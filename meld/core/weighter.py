"""Importance weighting for replay-free MELD updates."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from ..interfaces.base import TaskSnapshot


class KLIEPWeighter:
    """Gaussian KLIEP-style importance weighting without stored old samples.

    This is a parametric approximation: the old distribution is represented by
    the snapshot's diagonal Gaussian class manifolds, and the new-task
    distribution is fit from current-task embeddings only.
    """

    def __init__(self, clip_min: float = 0.25, clip_max: float = 4.0, eps: float = 1e-6) -> None:
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps
        self._old_means: Tensor | None = None
        self._old_vars: Tensor | None = None
        self._new_means: Tensor | None = None
        self._new_vars: Tensor | None = None
        self._old_prior_log = 0.0
        self._new_prior_log = 0.0
        self._fitted = False

    def fit(
        self,
        new_embeddings: Tensor,
        old_snapshot: TaskSnapshot,
        new_targets: Tensor | None = None,
    ) -> dict[int, np.ndarray]:
        device = new_embeddings.device
        dtype = new_embeddings.dtype
        if not old_snapshot.class_ids or new_embeddings.numel() == 0:
            self._fitted = False
            return {}

        old_means = [
            torch.from_numpy(old_snapshot.class_means[class_id]).to(device=device, dtype=dtype)
            for class_id in old_snapshot.class_ids
            if class_id in old_snapshot.class_means
        ]
        old_vars = [
            torch.from_numpy(old_snapshot.class_covs[class_id]).to(device=device, dtype=dtype)
            for class_id in old_snapshot.class_ids
            if class_id in old_snapshot.class_covs
        ]
        if not old_means or not old_vars:
            self._fitted = False
            return {}

        self._old_means = torch.stack(old_means, dim=0)
        self._old_vars = torch.stack(old_vars, dim=0).clamp_min(self.eps)

        if new_targets is None or new_targets.numel() == 0:
            self._new_means = new_embeddings.mean(dim=0, keepdim=True)
            self._new_vars = new_embeddings.var(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        else:
            means: list[Tensor] = []
            vars_: list[Tensor] = []
            for class_id in sorted(int(value) for value in new_targets.unique().tolist()):
                mask = new_targets == class_id
                if not torch.any(mask):
                    continue
                class_embeddings = new_embeddings[mask]
                means.append(class_embeddings.mean(dim=0))
                vars_.append(class_embeddings.var(dim=0, unbiased=False).clamp_min(self.eps))
            if not means:
                self._new_means = new_embeddings.mean(dim=0, keepdim=True)
                self._new_vars = new_embeddings.var(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
            else:
                self._new_means = torch.stack(means, dim=0)
                self._new_vars = torch.stack(vars_, dim=0)

        old_count = max(1, len(old_snapshot.class_ids))
        new_count = max(1, int(self._new_means.size(0)))
        total = float(old_count + new_count)
        self._old_prior_log = math.log(old_count / total)
        self._new_prior_log = math.log(new_count / total)
        self._fitted = True

        weights = self.score(new_embeddings)
        grouped: dict[int, np.ndarray] = {}
        if new_targets is not None and new_targets.numel() == weights.numel():
            for class_id in sorted(int(value) for value in new_targets.unique().tolist()):
                mask = new_targets == class_id
                grouped[class_id] = weights[mask].detach().cpu().numpy()
        return grouped

    def score(self, embeddings: Tensor) -> Tensor:
        if not self._fitted or self._old_means is None or self._old_vars is None:
            return torch.ones(embeddings.size(0), device=embeddings.device, dtype=embeddings.dtype)
        if self._new_means is None or self._new_vars is None:
            return torch.ones(embeddings.size(0), device=embeddings.device, dtype=embeddings.dtype)

        old_log = self._log_diag_gaussian_mixture(embeddings, self._old_means, self._old_vars)
        new_log = self._log_diag_gaussian_mixture(embeddings, self._new_means, self._new_vars)
        full_log = torch.logaddexp(
            old_log + self._old_prior_log,
            new_log + self._new_prior_log,
        )
        ratio = torch.exp(full_log - new_log)
        ratio = ratio / ratio.mean().clamp_min(self.eps)
        return ratio.clamp_(min=self.clip_min, max=self.clip_max)

    def _log_diag_gaussian_mixture(self, embeddings: Tensor, means: Tensor, vars_: Tensor) -> Tensor:
        dims = embeddings.size(1)
        diff = embeddings.unsqueeze(1) - means.unsqueeze(0)
        log_prob = -0.5 * (
            torch.log(vars_).sum(dim=1).unsqueeze(0)
            + (diff.pow(2) / vars_.unsqueeze(0)).sum(dim=2)
            + dims * math.log(2.0 * math.pi)
        )
        return torch.logsumexp(log_prob, dim=1) - math.log(max(1, means.size(0)))
