"""Snapshot capture logic for MELD."""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..interfaces.base import SnapshotStrategy, TaskSnapshot


class FisherManifoldSnapshot(SnapshotStrategy):
    def __init__(self, fisher_samples: int = 512, covariance_eps: float = 1e-6, anchors_per_class: int = 20) -> None:
        self.fisher_samples = fisher_samples
        self.covariance_eps = covariance_eps
        self.anchors_per_class = anchors_per_class
        self._ema_fisher: np.ndarray | None = None
        self._ema_decay: float = 0.9
        # EMA factors for K-FAC style curvature approximation on a small set of
        # parameters (last 2 linear layers).
        self._ema_kfac_A: dict[str, np.ndarray] = {}
        self._ema_kfac_G: dict[str, np.ndarray] = {}

    def capture(self, model: nn.Module, dataloader: DataLoader, class_ids: list[int], task_id: int) -> TaskSnapshot:
        device = next(model.parameters()).device
        model.eval()
        class_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
        class_logits: dict[int, list[np.ndarray]] = defaultdict(list)
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                embeddings_tensor = model.embed(inputs)
                logits_tensor = model.classifier(embeddings_tensor)
                embeddings = embeddings_tensor.detach().cpu().numpy()
                logits = logits_tensor.detach().cpu().numpy()
                targets_list = targets.tolist()
                for i, target in enumerate(targets_list):
                    tid = int(target)
                    if tid in class_ids:
                        class_embeddings[tid].append(embeddings[i])
                        class_logits[tid].append(logits[i])
                        total_samples += 1

        class_means: dict[int, np.ndarray] = {}
        class_covs: dict[int, np.ndarray] = {}
        class_anchors: dict[int, np.ndarray] = {}
        class_anchor_logits: dict[int, np.ndarray] = {}
        for class_id in class_ids:
            vectors = np.stack(class_embeddings[class_id], axis=0)
            logits = np.stack(class_logits[class_id], axis=0)
            class_means[class_id] = vectors.mean(axis=0)
            class_covs[class_id] = vectors.var(axis=0) + self.covariance_eps
            anchor_count = min(self.anchors_per_class, len(vectors))
            if anchor_count > 0:
                indices = np.linspace(0, len(vectors) - 1, num=anchor_count, dtype=int)
                class_anchors[class_id] = vectors[indices]
                class_anchor_logits[class_id] = logits[indices]
            else:
                class_anchors[class_id] = vectors[:0]
                class_anchor_logits[class_id] = logits[:0]

        fisher_samples = self.fisher_samples
        try:
            fisher_samples = min(fisher_samples, int(len(dataloader.dataset)))
        except Exception:
            pass

        (
            fisher_diagonal,
            mean_gradient_norm,
            kfac_weight_param_names,
            kfac_factors_A,
            kfac_factors_G,
            kfac_eig_max,
        ) = self._compute_fisher(model, dataloader, fisher_samples)
        if self._ema_fisher is not None and self._ema_fisher.shape == fisher_diagonal.shape:
            fisher_diagonal = self._ema_decay * self._ema_fisher + (1.0 - self._ema_decay) * fisher_diagonal
        self._ema_fisher = fisher_diagonal.copy()
        parameter_reference = [
            param.detach().cpu().numpy().copy() for param in model.parameters() if param.requires_grad
        ]
        steps_per_epoch = max(1, math.ceil(max(1, len(dataloader.dataset)) / max(1, dataloader.batch_size or 1)))
        diag_eig_max = float(np.max(fisher_diagonal)) if fisher_diagonal.size else 0.0
        fisher_eigenvalue_max = max(diag_eig_max, float(kfac_eig_max))

        return TaskSnapshot(
            task_id=task_id,
            class_ids=list(class_ids),
            class_means=class_means,
            class_covs=class_covs,
            class_anchors=class_anchors,
            class_anchor_logits=class_anchor_logits,
            classifier_norms=model.classifier.all_norms(),
            fisher_diagonal=fisher_diagonal,
            fisher_eigenvalue_max=fisher_eigenvalue_max,
            mean_gradient_norm=mean_gradient_norm,
            timestamp=time.time(),
            embedding_dim=int(next(iter(class_means.values())).shape[0]) if class_means else int(model.out_dim),
            dataset_size=len(dataloader.dataset),
            steps_per_epoch=steps_per_epoch,
            parameter_reference=parameter_reference,
            kfac_weight_param_names=kfac_weight_param_names,
            kfac_factors_A=kfac_factors_A,
            kfac_factors_G=kfac_factors_G,
        )

    def _compute_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        fisher_samples: int,
    ) -> tuple[
        np.ndarray,
        float,
        list[str],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        float,
    ]:
        device = next(model.parameters()).device
        params = [param for param in model.parameters() if param.requires_grad]
        fisher = [torch.zeros_like(param, device=device) for param in params]
        grad_norms = []
        total = 0
        criterion = nn.CrossEntropyLoss()

        # K-FAC factors for the last 2 Linear layers (by module order).
        linear_modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
        kfac_modules = linear_modules[-2:] if len(linear_modules) >= 2 else linear_modules
        name_by_param = {p: n for n, p in model.named_parameters()}

        kfac_weight_param_names: list[str] = []
        for m in kfac_modules:
            pname = name_by_param.get(m.weight)
            if pname is not None:
                kfac_weight_param_names.append(pname)

        activations: dict[str, Tensor] = {}
        grad_outputs: dict[str, Tensor] = {}
        A_acc: dict[str, Tensor] = {}
        G_acc: dict[str, Tensor] = {}
        handles = []

        for m in kfac_modules:
            pname = name_by_param.get(m.weight)
            if pname is None:
                continue
            in_dim = int(m.in_features)
            out_dim = int(m.out_features)
            A_acc[pname] = torch.zeros((in_dim, in_dim), device=device)
            G_acc[pname] = torch.zeros((out_dim, out_dim), device=device)

            def _make_fwd_hook(pn: str):
                def _hook(mod: nn.Module, inp: tuple[Tensor, ...], out: Tensor) -> None:
                    activations[pn] = inp[0].detach()
                return _hook

            def _make_bwd_hook(pn: str):
                def _hook(mod: nn.Module, grad_in: tuple[Tensor, ...], grad_out: tuple[Tensor, ...]) -> None:
                    grad_outputs[pn] = grad_out[0].detach()
                return _hook

            handles.append(m.register_forward_hook(_make_fwd_hook(pname)))
            handles.append(m.register_full_backward_hook(_make_bwd_hook(pname)))

        try:
            for inputs, targets in dataloader:
                if total >= fisher_samples:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)
                remaining = fisher_samples - total
                if inputs.size(0) > remaining:
                    inputs = inputs[:remaining]
                    targets = targets[:remaining]

                model.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()

                batch_size = inputs.size(0)
                for index, param in enumerate(params):
                    if param.grad is None:
                        continue
                    fisher[index] += param.grad.detach().pow(2) * batch_size

                # Mean grad norm proxy (used for oracle calibration).
                per_param_norms = [param.grad.detach().pow(2).sum() for param in params if param.grad is not None]
                if per_param_norms:
                    grad_norm = torch.sqrt(torch.stack(per_param_norms).sum())
                    grad_norms.append(float(grad_norm.item()))

                # K-FAC accumulation from hooked activations/gradients.
                if kfac_weight_param_names:
                    for pname in kfac_weight_param_names:
                        a = activations.get(pname)
                        g = grad_outputs.get(pname)
                        if a is None or g is None:
                            continue
                        A_acc[pname] += a.t() @ a
                        G_acc[pname] += g.t() @ g

                total += batch_size
        finally:
            for h in handles:
                h.remove()

        if total == 0:
            return np.array([], dtype=np.float32), 0.0, [], {}, {}, 0.0

        flat = torch.cat([(entry / total).reshape(-1) for entry in fisher])
        fisher_np = flat.detach().cpu().numpy()
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

        kfac_A_np: dict[str, np.ndarray] = {}
        kfac_G_np: dict[str, np.ndarray] = {}
        kfac_eig_max = 0.0

        for pname in kfac_weight_param_names:
            A = (A_acc[pname] / total).detach()
            G = (G_acc[pname] / total).detach()

            # Optional EMA smoothing for K-FAC factors.
            old_A = self._ema_kfac_A.get(pname)
            old_G = self._ema_kfac_G.get(pname)
            if old_A is not None and old_A.shape == tuple(A.shape):
                A = self._ema_decay * torch.from_numpy(old_A).to(device) + (1.0 - self._ema_decay) * A
            if old_G is not None and old_G.shape == tuple(G.shape):
                G = self._ema_decay * torch.from_numpy(old_G).to(device) + (1.0 - self._ema_decay) * G

            A_np = A.detach().cpu().numpy().astype(np.float32, copy=False)
            G_np = G.detach().cpu().numpy().astype(np.float32, copy=False)
            self._ema_kfac_A[pname] = A_np.copy()
            self._ema_kfac_G[pname] = G_np.copy()

            # Upper-bound-ish spectral proxy: maxeig(A) * maxeig(G).
            eigA = np.linalg.eigvalsh(A_np)
            eigG = np.linalg.eigvalsh(G_np)
            eigA = np.clip(eigA, 0.0, None)
            eigG = np.clip(eigG, 0.0, None)
            est = float(np.max(eigA) * np.max(eigG))
            kfac_eig_max = max(kfac_eig_max, est)

            kfac_A_np[pname] = A_np
            kfac_G_np[pname] = G_np

        return fisher_np, mean_grad_norm, kfac_weight_param_names, kfac_A_np, kfac_G_np, kfac_eig_max
