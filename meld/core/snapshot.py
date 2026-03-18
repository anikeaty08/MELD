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
                for embedding, logit, target in zip(embeddings, logits, targets.tolist()):
                    if int(target) in class_ids:
                        class_embeddings[int(target)].append(embedding)
                        class_logits[int(target)].append(logit)
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

        fisher_diagonal, mean_gradient_norm = self._compute_fisher(model, dataloader, self.fisher_samples)
        if self._ema_fisher is not None:
            fisher_diagonal = self._ema_decay * self._ema_fisher + (1.0 - self._ema_decay) * fisher_diagonal
        self._ema_fisher = fisher_diagonal.copy()
        parameter_reference = [param.detach().cpu().numpy().copy() for param in model.parameters()]
        steps_per_epoch = max(1, math.ceil(max(1, len(dataloader.dataset)) / max(1, dataloader.batch_size or 1)))

        return TaskSnapshot(
            task_id=task_id,
            class_ids=list(class_ids),
            class_means=class_means,
            class_covs=class_covs,
            class_anchors=class_anchors,
            class_anchor_logits=class_anchor_logits,
            classifier_norms=model.classifier.all_norms(),
            fisher_diagonal=fisher_diagonal,
            fisher_eigenvalue_max=float(np.max(fisher_diagonal)) if fisher_diagonal.size else 0.0,
            mean_gradient_norm=mean_gradient_norm,
            timestamp=time.time(),
            embedding_dim=int(next(iter(class_means.values())).shape[0]) if class_means else int(model.out_dim),
            dataset_size=len(dataloader.dataset),
            steps_per_epoch=steps_per_epoch,
            parameter_reference=parameter_reference,
        )

    def _compute_fisher(self, model: nn.Module, dataloader: DataLoader, fisher_samples: int) -> tuple[np.ndarray, float]:
        device = next(model.parameters()).device
        params = [param for param in model.parameters() if param.requires_grad]
        fisher = [torch.zeros_like(param, device=device) for param in params]
        grad_norms = []
        total = 0
        criterion = nn.CrossEntropyLoss()

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
            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
            grad_norm = torch.sqrt(torch.stack([grad.detach().pow(2).sum() for grad in grads]).sum())
            grad_norms.append(float(grad_norm.item()))
            batch_size = inputs.size(0)
            for index, grad in enumerate(grads):
                fisher[index] += grad.detach().pow(2) * batch_size
            total += batch_size

        if total == 0:
            return np.array([], dtype=np.float32), 0.0
        flat = torch.cat([(entry / total).reshape(-1) for entry in fisher])
        return flat.detach().cpu().numpy(), float(np.mean(grad_norms)) if grad_norms else 0.0
