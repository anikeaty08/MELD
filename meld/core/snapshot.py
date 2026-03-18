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
    def __init__(self, fisher_samples: int = 512, covariance_eps: float = 1e-6) -> None:
        self.fisher_samples = fisher_samples
        self.covariance_eps = covariance_eps

    def capture(self, model: nn.Module, dataloader: DataLoader, class_ids: list[int], task_id: int) -> TaskSnapshot:
        device = next(model.parameters()).device
        model.eval()
        class_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                embeddings = model.embed(inputs).detach().cpu().numpy()
                for embedding, target in zip(embeddings, targets.tolist()):
                    if int(target) in class_ids:
                        class_embeddings[int(target)].append(embedding)
                        total_samples += 1

        class_means: dict[int, np.ndarray] = {}
        class_covs: dict[int, np.ndarray] = {}
        for class_id in class_ids:
            vectors = np.stack(class_embeddings[class_id], axis=0)
            class_means[class_id] = vectors.mean(axis=0)
            class_covs[class_id] = vectors.var(axis=0) + self.covariance_eps

        fisher_diagonal = self._compute_fisher(model, dataloader, self.fisher_samples)
        parameter_reference = [param.detach().cpu().numpy().copy() for param in model.parameters()]
        steps_per_epoch = max(1, math.ceil(max(1, len(dataloader.dataset)) / max(1, dataloader.batch_size or 1)))

        return TaskSnapshot(
            task_id=task_id,
            class_ids=list(class_ids),
            class_means=class_means,
            class_covs=class_covs,
            classifier_norms=model.classifier.all_norms(),
            fisher_diagonal=fisher_diagonal,
            fisher_eigenvalue_max=float(np.max(fisher_diagonal)) if fisher_diagonal.size else 0.0,
            timestamp=time.time(),
            embedding_dim=int(next(iter(class_means.values())).shape[0]) if class_means else int(model.out_dim),
            dataset_size=len(dataloader.dataset),
            steps_per_epoch=steps_per_epoch,
            parameter_reference=parameter_reference,
        )

    def _compute_fisher(self, model: nn.Module, dataloader: DataLoader, fisher_samples: int) -> np.ndarray:
        device = next(model.parameters()).device
        params = [param for param in model.parameters() if param.requires_grad]
        fisher = [torch.zeros_like(param, device=device) for param in params]
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
            batch_size = inputs.size(0)
            for index, grad in enumerate(grads):
                fisher[index] += grad.detach().pow(2) * batch_size
            total += batch_size

        if total == 0:
            return np.array([], dtype=np.float32)
        flat = torch.cat([(entry / total).reshape(-1) for entry in fisher])
        return flat.detach().cpu().numpy()
