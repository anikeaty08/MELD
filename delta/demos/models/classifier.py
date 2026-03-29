"""Incremental classifier head for MELD."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


class IncrementalClassifier(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.heads = nn.ModuleList()
        self.class_to_head: dict[int, tuple[int, int]] = {}
        self.next_class_id = 0

    def adaption(self, nb_new_classes: int) -> list[int]:
        head = nn.Linear(self.in_dim, nb_new_classes)
        nn.init.kaiming_normal_(head.weight, nonlinearity="linear")
        nn.init.constant_(head.bias, 0.0)
        # Move new head to the same device as the rest of the model
        # so MPS / CUDA runs don't get a CPU/device mismatch on task 1+.
        if self.heads:
            head = head.to(next(self.heads.parameters()).device)
        head_index = len(self.heads)
        self.heads.append(head)
        class_ids = list(range(self.next_class_id, self.next_class_id + nb_new_classes))
        for offset, class_id in enumerate(class_ids):
            self.class_to_head[class_id] = (head_index, offset)
        self.next_class_id += nb_new_classes
        return class_ids

    def forward(self, embeddings: Tensor) -> Tensor:
        if not self.heads:
            raise RuntimeError("Classifier has no heads. Call adaption() first.")
        logits = [head(embeddings) for head in self.heads]
        return torch.cat(logits, dim=1)

    def weight_vector(self, class_id: int) -> Tensor:
        head_index, offset = self.class_to_head[class_id]
        return self.heads[head_index].weight[offset]

    def bias_value(self, class_id: int) -> Tensor:
        head_index, offset = self.class_to_head[class_id]
        return self.heads[head_index].bias[offset]

    def all_norms(self) -> dict[int, float]:
        return {class_id: float(self.weight_vector(class_id).norm(p=2).item()) for class_id in self.class_to_head}

    @property
    def num_classes(self) -> int:
        return len(self.class_to_head)