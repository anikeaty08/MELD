"""Composite model wrapper for MELD."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .models.classifier import IncrementalClassifier


class MELDModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: IncrementalClassifier) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.out_dim = getattr(backbone, "out_dim")

    def forward(self, x: Tensor) -> Tensor:
        embeddings = self.backbone(x)
        return self.classifier(embeddings)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def clone(self) -> "MELDModel":
        return copy.deepcopy(self)
