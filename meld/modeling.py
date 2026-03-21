"""Composite model wrapper for MELD."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .models.classifier import IncrementalClassifier


class MELDModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: IncrementalClassifier) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.out_dim = getattr(backbone, "out_dim")

    def forward(self, x: Any) -> Tensor:
        embeddings = self._encode_inputs(x)
        return self.classifier(embeddings)

    def embed(self, x: Any) -> Tensor:
        return self._encode_inputs(x)

    def clone(self) -> "MELDModel":
        return copy.deepcopy(self)

    def _encode_inputs(self, x: Any) -> Tensor:
        if isinstance(x, dict):
            if hasattr(self.backbone, "embed"):
                return self.backbone.embed(x)
            return self.backbone(**x)
        if isinstance(x, (list, tuple)) and not isinstance(x, Tensor):
            if hasattr(self.backbone, "embed"):
                return self.backbone.embed(x)
            return self.backbone(*x)
        return self.backbone(x)
