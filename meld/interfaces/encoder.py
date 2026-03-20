"""Encoder abstraction for future non-vision MELD backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class EncoderAdapter(ABC):
    """Minimal interface expected by MELD updater/oracle components."""

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Embedding dimension produced by the adapter."""

    @abstractmethod
    def encode(self, inputs: Any) -> torch.Tensor:
        """Convert task input into embedding vectors."""


@dataclass(slots=True)
class MLPEncoderConfig:
    input_dim: int
    hidden_dim: int = 128
    out_dim: int = 64


class MLPEncoderAdapter(nn.Module, EncoderAdapter):
    """Simple tabular/text placeholder encoder for research experiments."""

    def __init__(self, cfg: MLPEncoderConfig) -> None:
        super().__init__()
        self._out_dim = int(cfg.out_dim)
        self.net = nn.Sequential(
            nn.Linear(int(cfg.input_dim), int(cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(cfg.hidden_dim), int(cfg.out_dim)),
        )

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def encode(self, inputs: Any) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        flat = inputs.float().view(inputs.size(0), -1)
        return self.net(flat)
