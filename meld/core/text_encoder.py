"""Frozen text encoder backbone for MELD NLP datasets.

Uses a pretrained HuggingFace sentence-transformer as a fixed feature extractor.
The backbone is frozen — only the incremental classifier head is trained per task.
This matches the frozen_analytic strategy philosophically but uses gradient-based
head training for better calibration.

Supported models (all produce 384-dim embeddings):
  - sentence-transformers/all-MiniLM-L6-v2  (default, fast, 22M params)
  - sentence-transformers/all-mpnet-base-v2  (higher quality, 110M params)

Usage:
    from meld.core.text_encoder import TextEncoderBackbone
    backbone = TextEncoderBackbone()   # downloads on first use
    emb = backbone(input_ids, attention_mask)  # (B, 384)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class TextEncoderBackbone(nn.Module):
    """Frozen HuggingFace encoder — produces sentence embeddings.

    The encoder weights are fully frozen. Only the MELD classifier
    head trains. This means:
    - No catastrophic forgetting in the backbone (it never changes)
    - Geometry / EWC losses still protect the classifier head
    - Very fast per-task training (head only, not backbone)

    Args:
        model_name: HuggingFace model identifier.
        pooling: How to reduce token embeddings to one vector.
            'mean' (default) — mean of all non-padding tokens.
            'cls'            — CLS token only.
        normalize: L2-normalise the output embedding (default True).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        pooling: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling
        self.normalize = normalize
        self._encoder: Any = None
        self._tokenizer: Any = None
        self.out_dim: int = self._resolve_out_dim(model_name)
        self._loaded = False

    @staticmethod
    def _resolve_out_dim(model_name: str) -> int:
        dims = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-MiniLM-L12-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
            "bert-base-uncased": 768,
            "distilbert-base-uncased": 768,
        }
        return dims.get(model_name, 384)

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is required for NLP datasets. "
                "Install with: pip install transformers"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        encoder = AutoModel.from_pretrained(self.model_name)
        # Freeze all encoder parameters — backbone never trains
        for param in encoder.parameters():
            param.requires_grad_(False)
        self._encoder = encoder
        self._loaded = True

    def get_tokenizer(self) -> Any:
        self._load()
        return self._tokenizer

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode tokenised text to a fixed embedding vector.

        Args:
            input_ids: (B, seq_len) int64 token ids.
            attention_mask: (B, seq_len) float mask.

        Returns:
            (B, out_dim) embedding tensor.
        """
        self._load()
        self._encoder = self._encoder.to(input_ids.device)
        out = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, seq_len, hidden)
        if self.pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            # mean pooling — mask padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.normalize:
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def embed(self, x: Any) -> Tensor:
        """Alias so the backbone satisfies the MELDModel.embed() interface."""
        if isinstance(x, dict):
            return self.forward(x["input_ids"], x["attention_mask"])
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return self.forward(x[0], x[1])
        raise TypeError(
            "TextEncoderBackbone.embed() expects a dict with 'input_ids'/'attention_mask' "
            "or a (input_ids, attention_mask) tuple."
        )