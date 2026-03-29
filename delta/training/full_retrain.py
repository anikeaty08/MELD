"""FullRetrainStrategy — baseline that retrains from scratch each task.

Accumulates all seen data and retrains from random init each task.
No regularization, no KFAC, no certificate. Pure comparison baseline.
"""

from __future__ import annotations
import copy
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .base import BaseStrategy


class FullRetrainStrategy(BaseStrategy):
    """Full retrain baseline — retrains on all accumulated data."""

    def __init__(self, model, optimizer, criterion, evaluator=None,
                 device=None, train_epochs=10, train_mb_size=64):
        super().__init__(model, optimizer, criterion, evaluator,
                         device, train_epochs, train_mb_size)
        self._all_datasets: list[Any] = []
        self._seen_classes: set[int] = set()
        self._initial_state_dict = copy.deepcopy(model.state_dict())

    def _prepare_training_experience(self, experience) -> None:
        before_keys = set(self.model.state_dict().keys())
        self._expand_incremental_head_if_needed(experience)
        after_state = self.model.state_dict()
        for key, value in after_state.items():
            if key not in before_keys and key not in self._initial_state_dict:
                self._initial_state_dict[key] = value.detach().cpu().clone()
        self._seen_classes.update(
            int(c) for c in getattr(experience, "classes_in_this_experience", [])
        )
        self._set_active_classes(self._seen_classes)

    def _before_training_experience(self, experience) -> None:
        self._all_datasets.append(experience.train_dataset)

        # Reset model to initial weights
        self.model.load_state_dict(self._initial_state_dict, strict=False)

        # Reset optimizer state
        self.optimizer.state.clear()

    def train(self, experience) -> None:
        """Override train to use accumulated dataset."""
        self.experience = experience
        self.current_task_id = experience.task_id

        self._prepare_training_experience(experience)
        self._before_training_experience(experience)
        self._call_plugins("before_training_experience", self, experience)

        # Build loader from ALL accumulated data
        combined = ConcatDataset(self._all_datasets)
        train_loader = DataLoader(
            combined,
            batch_size=self.train_mb_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(self.train_epochs):
            self._before_training_epoch()
            self._call_plugins("before_training_epoch", self)

            for mb_x, mb_y in train_loader:
                from .base import _to_device
                self.mb_x = _to_device(mb_x, self.device)
                self.mb_y = mb_y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                self.mb_logits = self.model(self.mb_x)
                masked_logits, masked_targets = self._masked_logits_and_targets(
                    self.mb_logits, self.mb_y, self.active_classes
                )
                self.loss = self.criterion(masked_logits, masked_targets)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0)
                self.optimizer.step()

            self._after_training_epoch()
            self._call_plugins("after_training_epoch", self)

        self._after_training_experience(experience)
        self._call_plugins("after_training_experience", self, experience)
        self._tasks_trained += 1
