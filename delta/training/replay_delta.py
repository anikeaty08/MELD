"""ReplayDeltaStrategy: practical replay-first continual learning."""

from __future__ import annotations

import math
from typing import Any

import torch

from .base import _to_device

from .fisher_delta import FisherDeltaStrategy


class ReplayDeltaStrategy(FisherDeltaStrategy):
    """Practical replay-focused strategy for stronger empirical accuracy.

    This mode keeps the same framework structure as FisherDeltaStrategy
    but biases the defaults toward rehearsal-based continual learning:
    larger replay memory, mixed replay batches, stronger classifier
    balancing, and nearest-mean evaluation enabled by default.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        evaluator=None,
        device=None,
        train_epochs=10,
        train_mb_size=64,
        kd_alpha=0.5,
        kd_temperature=2.0,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            evaluator=evaluator,
            device=device,
            train_epochs=train_epochs,
            train_mb_size=train_mb_size,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
        )
        self._base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.use_nme_classifier = False
        self.mix_replay_into_batches = True
        self.replay_memory_per_class = max(32, self.replay_memory_per_class)
        self.replay_batch_size = max(self.train_mb_size, 64, self.replay_batch_size)
        self.replay_alpha = 1.0
        self.replay_kd_alpha = 0.75
        self.feature_kd_alpha = 0.5
        self.classifier_balance_steps = max(20, self.classifier_balance_steps)
        self.classifier_balance_lr = 0.05
        self.use_cosine_lr = True
        self.min_lr_scale = 0.1
        self.bias_correction_steps = 40
        self.bias_correction_lr = 0.01
        self.validation_batch_size = 64
        self._head_bias_params: dict[int, tuple[float, float]] = {}
        self.use_task_identity_inference = False
        self._effective_task_identity_inference = False
        self._eval_task_classes: list[int] = []

    def _before_training_experience(self, experience) -> None:
        super()._before_training_experience(experience)
        self._effective_task_identity_inference = bool(
            self.use_task_identity_inference
            or getattr(experience, "scenario", "class_incremental") == "task_incremental"
        )
        if len(self._base_lrs) != len(self.optimizer.param_groups):
            self._base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr
        if self.state is not None:
            # Replay is the main retention mechanism in this mode.
            # Keep a lighter Fisher pull so old weights are preserved
            # without overwhelming the replay signal.
            self.ewc_scale = min(self.ewc_scale, 0.25)
            self._ewc_target_scale = min(self._ewc_target_scale, 0.25)
            self._ewc_effective_scale = max(
                self._ewc_target_scale * 0.25,
                self._ewc_effective_scale,
            )

    def _before_training_epoch(self) -> None:
        super()._before_training_epoch()
        if not self.use_cosine_lr or not self.optimizer.param_groups:
            return
        if self.train_epochs <= 1:
            scale = 1.0
        else:
            progress = float(self._epoch_index - 1) / float(max(self.train_epochs - 1, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * scale

    def _predict_eval(self, x):
        logits = self.model(x)
        logits = self._apply_bias_corrections(logits)
        if self._effective_task_identity_inference and self._eval_task_classes:
            logits = self._mask_logits_to_task(logits, self._eval_task_classes)
        return logits

    def _before_eval_experience(self, experience) -> None:
        self._eval_task_classes = [
            int(c) for c in getattr(experience, "classes_in_this_experience", [])
        ]
        self._effective_task_identity_inference = bool(
            self.use_task_identity_inference
            or getattr(experience, "scenario", "class_incremental") == "task_incremental"
        )

    def _before_eval_stream(self, experiences) -> None:
        self._eval_task_classes = []

    def _mask_logits_to_task(self, logits: torch.Tensor, class_ids: list[int]) -> torch.Tensor:
        valid = [int(c) for c in class_ids if 0 <= int(c) < logits.shape[1]]
        if not valid or len(valid) == logits.shape[1]:
            return logits
        masked = torch.full_like(logits, fill_value=-1e9)
        index = torch.tensor(valid, device=logits.device, dtype=torch.long)
        masked[:, index] = logits.index_select(1, index)
        return masked

    def _head_slice(self, classifier, head_index: int) -> slice | None:
        start = 0
        for index, head in enumerate(classifier.heads):
            end = start + int(head.out_features)
            if index == head_index:
                return slice(start, end)
            start = end
        return None

    def _apply_bias_corrections(self, logits: torch.Tensor) -> torch.Tensor:
        classifier = getattr(self.model, "classifier", None)
        if classifier is None or not self._head_bias_params:
            return logits
        corrected = logits
        for head_index, (alpha, beta) in self._head_bias_params.items():
            sl = self._head_slice(classifier, head_index)
            if sl is None:
                continue
            corrected = self._bias_correct_head(
                corrected,
                sl,
                float(alpha),
                float(beta),
            )
        return corrected

    def _bias_correct_head(
        self,
        logits: torch.Tensor,
        head_slice: slice,
        alpha: torch.Tensor | float,
        beta: torch.Tensor | float,
    ) -> torch.Tensor:
        start = 0 if head_slice.start is None else int(head_slice.start)
        stop = logits.shape[1] if head_slice.stop is None else int(head_slice.stop)
        pieces: list[torch.Tensor] = []
        if start > 0:
            pieces.append(logits[:, :start])
        pieces.append(logits[:, start:stop] * alpha + beta)
        if stop < logits.shape[1]:
            pieces.append(logits[:, stop:])
        return torch.cat(pieces, dim=1)

    def _sample_current_task_batch(self, loader, total_size: int) -> tuple[Any, torch.Tensor] | None:
        if total_size <= 0 or not self._current_classes:
            return None
        target_classes = [int(c) for c in self._current_classes]
        target_set = set(target_classes)
        samples_per_class = max(1, total_size // max(1, len(target_classes)))
        chosen_inputs: list[Any] = []
        chosen_labels: list[int] = []
        counts = {class_id: 0 for class_id in target_classes}

        for mb_x, mb_y in loader:
            labels = mb_y.detach().cpu().tolist()
            batch_inputs = self._clone_memory_value(mb_x)
            for idx, label in enumerate(labels):
                class_id = int(label)
                if class_id not in target_set or counts[class_id] >= samples_per_class:
                    continue
                chosen_inputs.append(
                    self._clone_memory_value(self._slice_memory_input(batch_inputs, idx))
                )
                chosen_labels.append(class_id)
                counts[class_id] += 1
            if all(count >= samples_per_class for count in counts.values()):
                break

        if not chosen_inputs:
            return None
        batch_x = self._stack_memory_inputs(chosen_inputs)
        batch_x = _to_device(batch_x, self.device)
        batch_y = torch.tensor(chosen_labels, device=self.device, dtype=torch.long)
        return batch_x, batch_y

    def _fit_bias_correction(self, loader) -> None:
        if self.bias_correction_steps <= 0:
            return
        classifier = getattr(self.model, "classifier", None)
        if (
            classifier is None
            or not hasattr(classifier, "heads")
            or not hasattr(classifier, "class_to_head")
            or not self._current_classes
        ):
            return

        current_head = classifier.class_to_head.get(int(self._current_classes[0]))
        if current_head is None:
            return
        head_index = int(current_head[0])
        old_classes = [
            int(c) for c in classifier.class_to_head
            if int(c) not in set(self._current_classes)
        ]
        if not old_classes:
            self._head_bias_params[head_index] = (1.0, 0.0)
            return

        old_batch = self._sample_balanced_replay_batch(old_classes)
        if old_batch is None:
            return
        old_x, old_y = old_batch
        new_batch = self._sample_current_task_batch(loader, int(old_y.shape[0]))
        if new_batch is None:
            return
        new_x, new_y = new_batch

        val_x = self._concat_inputs(old_x, new_x)
        val_y = torch.cat([old_y, new_y], dim=0)
        with torch.no_grad():
            base_logits = self.model(val_x).detach()
        sl = self._head_slice(classifier, head_index)
        if sl is None:
            return

        alpha = torch.nn.Parameter(torch.tensor(1.0, device=self.device))
        beta = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        optimizer = torch.optim.Adam([alpha, beta], lr=self.bias_correction_lr)
        active_classes = sorted(int(c) for c in classifier.class_to_head)
        for _ in range(self.bias_correction_steps):
            optimizer.zero_grad(set_to_none=True)
            corrected = self._bias_correct_head(base_logits, sl, alpha, beta)
            masked_logits, masked_targets = self._masked_logits_and_targets(
                corrected, val_y, active_classes
            )
            loss = self.criterion(masked_logits, masked_targets)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()

        self._head_bias_params[head_index] = (
            float(alpha.detach().item()),
            float(beta.detach().item()),
        )
