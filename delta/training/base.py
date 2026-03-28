"""BaseStrategy — the hook-based training abstraction.

All strategies inherit from this class. The hook system allows
plugins to inject behavior without subclassing.

Hook execution order per train() call:
  _before_training_experience
    for each epoch:
      _before_training_epoch
        for each batch:
          _before_training_iteration
          forward
          _after_forward
          loss
          _before_backward
          backward
          _after_backward
          optimizer step
          _after_training_iteration
      _after_training_epoch
  _after_training_experience
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


def _to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, dict):
        return {k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in inputs.items()}
    if isinstance(inputs, tuple):
        return tuple(_to_device(v, device) for v in inputs)
    if isinstance(inputs, list):
        return [_to_device(v, device) for v in inputs]
    if isinstance(inputs, Tensor):
        return inputs.to(device)
    return inputs


class BaseStrategy:
    """Base class for all continual learning strategies.

    Provides the hook system and plugin architecture.
    Subclasses override specific hooks to implement their logic.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        evaluator: Any = None,
        device: torch.device | str | None = None,
        train_epochs: int = 10,
        train_mb_size: int = 64,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.evaluator = evaluator
        self.device = torch.device(device) if isinstance(device, str) else device
        self.train_epochs = train_epochs
        self.train_mb_size = train_mb_size

        self.plugins: list[Any] = []
        self.experience = None
        self.current_task_id: int = -1
        self._tasks_trained: int = 0
        self.active_classes: list[int] = []

        # Per-iteration state — set during training
        self.mb_x: Tensor | None = None
        self.mb_y: Tensor | None = None
        self.mb_logits: Tensor | None = None
        self.loss: Tensor | None = None

    def add_plugin(self, plugin: Any) -> None:
        """Add a plugin that receives hook callbacks."""
        self.plugins.append(plugin)

    # ── Public API ───────────────────────────────────────────

    def train(self, experience) -> None:
        """Train on one experience (task)."""
        self.experience = experience
        self.current_task_id = experience.task_id

        self._prepare_training_experience(experience)
        train_loader = experience.train_dataloader(
            batch_size=self.train_mb_size, shuffle=True)

        self._before_training_experience(experience)
        self._call_plugins("before_training_experience", self, experience)

        self.model.train()
        for epoch in range(self.train_epochs):
            self._before_training_epoch()
            self._call_plugins("before_training_epoch", self)

            for mb_x, mb_y in train_loader:
                self.mb_x = _to_device(mb_x, self.device)
                self.mb_y = mb_y.to(self.device)

                self._before_training_iteration(self.mb_x, self.mb_y)
                self._call_plugins("before_training_iteration", self)

                self.optimizer.zero_grad(set_to_none=True)
                self.mb_logits = self.model(self.mb_x)

                self._after_forward(self.mb_x, self.mb_y, self.mb_logits)
                self._call_plugins("after_forward", self)

                self.loss = self._compute_loss(
                    self.mb_x, self.mb_y, self.mb_logits)

                self._before_backward(self.loss)
                self._call_plugins("before_backward", self)

                if self.loss is not None and torch.isfinite(self.loss):
                    self.loss.backward()

                self._after_backward()
                self._call_plugins("after_backward", self)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 5.0)
                self.optimizer.step()

                self._after_training_iteration(
                    self.mb_x, self.mb_y, self.mb_logits, self.loss)
                self._call_plugins("after_training_iteration", self)

            self._after_training_epoch()
            self._call_plugins("after_training_epoch", self)

        self._after_training_experience(experience)
        self._call_plugins("after_training_experience", self, experience)
        self._tasks_trained += 1

    def eval(self, stream_or_experiences) -> dict[str, Any]:
        """Evaluate on a stream or list of experiences."""
        if not isinstance(stream_or_experiences, (list, tuple)):
            stream_or_experiences = [stream_or_experiences]

        self.model.eval()
        all_results: dict[str, Any] = {}
        self._before_eval_stream(stream_or_experiences)
        self._call_plugins("before_eval_stream", self, stream_or_experiences)

        for exp in stream_or_experiences:
            loader = exp.test_dataloader(batch_size=self.train_mb_size)
            correct = total = 0

            self._before_eval_experience(exp)
            self._call_plugins("before_eval_experience", self, exp)

            with torch.no_grad():
                for mb_x, mb_y in loader:
                    mb_x = _to_device(mb_x, self.device)
                    mb_y = mb_y.to(self.device)
                    logits = self._predict_eval(mb_x)
                    preds = logits.argmax(dim=1)
                    correct += int((preds == mb_y).sum().item())
                    total += int(mb_y.numel())

            acc = correct / max(total, 1)
            all_results[f"accuracy/task_{exp.task_id}"] = acc

            if not hasattr(self, "_last_eval_acc"):
                self._last_eval_acc = {}
            self._last_eval_acc.update(all_results)

            self._after_eval_experience(exp)
            self._call_plugins("after_eval_experience", self, exp)

        if all_results:
            acc_vals = [v for k, v in all_results.items()
                        if k.startswith("accuracy/task_")]
            all_results["accuracy/stream"] = (
                sum(acc_vals) / len(acc_vals) if acc_vals else 0.0
            )

        self._call_plugins("after_eval_stream", self, all_results)
        return all_results

    # ── Loss computation ─────────────────────────────────────

    def _compute_loss(self, x, y, logits) -> Tensor:
        masked_logits, masked_targets = self._masked_logits_and_targets(
            logits, y, self.active_classes
        )
        return self.criterion(masked_logits, masked_targets)

    def _predict_eval(self, x) -> Tensor:
        return self.model(x)

    def _prepare_training_experience(self, experience) -> None:
        pass

    # ── Hooks — override in subclasses ───────────────────────

    def _before_training_experience(self, experience) -> None:
        pass

    def _after_training_experience(self, experience) -> None:
        pass

    def _before_training_epoch(self) -> None:
        pass

    def _after_training_epoch(self) -> None:
        pass

    def _before_training_iteration(self, x, y) -> None:
        pass

    def _after_forward(self, x, y, logits) -> None:
        pass

    def _before_backward(self, loss) -> None:
        pass

    def _after_backward(self) -> None:
        pass

    def _after_training_iteration(self, x, y, logits, loss) -> None:
        pass

    def _before_eval_experience(self, experience) -> None:
        pass

    def _before_eval_stream(self, experiences) -> None:
        pass

    def _after_eval_experience(self, experience) -> None:
        pass

    def _set_active_classes(self, class_ids) -> None:
        self.active_classes = sorted({int(c) for c in class_ids if int(c) >= 0})

    def _masked_logits_and_targets(
        self,
        logits: Tensor,
        targets: Tensor,
        class_ids=None,
    ) -> tuple[Tensor, Tensor]:
        class_ids = list(class_ids or [])
        if not class_ids:
            return logits, targets

        valid_classes = [c for c in class_ids if 0 <= int(c) < logits.shape[1]]
        if not valid_classes or len(valid_classes) == logits.shape[1]:
            return logits, targets

        class_index = torch.tensor(
            valid_classes, device=logits.device, dtype=torch.long
        )
        masked_logits = logits.index_select(1, class_index)

        mapping = {class_id: idx for idx, class_id in enumerate(valid_classes)}
        target_list = [int(t) for t in targets.detach().cpu().tolist()]
        if any(t not in mapping for t in target_list):
            return logits, targets
        remapped_targets = torch.tensor(
            [mapping[t] for t in target_list],
            device=targets.device,
            dtype=targets.dtype,
        )
        return masked_logits, remapped_targets

    def _expand_incremental_head_if_needed(self, experience) -> list[int]:
        classifier = getattr(self.model, "classifier", None)
        if classifier is None or not hasattr(classifier, "adaption"):
            return []

        existing_classes = set(getattr(classifier, "class_to_head", {}).keys())
        requested_classes = [
            int(c) for c in getattr(experience, "classes_in_this_experience", [])
        ]
        missing = [c for c in requested_classes if c not in existing_classes]
        if not missing:
            return []

        next_class_id = getattr(classifier, "next_class_id", len(existing_classes))
        expected = list(range(next_class_id, next_class_id + len(missing)))
        if missing != expected:
            raise ValueError(
                f"Incremental classifier expected new classes {expected}, got {missing}."
            )

        classifier.adaption(len(missing))
        if classifier.heads:
            self.optimizer.add_param_group(
                {"params": list(classifier.heads[-1].parameters())}
            )
        return missing

    # ── Plugin dispatch ──────────────────────────────────────

    def _call_plugins(self, hook_name: str, *args) -> None:
        for plugin in self.plugins:
            fn = getattr(plugin, hook_name, None)
            if fn is not None:
                fn(*args)
        # Also call evaluator hooks
        if self.evaluator is not None:
            fn = getattr(self.evaluator, hook_name, None)
            if fn is not None:
                fn(*args)

# Update 9 - 2026-03-28 22:27:54
# Update 10 - 2026-03-29 00:34:44
# Update 18 - 2026-03-29 02:16:19
# Update 28 - 2026-03-28 14:34:58
# Update 2 @ 2026-03-28 12:27:08
# Update 13 @ 2026-03-29 06:44:39
# Update 21 @ 2026-03-28 16:51:53
# Update 24 @ 2026-03-28 22:29:06
# Update 1 @ 2026-03-29 08:46:00
# Update 13 @ 2026-03-28 12:24:04
# Update 14 @ 2026-03-28 10:27:48
# Update 21 @ 2026-03-29 06:53:48
# Update 27 @ 2026-03-28 14:02:38