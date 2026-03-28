"""FisherDeltaStrategy with bounded delta updates and task-time head growth."""

from __future__ import annotations

import copy
import time
import warnings
import numpy as np
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BaseStrategy, _to_device
from ..core.calibration import CalibrationTracker
from ..core.certificate import CertificateComputer, EquivalenceCertificate
from ..core.fisher import KFACComputer
from ..core.shift import ShiftDetector
from ..core.state import DeltaState


class FisherDeltaStrategy(BaseStrategy):
    """Delta-only update strategy with Fisher regularization and KD."""

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
            evaluator,
            device,
            train_epochs,
            train_mb_size,
        )
        self._fisher_computer = KFACComputer()
        self._shift_detector = ShiftDetector()
        self._calibration = CalibrationTracker()
        self._cert_computer = CertificateComputer()

        self.state: DeltaState | None = None
        self.last_certificate: EquivalenceCertificate | None = None

        self.kd_alpha = float(kd_alpha)
        self.kd_temperature = float(kd_temperature)

        self.ce_scale = 1.0
        self.ewc_scale = 0.0
        self.shift_type = "none"
        self._ece_before = float("nan")
        self._train_start = 0.0
        self._task0_time = 0.0
        self._task0_samples = 0
        self._last_full_retrain_time = 0.0
        self._epoch_index = 0
        self._ewc_target_scale = 0.0
        self._ewc_effective_scale = 0.0

        self._old_model: torch.nn.Module | None = None
        self._ref_params: dict[str, Tensor] = {}
        self._kfac_A: dict[str, Tensor] = {}
        self._kfac_G: dict[str, Tensor] = {}
        self._kfac_names: set[str] = set()
        self._fisher_splits: dict[str, Tensor] = {}
        self._fisher_trace = 1.0
        self._current_classes: list[int] = []

        self.feature_kd_alpha = 0.5
        self.replay_alpha = 1.0
        self.replay_kd_alpha = 0.5
        self.replay_batch_size = 64
        self.replay_memory_per_class = 20
        self.replay_memory: dict[int, list[Any]] = {}
        self.classifier_balance_steps = 25
        self.classifier_balance_lr = 0.05
        self.mix_replay_into_batches = True
        self._batch_has_mixed_replay = False
        self._batch_replay_count = 0
        self.use_nme_classifier = False
        self.normalize_nme_features = True

    def _prepare_training_experience(self, experience) -> None:
        self._expand_incremental_head_if_needed(experience)
        seen_classes = []
        if self.state is not None:
            seen_classes = [int(c) for c in self.state.label_counts]
        current_classes = [
            int(c) for c in getattr(experience, "classes_in_this_experience", [])
        ]
        self._current_classes = current_classes
        self._set_active_classes(seen_classes + current_classes)

    def _before_training_experience(self, experience) -> None:
        self._train_start = time.time()
        self._epoch_index = 0
        self._old_model = None
        self._ref_params = {}
        self._kfac_A = {}
        self._kfac_G = {}
        self._kfac_names = set()
        self._fisher_splits = {}
        self._fisher_trace = 1.0

        if self.state is None:
            self.ce_scale = 1.0
            self.ewc_scale = 0.0
            self._ewc_target_scale = 0.0
            self._ewc_effective_scale = 0.0
            self.shift_type = "none"
            return

        n_new = len(experience.train_dataset)
        n_old = self.state.n_old
        self.ce_scale = 1.0
        self.ewc_scale = min(float(n_old) / float(max(n_new, 1)), 1.0)
        self._ewc_target_scale = self.ewc_scale
        self._ewc_effective_scale = 0.0

        loader = experience.train_dataloader(batch_size=self.train_mb_size)
        self.shift_type = self._shift_detector.detect(
            loader, self.state, self.model, self.device
        )

        if self.shift_type == "concept":
            warnings.warn(
                "Concept drift detected. Setting ewc_scale=0.0. "
                "Consider full retraining.",
                UserWarning,
                stacklevel=2,
            )
            self.ce_scale = 1.0
            self.ewc_scale = 0.0
            self._ewc_target_scale = 0.0
            self._ewc_effective_scale = 0.0

        test_loader = experience.test_dataloader(batch_size=self.train_mb_size)
        self._ece_before = self._calibration.compute_ece(
            self.model, test_loader, self.device
        )

        self._old_model = copy.deepcopy(self.model)
        self._old_model.eval()
        for param in self._old_model.parameters():
            param.requires_grad = False

        self._ref_params = {
            name: torch.from_numpy(value).to(self.device)
            for name, value in self.state.theta_ref.items()
        }
        self._kfac_A = {
            name: torch.from_numpy(value).to(self.device)
            for name, value in self.state.kfac_A.items()
        }
        self._kfac_G = {
            name: torch.from_numpy(value).to(self.device)
            for name, value in self.state.kfac_G.items()
        }
        self._kfac_names = set(self.state.kfac_param_names)

        if self.state.fisher_diag is not None:
            fisher_diag = torch.from_numpy(self.state.fisher_diag).to(self.device)
            self._fisher_trace = max(float(fisher_diag.sum().item()), 1e-8)
            cursor = 0
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                count = param.numel()
                if cursor + count <= fisher_diag.numel():
                    self._fisher_splits[name] = fisher_diag[cursor:cursor + count].view_as(
                        param
                    )
                cursor += count

    def _before_training_epoch(self) -> None:
        self._batch_has_mixed_replay = False
        self._batch_replay_count = 0
        if self._ewc_target_scale <= 0.0:
            self._ewc_effective_scale = 0.0
            return
        self._epoch_index += 1
        warmup_epochs = max(1, self.train_epochs)
        self._ewc_effective_scale = self._ewc_target_scale * (
            float(self._epoch_index) / float(warmup_epochs)
        )

    def _predict_eval(self, x) -> Tensor:
        if (
            self.use_nme_classifier
            and self.state is not None
            and self.state.class_feature_means
            and hasattr(self.model, "embed")
        ):
            features = self.model.embed(x)
            if self.normalize_nme_features:
                features = F.normalize(features, p=2, dim=1)
            num_classes = 0
            classifier = getattr(self.model, "classifier", None)
            if classifier is not None and hasattr(classifier, "num_classes"):
                num_classes = int(classifier.num_classes)
            else:
                num_classes = max(int(c) for c in self.state.class_feature_means) + 1
            logits = torch.full(
                (features.shape[0], num_classes),
                fill_value=-1e9,
                device=features.device,
                dtype=features.dtype,
            )
            for class_id, mean_np in self.state.class_feature_means.items():
                if int(class_id) >= num_classes:
                    continue
                mean = torch.as_tensor(mean_np, device=features.device, dtype=features.dtype)
                if self.normalize_nme_features:
                    mean = F.normalize(mean.unsqueeze(0), p=2, dim=1).squeeze(0)
                dist = (features - mean.unsqueeze(0)).pow(2).sum(dim=1)
                logits[:, int(class_id)] = -dist
            return logits
        return super()._predict_eval(x)

    def _before_training_iteration(self, x, y) -> None:
        self._batch_has_mixed_replay = False
        self._batch_replay_count = 0
        if not self.mix_replay_into_batches:
            return
        replay_batch = self._sample_replay_batch()
        if replay_batch is None:
            return
        replay_x, replay_y = replay_batch
        self.mb_x = self._concat_inputs(self.mb_x, replay_x)
        self.mb_y = torch.cat([self.mb_y, replay_y], dim=0)
        self._batch_has_mixed_replay = True
        self._batch_replay_count = int(replay_y.shape[0])

    def _compute_loss(self, x, y, logits) -> Tensor:
        ce_logits, ce_targets = self._masked_logits_and_targets(
            logits, y, self.active_classes
        )
        loss = self.ce_scale * self.criterion(ce_logits, ce_targets)

        if self.state is not None and self._ewc_effective_scale > 0.0:
            kfac_pen = torch.tensor(0.0, device=self.device)
            for name, param in self.model.named_parameters():
                if not param.requires_grad or name not in self._ref_params:
                    continue
                d_w = param - self._ref_params[name]
                layer_name = name.replace(".weight", "")
                if (
                    name in self._kfac_names
                    and layer_name in self._kfac_A
                    and layer_name in self._kfac_G
                ):
                    A = self._kfac_A[layer_name]
                    G = self._kfac_G[layer_name]
                    if d_w.dim() == 2:
                        kfac_pen = kfac_pen + (G @ d_w @ A * d_w).sum()
                    elif d_w.dim() == 4:
                        d_w_2d = d_w.reshape(d_w.shape[0], -1)
                        kfac_pen = kfac_pen + (G @ d_w_2d @ A * d_w_2d).sum()
                    else:
                        kfac_pen = kfac_pen + (
                            self._fisher_splits.get(name, torch.zeros_like(d_w))
                            * d_w.pow(2)
                        ).sum()
                elif name in self._fisher_splits:
                    kfac_pen = kfac_pen + (
                        self._fisher_splits[name] * d_w.pow(2)
                    ).sum()
            kfac_pen = kfac_pen / max(self._fisher_trace, 1e-8)
            loss = loss + self._ewc_effective_scale * kfac_pen

        if self._old_model is not None and self.kd_alpha > 0.0:
            with torch.no_grad():
                old_logits = self._old_model(x)
            seen_classes = []
            if self.state is not None:
                seen_classes = sorted(
                    int(class_id)
                    for class_id in self.state.label_counts
                    if 0 <= int(class_id) < logits.shape[1]
                )
            if seen_classes:
                seen_index = torch.tensor(
                    seen_classes, device=logits.device, dtype=torch.long
                )
                temperature = self.kd_temperature
                kd_loss = F.kl_div(
                    F.log_softmax(logits.index_select(1, seen_index) / temperature, dim=1),
                    F.softmax(old_logits.index_select(1, seen_index) / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = loss + self.kd_alpha * kd_loss

        if (
            self._old_model is not None
            and self.feature_kd_alpha > 0.0
            and hasattr(self.model, "embed")
            and hasattr(self._old_model, "embed")
        ):
            new_features = self.model.embed(x)
            with torch.no_grad():
                old_features = self._old_model.embed(x)
            loss = loss + self.feature_kd_alpha * F.mse_loss(new_features, old_features)

        replay_batch = None if self._batch_has_mixed_replay else self._sample_replay_batch()
        if replay_batch is not None:
            replay_x, replay_targets = replay_batch
            replay_logits_full = self.model(replay_x)
            replay_logits, replay_targets = self._masked_logits_and_targets(
                replay_logits_full, replay_targets, self.active_classes
            )
            loss = loss + self.replay_alpha * self.criterion(replay_logits, replay_targets)
            if (
                self._old_model is not None
                and self.replay_kd_alpha > 0.0
            ):
                with torch.no_grad():
                    old_replay_logits = self._old_model(replay_x)
                replay_seen_classes = sorted(
                    int(c)
                    for c in self.replay_memory
                    if 0 <= int(c) < replay_logits_full.shape[1]
                )
                replay_seen_index = torch.tensor(
                    replay_seen_classes,
                    device=replay_logits_full.device,
                    dtype=torch.long,
                )
                if replay_seen_index.numel() > 0:
                    temperature = self.kd_temperature
                    replay_kd = F.kl_div(
                        F.log_softmax(
                            replay_logits_full.index_select(1, replay_seen_index)
                            / temperature,
                            dim=1,
                        ),
                        F.softmax(
                            old_replay_logits.index_select(1, replay_seen_index)
                            / temperature,
                            dim=1,
                        ),
                        reduction="batchmean",
                    ) * (temperature * temperature)
                    loss = loss + self.replay_kd_alpha * replay_kd
                if (
                    self.feature_kd_alpha > 0.0
                    and hasattr(self.model, "embed")
                    and hasattr(self._old_model, "embed")
                ):
                    new_replay_features = self.model.embed(replay_x)
                    with torch.no_grad():
                        old_replay_features = self._old_model.embed(replay_x)
                    loss = loss + self.feature_kd_alpha * F.mse_loss(
                        new_replay_features, old_replay_features
                    )

        return loss

    def _concat_inputs(self, current: Any, replay: Any) -> Any:
        if isinstance(current, Tensor) and isinstance(replay, Tensor):
            return torch.cat([current, replay], dim=0)
        if isinstance(current, dict) and isinstance(replay, dict):
            return {
                key: self._concat_inputs(current[key], replay[key])
                for key in current
            }
        if isinstance(current, tuple) and isinstance(replay, tuple):
            return tuple(
                self._concat_inputs(cur_item, rep_item)
                for cur_item, rep_item in zip(current, replay)
            )
        if isinstance(current, list) and isinstance(replay, list):
            return [
                self._concat_inputs(cur_item, rep_item)
                for cur_item, rep_item in zip(current, replay)
            ]
        raise TypeError(
            f"Cannot concatenate replay batch of type {type(replay)!r} "
            f"into current batch of type {type(current)!r}."
        )

    def _clone_memory_value(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: self._clone_memory_value(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._clone_memory_value(v) for v in value)
        if isinstance(value, list):
            return [self._clone_memory_value(v) for v in value]
        return copy.deepcopy(value)

    def _slice_memory_input(self, batch_inputs: Any, index: int) -> Any:
        if isinstance(batch_inputs, Tensor):
            return batch_inputs[index]
        if isinstance(batch_inputs, dict):
            return {k: self._slice_memory_input(v, index) for k, v in batch_inputs.items()}
        if isinstance(batch_inputs, tuple):
            return tuple(self._slice_memory_input(v, index) for v in batch_inputs)
        if isinstance(batch_inputs, list):
            return [self._slice_memory_input(v, index) for v in batch_inputs]
        raise TypeError(f"Unsupported replay input type: {type(batch_inputs)!r}")

    def _stack_memory_inputs(self, samples: list[Any]) -> Any:
        first = samples[0]
        if isinstance(first, Tensor):
            return torch.stack(samples, dim=0)
        if isinstance(first, dict):
            return {
                k: self._stack_memory_inputs([sample[k] for sample in samples])
                for k in first
            }
        if isinstance(first, tuple):
            return tuple(
                self._stack_memory_inputs([sample[i] for sample in samples])
                for i in range(len(first))
            )
        if isinstance(first, list):
            return [
                self._stack_memory_inputs([sample[i] for sample in samples])
                for i in range(len(first))
            ]
        raise TypeError(f"Unsupported replay sample type: {type(first)!r}")

    def _update_replay_memory(self, loader) -> None:
        if self.replay_memory_per_class <= 0 or not self._current_classes:
            return

        target_classes = set(int(c) for c in self._current_classes)
        per_class_samples: dict[int, list[Any]] = {class_id: [] for class_id in target_classes}
        per_class_features: dict[int, list[Tensor]] = {class_id: [] for class_id in target_classes}
        use_feature_selection = hasattr(self.model, "embed")

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for mb_x, mb_y in loader:
                labels = mb_y.detach().cpu()
                batch_inputs = self._clone_memory_value(mb_x)
                features = None
                if use_feature_selection:
                    features = self.model.embed(_to_device(mb_x, self.device)).detach().cpu()
                    if self.normalize_nme_features:
                        features = F.normalize(features, p=2, dim=1)
                for idx, label in enumerate(labels.tolist()):
                    class_id = int(label)
                    if class_id not in target_classes:
                        continue
                    per_class_samples[class_id].append(
                        self._clone_memory_value(self._slice_memory_input(batch_inputs, idx))
                    )
                    if features is not None:
                        per_class_features[class_id].append(features[idx].clone())
        if was_training:
            self.model.train()

        for class_id in target_classes:
            samples = per_class_samples.get(class_id, [])
            if not samples:
                continue
            take = min(self.replay_memory_per_class, len(samples))
            features = per_class_features.get(class_id, [])
            if features:
                feature_tensor = torch.stack(features, dim=0)
                class_mean = feature_tensor.mean(dim=0, keepdim=True)
                distances = (feature_tensor - class_mean).pow(2).sum(dim=1)
                topk = torch.topk(distances, k=take, largest=False).indices.tolist()
            else:
                topk = torch.randperm(len(samples))[:take].tolist()
            self.replay_memory[class_id] = [
                self._clone_memory_value(samples[idx]) for idx in topk
            ]

    def _sample_replay_batch(self) -> tuple[Any, Tensor] | None:
        if self.replay_memory_per_class <= 0 or not self.replay_memory:
            return None

        old_classes = [
            int(c)
            for c in self.replay_memory
            if int(c) not in set(self._current_classes) and self.replay_memory.get(int(c))
        ]
        if not old_classes:
            return None

        max_total = max(1, self.replay_batch_size)
        samples_per_class = max(1, max_total // max(1, len(old_classes)))
        chosen_inputs: list[Any] = []
        chosen_labels: list[int] = []
        for class_id in old_classes:
            bucket = self.replay_memory.get(class_id, [])
            if not bucket:
                continue
            take = min(samples_per_class, len(bucket))
            order = torch.randperm(len(bucket))[:take].tolist()
            for idx in order:
                chosen_inputs.append(self._clone_memory_value(bucket[idx]))
                chosen_labels.append(class_id)

        if not chosen_inputs:
            return None

        replay_x = self._stack_memory_inputs(chosen_inputs)
        replay_x = _to_device(replay_x, self.device)
        replay_y = torch.tensor(chosen_labels, device=self.device, dtype=torch.long)
        return replay_x, replay_y

    def _update_class_feature_stats(self, loader) -> None:
        if self.state is None or not hasattr(self.model, "embed"):
            return

        self.model.eval()
        sums: dict[int, Tensor] = {}
        sq_sums: dict[int, Tensor] = {}
        counts: dict[int, int] = {}
        with torch.no_grad():
            for mb_x, mb_y in loader:
                mb_x = _to_device(mb_x, self.device)
                mb_y = mb_y.to(self.device)
                features = self.model.embed(mb_x).detach().cpu()
                labels = mb_y.detach().cpu()
                for class_id in labels.unique().tolist():
                    class_id = int(class_id)
                    mask = labels == class_id
                    class_features = features[mask]
                    if class_features.numel() == 0:
                        continue
                    class_sum = class_features.sum(dim=0)
                    class_sq_sum = class_features.pow(2).sum(dim=0)
                    sums[class_id] = sums.get(class_id, torch.zeros_like(class_sum)) + class_sum
                    sq_sums[class_id] = sq_sums.get(
                        class_id, torch.zeros_like(class_sq_sum)
                    ) + class_sq_sum
                    counts[class_id] = counts.get(class_id, 0) + int(class_features.shape[0])

        for class_id, count in counts.items():
            if count <= 0:
                continue
            mean = (sums[class_id] / count).numpy()
            second = (sq_sums[class_id] / count).numpy()
            var = np.maximum(second - mean ** 2, 1e-6)

            prev_count = int(self.state.label_counts.get(class_id, 0))
            old_mean = self.state.class_feature_means.get(class_id)
            old_var = self.state.class_feature_vars.get(class_id)
            if old_mean is not None and old_var is not None and prev_count > 0:
                total = prev_count + count
                blended_mean = (
                    prev_count * old_mean + count * mean
                ) / float(total)
                old_second = old_var + old_mean ** 2
                new_second = var + mean ** 2
                blended_second = (
                    prev_count * old_second + count * new_second
                ) / float(total)
                blended_var = np.maximum(blended_second - blended_mean ** 2, 1e-6)
                self.state.class_feature_means[class_id] = blended_mean.astype(np.float32)
                self.state.class_feature_vars[class_id] = blended_var.astype(np.float32)
            else:
                self.state.class_feature_means[class_id] = mean.astype(np.float32)
                self.state.class_feature_vars[class_id] = var.astype(np.float32)

    def _refresh_memory_feature_stats(self) -> None:
        if self.state is None or not hasattr(self.model, "embed") or not self.replay_memory:
            return

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for class_id, samples in self.replay_memory.items():
                if not samples:
                    continue
                batch_x = self._stack_memory_inputs(
                    [self._clone_memory_value(sample) for sample in samples]
                )
                batch_x = _to_device(batch_x, self.device)
                features = self.model.embed(batch_x).detach().cpu()
                if self.normalize_nme_features:
                    features = F.normalize(features, p=2, dim=1)
                mean = features.mean(dim=0).numpy()
                if features.shape[0] > 1:
                    var = features.var(dim=0, unbiased=False).numpy()
                else:
                    var = np.full_like(mean, 1e-6)
                self.state.class_feature_means[int(class_id)] = mean.astype(np.float32)
                self.state.class_feature_vars[int(class_id)] = np.maximum(
                    var, 1e-6
                ).astype(np.float32)
        if was_training:
            self.model.train()

    def _align_new_class_weights(self) -> None:
        classifier = getattr(self.model, "classifier", None)
        if classifier is None or not hasattr(classifier, "class_to_head"):
            return
        if not self._current_classes:
            return
        old_classes = [
            int(c)
            for c in classifier.class_to_head
            if int(c) not in set(self._current_classes)
        ]
        new_classes = [int(c) for c in self._current_classes if int(c) in classifier.class_to_head]
        if not old_classes or not new_classes:
            return

        old_norms = []
        new_norms = []
        for class_id in old_classes:
            old_norms.append(float(classifier.weight_vector(class_id).norm(p=2).item()))
        for class_id in new_classes:
            new_norms.append(float(classifier.weight_vector(class_id).norm(p=2).item()))
        mean_old = float(np.mean(old_norms)) if old_norms else 0.0
        mean_new = float(np.mean(new_norms)) if new_norms else 0.0
        if mean_old <= 0.0 or mean_new <= 0.0:
            return
        gamma = mean_old / mean_new
        with torch.no_grad():
            for class_id in new_classes:
                head_index, offset = classifier.class_to_head[class_id]
                classifier.heads[head_index].weight[offset].mul_(gamma)
                classifier.heads[head_index].bias[offset].mul_(gamma)

    def _rebalance_classifier(self) -> None:
        classifier = getattr(self.model, "classifier", None)
        if (
            classifier is None
            or self.state is None
            or not hasattr(self.model, "embed")
        ):
            return
        all_classes = sorted(int(c) for c in self.replay_memory if self.replay_memory.get(int(c)))
        if len(all_classes) < 2:
            return

        was_training = self.model.training
        self.model.train()
        backbone = getattr(self.model, "backbone", None)
        frozen_backbone_params = []
        if backbone is not None:
            for param in backbone.parameters():
                frozen_backbone_params.append((param, param.requires_grad))
                param.requires_grad = False

        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=self.classifier_balance_lr,
            momentum=0.0,
        )

        try:
            for _ in range(self.classifier_balance_steps):
                replay_batch = self._sample_balanced_replay_batch(all_classes)
                if replay_batch is None:
                    break
                replay_x, targets = replay_batch
                with torch.no_grad():
                    features = self.model.embed(replay_x)
                optimizer.zero_grad(set_to_none=True)
                logits = classifier(features)
                logits, targets = self._masked_logits_and_targets(
                    logits, targets, all_classes
                )
                loss = self.criterion(logits, targets)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
                optimizer.step()
        finally:
            for param, requires_grad in frozen_backbone_params:
                param.requires_grad = requires_grad
            if not was_training:
                self.model.eval()

    def _sample_balanced_replay_batch(self, classes: list[int]) -> tuple[Any, Tensor] | None:
        if not self.replay_memory:
            return None

        selected = [int(c) for c in classes if self.replay_memory.get(int(c))]
        if not selected:
            return None
        samples_per_class = max(1, self.replay_batch_size // max(1, len(selected)))
        chosen_inputs: list[Any] = []
        chosen_labels: list[int] = []
        for class_id in selected:
            bucket = self.replay_memory.get(class_id, [])
            if not bucket:
                continue
            take = min(samples_per_class, len(bucket))
            order = torch.randperm(len(bucket))[:take].tolist()
            for idx in order:
                chosen_inputs.append(self._clone_memory_value(bucket[idx]))
                chosen_labels.append(class_id)
        if not chosen_inputs:
            return None
        replay_x = self._stack_memory_inputs(chosen_inputs)
        replay_x = _to_device(replay_x, self.device)
        replay_y = torch.tensor(chosen_labels, device=self.device, dtype=torch.long)
        return replay_x, replay_y

    def _after_training_experience(self, experience) -> None:
        delta_time = time.time() - self._train_start
        loader = experience.train_dataloader(batch_size=self.train_mb_size)

        if self.state is None:
            self.state = self._fisher_computer.compute(self.model, loader, self.device)
            self.state.n_old = len(experience.train_dataset)
            self._update_class_feature_stats(loader)
            self._update_replay_memory(loader)
            self._shift_detector.update_state(loader, self.model, self.state, self.device)
            self._task0_time = delta_time
            self._task0_samples = self.state.n_old
            self._last_full_retrain_time = delta_time
            self.last_certificate = EquivalenceCertificate(
                epsilon_param=0.0,
                kl_bound=0.0,
                kl_bound_normalized=0.0,
                is_equivalent=True,
                shift_type="none",
                ece_before=0.0,
                ece_after=0.0,
                ece_delta=0.0,
                compute_ratio=1.0,
                n_old=0,
                n_new=self.state.n_old,
                ce_scale=1.0,
                ewc_scale=0.0,
                tier="initial",
            )
            return

        self._update_class_feature_stats(loader)
        self._update_replay_memory(loader)
        self._refresh_memory_feature_stats()
        self._rebalance_classifier()
        self._align_new_class_weights()
        fit_bias_correction = getattr(self, "_fit_bias_correction", None)
        if fit_bias_correction is not None:
            fit_bias_correction(loader)

        test_loader = experience.test_dataloader(batch_size=self.train_mb_size)
        ece_after = self._calibration.compute_ece(
            self.model, test_loader, self.device
        )
        n_new = len(experience.train_dataset)
        n_old = self.state.n_old
        n_total = n_old + n_new

        if self._task0_samples > 0 and self._task0_time > 0:
            estimated_full_time = self._task0_time * (
                float(n_total) / float(self._task0_samples)
            )
        else:
            estimated_full_time = self._last_full_retrain_time

        self.last_certificate = self._cert_computer.compute(
            model=self.model,
            state=self.state,
            new_loader=loader,
            shift_type=self.shift_type,
            fisher_computer=self._fisher_computer,
            ece_before=self._ece_before,
            ece_after=ece_after,
            delta_time=delta_time,
            full_retrain_time=estimated_full_time,
            device=self.device,
            n_old=n_old,
            n_new=n_new,
            ce_scale=self.ce_scale,
            ewc_scale=self.ewc_scale,
        )

        new_state = self._fisher_computer.compute(self.model, loader, self.device)
        alpha = float(n_new) / float(max(n_total, 1))
        for name, value in new_state.kfac_A.items():
            if name in self.state.kfac_A:
                self.state.kfac_A[name] = (1 - alpha) * self.state.kfac_A[name] + alpha * value
            else:
                self.state.kfac_A[name] = value
        for name, value in new_state.kfac_G.items():
            if name in self.state.kfac_G:
                self.state.kfac_G[name] = (1 - alpha) * self.state.kfac_G[name] + alpha * value
            else:
                self.state.kfac_G[name] = value
        self.state.kfac_param_names = sorted(
            set(self.state.kfac_param_names) | set(new_state.kfac_param_names)
        )
        self.state.theta_ref = {
            name: param.detach().cpu().numpy().copy()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        if (
            new_state.fisher_diag is not None
            and self.state.fisher_diag is not None
            and new_state.fisher_diag.shape == self.state.fisher_diag.shape
        ):
            self.state.fisher_diag = (
                (1 - alpha) * self.state.fisher_diag + alpha * new_state.fisher_diag
            )
        elif new_state.fisher_diag is not None:
            self.state.fisher_diag = new_state.fisher_diag

        self._shift_detector.update_state(loader, self.model, self.state, self.device)
        self.state.n_old = n_total
        self._old_model = None

# Update 10 @ 2026-03-28 18:02:27
# Update 17 @ 2026-03-29 08:44:55
# Update 19 @ 2026-03-28 13:56:17
# Update 22 @ 2026-03-29 07:33:04
# Update 22 @ 2026-03-29 08:14:39
# Update 33 @ 2026-03-29 06:39:07
# Update 7 @ 2026-03-29 01:08:54
# Update 17 @ 2026-03-28 12:15:43