"""Safety oracle implementations."""

from __future__ import annotations

import math

import numpy as np

from ..interfaces.base import SafetyOracle, TaskSnapshot


class SpectralSafetyOracle(SafetyOracle):
    def __init__(self) -> None:
        self._calibration_history: list[float] = []
        self._last_pre_bound: float | None = None

    def pre_bound(self, snapshot: TaskSnapshot, train_config: object) -> float:
        total_steps = max(1, int(train_config.epochs) * int(snapshot.steps_per_epoch))
        spectral = float(snapshot.fisher_eigenvalue_max * float(train_config.lr) * math.sqrt(total_steps * snapshot.embedding_dim))
        if snapshot.mean_gradient_norm > 0.0:
            data_dependent = float(snapshot.fisher_eigenvalue_max * snapshot.mean_gradient_norm * float(train_config.lr) * total_steps)
            bound = min(spectral, data_dependent)
        else:
            bound = spectral
        if self._calibration_history:
            calibration = float(np.clip(np.mean(self._calibration_history), 0.05, 1.0))
            bound *= calibration
        self._last_pre_bound = bound
        return bound

    def post_bound(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> float:
        shared = sorted(set(snapshot_before.class_ids).intersection(snapshot_after.class_ids))
        if not shared:
            return 0.0
        drifts = []
        for class_id in shared:
            before = snapshot_before.class_means[class_id]
            after = snapshot_after.class_means[class_id]
            denom = float(np.linalg.norm(before)) + 1e-12
            drifts.append(float(np.linalg.norm(after - before) / denom))
        value = float(np.mean(drifts))
        if self._last_pre_bound and self._last_pre_bound > 0:
            self._calibration_history.append(min(1.0, value / self._last_pre_bound))
            self._calibration_history = self._calibration_history[-32:]
        return value
