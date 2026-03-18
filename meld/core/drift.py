"""Drift detection for MELD."""

from __future__ import annotations

import numpy as np

from ..interfaces.base import DriftDetector, DriftResult, TaskSnapshot


class KLManifoldDriftDetector(DriftDetector):
    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def detect(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> DriftResult:
        shared = sorted(set(snapshot_before.class_ids).intersection(snapshot_after.class_ids))
        per_class_drift: dict[int, float] = {}
        for class_id in shared:
            mu_before = snapshot_before.class_means[class_id]
            mu_after = snapshot_after.class_means[class_id]
            cov_before = np.clip(snapshot_before.class_covs[class_id], 1e-6, None)
            cov_after = np.clip(snapshot_after.class_covs[class_id], 1e-6, None)
            value = 0.5 * np.sum(np.log(cov_after / cov_before) + (cov_before + (mu_before - mu_after) ** 2) / cov_after - 1.0)
            per_class_drift[class_id] = float(value)

        shift_score = float(np.mean(list(per_class_drift.values()))) if per_class_drift else 0.0
        if shift_score <= self.threshold:
            severity = "none"
        elif shift_score <= 2.0 * self.threshold:
            severity = "minor"
        else:
            severity = "critical"
        return DriftResult(
            shift_score=shift_score,
            shift_detected=shift_score > self.threshold,
            per_class_drift=per_class_drift,
            severity=severity,
        )
