"""Safety oracle implementations."""

from __future__ import annotations

import math

import numpy as np

from ..interfaces.base import SafetyOracle, TaskSnapshot


class SpectralSafetyOracle(SafetyOracle):
    def pre_bound(self, snapshot: TaskSnapshot, train_config: object) -> float:
        total_steps = max(1, int(train_config.epochs) * int(snapshot.steps_per_epoch))
        return float(snapshot.fisher_eigenvalue_max * float(train_config.lr) * math.sqrt(total_steps * snapshot.embedding_dim))

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
        return float(np.mean(drifts))
