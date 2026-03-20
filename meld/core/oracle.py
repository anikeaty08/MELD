"""Safety oracle implementations."""

from __future__ import annotations

import math

import numpy as np

from ..interfaces.base import OracleEstimate, SafetyOracle, TaskSnapshot


class SpectralSafetyOracle(SafetyOracle):
    def __init__(self) -> None:
        self._calibration_history: list[float] = []
        self._last_pre_risk_estimate: OracleEstimate | None = None

    def pre_risk_estimate(self, snapshot: TaskSnapshot, train_config: object) -> OracleEstimate:
        total_steps = max(1, int(train_config.epochs) * int(snapshot.steps_per_epoch))
        spectral = float(
            snapshot.fisher_eigenvalue_max
            * float(train_config.lr)
            * math.sqrt(total_steps * snapshot.embedding_dim)
        )
        if snapshot.mean_gradient_norm > 0.0:
            data_dependent = float(
                snapshot.fisher_eigenvalue_max
                * snapshot.mean_gradient_norm
                * float(train_config.lr)
                * total_steps
            )
            value = min(spectral, data_dependent)
        else:
            value = spectral
        estimate = OracleEstimate(
            value=value,
            bound_type="empirical_spectral",
            calibrated=False,
            bound_is_formal=False,
        )
        self._last_pre_risk_estimate = estimate
        return estimate

    def empirical_calibrated_estimate(self, snapshot: TaskSnapshot, train_config: object) -> OracleEstimate:
        base = self.pre_risk_estimate(snapshot, train_config)
        if not self._calibration_history:
            return OracleEstimate(
                value=base.value,
                bound_type="empirical_spectral_calibrated",
                calibrated=True,
                bound_is_formal=False,
            )
        calibration = float(np.clip(np.mean(self._calibration_history), 0.05, 1.0))
        return OracleEstimate(
            value=base.value * calibration,
            bound_type="empirical_spectral_calibrated",
            calibrated=True,
            bound_is_formal=False,
        )

    def post_drift_realized(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> OracleEstimate:
        shared = sorted(set(snapshot_before.class_ids).intersection(snapshot_after.class_ids))
        if not shared:
            estimate = OracleEstimate(
                value=0.0,
                bound_type="realized_old_manifold_drift",
                calibrated=False,
                bound_is_formal=False,
            )
            return estimate

        drifts = []
        for class_id in shared:
            before = snapshot_before.class_means[class_id]
            after = snapshot_after.class_means[class_id]
            denom = float(np.linalg.norm(before)) + 1e-12
            drifts.append(float(np.linalg.norm(after - before) / denom))
        value = float(np.mean(drifts))
        if self._last_pre_risk_estimate is not None and self._last_pre_risk_estimate.value > 0.0:
            ratio = value / self._last_pre_risk_estimate.value
            self._calibration_history.append(float(np.clip(ratio, 0.05, 1.0)))
            self._calibration_history = self._calibration_history[-32:]
        return OracleEstimate(
            value=value,
            bound_type="realized_old_manifold_drift",
            calibrated=False,
            bound_is_formal=False,
        )

    def pac_style_gap(self, snapshot: TaskSnapshot, delta: float = 0.05) -> OracleEstimate:
        clipped_delta = float(np.clip(delta, 1e-12, 1.0 - 1e-12))
        n = max(1, int(snapshot.dataset_size))
        value = float(math.sqrt(math.log(1.0 / clipped_delta) / (2.0 * n)))
        return OracleEstimate(
            value=value,
            bound_type="pac_style_hoeffding",
            calibrated=False,
            bound_is_formal=True,
            delta=clipped_delta,
        )

    def pac_equivalence_bound(
        self,
        snapshot_before: TaskSnapshot,
        snapshot_after: TaskSnapshot | None = None,
        delta: float = 0.05,
    ) -> OracleEstimate:
        clipped_delta = float(np.clip(delta, 1e-12, 1.0 - 1e-12))
        n = max(1, int(snapshot_before.dataset_size))
        weights = self._importance_weights(snapshot_before)
        weight_var = float(np.var(weights)) if weights.size else 0.0
        iw_term = math.sqrt(max(weight_var, 0.0) / n)
        curvature_term = 0.0
        if snapshot_after is not None:
            curvature_term = snapshot_before.fisher_eigenvalue_max * self._parameter_shift_norm(
                snapshot_before,
                snapshot_after,
            )
        pac_term = math.sqrt(math.log(1.0 / clipped_delta) / (2.0 * n))
        value = float(iw_term + curvature_term + pac_term)
        return OracleEstimate(
            value=value,
            bound_type="pac_importance_weighted",
            calibrated=False,
            bound_is_formal=True,
            delta=clipped_delta,
        )

    def pac_equivalence_gap(self, snapshot: TaskSnapshot | None = None, confidence: float = 0.95) -> tuple[float, float]:
        if snapshot is None:
            if self._last_pre_risk_estimate is None:
                return 0.0, float(np.clip(1.0 - confidence, 1e-6, 1.0))
            return self._last_pre_risk_estimate.value, float(np.clip(1.0 - confidence, 1e-6, 1.0))
        estimate = self.pac_style_gap(snapshot, delta=1.0 - confidence)
        return estimate.value, float(estimate.delta or (1.0 - confidence))

    def pre_bound(self, snapshot: TaskSnapshot, train_config: object) -> float:
        return self.pre_risk_estimate(snapshot, train_config).value

    def post_bound(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> float:
        return self.post_drift_realized(snapshot_before, snapshot_after).value

    @staticmethod
    def _importance_weights(snapshot: TaskSnapshot) -> np.ndarray:
        if not snapshot.importance_weights:
            return np.array([], dtype=np.float32)
        arrays = [
            np.asarray(values, dtype=np.float32).reshape(-1)
            for values in snapshot.importance_weights.values()
            if np.asarray(values).size > 0
        ]
        if not arrays:
            return np.array([], dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    @staticmethod
    def _parameter_shift_norm(snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> float:
        if not snapshot_before.parameter_reference or not snapshot_after.parameter_reference:
            return 0.0
        total = 0.0
        for before, after in zip(snapshot_before.parameter_reference, snapshot_after.parameter_reference):
            delta = np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)
            total += float(np.sum(delta * delta))
        return float(math.sqrt(total))
