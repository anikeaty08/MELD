"""Drift detection for MELD."""

from __future__ import annotations

import math

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
            value = 0.5 * np.mean(
                np.log(cov_after / cov_before)
                + (cov_before + (mu_before - mu_after) ** 2) / cov_after
                - 1.0
            )
            per_class_drift[class_id] = float(value)

        shift_score = float(np.mean(list(per_class_drift.values()))) if per_class_drift else 0.0
        severity = _severity_from_ratio(shift_score / max(self.threshold, 1e-12))
        return DriftResult(
            shift_score=shift_score,
            shift_detected=shift_score > self.threshold,
            per_class_drift=per_class_drift,
            severity=severity,
            detector_scores={"kl_manifold": shift_score},
            input_shift_score=0.0,
        )


class MMDDriftDetector(DriftDetector):
    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def detect(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> DriftResult:
        x = snapshot_before.input_feature_samples
        y = snapshot_after.input_feature_samples
        score = self._mmd_rbf(x, y)
        severity = _severity_from_ratio(score / max(self.threshold, 1e-12))
        return DriftResult(
            shift_score=score,
            shift_detected=score > self.threshold,
            per_class_drift={},
            severity=severity,
            detector_scores={"mmd_input": score},
            input_shift_score=score,
        )

    def _mmd_rbf(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0 or y.size == 0:
            return 0.0
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        bandwidth = self._median_bandwidth(x, y)
        gamma = 1.0 / max(2.0 * bandwidth * bandwidth, 1e-12)
        k_xx = np.exp(-gamma * self._squared_distances(x, x))
        k_yy = np.exp(-gamma * self._squared_distances(y, y))
        k_xy = np.exp(-gamma * self._squared_distances(x, y))
        return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())

    @staticmethod
    def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)

    @staticmethod
    def _median_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
        combined = np.concatenate((x, y), axis=0)
        if combined.shape[0] < 2:
            return 1.0
        dists = MMDDriftDetector._squared_distances(combined, combined)
        upper = dists[np.triu_indices_from(dists, k=1)]
        upper = upper[upper > 0]
        if upper.size == 0:
            return 1.0
        return float(math.sqrt(np.median(upper)))


class CUSUMDriftDetector(DriftDetector):
    def __init__(self, threshold: float = 0.5, slack: float = 0.05) -> None:
        self.threshold = threshold
        self.slack = slack
        self._accumulator = 0.0

    def detect(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> DriftResult:
        if snapshot_before.input_feature_mean.size == 0 or snapshot_after.input_feature_mean.size == 0:
            score = 0.0
        else:
            pooled = np.sqrt(
                np.clip(snapshot_before.input_feature_var, 1e-6, None)
                + np.clip(snapshot_after.input_feature_var, 1e-6, None)
            )
            standardized_shift = float(
                np.mean(np.abs(snapshot_after.input_feature_mean - snapshot_before.input_feature_mean) / pooled)
            )
            self._accumulator = max(0.0, self._accumulator + standardized_shift - self.slack)
            score = self._accumulator
        severity = _severity_from_ratio(score / max(self.threshold, 1e-12))
        return DriftResult(
            shift_score=score,
            shift_detected=score > self.threshold,
            per_class_drift={},
            severity=severity,
            detector_scores={"cusum_input": score},
            input_shift_score=score,
        )


class CompositeDriftDetector(DriftDetector):
    def __init__(
        self,
        threshold: float = 0.3,
        mmd_threshold: float = 0.05,
        cusum_threshold: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.manifold = KLManifoldDriftDetector(threshold=threshold)
        self.mmd = MMDDriftDetector(threshold=mmd_threshold)
        self.cusum = CUSUMDriftDetector(threshold=cusum_threshold)

    def detect(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> DriftResult:
        kl_result = self.manifold.detect(snapshot_before, snapshot_after)
        mmd_result = self.mmd.detect(snapshot_before, snapshot_after)
        cusum_result = self.cusum.detect(snapshot_before, snapshot_after)

        normalized = [
            kl_result.shift_score / max(self.manifold.threshold, 1e-12),
            mmd_result.shift_score / max(self.mmd.threshold, 1e-12),
            cusum_result.shift_score / max(self.cusum.threshold, 1e-12),
        ]
        ratio = max(normalized)
        composite_score = ratio * self.threshold
        severity = _severity_from_ratio(ratio)
        detector_scores = {
            "kl_manifold": kl_result.shift_score,
            "mmd_input": mmd_result.shift_score,
            "cusum_input": cusum_result.shift_score,
        }
        return DriftResult(
            shift_score=float(composite_score),
            shift_detected=ratio > 1.0,
            per_class_drift=kl_result.per_class_drift,
            severity=severity,
            detector_scores=detector_scores,
            input_shift_score=float(max(mmd_result.shift_score, cusum_result.shift_score)),
        )


def _severity_from_ratio(ratio: float) -> str:
    if ratio <= 1.0:
        return "none"
    if ratio <= 2.0:
        return "minor"
    return "critical"
