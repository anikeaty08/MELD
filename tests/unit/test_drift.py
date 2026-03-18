import numpy as np

from meld.core.drift import KLManifoldDriftDetector
from meld.interfaces.base import TaskSnapshot


def _snapshot(mean: float) -> TaskSnapshot:
    return TaskSnapshot(
        task_id=0,
        class_ids=[0],
        class_means={0: np.array([mean, 0.0])},
        class_covs={0: np.array([1.0, 1.0])},
        class_anchors={0: np.array([[mean, 0.0]])},
        class_anchor_logits={0: np.array([[1.0]])},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.array([0.1, 0.2]),
        fisher_eigenvalue_max=0.2,
        timestamp=0.0,
        embedding_dim=2,
        dataset_size=16,
        steps_per_epoch=2,
        parameter_reference=[],
    )


def test_drift_detector_flags_critical_shift():
    detector = KLManifoldDriftDetector(threshold=0.3)
    drift = detector.detect(_snapshot(0.0), _snapshot(2.0))
    assert drift.shift_detected is True
    assert drift.severity == "critical"
