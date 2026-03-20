import numpy as np
import torch

from meld.core.drift import CompositeDriftDetector
from meld.core.weighter import KLIEPWeighter
from meld.interfaces.base import TaskSnapshot


def _snapshot(mean: float, input_shift: float = 0.0) -> TaskSnapshot:
    samples = np.stack(
        [
            np.array([input_shift, 0.1], dtype=np.float32),
            np.array([input_shift + 0.2, -0.1], dtype=np.float32),
            np.array([input_shift - 0.2, 0.0], dtype=np.float32),
        ],
        axis=0,
    )
    return TaskSnapshot(
        task_id=0,
        class_ids=[0],
        class_means={0: np.array([mean, 0.0], dtype=np.float32)},
        class_covs={0: np.array([1.0, 1.0], dtype=np.float32)},
        class_anchors={0: np.array([[mean, 0.0]], dtype=np.float32)},
        class_anchor_logits={0: np.array([[1.0]], dtype=np.float32)},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.array([0.1, 0.2], dtype=np.float32),
        fisher_eigenvalue_max=0.2,
        mean_gradient_norm=0.1,
        timestamp=0.0,
        embedding_dim=2,
        dataset_size=16,
        steps_per_epoch=2,
        parameter_reference=[],
        input_feature_mean=samples.mean(axis=0),
        input_feature_var=samples.var(axis=0) + 1e-6,
        input_feature_samples=samples,
    )


def test_kliep_weighter_returns_bounded_weights():
    weighter = KLIEPWeighter()
    snapshot = _snapshot(0.0)
    new_embeddings = torch.tensor(
        [[0.1, 0.0], [0.2, 0.1], [2.0, 2.0], [2.1, 1.9]],
        dtype=torch.float32,
    )
    new_targets = torch.tensor([1, 1, 2, 2], dtype=torch.long)

    grouped = weighter.fit(new_embeddings, snapshot, new_targets)
    weights = weighter.score(new_embeddings)

    assert set(grouped) == {1, 2}
    assert torch.all(weights > 0)
    assert float(weights.min().item()) >= 0.05 - 1e-6
    assert float(weights.max().item()) <= 20.0 + 1e-6
    assert 0.8 <= float(weights.mean().item()) <= 1.2


def test_composite_drift_detector_catches_input_shift():
    detector = CompositeDriftDetector(threshold=0.3, mmd_threshold=0.01, cusum_threshold=0.05)
    drift = detector.detect(_snapshot(0.0, input_shift=0.0), _snapshot(0.0, input_shift=2.0))

    assert drift.shift_detected is True
    assert drift.detector_scores["mmd_input"] > 0.0 or drift.detector_scores["cusum_input"] > 0.0
    assert drift.input_shift_score > 0.0
