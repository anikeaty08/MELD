import numpy as np

from meld.core.oracle import SpectralSafetyOracle
from meld.interfaces.base import TaskSnapshot


class _Train:
    epochs = 5
    lr = 0.1


def test_pre_bound_uses_snapshot_geometry():
    oracle = SpectralSafetyOracle()
    snapshot = TaskSnapshot(
        task_id=0,
        class_ids=[0],
        class_means={0: np.array([1.0, 0.0])},
        class_covs={0: np.array([1.0, 1.0])},
        class_anchors={0: np.array([[1.0, 0.0]])},
        class_anchor_inputs={0: np.zeros((1, 3, 32, 32), dtype=np.float32)},
        class_anchor_logits={0: np.array([[1.0]])},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.array([0.2, 0.3]),
        fisher_eigenvalue_max=0.3,
        mean_gradient_norm=0.25,
        timestamp=0.0,
        embedding_dim=2,
        dataset_size=32,
        steps_per_epoch=4,
        parameter_reference=[],
    )
    estimate = oracle.pre_risk_estimate(snapshot, _Train())
    assert estimate.value > 0.0
    assert estimate.bound_type == "empirical_spectral"
    assert estimate.bound_is_formal is False


def test_post_bound_is_zero_for_identical_snapshots():
    oracle = SpectralSafetyOracle()
    snapshot = TaskSnapshot(
        task_id=0,
        class_ids=[0],
        class_means={0: np.array([1.0, 0.0])},
        class_covs={0: np.array([1.0, 1.0])},
        class_anchors={0: np.array([[1.0, 0.0]])},
        class_anchor_inputs={0: np.zeros((1, 3, 32, 32), dtype=np.float32)},
        class_anchor_logits={0: np.array([[1.0]])},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.array([0.2, 0.3]),
        fisher_eigenvalue_max=0.3,
        mean_gradient_norm=0.25,
        timestamp=0.0,
        embedding_dim=2,
        dataset_size=32,
        steps_per_epoch=4,
        parameter_reference=[],
    )
    realized = oracle.post_drift_realized(snapshot, snapshot)
    assert realized.value == 0.0
    assert realized.bound_type == "realized_old_manifold_drift"


def test_pac_style_gap_reports_formal_metadata():
    oracle = SpectralSafetyOracle()
    snapshot = TaskSnapshot(
        task_id=0,
        class_ids=[0],
        class_means={0: np.array([1.0, 0.0])},
        class_covs={0: np.array([1.0, 1.0])},
        class_anchors={0: np.array([[1.0, 0.0]])},
        class_anchor_inputs={0: np.zeros((1, 3, 32, 32), dtype=np.float32)},
        class_anchor_logits={0: np.array([[1.0]])},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.array([0.2, 0.3]),
        fisher_eigenvalue_max=0.3,
        mean_gradient_norm=0.25,
        timestamp=0.0,
        embedding_dim=2,
        dataset_size=32,
        steps_per_epoch=4,
        parameter_reference=[],
    )
    estimate = oracle.pac_style_gap(snapshot, delta=0.05)
    assert estimate.value > 0.0
    assert estimate.bound_type == "pac_style_hoeffding"
    assert estimate.bound_is_formal is True


def test_pac_equivalence_bound_uses_importance_weights_and_shift():
    oracle = SpectralSafetyOracle()
    before = TaskSnapshot(
        task_id=0,
        class_ids=[0],
        class_means={0: np.array([1.0, 0.0])},
        class_covs={0: np.array([1.0, 1.0])},
        class_anchors={0: np.array([[1.0, 0.0]])},
        class_anchor_inputs={0: np.zeros((1, 3, 32, 32), dtype=np.float32)},
        class_anchor_logits={0: np.array([[1.0]])},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.array([0.2, 0.3]),
        fisher_eigenvalue_max=0.3,
        mean_gradient_norm=0.25,
        timestamp=0.0,
        embedding_dim=2,
        dataset_size=32,
        steps_per_epoch=4,
        parameter_reference=[np.array([0.0, 0.0]), np.array([1.0])],
        importance_weights={0: np.array([0.5, 1.5, 1.0], dtype=np.float32)},
    )
    after = TaskSnapshot(
        task_id=1,
        class_ids=[0],
        class_means={0: np.array([1.1, 0.1])},
        class_covs={0: np.array([1.0, 1.0])},
        class_anchors={0: np.array([[1.1, 0.1]])},
        class_anchor_inputs={0: np.zeros((1, 3, 32, 32), dtype=np.float32)},
        class_anchor_logits={0: np.array([[1.0]])},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.array([0.2, 0.3]),
        fisher_eigenvalue_max=0.3,
        mean_gradient_norm=0.25,
        timestamp=1.0,
        embedding_dim=2,
        dataset_size=32,
        steps_per_epoch=4,
        parameter_reference=[np.array([0.1, 0.0]), np.array([1.1])],
    )
    estimate = oracle.pac_equivalence_bound(before, after, delta=0.05)
    expected_n = 3
    expected_var = float(np.var(np.array([0.5, 1.5, 1.0], dtype=np.float32)))
    expected_shift = float(np.sqrt((0.1**2) + (0.1**2)))
    expected = (
        np.sqrt(expected_var / expected_n)
        + (0.3 * expected_shift)
        + np.sqrt(np.log(1.0 / 0.05) / (2.0 * expected_n))
    )
    assert estimate.value > 0.0
    assert np.isclose(estimate.value, expected)
    assert estimate.bound_type == "pac_importance_weighted"
    assert estimate.bound_is_formal is True
    assert estimate.delta == 0.05
