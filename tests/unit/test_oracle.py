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
        class_anchor_logits={0: np.array([[1.0]])},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.array([0.2, 0.3]),
        fisher_eigenvalue_max=0.3,
        timestamp=0.0,
        embedding_dim=2,
        dataset_size=32,
        steps_per_epoch=4,
        parameter_reference=[],
    )
    value = oracle.pre_bound(snapshot, _Train())
    assert value > 0.0


def test_post_bound_is_zero_for_identical_snapshots():
    oracle = SpectralSafetyOracle()
    snapshot = TaskSnapshot(
        task_id=0,
        class_ids=[0],
        class_means={0: np.array([1.0, 0.0])},
        class_covs={0: np.array([1.0, 1.0])},
        class_anchors={0: np.array([[1.0, 0.0]])},
        class_anchor_logits={0: np.array([[1.0]])},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.array([0.2, 0.3]),
        fisher_eigenvalue_max=0.3,
        timestamp=0.0,
        embedding_dim=2,
        dataset_size=32,
        steps_per_epoch=4,
        parameter_reference=[],
    )
    assert oracle.post_bound(snapshot, snapshot) == 0.0
