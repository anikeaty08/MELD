import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from meld.api import MELDConfig, TrainConfig
from meld.delta import DeltaModel
from meld.interfaces.base import Decision, DriftResult, OracleEstimate, TaskSnapshot, TrainArtifacts


def test_delta_model_from_scratch_exposes_public_api():
    model = DeltaModel.from_scratch(
        num_classes=2,
        backbone="resnet20",
        prefer_cuda=False,
        train_config=TrainConfig(backbone="resnet20", epochs=1, batch_size=2),
    )

    summary = model.summary()

    assert "DeltaModel" in summary
    assert "classes=2" in summary


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(12, 4)
        self.out_dim = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.flatten(x))


class _FakeOracle:
    def __init__(self, pac_value: float) -> None:
        self.pac_value = pac_value

    def pre_risk_estimate(self, snapshot: TaskSnapshot, train_config: object) -> OracleEstimate:
        return OracleEstimate(0.0, "empirical_spectral", False, False)

    def pac_style_gap(self, snapshot: TaskSnapshot) -> OracleEstimate:
        return OracleEstimate(self.pac_value, "pac_style_hoeffding", False, True, delta=0.05)

    def post_drift_realized(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> OracleEstimate:
        return OracleEstimate(0.0, "realized_old_manifold_drift", False, False)

    def pac_equivalence_bound(
        self,
        snapshot_before: TaskSnapshot,
        snapshot_after: TaskSnapshot,
    ) -> OracleEstimate:
        return OracleEstimate(0.0, "pac_importance_weighted", False, True, delta=0.05)


class _FakeUpdater:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, model: nn.Module, new_data_loader: DataLoader, snapshot: TaskSnapshot | None, config: object):
        self.calls += 1
        return model, TrainArtifacts(
            epochs_run=1,
            lambda_schedule=[],
            geometry_loss_per_epoch=[],
            ewc_loss_per_epoch=[],
            ce_loss_per_epoch=[0.0],
            wall_time_seconds=0.0,
            train_accuracy_per_epoch=[1.0],
        )


class _FakeSnapshotStrategy:
    def __init__(self, snapshot: TaskSnapshot) -> None:
        self.snapshot = snapshot

    def capture(self, model: nn.Module, dataloader: DataLoader, class_ids: list[int], task_id: int) -> TaskSnapshot:
        return self.snapshot


class _FakeCorrector:
    def correct(self, model: nn.Module, snapshot: TaskSnapshot) -> nn.Module:
        return model


class _FakeDriftDetector:
    def detect(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> DriftResult:
        return DriftResult(shift_score=0.0, shift_detected=False, per_class_drift={}, severity="none")


class _FakePolicy:
    def decide(self, pre_bound: float, post_bound: float, drift_result: DriftResult, config: object) -> Decision:
        return Decision(
            state="SAFE_DELTA",
            pre_bound=pre_bound,
            post_bound=post_bound,
            bound_held=True,
            shift_score=drift_result.shift_score,
            shift_detected=drift_result.shift_detected,
            reason="ok",
            compute_savings_percent=0.0,
            confidence=1.0,
            recommended_action="delta_update",
        )


def _snapshot() -> TaskSnapshot:
    return TaskSnapshot(
        task_id=0,
        class_ids=[0],
        class_means={0: np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)},
        class_covs={0: np.ones(4, dtype=np.float32)},
        class_anchors={0: np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)},
        class_anchor_inputs={0: np.zeros((1, 3, 2, 2), dtype=np.float32)},
        class_anchor_logits={0: np.array([[1.0]], dtype=np.float32)},
        classifier_norms={0: 1.0},
        fisher_diagonal=np.ones(4, dtype=np.float32) * 0.1,
        fisher_eigenvalue_max=0.1,
        mean_gradient_norm=0.1,
        timestamp=0.0,
        embedding_dim=4,
        dataset_size=1,
        steps_per_epoch=1,
        parameter_reference=[np.zeros((4,), dtype=np.float32)],
    )


def _delta_model_with_tolerance(pac_gate_tolerance: float) -> tuple[DeltaModel, _FakeUpdater]:
    train_config = TrainConfig(
        backbone="custom",
        epochs=1,
        batch_size=1,
        pac_gate_tolerance=pac_gate_tolerance,
    )
    model = DeltaModel.from_backbone(
        _TinyBackbone(),
        out_dim=4,
        num_classes=1,
        prefer_cuda=False,
        train_config=train_config,
    )
    snapshot = _snapshot()
    updater = _FakeUpdater()
    model._snapshot = snapshot
    model._snapshot_strategy = _FakeSnapshotStrategy(snapshot)
    model._safety_oracle = _FakeOracle(pac_value=0.2)
    model._updater = updater
    model._corrector = _FakeCorrector()
    model._drift_detector = _FakeDriftDetector()
    model._deploy_policy = _FakePolicy()
    return model, updater


def test_delta_model_respects_train_config_pac_gate_tolerance():
    inputs = torch.randn(1, 3, 2, 2)
    targets = torch.tensor([0], dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)

    gated_model, gated_updater = _delta_model_with_tolerance(0.1)
    gated_result = gated_model.update(loader)

    permissive_model, permissive_updater = _delta_model_with_tolerance(0.5)
    permissive_result = permissive_model.update(loader)

    assert gated_result.decision == "BOUND_EXCEEDED"
    assert gated_updater.calls == 0
    assert permissive_result.decision == "SAFE_DELTA"
    assert permissive_updater.calls == 1


def test_framework_defaults_use_consistent_pac_gate_tolerance():
    assert TrainConfig().pac_gate_tolerance == 0.5
    assert MELDConfig().pac_gate_tolerance == 0.5
