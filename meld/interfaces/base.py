"""Core interfaces and shared dataclasses for MELD."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class TaskSnapshot:
    task_id: int
    class_ids: list[int]
    class_means: dict[int, np.ndarray]
    class_covs: dict[int, np.ndarray]
    classifier_norms: dict[int, float]
    fisher_diagonal: np.ndarray
    fisher_eigenvalue_max: float
    timestamp: float
    embedding_dim: int
    dataset_size: int
    steps_per_epoch: int
    parameter_reference: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True)
class DriftResult:
    shift_score: float
    shift_detected: bool
    per_class_drift: dict[int, float]
    severity: str


@dataclass(slots=True)
class Decision:
    state: str
    pre_bound: float
    post_bound: float
    bound_held: bool
    shift_score: float
    shift_detected: bool
    reason: str
    compute_savings_percent: float
    confidence: float
    recommended_action: str


@dataclass(slots=True)
class TrainArtifacts:
    epochs_run: int
    lambda_schedule: list[float]
    geometry_loss_per_epoch: list[float]
    ewc_loss_per_epoch: list[float]
    ce_loss_per_epoch: list[float]
    wall_time_seconds: float
    loss_history: list[float] = field(default_factory=list)
    skipped: bool = False


class SnapshotStrategy(ABC):
    @abstractmethod
    def capture(self, model: Any, dataloader: Any, class_ids: list[int], task_id: int) -> TaskSnapshot:
        """Capture a replay-free statistical snapshot of model state."""


class SafetyOracle(ABC):
    @abstractmethod
    def pre_bound(self, snapshot: TaskSnapshot, train_config: Any) -> float:
        """Compute a pre-training safety bound."""

    @abstractmethod
    def post_bound(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> float:
        """Compute the post-training realized drift."""


class ManifoldUpdater(ABC):
    @abstractmethod
    def update(
        self,
        model: Any,
        new_data_loader: Any,
        snapshot: TaskSnapshot | None,
        config: Any,
    ) -> tuple[Any, TrainArtifacts]:
        """Update a model using only new task data."""


class BiasCorrector(ABC):
    @abstractmethod
    def correct(self, model: Any, snapshot: TaskSnapshot) -> Any:
        """Apply bias correction without replay data."""


class DriftDetector(ABC):
    @abstractmethod
    def detect(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> DriftResult:
        """Measure class manifold shift."""


class DeployPolicy(ABC):
    @abstractmethod
    def decide(self, pre_bound: float, post_bound: float, drift_result: DriftResult, config: Any) -> Decision:
        """Produce a structured deployment decision."""
