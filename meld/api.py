"""Public API for MELD."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmarks.runner import BenchmarkRunner
from .core.corrector import AnalyticNormCorrector
from .core.drift import KLManifoldDriftDetector
from .core.oracle import SpectralSafetyOracle
from .core.policy import FourStateDeployPolicy
from .core.snapshot import FisherManifoldSnapshot
from .core.updater import GeometryConstrainedUpdater


@dataclass(slots=True)
class TrainConfig:
    # Architecture — auto-upgraded to resnet56 for CIFAR-100 in runner
    backbone: str = "resnet32"
    pretrained_backbone: bool = False

    # Optimization
    epochs: int = 30           # 30 epochs with OneCycleLR ≈ 200 flat-LR epochs
    batch_size: int = 128      # larger batch → faster + more stable
    lr: float = 0.1            # OneCycleLR peak LR
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # Loss weights
    lambda_geometry: float = 0.5
    lambda_ewc: float = 0.3
    lambda_kd: float = 1.0
    kd_temperature: float = 2.0
    geometry_decay: float = 0.3

    # Fisher
    use_ema_fisher: bool = True
    fisher_ema_decay: float = 0.9
    auto_derive_hparams: bool = False
    protection_level: float = 0.5

    # Baseline
    full_retrain_epochs: int | None = None

    # Workers
    num_workers: int = 2


@dataclass(slots=True)
class MELDConfig:
    dataset: str = "CIFAR-10"
    num_tasks: int = 2
    classes_per_task: int = 5
    prefer_cuda: bool = False
    bound_tolerance: float = 10.0   # calibrated for normalized Fisher bounds
    shift_threshold: float = 0.3
    data_root: Path = Path("./data")
    seed: int = 7
    full_retrain_interval: int = 1
    train: TrainConfig = field(default_factory=TrainConfig)


def run(
    config: MELDConfig,
    results_path: str | None = None,
    *,
    snapshot_strategy: FisherManifoldSnapshot | None = None,
    safety_oracle: SpectralSafetyOracle | None = None,
    updater: GeometryConstrainedUpdater | None = None,
    corrector: AnalyticNormCorrector | None = None,
    drift_detector: KLManifoldDriftDetector | None = None,
    deploy_policy: FourStateDeployPolicy | None = None,
) -> dict[str, Any]:
    runner = BenchmarkRunner(
        config=config,
        snapshot_strategy=snapshot_strategy,
        safety_oracle=safety_oracle,
        updater=updater,
        corrector=corrector,
        drift_detector=drift_detector,
        deploy_policy=deploy_policy,
    )
    return runner.run(results_path=results_path)