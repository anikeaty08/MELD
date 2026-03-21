"""Public API for MELD."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmarks.runner import BenchmarkRunner
from .core.corrector import AnalyticNormCorrector
from .core.oracle import SpectralSafetyOracle
from .core.policy import FourStateDeployPolicy
from .core.snapshot import FisherManifoldSnapshot
from .core.updater import GeometryConstrainedUpdater
from .interfaces.base import DriftDetector, ManifoldUpdater


@dataclass(slots=True)
class TrainConfig:
    backbone: str = "auto"
    pretrained_backbone: bool = True
    incremental_strategy: str = "geometry"

    epochs: int = 30
    base_epochs: int | None = None
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    freeze_bn_stats: bool = False

    lambda_geometry: float = 5.0
    lambda_ewc: float = 1.0
    lambda_kd: float = 1.0
    kd_temperature: float = 2.0
    geometry_decay: float = 0.3
    analytic_ridge: float = 1e-3
    enable_importance_weighting: bool = True
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    max_grad_norm: float = 1.0
    use_imprinting: bool = True
    imprinting_max_samples_per_class: int = 64
    auto_scale_safe_update: bool = True
    min_safe_lr: float = 1e-5

    use_ema_fisher: bool = True
    fisher_ema_decay: float = 0.9
    auto_derive_hparams: bool = True
    protection_level: float = 0.5
    enable_grad_projection: bool = False

    # NLP settings — only used when backbone = "text_encoder"
    text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    full_retrain_epochs: int | None = None
    pac_gate_tolerance: float = 0.1
    num_workers: int = 0


@dataclass(slots=True)
class MELDConfig:
    dataset: str = "CIFAR-10"
    num_tasks: int = 2
    classes_per_task: int = 5
    prefer_cuda: bool = False
    bound_tolerance: float = 10.0
    pac_gate_tolerance: float = 0.1
    shift_threshold: float = 0.3
    data_root: Path = Path("./data")
    cifar_c_path: Path | None = None
    database_path: Path | None = Path("./meld_results.db")
    seed: int = 7
    full_retrain_interval: int = 3
    run_robustness_eval: bool = True
    run_avalanche_baselines: bool = False
    train: TrainConfig = field(default_factory=TrainConfig)


def run(
    config: MELDConfig,
    results_path: str | None = None,
    *,
    snapshot_strategy: FisherManifoldSnapshot | None = None,
    safety_oracle: SpectralSafetyOracle | None = None,
    updater: ManifoldUpdater | None = None,
    corrector: AnalyticNormCorrector | None = None,
    drift_detector: DriftDetector | None = None,
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
