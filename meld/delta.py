"""Simple public DeltaModel API for MELD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .api import TrainConfig
from .core.corrector import AnalyticNormCorrector
from .core.drift import CompositeDriftDetector
from .core.oracle import SpectralSafetyOracle
from .core.policy import FourStateDeployPolicy
from .core.snapshot import FisherManifoldSnapshot
from .core.updater import GeometryConstrainedUpdater
from .interfaces.base import DriftResult, OracleEstimate, TaskSnapshot, TrainArtifacts
from .modeling import MELDModel
from .models.backbone import resnet20, resnet32, resnet44, resnet56
from .models.classifier import IncrementalClassifier


BACKBONES = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
}


@dataclass(slots=True)
class DeltaUpdateResult:
    task_id: int
    decision: str
    recommended_action: str
    confidence: float
    oracle: dict[str, Any]
    drift: dict[str, Any]
    train: dict[str, Any]
    compute_savings_percent: float
    # Deployment safety fields — mirroring runner.py formal_guarantee logic
    safe_to_deploy: bool = False
    formal_guarantee: bool = False

    def __str__(self) -> str:
        guarantee = " [formal guarantee]" if self.formal_guarantee else ""
        return (
            f"DeltaUpdateResult(task={self.task_id}{guarantee})\n"
            f"  decision         : {self.decision}\n"
            f"  safe_to_deploy   : {self.safe_to_deploy}\n"
            f"  recommended      : {self.recommended_action}\n"
            f"  drift severity   : {self.drift.get('severity', '?')}  "
            f"(score={float(self.drift.get('shift_score', 0)):.4f})\n"
            f"  risk pre→post    : "
            f"{float(self.oracle.get('risk_estimate_pre', 0)):.4f} → "
            f"{float(self.oracle.get('drift_realized_post', 0)):.4f}\n"
            f"  pac_style_gap    : "
            f"{self.oracle.get('pac_style_gap', {}).get('value', 'n/a')}\n"
            f"  compute saved    : {self.compute_savings_percent:.1f}%\n"
            f"  epochs run       : {self.train.get('epochs_run', 0)}"
        )


class DeltaModel:
    def __init__(
        self,
        model: MELDModel,
        train_config: TrainConfig,
        *,
        bound_tolerance: float = 10.0,
        shift_threshold: float = 0.3,
        prefer_cuda: bool = False,
        backbone_name: str = "custom",
        snapshot_strategy: FisherManifoldSnapshot | None = None,
        safety_oracle: SpectralSafetyOracle | None = None,
        updater: GeometryConstrainedUpdater | None = None,
        corrector: AnalyticNormCorrector | None = None,
        drift_detector: CompositeDriftDetector | None = None,
        deploy_policy: FourStateDeployPolicy | None = None,
        snapshot: TaskSnapshot | None = None,
        history: list[dict[str, Any]] | None = None,
        task_id: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.train_config = train_config
        self.bound_tolerance = float(bound_tolerance)
        self.shift_threshold = float(shift_threshold)
        self.device = device or self._select_device(prefer_cuda)
        self._model = model.to(self.device)
        self._backbone_name = backbone_name
        self._snapshot_strategy = snapshot_strategy or FisherManifoldSnapshot()
        self._safety_oracle = safety_oracle or SpectralSafetyOracle()
        self._updater = updater or GeometryConstrainedUpdater()
        self._corrector = corrector or AnalyticNormCorrector()
        self._drift_detector = drift_detector or CompositeDriftDetector(self.shift_threshold)
        self._deploy_policy = deploy_policy or FourStateDeployPolicy()
        self._snapshot = snapshot
        self._history = list(history or [])
        self._task_id = int(task_id)

    @classmethod
    def from_scratch(
        cls,
        *,
        num_classes: int = 0,
        backbone: str = "resnet32",
        bound_tolerance: float = 10.0,
        shift_threshold: float = 0.3,
        prefer_cuda: bool = False,
        train_config: TrainConfig | None = None,
    ) -> "DeltaModel":
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Valid choices: {sorted(BACKBONES)}")
        config = train_config or TrainConfig(backbone=backbone)
        backbone_module = BACKBONES[backbone](pretrained=bool(config.pretrained_backbone))
        classifier = IncrementalClassifier(backbone_module.out_dim)
        if num_classes > 0:
            classifier.adaption(int(num_classes))
        model = MELDModel(backbone_module, classifier)
        return cls(
            model,
            config,
            bound_tolerance=bound_tolerance,
            shift_threshold=shift_threshold,
            prefer_cuda=prefer_cuda,
            backbone_name=backbone,
        )

    @classmethod
    def from_backbone(
        cls,
        backbone: nn.Module,
        *,
        out_dim: int,
        num_classes: int = 0,
        bound_tolerance: float = 10.0,
        shift_threshold: float = 0.3,
        prefer_cuda: bool = False,
        train_config: TrainConfig | None = None,
    ) -> "DeltaModel":
        if not hasattr(backbone, "out_dim"):
            setattr(backbone, "out_dim", int(out_dim))
        config = train_config or TrainConfig(backbone="custom")
        classifier = IncrementalClassifier(int(out_dim))
        if num_classes > 0:
            classifier.adaption(int(num_classes))
        model = MELDModel(backbone, classifier)
        return cls(
            model,
            config,
            bound_tolerance=bound_tolerance,
            shift_threshold=shift_threshold,
            prefer_cuda=prefer_cuda,
            backbone_name="custom",
        )

    def update(self, loader: DataLoader) -> DeltaUpdateResult:
        current_classes = self._loader_class_ids(loader)
        self._ensure_classes(current_classes)
        snapshot_before = self._snapshot
        pac_gate_tolerance = float(getattr(self.train_config, "pac_gate_tolerance", 0.1))

        if snapshot_before is not None:
            pre_risk_estimate = self._safety_oracle.pre_risk_estimate(snapshot_before, self.train_config)
        else:
            pre_risk_estimate = OracleEstimate(
                value=0.0,
                bound_type="no_prior_task",
                calibrated=False,
                bound_is_formal=False,
            )

        # PAC gate: compute formal Hoeffding gap before training.
        # If it exceeds pac_gate_tolerance, skip the update — same logic as runner.py.
        pac_style: OracleEstimate | None = None
        pac_gate_triggered = False
        if snapshot_before is not None and hasattr(self._safety_oracle, "pac_style_gap"):
            pac_style = self._safety_oracle.pac_style_gap(snapshot_before)
            pac_gate_triggered = bool(
                pac_style.bound_is_formal
                and pac_style.value > pac_gate_tolerance
            )

        spectral_gate = snapshot_before is not None and pre_risk_estimate.value > self.bound_tolerance

        if spectral_gate or pac_gate_triggered:
            post_drift_realized = OracleEstimate(
                value=pre_risk_estimate.value,
                bound_type="training_skipped",
                calibrated=False,
                bound_is_formal=False,
            )
            drift_result = DriftResult(0.0, False, {}, "none")
            reason = "pac_formal_gate" if pac_gate_triggered and not spectral_gate else "spectral_bound_exceeded"
            result = DeltaUpdateResult(
                task_id=self._task_id,
                decision="BOUND_EXCEEDED",
                recommended_action="full_retrain",
                confidence=1.0,
                oracle=self._oracle_payload(pre_risk_estimate, post_drift_realized, False, pac_style),
                drift=self._drift_payload(drift_result),
                train=self._train_payload(
                    TrainArtifacts(
                        epochs_run=0,
                        lambda_schedule=[],
                        geometry_loss_per_epoch=[],
                        ewc_loss_per_epoch=[],
                        ce_loss_per_epoch=[],
                        wall_time_seconds=0.0,
                        skipped=True,
                    )
                ),
                compute_savings_percent=0.0,
                safe_to_deploy=False,
                formal_guarantee=False,
            )
            self._record_result(result)
            self._task_id += 1
            return result

        self._model, artifacts = self._updater.update(self._model, loader, snapshot_before, self.train_config)
        if snapshot_before is not None:
            self._model = self._corrector.correct(self._model, snapshot_before)
        snapshot_after = self._snapshot_strategy.capture(
            self._model,
            loader,
            current_classes,
            self._task_id,
        )
        if snapshot_before is not None and snapshot_after.class_ids:
            post_drift_realized = self._safety_oracle.post_drift_realized(snapshot_before, snapshot_after)
            drift_result = self._drift_detector.detect(snapshot_before, snapshot_after)
        else:
            post_drift_realized = OracleEstimate(
                value=0.0,
                bound_type="no_prior_task",
                calibrated=False,
                bound_is_formal=False,
            )
            drift_result = DriftResult(0.0, False, {}, "none")

        # Compute PAC equivalence bound post-training for reporting.
        pac_equivalence: OracleEstimate | None = None
        if (snapshot_before is not None
                and hasattr(self._safety_oracle, "pac_equivalence_bound")):
            pac_equivalence = self._safety_oracle.pac_equivalence_bound(
                snapshot_before, snapshot_after
            )

        decision = self._deploy_policy.decide(
            pre_risk_estimate.value,
            post_drift_realized.value,
            drift_result,
            _DeltaPolicyConfig(
                shift_threshold=self.shift_threshold,
                delta_wall_time_seconds=float(artifacts.wall_time_seconds),
                full_retrain_wall_time_seconds=0.0,
            ),
        )

        # formal_guarantee: True only when IW is on, PAC bound is tight,
        # and the decision is safe — mirrors runner.py line 944 exactly.
        iw_on = bool(getattr(self.train_config, "enable_importance_weighting", False))
        formal_guarantee = bool(
            self._task_id > 0
            and iw_on
            and (
                (pac_equivalence is not None
                 and pac_equivalence.bound_is_formal
                 and pac_equivalence.value <= pac_gate_tolerance)
                or (pac_style is not None
                    and pac_style.bound_is_formal
                    and pac_style.value <= pac_gate_tolerance)
            )
            and decision.state in {"SAFE_DELTA", "CAUTIOUS_DELTA"}
        )
        safe_to_deploy = decision.state in {"SAFE_DELTA", "CAUTIOUS_DELTA"}

        result = DeltaUpdateResult(
            task_id=self._task_id,
            decision=decision.state,
            recommended_action=decision.recommended_action,
            confidence=float(decision.confidence),
            oracle=self._oracle_payload(pre_risk_estimate, post_drift_realized, bool(decision.bound_held), pac_style, pac_equivalence),
            drift=self._drift_payload(drift_result),
            train=self._train_payload(artifacts),
            compute_savings_percent=float(decision.compute_savings_percent),
            safe_to_deploy=safe_to_deploy,
            formal_guarantee=formal_guarantee,
        )
        self._snapshot = snapshot_after
        self._record_result(result)
        self._task_id += 1
        return result

    def predict(self, x: Tensor) -> Tensor:
        self._model.eval()
        with torch.no_grad():
            return self._model(x.to(self.device))

    def predict_labels(self, x: Tensor) -> Tensor:
        return self.predict(x).argmax(dim=1)

    def embed(self, x: Tensor) -> Tensor:
        self._model.eval()
        with torch.no_grad():
            return self._model.embed(x.to(self.device))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model = self._model.to("cpu")
        torch.save(
            {
                "model": model,
                "train_config": asdict(self.train_config),
                "bound_tolerance": self.bound_tolerance,
                "shift_threshold": self.shift_threshold,
                "backbone_name": self._backbone_name,
                "snapshot": self._snapshot,
                "history": self._history,
                "task_id": self._task_id,
            },
            target,
        )
        self._model = model.to(self.device)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        prefer_cuda: bool = False,
    ) -> "DeltaModel":
        device = cls._select_device(prefer_cuda)
        try:
            payload = torch.load(Path(path), map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location=device)
        train_config = TrainConfig(**payload["train_config"])
        model = payload["model"].to(device)
        return cls(
            model=model,
            train_config=train_config,
            bound_tolerance=float(payload["bound_tolerance"]),
            shift_threshold=float(payload["shift_threshold"]),
            prefer_cuda=prefer_cuda,
            backbone_name=str(payload.get("backbone_name", "custom")),
            snapshot=payload.get("snapshot"),
            history=payload.get("history", []),
            task_id=int(payload.get("task_id", 0)),
            device=device,
        )

    def summary(self) -> str:
        lines = [
            f"DeltaModel  tasks={len(self._history)}  classes={self._model.classifier.num_classes}  device={self.device.type}",
            "  task  decision           drift    saved%   safe  formal  action",
        ]
        for entry in self._history:
            safe = "✓" if entry.get("safe_to_deploy") else "✗"
            formal = "✓" if entry.get("formal_guarantee") else "✗"
            lines.append(
                "  "
                f"{entry['task_id']:>4}  "
                f"{entry['decision']:<16}  "
                f"{float(entry['drift']['shift_score']):>6.4f}  "
                f"{float(entry['compute_savings_percent']):>6.1f}%  "
                f"{safe:>4}  "
                f"{formal:>6}  "
                f"{entry['recommended_action']}"
            )
        return "\n".join(lines)

    def _ensure_classes(self, observed_class_ids: list[int]) -> None:
        if not observed_class_ids:
            return
        required = max(observed_class_ids) + 1
        missing = required - self._model.classifier.num_classes
        if missing > 0:
            self._model.classifier.adaption(missing)

    def _loader_class_ids(self, loader: DataLoader) -> list[int]:
        class_ids: set[int] = set()
        for _inputs, targets in loader:
            if isinstance(targets, Tensor):
                class_ids.update(int(value) for value in targets.detach().cpu().tolist())
            else:
                class_ids.update(int(value) for value in list(targets))
        return sorted(class_ids)

    def _record_result(self, result: DeltaUpdateResult) -> None:
        self._history.append(
            {
                "task_id": int(result.task_id),
                "decision": result.decision,
                "recommended_action": result.recommended_action,
                "confidence": float(result.confidence),
                "safe_to_deploy": bool(result.safe_to_deploy),
                "formal_guarantee": bool(result.formal_guarantee),
                "oracle": result.oracle,
                "drift": result.drift,
                "train": result.train,
                "compute_savings_percent": float(result.compute_savings_percent),
            }
        )

    @staticmethod
    def _select_device(prefer_cuda: bool) -> torch.device:
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _oracle_payload(
        pre_risk_estimate: OracleEstimate,
        post_drift_realized: OracleEstimate,
        bound_held: bool,
        pac_style: OracleEstimate | None = None,
        pac_equivalence: OracleEstimate | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "risk_estimate_pre": float(pre_risk_estimate.value),
            "drift_realized_post": float(post_drift_realized.value),
            "risk_estimate_held": bool(bound_held),
            "bound_type": pre_risk_estimate.bound_type,
            "bound_is_formal": bool(pre_risk_estimate.bound_is_formal),
        }
        if pac_style is not None:
            payload["pac_style_gap"] = {
                "value": float(pac_style.value),
                "delta": pac_style.delta,
                "bound_type": pac_style.bound_type,
                "bound_is_formal": bool(pac_style.bound_is_formal),
            }
        if pac_equivalence is not None:
            payload["pac_equivalence_bound"] = {
                "value": float(pac_equivalence.value),
                "delta": pac_equivalence.delta,
                "bound_type": pac_equivalence.bound_type,
                "bound_is_formal": bool(pac_equivalence.bound_is_formal),
            }
        return payload

    @staticmethod
    def _drift_payload(drift_result: DriftResult) -> dict[str, Any]:
        return {
            "shift_score": float(drift_result.shift_score),
            "shift_detected": bool(drift_result.shift_detected),
            "severity": drift_result.severity,
            "per_class_drift": dict(drift_result.per_class_drift),
        }

    @staticmethod
    def _train_payload(artifacts: TrainArtifacts) -> dict[str, Any]:
        return {
            "epochs_run": int(artifacts.epochs_run),
            "ce_loss_per_epoch": list(artifacts.ce_loss_per_epoch),
            "geometry_loss_per_epoch": list(artifacts.geometry_loss_per_epoch),
            "ewc_loss_per_epoch": list(artifacts.ewc_loss_per_epoch),
            "train_accuracy_per_epoch": list(artifacts.train_accuracy_per_epoch),
            "wall_time_seconds": float(artifacts.wall_time_seconds),
            "skipped": bool(artifacts.skipped),
        }


@dataclass(slots=True)
class _DeltaPolicyConfig:
    shift_threshold: float
    delta_wall_time_seconds: float
    full_retrain_wall_time_seconds: float