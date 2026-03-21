"""End-to-end benchmark runner for MELD."""

from __future__ import annotations

import json
import os
import random
import time
import functools
from dataclasses import asdict, replace
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

from ..core.auto_config import derive_train_config
from ..core.corrector import AnalyticNormCorrector
from ..core.drift import CompositeDriftDetector, PretrainingMMDDriftDetector
from ..core.oracle import SpectralSafetyOracle
from ..core.policy import FourStateDeployPolicy
from ..core.snapshot import FisherManifoldSnapshot
from ..core.updater import FrozenBackboneAnalyticUpdater, FullRetrainUpdater, GeometryConstrainedUpdater
from ..datasets import get_dataset_provider, validate_task_bundle
from ..interfaces.base import (
    Decision,
    DriftDetector,
    DriftResult,
    ManifoldUpdater,
    OracleEstimate,
    TaskSnapshot,
    TrainArtifacts,
)
from ..modeling import MELDModel
from ..models.backbone import resnet20, resnet32, resnet44, resnet56, resnet18_imagenet
from ..models.classifier import IncrementalClassifier
from .metrics import (
    compute_classification_metrics,
    compute_compute_savings,
    compute_equivalence_gap,
    compute_ece_maybe,
)
from .storage import ResultStore
from .robustness import evaluate_cifar_c
from .avalanche_baselines import run_avalanche_baselines

if TYPE_CHECKING:
    from ..api import MELDConfig


BACKBONES = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet18_imagenet": resnet18_imagenet,
}

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_STL10_MEAN = (0.4467, 0.4398, 0.4066)
_STL10_STD = (0.2603, 0.2566, 0.2713)


def _move_inputs_to_device(
    inputs: Any,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Any:
    if isinstance(inputs, dict):
        return {
            key: value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
    if isinstance(inputs, tuple):
        return tuple(
            value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
            for value in inputs
        )
    return inputs.to(device, non_blocking=non_blocking)


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """DataLoader worker seeding for Python/numpy/torch RNGs.

    Implemented at module scope so it is picklable on Windows (spawn).
    """

    import random as _random

    import numpy as _np

    worker_seed = int(base_seed) + int(worker_id)
    _random.seed(worker_seed)
    _np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _auto_backbone(dataset: str) -> str:
    d = dataset.upper().replace("-", "").replace("_", "")
    if d == "SYNTHETIC":
        return "resnet20"
    if d == "CIFAR10":
        return "resnet32"
    if d == "CIFAR100":
        return "resnet56"
    if d in ("TINYIMAGENET", "TINYIMAGENET200", "STL10"):
        return "resnet18_imagenet"
    if d in ("AGNEWS", "DBPEDIA", "YAHOOANSWERSNLP"):
        return "text_encoder"
    return "resnet32"


class BenchmarkRunner:
    def __init__(
        self,
        config: "MELDConfig",
        snapshot_strategy: FisherManifoldSnapshot | None = None,
        safety_oracle: SpectralSafetyOracle | None = None,
        updater: ManifoldUpdater | None = None,
        corrector: AnalyticNormCorrector | None = None,
        drift_detector: DriftDetector | None = None,
        deploy_policy: FourStateDeployPolicy | None = None,
    ) -> None:
        self.config = config
        self.snapshot_strategy = snapshot_strategy or FisherManifoldSnapshot()
        self.safety_oracle = safety_oracle or SpectralSafetyOracle()
        self.updater = updater or GeometryConstrainedUpdater()
        self.analytic_updater = FrozenBackboneAnalyticUpdater()
        self._custom_updater_provided = updater is not None
        self.full_retrain_updater = FullRetrainUpdater()
        self.corrector = corrector or AnalyticNormCorrector()
        self.drift_detector = drift_detector or CompositeDriftDetector(config.shift_threshold)
        self.pretrain_shift_detector = PretrainingMMDDriftDetector()
        self.deploy_policy = deploy_policy or FourStateDeployPolicy()
        if config.prefer_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self._baseline_model_cache: MELDModel | None = None
        self._loader_num_workers = self._effective_num_workers()
        self._pin_memory = self.device.type == "cuda"
        self._result_store = ResultStore(getattr(config, "database_path", None))

    def _run_mode(self) -> str:
        mode = str(getattr(self.config, "run_mode", "compare"))
        if mode not in {"compare", "delta", "full_retrain"}:
            raise ValueError(
                f"Unknown run_mode '{mode}'. Valid choices: compare, delta, full_retrain."
            )
        return mode

    def _uses_custom_dataset_provider(self) -> bool:
        return bool(getattr(self.config, "dataset_provider", None) is not None or get_dataset_provider(self.config.dataset) is not None)

    def run(self, results_path: str | Path | None = None) -> dict[str, Any]:
        self._seed_everything(int(self.config.seed))
        tasks, eval_loaders = self._build_tasks()
        model = self._build_model()
        all_seen_loaders: list[DataLoader] = []
        last_drift_score: float | None = None
        num_tasks_run = len(tasks)
        # CIL tracking for MELD delta model:
        # - acc_after_task[t] : top1 after training task t
        # - best_acc[t]       : best top1 on task t over time
        delta_acc_after_task: list[float] = [0.0 for _ in range(num_tasks_run)]
        delta_best_acc: list[float] = [0.0 for _ in range(num_tasks_run)]
        results: dict[str, Any] = {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
            "status": "running",
            "config": self._config_dict(),
            "tasks": [],
            "bounds_timeline": [],
            "epoch_history": [],
            "final_summary": None,
        }
        self._write_results(results_path, results)

        for task_id, task_loader in enumerate(tasks):
            all_seen_loaders.append(task_loader)
            new_class_ids = model.classifier.adaption(self.config.classes_per_task)
            if bool(getattr(self.config.train, "use_imprinting", True)) and (
                task_id > 0 or bool(getattr(self.config.train, "pretrained_backbone", False))
            ):
                self._imprint_new_head(model, task_loader, new_class_ids)
            forward_transfer_top1: float | None = None
            if task_id > 0:
                # Zero-shot forward transfer: evaluate before any delta training.
                ft_metrics = self._evaluate(model, [eval_loaders[task_id]])
                forward_transfer_top1 = float(ft_metrics["top1"])

            snapshot_before = None
            pre_pac_style = None
            pac_gate_triggered = False
            pre_risk_estimate = OracleEstimate(
                value=0.0,
                bound_type="empirical_spectral",
                calibrated=False,
                bound_is_formal=False,
            )
            if task_id > 0:
                # Adaptive Fisher EMA decay: larger observed drift -> faster
                # decay (smaller EMA factor) for the next snapshot.
                base_decay = float(getattr(self.config.train, "fisher_ema_decay", 0.9))
                if last_drift_score is None:
                    decay = base_decay
                else:
                    ratio = float(last_drift_score) / max(1e-6, float(self.config.shift_threshold))
                    decay = base_decay / (1.0 + ratio)
                self.snapshot_strategy._ema_decay = float(np.clip(decay, 0.05, 0.95))
                snapshot_before = self.snapshot_strategy.capture(
                    model,
                    self._merged_loader(all_seen_loaders[:-1], shuffle=False),
                    list(range(model.classifier.num_classes - self.config.classes_per_task)),
                    task_id - 1,
                )
                if getattr(self.config.train, "auto_derive_hparams", False):
                    task_train_config = derive_train_config(
                        snapshot_before,
                        self.config.train,
                        getattr(self.config.train, "protection_level", 0.5),
                    )
                else:
                    task_train_config = self.config.train
                pretrain_shift = self._pretraining_shift(snapshot_before, model, task_loader)
                if pretrain_shift is not None and pretrain_shift.shift_detected:
                    mmd_ratio = float(pretrain_shift.shift_score) / max(1e-6, float(self.config.shift_threshold))
                    self.snapshot_strategy._ema_decay = float(
                        np.clip(base_decay / (1.0 + 2.0 * mmd_ratio), 0.05, 0.95)
                    )
                if self._uses_frozen_analytic(task_id):
                    pre_risk_estimate = OracleEstimate(
                        value=0.0,
                        bound_type="protected_params_frozen",
                        calibrated=False,
                        bound_is_formal=False,
                    )
                else:
                    pre_risk_estimate = self._pre_risk_estimate(snapshot_before, task_train_config)
                    task_train_config, pre_risk_estimate = self._make_safe_train_config(
                        snapshot_before,
                        task_train_config,
                        pre_risk_estimate,
                    )
                if hasattr(self.safety_oracle, "pac_style_gap"):
                    pre_pac_style = self.safety_oracle.pac_style_gap(snapshot_before)
                    pac_gate_triggered = bool(
                        pre_pac_style.bound_is_formal
                        and pre_pac_style.value > float(getattr(self.config, "pac_gate_tolerance", 0.1))
                    )
            else:
                task_train_config = self._base_train_config()
                pretrain_shift = None

            if task_id > 0 and (
                pre_risk_estimate.value > self.config.bound_tolerance or pac_gate_triggered
            ):
                delta_artifacts = TrainArtifacts(
                    epochs_run=0,
                    lambda_schedule=[],
                    geometry_loss_per_epoch=[],
                    ewc_loss_per_epoch=[],
                    ce_loss_per_epoch=[],
                    wall_time_seconds=0.0,
                    train_accuracy_per_epoch=[],
                    projected_step_fraction=None,
                    skipped=True,
                )
                snapshot_after = snapshot_before
                post_drift_realized = OracleEstimate(
                    value=pre_risk_estimate.value,
                    bound_type="training_skipped",
                    calibrated=False,
                    bound_is_formal=False,
                )
                drift_result = DriftResult(0.0, False, {}, "none")
                delta_metrics = None
                decision = Decision(
                    state="BOUND_EXCEEDED",
                    pre_bound=pre_risk_estimate.value,
                    post_bound=post_drift_realized.value,
                    bound_held=False,
                    shift_score=0.0,
                    shift_detected=False,
                    reason=(
                        "formal PAC gate exceeded tolerance before training"
                        if pac_gate_triggered
                        else "pre-training safety bound exceeded tolerance before training"
                    ),
                    compute_savings_percent=0.0,
                    confidence=1.0,
                    recommended_action="full_retrain",
                    formal_guarantee=False,
                )
            else:
                delta_updater = self._delta_updater_for_task(task_id)
                model, delta_artifacts = delta_updater.update(
                    model, task_loader, snapshot_before, task_train_config
                )
                if snapshot_before is not None:
                    model = self.corrector.correct(model, snapshot_before)
                    all_known_class_ids = list(range(model.classifier.num_classes))
                    snapshot_after = self.snapshot_strategy.capture(
                        model,
                        self._merged_loader(all_seen_loaders, shuffle=False),
                        all_known_class_ids,
                        task_id,
                    )
                    post_drift_realized = self._post_drift_realized(snapshot_before, snapshot_after)
                    drift_result = self.drift_detector.detect(snapshot_before, snapshot_after)
                else:
                    snapshot_after = self.snapshot_strategy.capture(
                        model, task_loader, new_class_ids, task_id
                    )
                    post_drift_realized = OracleEstimate(
                        value=0.0,
                        bound_type="no_prior_task",
                        calibrated=False,
                        bound_is_formal=False,
                    )
                    drift_result = DriftResult(0.0, False, {}, "none")

                old_class_ids = list(range(0, task_id * self.config.classes_per_task))
                new_class_ids = list(
                    range(task_id * self.config.classes_per_task, (task_id + 1) * self.config.classes_per_task)
                )
                full_metrics, full_time = self._run_full_retrain_baseline(
                    all_seen_loaders,
                    eval_loaders[: task_id + 1],
                    old_class_ids=old_class_ids,
                    new_class_ids=new_class_ids,
                )
                decision = self.deploy_policy.decide(
                    pre_risk_estimate.value,
                    post_drift_realized.value,
                    drift_result,
                    _PolicyConfigProxy(
                        shift_threshold=self.config.shift_threshold,
                        delta_wall_time_seconds=delta_artifacts.wall_time_seconds,
                        full_retrain_wall_time_seconds=full_time,
                    ),
                )
                delta_metrics = self._evaluate(
                    model,
                    eval_loaders[: task_id + 1],
                    old_class_ids=old_class_ids,
                    new_class_ids=new_class_ids,
                )
                decision.compute_savings_percent = compute_compute_savings(
                    delta_artifacts.wall_time_seconds, full_time
                )
                delta_metrics["wall_time_seconds"] = delta_artifacts.wall_time_seconds

                # Phase 4 CIL metrics: per-task forgetting/backward transfer
                # for the MELD delta model.
                per_task_top1: list[float] = []
                for k in range(task_id + 1):
                    m_k = self._evaluate(model, [eval_loaders[k]])
                    per_task_top1.append(float(m_k["top1"]))
                if task_id == 0:
                    delta_acc_after_task[0] = per_task_top1[0]
                    delta_best_acc[0] = per_task_top1[0]
                    forgetting_per_task: dict[int, float] = {}
                    backward_transfer_per_task: dict[int, float] = {}
                else:
                    forgetting_per_task = {}
                    backward_transfer_per_task = {}
                    for k in range(task_id):
                        delta_best_acc[k] = max(delta_best_acc[k], per_task_top1[k])
                        forgetting_per_task[k] = float(delta_best_acc[k] - per_task_top1[k])
                        backward_transfer_per_task[k] = float(
                            per_task_top1[k] - delta_acc_after_task[k]
                        )
                    delta_acc_after_task[task_id] = per_task_top1[task_id]
                    delta_best_acc[task_id] = per_task_top1[task_id]

                cil_metrics = {
                    "forward_transfer_top1": forward_transfer_top1,
                    "forgetting_per_task": forgetting_per_task,
                    "backward_transfer_per_task": backward_transfer_per_task,
                }
                if forgetting_per_task:
                    cil_metrics["mean_forgetting"] = float(
                        np.mean(list(forgetting_per_task.values()))
                    )
                else:
                    cil_metrics["mean_forgetting"] = None
                if backward_transfer_per_task:
                    cil_metrics["mean_backward_transfer"] = float(
                        np.mean(list(backward_transfer_per_task.values()))
                    )
                else:
                    cil_metrics["mean_backward_transfer"] = None
                task_result = self._task_result(
                    task_id,
                    delta_metrics,
                    full_metrics,
                    snapshot_before,
                    snapshot_after,
                    pre_risk_estimate,
                    post_drift_realized,
                    drift_result,
                    decision,
                    delta_wall_time_seconds=float(delta_artifacts.wall_time_seconds),
                    cil_metrics=cil_metrics,
                    train_artifacts=delta_artifacts,
                    pretrain_shift=pretrain_shift,
                    pre_pac_style=pre_pac_style,
                )
                results["tasks"].append(task_result)
                results["bounds_timeline"].append(
                    {
                        "task_id": task_id,
                        "risk_estimate_pre": float(pre_risk_estimate.value),
                        "drift_realized_post": float(post_drift_realized.value),
                        "risk_estimate_held": bool(decision.bound_held if task_id > 0 else True),
                        "bound_type": pre_risk_estimate.bound_type,
                        "bound_is_formal": pre_risk_estimate.bound_is_formal,
                        "pac_style_gap": task_result["oracle"].get("pac_style_gap"),
                        "decision_state": decision.state,
                    }
                )
                results["epoch_history"].append(
                    {
                        "task_id": task_id,
                        "delta": {
                            "epochs_run": int(delta_artifacts.epochs_run),
                            "ce_loss_per_epoch": list(delta_artifacts.ce_loss_per_epoch),
                            "geometry_loss_per_epoch": list(delta_artifacts.geometry_loss_per_epoch),
                            "ewc_loss_per_epoch": list(delta_artifacts.ewc_loss_per_epoch),
                            "train_accuracy_per_epoch": list(delta_artifacts.train_accuracy_per_epoch),
                            "projected_step_fraction": delta_artifacts.projected_step_fraction,
                            "skipped": bool(delta_artifacts.skipped),
                        },
                    }
                )
                self._write_results(results_path, results)
                last_drift_score = float(drift_result.shift_score)
                continue

            old_class_ids = list(range(0, task_id * self.config.classes_per_task))
            new_class_ids = list(
                range(task_id * self.config.classes_per_task, (task_id + 1) * self.config.classes_per_task)
            )
            full_metrics, full_time = self._run_full_retrain_baseline(
                all_seen_loaders,
                eval_loaders[: task_id + 1],
                old_class_ids=old_class_ids,
                new_class_ids=new_class_ids,
            )
            decision.compute_savings_percent = compute_compute_savings(
                delta_artifacts.wall_time_seconds, full_time
            )

            # Phase 4 CIL metrics for the MELD delta model (even when
            # training was skipped).
            per_task_top1: list[float] = []
            for k in range(task_id + 1):
                m_k = self._evaluate(model, [eval_loaders[k]])
                per_task_top1.append(float(m_k["top1"]))
            if task_id == 0:
                delta_acc_after_task[0] = per_task_top1[0]
                delta_best_acc[0] = per_task_top1[0]
                forgetting_per_task = {}
                backward_transfer_per_task = {}
            else:
                forgetting_per_task = {}
                backward_transfer_per_task = {}
                for k in range(task_id):
                    delta_best_acc[k] = max(delta_best_acc[k], per_task_top1[k])
                    forgetting_per_task[k] = float(delta_best_acc[k] - per_task_top1[k])
                    backward_transfer_per_task[k] = float(
                        per_task_top1[k] - delta_acc_after_task[k]
                    )
                delta_acc_after_task[task_id] = per_task_top1[task_id]
                delta_best_acc[task_id] = per_task_top1[task_id]

            cil_metrics = {
                "forward_transfer_top1": forward_transfer_top1,
                "forgetting_per_task": forgetting_per_task,
                "backward_transfer_per_task": backward_transfer_per_task,
            }
            if forgetting_per_task:
                cil_metrics["mean_forgetting"] = float(
                    np.mean(list(forgetting_per_task.values()))
                )
            else:
                cil_metrics["mean_forgetting"] = None
            if backward_transfer_per_task:
                cil_metrics["mean_backward_transfer"] = float(
                    np.mean(list(backward_transfer_per_task.values()))
                )
            else:
                cil_metrics["mean_backward_transfer"] = None
            task_result = self._task_result(
                task_id,
                delta_metrics,
                full_metrics,
                snapshot_before,
                snapshot_after,
                pre_risk_estimate,
                post_drift_realized,
                drift_result,
                decision,
                delta_wall_time_seconds=float(delta_artifacts.wall_time_seconds),
                cil_metrics=cil_metrics,
                train_artifacts=delta_artifacts,
                pretrain_shift=pretrain_shift,
                pre_pac_style=pre_pac_style,
            )
            results["tasks"].append(task_result)
            results["bounds_timeline"].append(
                {
                    "task_id": task_id,
                    "risk_estimate_pre": float(pre_risk_estimate.value),
                    "drift_realized_post": float(post_drift_realized.value),
                    "risk_estimate_held": bool(decision.bound_held if task_id > 0 else True),
                    "bound_type": pre_risk_estimate.bound_type,
                    "bound_is_formal": pre_risk_estimate.bound_is_formal,
                    "pac_style_gap": task_result["oracle"].get("pac_style_gap"),
                    "decision_state": decision.state,
                }
            )
            results["epoch_history"].append(
                {
                    "task_id": task_id,
                    "delta": {
                        "epochs_run": int(delta_artifacts.epochs_run),
                        "ce_loss_per_epoch": list(delta_artifacts.ce_loss_per_epoch),
                        "geometry_loss_per_epoch": list(delta_artifacts.geometry_loss_per_epoch),
                        "ewc_loss_per_epoch": list(delta_artifacts.ewc_loss_per_epoch),
                        "train_accuracy_per_epoch": list(delta_artifacts.train_accuracy_per_epoch),
                        "projected_step_fraction": delta_artifacts.projected_step_fraction,
                        "skipped": bool(delta_artifacts.skipped),
                    },
                }
            )
            self._write_results(results_path, results)
            last_drift_score = float(drift_result.shift_score)

        results["status"] = "completed"
        if bool(getattr(self.config, "run_robustness_eval", False)):
            try:
                results["robustness"] = evaluate_cifar_c(
                    model,
                    dataset=self.config.dataset,
                    data_root=Path(getattr(self.config, "cifar_c_path", None) or self.config.data_root),
                    device=self.device,
                    batch_size=self.config.train.batch_size,
                )
            except Exception as exc:
                results["robustness"] = {"status": "skipped", "reason": f"Robustness eval failed: {exc}"}
        if bool(getattr(self.config, "run_avalanche_baselines", False)):
            results["baselines"] = run_avalanche_baselines(
                config=self.config,
                device=self.device,
            )
        results["final_summary"] = self._summarize(results["tasks"], results.get("robustness"))
        self._write_results(results_path, results)
        return results

    def _build_model(self) -> MELDModel:
        backbone_name: str = self.config.train.backbone
        if backbone_name == "auto":
            backbone_name = _auto_backbone(self.config.dataset)

        # Text encoder backbone — lazy-loaded, no registry entry needed
        if backbone_name == "text_encoder":
            from ..core.text_encoder import TextEncoderBackbone
            text_model = self.config.train.text_encoder_model
            backbone = TextEncoderBackbone(model_name=text_model)
            classifier = IncrementalClassifier(backbone.out_dim)
            return MELDModel(backbone, classifier).to(self.device)

        if backbone_name not in BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Valid choices: {sorted(BACKBONES)} or 'text_encoder'"
            )
        pretrained: bool = self.config.train.pretrained_backbone
        backbone = BACKBONES[backbone_name](pretrained=pretrained)
        classifier = IncrementalClassifier(backbone.out_dim)
        return MELDModel(backbone, classifier).to(self.device)

    def _build_tasks(self) -> tuple[list[DataLoader], list[DataLoader]]:
        bundle = self._load_dataset_bundle()
        train_tasks, eval_tasks = [], []
        generator = torch.Generator()
        generator.manual_seed(int(self.config.seed))
        worker_init = (
            functools.partial(_seed_worker, base_seed=int(self.config.seed))
            if self._loader_num_workers > 0
            else None
        )

        for train_ds, test_ds in bundle:
            train_tasks.append(
                DataLoader(
                    train_ds,
                    batch_size=self.config.train.batch_size,
                    shuffle=True,
                    num_workers=self._loader_num_workers,
                    generator=generator,
                    worker_init_fn=worker_init,
                    pin_memory=self._pin_memory,
                    persistent_workers=bool(self._loader_num_workers > 0),
                )
            )
            eval_tasks.append(
                DataLoader(
                    test_ds,
                    batch_size=self.config.train.batch_size,
                    shuffle=False,
                    num_workers=self._loader_num_workers,
                    generator=generator,
                    worker_init_fn=worker_init,
                    pin_memory=self._pin_memory,
                    persistent_workers=bool(self._loader_num_workers > 0),
                )
            )
        return train_tasks, eval_tasks

    def _load_dataset_bundle(self) -> list[tuple[Dataset[Any], Dataset[Any]]]:
        custom_provider = getattr(self.config, "dataset_provider", None)
        if custom_provider is not None:
            return validate_task_bundle(custom_provider(self.config))

        registered_provider = get_dataset_provider(self.config.dataset)
        if registered_provider is not None:
            return validate_task_bundle(registered_provider(self.config))

        dataset_name = self.config.dataset.upper().replace("-", "").replace("_", "")
        if dataset_name == "SYNTHETIC":
            return self._synthetic_bundle()

        # ── NLP datasets ────────────────────────────────────────────────────
        if dataset_name in ("AGNEWS", "DBPEDIA", "YAHOOANSWERSNLP"):
            return self._nlp_bundle(dataset_name)

        # ── Image datasets ───────────────────────────────────────────────────
        try:
            from continuum import ClassIncremental
            from continuum.datasets import CIFAR10, CIFAR100
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Continuum is required for real datasets. "
                "Install project dependencies with `pip install -r requirements.txt` "
                "or run `python -m meld.bootstrap --download-datasets` after install."
            ) from exc

        try:
            import torchvision  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torchvision is required for image dataset support. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

        d = dataset_name
        if d == "CIFAR10":
            try:
                tr = CIFAR10(data_path=str(self.config.data_root), train=True, download=True)
                te = CIFAR10(data_path=str(self.config.data_root), train=False, download=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to prepare CIFAR-10 at {self.config.data_root}. "
                    "Check the data path, network access, and Continuum installation."
                ) from exc
            mean, std = _CIFAR10_MEAN, _CIFAR10_STD

        elif d == "CIFAR100":
            try:
                tr = CIFAR100(data_path=str(self.config.data_root), train=True, download=True)
                te = CIFAR100(data_path=str(self.config.data_root), train=False, download=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to prepare CIFAR-100 at {self.config.data_root}. "
                    "Check the data path, network access, and Continuum installation."
                ) from exc
            mean, std = _CIFAR100_MEAN, _CIFAR100_STD

        elif d in ("TINYIMAGENET", "TINYIMAGENET200"):
            return self._tiny_imagenet_bundle()

        elif d == "STL10":
            return self._stl10_bundle()

        else:
            raise ValueError(
                f"Unsupported dataset '{self.config.dataset}'. "
                "Supported values: synthetic, CIFAR-10, CIFAR-100, "
                "TinyImageNet, STL-10, AGNews, DBpedia."
            )

        s_tr = ClassIncremental(tr, increment=self.config.classes_per_task, transformations=None)
        s_te = ClassIncremental(te, increment=self.config.classes_per_task, transformations=None)
        bundle: list[tuple[Dataset[Any], Dataset[Any]]] = []
        for i in range(min(len(s_tr), self.config.num_tasks)):
            bundle.append(
                (
                    _TaskDatasetAdapter(s_tr[i], mean=mean, std=std, is_train=True),
                    _TaskDatasetAdapter(s_te[i], mean=mean, std=std, is_train=False),
                )
            )
        return bundle

    def _tiny_imagenet_bundle(self) -> list[tuple[Dataset[Any], Dataset[Any]]]:
        """Tiny ImageNet-200: 200 classes, 64×64 images, 100k train / 10k val."""
        try:
            import torchvision.transforms as T
            from torchvision.datasets import ImageFolder
        except ModuleNotFoundError as exc:
            raise RuntimeError("torchvision is required for Tiny ImageNet.") from exc

        root = Path(self.config.data_root) / "tiny-imagenet-200"
        if not root.exists():
            raise RuntimeError(
                f"Tiny ImageNet not found at {root}. "
                "Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                "and extract to your data root."
            )

        train_transform = T.Compose([
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        train_ds = ImageFolder(root / "train", transform=train_transform)
        val_ds = ImageFolder(root / "val", transform=val_transform)

        return self._split_into_tasks(train_ds, val_ds, num_classes=200)

    def _stl10_bundle(self) -> list[tuple[Dataset[Any], Dataset[Any]]]:
        """STL-10: 10 classes, 96×96 images, 5k train / 8k test."""
        try:
            import torchvision.transforms as T
            from torchvision.datasets import STL10
        except ModuleNotFoundError as exc:
            raise RuntimeError("torchvision is required for STL-10.") from exc

        train_transform = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_STL10_MEAN, _STL10_STD),
        ])
        val_transform = T.Compose([
            T.Resize(64),
            T.ToTensor(),
            T.Normalize(_STL10_MEAN, _STL10_STD),
        ])

        data_root = str(self.config.data_root)
        train_ds = STL10(data_root, split="train", download=True, transform=train_transform)
        val_ds = STL10(data_root, split="test", download=True, transform=val_transform)

        return self._split_into_tasks(train_ds, val_ds, num_classes=10)

    def _split_into_tasks(
        self,
        train_ds: Any,
        val_ds: Any,
        num_classes: int,
    ) -> list[tuple[Dataset[Any], Dataset[Any]]]:
        """Split a labelled dataset into class-incremental tasks."""
        import torch
        from torch.utils.data import Subset

        classes_per_task = self.config.classes_per_task
        num_tasks = min(self.config.num_tasks, num_classes // classes_per_task)

        def _indices_for_classes(dataset: Any, class_ids: list[int]) -> list[int]:
            if hasattr(dataset, "targets"):
                targets = torch.as_tensor(dataset.targets)
            elif hasattr(dataset, "labels"):
                targets = torch.as_tensor(dataset.labels)
            else:
                targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
            mask = torch.zeros(len(targets), dtype=torch.bool)
            for c in class_ids:
                mask |= targets == c
            return mask.nonzero(as_tuple=False).squeeze(1).tolist()

        bundle: list[tuple[Dataset[Any], Dataset[Any]]] = []
        for task_id in range(num_tasks):
            start = task_id * classes_per_task
            class_ids = list(range(start, start + classes_per_task))
            tr_idx = _indices_for_classes(train_ds, class_ids)
            te_idx = _indices_for_classes(val_ds, class_ids)
            bundle.append((
                _ClassRemapDataset(Subset(train_ds, tr_idx), class_ids),
                _ClassRemapDataset(Subset(val_ds, te_idx), class_ids),
            ))
        return bundle

    def _nlp_bundle(self, dataset_name: str) -> list[tuple[Dataset[Any], Dataset[Any]]]:
        """Build class-incremental tasks from a text classification dataset.

        Supported: AGNEWS (4 classes), DBPEDIA (14 classes).
        Requires: pip install transformers datasets
        """
        try:
            from datasets import load_dataset as hf_load
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "HuggingFace `datasets` is required for NLP datasets. "
                "Install with: pip install datasets transformers"
            ) from exc

        from ..core.text_encoder import TextEncoderBackbone

        # Load tokenizer once to encode all texts
        encoder = TextEncoderBackbone(model_name=self.config.train.text_encoder_model)
        tokenizer = encoder.get_tokenizer()

        hf_name_map = {
            "AGNEWS": ("ag_news", None),
            "DBPEDIA": ("dbpedia_14", None),
            "YAHOOANSWERSNLP": ("yahoo_answers_topics", None),
        }
        hf_name, hf_config = hf_name_map[dataset_name]
        num_classes_map = {"AGNEWS": 4, "DBPEDIA": 14, "YAHOOANSWERSNLP": 10}
        num_classes = num_classes_map[dataset_name]
        # (body_col, optional_title_col) — title is prepended when present
        text_col_map = {
            "AGNEWS": ("text", None),
            "DBPEDIA": ("content", "title"),
            "YAHOOANSWERSNLP": ("best_answer", "question_title"),
        }

        raw = hf_load(hf_name, hf_config, cache_dir=str(self.config.data_root))
        train_split = raw["train"]
        test_split = raw["test"]
        text_col, title_col = text_col_map[dataset_name]

        def _extract_texts(split: Any) -> list[str]:
            bodies = split[text_col]
            if title_col and title_col in split.column_names:
                titles = split[title_col]
                return [f"{str(t).strip()} {str(b).strip()}" for t, b in zip(titles, bodies)]
            return [str(b) for b in bodies]

        def _tokenize(texts: list[str]) -> dict[str, Any]:
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

        def _build_tensor_dataset(split: Any) -> "_NLPTensorDataset":
            texts = _extract_texts(split)
            labels = split["label"]
            enc = _tokenize(texts)
            return _NLPTensorDataset(
                enc["input_ids"],
                enc["attention_mask"],
                torch.tensor(labels, dtype=torch.long),
            )

        import logging
        logging.getLogger("meld").info(
            "Tokenising %s (train=%d, test=%d) — this takes ~1 min on first run.",
            dataset_name, len(train_split), len(test_split),
        )
        train_full = _build_tensor_dataset(train_split)
        test_full = _build_tensor_dataset(test_split)

        return self._split_nlp_into_tasks(train_full, test_full, num_classes)
        return bundle

    def _synthetic_bundle(self) -> list[tuple[Dataset[Any], Dataset[Any]]]:
        torch.manual_seed(self.config.seed)
        nc = self.config.num_tasks * self.config.classes_per_task
        tasks = []
        for tid in range(self.config.num_tasks):
            tx, ty, ex, ey = [], [], [], []
            for off in range(self.config.classes_per_task):
                cid = tid * self.config.classes_per_task + off
                base = torch.full((3, 32, 32), float(cid) / max(1, nc))
                tx.append(base + 0.05 * torch.randn(32, 3, 32, 32))
                ty.append(torch.full((32,), cid, dtype=torch.long))
                ex.append(base + 0.05 * torch.randn(16, 3, 32, 32))
                ey.append(torch.full((16,), cid, dtype=torch.long))
            tasks.append(
                (
                    TensorDataset(torch.cat(tx), torch.cat(ty)),
                    TensorDataset(torch.cat(ex), torch.cat(ey)),
                )
            )
        return tasks

    def _split_nlp_into_tasks(
        self,
        train_ds: "_NLPTensorDataset",
        test_ds: "_NLPTensorDataset",
        num_classes: int,
    ) -> list[tuple[Dataset[Any], Dataset[Any]]]:
        """Split NLP dataset into class-incremental tasks."""
        from torch.utils.data import Subset

        classes_per_task = self.config.classes_per_task
        num_tasks = min(self.config.num_tasks, num_classes // classes_per_task)

        def _idx_for(ds: "_NLPTensorDataset", class_ids: list[int]) -> list[int]:
            mask = torch.zeros(len(ds), dtype=torch.bool)
            for c in class_ids:
                mask |= ds.labels == c
            return mask.nonzero(as_tuple=False).squeeze(1).tolist()

        bundle: list[tuple[Dataset[Any], Dataset[Any]]] = []
        for task_id in range(num_tasks):
            start = task_id * classes_per_task
            class_ids = list(range(start, start + classes_per_task))
            tr_idx = _idx_for(train_ds, class_ids)
            te_idx = _idx_for(test_ds, class_ids)
            bundle.append((
                _NLPSubset(train_ds, tr_idx, class_ids),
                _NLPSubset(test_ds, te_idx, class_ids),
            ))
        return bundle

    def _run_full_retrain_baseline(
        self,
        train_loaders: list[DataLoader],
        eval_loaders: list[DataLoader],
        *,
        old_class_ids: list[int] | None = None,
        new_class_ids: list[int] | None = None,
    ) -> tuple[MELDModel, dict[str, Any], float, TrainArtifacts]:
        torch.manual_seed(int(self.config.seed) + len(train_loaders))
        use_cached = (
            self.config.full_retrain_interval > 1
            and self._baseline_model_cache is not None
            and (len(train_loaders) % self.config.full_retrain_interval != 0)
        )
        if use_cached:
            bm = self._baseline_model_cache.clone().to(self.device)
            missing = len(train_loaders) * self.config.classes_per_task - bm.classifier.num_classes
            if missing > 0:
                bm.classifier.adaption(missing)
            train_loader = train_loaders[-1]
        else:
            bm = self._build_model()
            bm.classifier.adaption(len(train_loaders) * self.config.classes_per_task)
            train_loader = self._merged_loader(train_loaders, shuffle=True)

        baseline_epochs = self.config.train.full_retrain_epochs or self.config.train.epochs
        train_cfg = replace(self.config.train, epochs=baseline_epochs)
        start = time.time()
        bm, artifacts = self.full_retrain_updater.update(bm, train_loader, None, train_cfg)
        wall = time.time() - start
        self._baseline_model_cache = bm.clone()
        metrics = self._evaluate(
            bm,
            eval_loaders,
            old_class_ids=old_class_ids,
            new_class_ids=new_class_ids,
        )
        metrics["wall_time_seconds"] = wall
        return bm, metrics, wall, artifacts

    @staticmethod
    def _not_applicable_estimate(bound_type: str) -> OracleEstimate:
        return OracleEstimate(
            value=0.0,
            bound_type=bound_type,
            calibrated=False,
            bound_is_formal=False,
        )

    @staticmethod
    def _artifacts_payload(artifacts: TrainArtifacts | None) -> dict[str, Any]:
        if artifacts is None:
            return {
                "epochs_run": 0,
                "ce_loss_per_epoch": [],
                "geometry_loss_per_epoch": [],
                "ewc_loss_per_epoch": [],
                "train_accuracy_per_epoch": [],
                "projected_step_fraction": None,
                "skipped": False,
            }
        return {
            "epochs_run": int(artifacts.epochs_run),
            "ce_loss_per_epoch": list(artifacts.ce_loss_per_epoch),
            "geometry_loss_per_epoch": list(artifacts.geometry_loss_per_epoch),
            "ewc_loss_per_epoch": list(artifacts.ewc_loss_per_epoch),
            "train_accuracy_per_epoch": list(artifacts.train_accuracy_per_epoch),
            "projected_step_fraction": artifacts.projected_step_fraction,
            "skipped": bool(artifacts.skipped),
        }

    def _run_primary_full_retrain(
        self,
        *,
        task_id: int,
        model: MELDModel,
        all_seen_loaders: list[DataLoader],
        eval_loaders: list[DataLoader],
    ) -> tuple[MELDModel, dict[str, Any], TaskSnapshot, Decision, TrainArtifacts]:
        all_known_class_ids = list(range((task_id + 1) * self.config.classes_per_task))
        model, full_metrics, _full_time, full_artifacts = self._run_full_retrain_baseline(
            all_seen_loaders,
            eval_loaders[: task_id + 1],
            old_class_ids=list(range(0, task_id * self.config.classes_per_task)),
            new_class_ids=list(range(task_id * self.config.classes_per_task, (task_id + 1) * self.config.classes_per_task)),
        )
        snapshot = self.snapshot_strategy.capture(
            model,
            self._merged_loader(all_seen_loaders, shuffle=False),
            all_known_class_ids,
            task_id,
        )
        decision = Decision(
            state="FULL_RETRAIN",
            pre_bound=0.0,
            post_bound=0.0,
            bound_held=True,
            shift_score=0.0,
            shift_detected=False,
            reason="Configured to run the primary training path in full_retrain mode.",
            compute_savings_percent=0.0,
            confidence=1.0,
            recommended_action="full_retrain",
            formal_guarantee=False,
        )
        return model, full_metrics, snapshot, decision, full_artifacts

    def _evaluate(
        self,
        model: MELDModel,
        eval_loaders: list[DataLoader],
        *,
        old_class_ids: list[int] | None = None,
        new_class_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        model.eval()
        logits_all, targets_all = [], []
        with torch.no_grad():
            for loader in eval_loaders:
                for inp, tgt in loader:
                    logits_all.append(model(_move_inputs_to_device(inp, self.device, non_blocking=self._pin_memory)))
                    targets_all.append(tgt.to(self.device))
        all_logits = torch.cat(logits_all)
        all_targets = torch.cat(targets_all)
        metrics = compute_classification_metrics(all_logits, all_targets)

        # Optional ECE split by "old" vs "new" class groups.
        if old_class_ids is not None or new_class_ids is not None:
            probs = torch.softmax(all_logits, dim=1).detach().cpu().numpy()
            y_true = all_targets.detach().cpu().numpy()
            if old_class_ids is not None:
                mask_old = np.isin(y_true, np.asarray(old_class_ids))
                metrics["ece_old"] = compute_ece_maybe(probs[mask_old], y_true[mask_old])
            else:
                metrics["ece_old"] = None
            if new_class_ids is not None:
                mask_new = np.isin(y_true, np.asarray(new_class_ids))
                metrics["ece_new"] = compute_ece_maybe(probs[mask_new], y_true[mask_new])
            else:
                metrics["ece_new"] = None
        metrics["wall_time_seconds"] = 0.0
        return metrics

    def _merged_loader(self, loaders: list[DataLoader], shuffle: bool = False) -> DataLoader:
        combined = ConcatDataset([loader.dataset for loader in loaders])
        return DataLoader(
            combined,
            batch_size=self.config.train.batch_size,
            shuffle=shuffle,
            num_workers=self._loader_num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=bool(self._loader_num_workers > 0),
        )

    def _effective_num_workers(self) -> int:
        configured = int(getattr(self.config.train, "num_workers", 0))
        if self.device.type != "cuda":
            return 0
        return max(0, configured)

    def _imprint_new_head(self, model: MELDModel, task_loader: DataLoader, class_ids: list[int]) -> None:
        if not class_ids:
            return

        max_per_class = int(getattr(self.config.train, "imprinting_max_samples_per_class", 64))
        sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {class_id: 0 for class_id in class_ids}
        target_norms = [
            norm
            for class_id, norm in model.classifier.all_norms().items()
            if class_id not in class_ids
        ]
        target_norm = float(np.mean(target_norms)) if target_norms else 1.0

        model.eval()
        with torch.no_grad():
            for inputs, targets in task_loader:
                inputs = _move_inputs_to_device(inputs, self.device, non_blocking=self._pin_memory)
                targets = targets.to(self.device, non_blocking=self._pin_memory)
                embeddings = model.embed(inputs)
                for class_id in class_ids:
                    remaining = max_per_class - counts[class_id]
                    if remaining <= 0:
                        continue
                    mask = targets == class_id
                    if not torch.any(mask):
                        continue
                    selected = embeddings[mask][:remaining]
                    if selected.numel() == 0:
                        continue
                    sums[class_id] = sums.get(class_id, torch.zeros_like(selected[0])) + selected.sum(dim=0)
                    counts[class_id] += int(selected.size(0))
                if all(count >= max_per_class for count in counts.values()):
                    break

        for class_id in class_ids:
            count = counts.get(class_id, 0)
            if count <= 0:
                continue
            prototype = sums[class_id] / float(count)
            prototype = F.normalize(prototype, dim=0) * target_norm
            head_index, offset = model.classifier.class_to_head[class_id]
            head = model.classifier.heads[head_index]
            head.weight.data[offset].copy_(prototype)
            head.bias.data[offset].zero_()

    def _collect_loader_embeddings(self, model: MELDModel, loader: DataLoader, max_samples: int = 512) -> np.ndarray:
        model.eval()
        chunks: list[np.ndarray] = []
        collected = 0
        with torch.no_grad():
            for inputs, _targets in loader:
                inputs = _move_inputs_to_device(inputs, self.device, non_blocking=self._pin_memory)
                embeddings = model.embed(inputs).detach().cpu().numpy()
                if collected + len(embeddings) > max_samples:
                    embeddings = embeddings[: max_samples - collected]
                chunks.append(embeddings)
                collected += len(embeddings)
                if collected >= max_samples:
                    break
        if not chunks:
            return np.empty((0, int(model.out_dim)), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def _pre_risk_estimate(self, snapshot: TaskSnapshot, train_config: Any) -> OracleEstimate:
        if hasattr(self.safety_oracle, "pre_risk_estimate"):
            return self.safety_oracle.pre_risk_estimate(snapshot, train_config)
        return OracleEstimate(
            value=float(self.safety_oracle.pre_bound(snapshot, train_config)),
            bound_type="empirical_spectral",
            calibrated=False,
            bound_is_formal=False,
        )

    def _post_drift_realized(self, snapshot_before: TaskSnapshot, snapshot_after: TaskSnapshot) -> OracleEstimate:
        if hasattr(self.safety_oracle, "post_drift_realized"):
            return self.safety_oracle.post_drift_realized(snapshot_before, snapshot_after)
        return OracleEstimate(
            value=float(self.safety_oracle.post_bound(snapshot_before, snapshot_after)),
            bound_type="realized_old_manifold_drift",
            calibrated=False,
            bound_is_formal=False,
        )

    def _make_safe_train_config(
        self,
        snapshot_before: TaskSnapshot,
        task_train_config: Any,
        pre_risk_estimate: OracleEstimate,
    ) -> tuple[Any, OracleEstimate]:
        if pre_risk_estimate.value <= self.config.bound_tolerance:
            return task_train_config, pre_risk_estimate
        if not bool(getattr(task_train_config, "auto_scale_safe_update", True)):
            return task_train_config, pre_risk_estimate

        min_safe_lr = float(getattr(task_train_config, "min_safe_lr", 1e-5))
        adjusted_config = task_train_config
        adjusted_estimate = pre_risk_estimate

        for _ in range(4):
            if adjusted_estimate.value <= self.config.bound_tolerance:
                break
            ratio = self.config.bound_tolerance / max(adjusted_estimate.value, 1e-12)
            safe_lr = max(min_safe_lr, float(adjusted_config.lr) * ratio * 0.95)
            if safe_lr >= float(adjusted_config.lr):
                break
            adjusted_config = replace(adjusted_config, lr=safe_lr)
            adjusted_estimate = self._pre_risk_estimate(snapshot_before, adjusted_config)

        return adjusted_config, adjusted_estimate

    def _base_train_config(self) -> Any:
        base_epochs = getattr(self.config.train, "base_epochs", None)
        if base_epochs is None:
            return self.config.train
        return replace(self.config.train, epochs=int(base_epochs))

    def _uses_frozen_analytic(self, task_id: int) -> bool:
        if self._custom_updater_provided:
            return False
        return task_id > 0 and str(getattr(self.config.train, "incremental_strategy", "geometry")) == "frozen_analytic"

    def _delta_updater_for_task(self, task_id: int) -> ManifoldUpdater:
        if self._uses_frozen_analytic(task_id):
            return self.analytic_updater
        return self.updater

    def _pretraining_shift(
        self,
        snapshot_before: TaskSnapshot,
        model: MELDModel,
        task_loader: DataLoader,
    ) -> DriftResult:
        embeddings = self._collect_loader_embeddings(model, task_loader)
        return self.pretrain_shift_detector.detect_from_embeddings(snapshot_before, embeddings)

    def _task_result(
        self,
        task_id: int,
        delta_metrics: dict[str, Any] | None,
        full_metrics: dict[str, Any],
        snapshot_before: TaskSnapshot | None,
        snapshot: TaskSnapshot | None,
        pre_risk_estimate: OracleEstimate,
        post_drift_realized: OracleEstimate,
        drift_result: DriftResult,
        decision: Decision,
        delta_wall_time_seconds: float,
        cil_metrics: dict[str, Any] | None = None,
        train_artifacts: TrainArtifacts | None = None,
        pretrain_shift: DriftResult | None = None,
        pre_pac_style: OracleEstimate | None = None,
    ) -> dict[str, Any]:
        if (
            task_id > 0
            and pre_risk_estimate.value > 0.0
            and hasattr(self.safety_oracle, "empirical_calibrated_estimate")
            and snapshot is not None
        ):
            calibrated_estimate = self.safety_oracle.empirical_calibrated_estimate(snapshot, self.config.train)
        else:
            calibrated_estimate = None
        if pre_pac_style is not None:
            pac_style = pre_pac_style
        elif task_id > 0 and hasattr(self.safety_oracle, "pac_style_gap") and snapshot is not None:
            pac_style = self.safety_oracle.pac_style_gap(snapshot)
        else:
            pac_style = None
        if (
            task_id > 0
            and snapshot_before is not None
            and snapshot is not None
            and hasattr(self.safety_oracle, "pac_equivalence_bound")
        ):
            pac_equivalence = self.safety_oracle.pac_equivalence_bound(snapshot_before, snapshot)
        else:
            pac_equivalence = None
        decision.formal_guarantee = bool(
            task_id > 0
            and bool(getattr(self.config.train, "enable_importance_weighting", False))
            and (
                (pac_equivalence is not None and pac_equivalence.bound_is_formal and pac_equivalence.value <= float(getattr(self.config, "pac_gate_tolerance", 0.1)))
                or (pac_style is not None and pac_style.bound_is_formal and pac_style.value <= float(getattr(self.config, "pac_gate_tolerance", 0.1)))
            )
            and decision.state in {"SAFE_DELTA", "CAUTIOUS_DELTA"}
        )
        fc = np.asarray(full_metrics["confusion_matrix"])
        if delta_metrics is not None:
            dc = np.asarray(delta_metrics["confusion_matrix"])
            equivalence_gap = compute_equivalence_gap(dc, fc)
            forgetting = max(0.0, full_metrics["top1"] - delta_metrics["top1"])
            delta_payload = {
                "top1": delta_metrics["top1"],
                "top5": delta_metrics["top5"],
                "ece": delta_metrics["ece"],
                "ece_old": delta_metrics.get("ece_old"),
                "ece_new": delta_metrics.get("ece_new"),
                "per_class_acc": delta_metrics["per_class_accuracy"],
                "wall_time_seconds": delta_metrics.get("wall_time_seconds", delta_wall_time_seconds),
            }
        else:
            equivalence_gap = None
            forgetting = None
            delta_payload = {
                "top1": None,
                "top5": None,
                "ece": None,
                "ece_old": None,
                "ece_new": None,
                "per_class_acc": {},
                "wall_time_seconds": delta_wall_time_seconds,
                "skipped": True,
            }
        return {
            "task_id": task_id,
            "delta": delta_payload,
            "full_retrain": {
                "top1": full_metrics["top1"],
                "top5": full_metrics["top5"],
                "ece": full_metrics["ece"],
                "ece_old": full_metrics.get("ece_old"),
                "ece_new": full_metrics.get("ece_new"),
                "per_class_acc": full_metrics["per_class_accuracy"],
                "wall_time_seconds": full_metrics.get("wall_time_seconds", 0.0),
            },
            "snapshot": {"fisher_eigenvalue_max": snapshot.fisher_eigenvalue_max if snapshot else 0.0, "class_ids": snapshot.class_ids if snapshot else []},
            "oracle": {
                "risk_estimate_pre": pre_risk_estimate.value,
                "drift_realized_post": post_drift_realized.value,
                "risk_estimate_held": decision.bound_held if task_id > 0 else True,
                "bound_type": pre_risk_estimate.bound_type,
                "calibrated": pre_risk_estimate.calibrated,
                "bound_is_formal": pre_risk_estimate.bound_is_formal,
                "fisher_saturated": pre_risk_estimate.fisher_saturated,
                "derivation": pre_risk_estimate.derivation,
                "empirical_calibrated_estimate": (
                    {
                        "value": calibrated_estimate.value,
                        "bound_type": calibrated_estimate.bound_type,
                        "calibrated": calibrated_estimate.calibrated,
                        "bound_is_formal": calibrated_estimate.bound_is_formal,
                        "fisher_saturated": calibrated_estimate.fisher_saturated,
                        "derivation": calibrated_estimate.derivation,
                    }
                    if calibrated_estimate is not None
                    else None
                ),
                "pac_style_gap": (
                    {
                        "value": pac_style.value,
                        "delta": pac_style.delta,
                        "bound_type": pac_style.bound_type,
                        "calibrated": pac_style.calibrated,
                        "bound_is_formal": pac_style.bound_is_formal,
                        "derivation": pac_style.derivation,
                    }
                    if pac_style is not None
                    else None
                ),
                "pac_equivalence_bound": (
                    {
                        "value": pac_equivalence.value,
                        "delta": pac_equivalence.delta,
                        "bound_type": pac_equivalence.bound_type,
                        "calibrated": pac_equivalence.calibrated,
                        "bound_is_formal": pac_equivalence.bound_is_formal,
                        "derivation": pac_equivalence.derivation,
                    }
                    if pac_equivalence is not None
                    else None
                ),
            },
            "drift": {
                "shift_score": drift_result.shift_score,
                "shift_detected": drift_result.shift_detected,
                "severity": drift_result.severity,
                "per_class_drift": drift_result.per_class_drift,
                "detector_scores": drift_result.detector_scores,
                "input_shift_score": drift_result.input_shift_score,
                "pretrain_mmd_score": pretrain_shift.shift_score if pretrain_shift is not None else None,
                "pretrain_mmd_detected": pretrain_shift.shift_detected if pretrain_shift is not None else None,
                "pretrain_mmd_severity": pretrain_shift.severity if pretrain_shift is not None else None,
            },
            "decision": asdict(decision),
            "equivalence_gap": equivalence_gap,
            "forgetting": forgetting,
            "compute_savings_percent": decision.compute_savings_percent,
            "cil_metrics": cil_metrics,
            "train": {
                "epochs_run": train_artifacts.epochs_run if train_artifacts else 0,
                "ce_loss_per_epoch": train_artifacts.ce_loss_per_epoch if train_artifacts else [],
                "geometry_loss_per_epoch": train_artifacts.geometry_loss_per_epoch if train_artifacts else [],
                "ewc_loss_per_epoch": train_artifacts.ewc_loss_per_epoch if train_artifacts else [],
                "train_accuracy_per_epoch": train_artifacts.train_accuracy_per_epoch if train_artifacts else [],
                "projected_step_fraction": train_artifacts.projected_step_fraction if train_artifacts else None,
                "skipped": train_artifacts.skipped if train_artifacts else False,
            },
        }

    def _summarize(
        self,
        tasks: list[dict[str, Any]],
        robustness: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not tasks:
            return {}
        delta_top1s = [t["delta"].get("top1") for t in tasks if isinstance(t.get("delta"), dict) and t["delta"].get("top1") is not None]
        delta_eces = [t["delta"].get("ece") for t in tasks if isinstance(t.get("delta"), dict) and t["delta"].get("ece") is not None]
        full_eces = [t["full_retrain"]["ece"] for t in tasks if t["full_retrain"].get("ece") is not None]
        eq_gaps = [t.get("equivalence_gap") for t in tasks if t.get("equivalence_gap") is not None]
        summary = {
            "mean_delta_top1": float(np.mean(delta_top1s)) if delta_top1s else None,
            "mean_full_retrain_top1": float(np.mean([t["full_retrain"]["top1"] for t in tasks])),
            "mean_delta_ece": float(np.mean(delta_eces)) if delta_eces else None,
            "mean_full_retrain_ece": float(np.mean(full_eces)) if full_eces else None,
            "mean_equivalence_gap": float(np.mean(eq_gaps)) if eq_gaps else None,
            "mean_compute_savings": float(np.mean([t["compute_savings_percent"] for t in tasks])),
            "decisions": [t["decision"]["state"] for t in tasks],
            "total_wall_time_delta": float(np.sum([float(t["delta"].get("wall_time_seconds", 0.0)) for t in tasks if isinstance(t.get("delta"), dict)])),
            "total_wall_time_full_retrain": float(np.sum([t["full_retrain"]["wall_time_seconds"] for t in tasks])),
        }
        if summary["mean_delta_ece"] is not None and summary["mean_full_retrain_ece"] is not None:
            summary["ece_preserved"] = bool(
                summary["mean_delta_ece"] <= (summary["mean_full_retrain_ece"] + 0.02)
            )
        else:
            summary["ece_preserved"] = None
        if robustness and robustness.get("status") == "completed":
            mean_robustness = robustness.get("mean_top1")
            summary["mean_robustness_accuracy"] = mean_robustness
            if mean_robustness is not None and summary["mean_full_retrain_top1"] is not None:
                summary["robustness_gap"] = float(summary["mean_full_retrain_top1"] - mean_robustness)
            else:
                summary["robustness_gap"] = None
        return summary

    def _seed_everything(self, seed: int) -> None:
        import random as _random

        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _config_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.config.dataset,
            "dataset_provider": (
                getattr(getattr(self.config, "dataset_provider", None), "__name__", type(getattr(self.config, "dataset_provider", None)).__name__)
                if getattr(self.config, "dataset_provider", None) is not None
                else None
            ),
            "num_tasks": self.config.num_tasks,
            "classes_per_task": self.config.classes_per_task,
            "run_mode": self._run_mode(),
            "bound_tolerance": self.config.bound_tolerance,
            "pac_gate_tolerance": self.config.pac_gate_tolerance,
            "shift_threshold": self.config.shift_threshold,
            "prefer_cuda": self.config.prefer_cuda,
            "cifar_c_path": str(self.config.cifar_c_path) if getattr(self.config, "cifar_c_path", None) else None,
            "database_path": str(self.config.database_path) if getattr(self.config, "database_path", None) else None,
            "seed": self.config.seed,
            "train": asdict(self.config.train),
        }

    def _write_results(self, results_path: str | Path | None, payload: dict[str, Any]) -> None:
        if results_path is None:
            self._result_store.sync_run(payload)
            return
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        self._result_store.sync_run(payload)


class _PolicyConfigProxy:
    def __init__(
        self,
        shift_threshold: float,
        delta_wall_time_seconds: float,
        full_retrain_wall_time_seconds: float,
    ) -> None:
        self.shift_threshold = shift_threshold
        self.delta_wall_time_seconds = delta_wall_time_seconds
        self.full_retrain_wall_time_seconds = full_retrain_wall_time_seconds


class _TaskDatasetAdapter(Dataset[Any]):
    """Wraps a Continuum task dataset with CIFAR-standard augmentation.

    Training split:  random crop (pad=4, reflect) + random h-flip + normalize
    Eval split:      normalize only

    Args:
        dataset:   A Continuum task dataset (supports __len__ and __getitem__).
        mean:      Per-channel mean for normalization (default: CIFAR-10 stats).
        std:       Per-channel std  for normalization (default: CIFAR-10 stats).
        is_train:  If True, applies random crop and flip augmentation.
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        mean: tuple[float, float, float] = _CIFAR10_MEAN,
        std: tuple[float, float, float] = _CIFAR10_STD,
        is_train: bool = False,
    ) -> None:
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.dataset[index]
        if not isinstance(sample, tuple):
            raise TypeError(
                f"Expected dataset sample to be a tuple, got {type(sample).__name__}."
            )
        inputs, target = sample[0], sample[1]
        tensor = self._to_tensor(inputs)

        if self.is_train:
            tensor = TF.pad(tensor, padding=4, padding_mode="reflect")
            i = random.randint(0, 8)
            j = random.randint(0, 8)
            tensor = TF.crop(tensor, i, j, 32, 32)
            if random.random() > 0.5:
                tensor = TF.hflip(tensor)

        mean_t = torch.tensor(self.mean, dtype=tensor.dtype).view(3, 1, 1)
        std_t = torch.tensor(self.std, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t
        return tensor, int(target)

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            t = value.float()
            if t.max().item() > 2.0:
                t = t / 255.0
            return t
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] in {1, 3}:
            arr = np.transpose(arr, (2, 0, 1))
        t = torch.from_numpy(arr).float()
        if t.max().item() > 2.0:
            t = t / 255.0
        return t


class _ClassRemapDataset(Dataset):  # type: ignore[type-arg]
    """Wraps a Subset and remaps absolute class ids to task-local ids.

    E.g. for task 2 with class_ids=[20,21,22], label 20 → 20 (global ids
    are preserved so the incremental classifier can accumulate them).
    """

    def __init__(self, subset: Any, class_ids: list[int]) -> None:
        self.subset = subset
        self.class_ids = class_ids

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        x, y = self.subset[idx]
        return x, int(y)


class _NLPTensorDataset(Dataset):  # type: ignore[type-arg]
    """Pre-tokenised text dataset backed by tensors.

    Each item returns (input_ids, attention_mask, label) as a dict so
    the DataLoader can batch them. The model forward pass receives the dict
    and passes input_ids + attention_mask to the text encoder.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], int]:
        return (
            {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            },
            int(self.labels[idx].item()),
        )


class _NLPSubset(Dataset):  # type: ignore[type-arg]
    """Task-specific slice of an _NLPTensorDataset."""

    def __init__(
        self,
        full: "_NLPTensorDataset",
        indices: list[int],
        class_ids: list[int],
    ) -> None:
        idx_t = torch.tensor(indices, dtype=torch.long)
        self.input_ids = full.input_ids[idx_t]
        self.attention_mask = full.attention_mask[idx_t]
        self.labels = full.labels[idx_t]
        self.class_ids = class_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], int]:
        return (
            {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            },
            int(self.labels[idx].item()),
        )
