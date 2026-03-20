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
from ..core.drift import CompositeDriftDetector
from ..core.oracle import SpectralSafetyOracle
from ..core.policy import FourStateDeployPolicy
from ..core.snapshot import FisherManifoldSnapshot
from ..core.updater import FrozenBackboneAnalyticUpdater, FullRetrainUpdater, GeometryConstrainedUpdater
from ..interfaces.base import Decision, DriftDetector, DriftResult, ManifoldUpdater, TaskSnapshot, TrainArtifacts
from ..modeling import MELDModel
from ..models.backbone import resnet20, resnet32, resnet44, resnet56
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
}

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


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
    d = dataset.upper().replace("-", "")
    if d == "CIFAR10":
        return "resnet32"
    return "resnet56"


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
            pre_bound = 0.0
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
                if self._uses_frozen_analytic(task_id):
                    pre_bound = 0.0
                else:
                    pre_bound = self.safety_oracle.pre_bound(snapshot_before, task_train_config)
                    task_train_config, pre_bound = self._make_safe_train_config(
                        snapshot_before,
                        task_train_config,
                        pre_bound,
                    )
            else:
                task_train_config = self._base_train_config()

            if task_id > 0 and pre_bound > self.config.bound_tolerance:
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
                post_bound = pre_bound
                drift_result = DriftResult(0.0, False, {}, "none")
                delta_metrics = None
                decision = Decision(
                    state="BOUND_EXCEEDED",
                    pre_bound=pre_bound,
                    post_bound=post_bound,
                    bound_held=False,
                    shift_score=0.0,
                    shift_detected=False,
                    reason="pre-training safety bound exceeded tolerance before training",
                    compute_savings_percent=0.0,
                    confidence=1.0,
                    recommended_action="full_retrain",
                )
            else:
                delta_updater = self._delta_updater_for_task(task_id)
                model, delta_artifacts = delta_updater.update(
                    model, task_loader, snapshot_before, task_train_config
                )
                if snapshot_before is not None:
                    model = self.corrector.correct(model, snapshot_before)
                    snapshot_after = self.snapshot_strategy.capture(
                        model,
                        self._merged_loader(all_seen_loaders[:-1], shuffle=False),
                        snapshot_before.class_ids,
                        task_id,
                    )
                    post_bound = self.safety_oracle.post_bound(snapshot_before, snapshot_after)
                    drift_result = self.drift_detector.detect(snapshot_before, snapshot_after)
                else:
                    snapshot_after = self.snapshot_strategy.capture(
                        model, task_loader, new_class_ids, task_id
                    )
                    post_bound = 0.0
                    drift_result = DriftResult(0.0, False, {}, "none")

                decision = self.deploy_policy.decide(
                    pre_bound,
                    post_bound,
                    drift_result,
                    _PolicyConfigProxy(
                        shift_threshold=self.config.shift_threshold,
                        delta_wall_time_seconds=delta_artifacts.wall_time_seconds,
                        full_retrain_wall_time_seconds=0.0,
                    ),
                )
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
                    snapshot_after,
                    pre_bound,
                    post_bound,
                    drift_result,
                    decision,
                    delta_wall_time_seconds=float(delta_artifacts.wall_time_seconds),
                    cil_metrics=cil_metrics,
                    train_artifacts=delta_artifacts,
                )
                results["tasks"].append(task_result)
                results["bounds_timeline"].append(
                    {
                        "task_id": task_id,
                        "epsilon_max": float(pre_bound),
                        "epsilon_actual": float(post_bound),
                        "bound_held": bool(decision.bound_held if task_id > 0 else True),
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
                snapshot_after,
                pre_bound,
                post_bound,
                drift_result,
                decision,
                delta_wall_time_seconds=float(delta_artifacts.wall_time_seconds),
                cil_metrics=cil_metrics,
                train_artifacts=delta_artifacts,
            )
            results["tasks"].append(task_result)
            results["bounds_timeline"].append(
                {
                    "task_id": task_id,
                    "epsilon_max": float(pre_bound),
                    "epsilon_actual": float(post_bound),
                    "bound_held": bool(decision.bound_held if task_id > 0 else True),
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
        results["final_summary"] = self._summarize(results["tasks"])
        if bool(getattr(self.config, "run_robustness_eval", False)):
            try:
                results["robustness"] = evaluate_cifar_c(
                    model,
                    dataset=self.config.dataset,
                    data_root=self.config.data_root,
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
        self._write_results(results_path, results)
        return results

    def _build_model(self) -> MELDModel:
        backbone_name: str = self.config.train.backbone
        if backbone_name == "resnet32":
            suggested = _auto_backbone(self.config.dataset)
            if suggested != backbone_name:
                import warnings

                warnings.warn(
                    f"backbone='resnet32' was auto-upgraded to '{suggested}' "
                    f"for dataset '{self.config.dataset}'. "
                    "Set TrainConfig(backbone=...) explicitly to suppress this.",
                    stacklevel=2,
                )
                backbone_name = suggested
        if backbone_name not in BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Valid choices: {sorted(BACKBONES)}"
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
        dataset_name = self.config.dataset.upper().replace("-", "")
        if dataset_name == "SYNTHETIC":
            return self._synthetic_bundle()

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
                "torchvision is required for CIFAR dataset support. "
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
        else:
            raise ValueError(
                f"Unsupported dataset '{self.config.dataset}'. "
                "Supported values: synthetic, CIFAR-10, CIFAR-100."
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

    def _run_full_retrain_baseline(
        self,
        train_loaders: list[DataLoader],
        eval_loaders: list[DataLoader],
        *,
        old_class_ids: list[int] | None = None,
        new_class_ids: list[int] | None = None,
    ) -> tuple[dict[str, Any], float]:
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
        bm, _ = self.full_retrain_updater.update(bm, train_loader, None, train_cfg)
        wall = time.time() - start
        self._baseline_model_cache = bm.clone()
        metrics = self._evaluate(
            bm,
            eval_loaders,
            old_class_ids=old_class_ids,
            new_class_ids=new_class_ids,
        )
        metrics["wall_time_seconds"] = wall
        return metrics, wall

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
                    logits_all.append(model(inp.to(self.device)))
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
                inputs = inputs.to(self.device, non_blocking=self._pin_memory)
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

    def _make_safe_train_config(
        self,
        snapshot_before: TaskSnapshot,
        task_train_config: Any,
        pre_bound: float,
    ) -> tuple[Any, float]:
        if pre_bound <= self.config.bound_tolerance:
            return task_train_config, pre_bound
        if not bool(getattr(task_train_config, "auto_scale_safe_update", True)):
            return task_train_config, pre_bound

        min_safe_lr = float(getattr(task_train_config, "min_safe_lr", 1e-5))
        adjusted_config = task_train_config
        adjusted_bound = pre_bound

        for _ in range(4):
            if adjusted_bound <= self.config.bound_tolerance:
                break
            ratio = self.config.bound_tolerance / max(adjusted_bound, 1e-12)
            safe_lr = max(min_safe_lr, float(adjusted_config.lr) * ratio * 0.95)
            if safe_lr >= float(adjusted_config.lr):
                break
            adjusted_config = replace(adjusted_config, lr=safe_lr)
            adjusted_bound = self.safety_oracle.pre_bound(snapshot_before, adjusted_config)

        return adjusted_config, adjusted_bound

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

    def _task_result(
        self,
        task_id: int,
        delta_metrics: dict[str, Any] | None,
        full_metrics: dict[str, Any],
        snapshot: TaskSnapshot | None,
        pre_bound: float,
        post_bound: float,
        drift_result: DriftResult,
        decision: Decision,
        delta_wall_time_seconds: float,
        cil_metrics: dict[str, Any] | None = None,
        train_artifacts: TrainArtifacts | None = None,
    ) -> dict[str, Any]:
        if hasattr(self.safety_oracle, "pac_equivalence_gap"):
            pac_epsilon, pac_delta = self.safety_oracle.pac_equivalence_gap()
        else:
            pac_epsilon, pac_delta = 0.0, 0.0
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
                "pre_bound": pre_bound,
                "post_bound": post_bound,
                "bound_held": decision.bound_held if task_id > 0 else True,
                "pac_epsilon": pac_epsilon,
                "pac_delta": pac_delta,
            },
            "drift": {
                "shift_score": drift_result.shift_score,
                "shift_detected": drift_result.shift_detected,
                "severity": drift_result.severity,
                "per_class_drift": drift_result.per_class_drift,
                "detector_scores": drift_result.detector_scores,
                "input_shift_score": drift_result.input_shift_score,
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

    def _summarize(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        if not tasks:
            return {}
        delta_top1s = [t["delta"].get("top1") for t in tasks if isinstance(t.get("delta"), dict) and t["delta"].get("top1") is not None]
        eq_gaps = [t.get("equivalence_gap") for t in tasks if t.get("equivalence_gap") is not None]
        return {
            "mean_delta_top1": float(np.mean(delta_top1s)) if delta_top1s else None,
            "mean_full_retrain_top1": float(np.mean([t["full_retrain"]["top1"] for t in tasks])),
            "mean_equivalence_gap": float(np.mean(eq_gaps)) if eq_gaps else None,
            "mean_compute_savings": float(np.mean([t["compute_savings_percent"] for t in tasks])),
            "decisions": [t["decision"]["state"] for t in tasks],
            "total_wall_time_delta": float(np.sum([float(t["delta"].get("wall_time_seconds", 0.0)) for t in tasks if isinstance(t.get("delta"), dict)])),
            "total_wall_time_full_retrain": float(np.sum([t["full_retrain"]["wall_time_seconds"] for t in tasks])),
        }

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
            "num_tasks": self.config.num_tasks,
            "classes_per_task": self.config.classes_per_task,
            "bound_tolerance": self.config.bound_tolerance,
            "shift_threshold": self.config.shift_threshold,
            "prefer_cuda": self.config.prefer_cuda,
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
