"""End-to-end benchmark runner for MELD."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

from ..core.auto_config import derive_train_config
from ..core.corrector import AnalyticNormCorrector
from ..core.drift import KLManifoldDriftDetector
from ..core.oracle import SpectralSafetyOracle
from ..core.policy import FourStateDeployPolicy
from ..core.snapshot import FisherManifoldSnapshot
from ..core.updater import GeometryConstrainedUpdater
from ..interfaces.base import Decision, DriftResult, TaskSnapshot, TrainArtifacts
from ..modeling import MELDModel
from ..models.backbone import resnet20, resnet32, resnet44, resnet56
from ..models.classifier import IncrementalClassifier
from .metrics import compute_classification_metrics, compute_compute_savings, compute_equivalence_gap

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
        updater: GeometryConstrainedUpdater | None = None,
        corrector: AnalyticNormCorrector | None = None,
        drift_detector: KLManifoldDriftDetector | None = None,
        deploy_policy: FourStateDeployPolicy | None = None,
    ) -> None:
        self.config = config
        self.snapshot_strategy = snapshot_strategy or FisherManifoldSnapshot()
        self.safety_oracle = safety_oracle or SpectralSafetyOracle()
        self.updater = updater or GeometryConstrainedUpdater()
        self.corrector = corrector or AnalyticNormCorrector()
        self.drift_detector = drift_detector or KLManifoldDriftDetector(config.shift_threshold)
        self.deploy_policy = deploy_policy or FourStateDeployPolicy()
        if config.prefer_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self._baseline_model_cache: MELDModel | None = None

    def run(self, results_path: str | Path | None = None) -> dict[str, Any]:
        tasks, eval_loaders = self._build_tasks()
        model = self._build_model()
        all_seen_loaders: list[DataLoader] = []
        results: dict[str, Any] = {
            "status": "running",
            "config": self._config_dict(),
            "tasks": [],
            "final_summary": None,
        }
        self._write_results(results_path, results)

        for task_id, task_loader in enumerate(tasks):
            all_seen_loaders.append(task_loader)
            new_class_ids = model.classifier.adaption(self.config.classes_per_task)

            snapshot_before = None
            pre_bound = 0.0
            if task_id > 0:
                snapshot_before = self.snapshot_strategy.capture(
                    model,
                    self._merged_loader(all_seen_loaders[:-1], shuffle=False),
                    list(range(model.classifier.num_classes - self.config.classes_per_task)),
                    task_id - 1,
                )
                self.snapshot_strategy._ema_decay = float(
                    getattr(self.config.train, "fisher_ema_decay", 0.9)
                )
                if getattr(self.config.train, "auto_derive_hparams", False):
                    task_train_config = derive_train_config(
                        snapshot_before,
                        self.config.train,
                        getattr(self.config.train, "protection_level", 0.5),
                    )
                else:
                    task_train_config = self.config.train
                pre_bound = self.safety_oracle.pre_bound(snapshot_before, self.config.train)
            else:
                task_train_config = self.config.train

            if task_id > 0 and pre_bound > self.config.bound_tolerance:
                delta_artifacts = TrainArtifacts(0, [], [], [], [], 0.0, skipped=True)
                snapshot_after = snapshot_before
                post_bound = pre_bound
                drift_result = DriftResult(0.0, False, {}, "none")
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
                model, delta_artifacts = self.updater.update(
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
                full_metrics, full_time = self._run_full_retrain_baseline(
                    all_seen_loaders, eval_loaders[: task_id + 1]
                )
                delta_metrics = self._evaluate(model, eval_loaders[: task_id + 1])
                decision.compute_savings_percent = compute_compute_savings(
                    delta_artifacts.wall_time_seconds, full_time
                )
                delta_metrics["wall_time_seconds"] = delta_artifacts.wall_time_seconds
                task_result = self._task_result(
                    task_id,
                    delta_metrics,
                    full_metrics,
                    snapshot_after,
                    pre_bound,
                    post_bound,
                    drift_result,
                    decision,
                )
                results["tasks"].append(task_result)
                self._write_results(results_path, results)
                continue

            full_metrics, full_time = self._run_full_retrain_baseline(
                all_seen_loaders, eval_loaders[: task_id + 1]
            )
            delta_metrics = self._evaluate(model, eval_loaders[: task_id + 1])
            delta_metrics["wall_time_seconds"] = delta_artifacts.wall_time_seconds
            decision.compute_savings_percent = compute_compute_savings(
                delta_artifacts.wall_time_seconds, full_time
            )
            task_result = self._task_result(
                task_id,
                delta_metrics,
                full_metrics,
                snapshot_after,
                pre_bound,
                post_bound,
                drift_result,
                decision,
            )
            results["tasks"].append(task_result)
            self._write_results(results_path, results)

        results["status"] = "completed"
        results["final_summary"] = self._summarize(results["tasks"])
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
        for train_ds, test_ds in bundle:
            train_tasks.append(
                DataLoader(
                    train_ds,
                    batch_size=self.config.train.batch_size,
                    shuffle=True,
                    num_workers=self.config.train.num_workers,
                )
            )
            eval_tasks.append(
                DataLoader(
                    test_ds,
                    batch_size=self.config.train.batch_size,
                    shuffle=False,
                    num_workers=self.config.train.num_workers,
                )
            )
        return train_tasks, eval_tasks

    def _load_dataset_bundle(self) -> list[tuple[Dataset[Any], Dataset[Any]]]:
        try:
            from continuum import ClassIncremental
            from continuum.datasets import CIFAR10, CIFAR100
        except ModuleNotFoundError as exc:
            import warnings

            warnings.warn(
                f"Continuum not installed ({exc}). "
                "Falling back to synthetic data. "
                "Install with: pip install continuum-learn",
                stacklevel=2,
            )
            return self._synthetic_bundle()

        d = self.config.dataset.upper().replace("-", "")
        if d == "CIFAR10":
            tr = CIFAR10(data_path=str(self.config.data_root), train=True, download=True)
            te = CIFAR10(data_path=str(self.config.data_root), train=False, download=True)
            mean, std = _CIFAR10_MEAN, _CIFAR10_STD
        elif d == "CIFAR100":
            tr = CIFAR100(data_path=str(self.config.data_root), train=True, download=True)
            te = CIFAR100(data_path=str(self.config.data_root), train=False, download=True)
            mean, std = _CIFAR100_MEAN, _CIFAR100_STD
        else:
            raise ValueError(
                f"Unsupported dataset '{self.config.dataset}'. "
                "Supported: CIFAR-10, CIFAR-100."
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
        cfg = asdict(self.config.train)
        cfg["epochs"] = baseline_epochs
        start = time.time()
        bm, _ = self.updater.update(bm, train_loader, None, _ConfigView(cfg))
        wall = time.time() - start
        self._baseline_model_cache = bm.clone()
        metrics = self._evaluate(bm, eval_loaders)
        metrics["wall_time_seconds"] = wall
        return metrics, wall

    def _evaluate(self, model: MELDModel, eval_loaders: list[DataLoader]) -> dict[str, Any]:
        model.eval()
        logits_all, targets_all = [], []
        with torch.no_grad():
            for loader in eval_loaders:
                for inp, tgt in loader:
                    logits_all.append(model(inp.to(self.device)))
                    targets_all.append(tgt.to(self.device))
        metrics = compute_classification_metrics(
            torch.cat(logits_all), torch.cat(targets_all)
        )
        metrics["wall_time_seconds"] = 0.0
        return metrics

    def _merged_loader(self, loaders: list[DataLoader], shuffle: bool = False) -> DataLoader:
        combined = ConcatDataset([loader.dataset for loader in loaders])
        return DataLoader(
            combined,
            batch_size=self.config.train.batch_size,
            shuffle=shuffle,
            num_workers=self.config.train.num_workers,
        )

    def _task_result(
        self,
        task_id: int,
        delta_metrics: dict[str, Any],
        full_metrics: dict[str, Any],
        snapshot: TaskSnapshot | None,
        pre_bound: float,
        post_bound: float,
        drift_result: DriftResult,
        decision: Decision,
    ) -> dict[str, Any]:
        dc = np.asarray(delta_metrics["confusion_matrix"])
        fc = np.asarray(full_metrics["confusion_matrix"])
        return {
            "task_id": task_id,
            "delta": {"top1": delta_metrics["top1"], "top5": delta_metrics["top5"], "ece": delta_metrics["ece"], "per_class_acc": delta_metrics["per_class_accuracy"], "wall_time_seconds": delta_metrics.get("wall_time_seconds", 0.0)},
            "full_retrain": {"top1": full_metrics["top1"], "top5": full_metrics["top5"], "ece": full_metrics["ece"], "per_class_acc": full_metrics["per_class_accuracy"], "wall_time_seconds": full_metrics.get("wall_time_seconds", 0.0)},
            "snapshot": {"fisher_eigenvalue_max": snapshot.fisher_eigenvalue_max if snapshot else 0.0, "class_ids": snapshot.class_ids if snapshot else []},
            "oracle": {"pre_bound": pre_bound, "post_bound": post_bound, "bound_held": decision.bound_held if task_id > 0 else True},
            "drift": {"shift_score": drift_result.shift_score, "shift_detected": drift_result.shift_detected, "severity": drift_result.severity, "per_class_drift": drift_result.per_class_drift},
            "decision": asdict(decision),
            "equivalence_gap": compute_equivalence_gap(dc, fc),
            "forgetting": max(0.0, full_metrics["top1"] - delta_metrics["top1"]),
            "compute_savings_percent": decision.compute_savings_percent,
        }

    def _summarize(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        if not tasks:
            return {}
        return {
            "mean_delta_top1": float(np.mean([t["delta"]["top1"] for t in tasks])),
            "mean_full_retrain_top1": float(np.mean([t["full_retrain"]["top1"] for t in tasks])),
            "mean_equivalence_gap": float(np.mean([t["equivalence_gap"] for t in tasks])),
            "mean_compute_savings": float(np.mean([t["compute_savings_percent"] for t in tasks])),
            "decisions": [t["decision"]["state"] for t in tasks],
            "total_wall_time_delta": float(np.sum([t["delta"]["wall_time_seconds"] for t in tasks])),
            "total_wall_time_full_retrain": float(np.sum([t["full_retrain"]["wall_time_seconds"] for t in tasks])),
        }

    def _config_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.config.dataset,
            "num_tasks": self.config.num_tasks,
            "classes_per_task": self.config.classes_per_task,
            "bound_tolerance": self.config.bound_tolerance,
            "shift_threshold": self.config.shift_threshold,
            "prefer_cuda": self.config.prefer_cuda,
            "seed": self.config.seed,
            "train": asdict(self.config.train),
        }

    def _write_results(self, results_path: str | Path | None, payload: dict[str, Any]) -> None:
        if results_path is None:
            return
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)


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


class _ConfigView:
    """Thin attribute-access wrapper around a plain dict (for baseline config)."""

    def __init__(self, values: dict[str, Any]) -> None:
        for k, v in values.items():
            setattr(self, k, v)


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
