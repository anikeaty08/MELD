"""End-to-end benchmark runner for MELD."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..core.corrector import AnalyticNormCorrector
from ..core.drift import KLManifoldDriftDetector
from ..core.oracle import SpectralSafetyOracle
from ..core.policy import FourStateDeployPolicy
from ..core.snapshot import FisherManifoldSnapshot
from ..core.updater import GeometryConstrainedUpdater
from ..core.auto_config import derive_train_config
from ..interfaces.base import Decision, DriftResult, TaskSnapshot, TrainArtifacts
from ..modeling import MELDModel
from ..models.backbone import resnet20, resnet32, resnet44, resnet56
from ..models.classifier import IncrementalClassifier
from .metrics import compute_classification_metrics, compute_compute_savings, compute_equivalence_gap


BACKBONES = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
}


class BenchmarkRunner:
    def __init__(
        self,
        config: Any,
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
        self.device = torch.device("cuda" if config.prefer_cuda and torch.cuda.is_available() else "cpu")
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
            all_seen_class_ids = list(range(model.classifier.num_classes))

            snapshot_before = None
            pre_bound = 0.0
            if task_id > 0:
                snapshot_before = self.snapshot_strategy.capture(
                    model,
                    self._merged_loader(all_seen_loaders[:-1], shuffle=False),
                    list(range(model.classifier.num_classes - self.config.classes_per_task)),
                    task_id - 1,
                )
                self.snapshot_strategy._ema_decay = float(getattr(self.config.train, "fisher_ema_decay", 0.9))
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
                model, delta_artifacts = self.updater.update(model, task_loader, snapshot_before, task_train_config)
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
                    snapshot_after = self.snapshot_strategy.capture(model, task_loader, new_class_ids, task_id)
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
                full_metrics, full_time = self._run_full_retrain_baseline(all_seen_loaders, eval_loaders[: task_id + 1])
                delta_metrics = self._evaluate(model, eval_loaders[: task_id + 1])
                decision.compute_savings_percent = compute_compute_savings(delta_artifacts.wall_time_seconds, full_time)
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

            full_metrics, full_time = self._run_full_retrain_baseline(all_seen_loaders, eval_loaders[: task_id + 1])
            delta_metrics = self._evaluate(model, eval_loaders[: task_id + 1])
            delta_metrics["wall_time_seconds"] = delta_artifacts.wall_time_seconds
            decision.compute_savings_percent = compute_compute_savings(delta_artifacts.wall_time_seconds, full_time)
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
        backbone = BACKBONES[self.config.train.backbone](pretrained=getattr(self.config.train, "pretrained_backbone", False))
        classifier = IncrementalClassifier(backbone.out_dim)
        return MELDModel(backbone, classifier).to(self.device)

    def _build_tasks(self) -> tuple[list[DataLoader], list[DataLoader]]:
        dataset_bundle = self._load_dataset_bundle()
        train_tasks: list[DataLoader] = []
        eval_tasks: list[DataLoader] = []
        for train_subset, test_subset in dataset_bundle:
            train_tasks.append(
                DataLoader(
                    train_subset,
                    batch_size=self.config.train.batch_size,
                    shuffle=True,
                    num_workers=self.config.train.num_workers,
                )
            )
            eval_tasks.append(
                DataLoader(
                    test_subset,
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

            dataset_name = self.config.dataset.upper().replace("-", "")
            if dataset_name == "CIFAR10":
                train_dataset = CIFAR10(data_path=str(self.config.data_root), train=True, download=True)
                test_dataset = CIFAR10(data_path=str(self.config.data_root), train=False, download=True)
            elif dataset_name == "CIFAR100":
                train_dataset = CIFAR100(data_path=str(self.config.data_root), train=True, download=True)
                test_dataset = CIFAR100(data_path=str(self.config.data_root), train=False, download=True)
            else:
                raise ValueError(f"Unsupported dataset for Continuum loader: {self.config.dataset}")

            scenario_train = ClassIncremental(train_dataset, increment=self.config.classes_per_task, transformations=None)
            scenario_test = ClassIncremental(test_dataset, increment=self.config.classes_per_task, transformations=None)
            bundle = []
            for index in range(min(len(scenario_train), self.config.num_tasks)):
                bundle.append(
                    (
                        _TaskDatasetAdapter(scenario_train[index], normalize=False),
                        _TaskDatasetAdapter(scenario_test[index], normalize=False),
                    )
                )
            return bundle
        except Exception:
            return self._synthetic_bundle()

    def _synthetic_bundle(self) -> list[tuple[Dataset[Any], Dataset[Any]]]:
        torch.manual_seed(self.config.seed)
        num_classes = self.config.num_tasks * self.config.classes_per_task
        samples_per_class = 32
        test_per_class = 16
        tasks = []
        image_shape = (3, 32, 32)
        for task_id in range(self.config.num_tasks):
            train_tensors = []
            train_targets = []
            test_tensors = []
            test_targets = []
            for offset in range(self.config.classes_per_task):
                class_id = task_id * self.config.classes_per_task + offset
                base = torch.full(image_shape, float(class_id) / max(1, num_classes))
                train_tensors.append(base + 0.05 * torch.randn(samples_per_class, *image_shape))
                train_targets.append(torch.full((samples_per_class,), class_id, dtype=torch.long))
                test_tensors.append(base + 0.05 * torch.randn(test_per_class, *image_shape))
                test_targets.append(torch.full((test_per_class,), class_id, dtype=torch.long))
            train_x = torch.cat(train_tensors, dim=0)
            train_y = torch.cat(train_targets, dim=0)
            test_x = torch.cat(test_tensors, dim=0)
            test_y = torch.cat(test_targets, dim=0)
            tasks.append((TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)))
        return tasks

    def _run_full_retrain_baseline(
        self,
        train_loaders: list[DataLoader],
        eval_loaders: list[DataLoader],
    ) -> tuple[dict[str, Any], float]:
        torch.manual_seed(int(self.config.seed) + len(train_loaders))
        use_cached = self.config.full_retrain_interval > 1 and self._baseline_model_cache is not None and (len(train_loaders) % self.config.full_retrain_interval != 0)
        if use_cached:
            baseline_model = self._baseline_model_cache.clone().to(self.device)
            latest_train = train_loaders[-1]
            required_classes = len(train_loaders) * self.config.classes_per_task
            missing = required_classes - baseline_model.classifier.num_classes
            if missing > 0:
                baseline_model.classifier.adaption(missing)
            train_loader = latest_train
        else:
            baseline_model = self._build_model()
            baseline_model.classifier.adaption(len(train_loaders) * self.config.classes_per_task)
            train_loader = self._merged_loader(train_loaders, shuffle=True)
        start = time.time()
        baseline_epochs = self.config.train.full_retrain_epochs or self.config.train.epochs
        baseline_config = asdict(self.config.train)
        baseline_config["epochs"] = baseline_epochs
        baseline_model, _ = self.updater.update(baseline_model, train_loader, None, _ConfigView(baseline_config))
        self._baseline_model_cache = baseline_model.clone()
        metrics = self._evaluate(baseline_model, eval_loaders)
        wall_time = time.time() - start
        metrics["wall_time_seconds"] = wall_time
        return metrics, wall_time

    def _evaluate(self, model: MELDModel, eval_loaders: list[DataLoader]) -> dict[str, Any]:
        model.eval()
        logits_batches = []
        target_batches = []
        with torch.no_grad():
            for loader in eval_loaders:
                for inputs, targets in loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    logits_batches.append(model(inputs))
                    target_batches.append(targets)
        logits = torch.cat(logits_batches, dim=0)
        targets = torch.cat(target_batches, dim=0)
        metrics = compute_classification_metrics(logits, targets)
        metrics["wall_time_seconds"] = 0.0
        return metrics

    def _merged_loader(self, loaders: list[DataLoader], shuffle: bool = False) -> DataLoader:
        tensors_x = []
        tensors_y = []
        for loader in loaders:
            dataset = loader.dataset
            if isinstance(dataset, TensorDataset):
                x, y = dataset.tensors
                tensors_x.append(x)
                tensors_y.append(y)
            else:
                batch_x = []
                batch_y = []
                for inputs, targets in loader:
                    batch_x.append(inputs)
                    batch_y.append(targets)
                tensors_x.append(torch.cat(batch_x, dim=0))
                tensors_y.append(torch.cat(batch_y, dim=0))
        dataset = TensorDataset(torch.cat(tensors_x, dim=0), torch.cat(tensors_y, dim=0))
        return DataLoader(
            dataset,
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
        delta_confusion = np.asarray(delta_metrics["confusion_matrix"])
        full_confusion = np.asarray(full_metrics["confusion_matrix"])
        return {
            "task_id": task_id,
            "delta": {
                "top1": delta_metrics["top1"],
                "top5": delta_metrics["top5"],
                "ece": delta_metrics["ece"],
                "per_class_acc": delta_metrics["per_class_accuracy"],
                "wall_time_seconds": delta_metrics.get("wall_time_seconds", 0.0),
            },
            "full_retrain": {
                "top1": full_metrics["top1"],
                "top5": full_metrics["top5"],
                "ece": full_metrics["ece"],
                "per_class_acc": full_metrics["per_class_accuracy"],
                "wall_time_seconds": full_metrics.get("wall_time_seconds", 0.0),
            },
            "snapshot": {
                "fisher_eigenvalue_max": snapshot.fisher_eigenvalue_max if snapshot is not None else 0.0,
                "class_ids": snapshot.class_ids if snapshot is not None else [],
            },
            "oracle": {
                "pre_bound": pre_bound,
                "post_bound": post_bound,
                "bound_held": decision.bound_held if task_id > 0 else True,
            },
            "drift": {
                "shift_score": drift_result.shift_score,
                "shift_detected": drift_result.shift_detected,
                "severity": drift_result.severity,
                "per_class_drift": drift_result.per_class_drift,
            },
            "decision": asdict(decision),
            "equivalence_gap": compute_equivalence_gap(delta_confusion, full_confusion),
            "forgetting": max(0.0, full_metrics["top1"] - delta_metrics["top1"]),
            "compute_savings_percent": decision.compute_savings_percent,
        }

    def _summarize(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        if not tasks:
            return {}
        return {
            "mean_delta_top1": float(np.mean([task["delta"]["top1"] for task in tasks])),
            "mean_full_retrain_top1": float(np.mean([task["full_retrain"]["top1"] for task in tasks])),
            "mean_equivalence_gap": float(np.mean([task["equivalence_gap"] for task in tasks])),
            "mean_compute_savings": float(np.mean([task["compute_savings_percent"] for task in tasks])),
            "decisions": [task["decision"]["state"] for task in tasks],
            "total_wall_time_delta": float(np.sum([task["delta"]["wall_time_seconds"] for task in tasks])),
            "total_wall_time_full_retrain": float(np.sum([task["full_retrain"]["wall_time_seconds"] for task in tasks])),
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
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp_path, path)


class _PolicyConfigProxy:
    def __init__(self, shift_threshold: float, delta_wall_time_seconds: float, full_retrain_wall_time_seconds: float) -> None:
        self.shift_threshold = shift_threshold
        self.delta_wall_time_seconds = delta_wall_time_seconds
        self.full_retrain_wall_time_seconds = full_retrain_wall_time_seconds


class _ConfigView:
    def __init__(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(self, key, value)


class _TaskDatasetAdapter(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], normalize: bool = False) -> None:
        self.dataset = dataset
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.dataset[index]
        if isinstance(sample, tuple):
            inputs = sample[0]
            target = sample[1]
        else:
            raise TypeError("Expected dataset sample to be a tuple.")
        tensor = self._to_tensor(inputs)
        if self.normalize:
            mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=tensor.dtype).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor, int(target)

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.float()
            if tensor.max().item() > 2.0:
                tensor = tensor / 255.0
            return tensor
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 3 and array.shape[-1] in {1, 3}:
            array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array).float()
        if tensor.max().item() > 2.0:
            tensor = tensor / 255.0
        return tensor
