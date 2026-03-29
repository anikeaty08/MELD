"""Slim benchmark runner for the delta framework.

Replaces meld/benchmarks/runner.py. Uses delta/ framework end-to-end.
No meld imports anywhere.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW

from delta import (
    DeltaStream,
    Experience,
    FisherDeltaStrategy,
    FullRetrainStrategy,
    ReplayDeltaStrategy,
    EvaluationPlugin,
    accuracy_metrics,
    equivalence_metrics,
    calibration_metrics,
    compute_metrics,
    InteractiveLogger,
    CSVLogger,
)
from .models import (
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet18_imagenet,
    IncrementalClassifier,
    MELDModel,
)
from .storage import ResultStore


BACKBONES = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet18_imagenet": resnet18_imagenet,
}


def _auto_backbone(dataset: str) -> str:
    d = dataset.upper().replace("-", "").replace("_", "")
    if d == "SYNTHETIC":
        return "resnet20"
    if d == "CIFAR10":
        return "resnet32"
    if d == "CIFAR100":
        return "resnet56"
    return "resnet32"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DemoRunner:
    """Lightweight runner using delta/ framework directly."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = _pick_device()

    def run(self, results_path: str | Path | None = None) -> dict[str, Any]:
        cfg = self.config
        _seed_everything(int(cfg.get("seed", 42)))

        run_mode = cfg.get("run_mode", "compare")
        stream = DeltaStream(
            dataset_name=cfg.get("dataset", "synthetic"),
            n_tasks=int(cfg.get("num_tasks", 2)),
            scenario=cfg.get("scenario", "class_incremental"),
            classes_per_task=int(cfg.get("classes_per_task", 2)),
            data_root=cfg.get("data_root", "./data"),
            seed=int(cfg.get("seed", 42)),
            batch_size=int(cfg.get("batch_size", 64)),
            preset=cfg.get("preset", "standard"),
            pretrained_backbone=bool(cfg.get("pretrained_backbone", False)),
            image_size=cfg.get("image_size"),
            use_imagenet_stats=cfg.get("use_imagenet_stats"),
        )

        results: dict[str, Any] = {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
            "status": "running",
            "config": cfg,
            "tasks": [],
            "final_summary": None,
        }

        delta_results = None
        full_results = None

        if run_mode in ("fisher_delta", "compare", "compare_fisher"):
            delta_results = self._run_strategy("fisher_delta", stream, cfg)
            results["delta_tasks"] = delta_results["tasks"]

        if run_mode in ("replay_delta", "compare_replay"):
            delta_results = self._run_strategy("replay_delta", stream, cfg)
            results["delta_tasks"] = delta_results["tasks"]

        if run_mode in ("full_retrain", "compare", "compare_fisher", "compare_replay"):
            stream2 = DeltaStream(
                dataset_name=cfg.get("dataset", "synthetic"),
                n_tasks=int(cfg.get("num_tasks", 2)),
                scenario=cfg.get("scenario", "class_incremental"),
                classes_per_task=int(cfg.get("classes_per_task", 2)),
                data_root=cfg.get("data_root", "./data"),
                seed=int(cfg.get("seed", 42)),
                batch_size=int(cfg.get("batch_size", 64)),
                preset=cfg.get("preset", "standard"),
                pretrained_backbone=bool(cfg.get("pretrained_backbone", False)),
                image_size=cfg.get("image_size"),
                use_imagenet_stats=cfg.get("use_imagenet_stats"),
            )
            full_results = self._run_strategy("full_retrain", stream2, cfg)
            results["full_retrain_tasks"] = full_results["tasks"]

        results["status"] = "completed"
        results["final_summary"] = self._summarize(
            delta_results, full_results, run_mode)
        self._write_results(results_path, results)
        return results

    def _run_strategy(
        self, mode: str, stream: DeltaStream, cfg: dict[str, Any]
    ) -> dict[str, Any]:
        backbone_name = cfg.get("backbone", "auto")
        if backbone_name == "auto":
            backbone_name = _auto_backbone(cfg.get("dataset", "synthetic"))
        n_classes = int(cfg.get("num_tasks", 2)) * int(cfg.get("classes_per_task", 2))

        ds_name = cfg.get("dataset", "synthetic").upper().replace("-", "").replace("_", "")
        if ds_name == "SYNTHETIC":
            # Synthetic produces flat vectors — use simple MLP
            feature_dim = 32
            n_out = n_classes
            model = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, n_out),
            )
        else:
            backbone_fn = BACKBONES.get(backbone_name, resnet20)
            pretrained_backbone = bool(cfg.get("pretrained_backbone", False))
            backbone = backbone_fn(pretrained=pretrained_backbone)
            classifier = IncrementalClassifier(backbone.out_dim)
            model = MELDModel(backbone, classifier)

        raw_lr = float(cfg.get("lr", 0.01))
        text_datasets = {"AGNEWS", "DBPEDIA", "YAHOOANSWERSNLP", "IMDB", "SST2"}
        if ds_name in text_datasets:
            lr = 0.001 if raw_lr == 0.01 else raw_lr
            optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=1e-2,
            )
        else:
            optimizer = SGD(
                model.parameters(),
                lr=raw_lr,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True,
            )
        criterion = nn.CrossEntropyLoss()

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(experience=True, stream=True),
            equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
            calibration_metrics(ece_before=True, ece_after=True),
            compute_metrics(savings_ratio=True),
            loggers=[InteractiveLogger(verbose=bool(cfg.get("verbose", True)))],
        )

        epochs = int(cfg.get("epochs", 5))
        batch_size = int(cfg.get("batch_size", 64))

        if mode == "fisher_delta":
            strategy = FisherDeltaStrategy(
                model, optimizer, criterion,
                evaluator=eval_plugin,
                device=self.device,
                train_epochs=epochs,
                train_mb_size=batch_size,
            )
        elif mode == "replay_delta":
            strategy = ReplayDeltaStrategy(
                model, optimizer, criterion,
                evaluator=eval_plugin,
                device=self.device,
                train_epochs=epochs,
                train_mb_size=batch_size,
            )
            strategy.replay_memory_per_class = int(
                cfg.get("replay_memory_per_class", strategy.replay_memory_per_class)
            )
            strategy.replay_batch_size = int(
                cfg.get("replay_batch_size", strategy.replay_batch_size)
            )
            strategy.use_task_identity_inference = bool(
                cfg.get("task_identity_inference", False)
                or cfg.get("scenario", "class_incremental") == "task_incremental"
            )
        else:
            strategy = FullRetrainStrategy(
                model, optimizer, criterion,
                evaluator=eval_plugin,
                device=self.device,
                train_epochs=epochs,
                train_mb_size=batch_size,
            )

        tasks_data = []
        for exp in stream.train_stream:
            t0 = time.time()
            strategy.train(exp)
            wall_time = time.time() - t0
            eval_results = strategy.eval(stream.test_stream)
            metrics = eval_plugin.get_last_metrics()

            task_entry: dict[str, Any] = {
                "task_id": exp.task_id,
                "wall_time_seconds": wall_time,
                "accuracy_stream": metrics.get("accuracy/stream"),
                **metrics,
            }

            cert = getattr(strategy, "last_certificate", None)
            if cert is not None:
                task_entry.update({
                    "epsilon_param": cert.epsilon_param,
                    "kl_bound": cert.kl_bound,
                    "kl_bound_normalized": cert.kl_bound_normalized,
                    "is_equivalent": cert.is_equivalent,
                    "shift_type": cert.shift_type,
                    "ece_before": cert.ece_before,
                    "ece_after": cert.ece_after,
                    "ece_delta": cert.ece_delta,
                    "compute_ratio": cert.compute_ratio,
                    "ce_scale": cert.ce_scale,
                    "ewc_scale": cert.ewc_scale,
                })

            tasks_data.append(task_entry)

        return {
            "tasks": tasks_data,
            "strategy": strategy,
            "model": model,
        }

    def _summarize(
        self,
        delta_res: dict[str, Any] | None,
        full_res: dict[str, Any] | None,
        run_mode: str,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {"run_mode": run_mode}

        if delta_res:
            tasks = delta_res["tasks"]
            accs = [t.get("accuracy_stream") for t in tasks if t.get("accuracy_stream") is not None]
            summary["mean_delta_accuracy"] = float(np.mean(accs)) if accs else None
            wall_times = [t.get("wall_time_seconds", 0) for t in tasks]
            summary["total_delta_wall_time"] = float(sum(wall_times))

            epsilons = [t.get("epsilon_param") for t in tasks if t.get("epsilon_param") is not None]
            summary["mean_epsilon_param"] = float(np.mean(epsilons)) if epsilons else None

            ece_deltas = [t.get("ece_delta") for t in tasks if t.get("ece_delta") is not None]
            summary["mean_ece_delta"] = float(np.mean(ece_deltas)) if ece_deltas else None

        if full_res:
            tasks = full_res["tasks"]
            accs = [t.get("accuracy_stream") for t in tasks if t.get("accuracy_stream") is not None]
            summary["mean_full_retrain_accuracy"] = float(np.mean(accs)) if accs else None
            wall_times = [t.get("wall_time_seconds", 0) for t in tasks]
            summary["total_full_retrain_wall_time"] = float(sum(wall_times))

        if delta_res and full_res:
            dt = summary.get("total_delta_wall_time", 0)
            ft = summary.get("total_full_retrain_wall_time", 0)
            summary["speedup_ratio"] = ft / max(dt, 1e-6)

        return summary

    def _write_results(self, path: str | Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serializable objects
        clean = self._make_serializable(payload)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)

    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()
                    if not isinstance(v, (torch.nn.Module, torch.optim.Optimizer))
                    and k not in ("strategy", "model")}
        if isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
