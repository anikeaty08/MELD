"""Optional Avalanche baseline integrations for MELD benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.backbone import resnet32
from .metrics import compute_classification_metrics


@dataclass(slots=True)
class _BaselineConfig:
    name: str
    status: str
    top1: float | None
    ece: float | None
    forgetting: float | None
    wall_time_seconds: float
    reason: str | None = None


class _AvalancheResNet(nn.Module):
    """Simple fixed-class classifier wrapper for Avalanche strategies."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = resnet32(pretrained=False)
        self.classifier = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def run_avalanche_baselines(config: Any, device: torch.device) -> dict[str, Any]:
    try:
        from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100
        from avalanche.training.supervised import DER, EWC, LwF, Naive
        from avalanche.training.plugins import ReplayPlugin
        from torchvision import transforms
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"Avalanche baselines unavailable: {exc}",
            "results": {
                name: _serialize(_BaselineConfig(name, "skipped", None, None, None, 0.0, reason="dependency_unavailable"))
                for name in ("naive", "ewc", "lwf", "derpp")
            },
        }

    try:
        benchmark = _build_benchmark(config, SplitCIFAR10, SplitCIFAR100, transforms)
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"Failed to build Avalanche benchmark: {exc}",
            "results": {
                name: _serialize(_BaselineConfig(name, "skipped", None, None, None, 0.0, reason="benchmark_build_failed"))
                for name in ("naive", "ewc", "lwf", "derpp")
            },
        }

    num_classes = int(config.num_tasks * config.classes_per_task)
    results: dict[str, Any] = {}
    strategies = {
        "naive": lambda model, optimizer, criterion: Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=int(config.train.batch_size),
            train_epochs=int(config.train.epochs),
            eval_mb_size=int(config.train.batch_size),
            device=device,
        ),
        "ewc": lambda model, optimizer, criterion: EWC(
            model,
            optimizer,
            criterion,
            ewc_lambda=float(getattr(config.train, "lambda_ewc", 0.3)),
            mode="separate",
            train_mb_size=int(config.train.batch_size),
            train_epochs=int(config.train.epochs),
            eval_mb_size=int(config.train.batch_size),
            device=device,
        ),
        "lwf": lambda model, optimizer, criterion: LwF(
            model,
            optimizer,
            criterion,
            alpha=float(getattr(config.train, "lambda_kd", 1.0)),
            temperature=float(getattr(config.train, "kd_temperature", 2.0)),
            train_mb_size=int(config.train.batch_size),
            train_epochs=int(config.train.epochs),
            eval_mb_size=int(config.train.batch_size),
            device=device,
        ),
        "derpp": lambda model, optimizer, criterion: DER(
            model,
            optimizer,
            criterion,
            alpha=0.1,
            beta=0.5,
            train_mb_size=int(config.train.batch_size),
            train_epochs=int(config.train.epochs),
            eval_mb_size=int(config.train.batch_size),
            device=device,
            plugins=[ReplayPlugin(mem_size=max(200, num_classes * 20))],
        ),
    }

    for name, factory in strategies.items():
        try:
            model = _AvalancheResNet(num_classes=num_classes).to(device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=float(config.train.lr),
                momentum=float(getattr(config.train, "momentum", 0.9)),
                weight_decay=float(getattr(config.train, "weight_decay", 5e-4)),
            )
            criterion = nn.CrossEntropyLoss()
            strategy = factory(model, optimizer, criterion)

            start = time.time()
            best_per_task: list[float] = []
            mean_forgetting = 0.0
            top1 = 0.0
            ece = 0.0

            for task_id, experience in enumerate(benchmark.train_stream[: int(config.num_tasks)]):
                strategy.train(experience, num_workers=int(getattr(config.train, "num_workers", 0)))
                eval_experiences = benchmark.test_stream[: task_id + 1]
                current_per_task: list[float] = []
                for eval_task_id, eval_experience in enumerate(eval_experiences):
                    task_logits, task_targets = _collect_logits(
                        strategy.model,
                        [eval_experience],
                        int(config.train.batch_size),
                        device,
                    )
                    task_metrics = compute_classification_metrics(task_logits, task_targets)
                    task_top1 = float(task_metrics["top1"])
                    current_per_task.append(task_top1)
                    if eval_task_id >= len(best_per_task):
                        best_per_task.append(task_top1)
                    else:
                        best_per_task[eval_task_id] = max(best_per_task[eval_task_id], task_top1)

                forgetting_values = [
                    max(0.0, best_per_task[k] - current_per_task[k])
                    for k in range(task_id)
                ]
                mean_forgetting = (
                    float(sum(forgetting_values) / len(forgetting_values))
                    if forgetting_values
                    else 0.0
                )
                logits, targets = _collect_logits(strategy.model, eval_experiences, int(config.train.batch_size), device)
                metrics = compute_classification_metrics(logits, targets)
                top1 = float(metrics["top1"])
                ece = float(metrics["ece"])

            results[name] = _serialize(
                _BaselineConfig(
                    name=name,
                    status="completed",
                    top1=top1,
                    ece=ece,
                    forgetting=mean_forgetting,
                    wall_time_seconds=time.time() - start,
                )
            )
        except Exception as exc:
            results[name] = _serialize(
                _BaselineConfig(
                    name=name,
                    status="skipped",
                    top1=None,
                    ece=None,
                    forgetting=None,
                    wall_time_seconds=0.0,
                    reason=str(exc),
                )
            )

    return {
        "status": "completed",
        "results": results,
    }


def _build_benchmark(config: Any, split_cifar10: Any, split_cifar100: Any, transforms: Any) -> Any:
    dataset = str(config.dataset).upper().replace("-", "")
    if dataset == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset == "CIFAR100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unsupported Avalanche dataset: {config.dataset}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    common = {
        "n_experiences": int(config.num_tasks),
        "return_task_id": True,
        "seed": int(config.seed),
        "train_transform": transform,
        "eval_transform": transform,
        "dataset_root": str(Path(config.data_root)),
    }
    if dataset == "CIFAR10":
        return split_cifar10(**common)
    if dataset == "CIFAR100":
        return split_cifar100(**common)
    raise AssertionError("unreachable")


def _collect_logits(
    model: nn.Module,
    experiences: list[Any],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    with torch.no_grad():
        for experience in experiences:
            loader = DataLoader(experience.dataset, batch_size=batch_size, shuffle=False)
            for batch in loader:
                if len(batch) < 2:
                    continue
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                logits_all.append(model(inputs))
                targets_all.append(targets)
    if not logits_all:
        return torch.empty((0, 1), device=device), torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat(logits_all, dim=0), torch.cat(targets_all, dim=0)


def _serialize(baseline: _BaselineConfig) -> dict[str, Any]:
    return {
        "status": baseline.status,
        "top1": baseline.top1,
        "ece": baseline.ece,
        "forgetting": baseline.forgetting,
        "wall_time_seconds": baseline.wall_time_seconds,
        "reason": baseline.reason,
    }
