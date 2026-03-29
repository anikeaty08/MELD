"""CIFAR-10 equivalence benchmark — competition proof script.

Demonstrates all 8 PS requirements on real or synthetic data.
No meld imports. Uses delta/ framework throughout.

Usage:
    python -m delta.demos.cifar10_benchmark --quick
    python -m delta.demos.cifar10_benchmark --data-root ./data --epochs 10
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from delta import (
    DeltaStream,
    FisherDeltaStrategy,
    FullRetrainStrategy,
    EvaluationPlugin,
    accuracy_metrics,
    equivalence_metrics,
    calibration_metrics,
    compute_metrics,
    InteractiveLogger,
    CSVLogger,
)
from .models import resnet20, IncrementalClassifier, MELDModel


def _build_model(n_tasks: int, classes_per_task: int, use_cnn: bool = True) -> nn.Module:
    if use_cnn:
        backbone = resnet20()
        classifier = IncrementalClassifier(backbone.out_dim)
        return MELDModel(backbone, classifier)
    else:
        n_classes = n_tasks * classes_per_task
        return nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    n_tasks = 5
    classes_per_task = 2
    dataset = "CIFAR-10"

    print(f"\n=== Delta Framework CIFAR-10 Benchmark ===")
    print(f"    Tasks: {n_tasks}, Classes/task: {classes_per_task}")
    print(f"    Epochs: {args.epochs}, Quick: {args.quick}\n")

    stream = DeltaStream(
        dataset_name=dataset,
        n_tasks=n_tasks,
        scenario="class_incremental",
        classes_per_task=classes_per_task,
        data_root=args.data_root,
    )

    # Detect if real dataset loaded or fell back to synthetic flat data
    sample_x = stream.train_stream[0].train_dataset[0][0]
    use_cnn = sample_x.dim() >= 3  # True for images (C,H,W), False for flat vectors

    if not use_cnn:
        print("  (Real CIFAR-10 not available — using synthetic flat data with MLP)\n")

    # --- FisherDelta ---
    delta_model = _build_model(n_tasks, classes_per_task, use_cnn=use_cnn)
    delta_opt = SGD(delta_model.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=5e-4, nesterov=True)

    csv_path = args.results_path.replace(".json", ".csv") if args.results_path else None
    loggers = [InteractiveLogger()]
    if csv_path:
        loggers.append(CSVLogger(csv_path))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
        calibration_metrics(ece_before=True, ece_after=True),
        compute_metrics(savings_ratio=True),
        loggers=loggers,
    )

    delta_strategy = FisherDeltaStrategy(
        delta_model, delta_opt, nn.CrossEntropyLoss(),
        evaluator=eval_plugin,
        train_epochs=args.epochs,
        train_mb_size=args.batch_size,
    )

    # --- FullRetrain ---
    full_model = _build_model(n_tasks, classes_per_task, use_cnn=use_cnn)
    full_opt = SGD(full_model.parameters(), lr=args.lr,
                   momentum=0.9, weight_decay=5e-4, nesterov=True)
    full_strategy = FullRetrainStrategy(
        full_model, full_opt, nn.CrossEntropyLoss(),
        train_epochs=args.epochs,
        train_mb_size=args.batch_size,
    )

    # --- Run FisherDelta ---
    print("=== Running FisherDelta Strategy ===")
    delta_task_data: list[dict[str, Any]] = []
    total_delta_time = 0.0
    for exp in stream.train_stream:
        print(f"\n--- Task {exp.task_id} ---")
        t0 = time.time()
        delta_strategy.train(exp)
        dt = time.time() - t0
        total_delta_time += dt
        delta_strategy.eval(stream.test_stream)
        metrics = eval_plugin.get_last_metrics()
        cert = delta_strategy.last_certificate
        delta_task_data.append({
            "task_id": exp.task_id,
            "wall_time": dt,
            "accuracy_stream": metrics.get("accuracy/stream", 0),
            "epsilon_param": cert.epsilon_param if cert else None,
            "kl_bound": cert.kl_bound if cert else None,
            "kl_bound_normalized": cert.kl_bound_normalized if cert else None,
            "is_equivalent": cert.is_equivalent if cert else None,
            "ece_before": cert.ece_before if cert else None,
            "ece_after": cert.ece_after if cert else None,
            "ece_delta": cert.ece_delta if cert else None,
            "shift_type": cert.shift_type if cert else None,
            "ce_scale": cert.ce_scale if cert else None,
            "ewc_scale": cert.ewc_scale if cert else None,
            "compute_ratio": cert.compute_ratio if cert else None,
        })

    # --- Run FullRetrain ---
    print("\n=== Running FullRetrain Baseline ===")
    stream2 = DeltaStream(
        dataset_name=dataset,
        n_tasks=n_tasks,
        scenario="class_incremental",
        classes_per_task=classes_per_task,
        data_root=args.data_root,
    )
    total_full_time = 0.0
    full_acc = 0.0
    for exp in stream2.train_stream:
        t0 = time.time()
        full_strategy.train(exp)
        total_full_time += time.time() - t0
    full_results = full_strategy.eval(stream2.test_stream)
    full_acc = full_results.get("accuracy/stream", 0)

    # --- Robustness ---
    delta_rob = None
    full_rob = None
    robustness_gap = None
    try:
        from .robustness import evaluate_cifar_c
        delta_rob_result = evaluate_cifar_c(
            delta_model, dataset=dataset,
            data_root=Path(args.data_root),
            device=next(delta_model.parameters()).device)
        delta_rob = delta_rob_result.get("mean_top1")
        full_rob_result = evaluate_cifar_c(
            full_model, dataset=dataset,
            data_root=Path(args.data_root),
            device=next(full_model.parameters()).device)
        full_rob = full_rob_result.get("mean_top1")
        if delta_rob is not None and full_rob is not None:
            robustness_gap = full_rob - delta_rob
    except Exception:
        pass

    # --- Aggregate ---
    final_delta_acc = delta_task_data[-1]["accuracy_stream"] if delta_task_data else 0
    epsilons = [t["epsilon_param"] for t in delta_task_data if t["epsilon_param"] is not None and t["epsilon_param"] != 0]
    ece_befores = [t["ece_before"] for t in delta_task_data if t["ece_before"] is not None]
    ece_afters = [t["ece_after"] for t in delta_task_data if t["ece_after"] is not None]
    ece_deltas = [t["ece_delta"] for t in delta_task_data if t["ece_delta"] is not None]
    ce_scales = [t["ce_scale"] for t in delta_task_data if t["ce_scale"] is not None]
    ewc_scales = [t["ewc_scale"] for t in delta_task_data if t["ewc_scale"] is not None]
    equiv_count = sum(1 for t in delta_task_data if t.get("is_equivalent"))

    speedup_wall = total_full_time / max(total_delta_time, 1e-6)
    mean_ece_delta = float(np.mean(ece_deltas)) if ece_deltas else float("nan")
    cal_preserved = abs(mean_ece_delta) < 0.05 if ece_deltas else False

    # --- Print comparison table ---
    def fmt(v: Any, pct: bool = False, dec: int = 4) -> str:
        if v is None:
            return "--"
        if isinstance(v, bool):
            return "YES" if v else "NO"
        if pct:
            return f"{v*100:.1f}%"
        if isinstance(v, float):
            if abs(v) >= 10:
                return f"{v:.1f}x"
            return f"{v:.{dec}f}"
        return str(v)

    rows = [
        ("Final Accuracy", fmt(final_delta_acc, pct=True), fmt(full_acc, pct=True)),
        ("Accuracy Gap", fmt(abs(final_delta_acc - full_acc) if full_acc else None, pct=True), "baseline"),
        ("ECE Before (avg)", fmt(float(np.mean(ece_befores)) if ece_befores else None), "--"),
        ("ECE After (avg)", fmt(float(np.mean(ece_afters)) if ece_afters else None), "--"),
        ("ECE Delta (avg)", fmt(mean_ece_delta), "--"),
        ("Calibration Preserved", fmt(cal_preserved), "--"),
        ("Epsilon Bound (avg)", fmt(float(np.mean(epsilons)) if epsilons else None), "--"),
        ("Is Equivalent (tasks)", f"{equiv_count}/{n_tasks}", "--"),
        ("Robustness Score", fmt(delta_rob, pct=True), fmt(full_rob, pct=True)),
        ("Robustness Gap", fmt(robustness_gap, pct=True) if robustness_gap is not None else "--", "baseline"),
        ("Wall Time Savings", fmt(speedup_wall), "1.0x"),
        ("ce_scale range", f"{min(ce_scales):.2f}-{max(ce_scales):.2f}" if ce_scales else "--", "--"),
        ("ewc_scale range", f"{min(ewc_scales):.4f}-{max(ewc_scales):.4f}" if ewc_scales else "--", "--"),
    ]

    c1 = max(len(r[0]) for r in rows) + 2
    c2 = 14
    c3 = 14
    sep = "+" + "-" * (c1 + c2 + c3 + 6) + "+"
    print(f"\n{sep}")
    print(f"| {'Metric':<{c1}} | {'FisherDelta':>{c2}} | {'Full Retrain':>{c3}} |")
    print(sep)
    for name, dv, fv in rows:
        print(f"| {name:<{c1}} | {dv:>{c2}} | {fv:>{c3}} |")
    print(sep)
    print()

    # --- Save JSON ---
    output = {
        "delta_tasks": delta_task_data,
        "full_retrain_accuracy": full_acc,
        "total_delta_wall_time": total_delta_time,
        "total_full_retrain_wall_time": total_full_time,
        "speedup_ratio": speedup_wall,
        "mean_epsilon": float(np.mean(epsilons)) if epsilons else None,
        "mean_ece_delta": mean_ece_delta,
        "calibration_preserved": cal_preserved,
        "delta_robustness": delta_rob,
        "full_robustness": full_rob,
        "robustness_gap": robustness_gap,
        "robustness_equivalent": bool(abs(robustness_gap) < 2.0) if robustness_gap is not None else None,
    }
    if args.results_path:
        Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results saved to {args.results_path}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 Equivalence Benchmark")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--results-path", default="results_cifar10.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true",
                        help="1 epoch smoke test")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 1
        args.batch_size = 8
    run_benchmark(args)


if __name__ == "__main__":
    main()
