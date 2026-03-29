"""
Test 9 — Full analysis: run all available datasets, collect results, print report
Runs whatever datasets are available. Skips missing deps gracefully.
Run: python test_runs/test_9_analysis.py
"""

import time
import torch
import torch.nn as nn
from torch.optim import SGD
from delta import (
    DeltaStream, FisherDeltaStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)

DATASETS = [
    # (name, n_tasks, cpt, model_fn, epochs, batch)
    ("synthetic", 3, 2, lambda: nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 6)), 3, 16),
    ("MNIST", 5, 2, lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)), 2, 64),
    ("FashionMNIST", 5, 2, lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)), 2, 64),
    ("CIFAR-10", 5, 2, lambda: nn.Sequential(
        nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, 10)), 2, 64),
    ("SVHN", 5, 2, lambda: nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        nn.Flatten(), nn.Linear(32*16, 10)), 2, 64),
]

report = []

for ds_name, n_tasks, cpt, model_fn, epochs, batch in DATASETS:
    print(f"\n{'='*60}")
    print(f"  Running: {ds_name}")
    print(f"{'='*60}")

    try:
        model = model_fn()
        strategy = FisherDeltaStrategy(
            model, SGD(model.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
            evaluator=EvaluationPlugin(
                accuracy_metrics(stream=True),
                equivalence_metrics(epsilon=True),
                calibration_metrics(),
                compute_metrics(),
                loggers=[InteractiveLogger(verbose=False)],
            ),
            train_epochs=epochs, train_mb_size=batch,
        )

        stream = DeltaStream(ds_name, n_tasks=n_tasks, classes_per_task=cpt, data_root="./data")

        t0 = time.time()
        for exp in stream.train_stream:
            strategy.train(exp)
            strategy.eval(stream.test_stream)
        wall = time.time() - t0

        cert = strategy.last_certificate
        metrics = strategy.evaluator.get_last_metrics() if strategy.evaluator else {}

        report.append({
            "dataset": ds_name,
            "status": "PASS",
            "accuracy": metrics.get("accuracy/stream", 0),
            "epsilon": cert.epsilon_param,
            "kl_bound": cert.kl_bound,
            "equivalent": cert.is_equivalent,
            "ece_delta": cert.ece_delta,
            "speedup": cert.compute_ratio,
            "shift": cert.shift_type,
            "wall_time": wall,
            "tasks": n_tasks,
        })
        print(f"  -> PASS ({wall:.1f}s)")

    except Exception as e:
        report.append({
            "dataset": ds_name,
            "status": f"SKIP: {type(e).__name__}",
            "accuracy": None, "epsilon": None, "kl_bound": None,
            "equivalent": None, "ece_delta": None, "speedup": None,
            "shift": None, "wall_time": None, "tasks": n_tasks,
        })
        print(f"  -> SKIPPED: {e}")

# ---- Final Report ----
print(f"\n\n{'='*100}")
print(f"  DELTA FRAMEWORK — CROSS-DATASET ANALYSIS REPORT")
print(f"{'='*100}")
print(f"  {'Dataset':<15} | {'Status':<8} | {'Acc':>6} | {'Epsilon':>10} | {'KL':>10} | {'ECE d':>8} | {'Speed':>6} | {'Shift':<10} | {'Time':>6}")
print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}")

for r in report:
    if r["accuracy"] is not None:
        print(f"  {r['dataset']:<15} | {'PASS':<8} | {r['accuracy']*100:>5.1f}% | {r['epsilon']:>10.6f} | {r['kl_bound']:>10.4f} | {r['ece_delta']:>+7.4f} | {r['speedup']:>5.1f}x | {r['shift']:<10} | {r['wall_time']:>5.1f}s")
    else:
        print(f"  {r['dataset']:<15} | {'SKIP':<8} | {'--':>6} | {'--':>10} | {'--':>10} | {'--':>8} | {'--':>6} | {'--':<10} | {'--':>6}")

passed = sum(1 for r in report if r["status"] == "PASS")
total = len(report)
print(f"{'='*100}")
print(f"  RESULT: {passed}/{total} datasets passed")
print(f"{'='*100}")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_9_analysis", "passed": passed, "total": total, "report": report,
}, open(out / "test_9_analysis.json", "w"), indent=2, default=str)
print(f"  Saved: {out / 'test_9_analysis.json'}")
