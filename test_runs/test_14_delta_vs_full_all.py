"""
Test 14 — Head-to-head: FisherDelta vs FullRetrain on every available dataset
Runs both strategies, prints accuracy comparison table.
Run: python test_runs/test_14_delta_vs_full_all.py
"""

import time
import torch
import torch.nn as nn
from torch.optim import SGD
from delta import (
    DeltaStream, FisherDeltaStrategy, FullRetrainStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)

BENCHMARKS = [
    ("synthetic", 3, 2,
     lambda nc: nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, nc)),
     3, 16),
    ("MNIST", 5, 2,
     lambda nc: nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, nc)),
     2, 64),
    ("FashionMNIST", 5, 2,
     lambda nc: nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, nc)),
     2, 64),
    ("CIFAR-10", 5, 2,
     lambda nc: nn.Sequential(
         nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
         nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
         nn.Flatten(), nn.Linear(32, nc)),
     2, 64),
]

results = []

for ds_name, n_tasks, cpt, model_fn, epochs, batch in BENCHMARKS:
    nc = n_tasks * cpt
    print(f"\n{'='*60}")
    print(f"  {ds_name}: delta vs full retrain ({n_tasks} tasks, {epochs} epochs)")
    print(f"{'='*60}")

    try:
        stream = DeltaStream(ds_name, n_tasks=n_tasks, classes_per_task=cpt, data_root="./data")

        # ---- Delta ----
        model_d = model_fn(nc)
        ep = EvaluationPlugin(
            accuracy_metrics(stream=True), equivalence_metrics(epsilon=True),
            loggers=[InteractiveLogger(verbose=False)],
        )
        delta_s = FisherDeltaStrategy(
            model_d, SGD(model_d.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
            evaluator=ep, train_epochs=epochs, train_mb_size=batch,
        )
        t0 = time.time()
        for exp in stream.train_stream:
            delta_s.train(exp)
            delta_s.eval(stream.test_stream)
        delta_time = time.time() - t0
        delta_acc = ep.get_last_metrics().get("accuracy/stream", 0)
        cert = delta_s.last_certificate

        # ---- Full retrain ----
        model_f = model_fn(nc)
        full_s = FullRetrainStrategy(
            model_f, SGD(model_f.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
            train_epochs=epochs, train_mb_size=batch,
        )
        stream2 = DeltaStream(ds_name, n_tasks=n_tasks, classes_per_task=cpt, data_root="./data")
        t0 = time.time()
        for exp in stream2.train_stream:
            full_s.train(exp)
        full_time = time.time() - t0
        full_results = full_s.eval(stream2.test_stream)
        full_acc = full_results.get("accuracy/stream", 0)

        results.append({
            "dataset": ds_name, "status": "OK",
            "delta_acc": delta_acc, "full_acc": full_acc,
            "gap": abs(delta_acc - full_acc),
            "epsilon": cert.epsilon_param,
            "delta_time": delta_time, "full_time": full_time,
            "speedup": full_time / max(delta_time, 1e-6),
        })
        print(f"  Delta: {delta_acc*100:.1f}%  Full: {full_acc*100:.1f}%  Gap: {abs(delta_acc-full_acc)*100:.1f}%  Speedup: {full_time/max(delta_time,1e-6):.1f}x")

    except Exception as e:
        results.append({"dataset": ds_name, "status": f"SKIP: {e}"})
        print(f"  SKIPPED: {e}")

# ---- Final Table ----
print(f"\n\n{'='*90}")
print(f"  DELTA vs FULL RETRAIN — HEAD TO HEAD")
print(f"{'='*90}")
print(f"  {'Dataset':<15} | {'Delta':>7} | {'Full':>7} | {'Gap':>6} | {'Epsilon':>10} | {'D time':>7} | {'F time':>7} | {'Speedup':>8}")
print(f"  {'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")
for r in results:
    if "delta_acc" in r:
        print(f"  {r['dataset']:<15} | {r['delta_acc']*100:>6.1f}% | {r['full_acc']*100:>6.1f}% | {r['gap']*100:>5.1f}% | {r['epsilon']:>10.4f} | {r['delta_time']:>6.1f}s | {r['full_time']:>6.1f}s | {r['speedup']:>7.1f}x")
    else:
        print(f"  {r['dataset']:<15} | {'--':>7} | {'--':>7} | {'--':>6} | {'--':>10} | {'--':>7} | {'--':>7} | {'--':>8}")
print(f"{'='*90}")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_14_delta_vs_full", "results": results,
}, open(out / "test_14_delta_vs_full.json", "w"), indent=2, default=str)
print(f"  Saved: {out / 'test_14_delta_vs_full.json'}")
