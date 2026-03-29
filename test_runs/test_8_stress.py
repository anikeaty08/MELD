"""
Test 8 — Stress test: many tasks, large vs small data, epsilon tracking
No extra deps needed. Just torch + numpy.
Run: python test_runs/test_8_stress.py
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset
from delta import (
    Experience, FisherDeltaStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    InteractiveLogger,
)

print("=" * 60)
print("  TEST 8: Stress — 10 tasks, tracking all metrics")
print("  Proves: epsilon stable, ce_scale grows, speedup increases")
print("=" * 60)

model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 20),
)

strategy = FisherDeltaStrategy(
    model, SGD(model.parameters(), lr=0.005, momentum=0.9), nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True),
        loggers=[InteractiveLogger(verbose=False)],
    ),
    train_epochs=3, train_mb_size=32,
)

torch.manual_seed(42)
n_tasks = 10
classes_per_task = 2
all_exps = []

# Task 0 gets 500 samples, rest get 50 each
for tid in range(n_tasks):
    n_train = 500 if tid == 0 else 50
    n_test = 50
    classes = list(range(tid * classes_per_task, (tid + 1) * classes_per_task))
    tx, ty, ex, ey = [], [], [], []
    for c in classes:
        center = torch.randn(64) * (c + 1)
        tx.append(center + torch.randn(n_train // classes_per_task, 64) * 0.5)
        ty.append(torch.full((n_train // classes_per_task,), c, dtype=torch.long))
        ex.append(center + torch.randn(n_test // classes_per_task, 64) * 0.5)
        ey.append(torch.full((n_test // classes_per_task,), c, dtype=torch.long))
    all_exps.append(Experience(
        train_dataset=TensorDataset(torch.cat(tx), torch.cat(ty)),
        test_dataset=TensorDataset(torch.cat(ex), torch.cat(ey)),
        task_id=tid, classes_in_this_experience=classes,
    ))

# ---- Run ----
results = []
for exp in all_exps:
    strategy.train(exp)
    strategy.eval(all_exps)
    cert = strategy.last_certificate
    results.append({
        "task": exp.task_id,
        "epsilon": cert.epsilon_param,
        "kl_bound": cert.kl_bound,
        "ce_scale": cert.ce_scale,
        "ewc_scale": cert.ewc_scale,
        "speedup": cert.compute_ratio,
        "equivalent": cert.is_equivalent,
        "shift": cert.shift_type,
    })

# ---- Analysis ----
print(f"\n{'='*80}")
print(f"  {'Task':>4} | {'Epsilon':>10} | {'KL Bound':>10} | {'ce_scale':>10} | {'ewc_scale':>10} | {'Speedup':>8} | {'Equiv':>6} | Shift")
print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+------")
for r in results:
    print(f"  {r['task']:>4} | {r['epsilon']:>10.6f} | {r['kl_bound']:>10.6f} | {r['ce_scale']:>10.4f} | {r['ewc_scale']:>10.4f} | {r['speedup']:>7.1f}x | {'YES' if r['equivalent'] else 'NO':>6} | {r['shift']}")
print(f"{'='*80}")

epsilons = [r["epsilon"] for r in results]
ce_scales = [r["ce_scale"] for r in results]
speedups = [r["speedup"] for r in results]

all_finite = all(e < float("inf") for e in epsilons)
ce_growing = all(ce_scales[i] <= ce_scales[i+1] for i in range(1, len(ce_scales)-1))
speedup_growing = speedups[-1] > speedups[1] if len(speedups) > 2 else True

print(f"\n  All epsilon finite:     {all_finite}")
print(f"  ce_scale grows:         {ce_growing} ({ce_scales[0]:.1f} -> {ce_scales[-1]:.1f})")
print(f"  Speedup grows:          {speedup_growing} ({speedups[0]:.1f}x -> {speedups[-1]:.1f}x)")
passed = all_finite and ce_growing
print(f"  TEST 8 {'PASSED' if passed else 'FAILED'}")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_8_stress", "status": "PASS" if passed else "FAIL",
    "tasks": results, "all_epsilon_finite": all_finite,
    "ce_scale_growing": ce_growing, "speedup_growing": speedup_growing,
    "ce_scale_range": [ce_scales[0], ce_scales[-1]],
    "speedup_range": [speedups[0], speedups[-1]],
}, open(out / "test_8_stress.json", "w"), indent=2)
print(f"  Saved: {out / 'test_8_stress.json'}")
