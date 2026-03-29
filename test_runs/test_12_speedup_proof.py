"""
Test 12 — Speedup proof: large task 0, tiny subsequent tasks
No extra deps needed. Just torch + numpy.
Run: python test_runs/test_12_speedup_proof.py
"""

import time
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset
from delta import (
    Experience, FisherDeltaStrategy, FullRetrainStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    compute_metrics, InteractiveLogger,
)

print("=" * 60)
print("  TEST 12: Speedup proof — 2000 initial, 100 per task")
print("  Proves: delta is genuinely faster when history >> new data")
print("=" * 60)

torch.manual_seed(42)
in_dim, n_classes = 64, 10

def make_exp(tid, n_train, n_test=100):
    tx = torch.randn(n_train, in_dim) + torch.randn(1, in_dim) * tid
    ty = torch.randint(0, n_classes, (n_train,))
    ex = torch.randn(n_test, in_dim)
    ey = torch.randint(0, n_classes, (n_test,))
    return Experience(
        train_dataset=TensorDataset(tx, ty),
        test_dataset=TensorDataset(ex, ey),
        task_id=tid, classes_in_this_experience=list(range(n_classes)),
    )

exps = [make_exp(0, 2000)] + [make_exp(i, 100) for i in range(1, 6)]

# ---- FisherDelta ----
model_d = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))
delta_strat = FisherDeltaStrategy(
    model_d, SGD(model_d.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True), equivalence_metrics(epsilon=True),
        compute_metrics(), loggers=[InteractiveLogger(verbose=False)],
    ),
    train_epochs=5, train_mb_size=32,
)

print("\n--- FisherDelta ---")
delta_times = []
for exp in exps:
    t0 = time.time()
    delta_strat.train(exp)
    delta_times.append(time.time() - t0)
    delta_strat.eval(exps)
    print(f"  Task {exp.task_id}: {len(exp.train_dataset):>5} samples, {delta_times[-1]:.2f}s")

# ---- FullRetrain ----
model_f = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))
full_strat = FullRetrainStrategy(
    model_f, SGD(model_f.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
    train_epochs=5, train_mb_size=32,
)

print("\n--- FullRetrain ---")
full_times = []
for exp in exps:
    t0 = time.time()
    full_strat.train(exp)
    full_times.append(time.time() - t0)
    print(f"  Task {exp.task_id}: {len(exp.train_dataset):>5} samples, {full_times[-1]:.2f}s (cumulative retrain)")

delta_total = sum(delta_times)
full_total = sum(full_times)
actual_speedup = full_total / max(delta_total, 1e-6)

cert = delta_strat.last_certificate
delta_acc = delta_strat.evaluator.get_last_metrics().get("accuracy/stream", 0)
full_results = full_strat.eval(exps)
full_acc = full_results.get("accuracy/stream", 0)

print(f"\n{'='*60}")
print(f"  Delta total time:       {delta_total:.2f}s")
print(f"  Full retrain total:     {full_total:.2f}s")
print(f"  Actual wall speedup:    {actual_speedup:.2f}x")
print(f"  Estimated speedup:      {cert.compute_ratio:.1f}x")
print(f"  Delta accuracy:         {delta_acc*100:.1f}%")
print(f"  Full accuracy:          {full_acc*100:.1f}%")
print(f"  Epsilon:                {cert.epsilon_param:.6f}")
print(f"{'='*60}")
passed = actual_speedup > 1.0
print(f"  TEST 12 {'PASSED' if passed else 'FAILED'} — actual speedup {actual_speedup:.2f}x")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_12_speedup_proof", "status": "PASS" if passed else "FAIL",
    "delta_total_time": round(delta_total, 3), "full_total_time": round(full_total, 3),
    "actual_wall_speedup": round(actual_speedup, 2),
    "estimated_speedup": cert.compute_ratio,
    "delta_accuracy": delta_acc, "full_accuracy": full_acc,
    "epsilon_param": cert.epsilon_param,
    "delta_times_per_task": [round(t, 3) for t in delta_times],
    "full_times_per_task": [round(t, 3) for t in full_times],
}, open(out / "test_12_speedup_proof.json", "w"), indent=2)
print(f"  Saved: {out / 'test_12_speedup_proof.json'}")
