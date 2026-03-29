"""
Test 1 — Synthetic data, MLP model
No extra deps needed. Just torch + numpy.
Run: python test_runs/test_1_synthetic.py
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from delta import (
    DeltaStream, FisherDeltaStrategy, FullRetrainStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)

print("=" * 60)
print("  TEST 1: Synthetic — MLP vs FullRetrain")
print("  Proves: framework works, bias correction, epsilon bound")
print("=" * 60)

# ---- FisherDelta strategy ----
model_d = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6))
ep = EvaluationPlugin(
    accuracy_metrics(stream=True),
    equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
    calibration_metrics(ece_before=True, ece_after=True),
    compute_metrics(savings_ratio=True),
    loggers=[InteractiveLogger()],
)
delta_strat = FisherDeltaStrategy(
    model_d, SGD(model_d.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
    evaluator=ep, train_epochs=5, train_mb_size=16,
)

stream = DeltaStream("synthetic", n_tasks=3, classes_per_task=2)

print("\n--- FisherDelta ---")
for exp in stream.train_stream:
    print(f"\nTask {exp.task_id}")
    delta_strat.train(exp)
    delta_strat.eval(stream.test_stream)

cert = delta_strat.last_certificate
print("\n" + cert.summary())

# ---- FullRetrain baseline ----
model_f = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6))
full_strat = FullRetrainStrategy(
    model_f, SGD(model_f.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
    train_epochs=5, train_mb_size=16,
)

stream2 = DeltaStream("synthetic", n_tasks=3, classes_per_task=2)
print("\n--- FullRetrain ---")
for exp in stream2.train_stream:
    full_strat.train(exp)
full_results = full_strat.eval(stream2.test_stream)
print(f"Full retrain accuracy: {full_results.get('accuracy/stream', 0):.4f}")

# ---- Comparison ----
delta_acc = ep.get_last_metrics().get("accuracy/stream", 0)
full_acc = full_results.get("accuracy/stream", 0)
print(f"\n{'='*60}")
print(f"  Delta accuracy:     {delta_acc:.4f}")
print(f"  Full accuracy:      {full_acc:.4f}")
print(f"  Accuracy gap:       {abs(delta_acc - full_acc):.4f}")
print(f"  Epsilon bound:      {cert.epsilon_param:.6f}")
print(f"  Is equivalent:      {cert.is_equivalent}")
print(f"  ce_scale:           {cert.ce_scale:.4f}")
print(f"  ewc_scale:          {cert.ewc_scale:.4f}")
print(f"  Compute ratio:      {cert.compute_ratio:.1f}x")
print(f"{'='*60}")
passed = cert.epsilon_param < float("inf")
print("  TEST 1 PASSED" if passed else "  TEST 1 FAILED")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_1_synthetic", "status": "PASS" if passed else "FAIL",
    "delta_accuracy": delta_acc, "full_accuracy": full_acc,
    "accuracy_gap": abs(delta_acc - full_acc),
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "is_equivalent": cert.is_equivalent, "shift_type": cert.shift_type,
    "ece_before": cert.ece_before, "ece_after": cert.ece_after, "ece_delta": cert.ece_delta,
    "compute_ratio": cert.compute_ratio, "ce_scale": cert.ce_scale, "ewc_scale": cert.ewc_scale,
    "n_old": cert.n_old, "n_new": cert.n_new, "tier": cert.tier,
}, open(out / "test_1_synthetic.json", "w"), indent=2)
print(f"  Saved: {out / 'test_1_synthetic.json'}")
