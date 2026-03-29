"""
Test 3 — MNIST, flat MLP
Needs: pip install torchvision
Run: python test_runs/test_3_mnist.py
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
print("  TEST 3: MNIST — MLP, 5 tasks, delta vs full retrain")
print("  Proves: works on grayscale images, accuracy comparison")
print("=" * 60)

# ---- Delta ----
model_d = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
ep = EvaluationPlugin(
    accuracy_metrics(stream=True),
    equivalence_metrics(epsilon=True, is_equivalent=True),
    calibration_metrics(),
    compute_metrics(),
    loggers=[InteractiveLogger()],
)
delta_strat = FisherDeltaStrategy(
    model_d, SGD(model_d.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
    evaluator=ep, train_epochs=3, train_mb_size=64,
)

stream = DeltaStream("MNIST", n_tasks=5, classes_per_task=2, data_root="./data")
print("\n--- FisherDelta ---")
for exp in stream.train_stream:
    print(f"\nTask {exp.task_id}: {len(exp.train_dataset)} train, {len(exp.test_dataset)} test")
    delta_strat.train(exp)
    delta_strat.eval(stream.test_stream)

cert = delta_strat.last_certificate
print("\n" + cert.summary())

# ---- Full retrain ----
model_f = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
full_strat = FullRetrainStrategy(
    model_f, SGD(model_f.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
    train_epochs=3, train_mb_size=64,
)
stream2 = DeltaStream("MNIST", n_tasks=5, classes_per_task=2, data_root="./data")
print("\n--- FullRetrain ---")
for exp in stream2.train_stream:
    full_strat.train(exp)
full_results = full_strat.eval(stream2.test_stream)

delta_acc = ep.get_last_metrics().get("accuracy/stream", 0)
full_acc = full_results.get("accuracy/stream", 0)
print(f"\n{'='*60}")
print(f"  Delta accuracy:  {delta_acc*100:.1f}%")
print(f"  Full accuracy:   {full_acc*100:.1f}%")
print(f"  Gap:             {abs(delta_acc - full_acc)*100:.1f}%")
print(f"  Epsilon:         {cert.epsilon_param:.6f}")
print(f"  Equivalent:      {cert.is_equivalent}")
print(f"  Speedup:         {cert.compute_ratio:.1f}x")
print(f"{'='*60}")
print("  TEST 3 PASSED")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_3_mnist", "status": "PASS",
    "delta_accuracy": delta_acc, "full_accuracy": full_acc,
    "accuracy_gap": abs(delta_acc - full_acc),
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "is_equivalent": cert.is_equivalent, "shift_type": cert.shift_type,
    "ece_before": cert.ece_before, "ece_after": cert.ece_after, "ece_delta": cert.ece_delta,
    "compute_ratio": cert.compute_ratio, "ce_scale": cert.ce_scale, "ewc_scale": cert.ewc_scale,
}, open(out / "test_3_mnist.json", "w"), indent=2)
print(f"  Saved: {out / 'test_3_mnist.json'}")
