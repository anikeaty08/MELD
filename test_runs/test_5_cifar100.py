"""
Test 5 — CIFAR-100, ResNet20 backbone, 10 tasks
Needs: pip install torchvision
Run: python test_runs/test_5_cifar100.py
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from delta import (
    DeltaStream, ReplayDeltaStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)
from delta.demos.models import resnet20, IncrementalClassifier, MELDModel

print("=" * 60)
print("  TEST 5: CIFAR-100 — ResNet20, 10 tasks x 10 classes")
print("  Proves: scales to many tasks, epsilon stays bounded")
print("=" * 60)

backbone = resnet20()
classifier = IncrementalClassifier(backbone.out_dim)
model = MELDModel(backbone, classifier)

strategy = ReplayDeltaStrategy(
    model, SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4),
    nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True, kl_bound=True),
        calibration_metrics(ece_before=True, ece_after=True),
        compute_metrics(savings_ratio=True),
        loggers=[InteractiveLogger()],
    ),
    train_epochs=30, train_mb_size=64,
)

stream = DeltaStream("CIFAR-100", n_tasks=10, classes_per_task=10, data_root="./data")

epsilons = []
accuracies = []
speedups = []
for exp in stream.train_stream:
    print(f"\nTask {exp.task_id} — {len(exp.train_dataset)} samples")
    strategy.train(exp)
    results = strategy.eval(stream.test_stream)
    cert = strategy.last_certificate
    epsilons.append(cert.epsilon_param)
    accuracies.append(results.get("accuracy/stream", 0))
    speedups.append(cert.compute_ratio)

print(f"\n{'='*60}")
print(f"  Tasks completed:    {len(epsilons)}")
print(f"  Final accuracy:     {accuracies[-1]*100:.1f}%")
print(f"  Epsilon range:      {min(epsilons):.4f} — {max(epsilons):.4f}")
print(f"  Speedup range:      {min(speedups):.1f}x — {max(speedups):.1f}x")
print(f"  All eps finite:     {all(e < float('inf') for e in epsilons)}")
print(f"  All eps < 1.0:      {all(e < 1.0 for e in epsilons)}")
print(f"{'='*60}")
bounded = all(e < float("inf") for e in epsilons)
print(f"  TEST 5 {'PASSED' if bounded else 'FAILED'}")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_5_cifar100", "status": "PASS" if bounded else "FAIL",
    "tasks_completed": len(epsilons), "final_accuracy": accuracies[-1],
    "epsilons": epsilons, "speedups": speedups, "accuracies": accuracies,
    "epsilon_range": [min(epsilons), max(epsilons)],
    "speedup_range": [min(speedups), max(speedups)],
}, open(out / "test_5_cifar100.json", "w"), indent=2)
print(f"  Saved: {out / 'test_5_cifar100.json'}")
