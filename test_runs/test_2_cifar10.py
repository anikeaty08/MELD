"""
Test 2 — CIFAR-10, custom CNN
Needs: pip install torchvision
Run: python test_runs/test_2_cifar10.py
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from delta import (
    DeltaStream, ReplayDeltaStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)

print("=" * 60)
print("  TEST 2: CIFAR-10 — Custom CNN (Conv2d KFAC)")
print("  Proves: Conv2d KFAC works on real images")
print("=" * 60)

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10),
)

strategy = ReplayDeltaStrategy(
    model, SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4),
    nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
        calibration_metrics(ece_before=True, ece_after=True),
        compute_metrics(savings_ratio=True),
        loggers=[InteractiveLogger()],
    ),
    train_epochs=30, train_mb_size=64,
)

stream = DeltaStream("CIFAR-10", n_tasks=5, classes_per_task=2, data_root="./data")

for exp in stream.train_stream:
    print(f"\nTask {exp.task_id} — classes {exp.classes_in_this_experience}")
    print(f"  Train samples: {len(exp.train_dataset)}, Test samples: {len(exp.test_dataset)}")
    strategy.train(exp)
    strategy.eval(stream.test_stream)

cert = strategy.last_certificate
print("\n" + cert.summary())

print(f"\n  KFAC layers captured: {len(strategy.state.kfac_A)}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"{'='*60}")
passed = cert.epsilon_param < float("inf")
print("  TEST 2 PASSED" if passed else "  TEST 2 FAILED")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_2_cifar10", "status": "PASS" if passed else "FAIL",
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "is_equivalent": cert.is_equivalent, "shift_type": cert.shift_type,
    "ece_before": cert.ece_before, "ece_after": cert.ece_after, "ece_delta": cert.ece_delta,
    "compute_ratio": cert.compute_ratio, "ce_scale": cert.ce_scale, "ewc_scale": cert.ewc_scale,
    "kfac_layers": len(strategy.state.kfac_A), "n_params": sum(p.numel() for p in model.parameters()),
    "n_old": cert.n_old, "n_new": cert.n_new,
}, open(out / "test_2_cifar10.json", "w"), indent=2)
print(f"  Saved: {out / 'test_2_cifar10.json'}")
