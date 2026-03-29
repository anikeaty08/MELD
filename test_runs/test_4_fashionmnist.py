"""
Test 4 — FashionMNIST, CNN
Needs: pip install torchvision
Run: python test_runs/test_4_fashionmnist.py
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from delta import (
    DeltaStream, FisherDeltaStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)

print("=" * 60)
print("  TEST 4: FashionMNIST — CNN (1-channel Conv2d KFAC)")
print("  Proves: Conv2d KFAC works on grayscale, shift detection")
print("=" * 60)

model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10),
)

strategy = FisherDeltaStrategy(
    model, SGD(model.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
        calibration_metrics(ece_before=True, ece_after=True),
        compute_metrics(savings_ratio=True),
        loggers=[InteractiveLogger()],
    ),
    train_epochs=3, train_mb_size=64,
)

stream = DeltaStream("FashionMNIST", n_tasks=5, classes_per_task=2, data_root="./data")

epsilons = []
shift_types = []
for exp in stream.train_stream:
    print(f"\nTask {exp.task_id}")
    strategy.train(exp)
    strategy.eval(stream.test_stream)
    cert = strategy.last_certificate
    epsilons.append(cert.epsilon_param)
    shift_types.append(cert.shift_type)

print(f"\n{'='*60}")
print(f"  Epsilons per task:  {[f'{e:.4f}' for e in epsilons]}")
print(f"  Shift types:        {shift_types}")
print(f"  ECE delta:          {cert.ece_delta:+.4f}")
print(f"  Speedup:            {cert.compute_ratio:.1f}x")
print(f"  All finite:         {all(e < float('inf') for e in epsilons)}")
print(f"{'='*60}")
passed = all(e < float("inf") for e in epsilons)
print("  TEST 4 PASSED" if passed else "  TEST 4 FAILED")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
cert = strategy.last_certificate
json.dump({
    "test": "test_4_fashionmnist", "status": "PASS" if passed else "FAIL",
    "epsilons": epsilons, "shift_types": shift_types,
    "ece_delta": cert.ece_delta, "compute_ratio": cert.compute_ratio,
    "ce_scale": cert.ce_scale, "ewc_scale": cert.ewc_scale,
}, open(out / "test_4_fashionmnist.json", "w"), indent=2)
print(f"  Saved: {out / 'test_4_fashionmnist.json'}")
