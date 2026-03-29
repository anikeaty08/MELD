"""
Test 7 — SVHN (Street View House Numbers), CNN
Needs: pip install torchvision
Run: python test_runs/test_7_svhn.py
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
print("  TEST 7: SVHN — CNN, 5 tasks x 2 digits")
print("  Proves: works on real-world digit recognition data")
print("=" * 60)

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

strategy = FisherDeltaStrategy(
    model, SGD(model.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True, kl_bound=True),
        calibration_metrics(),
        compute_metrics(),
        loggers=[InteractiveLogger()],
    ),
    train_epochs=2, train_mb_size=64,
)

stream = DeltaStream("SVHN", n_tasks=5, classes_per_task=2, data_root="./data")

for exp in stream.train_stream:
    print(f"\nTask {exp.task_id} — {len(exp.train_dataset)} samples")
    strategy.train(exp)
    strategy.eval(stream.test_stream)

cert = strategy.last_certificate
print("\n" + cert.summary())
passed = cert.epsilon_param < float("inf")
print(f"  TEST 7 {'PASSED' if passed else 'FAILED'}")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_7_svhn", "status": "PASS" if passed else "FAIL",
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "is_equivalent": cert.is_equivalent, "shift_type": cert.shift_type,
    "ece_before": cert.ece_before, "ece_after": cert.ece_after, "ece_delta": cert.ece_delta,
    "compute_ratio": cert.compute_ratio, "ce_scale": cert.ce_scale, "ewc_scale": cert.ewc_scale,
}, open(out / "test_7_svhn.json", "w"), indent=2)
print(f"  Saved: {out / 'test_7_svhn.json'}")
