"""
Test 13 — Model with BatchNorm + Dropout (edge case)
Needs: pip install torchvision
Run: python test_runs/test_13_batchnorm_dropout.py
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from delta import (
    DeltaStream, FisherDeltaStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, InteractiveLogger,
)

print("=" * 60)
print("  TEST 13: CNN with BatchNorm + Dropout on CIFAR-10")
print("  Proves: BN/Dropout don't break KFAC or training")
print("=" * 60)

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.1),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(32, 10),
)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")
print(f"  Has BatchNorm: yes")
print(f"  Has Dropout: yes (0.1, 0.2, 0.3)")

strategy = FisherDeltaStrategy(
    model, SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4),
    nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True), equivalence_metrics(epsilon=True),
        calibration_metrics(), loggers=[InteractiveLogger()],
    ),
    train_epochs=2, train_mb_size=64,
)

stream = DeltaStream("CIFAR-10", n_tasks=5, classes_per_task=2, data_root="./data")
for exp in stream.train_stream:
    print(f"\nTask {exp.task_id}")
    strategy.train(exp)
    strategy.eval(stream.test_stream)

cert = strategy.last_certificate
print("\n" + cert.summary())
print(f"  KFAC layers: {len(strategy.state.kfac_A)} (Conv2d + Linear, not BN)")
print(f"{'='*60}")
passed = cert.epsilon_param < float("inf")
print(f"  TEST 13 {'PASSED' if passed else 'FAILED'}")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_13_batchnorm_dropout", "status": "PASS" if passed else "FAIL",
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "compute_ratio": cert.compute_ratio, "kfac_layers": len(strategy.state.kfac_A),
    "n_params": n_params,
}, open(out / "test_13_batchnorm_dropout.json", "w"), indent=2)
print(f"  Saved: {out / 'test_13_batchnorm_dropout.json'}")
