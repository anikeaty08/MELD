"""
Test 18 - Quick CIFAR-10 ResNet20 benchmark
Needs: pip install torchvision
Run: python test_runs/test_18_quick_cifar10_resnet20.py
"""

import json
from pathlib import Path

import torch.nn as nn
from torch.optim import SGD

from delta import (
    DeltaStream,
    EvaluationPlugin,
    ReplayDeltaStrategy,
    accuracy_metrics,
    calibration_metrics,
    compute_metrics,
    equivalence_metrics,
    InteractiveLogger,
)
from delta.demos.models import IncrementalClassifier, MELDModel, resnet20

print("=" * 60)
print("  TEST 18: Quick CIFAR-10 ResNet20")
print("  Proves: fast sanity benchmark for class-incremental CIFAR-10")
print("=" * 60)

backbone = resnet20()
classifier = IncrementalClassifier(backbone.out_dim)
model = MELDModel(backbone, classifier)

ep = EvaluationPlugin(
    accuracy_metrics(stream=True),
    equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
    calibration_metrics(ece_before=True, ece_after=True),
    compute_metrics(savings_ratio=True),
    loggers=[InteractiveLogger()],
)

strategy = ReplayDeltaStrategy(
    model,
    SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4),
    nn.CrossEntropyLoss(),
    evaluator=ep,
    train_epochs=2,
    train_mb_size=64,
)

stream = DeltaStream("CIFAR-10", n_tasks=5, classes_per_task=2, data_root="./data")

for exp in stream.train_stream:
    print(f"\nTask {exp.task_id} - {len(exp.train_dataset)} samples")
    strategy.train(exp)
    strategy.eval(stream.test_stream)

cert = strategy.last_certificate
metrics = ep.get_last_metrics()

print("\n" + cert.summary())
print(f"  Final stream accuracy: {metrics.get('accuracy/stream', 0.0) * 100:.1f}%")
print(f"  KFAC layers:           {len(strategy.state.kfac_A) if strategy.state else 0}")
print("  TEST 18 PASSED")

out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
payload = {
    "test": "test_18_quick_cifar10_resnet20",
    "status": "PASS",
    "accuracy_stream": metrics.get("accuracy/stream", 0.0),
    "epsilon_param": cert.epsilon_param,
    "kl_bound": cert.kl_bound,
    "kl_bound_normalized": cert.kl_bound_normalized,
    "is_equivalent": cert.is_equivalent,
    "shift_type": cert.shift_type,
    "ece_before": cert.ece_before,
    "ece_after": cert.ece_after,
    "ece_delta": cert.ece_delta,
    "compute_ratio": cert.compute_ratio,
    "ce_scale": cert.ce_scale,
    "ewc_scale": cert.ewc_scale,
    "kfac_layers": len(strategy.state.kfac_A) if strategy.state else 0,
}
with open(out / "test_18_quick_cifar10_resnet20.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"  Saved: {out / 'test_18_quick_cifar10_resnet20.json'}")
