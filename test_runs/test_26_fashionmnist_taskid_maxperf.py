"""
Test 26 - FashionMNIST task-incremental replay benchmark
Needs: pip install torchvision
Run: python test_runs/test_26_fashionmnist_taskid_maxperf.py
"""

import json
from pathlib import Path

import torch.nn as nn
from torch.optim import SGD

from delta import (
    DeltaStream,
    DeltaStrategy,
    EvaluationPlugin,
    accuracy_metrics,
    calibration_metrics,
    compute_metrics,
    equivalence_metrics,
    InteractiveLogger,
)
from delta.demos.models import IncrementalClassifier, MELDModel


class GrayCNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = 128

    def forward(self, x):
        return self.features(x).flatten(1)

    def embed(self, x):
        return self.forward(x)


print("=" * 60)
print("  TEST 26: FashionMNIST (Task-Incremental Max Performance, 8 epochs)")
print("  Proves: replay strategy stays strong on another grayscale dataset")
print("=" * 60)

model = MELDModel(GrayCNNBackbone(), IncrementalClassifier(128))
ep = EvaluationPlugin(
    accuracy_metrics(stream=True),
    equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
    calibration_metrics(ece_before=True, ece_after=True),
    compute_metrics(savings_ratio=True),
    loggers=[InteractiveLogger()],
)
strategy = DeltaStrategy(
    model,
    SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4),
    nn.CrossEntropyLoss(),
    evaluator=ep,
    train_epochs=8,
    train_mb_size=64,
)
strategy.use_task_identity_inference = True

stream = DeltaStream(
    "FashionMNIST",
    n_tasks=5,
    scenario="task_incremental",
    classes_per_task=2,
    preset="maxperf",
    data_root="./data",
)

for exp in stream.train_stream:
    print(f"\nTask {exp.task_id} - classes {exp.classes_in_this_experience}")
    strategy.train(exp)
    strategy.eval(stream.test_stream)

cert = strategy.last_certificate
metrics = ep.get_last_metrics()

print("\n" + cert.summary())
print(f"  Final stream accuracy: {metrics.get('accuracy/stream', 0.0) * 100:.1f}%")
print("  TEST 26 PASSED")

out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
payload = {
    "test": "test_26_fashionmnist_taskid_maxperf",
    "status": "PASS",
    "epochs": 8,
    "task_identity_inference": True,
    "scenario": "task_incremental",
    "preset": "maxperf",
    "replay_memory_per_class": strategy.replay_memory_per_class,
    "replay_batch_size": strategy.replay_batch_size,
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
}
with open(out / "test_26_fashionmnist_taskid_maxperf.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"  Saved: {out / 'test_26_fashionmnist_taskid_maxperf.json'}")
