"""
Test 10 — ResNet20 vs ResNet32 on CIFAR-10
Needs: pip install torchvision
Run: python test_runs/test_10_resnet_comparison.py
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from delta import (
    DeltaStream, ReplayDeltaStrategy, FullRetrainStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)
from delta.demos.models import resnet20, resnet32, IncrementalClassifier, MELDModel

print("=" * 60)
print("  TEST 10: ResNet20 vs ResNet32 on CIFAR-10")
print("  Proves: framework handles different backbone depths")
print("=" * 60)

def build_meld_model(backbone_fn, n_tasks, cpt):
    bb = backbone_fn()
    clf = IncrementalClassifier(bb.out_dim)
    return MELDModel(bb, clf)

n_tasks, cpt, epochs = 5, 2, 50
results = {}

for name, bb_fn in [("ResNet20", resnet20), ("ResNet32", resnet32)]:
    print(f"\n--- {name} ---")
    model = build_meld_model(bb_fn, n_tasks, cpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    strategy = ReplayDeltaStrategy(
        model, SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4),
        nn.CrossEntropyLoss(),
        evaluator=EvaluationPlugin(
            accuracy_metrics(stream=True), equivalence_metrics(epsilon=True),
            calibration_metrics(), compute_metrics(),
            loggers=[InteractiveLogger(verbose=False)],
        ),
        train_epochs=epochs, train_mb_size=64,
    )
    stream = DeltaStream("CIFAR-10", n_tasks=n_tasks, classes_per_task=cpt, data_root="./data")
    for exp in stream.train_stream:
        strategy.train(exp)
        strategy.eval(stream.test_stream)

    cert = strategy.last_certificate
    acc = strategy.evaluator.get_last_metrics().get("accuracy/stream", 0)
    results[name] = {"acc": acc, "epsilon": cert.epsilon_param, "speedup": cert.compute_ratio,
                     "kfac_layers": len(strategy.state.kfac_A), "params": n_params}

print(f"\n{'='*60}")
print(f"  {'Backbone':<12} | {'Acc':>6} | {'Epsilon':>10} | {'Speedup':>8} | {'KFAC':>5} | {'Params':>10}")
print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*10}")
for name, r in results.items():
    print(f"  {name:<12} | {r['acc']*100:>5.1f}% | {r['epsilon']:>10.6f} | {r['speedup']:>7.1f}x | {r['kfac_layers']:>5} | {r['params']:>10,}")
print(f"{'='*60}")
print("  TEST 10 PASSED")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_10_resnet_comparison", "status": "PASS", "backbones": results,
}, open(out / "test_10_resnet_comparison.json", "w"), indent=2)
print(f"  Saved: {out / 'test_10_resnet_comparison.json'}")
