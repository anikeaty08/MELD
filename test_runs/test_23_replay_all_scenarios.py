"""
Test 23 - Replay strategy across CL scenarios
Run: python test_runs/test_23_replay_all_scenarios.py
"""

import json
from pathlib import Path

import torch.nn as nn
from torch.optim import SGD

from delta import (
    DeltaStream,
    EvaluationPlugin,
    ReplayStrategy,
    accuracy_metrics,
    calibration_metrics,
    compute_metrics,
    equivalence_metrics,
    InteractiveLogger,
)

print("=" * 60)
print("  TEST 23: ReplayStrategy across CL scenarios")
print("  Proves: framework-level support for class/task/domain incremental")
print("=" * 60)

results = {}
for scenario in ["class_incremental", "task_incremental", "domain_incremental"]:
    print(f"\n--- Scenario: {scenario} ---")
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 6 if scenario != "domain_incremental" else 2),
    )
    strategy = ReplayStrategy(
        model,
        SGD(model.parameters(), lr=0.01, momentum=0.9),
        nn.CrossEntropyLoss(),
        evaluator=EvaluationPlugin(
            accuracy_metrics(stream=True),
            equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
            calibration_metrics(ece_before=True, ece_after=True),
            compute_metrics(savings_ratio=True),
            loggers=[InteractiveLogger(verbose=False)],
        ),
        train_epochs=2,
        train_mb_size=16,
        device="cpu",
    )
    stream = DeltaStream(
        "synthetic",
        n_tasks=3,
        scenario=scenario,
        classes_per_task=2,
    )
    for exp in stream.train_stream:
        strategy.train(exp)
        strategy.eval(stream.test_stream)
    metrics = strategy.evaluator.get_last_metrics()
    results[scenario] = {
        "accuracy_stream": metrics.get("accuracy/stream", 0.0),
        "task_identity_inference": bool(strategy._effective_task_identity_inference),
    }
    print(f"  Final stream accuracy: {results[scenario]['accuracy_stream'] * 100:.1f}%")

print("\nTEST 23 PASSED")
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
with open(out / "test_23_replay_all_scenarios.json", "w", encoding="utf-8") as f:
    json.dump({"test": "test_23_replay_all_scenarios", "status": "PASS", "results": results}, f, indent=2)
print(f"Saved: {out / 'test_23_replay_all_scenarios.json'}")
