"""
Test 17 — DBpedia 14-class topic classification
Needs: pip install datasets transformers sentencepiece
   Or: pip install delta-framework[text]
Run: python test_runs/test_17_dbpedia_nlp.py
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from delta import (
    DeltaStream, FisherDeltaStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)

print("=" * 60)
print("  TEST 17: DBpedia — 14 classes, 7 tasks x 2 classes")
print("  Proves: scales to many NLP classes, epsilon over 7 tasks")
print("  Note: first run downloads dataset (~70MB)")
print("=" * 60)

# DBpedia has 14 classes. Embeddings are 384-dim.
model = nn.Sequential(
    nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 14),
)

strategy = FisherDeltaStrategy(
    model, Adam(model.parameters(), lr=0.001), nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
        calibration_metrics(ece_before=True, ece_after=True),
        compute_metrics(savings_ratio=True),
        loggers=[InteractiveLogger()],
    ),
    train_epochs=2, train_mb_size=64,
)

stream = DeltaStream("DBpedia", n_tasks=7, classes_per_task=2, data_root="./data")

epsilons = []
for exp in stream.train_stream:
    print(f"\nTask {exp.task_id} — {len(exp.train_dataset)} samples")
    strategy.train(exp)
    strategy.eval(stream.test_stream)
    epsilons.append(strategy.last_certificate.epsilon_param)

cert = strategy.last_certificate
print("\n" + cert.summary())

print(f"\n  Epsilons across 7 tasks: {[f'{e:.4f}' for e in epsilons]}")
print(f"  All finite: {all(e < float('inf') for e in epsilons)}")
print(f"  Speedup: {cert.compute_ratio:.1f}x")
print(f"{'='*60}")
passed = all(e < float("inf") for e in epsilons)
print(f"  TEST 17 {'PASSED' if passed else 'FAILED'}")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_17_dbpedia", "status": "PASS" if passed else "FAIL",
    "epsilons": epsilons, "compute_ratio": cert.compute_ratio,
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "shift_type": cert.shift_type,
}, open(out / "test_17_dbpedia.json", "w"), indent=2)
print(f"  Saved: {out / 'test_17_dbpedia.json'}")
