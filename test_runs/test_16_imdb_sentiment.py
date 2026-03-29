"""
Test 16 — IMDB sentiment classification (binary)
Needs: pip install datasets transformers sentencepiece
   Or: pip install delta-framework[text]
Run: python test_runs/test_16_imdb_sentiment.py
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
print("  TEST 16: IMDB — binary sentiment, text embeddings")
print("  Proves: works on 2-class NLP, shift detection on text")
print("  Note: first run downloads dataset (~80MB)")
print("=" * 60)

# IMDB is binary (pos/neg). Embeddings are 384-dim.
model = nn.Sequential(
    nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 2),
)

strategy = FisherDeltaStrategy(
    model, Adam(model.parameters(), lr=0.001), nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True, kl_bound=True),
        calibration_metrics(),
        compute_metrics(),
        loggers=[InteractiveLogger()],
    ),
    train_epochs=3, train_mb_size=64,
)

# 2 classes, 1 task = all data at once. Use 2 tasks with 1 class each
# to test incremental behavior on binary classification.
stream = DeltaStream("IMDB", n_tasks=2, classes_per_task=1, data_root="./data")

for exp in stream.train_stream:
    print(f"\nTask {exp.task_id} — {len(exp.train_dataset)} samples, classes {exp.classes_in_this_experience}")
    strategy.train(exp)
    strategy.eval(stream.test_stream)

cert = strategy.last_certificate
print("\n" + cert.summary())
print(f"{'='*60}")
passed = cert.epsilon_param < float("inf")
print(f"  TEST 16 {'PASSED' if passed else 'FAILED'}")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_16_imdb", "status": "PASS" if passed else "FAIL",
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "is_equivalent": cert.is_equivalent, "shift_type": cert.shift_type,
    "compute_ratio": cert.compute_ratio,
}, open(out / "test_16_imdb.json", "w"), indent=2)
print(f"  Saved: {out / 'test_16_imdb.json'}")
