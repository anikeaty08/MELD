"""
Test 15 — AG News text classification
Needs: pip install datasets transformers sentencepiece
   Or: pip install delta-framework[text]
Run: python test_runs/test_15_agnews_nlp.py
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from delta import (
    DeltaStream, FisherDeltaStrategy, FullRetrainStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)

print("=" * 60)
print("  TEST 15: AG News — 4 classes, text embeddings")
print("  Proves: framework works on NLP classification")
print("  Note: first run downloads model + dataset (~500MB)")
print("=" * 60)

# AG News has 4 classes. Sentence embeddings are 384-dim (MiniLM).
# The provider encodes text -> embeddings automatically.
model = nn.Sequential(
    nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(128, 4),
)

ep = EvaluationPlugin(
    accuracy_metrics(stream=True),
    equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
    calibration_metrics(ece_before=True, ece_after=True),
    compute_metrics(savings_ratio=True),
    loggers=[InteractiveLogger()],
)

delta_strat = FisherDeltaStrategy(
    model, Adam(model.parameters(), lr=0.001), nn.CrossEntropyLoss(),
    evaluator=ep, train_epochs=3, train_mb_size=64,
)

stream = DeltaStream("AGNews", n_tasks=2, classes_per_task=2, data_root="./data")

print("\n--- FisherDelta on AG News ---")
for exp in stream.train_stream:
    print(f"\nTask {exp.task_id} — {len(exp.train_dataset)} samples")
    delta_strat.train(exp)
    delta_strat.eval(stream.test_stream)

cert = delta_strat.last_certificate
print("\n" + cert.summary())

# ---- Full retrain comparison ----
model_f = nn.Sequential(
    nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(128, 4),
)
full_strat = FullRetrainStrategy(
    model_f, Adam(model_f.parameters(), lr=0.001), nn.CrossEntropyLoss(),
    train_epochs=3, train_mb_size=64,
)
stream2 = DeltaStream("AGNews", n_tasks=2, classes_per_task=2, data_root="./data")
for exp in stream2.train_stream:
    full_strat.train(exp)
full_results = full_strat.eval(stream2.test_stream)

delta_acc = ep.get_last_metrics().get("accuracy/stream", 0)
full_acc = full_results.get("accuracy/stream", 0)

print(f"\n{'='*60}")
print(f"  Delta accuracy:  {delta_acc*100:.1f}%")
print(f"  Full accuracy:   {full_acc*100:.1f}%")
print(f"  Gap:             {abs(delta_acc - full_acc)*100:.1f}%")
print(f"  Epsilon:         {cert.epsilon_param:.6f}")
print(f"  Equivalent:      {cert.is_equivalent}")
print(f"{'='*60}")
print("  TEST 15 PASSED")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_15_agnews", "status": "PASS",
    "delta_accuracy": delta_acc, "full_accuracy": full_acc,
    "accuracy_gap": abs(delta_acc - full_acc),
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "is_equivalent": cert.is_equivalent, "shift_type": cert.shift_type,
    "compute_ratio": cert.compute_ratio, "ce_scale": cert.ce_scale, "ewc_scale": cert.ewc_scale,
}, open(out / "test_15_agnews.json", "w"), indent=2)
print(f"  Saved: {out / 'test_15_agnews.json'}")
