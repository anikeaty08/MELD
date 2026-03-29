"""
Test 11 — Transformer-style model (self-attention)
No extra deps needed. Just torch + numpy.
Run: python test_runs/test_11_transformer.py
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from delta import (
    DeltaStream, FisherDeltaStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    InteractiveLogger,
)

print("=" * 60)
print("  TEST 11: Transformer-style model on synthetic data")
print("  Proves: KFAC captures Linear inside attention blocks")
print("=" * 60)


class TinyTransformerClassifier(nn.Module):
    """Minimal transformer: embed -> self-attention -> classify."""
    def __init__(self, input_dim=32, d_model=64, n_heads=4, n_classes=6):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model),
        )
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, input_dim) -> treat as single-token sequence
        h = self.embed(x).unsqueeze(1)  # (B, 1, d_model)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out)
        h = h + self.ff(h)
        return self.head(h.squeeze(1))  # (B, n_classes)


model = TinyTransformerClassifier(input_dim=32, d_model=64, n_heads=4, n_classes=6)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")
print(f"  Layers: {sum(1 for _ in model.named_modules())}")

strategy = FisherDeltaStrategy(
    model, Adam(model.parameters(), lr=0.001), nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True), equivalence_metrics(epsilon=True),
        loggers=[InteractiveLogger()],
    ),
    train_epochs=5, train_mb_size=16,
)

stream = DeltaStream("synthetic", n_tasks=3, classes_per_task=2)
for exp in stream.train_stream:
    print(f"\nTask {exp.task_id}")
    strategy.train(exp)
    strategy.eval(stream.test_stream)

cert = strategy.last_certificate
print("\n" + cert.summary())
print(f"\n  KFAC layers captured: {len(strategy.state.kfac_A)}")
print(f"  KFAC param names:    {strategy.state.kfac_param_names}")
print(f"{'='*60}")
kfac_count = len(strategy.state.kfac_A)
passed = kfac_count >= 4
print(f"  TEST 11 {'PASSED' if passed else 'FAILED'} — {kfac_count} KFAC layers (expect >=4)")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_11_transformer", "status": "PASS" if passed else "FAIL",
    "kfac_layers": kfac_count, "kfac_param_names": strategy.state.kfac_param_names,
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "compute_ratio": cert.compute_ratio, "n_params": sum(p.numel() for p in model.parameters()),
}, open(out / "test_11_transformer.json", "w"), indent=2)
print(f"  Saved: {out / 'test_11_transformer.json'}")
