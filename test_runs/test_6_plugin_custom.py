"""
Test 6 — Custom model + custom plugin + custom dataset
No extra deps needed. Just torch + numpy.
Run: python test_runs/test_6_plugin_custom.py
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset
from delta import (
    DeltaStream, Experience, FisherDeltaStrategy, BaseStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    InteractiveLogger,
)

print("=" * 60)
print("  TEST 6: Custom model + plugin + hand-built dataset")
print("  Proves: any model works, plugins inject, custom data works")
print("=" * 60)


# ---- Custom model: wide MLP with dropout ----
class WideMLPWithDropout(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---- Custom plugin: tracks gradient norms ----
class GradNormTracker:
    def __init__(self):
        self.norms = []

    def after_backward(self, strategy):
        total = sum(
            p.grad.norm().item() ** 2
            for p in strategy.model.parameters()
            if p.grad is not None
        ) ** 0.5
        self.norms.append(total)


# ---- Custom plugin: counts training steps ----
class StepCounter:
    def __init__(self):
        self.steps = 0

    def after_training_iteration(self, strategy):
        self.steps += 1


# ---- Hand-built dataset: 4 classes, 200 samples each ----
print("\n--- Building custom dataset ---")
torch.manual_seed(123)
in_dim = 50
n_classes = 8
experiences = []
for task_id in range(4):
    classes = [task_id * 2, task_id * 2 + 1]
    tx, ty, ex, ey = [], [], [], []
    for c in classes:
        center = torch.randn(in_dim) * 2
        tx.append(center + torch.randn(100, in_dim) * 0.3)
        ty.append(torch.full((100,), c, dtype=torch.long))
        ex.append(center + torch.randn(30, in_dim) * 0.3)
        ey.append(torch.full((30,), c, dtype=torch.long))
    experiences.append(Experience(
        train_dataset=TensorDataset(torch.cat(tx), torch.cat(ty)),
        test_dataset=TensorDataset(torch.cat(ex), torch.cat(ey)),
        task_id=task_id,
        classes_in_this_experience=classes,
        dataset_name="custom",
    ))
print(f"  Created {len(experiences)} tasks, {n_classes} classes, {in_dim}D features")

# ---- Run with plugins ----
model = WideMLPWithDropout(in_dim, 128, n_classes)
grad_tracker = GradNormTracker()
step_counter = StepCounter()

strategy = FisherDeltaStrategy(
    model, SGD(model.parameters(), lr=0.005, momentum=0.9), nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True),
        loggers=[InteractiveLogger()],
    ),
    train_epochs=5, train_mb_size=32,
)
strategy.add_plugin(grad_tracker)
strategy.add_plugin(step_counter)

print("\n--- Training with plugins ---")
for exp in experiences:
    print(f"\nTask {exp.task_id}")
    strategy.train(exp)
    strategy.eval(experiences)  # eval on all tasks

cert = strategy.last_certificate
print("\n" + cert.summary())

print(f"\n{'='*60}")
print(f"  Custom model params:    {sum(p.numel() for p in model.parameters()):,}")
print(f"  Total training steps:   {step_counter.steps}")
print(f"  Grad norms recorded:    {len(grad_tracker.norms)}")
print(f"  Mean grad norm:         {sum(grad_tracker.norms)/len(grad_tracker.norms):.4f}")
print(f"  Max grad norm:          {max(grad_tracker.norms):.4f}")
print(f"  Epsilon bound:          {cert.epsilon_param:.6f}")
print(f"  ce_scale:               {cert.ce_scale:.4f}")
print(f"  ewc_scale:              {cert.ewc_scale:.4f}")
print(f"  Speedup:                {cert.compute_ratio:.1f}x")
print(f"{'='*60}")
passed = step_counter.steps > 0 and cert.epsilon_param < float("inf")
print(f"  TEST 6 {'PASSED' if passed else 'FAILED'}")

import json
from pathlib import Path
out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
json.dump({
    "test": "test_6_plugin_custom", "status": "PASS" if passed else "FAIL",
    "n_params": sum(p.numel() for p in model.parameters()),
    "training_steps": step_counter.steps, "grad_norms_count": len(grad_tracker.norms),
    "mean_grad_norm": sum(grad_tracker.norms)/len(grad_tracker.norms),
    "max_grad_norm": max(grad_tracker.norms),
    "epsilon_param": cert.epsilon_param, "kl_bound": cert.kl_bound,
    "ce_scale": cert.ce_scale, "ewc_scale": cert.ewc_scale,
    "compute_ratio": cert.compute_ratio,
}, open(out / "test_6_plugin_custom.json", "w"), indent=2)
print(f"  Saved: {out / 'test_6_plugin_custom.json'}")
