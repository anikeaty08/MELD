# Delta

Provably equivalent continual learning framework. Train on new data only, get mathematically bounded equivalence to full retraining.

```python
from delta import (
    DeltaStream, FisherDeltaStrategy, EvaluationPlugin,
    accuracy_metrics, equivalence_metrics, InteractiveLogger
)
import torch.nn as nn
from torch.optim import SGD

model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = SGD(model.parameters(), lr=0.01)

stream = DeltaStream("synthetic", n_tasks=5, scenario="class_incremental")

strategy = FisherDeltaStrategy(
    model, optimizer, nn.CrossEntropyLoss(),
    evaluator=EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True),
        loggers=[InteractiveLogger()]
    )
)

for experience in stream.train_stream:
    strategy.train(experience)
    strategy.eval(stream.test_stream)

print(strategy.last_certificate.summary())
```

## What it does

Given a model trained on old data D_old, and new data D_new arriving:
- Trains on **only D_new** (old data is never replayed)
- Uses KFAC-approximated Fisher information to regularize against catastrophic forgetting
- Derives bias correction weights mathematically from dataset sizes (not hyperparameters)
- Produces an **equivalence certificate** proving the update is close to full retraining
- Detects distribution shift (none / covariate / concept) and routes accordingly

## Install

```bash
git clone https://github.com/anikeaty08/MELD.git
cd MELD
pip install -e .
```

Core dependencies: `torch>=2.1`, `numpy>=1.24`. That's it.

Optional (for real dataset benchmarks):
```bash
pip install -e ".[full]"  # adds scipy, torchvision, continuum
```

## Works with any model

| Model Type | KFAC Coverage | Status |
|---|---|---|
| MLPs | Full KFAC on all Linear layers | Tested |
| CNNs | Full KFAC on Conv2d + Linear | Tested |
| Transformers | KFAC on attention/FFN Linear layers | Works |
| Mixed architectures | KFAC where possible, diagonal Fisher fallback | Works |

```python
# Your CNN
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(32, 10)
)
# Just wrap it
strategy = FisherDeltaStrategy(model, optimizer, criterion)
```

## CLI

```bash
# Compare delta vs full retrain on synthetic data
python -m delta.demos.cli \
    --dataset synthetic --num-tasks 3 --classes-per-task 2 \
    --run-mode compare --epochs 2 --batch-size 16

# Fisher delta only
python -m delta.demos.cli \
    --dataset synthetic --run-mode fisher_delta --epochs 5

# CIFAR-10 benchmark (requires torchvision + continuum)
python -m delta.demos.cifar10_benchmark --quick
```

## Framework structure

```
delta/
  core/           # Mathematical engine (KFAC, shift detection, certificates)
    fisher.py     # KFAC computation for Linear + Conv2d
    shift.py      # 3-way distribution shift detector
    certificate.py# Equivalence certificate with formal bounds
    calibration.py# ECE tracking + temperature scaling
    state.py      # Compact statistics snapshot (no raw data)
  training/       # Strategy layer (hooks + plugins)
    base.py       # BaseStrategy with hook system
    fisher_delta.py # FisherDeltaStrategy
    full_retrain.py # FullRetrainStrategy (baseline)
  benchmarks/     # Data abstraction
    stream.py     # DeltaStream + Experience
    dataset_base.py # ContinualDataset registry
  evaluation/     # Metrics + plugin system
  logging/        # InteractiveLogger, CSVLogger
  demos/          # Models, datasets, CLI, benchmarks
```

## What the certificate reports

After each task update, `strategy.last_certificate` contains:

| Field | Meaning |
|---|---|
| `epsilon_param` | Parameter-space distance bound (Theorem 1) |
| `kl_bound` | KL divergence bound on predictions (Theorem 2) |
| `is_equivalent` | Whether both bounds are within tolerance |
| `shift_type` | "none", "covariate", or "concept" |
| `ece_before` / `ece_after` | Calibration before/after update |
| `compute_ratio` | Estimated speedup vs full retraining |
| `ce_scale` / `ewc_scale` | Derived bias correction weights |

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

25 tests covering: PS requirements, hook system, plugin injection, Conv2d KFAC, temperature scaling, speedup ratio, epsilon stability, cross-platform compatibility.

## Cross-platform

Works on Windows, macOS (including Apple Silicon via MPS), and Linux. Auto-detects CUDA > MPS > CPU. No platform-specific code in the framework.

## Theory

See [docs/theory.md](docs/theory.md) for the full mathematical foundations including:
- DeltaGrad bias correction derivation
- KFAC Hessian approximation (Linear + Conv2d)
- Formal equivalence bounds (Theorems 1 and 2)
- 3-way shift detection and routing
- Calibration preservation guarantees

# Commit 4 - 2026-03-29 05:53:27

# Commit 6 - 2026-03-29 08:34:28

# Commit 8 - 2026-03-29 04:18:59

# Commit 11 - 2026-03-29 06:42:09

# Commit 16 - 2026-03-29 08:40:30

# Commit 18 - 2026-03-28 22:34:34

# Commit 20 - 2026-03-28 18:57:33

# Commit 23 - 2026-03-28 13:55:50

# Commit 26 - 2026-03-29 05:33:30

# Commit 32 - 2026-03-28 18:36:05

# Commit 35 - 2026-03-29 08:33:04

# Update 7 - 2026-03-29 09:06:01
# Update 1 @ 2026-03-28 11:01:09
# Update 13 @ 2026-03-29 09:48:49
# Update 14 @ 2026-03-28 23:50:16
# Update 17 @ 2026-03-28 22:18:23
# Update 30 @ 2026-03-29 08:15:44
# Update 2 @ 2026-03-29 09:10:01
# Update 5 @ 2026-03-29 05:05:41