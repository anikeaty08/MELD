# MELD

MELD (Manifold-Equivalent Learning with Deployment guarantees) is a class-incremental learning framework built around three hard requirements: zero replay, pre-training safety checks, and structured deployment decisions. It stores model geometry instead of historical samples, derives a safety bound before training, and decides whether a delta update is safe to ship after training.

## What MELD does

- Uses only new task data during incremental updates.
- Captures replay-free snapshots with per-class means, diagonal covariances, classifier norms, and diagonal Fisher information.
- Computes a spectral pre-training safety bound and skips unsafe runs.
- Preserves old-class geometry with a KL penalty and weight importance with an EWC-style penalty.
- Corrects classifier norm drift analytically after training.
- Detects class manifold shift and outputs a four-state deployment decision.

## Quick start

```python
from meld.api import MELDConfig, TrainConfig, run

results = run(
    MELDConfig(
        dataset="CIFAR-10",
        num_tasks=2,
        classes_per_task=5,
        train=TrainConfig(epochs=1, batch_size=32),
    ),
    results_path="results.json",
)
```

```bash
python -m meld.cli --dataset CIFAR-10 --num-tasks 2 --classes-per-task 5 --epochs 1
```

If Continuum is unavailable, MELD falls back to a synthetic class-incremental dataset so the framework remains runnable for development and tests.

## Web dashboard

The dashboard now uses FastAPI and exposes:

- `/` for setup and launch
- `/monitor` for SSE-based live updates
- `/results` for the latest structured run output
