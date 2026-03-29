# Usage

## Python API

### Basic continual learning loop

```python
from delta import (
    DeltaStream, FisherDeltaStrategy, EvaluationPlugin,
    accuracy_metrics, equivalence_metrics, calibration_metrics,
    compute_metrics, InteractiveLogger
)
import torch.nn as nn
from torch.optim import SGD

# Any nn.Module works
model = nn.Sequential(
    nn.Linear(32, 64), nn.ReLU(),
    nn.Linear(64, 10)
)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Create a stream of tasks
stream = DeltaStream(
    dataset_name="synthetic",
    n_tasks=5,
    scenario="class_incremental",
    classes_per_task=2,
)

# Setup evaluation
eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    equivalence_metrics(epsilon=True, kl_bound=True),
    calibration_metrics(ece_before=True, ece_after=True),
    compute_metrics(savings_ratio=True),
    loggers=[InteractiveLogger()],
)

# Create strategy
strategy = FisherDeltaStrategy(
    model, optimizer, nn.CrossEntropyLoss(),
    evaluator=eval_plugin,
    train_epochs=5,
    train_mb_size=32,
)

# Train across tasks
for experience in stream.train_stream:
    strategy.train(experience)
    strategy.eval(stream.test_stream)

# Inspect certificate
cert = strategy.last_certificate
print(cert.summary())
print(f"Equivalent: {cert.is_equivalent}")
print(f"Speedup: {cert.compute_ratio:.1f}x")
```

### Using CNNs

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10),
)

# KFAC captures both Conv2d and Linear layers automatically
strategy = FisherDeltaStrategy(model, optimizer, criterion)
```

### Using MELD's ResNet backbones

```python
from delta.demos.models import resnet20, IncrementalClassifier, MELDModel

backbone = resnet20()
classifier = IncrementalClassifier(backbone.out_dim)
classifier.adaption(5)  # 5 classes for first task
model = MELDModel(backbone, classifier)
```

### Comparing strategies

```python
from delta import FisherDeltaStrategy, FullRetrainStrategy

# Same model architecture, different strategies
delta_strategy = FisherDeltaStrategy(model_a, opt_a, criterion)
full_strategy = FullRetrainStrategy(model_b, opt_b, criterion)

for exp in stream.train_stream:
    delta_strategy.train(exp)
    full_strategy.train(exp)
```

### Custom plugins

```python
class GradientMonitor:
    def after_backward(self, strategy):
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in strategy.model.parameters()
            if p.grad is not None
        ) ** 0.5
        print(f"Grad norm: {total_norm:.4f}")

strategy.add_plugin(GradientMonitor())
```

## CLI

```bash
# Compare delta vs full retrain
python -m delta.demos.cli \
    --dataset synthetic \
    --num-tasks 3 \
    --classes-per-task 2 \
    --run-mode compare \
    --epochs 2 \
    --batch-size 16

# Delta only
python -m delta.demos.cli --run-mode fisher_delta

# Full retrain only
python -m delta.demos.cli --run-mode full_retrain

# All options
python -m delta.demos.cli --help
```

## Run modes

| Mode | What it does |
|---|---|
| `compare` | Runs both FisherDelta and FullRetrain, reports side-by-side |
| `fisher_delta` | FisherDelta only with full metrics |
| `full_retrain` | FullRetrain baseline only |

## Supported datasets

The `DeltaStream` class handles dataset loading:

| Dataset | Type | Needs extras |
|---|---|---|
| `synthetic` | Flat vectors | None |
| `CIFAR-10` | Images | `torchvision`, `continuum` |
| `CIFAR-100` | Images | `torchvision`, `continuum` |
| `TinyImageNet` | Images | `torchvision`, `continuum` |
| `AGNews` | Text | `transformers`, `datasets` |

If a real dataset can't be loaded, DeltaStream falls back to synthetic data automatically.

## Platform notes

- **Linux**: full support, CUDA auto-detected
- **macOS**: full support, MPS auto-detected on Apple Silicon
- **Windows**: full support, CUDA auto-detected
- Always safe to start with `--num-workers 0`
