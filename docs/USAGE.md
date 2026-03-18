# Usage

## Python API

```python
from meld.api import MELDConfig, TrainConfig, run

config = MELDConfig(
    dataset="synthetic",
    num_tasks=2,
    classes_per_task=2,
    bound_tolerance=10.0,
    train=TrainConfig(
        backbone="resnet20",
        epochs=1,
        batch_size=8,
    ),
)

results = run(config, results_path="results.json")
print(results["final_summary"])
```

## CLI

```bash
python -m meld.cli \
  --dataset synthetic \
  --num-tasks 2 \
  --classes-per-task 2 \
  --epochs 1 \
  --batch-size 8 \
  --backbone resnet20 \
  --results-path results.json
```

## Web

```bash
python -m meld.web.server
```

Then open [http://127.0.0.1:8080](http://127.0.0.1:8080).

## Dataset behavior

- `CIFAR-10` and `CIFAR-100` try to use Continuum if it is installed.
- Other dataset names currently fall back to the synthetic incremental dataset for development.
- The synthetic path is intentionally useful for smoke tests, API checks, and dashboard wiring.
