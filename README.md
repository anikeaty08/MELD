# MELD

MELD is a domain-specific framework for continual-learning experiments and
replay-free model updates. It focuses on pre-update safety checks, post-update
drift auditing, and deployment-aware decision outputs for image and text
benchmarks.


It ships with a CLI and includes a React dashboard served by FastAPI for
experiment launch, monitoring, and asset preparation.



MELD is a good fit when you want:

- replay-free continual-learning experiments
- safe incremental updates on new tasks
- the option to compare delta updates against full retraining
- vision and text benchmark runs with comparable outputs
- a reusable runner and API that you can extend with your own datasets



## Install

### From PyPI

```bash
pip install meld-framework
```

### From source

```bash
git clone https://github.com/anikeaty08/MELD.git
cd MELD
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## Quick start

### Python API

```python
from meld import MELDConfig, TrainConfig, run

results = run(
    MELDConfig(
        dataset="CIFAR-10",
        num_tasks=5,
        classes_per_task=2,
        train=TrainConfig(
            backbone="resnet32",
            epochs=5,
            batch_size=64,
            lr=0.01,
        ),
    ),
    results_path="results.json",
)

print(results["final_summary"])
```

You can choose the primary execution path with `run_mode`:

- `compare`: run MELD delta updates and a full-retrain baseline side by side
- `delta`: run only the replay-free update path
- `full_retrain`: always retrain on all seen task data

### CLI

```bash
meld \
  --dataset CIFAR-10 \
  --num-tasks 5 \
  --classes-per-task 2 \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --results-path results.json
```

Equivalent module form:

```bash
python -m meld.cli --help
```

## Custom datasets

MELD ships with built-in benchmark adapters, but you can also register your own
task bundle provider through the Python API. A provider returns a list of
`(train_dataset, test_dataset)` pairs, one pair per continual-learning task.

```python
from torch.utils.data import TensorDataset
import torch

from meld import MELDConfig, TrainConfig, register_dataset, run
from meld.datasets import split_classification_dataset_into_tasks


def my_dataset_provider(config: MELDConfig):
    train = TensorDataset(
        torch.randn(12, 3, 32, 32),
        torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
    )
    test = TensorDataset(
        torch.randn(8, 3, 32, 32),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
    )
    return split_classification_dataset_into_tasks(
        train,
        test,
        num_tasks=config.num_tasks,
        classes_per_task=config.classes_per_task,
    )


register_dataset("MyDataset", my_dataset_provider, overwrite=True)

results = run(
    MELDConfig(
        dataset="MyDataset",
        num_tasks=2,
        classes_per_task=2,
        run_mode="compare",
        train=TrainConfig(backbone="resnet20", epochs=1, batch_size=8),
    )
)
```

## Dataset preparation

If you want datasets cached before the first benchmark, use:

```bash
meld-bootstrap --datasets CIFAR-10 CIFAR-100 --data-root ./data
```

The web dashboard also exposes a preparation flow for:

- `CIFAR-10`
- `CIFAR-100`
- `STL-10`
- `AGNews`
- `DBpedia`
- `YahooAnswersNLP`

`TinyImageNet` is supported by the runner, but it still expects a manually
extracted `tiny-imagenet-200` folder under the selected data root.

## Supported datasets

- `synthetic`
- `CIFAR-10`
- `CIFAR-100`
- `STL-10`
- `TinyImageNet`
- `AGNews`
- `DBpedia`
- `YahooAnswersNLP`

## Supported backbones

- `auto`
- `resnet20`
- `resnet32`
- `resnet44`
- `resnet56`
- `resnet18_imagenet`
- `text_encoder`

Supported text encoders include:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-MiniLM-L12-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/paraphrase-MiniLM-L6-v2`
- `bert-base-uncased`
- `distilbert-base-uncased`

## Project layout

```text
meld/
|- api.py
|- datasets.py
|- bootstrap.py
|- cli.py
|- delta.py
|- modeling.py
|- benchmarks/
|- core/
|- interfaces/
|- models/
`- web/
```

## Development

Install dev tools with:

```bash
pip install .[dev]
```

Run unit tests with:

```bash
python -m pytest tests/unit -q
```

## Links

- Homepage: https://github.com/anikeaty08/MELD
- Repository: https://github.com/anikeaty08/MELD.git
- Usage guide: https://github.com/anikeaty08/MELD/blob/main/docs/USAGE.md
