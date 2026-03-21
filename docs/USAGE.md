# Usage

MELD is a continual-learning framework for replay-free incremental-update
workflows. The commands below focus on the package API, CLI, supported run
modes, and custom dataset extension points.

## Install

### PyPI

```bash
pip install meld-framework
```

### Source checkout

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

## Main entry points

CLI:

```bash
meld --help
```

Bootstrap helper:

```bash
meld-bootstrap --help
```

Module forms are also available:

```bash
python -m meld.cli --help
python -m meld.bootstrap --help
```

## Python API example

```python
from meld import MELDConfig, TrainConfig, run

config = MELDConfig(
    dataset="synthetic",
    num_tasks=2,
    classes_per_task=2,
    run_mode="compare",
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

## Run modes

MELD exposes three primary run modes:

- `compare`: run the replay-free delta update path and a full-retrain baseline
- `delta`: run only MELD's incremental update path
- `full_retrain`: always retrain on all seen task data

CLI example:

```bash
meld --dataset synthetic --num-tasks 2 --classes-per-task 2 --run-mode full_retrain
```

Python API example:

```python
config = MELDConfig(dataset="synthetic", run_mode="delta")
```

## CLI example

```bash
meld \
  --dataset synthetic \
  --num-tasks 2 \
  --classes-per-task 2 \
  --run-mode compare \
  --epochs 1 \
  --batch-size 8 \
  --backbone resnet20 \
  --num-workers 0 \
  --results-path results.json
```

NLP example:

```bash
meld \
  --dataset AGNews \
  --num-tasks 1 \
  --classes-per-task 4 \
  --epochs 1 \
  --batch-size 8 \
  --backbone text_encoder \
  --text-encoder-model sentence-transformers/all-MiniLM-L6-v2 \
  --num-workers 0 \
  --results-path results_agnews.json
```

## Dataset notes

Supported runner datasets:

- `synthetic`
- `CIFAR-10`
- `CIFAR-100`
- `STL-10`
- `TinyImageNet`
- `AGNews`
- `DBpedia`
- `YahooAnswersNLP`

Bootstrap helper coverage:

- `CIFAR-10`
- `CIFAR-100`
- `CIFAR-10-C`

`TinyImageNet` still requires a manually extracted `tiny-imagenet-200` folder
inside the chosen data root.

## macOS notes

- MELD auto-detects Apple `mps` when it is available.
- `--prefer-cuda` only applies to CUDA-capable NVIDIA systems.
- Keep `--num-workers 0` for initial smoke tests on macOS.

## Custom dataset adapters

You can register dataset providers when you want MELD to operate on your own
data source instead of only the built-in benchmark names.

The provider contract is:

- input: `MELDConfig`
- output: a list of `(train_dataset, test_dataset)` pairs
- each dataset must implement `__len__` and `__getitem__`

Example:

```python
import torch
from torch.utils.data import TensorDataset

from meld import MELDConfig, register_dataset
from meld.datasets import split_classification_dataset_into_tasks


def custom_provider(config: MELDConfig):
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


register_dataset("CustomImages", custom_provider, overwrite=True)
```
