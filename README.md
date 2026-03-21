# MELD

MELD is a continual-learning framework for replay-free model updates on image
and text classification tasks. It gives you a reusable runner, a Python API,
CLI commands, safety checks before updates, drift audits after updates, and a
side-by-side comparison path against full retraining.

MELD is a good fit when you want:

- replay-free continual-learning experiments
- safe incremental updates on new tasks
- direct comparison between delta updates and full retraining
- built-in vision and NLP benchmark support
- custom dataset adapters through Python hooks

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

Check the installed commands:

```bash
meld --help
meld-bootstrap --help
```

Module form works too:

```bash
python -m meld.cli --help
python -m meld.bootstrap --help
```

## CLI usage

Default compare mode on a small synthetic smoke run:

```bash
meld \
  --dataset synthetic \
  --num-tasks 2 \
  --classes-per-task 2 \
  --epochs 1 \
  --batch-size 8 \
  --backbone resnet20 \
  --results-path results.json
```

Delta-only mode:

```bash
meld \
  --dataset synthetic \
  --num-tasks 2 \
  --classes-per-task 2 \
  --run-mode delta \
  --epochs 1 \
  --batch-size 8 \
  --backbone resnet20 \
  --results-path results_delta.json
```

Full-retrain mode:

```bash
meld \
  --dataset synthetic \
  --num-tasks 2 \
  --classes-per-task 2 \
  --run-mode full_retrain \
  --epochs 1 \
  --batch-size 8 \
  --backbone resnet20 \
  --results-path results_full_retrain.json
```

Text benchmark example:

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

## Python API

```python
from meld import MELDConfig, TrainConfig, run

results = run(
    MELDConfig(
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
    ),
    results_path="results.json",
)

print(results["final_summary"])
```

Run modes:

- `compare`: run MELD delta updates and a full-retrain baseline side by side
- `delta`: run only the replay-free update path
- `full_retrain`: always retrain on all seen task data

## Dataset preparation

Download benchmark assets before the first run:

```bash
meld-bootstrap --datasets CIFAR-10 CIFAR-100 --data-root ./data
```

Bootstrap helper coverage:

- `CIFAR-10`
- `CIFAR-100`
- `CIFAR-10-C`

Runner dataset coverage:

- `synthetic`
- `CIFAR-10`
- `CIFAR-100`
- `STL-10`
- `TinyImageNet`
- `AGNews`
- `DBpedia`
- `YahooAnswersNLP`

`TinyImageNet` still expects a manually extracted `tiny-imagenet-200` folder
inside the selected data root.

## Backbones

Available backbone choices:

- `auto`
- `resnet20`
- `resnet32`
- `resnet44`
- `resnet56`
- `resnet18_imagenet`
- `text_encoder`

Common text encoders:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-MiniLM-L12-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/paraphrase-MiniLM-L6-v2`
- `bert-base-uncased`
- `distilbert-base-uncased`

## Custom datasets

You can register your own dataset provider through the Python API. A provider
returns a list of `(train_dataset, test_dataset)` pairs, one pair per
continual-learning task.

```python
import torch
from torch.utils.data import TensorDataset

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

## macOS notes

- MELD automatically uses Apple `mps` when it is available.
- `--prefer-cuda` is only relevant on CUDA-capable NVIDIA setups.
- Keep `--num-workers 0` when you first smoke-test on macOS. That is the
  default and it avoids DataLoader worker-spawn surprises.

## Troubleshooting

- If an image dataset run says `Continuum is required`, install the full package
  dependencies with `pip install .` or `pip install meld-framework`.
- If a text run fails because a model or dataset is missing, retry once with an
  active internet connection so Hugging Face assets can be cached locally.
- If you only want a quick correctness check, start with `synthetic`,
  `--epochs 1`, and `--num-workers 0`.

## Verification

Current release checks used for this package:

- `python -m pytest tests -q`
- `python -m build`
- `python -m twine check dist/*`

A GitHub Actions matrix is also included to exercise install and test flows on
Windows, macOS, and Linux.

For local development tools:

```bash
python -m pip install '.[dev]'
```

## Links

- Homepage: https://github.com/anikeaty08/MELD
- Repository: https://github.com/anikeaty08/MELD.git
- Usage guide: https://github.com/anikeaty08/MELD/blob/main/docs/USAGE.md
