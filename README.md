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
- vision and text benchmark runs with comparable outputs
- a reusable runner, API, and dashboard around those workflows



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

### Web dashboard

```bash
meld-web
```

Then open `http://127.0.0.1:8080`.

The dashboard can:

- launch experiments
- monitor logs and metrics
- show all supported backbones and text encoders
- pre-download supported datasets
- warm selected or all text-model assets before a run

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
