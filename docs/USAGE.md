# Usage

MELD is a continual-learning framework for replay-free incremental-update
workflows. The commands below assume that scope: class-incremental benchmarks,
safe update checks, and experiment monitoring for supported vision and NLP
tasks.

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

Web dashboard:

```bash
meld-web
```

Module forms are also available:

```bash
python -m meld.cli --help
python -m meld.bootstrap --help
python -m meld.web.server
```

## Python API example

```python
from meld import MELDConfig, TrainConfig, run

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

## CLI example

```bash
meld \
  --dataset AGNews \
  --num-tasks 1 \
  --classes-per-task 4 \
  --epochs 1 \
  --batch-size 8 \
  --backbone text_encoder \
  --text-encoder-model sentence-transformers/all-MiniLM-L6-v2 \
  --results-path results.json
```

## Dashboard workflow

The React dashboard at `http://127.0.0.1:8080` can:

- launch new runs
- stream logs and metrics
- show all supported backbones and text encoders
- install requirements from the current environment
- pre-download supported datasets
- warm model assets before launch

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

Dashboard preparation coverage:

- `CIFAR-10`
- `CIFAR-100`
- `STL-10`
- `AGNews`
- `DBpedia`
- `YahooAnswersNLP`

`TinyImageNet` still requires a manually extracted `tiny-imagenet-200` folder
inside the chosen data root.
