# Usage

Repository: [anikeaty08/MELD](https://github.com/anikeaty08/MELD)

## Virtual Environment

Use MELD from the project virtual environment.

### Windows PowerShell

```powershell
git clone https://github.com/anikeaty08/MELD.git
cd MELD
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
python -m meld.bootstrap --datasets CIFAR-10 CIFAR-100 --data-root ./data
```

### Windows CMD

```bat
git clone https://github.com/anikeaty08/MELD.git
cd MELD
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install .
python -m meld.bootstrap --datasets CIFAR-10 CIFAR-100 --data-root ./data
```

### macOS / Linux

```bash
git clone https://github.com/anikeaty08/MELD.git
cd MELD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m meld.bootstrap --datasets CIFAR-10 CIFAR-100 --data-root ./data
```

Activation alone does not install dependencies. The install happens when you run
`python -m pip install .`.

To exit:

```bash
deactivate
```

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

- `CIFAR-10` and `CIFAR-100` use Continuum from the active virtual environment.
- Run `python -m meld.bootstrap --datasets CIFAR-10 CIFAR-100 --data-root ./data` after install if you want the datasets ready before the first benchmark.
- Other dataset names currently fall back to the synthetic incremental dataset for development.
- The synthetic path is intentionally useful for smoke tests, API checks, and dashboard wiring.

<div align="center">
  <h3>with love Anikeat ❤️</h3>
</div>
