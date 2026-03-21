import json
import subprocess
import sys
import uuid
from pathlib import Path


def _workspace_result_path(name: str) -> Path:
    root = Path.cwd() / "tests" / ".tmp_results"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}_{uuid.uuid4().hex}.json"


def test_cli_module_help_succeeds():
    result = subprocess.run(
        [sys.executable, "-m", "meld.cli", "--help"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run MELD continual-learning experiments." in result.stdout
    assert "--run-mode" in result.stdout


def test_bootstrap_module_help_succeeds():
    result = subprocess.run(
        [sys.executable, "-m", "meld.bootstrap", "--help"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Bootstrap MELD datasets and dependency checks." in result.stdout


def test_cli_module_runs_synthetic_smoke():
    results_path = _workspace_result_path("cli_smoke_results")
    database_path = _workspace_result_path("cli_smoke_db").with_suffix(".sqlite")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "meld.cli",
            "--dataset",
            "synthetic",
            "--num-tasks",
            "1",
            "--classes-per-task",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--backbone",
            "resnet20",
            "--num-workers",
            "0",
            "--results-path",
            str(results_path),
            "--database-path",
            str(database_path),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert results_path.exists()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["final_summary"]["run_mode"] == "compare"
