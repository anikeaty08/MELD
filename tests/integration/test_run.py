from pathlib import Path
import uuid

from meld.api import MELDConfig, TrainConfig, run


def _workspace_result_path(name: str) -> Path:
    root = Path.cwd() / "tests" / ".tmp_results"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}_{uuid.uuid4().hex}.json"


def test_run_completes_with_synthetic_data():
    path = _workspace_result_path("results")
    results = run(
        MELDConfig(
            dataset="synthetic",
            num_tasks=2,
            classes_per_task=2,
            bound_tolerance=10.0,
            train=TrainConfig(epochs=1, batch_size=8, backbone="resnet20"),
        ),
        results_path=str(path),
    )
    assert results["status"] == "completed"
    assert len(results["tasks"]) == 2
    assert all(entry["full_retrain"]["skipped"] is False for entry in results["epoch_history"])
    assert path.exists()


def test_run_completes_with_frozen_analytic_strategy():
    path = _workspace_result_path("results_analytic")
    results = run(
        MELDConfig(
            dataset="synthetic",
            num_tasks=2,
            classes_per_task=2,
            bound_tolerance=10.0,
            pac_gate_tolerance=1.0,
            train=TrainConfig(
                epochs=1,
                base_epochs=1,
                batch_size=8,
                backbone="resnet20",
                incremental_strategy="frozen_analytic",
            ),
        ),
        results_path=str(path),
    )
    assert results["status"] == "completed"
    assert len(results["tasks"]) == 2
    assert results["tasks"][1]["train"]["epochs_run"] == 1
    assert path.exists()


def test_run_supports_full_retrain_mode():
    path = _workspace_result_path("results_full_retrain")
    results = run(
        MELDConfig(
            dataset="synthetic",
            run_mode="full_retrain",
            num_tasks=2,
            classes_per_task=2,
            bound_tolerance=10.0,
            train=TrainConfig(
                epochs=1,
                full_retrain_epochs=1,
                batch_size=8,
                backbone="resnet20",
            ),
        ),
        results_path=str(path),
    )

    assert results["status"] == "completed"
    assert all(task["decision"]["state"] == "FULL_RETRAIN" for task in results["tasks"])
    assert all(task["delta"]["top1"] is None for task in results["tasks"])
    assert all(task["full_retrain"]["top1"] is not None for task in results["tasks"])
    assert all(entry["delta"]["skipped"] is True for entry in results["epoch_history"])


def test_run_supports_delta_only_mode():
    path = _workspace_result_path("results_delta_only")
    results = run(
        MELDConfig(
            dataset="synthetic",
            run_mode="delta",
            num_tasks=2,
            classes_per_task=2,
            bound_tolerance=10.0,
            train=TrainConfig(
                epochs=1,
                batch_size=8,
                backbone="resnet20",
            ),
        ),
        results_path=str(path),
    )

    assert results["status"] == "completed"
    assert results["final_summary"]["run_mode"] == "delta"
    assert all(task["full_retrain"]["top1"] is None for task in results["tasks"])
    assert all(entry["full_retrain"]["skipped"] is True for entry in results["epoch_history"])
