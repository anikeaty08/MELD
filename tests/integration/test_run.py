from pathlib import Path

from meld.api import MELDConfig, TrainConfig, run


def test_run_completes_with_synthetic_data(tmp_path: Path):
    results = run(
        MELDConfig(
            dataset="synthetic",
            num_tasks=2,
            classes_per_task=2,
            bound_tolerance=10.0,
            train=TrainConfig(epochs=1, batch_size=8, backbone="resnet20"),
        ),
        results_path=str(tmp_path / "results.json"),
    )
    assert results["status"] == "completed"
    assert len(results["tasks"]) == 2
    assert (tmp_path / "results.json").exists()
