from pathlib import Path

from fastapi.testclient import TestClient

import meld.web.server as server


def _reset_state() -> None:
    server.STATE.running = False
    server.STATE.preparing = False
    server.STATE.process = None
    server.STATE.error = None
    server.STATE.prepare_report = None


def test_options_endpoint_lists_dataset_and_model_choices():
    client = TestClient(server.app)

    response = client.get("/api/options")

    assert response.status_code == 200
    payload = response.json()
    assert any(item["id"] == "AGNews" for item in payload["datasets"])
    assert any(item["id"] == "text_encoder" for item in payload["backbones"])
    assert any(item["id"] == "sentence-transformers/all-MiniLM-L6-v2" for item in payload["textModels"])


def test_build_run_command_includes_web_form_fields():
    tmp_path = Path.cwd()
    command = server._build_run_command(
        {
            "dataset": "AGNews",
            "num_tasks": 1,
            "classes_per_task": 4,
            "epochs": 2,
            "batch_size": 8,
            "lr": 0.05,
            "backbone": "text_encoder",
            "pretrained_backbone": True,
            "text_encoder_model": "sentence-transformers/all-mpnet-base-v2",
            "bound_tolerance": 12.0,
            "pac_gate_tolerance": 0.5,
            "mixup_alpha": 0.3,
            "num_workers": 2,
            "data_root": str(tmp_path / "data"),
            "database_path": str(tmp_path / "db.sqlite"),
            "results_path": str(tmp_path / "results.json"),
            "prefer_cuda": True,
        }
    )

    assert "--dataset" in command and "AGNews" in command
    assert "--text-encoder-model" in command and "sentence-transformers/all-mpnet-base-v2" in command
    assert "--mixup-alpha" in command and "0.3" in command
    assert "--pretrained-backbone" in command
    assert "--prefer-cuda" in command


def test_prepare_endpoint_delegates_to_workspace_preparer(monkeypatch):
    tmp_path = Path.cwd()
    _reset_state()
    captured = {}

    def fake_prepare_workspace(**kwargs):
        captured.update(kwargs)
        return {"success": True, "datasets": [], "models": []}

    def fake_readiness(project_root: Path, data_root: Path):
        return {"requirements": {"ready": True}, "datasets": [], "textModels": [], "backbones": []}

    monkeypatch.setattr(server, "prepare_workspace", fake_prepare_workspace)
    monkeypatch.setattr(server, "inspect_workspace_readiness", fake_readiness)
    client = TestClient(server.app)

    response = client.post(
        "/api/prepare",
        json={
            "dataset": "AGNews",
            "backbone": "text_encoder",
            "text_encoder_model": "bert-base-uncased",
            "data_root": str(tmp_path / "data"),
            "prepare": {
                "install_requirements": True,
                "datasets": ["AGNews"],
                "text_models": ["distilbert-base-uncased"],
            },
        },
    )

    assert response.status_code == 200
    assert captured["install_requirements"] is True
    assert captured["datasets"] == ["AGNews"]
    assert captured["text_models"] == ["distilbert-base-uncased", "bert-base-uncased"]


def test_run_endpoint_can_prepare_before_launch(monkeypatch):
    tmp_path = Path.cwd()
    _reset_state()

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 4321

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    def fake_prepare_workspace(**kwargs):
        return {"success": True, "datasets": kwargs["datasets"], "models": kwargs["text_models"]}

    monkeypatch.setattr(server, "prepare_workspace", fake_prepare_workspace)
    monkeypatch.setattr(server.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(server, "LOG_PATH", tmp_path / "experiment_web_test.log")
    monkeypatch.setattr(server, "RESULTS_PATH", tmp_path / "results_web_test.json")
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "meld_results_web_test.db")
    client = TestClient(server.app)

    response = client.post(
        "/api/run",
        json={
            "dataset": "CIFAR-10",
            "num_tasks": 2,
            "classes_per_task": 5,
            "epochs": 1,
            "batch_size": 8,
            "lr": 0.1,
            "backbone": "resnet32",
            "pretrained_backbone": True,
            "text_encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "data_root": str(tmp_path / "data"),
            "prepare": {
                "enabled": True,
                "install_requirements": False,
                "datasets": ["CIFAR-10"],
                "text_models": [],
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["started"] is True
    assert payload["prepare"]["datasets"] == ["CIFAR-10"]
    assert "--pretrained-backbone" in payload["command"]
