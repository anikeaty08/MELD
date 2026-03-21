"""FastAPI dashboard server for MELD."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from web.assets import inspect_workspace_readiness, prepare_workspace
from web.catalog import build_options_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "web" / "static"
INDEX_PATH = STATIC_DIR / "index.html"
RESULTS_PATH = PROJECT_ROOT / "results.json"
LOG_PATH = PROJECT_ROOT / "experiment.log"
DB_PATH = PROJECT_ROOT / "meld_results.db"

FRONTEND_FALLBACK = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MELD Dashboard</title>
</head>
<body style="font-family: Segoe UI, sans-serif; padding: 2rem;">
  <h1>MELD Dashboard</h1>
  <p>The React frontend has not been built yet.</p>
  <p>Run <code>npm install</code> and <code>npm run build</code> in <code>web/frontend</code>.</p>
</body>
</html>
"""


@dataclass
class ExperimentState:
    running: bool = False
    preparing: bool = False
    process: subprocess.Popen[str] | None = None
    error: str | None = None
    prepare_report: dict[str, Any] | None = None


app = FastAPI(title="MELD Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
STATE = ExperimentState()


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_path(raw: Any, default: Path) -> Path:
    path = Path(str(raw)) if raw else default
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _frontend_response() -> FileResponse | HTMLResponse:
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return HTMLResponse(FRONTEND_FALLBACK)


def _sync_process() -> None:
    if STATE.process is None:
        STATE.running = False
        return
    code = STATE.process.poll()
    if code is None:
        STATE.running = True
        return
    STATE.running = False
    if code != 0:
        STATE.error = f"MELD process exited with code {code}"
    STATE.process = None


def _build_run_command(payload: dict[str, Any]) -> list[str]:
    dataset = str(payload.get("dataset", "CIFAR-10"))
    results_path = _resolve_path(payload.get("results_path"), RESULTS_PATH)
    database_path = _resolve_path(payload.get("database_path"), DB_PATH)
    data_root = _resolve_path(payload.get("data_root"), PROJECT_ROOT / "data")
    command = [
        sys.executable,
        "-m",
        "meld.cli",
        "--dataset",
        dataset,
        "--num-tasks",
        str(int(payload.get("num_tasks", 2))),
        "--classes-per-task",
        str(int(payload.get("classes_per_task", 5))),
        "--epochs",
        str(int(payload.get("epochs", 5))),
        "--batch-size",
        str(int(payload.get("batch_size", 64))),
        "--lr",
        str(float(payload.get("lr", 0.1))),
        "--backbone",
        str(payload.get("backbone", "auto")),
        "--text-encoder-model",
        str(payload.get("text_encoder_model", "sentence-transformers/all-MiniLM-L6-v2")),
        "--bound-tolerance",
        str(float(payload.get("bound_tolerance", 10.0))),
        "--pac-gate-tolerance",
        str(float(payload.get("pac_gate_tolerance", 0.5))),
        "--mixup-alpha",
        str(float(payload.get("mixup_alpha", 0.2))),
        "--num-workers",
        str(int(payload.get("num_workers", 0))),
        "--data-root",
        str(data_root),
        "--database-path",
        str(database_path),
        "--results-path",
        str(results_path),
    ]
    if _normalize_bool(payload.get("pretrained_backbone", False)):
        command.append("--pretrained-backbone")
    if _normalize_bool(payload.get("prefer_cuda", False)):
        command.append("--prefer-cuda")
    return command


def _prepare_from_payload(payload: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    data_root = _resolve_path(payload.get("data_root"), PROJECT_ROOT / "data")
    prepare_payload = payload.get("prepare")
    if not isinstance(prepare_payload, dict):
        prepare_payload = {}
    if prepare_payload.get("all_datasets"):
        options = build_options_payload()
        datasets = [item["id"] for item in options["datasets"]]
    else:
        datasets = prepare_payload.get("datasets") or [payload.get("dataset", "CIFAR-10")]
    text_models = prepare_payload.get("text_models") or []
    if (
        str(payload.get("backbone", "auto")) == "text_encoder"
        and payload.get("text_encoder_model")
        and not prepare_payload.get("skip_selected_text_model")
    ):
        text_models = [*text_models, str(payload["text_encoder_model"])]
    normalized = {
        "install_requirements": _normalize_bool(prepare_payload.get("install_requirements", False)),
        "datasets": [str(item) for item in datasets if item],
        "backbone": str(payload.get("backbone", "auto")),
        "pretrained_backbone": _normalize_bool(payload.get("pretrained_backbone", False)),
        "text_models": [str(item) for item in text_models if item],
    }
    return data_root, normalized


@app.get("/", response_model=None)
def index():
    return _frontend_response()


@app.get("/monitor", response_model=None)
def monitor():
    return _frontend_response()


@app.get("/api/options")
def options() -> JSONResponse:
    return JSONResponse(build_options_payload())


@app.get("/api/readiness")
def readiness(data_root: str | None = None) -> JSONResponse:
    resolved_data_root = _resolve_path(data_root, PROJECT_ROOT / "data")
    return JSONResponse(inspect_workspace_readiness(PROJECT_ROOT, resolved_data_root))


@app.post("/api/prepare")
def prepare(payload: dict[str, Any]) -> JSONResponse:
    _sync_process()
    if STATE.running:
        raise HTTPException(status_code=409, detail="Cannot prepare assets while an experiment is running.")
    if STATE.preparing:
        raise HTTPException(status_code=409, detail="A workspace preparation is already in progress.")

    data_root, prepare_payload = _prepare_from_payload(payload)
    STATE.preparing = True
    STATE.error = None
    try:
        report = prepare_workspace(
            project_root=PROJECT_ROOT,
            data_root=data_root,
            install_requirements=prepare_payload["install_requirements"],
            datasets=prepare_payload["datasets"],
            backbone=prepare_payload["backbone"],
            pretrained_backbone=prepare_payload["pretrained_backbone"],
            text_models=prepare_payload["text_models"],
        )
        STATE.prepare_report = report
        return JSONResponse(
            {
                "report": report,
                "readiness": inspect_workspace_readiness(PROJECT_ROOT, data_root),
            }
        )
    except Exception as exc:
        STATE.error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        STATE.preparing = False


@app.get("/results")
def results() -> JSONResponse:
    _sync_process()
    if not RESULTS_PATH.exists():
        return JSONResponse({"status": "no_results"})
    return JSONResponse(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))


@app.get("/results/db")
def results_db() -> JSONResponse:
    if not DB_PATH.exists():
        return JSONResponse({"status": "no_database"})
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT run_id, status, config_json, final_summary_json, payload_json FROM runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return JSONResponse({"status": "empty_database"})
        run_id, status, config_json, final_summary_json, payload_json = row
        tasks = conn.execute(
            """
            SELECT task_id, delta_json, full_retrain_json, oracle_json, drift_json, decision_json,
                   snapshot_json, train_json, cil_metrics_json, equivalence_gap, forgetting,
                   compute_savings_percent
            FROM tasks
            WHERE run_id = ?
            ORDER BY task_id ASC
            """,
            (run_id,),
        ).fetchall()
    return JSONResponse(
        {
            "status": status,
            "run_id": run_id,
            "config": json.loads(config_json) if config_json else None,
            "final_summary": json.loads(final_summary_json) if final_summary_json else None,
            "tasks": [
                {
                    "task_id": task_id,
                    "delta": json.loads(delta_json) if delta_json else None,
                    "full_retrain": json.loads(full_json) if full_json else None,
                    "oracle": json.loads(oracle_json) if oracle_json else None,
                    "drift": json.loads(drift_json) if drift_json else None,
                    "decision": json.loads(decision_json) if decision_json else None,
                    "snapshot": json.loads(snapshot_json) if snapshot_json else None,
                    "train": json.loads(train_json) if train_json else None,
                    "cil_metrics": json.loads(cil_json) if cil_json else None,
                    "equivalence_gap": equivalence_gap,
                    "forgetting": forgetting,
                    "compute_savings_percent": compute_savings_percent,
                }
                for (
                    task_id,
                    delta_json,
                    full_json,
                    oracle_json,
                    drift_json,
                    decision_json,
                    snapshot_json,
                    train_json,
                    cil_json,
                    equivalence_gap,
                    forgetting,
                    compute_savings_percent,
                ) in tasks
            ],
            "payload": json.loads(payload_json),
        }
    )


@app.get("/results/csv")
def results_csv() -> StreamingResponse:
    if not RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="No results.json found.")
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    lines = [
        "task_id,delta_top1,full_top1,risk_estimate_pre,drift_realized_post,risk_estimate_held,bound_is_formal,compute_savings_percent"
    ]
    for task in tasks:
        delta = task.get("delta", {}) if isinstance(task.get("delta"), dict) else {}
        full = task.get("full_retrain", {}) if isinstance(task.get("full_retrain"), dict) else {}
        oracle = task.get("oracle", {}) if isinstance(task.get("oracle"), dict) else {}
        lines.append(
            ",".join(
                [
                    str(task.get("task_id", "")),
                    str(delta.get("top1", "")),
                    str(full.get("top1", "")),
                    str(oracle.get("risk_estimate_pre", "")),
                    str(oracle.get("drift_realized_post", "")),
                    str(oracle.get("risk_estimate_held", "")),
                    str(oracle.get("bound_is_formal", "")),
                    str(task.get("compute_savings_percent", "")),
                ]
            )
        )
    data = ("\n".join(lines) + "\n").encode("utf-8")
    return StreamingResponse(
        iter([data]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=meld_results.csv"},
    )


@app.get("/api/state")
def state() -> JSONResponse:
    _sync_process()
    logs = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    return JSONResponse(
        {
            "running": STATE.running,
            "preparing": STATE.preparing,
            "error": STATE.error,
            "prepareReport": STATE.prepare_report,
            "results": _read_json(RESULTS_PATH),
            "logs": logs,
        }
    )


@app.get("/api/state/stream")
def state_stream() -> StreamingResponse:
    def event_stream():
        last_payload = None
        for _ in range(120):
            _sync_process()
            payload = json.dumps(
                {
                    "running": STATE.running,
                    "preparing": STATE.preparing,
                    "error": STATE.error,
                    "prepareReport": STATE.prepare_report,
                    "results": _read_json(RESULTS_PATH),
                }
            )
            if payload != last_payload:
                last_payload = payload
                yield f"data: {payload}\n\n"
            time.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/run")
def launch(payload: dict[str, Any]) -> JSONResponse:
    _sync_process()
    if STATE.running:
        raise HTTPException(status_code=409, detail="An experiment is already running.")
    if STATE.preparing:
        raise HTTPException(status_code=409, detail="Wait for workspace preparation to finish first.")

    prepare_report = None
    should_prepare = isinstance(payload.get("prepare"), dict) and _normalize_bool(payload["prepare"].get("enabled"))
    if should_prepare:
        data_root, prepare_payload = _prepare_from_payload(payload)
        STATE.preparing = True
        try:
            prepare_report = prepare_workspace(
                project_root=PROJECT_ROOT,
                data_root=data_root,
                install_requirements=prepare_payload["install_requirements"],
                datasets=prepare_payload["datasets"],
                backbone=prepare_payload["backbone"],
                pretrained_backbone=prepare_payload["pretrained_backbone"],
                text_models=prepare_payload["text_models"],
            )
            STATE.prepare_report = prepare_report
        finally:
            STATE.preparing = False
        if prepare_report and not prepare_report["success"]:
            STATE.error = "Workspace preparation failed."
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Workspace preparation failed.",
                    "report": prepare_report,
                },
            )

    STATE.error = None
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = _build_run_command(payload)
    log_handle = LOG_PATH.open("w", encoding="utf-8")
    STATE.process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    STATE.running = True
    return JSONResponse(
        {
            "started": True,
            "pid": STATE.process.pid,
            "command": command,
            "prepare": prepare_report,
        }
    )


@app.post("/api/stop")
def stop() -> JSONResponse:
    _sync_process()
    if not STATE.running:
        return JSONResponse({"stopped": False, "message": "No experiment is currently running."})
    assert STATE.process is not None
    STATE.process.terminate()
    try:
        STATE.process.wait(timeout=5)
    except Exception:
        STATE.process.kill()
    STATE.process = None
    STATE.running = False
    return JSONResponse({"stopped": True})


def main() -> None:
    uvicorn.run("web.server:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
