"""FastAPI dashboard for MELD."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import subprocess
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

ROOT_TEMPLATE = """
<!doctype html>
<html>
<head><title>MELD</title></head>
<body>
  <h1>MELD Runner</h1>
  <form id="run-form">
    <label>Dataset <input name="dataset" value="synthetic"></label><br>
    <label>Tasks <input name="num_tasks" type="number" value="2"></label><br>
    <label>Classes per task <input name="classes_per_task" type="number" value="2"></label><br>
    <label>Epochs <input name="epochs" type="number" value="1"></label><br>
    <label>Batch size <input name="batch_size" type="number" value="8"></label><br>
    <label>Backbone <input name="backbone" value="resnet20"></label><br>
    <button type="submit">Launch</button>
  </form>
  <p><a href="/monitor">Monitor</a> | <a href="/results">Results</a></p>
  <pre id="status"></pre>
  <script>
    const form = document.getElementById("run-form");
    const status = document.getElementById("status");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = Object.fromEntries(new FormData(form).entries());
      payload.num_tasks = Number(payload.num_tasks);
      payload.classes_per_task = Number(payload.classes_per_task);
      payload.epochs = Number(payload.epochs);
      const response = await fetch("/api/run", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      status.textContent = JSON.stringify(await response.json(), null, 2);
    });
  </script>
</body>
</html>
"""
MONITOR_TEMPLATE = """
<!doctype html>
<html>
<head><title>MELD Monitor</title></head>
<body>
  <h1>Live Monitor</h1>
  <pre id="stream">Waiting for updates...</pre>
  <script>
    const target = document.getElementById("stream");
    const source = new EventSource("/api/state/stream");
    source.onmessage = (event) => { target.textContent = event.data; };
  </script>
</body>
</html>
"""

RESULTS_PATH = Path("results.json")
LOG_PATH = Path("experiment.log")


@dataclass
class ExperimentState:
    running: bool = False
    process: subprocess.Popen[str] | None = None
    error: str | None = None


app = FastAPI(title="MELD Dashboard")
STATE = ExperimentState()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
    return [
        sys.executable,
        "-m",
        "meld.cli",
        "--dataset",
        str(payload.get("dataset", "synthetic")),
        "--num-tasks",
        str(int(payload.get("num_tasks", 2))),
        "--classes-per-task",
        str(int(payload.get("classes_per_task", 2))),
        "--epochs",
        str(int(payload.get("epochs", 1))),
        "--batch-size",
        str(int(payload.get("batch_size", 8))),
        "--backbone",
        str(payload.get("backbone", "resnet20")),
        "--results-path",
        str(RESULTS_PATH),
    ]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return ROOT_TEMPLATE


@app.get("/monitor", response_class=HTMLResponse)
def monitor() -> str:
    return MONITOR_TEMPLATE


@app.get("/results")
def results() -> JSONResponse:
    _sync_process()
    if not RESULTS_PATH.exists():
        return JSONResponse({"status": "no_results"})
    return JSONResponse(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))


@app.get("/api/state")
def state() -> JSONResponse:
    _sync_process()
    logs = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    return JSONResponse(
        {
            "running": STATE.running,
            "error": STATE.error,
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
                    "error": STATE.error,
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
        cwd=str(Path.cwd()),
    )
    STATE.running = True
    return JSONResponse({"started": True, "pid": STATE.process.pid, "command": command})


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
    uvicorn.run("meld.web.server:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
