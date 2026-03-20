"""FastAPI dashboard for MELD."""

from __future__ import annotations

import json
import sqlite3
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
    <label>Pretrained backbone <input name="pretrained_backbone" type="checkbox"></label><br>
    <button type="submit">Launch</button>
  </form>
  <p><a href="/monitor">Monitor</a> | <a href="/results">Results JSON</a> | <a href="/results/db">Results DB</a> | <a href="/results/csv">Export CSV</a></p>
  <h3>Bound Timeline</h3>
  <canvas id="bound-chart" width="800" height="240" style="border:1px solid #ddd"></canvas>
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
      payload.pretrained_backbone = form.elements.pretrained_backbone.checked;
      const response = await fetch("/api/run", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      status.textContent = JSON.stringify(await response.json(), null, 2);
    });
    async function refreshBoundChart() {
      const response = await fetch("/results");
      const data = await response.json();
      const timeline = Array.isArray(data.bounds_timeline) ? data.bounds_timeline : [];
      const canvas = document.getElementById("bound-chart");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!timeline.length) {
        ctx.fillText("No bound data yet.", 10, 24);
        return;
      }
      const xs = timeline.map(p => Number(p.task_id));
      const ys1 = timeline.map(p => Number(p.epsilon_max || 0));
      const ys2 = timeline.map(p => Number(p.epsilon_actual || 0));
      const yMax = Math.max(1e-6, ...ys1, ...ys2);
      const margin = 30;
      const w = canvas.width - margin * 2;
      const h = canvas.height - margin * 2;
      function xPx(i) { return margin + (w * i / Math.max(1, xs.length - 1)); }
      function yPx(y) { return margin + h - (h * y / yMax); }
      ctx.strokeStyle = "#999";
      ctx.beginPath();
      ctx.moveTo(margin, margin);
      ctx.lineTo(margin, margin + h);
      ctx.lineTo(margin + w, margin + h);
      ctx.stroke();
      function drawLine(vals, color) {
        ctx.strokeStyle = color;
        ctx.beginPath();
        vals.forEach((v, i) => {
          const x = xPx(i);
          const y = yPx(v);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
      drawLine(ys1, "#d33");
      drawLine(ys2, "#36c");
      ctx.fillStyle = "#000";
      ctx.fillText("red: epsilon_max, blue: epsilon_actual", margin, 16);
    }
    refreshBoundChart();
    setInterval(refreshBoundChart, 2000);
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
  <p><a href="/results/csv">Export CSV</a></p>
  <canvas id="acc-chart" width="800" height="240" style="border:1px solid #ddd"></canvas>
  <pre id="stream">Waiting for updates...</pre>
  <script>
    const target = document.getElementById("stream");
    const source = new EventSource("/api/state/stream");
    const canvas = document.getElementById("acc-chart");
    const ctx = canvas.getContext("2d");
    function drawAcc(history) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!Array.isArray(history) || !history.length) {
        ctx.fillText("No epoch history yet.", 10, 24);
        return;
      }
      const all = [];
      history.forEach(h => (h.delta && Array.isArray(h.delta.train_accuracy_per_epoch)) && all.push(...h.delta.train_accuracy_per_epoch));
      if (!all.length) {
        ctx.fillText("No training accuracy yet.", 10, 24);
        return;
      }
      const margin = 30, w = canvas.width - margin*2, h = canvas.height - margin*2;
      ctx.strokeStyle = "#999";
      ctx.beginPath(); ctx.moveTo(margin, margin); ctx.lineTo(margin, margin+h); ctx.lineTo(margin+w, margin+h); ctx.stroke();
      let offset = 0;
      history.forEach((task, taskIdx) => {
        const vals = (task.delta && task.delta.train_accuracy_per_epoch) || [];
        if (!vals.length) return;
        ctx.strokeStyle = ["#d33","#36c","#2a2","#a3a","#f80"][taskIdx % 5];
        ctx.beginPath();
        vals.forEach((v, i) => {
          const x = margin + (w * (offset + i) / Math.max(1, all.length - 1));
          const y = margin + h - (h * Math.max(0, Math.min(1, Number(v))));
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        offset += vals.length;
      });
    }
    source.onmessage = (event) => {
      target.textContent = event.data;
      try {
        const payload = JSON.parse(event.data);
        const history = payload.results && payload.results.epoch_history;
        drawAcc(history);
      } catch (_) {}
    };
  </script>
</body>
</html>
"""

RESULTS_PATH = Path("results.json")
LOG_PATH = Path("experiment.log")
DB_PATH = Path("meld_results.db")


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
        *(
            ["--pretrained-backbone"]
            if bool(payload.get("pretrained_backbone", False))
            else []
        ),
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
        "task_id,delta_top1,full_top1,pre_bound,post_bound,bound_held,compute_savings_percent"
    ]
    for t in tasks:
        delta = t.get("delta", {}) if isinstance(t.get("delta"), dict) else {}
        full = t.get("full_retrain", {}) if isinstance(t.get("full_retrain"), dict) else {}
        oracle = t.get("oracle", {}) if isinstance(t.get("oracle"), dict) else {}
        lines.append(
            ",".join(
                [
                    str(t.get("task_id", "")),
                    str(delta.get("top1", "")),
                    str(full.get("top1", "")),
                    str(oracle.get("pre_bound", "")),
                    str(oracle.get("post_bound", "")),
                    str(oracle.get("bound_held", "")),
                    str(t.get("compute_savings_percent", "")),
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
