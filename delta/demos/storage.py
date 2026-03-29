"""SQLite-backed result persistence for MELD benchmark runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ResultStore:
    def __init__(self, database_path: str | Path | None) -> None:
        self.database_path = Path(database_path) if database_path else None
        if self.database_path is not None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    def sync_run(self, results: dict[str, Any]) -> None:
        if self.database_path is None:
            return

        run_id = str(results.get("run_id", "default"))
        config = results.get("config")
        final_summary = results.get("final_summary")
        status = str(results.get("status", "unknown"))
        payload_json = json.dumps(results)

        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, status, config_json, final_summary_json, payload_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    config_json=excluded.config_json,
                    final_summary_json=excluded.final_summary_json,
                    payload_json=excluded.payload_json
                """,
                (
                    run_id,
                    status,
                    json.dumps(config),
                    json.dumps(final_summary),
                    payload_json,
                ),
            )

            conn.execute("DELETE FROM tasks WHERE run_id = ?", (run_id,))
            for task in results.get("tasks", []):
                conn.execute(
                    """
                    INSERT INTO tasks(
                        run_id,
                        task_id,
                        delta_json,
                        full_retrain_json,
                        oracle_json,
                        drift_json,
                        decision_json,
                        snapshot_json,
                        train_json,
                        cil_metrics_json,
                        equivalence_gap,
                        forgetting,
                        compute_savings_percent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(task.get("task_id", 0)),
                        json.dumps(task.get("delta")),
                        json.dumps(task.get("full_retrain")),
                        json.dumps(task.get("oracle")),
                        json.dumps(task.get("drift")),
                        json.dumps(task.get("decision")),
                        json.dumps(task.get("snapshot")),
                        json.dumps(task.get("train")),
                        json.dumps(task.get("cil_metrics")),
                        task.get("equivalence_gap"),
                        task.get("forgetting"),
                        task.get("compute_savings_percent"),
                    ),
                )
            conn.commit()

    def _initialize(self) -> None:
        assert self.database_path is not None
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_json TEXT,
                    final_summary_json TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    run_id TEXT NOT NULL,
                    task_id INTEGER NOT NULL,
                    delta_json TEXT,
                    full_retrain_json TEXT,
                    oracle_json TEXT,
                    drift_json TEXT,
                    decision_json TEXT,
                    snapshot_json TEXT,
                    train_json TEXT,
                    cil_metrics_json TEXT,
                    equivalence_gap REAL,
                    forgetting REAL,
                    compute_savings_percent REAL,
                    PRIMARY KEY (run_id, task_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
                """
            )
            conn.commit()
