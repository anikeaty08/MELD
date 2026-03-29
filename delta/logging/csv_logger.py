"""CSVLogger — writes metrics to a CSV file."""

from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Any


class CSVLogger:
    """Appends metric rows to a CSV file."""

    def __init__(self, filepath: str = "delta_metrics.csv") -> None:
        self.filepath = Path(filepath)
        self._writer = None
        self._file = None
        self._header_written = False

    def log(self, name: str, value: Any, task_id: int | None = None,
            step: int | None = None) -> None:
        if self._file is None:
            self._file = open(self.filepath, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            if not self._header_written and self.filepath.stat().st_size == 0:
                self._writer.writerow(
                    ["timestamp", "task_id", "step", "metric", "value"])
                self._header_written = True
        self._writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            task_id if task_id is not None else "",
            step if step is not None else "",
            name,
            value,
        ])
        self._file.flush()

    def log_summary(self, metrics_dict: dict[str, Any]) -> None:
        for name, value in metrics_dict.items():
            self.log(name, value)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
