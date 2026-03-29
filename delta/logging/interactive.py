"""InteractiveLogger — prints metrics to stdout."""

from __future__ import annotations
from typing import Any


class InteractiveLogger:
    """Prints metric values and summary tables to stdout."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._logged_this_round: bool = False

    def log(self, name: str, value: Any, task_id: int | None = None,
            step: int | None = None) -> None:
        if not self.verbose:
            return
        # Skip individual logs — we'll print the summary table
        self._logged_this_round = True

    def log_summary(self, metrics_dict: dict[str, Any]) -> None:
        if not metrics_dict:
            return

        # Find longest key for formatting
        keys = list(metrics_dict.keys())
        max_key_len = max(len(k) for k in keys) if keys else 20
        col1 = max(max_key_len + 2, 22)
        col2 = 12

        print()
        print("+" + "-" * (col1 + col2 + 3) + "+")
        print(f"| {'Metric':<{col1}} | {'Value':>{col2}} |")
        print("+" + "-" * (col1 + col2 + 3) + "+")

        for name, value in sorted(metrics_dict.items()):
            formatted = self._format_value(name, value)
            print(f"| {name:<{col1}} | {formatted:>{col2}} |")

        print("+" + "-" * (col1 + col2 + 3) + "+")
        print()
        self._logged_this_round = False

    @staticmethod
    def _format_value(name: str, value: Any) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            if "ratio" in name:
                return f"{value:.1f}x"
            if abs(value) < 0.0001 and value != 0.0:
                return f"{value:.6f}"
            return f"{value:.4f}"
        if isinstance(value, int):
            return str(value)
        return str(value)[:12]
