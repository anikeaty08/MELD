"""EvaluationPlugin — collects metrics and dispatches to loggers.

Wire into any BaseStrategy via the evaluator parameter.
Automatically called during strategy.eval() via the hook system.
"""

from __future__ import annotations
from typing import Any


class EvaluationPlugin:
    """Collects metrics and forwards them to loggers.

    Args:
        *metrics: Metric objects (from factory functions).
        loggers: List of logger objects with .log() and .log_summary().
    """

    def __init__(self, *metrics: Any, loggers: list[Any] | None = None):
        self.metrics = list(metrics)
        self.loggers = list(loggers or [])
        self._last_metrics: dict[str, Any] = {}

    def before_eval_stream(self, strategy: Any, experiences: Any) -> None:
        """Reset metric state before a new stream evaluation."""
        strategy._last_eval_acc = {}
        for metric in self.metrics:
            reset = getattr(metric, "reset", None)
            if reset is not None:
                reset()

    def after_eval_experience(self, strategy: Any, experience: Any) -> None:
        """Called after each experience evaluation."""
        # Store eval results on strategy for metrics to read
        if not hasattr(strategy, "_last_eval_acc"):
            strategy._last_eval_acc = {}

        for metric in self.metrics:
            metric.update(strategy, experience)

    def after_eval_stream(self, strategy: Any, results: dict[str, Any]) -> None:
        """Called after full stream evaluation."""
        # Store eval results for metrics
        strategy._last_eval_acc = dict(results)

        # Update all metrics with stream-level results
        for metric in self.metrics:
            metric.update(strategy)

        # Collect all results
        self._last_metrics.clear()
        self._last_metrics.update(results)
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                self._last_metrics.update(result)

        # Log to all loggers
        task_id = getattr(strategy, "current_task_id", None)
        for logger in self.loggers:
            for name, value in self._last_metrics.items():
                if hasattr(logger, "log"):
                    logger.log(name, value, task_id=task_id)
            if hasattr(logger, "log_summary"):
                logger.log_summary(self._last_metrics)

    def get_last_metrics(self) -> dict[str, Any]:
        """Return the most recent metric values."""
        return dict(self._last_metrics)
