"""Metric objects for the evaluation plugin.

Each metric function returns a Metric object with:
  update(strategy, experience) — called after each eval
  result() — returns current value(s)
  reset() — clears accumulated state
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


class Metric:
    """Base metric class."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._value: Any = None

    def update(self, strategy: Any, experience: Any = None) -> None:
        pass

    def result(self) -> Any:
        return self._value

    def reset(self) -> None:
        self._value = None

    def __repr__(self) -> str:
        return f"Metric({self.name}={self._value})"


class AccuracyMetric(Metric):
    def __init__(self, per_experience: bool = True, stream: bool = True):
        super().__init__("accuracy")
        self.per_experience = per_experience
        self.stream = stream
        self._per_task: dict[int, float] = {}
        self._stream_acc: float | None = None

    def update(self, strategy, experience=None):
        # Read from strategy's last eval results
        if experience is not None:
            task_id = experience.task_id
            key = f"accuracy/task_{task_id}"
            # Try to get from evaluator's collected results
            self._per_task[task_id] = getattr(strategy, "_last_eval_acc", {}).get(key, 0.0)
        stream_key = "accuracy/stream"
        self._stream_acc = getattr(strategy, "_last_eval_acc", {}).get(stream_key)

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.per_experience:
            for tid, acc in self._per_task.items():
                out[f"accuracy/task_{tid}"] = acc
        if self.stream and self._stream_acc is not None:
            out["accuracy/stream"] = self._stream_acc
        return out

    def reset(self):
        self._per_task.clear()
        self._stream_acc = None


class EquivalenceMetric(Metric):
    def __init__(self, epsilon: bool = True, kl_bound: bool = True,
                 is_equivalent: bool = True):
        super().__init__("equivalence")
        self.epsilon = epsilon
        self.kl_bound = kl_bound
        self.is_equivalent = is_equivalent

    def update(self, strategy, experience=None):
        cert = getattr(strategy, "last_certificate", None)
        if cert is not None:
            self._value = {
                "epsilon_param": cert.epsilon_param,
                "kl_bound": cert.kl_bound,
                "kl_bound_normalized": cert.kl_bound_normalized,
                "is_equivalent": cert.is_equivalent,
                "tier": cert.tier,
            }

    def result(self) -> dict[str, Any]:
        if self._value is None:
            return {}
        out = {}
        if self.epsilon:
            out["epsilon_param"] = self._value["epsilon_param"]
        if self.kl_bound:
            out["kl_bound"] = self._value["kl_bound"]
            out["kl_bound_normalized"] = self._value["kl_bound_normalized"]
        if self.is_equivalent:
            out["is_equivalent"] = self._value["is_equivalent"]
        return out


class CalibrationMetric(Metric):
    def __init__(self, ece_before: bool = True, ece_after: bool = True,
                 ece_delta: bool = True):
        super().__init__("calibration")
        self._ece_before = ece_before
        self._ece_after = ece_after
        self._ece_delta = ece_delta

    def update(self, strategy, experience=None):
        cert = getattr(strategy, "last_certificate", None)
        if cert is not None:
            self._value = {
                "ece_before": cert.ece_before,
                "ece_after": cert.ece_after,
                "ece_delta": cert.ece_delta,
            }

    def result(self) -> dict[str, Any]:
        if self._value is None:
            return {}
        out = {}
        if self._ece_before:
            out["ece_before"] = self._value["ece_before"]
        if self._ece_after:
            out["ece_after"] = self._value["ece_after"]
        if self._ece_delta:
            out["ece_delta"] = self._value["ece_delta"]
        return out


class ForgettingMetric(Metric):
    def __init__(self, per_experience: bool = True):
        super().__init__("forgetting")
        self.per_experience = per_experience
        self._best_acc: dict[int, float] = {}
        self._current_acc: dict[int, float] = {}

    def update(self, strategy, experience=None):
        eval_results = getattr(strategy, "_last_eval_acc", {})
        for key, val in eval_results.items():
            if key.startswith("accuracy/task_"):
                tid = int(key.split("_")[-1])
                self._current_acc[tid] = val
                if tid not in self._best_acc or val > self._best_acc[tid]:
                    self._best_acc[tid] = val

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        forgetting_vals = []
        for tid in self._current_acc:
            f = self._best_acc.get(tid, 0.0) - self._current_acc[tid]
            out[f"forgetting/task_{tid}"] = max(0.0, f)
            forgetting_vals.append(max(0.0, f))
        if forgetting_vals:
            out["forgetting/stream"] = sum(forgetting_vals) / len(forgetting_vals)
        return out

    def reset(self):
        self._current_acc.clear()


class ComputeMetric(Metric):
    def __init__(self, savings_ratio: bool = True):
        super().__init__("compute")
        self.savings_ratio = savings_ratio

    def update(self, strategy, experience=None):
        cert = getattr(strategy, "last_certificate", None)
        if cert is not None:
            self._value = {"compute_ratio": cert.compute_ratio}

    def result(self) -> dict[str, Any]:
        if self._value is None:
            return {}
        out = {}
        if self.savings_ratio:
            out["compute_ratio"] = self._value["compute_ratio"]
        return out


# ── Factory functions ──────────────────────────────────────

def accuracy_metrics(experience: bool = True, stream: bool = True) -> AccuracyMetric:
    return AccuracyMetric(per_experience=experience, stream=stream)

def equivalence_metrics(epsilon: bool = True, kl_bound: bool = True,
                        is_equivalent: bool = True) -> EquivalenceMetric:
    return EquivalenceMetric(epsilon=epsilon, kl_bound=kl_bound,
                             is_equivalent=is_equivalent)

def calibration_metrics(ece_before: bool = True, ece_after: bool = True,
                        ece_delta: bool = True) -> CalibrationMetric:
    return CalibrationMetric(ece_before=ece_before, ece_after=ece_after,
                             ece_delta=ece_delta)

def forgetting_metrics(experience: bool = True) -> ForgettingMetric:
    return ForgettingMetric(per_experience=experience)

def compute_metrics(savings_ratio: bool = True) -> ComputeMetric:
    return ComputeMetric(savings_ratio=savings_ratio)
