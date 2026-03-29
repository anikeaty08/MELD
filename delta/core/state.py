"""
DeltaState: compact statistics snapshot.
Stores ONLY what is needed to reconstruct gradient behavior
of old data. NEVER stores raw input data.

Mathematical basis:
  gradient of old loss at new theta ≈
  grad(theta_old, D_old) + H_old * (theta - theta_old)

  where H_old ≈ kron(G_l, A_l) per layer (KFAC approximation)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DeltaState:
    theta_ref: dict[str, np.ndarray] = field(default_factory=dict)
    kfac_A: dict[str, np.ndarray] = field(default_factory=dict)
    kfac_G: dict[str, np.ndarray] = field(default_factory=dict)
    fisher_diag: Optional[np.ndarray] = None
    kfac_param_names: list[str] = field(default_factory=list)
    n_old: int = 0
    input_mean: Optional[np.ndarray] = None
    input_var: Optional[np.ndarray] = None
    label_counts: dict[int, int] = field(default_factory=dict)
    class_feature_means: dict[int, np.ndarray] = field(default_factory=dict)
    class_feature_vars: dict[int, np.ndarray] = field(default_factory=dict)
    fisher_eigenvalue_max: float = 0.0

    def save(self, path: str) -> None:
        data = {
            "n_old": np.array([self.n_old]),
            "fisher_eigenvalue_max": np.array([self.fisher_eigenvalue_max]),
            "kfac_param_names": np.array(self.kfac_param_names),
        }
        if self.fisher_diag is not None:
            data["fisher_diag"] = self.fisher_diag
        if self.input_mean is not None:
            data["input_mean"] = self.input_mean
        if self.input_var is not None:
            data["input_var"] = self.input_var
        for k, v in self.theta_ref.items():
            data[f"theta_ref__{k}"] = v
        for k, v in self.kfac_A.items():
            data[f"kfac_A__{k}"] = v
        for k, v in self.kfac_G.items():
            data[f"kfac_G__{k}"] = v
        for k, v in self.label_counts.items():
            data[f"label_count__{k}"] = np.array([v])
        for k, v in self.class_feature_means.items():
            data[f"class_feature_mean__{k}"] = v
        for k, v in self.class_feature_vars.items():
            data[f"class_feature_var__{k}"] = v
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str) -> "DeltaState":
        data = np.load(path, allow_pickle=False)
        state = cls()
        state.n_old = int(data["n_old"][0])
        state.fisher_eigenvalue_max = float(data["fisher_eigenvalue_max"][0])
        state.kfac_param_names = list(data["kfac_param_names"])
        if "fisher_diag" in data:
            state.fisher_diag = data["fisher_diag"]
        if "input_mean" in data:
            state.input_mean = data["input_mean"]
        if "input_var" in data:
            state.input_var = data["input_var"]
        for key in data.files:
            if key.startswith("theta_ref__"):
                state.theta_ref[key[11:]] = data[key]
            elif key.startswith("kfac_A__"):
                state.kfac_A[key[8:]] = data[key]
            elif key.startswith("kfac_G__"):
                state.kfac_G[key[8:]] = data[key]
            elif key.startswith("label_count__"):
                state.label_counts[int(key[13:])] = int(data[key][0])
            elif key.startswith("class_feature_mean__"):
                state.class_feature_means[int(key[20:])] = data[key]
            elif key.startswith("class_feature_var__"):
                state.class_feature_vars[int(key[19:])] = data[key]
        return state

    @classmethod
    def from_task_snapshot(cls, snapshot) -> "DeltaState":
        state = cls()
        state.n_old = int(getattr(snapshot, "dataset_size", 0))
        state.fisher_eigenvalue_max = float(
            getattr(snapshot, "fisher_eigenvalue_max", 0.0))
        state.kfac_A = dict(getattr(snapshot, "kfac_factors_A", {}) or {})
        state.kfac_G = dict(getattr(snapshot, "kfac_factors_G", {}) or {})
        state.kfac_param_names = list(
            getattr(snapshot, "kfac_weight_param_names", []))
        state.input_mean = getattr(snapshot, "input_feature_mean", None)
        state.input_var = getattr(snapshot, "input_feature_var", None)
        if hasattr(snapshot, "fisher_diagonal") and snapshot.fisher_diagonal is not None:
            state.fisher_diag = np.array(snapshot.fisher_diagonal, dtype=np.float32)
        if hasattr(snapshot, "parameter_reference") and snapshot.parameter_reference:
            pnames = list(getattr(snapshot, "protected_parameter_names", []))
            for i, arr in enumerate(snapshot.parameter_reference):
                name = pnames[i] if i < len(pnames) else f"param_{i}"
                state.theta_ref[name] = np.array(arr, dtype=np.float32)
        return state
