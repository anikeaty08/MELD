"""Auto-derivation of protection settings from snapshot geometry."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..interfaces.base import TaskSnapshot


def derive_train_config(snapshot: TaskSnapshot, base_config: Any, protection_level: float = 0.5) -> Any:
    fisher_max = max(float(snapshot.fisher_eigenvalue_max), 1e-6)
    fisher_mean = max(float(np.mean(snapshot.fisher_diagonal)) if snapshot.fisher_diagonal.size else 1e-6, 1e-6)
    scale = max(0.0, min(1.0, float(protection_level)))
    return replace(
        base_config,
        lambda_geometry=(1.0 + scale) / fisher_max,
        lambda_ewc=(fisher_max / fisher_mean) * (0.5 + scale),
        geometry_decay=max(0.1, float(base_config.geometry_decay) / (1.0 + scale)),
    )
