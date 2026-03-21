"""Top-level MELD package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "DeltaModel",
    "DeltaUpdateResult",
    "MELDConfig",
    "TrainConfig",
    "__version__",
    "list_registered_datasets",
    "register_dataset",
    "run",
]


def __getattr__(name: str) -> Any:
    if name in {"MELDConfig", "TrainConfig", "list_registered_datasets", "register_dataset", "run"}:
        module = import_module(".api", __name__)
        return getattr(module, name)
    if name in {"DeltaModel", "DeltaUpdateResult"}:
        module = import_module(".delta", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
