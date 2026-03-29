"""ContinualDataset base class and registry for custom datasets."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import DataLoader, Dataset

DATASET_REGISTRY: dict[str, type] = {}


def register_dataset(name: str, cls: type) -> None:
    """Register a ContinualDataset subclass by name."""
    DATASET_REGISTRY[name.upper().replace("-", "").replace("_", "")] = cls


class ContinualDataset(ABC):
    """Abstract base for continual learning datasets.

    Subclass this to register custom datasets with the framework.

    Example::

        class MyDataset(ContinualDataset):
            @property
            def n_classes(self): return 10
            @property
            def n_tasks(self): return 5
            @property
            def name(self): return "MyDataset"
            def get_data_loaders(self, config):
                ...
        register_dataset("mydataset", MyDataset)
    """

    @abstractmethod
    def get_data_loaders(
        self, config: Any
    ) -> tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader) for the current task."""

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Total number of classes across all tasks."""

    @property
    @abstractmethod
    def n_tasks(self) -> int:
        """Total number of tasks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""
