"""Benchmark streams and dataset abstractions."""

from .stream import DeltaStream, Experience
from .dataset_base import ContinualDataset, register_dataset, DATASET_REGISTRY

__all__ = [
    "DeltaStream",
    "Experience",
    "ContinualDataset",
    "register_dataset",
    "DATASET_REGISTRY",
]
