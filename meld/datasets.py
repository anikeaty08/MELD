"""Dataset extension hooks and helpers for MELD."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

TaskBundle = list[tuple[Dataset[Any], Dataset[Any]]]
DatasetProvider = Callable[[Any], TaskBundle]

_DATASET_REGISTRY: dict[str, DatasetProvider] = {}
_BUILTIN_DATASETS = (
    "synthetic",
    "CIFAR-10",
    "CIFAR-100",
    "TinyImageNet",
    "STL-10",
    "AGNews",
    "DBpedia",
    "YahooAnswersNLP",
)


def normalize_dataset_name(name: str) -> str:
    return name.upper().replace("-", "").replace("_", "")


def register_dataset(
    name: str,
    provider: DatasetProvider,
    *,
    aliases: Sequence[str] | None = None,
    overwrite: bool = False,
) -> None:
    keys = [name, *(aliases or ())]
    for raw in keys:
        key = normalize_dataset_name(raw)
        if not overwrite and key in _DATASET_REGISTRY:
            raise ValueError(f"Dataset provider already registered for '{raw}'.")
        _DATASET_REGISTRY[key] = provider


def get_dataset_provider(name: str) -> DatasetProvider | None:
    return _DATASET_REGISTRY.get(normalize_dataset_name(name))


def list_registered_datasets() -> list[str]:
    names = sorted(set(_BUILTIN_DATASETS).union(_DATASET_REGISTRY))
    return names


def extract_labels(dataset: Dataset[Any]) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        return torch.as_tensor(getattr(dataset, "targets"))
    if hasattr(dataset, "labels"):
        return torch.as_tensor(getattr(dataset, "labels"))
    labels = [dataset[index][1] for index in range(len(dataset))]
    return torch.as_tensor(labels)


def split_classification_dataset_into_tasks(
    train_ds: Dataset[Any],
    test_ds: Dataset[Any],
    *,
    num_tasks: int,
    classes_per_task: int,
) -> TaskBundle:
    train_targets = extract_labels(train_ds)
    test_targets = extract_labels(test_ds)
    if train_targets.numel() == 0:
        return []

    classes = torch.unique(train_targets).tolist()
    classes = sorted(int(class_id) for class_id in classes)
    bundle: TaskBundle = []
    for task_id in range(min(int(num_tasks), max(1, len(classes) // max(1, int(classes_per_task))))):
        start = task_id * int(classes_per_task)
        class_ids = classes[start : start + int(classes_per_task)]
        if not class_ids:
            break
        train_mask = torch.zeros(len(train_targets), dtype=torch.bool)
        test_mask = torch.zeros(len(test_targets), dtype=torch.bool)
        for class_id in class_ids:
            train_mask |= train_targets == class_id
            test_mask |= test_targets == class_id
        train_indices = train_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        test_indices = test_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        bundle.append((Subset(train_ds, train_indices), Subset(test_ds, test_indices)))
    return bundle


def validate_task_bundle(bundle: Any) -> TaskBundle:
    if not isinstance(bundle, list):
        raise TypeError("Dataset provider must return a list of (train_ds, test_ds) task pairs.")
    validated: TaskBundle = []
    for index, item in enumerate(bundle):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"Task bundle entry {index} must be a (train_ds, test_ds) tuple, got {type(item).__name__}."
            )
        train_ds, test_ds = item
        if not hasattr(train_ds, "__len__") or not hasattr(train_ds, "__getitem__"):
            raise TypeError(f"Train dataset at task {index} is not dataset-like.")
        if not hasattr(test_ds, "__len__") or not hasattr(test_ds, "__getitem__"):
            raise TypeError(f"Eval dataset at task {index} is not dataset-like.")
        validated.append((train_ds, test_ds))
    return validated
