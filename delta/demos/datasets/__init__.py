"""Dataset utilities for delta demos."""

from .loaders import (
    get_dataset_provider,
    register_dataset,
    list_registered_datasets,
    split_classification_dataset_into_tasks,
    extract_labels,
    validate_task_bundle,
)

# Auto-register all built-in dataset providers
from . import providers as _providers  # noqa: F401
