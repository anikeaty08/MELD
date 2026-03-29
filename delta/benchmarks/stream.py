"""Experience and DeltaStream — the data abstraction layer.

An Experience represents one task's worth of train/test data.
A DeltaStream produces an ordered sequence of Experience objects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset


@dataclass
class Experience:
    """One task in a continual learning stream."""

    train_dataset: Dataset
    test_dataset: Dataset
    task_id: int
    classes_in_this_experience: list[int] = field(default_factory=list)
    dataset_name: str = ""
    scenario: str = "class_incremental"

    def train_dataloader(
        self, batch_size: int = 64, num_workers: int = 0, shuffle: bool = True,
    ) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def test_dataloader(
        self, batch_size: int = 64, num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )


class DeltaStream:
    """Ordered stream of Experience objects for continual learning.

    Wraps dataset loading so strategy code never touches raw files.

    Args:
        dataset_name: "synthetic", "CIFAR-10", "CIFAR-100", etc.
        n_tasks: Number of incremental tasks.
        scenario: "class_incremental", "domain_incremental", or
                  "task_incremental".
        classes_per_task: Override classes per task (auto if None).
        data_root: Root directory for dataset downloads.
        seed: Random seed for reproducibility.
        batch_size: Default batch size for dataloaders.
    """

    def __init__(
        self,
        dataset_name: str = "synthetic",
        n_tasks: int = 5,
        scenario: str = "class_incremental",
        classes_per_task: int | None = None,
        data_root: str = "./data",
        seed: int = 42,
        batch_size: int = 64,
        feature_dim: int = 32,
        preset: str = "standard",
        pretrained_backbone: bool = False,
        image_size: int | None = None,
        use_imagenet_stats: bool | None = None,
    ) -> None:
        assert scenario in (
            "class_incremental",
            "domain_incremental",
            "task_incremental",
        ), f"Unknown scenario: {scenario}"

        self.dataset_name = dataset_name
        self._n_tasks = n_tasks
        self.scenario = scenario
        self.data_root = data_root
        self.seed = seed
        self.batch_size = batch_size
        self._feature_dim = feature_dim
        self.preset = preset
        self.pretrained_backbone = pretrained_backbone
        self.image_size = image_size
        self.use_imagenet_stats = use_imagenet_stats

        self._experiences: list[Experience] = []
        self._build(classes_per_task)

    def _build(self, classes_per_task: int | None) -> None:
        name = self.dataset_name.upper().replace("-", "").replace("_", "")

        if name == "SYNTHETIC":
            self._build_synthetic(classes_per_task or 2)
            return

        # Try registered dataset providers
        try:
            from delta.demos.datasets import get_dataset_provider
            provider = get_dataset_provider(self.dataset_name)
        except ImportError:
            provider = None

        if provider is not None:
            from types import SimpleNamespace
            cpt = classes_per_task or self._auto_classes_per_task(name)
            config = SimpleNamespace(
                dataset=self.dataset_name,
                num_tasks=self._n_tasks,
                classes_per_task=cpt,
                seed=self.seed,
                data_root=self.data_root,
                preset=self.preset,
                pretrained_backbone=self.pretrained_backbone,
                image_size=self.image_size,
                use_imagenet_stats=self.use_imagenet_stats,
                prefer_cuda=False,
                train=SimpleNamespace(
                    batch_size=self.batch_size,
                    num_workers=0,
                    text_encoder_model="sentence-transformers/all-MiniLM-L6-v2",
                ),
            )
            # Let MissingDependencyError propagate with install instructions
            bundle = provider(config)
            self._from_bundle(bundle, cpt)
            return

        # No provider registered for this name
        import warnings
        warnings.warn(
            f"No dataset provider registered for '{self.dataset_name}'. "
            f"Falling back to synthetic data. "
            f"To use real datasets: pip install delta-framework[vision] "
            f"or pip install delta-framework[full]",
            UserWarning,
            stacklevel=2,
        )
        self._build_synthetic(classes_per_task or 2)

    def _build_synthetic(self, classes_per_task: int) -> None:
        torch.manual_seed(self.seed)
        if self.scenario == "domain_incremental":
            self._build_synthetic_domain_incremental(classes_per_task)
            return

        nc = self._n_tasks * classes_per_task
        dim = self._feature_dim
        for tid in range(self._n_tasks):
            tx, ty, ex, ey = [], [], [], []
            for off in range(classes_per_task):
                cid = tid * classes_per_task + off
                base = torch.full((dim,), float(cid) / max(1, nc))
                tx.append(base + 0.05 * torch.randn(32, dim))
                ty.append(torch.full((32,), cid, dtype=torch.long))
                ex.append(base + 0.05 * torch.randn(16, dim))
                ey.append(torch.full((16,), cid, dtype=torch.long))
            classes = list(range(tid * classes_per_task, (tid + 1) * classes_per_task))
            self._experiences.append(
                Experience(
                    train_dataset=TensorDataset(torch.cat(tx), torch.cat(ty)),
                    test_dataset=TensorDataset(torch.cat(ex), torch.cat(ey)),
                    task_id=tid,
                    classes_in_this_experience=classes,
                    dataset_name=self.dataset_name,
                    scenario=self.scenario,
                )
            )

    def _build_synthetic_domain_incremental(self, classes_per_task: int) -> None:
        dim = self._feature_dim
        nc = classes_per_task
        for tid in range(self._n_tasks):
            torch.manual_seed(self.seed + tid)
            tx, ty, ex, ey = [], [], [], []
            domain_shift = 0.15 * float(tid)
            for cid in range(classes_per_task):
                base = torch.full((dim,), float(cid) / max(1, nc))
                base = base + domain_shift
                tx.append(base + 0.05 * torch.randn(32, dim))
                ty.append(torch.full((32,), cid, dtype=torch.long))
                ex.append(base + 0.05 * torch.randn(16, dim))
                ey.append(torch.full((16,), cid, dtype=torch.long))
            classes = list(range(classes_per_task))
            self._experiences.append(
                Experience(
                    train_dataset=TensorDataset(torch.cat(tx), torch.cat(ty)),
                    test_dataset=TensorDataset(torch.cat(ex), torch.cat(ey)),
                    task_id=tid,
                    classes_in_this_experience=classes,
                    dataset_name=self.dataset_name,
                    scenario=self.scenario,
                )
            )

    def _from_bundle(
        self,
        bundle: list[tuple[Dataset, Dataset]],
        classes_per_task: int,
    ) -> None:
        for tid, (train_ds, test_ds) in enumerate(bundle):
            if tid >= self._n_tasks:
                break
            classes = list(range(tid * classes_per_task, (tid + 1) * classes_per_task))
            self._experiences.append(
                Experience(
                    train_dataset=train_ds,
                    test_dataset=test_ds,
                    task_id=tid,
                    classes_in_this_experience=classes,
                    dataset_name=self.dataset_name,
                    scenario=self.scenario,
                )
            )

    @staticmethod
    def _auto_classes_per_task(name: str) -> int:
        return {
            "CIFAR10": 2,
            "CIFAR100": 10,
            "TINYIMAGENET": 20,
            "AGNEWS": 2,
            "DBPEDIA": 7,
            "STL10": 2,
        }.get(name, 2)

    @property
    def train_stream(self) -> list[Experience]:
        return list(self._experiences)

    @property
    def test_stream(self) -> list[Experience]:
        return list(self._experiences)

    @property
    def n_tasks(self) -> int:
        return len(self._experiences)

    @property
    def all_test_dataset(self) -> Dataset:
        return ConcatDataset([e.test_dataset for e in self._experiences])
