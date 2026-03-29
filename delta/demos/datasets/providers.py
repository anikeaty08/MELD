"""Built-in dataset providers for the delta framework.

Each provider is a function that takes a config namespace and returns
a TaskBundle: list[tuple[train_dataset, test_dataset]].

Providers are auto-registered when this module is imported.
When a provider's dependencies are missing, it raises a clear
error telling the user exactly what to install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset

from .loaders import (
    register_dataset,
    split_classification_dataset_into_tasks,
    TaskBundle,
)


class MissingDependencyError(ImportError):
    """Raised when a dataset provider can't load because a package is missing."""
    pass


def _require(package: str, install_extra: str, dataset_name: str) -> None:
    """Import a package or raise a clear install instruction."""
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        raise MissingDependencyError(
            f"\n{'='*60}\n"
            f"  Dataset '{dataset_name}' requires '{package}'\n"
            f"  which is not installed.\n\n"
            f"  Install it with:\n"
            f"    pip install delta-framework[{install_extra}]\n\n"
            f"  Or install the dependency directly:\n"
            f"    pip install {package}\n"
            f"{'='*60}\n"
        )


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _image_preset(config: Any) -> str:
    return str(_cfg(config, "preset", "standard")).lower()


def _image_size(config: Any, default: int) -> int:
    size = _cfg(config, "image_size", None)
    return int(size) if size is not None else int(default)


def _use_imagenet_stats(config: Any) -> bool:
    explicit = _cfg(config, "use_imagenet_stats", None)
    if explicit is not None:
        return bool(explicit)
    return bool(_cfg(config, "pretrained_backbone", False))


def _normalize_stats(config: Any, dataset_mean: tuple[float, ...], dataset_std: tuple[float, ...]):
    if _use_imagenet_stats(config):
        if len(dataset_mean) == 1:
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return dataset_mean, dataset_std


def _maybe_add_resize(ops: list[Any], size: int, native_size: int, T: Any) -> None:
    if int(size) != int(native_size):
        ops.append(T.Resize((size, size)))


def _maybe_add_randaugment(ops: list[Any], T: Any) -> None:
    if hasattr(T, "RandAugment"):
        ops.append(T.RandAugment(num_ops=2, magnitude=9))


def _grayscale_pipeline(
    config: Any,
    dataset_mean: tuple[float, ...],
    dataset_std: tuple[float, ...],
    *,
    native_size: int = 28,
) -> tuple[list[Any], list[Any]]:
    import torchvision.transforms as T

    preset = _image_preset(config)
    size = _image_size(config, native_size)
    rgb = _use_imagenet_stats(config)
    mean, std = _normalize_stats(config, dataset_mean, dataset_std)
    train_ops: list[Any] = []
    test_ops: list[Any] = []
    if rgb:
        train_ops.append(T.Grayscale(num_output_channels=3))
        test_ops.append(T.Grayscale(num_output_channels=3))
    _maybe_add_resize(train_ops, size, native_size, T)
    _maybe_add_resize(test_ops, size, native_size, T)
    if preset == "maxperf":
        train_ops.append(T.RandomCrop(size, padding=max(2, size // 12)))
        train_ops.append(T.RandomAffine(degrees=12, translate=(0.08, 0.08)))
    train_ops.extend([T.ToTensor(), T.Normalize(mean, std)])
    test_ops.extend([T.ToTensor(), T.Normalize(mean, std)])
    return train_ops, test_ops


def _rgb_pipeline(
    config: Any,
    dataset_mean: tuple[float, float, float],
    dataset_std: tuple[float, float, float],
    *,
    native_size: int = 32,
    allow_flip: bool = True,
) -> tuple[list[Any], list[Any]]:
    import torchvision.transforms as T

    preset = _image_preset(config)
    size = _image_size(config, native_size)
    mean, std = _normalize_stats(config, dataset_mean, dataset_std)
    train_ops: list[Any] = []
    test_ops: list[Any] = []
    _maybe_add_resize(train_ops, size, native_size, T)
    _maybe_add_resize(test_ops, size, native_size, T)
    if preset == "maxperf":
        train_ops.append(T.RandomCrop(size, padding=max(4, size // 12)))
        if allow_flip:
            train_ops.append(T.RandomHorizontalFlip())
        _maybe_add_randaugment(train_ops, T)
    else:
        train_ops.append(T.RandomCrop(size, padding=max(4, size // 12)))
        if allow_flip:
            train_ops.append(T.RandomHorizontalFlip())
    train_ops.extend([T.ToTensor(), T.Normalize(mean, std)])
    if preset == "maxperf" and hasattr(T, "RandomErasing"):
        train_ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
    test_ops.extend([T.ToTensor(), T.Normalize(mean, std)])
    return train_ops, test_ops


# ===================================================================
# Image datasets (require torchvision)
# ===================================================================

def _cifar10_provider(config: Any) -> TaskBundle:
    """CIFAR-10: 10 classes, 50k train, 10k test, 32x32 RGB."""
    _require("torchvision", "vision", "CIFAR-10")
    import torchvision

    root = getattr(config, "data_root", "./data")
    train_ops, test_ops = _rgb_pipeline(
        config,
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
        native_size=32,
    )
    train_transform = torchvision.transforms.Compose(train_ops)
    test_transform = torchvision.transforms.Compose(test_ops)
    train_ds = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform)
    return split_classification_dataset_into_tasks(
        train_ds, test_ds,
        num_tasks=getattr(config, "num_tasks", 5),
        classes_per_task=getattr(config, "classes_per_task", 2),
    )


def _cifar100_provider(config: Any) -> TaskBundle:
    """CIFAR-100: 100 classes, 50k train, 10k test, 32x32 RGB."""
    _require("torchvision", "vision", "CIFAR-100")
    import torchvision

    root = getattr(config, "data_root", "./data")
    train_ops, test_ops = _rgb_pipeline(
        config,
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
        native_size=32,
    )
    train_transform = torchvision.transforms.Compose(train_ops)
    test_transform = torchvision.transforms.Compose(test_ops)
    train_ds = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=test_transform)
    return split_classification_dataset_into_tasks(
        train_ds, test_ds,
        num_tasks=getattr(config, "num_tasks", 10),
        classes_per_task=getattr(config, "classes_per_task", 10),
    )


def _stl10_provider(config: Any) -> TaskBundle:
    """STL-10: 10 classes, 5k train, 8k test, 96x96 RGB."""
    _require("torchvision", "vision", "STL-10")
    import torchvision
    import torchvision.transforms as T
    root = getattr(config, "data_root", "./data")
    preset = _image_preset(config)
    size = _image_size(config, 32)
    mean, std = _normalize_stats(config, (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
    train_ops: list[Any] = [T.Resize((size, size))]
    test_ops: list[Any] = [T.Resize((size, size))]
    if preset == "maxperf":
        train_ops.append(T.RandomCrop(size, padding=max(4, size // 12)))
        train_ops.append(T.RandomHorizontalFlip())
        _maybe_add_randaugment(train_ops, T)
    else:
        train_ops.append(T.RandomCrop(size, padding=max(4, size // 12)))
        train_ops.append(T.RandomHorizontalFlip())
    train_ops.extend([T.ToTensor(), T.Normalize(mean, std)])
    if preset == "maxperf" and hasattr(T, "RandomErasing"):
        train_ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
    test_ops.extend([T.ToTensor(), T.Normalize(mean, std)])
    train_transform = T.Compose(train_ops)
    test_transform = T.Compose(test_ops)
    train_ds = torchvision.datasets.STL10(
        root=root, split="train", download=True, transform=train_transform)
    test_ds = torchvision.datasets.STL10(
        root=root, split="test", download=True, transform=test_transform)
    return split_classification_dataset_into_tasks(
        train_ds, test_ds,
        num_tasks=getattr(config, "num_tasks", 5),
        classes_per_task=getattr(config, "classes_per_task", 2),
    )


def _svhn_provider(config: Any) -> TaskBundle:
    """SVHN: 10 digit classes, 73k train, 26k test, 32x32 RGB."""
    _require("torchvision", "vision", "SVHN")
    import torchvision
    root = getattr(config, "data_root", "./data")
    train_ops, test_ops = _rgb_pipeline(
        config,
        (0.4377, 0.4438, 0.4728),
        (0.1980, 0.2010, 0.1970),
        native_size=32,
        allow_flip=False,
    )
    train_transform = torchvision.transforms.Compose(train_ops)
    test_transform = torchvision.transforms.Compose(test_ops)
    train_ds = torchvision.datasets.SVHN(
        root=root, split="train", download=True, transform=train_transform)
    test_ds = torchvision.datasets.SVHN(
        root=root, split="test", download=True, transform=test_transform)
    return split_classification_dataset_into_tasks(
        train_ds, test_ds,
        num_tasks=getattr(config, "num_tasks", 5),
        classes_per_task=getattr(config, "classes_per_task", 2),
    )


def _fashionmnist_provider(config: Any) -> TaskBundle:
    """FashionMNIST: 10 classes, 60k train, 10k test, 28x28 grayscale."""
    _require("torchvision", "vision", "FashionMNIST")
    import torchvision
    root = getattr(config, "data_root", "./data")
    import torchvision.transforms as T

    train_ops, test_ops = _grayscale_pipeline(
        config,
        (0.2860,),
        (0.3530,),
        native_size=28,
    )
    train_transform = T.Compose(train_ops)
    test_transform = T.Compose(test_ops)
    train_ds = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=test_transform)
    return split_classification_dataset_into_tasks(
        train_ds, test_ds,
        num_tasks=getattr(config, "num_tasks", 5),
        classes_per_task=getattr(config, "classes_per_task", 2),
    )


def _mnist_provider(config: Any) -> TaskBundle:
    """MNIST: 10 digit classes, 60k train, 10k test, 28x28 grayscale."""
    _require("torchvision", "vision", "MNIST")
    import torchvision
    root = getattr(config, "data_root", "./data")
    import torchvision.transforms as T

    train_ops, test_ops = _grayscale_pipeline(
        config,
        (0.1307,),
        (0.3081,),
        native_size=28,
    )
    train_transform = T.Compose(train_ops)
    test_transform = T.Compose(test_ops)
    train_ds = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=test_transform)
    return split_classification_dataset_into_tasks(
        train_ds, test_ds,
        num_tasks=getattr(config, "num_tasks", 5),
        classes_per_task=getattr(config, "classes_per_task", 2),
    )


def _tinyimagenet_provider(config: Any) -> TaskBundle:
    """TinyImageNet: 200 classes, 100k train, 10k test, 64x64 RGB.

    Requires manual download — expects tiny-imagenet-200/ under data_root.
    """
    _require("torchvision", "vision", "TinyImageNet")
    import torchvision
    root = Path(getattr(config, "data_root", "./data")) / "tiny-imagenet-200"
    if not root.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"  TinyImageNet not found at {root}\n\n"
            f"  Download manually:\n"
            f"    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
            f"    unzip tiny-imagenet-200.zip -d {root.parent}\n"
            f"{'='*60}\n"
        )
    import torchvision.transforms as T

    train_ops, test_ops = _rgb_pipeline(
        config,
        (0.4802, 0.4481, 0.3975),
        (0.2302, 0.2265, 0.2262),
        native_size=64,
    )
    transform_train = T.Compose(train_ops)
    transform_test = T.Compose(test_ops)
    train_ds = torchvision.datasets.ImageFolder(
        str(root / "train"), transform=transform_train)
    test_ds = torchvision.datasets.ImageFolder(
        str(root / "val"), transform=transform_test)
    return split_classification_dataset_into_tasks(
        train_ds, test_ds,
        num_tasks=getattr(config, "num_tasks", 10),
        classes_per_task=getattr(config, "classes_per_task", 20),
    )


# ===================================================================
# Text datasets (require datasets + transformers from HuggingFace)
# ===================================================================

class _TextClassificationDataset(Dataset):
    """Wraps pre-computed text embeddings into a torch Dataset."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        self.embeddings = embeddings
        self.labels = labels
        self.targets = labels  # for extract_labels()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


def _encode_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode texts to embeddings using a sentence-transformer."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)


def _hf_text_provider(
    dataset_name: str,
    text_field: str,
    label_field: str,
    config: Any,
) -> TaskBundle:
    """Generic provider for HuggingFace text classification datasets."""
    _require("datasets", "text", dataset_name)
    _require("transformers", "text", dataset_name)
    from datasets import load_dataset

    model_name = getattr(
        getattr(config, "train", None),
        "text_encoder_model",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    ds = load_dataset(dataset_name)
    train_split = ds["train"]
    test_split = ds["test"]

    train_texts = train_split[text_field]
    train_labels = torch.tensor(train_split[label_field], dtype=torch.long)
    test_texts = test_split[text_field]
    test_labels = torch.tensor(test_split[label_field], dtype=torch.long)

    train_emb = _encode_texts(train_texts, model_name)
    test_emb = _encode_texts(test_texts, model_name)

    train_ds = _TextClassificationDataset(train_emb, train_labels)
    test_ds = _TextClassificationDataset(test_emb, test_labels)

    return split_classification_dataset_into_tasks(
        train_ds, test_ds,
        num_tasks=getattr(config, "num_tasks", 2),
        classes_per_task=getattr(config, "classes_per_task", 2),
    )


def _agnews_provider(config: Any) -> TaskBundle:
    """AG News: 4 classes, 120k train, 7.6k test."""
    return _hf_text_provider("ag_news", "text", "label", config)


def _dbpedia_provider(config: Any) -> TaskBundle:
    """DBpedia: 14 classes, 560k train, 70k test."""
    return _hf_text_provider("dbpedia_14", "content", "label", config)


def _yahooanswers_provider(config: Any) -> TaskBundle:
    """Yahoo Answers: 10 classes, 1.4M train, 60k test."""
    return _hf_text_provider(
        "yahoo_answers_topics", "question_title", "topic", config)


def _imdb_provider(config: Any) -> TaskBundle:
    """IMDB: 2 classes (sentiment), 25k train, 25k test."""
    return _hf_text_provider("imdb", "text", "label", config)


def _sst2_provider(config: Any) -> TaskBundle:
    """SST-2: 2 classes (sentiment), 67k train, 872 test."""
    return _hf_text_provider("glue", "sentence", "label", config)


# ===================================================================
# Registration
# ===================================================================

def _try_register(name: str, provider, aliases=None) -> None:
    register_dataset(name, provider, aliases=aliases, overwrite=True)


# Image datasets
_try_register("CIFAR-10", _cifar10_provider, aliases=["cifar10"])
_try_register("CIFAR-100", _cifar100_provider, aliases=["cifar100"])
_try_register("STL-10", _stl10_provider, aliases=["stl10"])
_try_register("SVHN", _svhn_provider, aliases=["svhn"])
_try_register("FashionMNIST", _fashionmnist_provider,
              aliases=["fashion-mnist", "fmnist"])
_try_register("MNIST", _mnist_provider, aliases=["mnist"])
_try_register("TinyImageNet", _tinyimagenet_provider,
              aliases=["tiny-imagenet", "tinyimagenet200"])

# Text datasets
_try_register("AGNews", _agnews_provider, aliases=["ag-news", "ag_news"])
_try_register("DBpedia", _dbpedia_provider, aliases=["dbpedia14", "dbpedia_14"])
_try_register("YahooAnswersNLP", _yahooanswers_provider,
              aliases=["yahoo-answers", "yahoo_answers"])
_try_register("IMDB", _imdb_provider, aliases=["imdb"])
_try_register("SST-2", _sst2_provider, aliases=["sst2", "glue-sst2"])
