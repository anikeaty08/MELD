"""Frontend-facing catalog data for the MELD dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..api import MELDConfig, TrainConfig
from ..core.text_encoder import TextEncoderBackbone


@dataclass(frozen=True)
class DatasetOption:
    id: str
    label: str
    domain: str
    class_count: int | None
    default_tasks: int
    default_classes_per_task: int
    recommended_backbone: str
    auto_download: bool
    note: str = ""


@dataclass(frozen=True)
class BackboneOption:
    id: str
    label: str
    family: str
    supports_pretrained: bool
    note: str = ""


DATASET_OPTIONS: tuple[DatasetOption, ...] = (
    DatasetOption(
        id="synthetic",
        label="Synthetic",
        domain="image",
        class_count=None,
        default_tasks=2,
        default_classes_per_task=2,
        recommended_backbone="resnet20",
        auto_download=False,
        note="Built-in toy benchmark for fast smoke tests.",
    ),
    DatasetOption(
        id="CIFAR-10",
        label="CIFAR-10",
        domain="image",
        class_count=10,
        default_tasks=2,
        default_classes_per_task=5,
        recommended_backbone="resnet32",
        auto_download=True,
        note="Auto-downloads through Continuum.",
    ),
    DatasetOption(
        id="CIFAR-100",
        label="CIFAR-100",
        domain="image",
        class_count=100,
        default_tasks=10,
        default_classes_per_task=10,
        recommended_backbone="resnet56",
        auto_download=True,
        note="Auto-downloads through Continuum.",
    ),
    DatasetOption(
        id="TinyImageNet",
        label="Tiny ImageNet",
        domain="image",
        class_count=200,
        default_tasks=20,
        default_classes_per_task=10,
        recommended_backbone="resnet18_imagenet",
        auto_download=False,
        note="Manual extract required at data/tiny-imagenet-200.",
    ),
    DatasetOption(
        id="STL-10",
        label="STL-10",
        domain="image",
        class_count=10,
        default_tasks=2,
        default_classes_per_task=5,
        recommended_backbone="resnet18_imagenet",
        auto_download=True,
        note="Auto-downloads through torchvision.",
    ),
    DatasetOption(
        id="AGNews",
        label="AG News",
        domain="text",
        class_count=4,
        default_tasks=1,
        default_classes_per_task=4,
        recommended_backbone="text_encoder",
        auto_download=True,
        note="Downloads via Hugging Face datasets.",
    ),
    DatasetOption(
        id="DBpedia",
        label="DBpedia",
        domain="text",
        class_count=14,
        default_tasks=7,
        default_classes_per_task=2,
        recommended_backbone="text_encoder",
        auto_download=True,
        note="Downloads via Hugging Face datasets.",
    ),
    DatasetOption(
        id="YahooAnswersNLP",
        label="Yahoo Answers",
        domain="text",
        class_count=10,
        default_tasks=5,
        default_classes_per_task=2,
        recommended_backbone="text_encoder",
        auto_download=True,
        note="Downloads via Hugging Face datasets.",
    ),
)


BACKBONE_OPTIONS: tuple[BackboneOption, ...] = (
    BackboneOption(
        id="auto",
        label="Auto",
        family="adaptive",
        supports_pretrained=False,
        note="Let MELD pick a dataset-appropriate backbone.",
    ),
    BackboneOption(
        id="resnet20",
        label="ResNet-20",
        family="cifar",
        supports_pretrained=True,
        note="Small CIFAR-style backbone.",
    ),
    BackboneOption(
        id="resnet32",
        label="ResNet-32",
        family="cifar",
        supports_pretrained=True,
        note="Balanced CIFAR-style backbone.",
    ),
    BackboneOption(
        id="resnet44",
        label="ResNet-44",
        family="cifar",
        supports_pretrained=True,
        note="Larger CIFAR-style backbone.",
    ),
    BackboneOption(
        id="resnet56",
        label="ResNet-56",
        family="cifar",
        supports_pretrained=True,
        note="Strongest CIFAR-style backbone in this repo.",
    ),
    BackboneOption(
        id="resnet18_imagenet",
        label="ResNet-18 ImageNet",
        family="imagenet",
        supports_pretrained=True,
        note="Recommended for larger images like TinyImageNet and STL-10.",
    ),
    BackboneOption(
        id="text_encoder",
        label="Frozen Text Encoder",
        family="nlp",
        supports_pretrained=True,
        note="Hugging Face encoder plus trainable incremental head.",
    ),
)


_TEXT_MODEL_LABELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM L6",
    "sentence-transformers/all-MiniLM-L12-v2": "MiniLM L12",
    "sentence-transformers/all-mpnet-base-v2": "MPNet Base",
    "sentence-transformers/paraphrase-MiniLM-L6-v2": "Paraphrase MiniLM",
    "bert-base-uncased": "BERT Base",
    "distilbert-base-uncased": "DistilBERT",
}


def normalize_dataset_key(name: str) -> str:
    return name.upper().replace("-", "").replace("_", "")


def find_dataset_option(name: str) -> DatasetOption:
    key = normalize_dataset_key(name)
    for option in DATASET_OPTIONS:
        if normalize_dataset_key(option.id) == key:
            return option
    raise KeyError(f"Unknown dataset option: {name}")


def dataset_options_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": option.id,
            "label": option.label,
            "domain": option.domain,
            "classCount": option.class_count,
            "defaultTasks": option.default_tasks,
            "defaultClassesPerTask": option.default_classes_per_task,
            "recommendedBackbone": option.recommended_backbone,
            "autoDownload": option.auto_download,
            "note": option.note,
        }
        for option in DATASET_OPTIONS
    ]


def backbone_options_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": option.id,
            "label": option.label,
            "family": option.family,
            "supportsPretrained": option.supports_pretrained,
            "note": option.note,
        }
        for option in BACKBONE_OPTIONS
    ]


def text_model_options_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": model_name,
            "label": _TEXT_MODEL_LABELS.get(model_name, model_name.rsplit("/", maxsplit=1)[-1]),
            "out_dim": out_dim,
        }
        for model_name, out_dim in TextEncoderBackbone.SUPPORTED_MODELS.items()
    ]


def build_options_payload() -> dict[str, Any]:
    config = MELDConfig()
    train = TrainConfig()
    return {
        "datasets": dataset_options_payload(),
        "backbones": backbone_options_payload(),
        "textModels": text_model_options_payload(),
        "defaults": {
            "dataset": config.dataset,
            "numTasks": config.num_tasks,
            "classesPerTask": config.classes_per_task,
            "epochs": train.epochs,
            "batchSize": train.batch_size,
            "lr": train.lr,
            "backbone": train.backbone,
            "pretrainedBackbone": train.pretrained_backbone,
            "textEncoderModel": train.text_encoder_model,
            "boundTolerance": config.bound_tolerance,
            "pacGateTolerance": config.pac_gate_tolerance,
            "mixupAlpha": train.mixup_alpha,
            "numWorkers": train.num_workers,
            "dataRoot": str(config.data_root),
            "databasePath": str(config.database_path) if config.database_path else "",
            "resultsPath": "results.json",
            "preferCuda": config.prefer_cuda,
        },
    }
