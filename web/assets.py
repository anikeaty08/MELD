"""Workspace preparation helpers for the MELD dashboard."""

from __future__ import annotations

from datetime import UTC, datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from meld.core.text_encoder import TextEncoderBackbone
from meld.models.backbone import (
    resnet18_imagenet,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
)
from web.catalog import BACKBONE_OPTIONS, DATASET_OPTIONS, find_dataset_option, normalize_dataset_key

PREP_STATE_FILE = ".meld_prepare_state.json"
WEB_REQUIREMENTS_PATH = Path("web") / "requirements.txt"

_REQUIREMENT_IMPORTS = {
    "datasets": "datasets",
    "fastapi": "fastapi",
    "numpy": "numpy",
    "scipy": "scipy",
    "torch": "torch",
    "transformers": "transformers",
    "torchvision": "torchvision",
    "continuum": "continuum",
    "uvicorn": "uvicorn",
}

_HF_DATASET_MAP: dict[str, tuple[str, str | None]] = {
    "AGNEWS": ("ag_news", None),
    "DBPEDIA": ("dbpedia_14", None),
    "YAHOOANSWERSNLP": ("yahoo_answers_topics", None),
}

_BACKBONE_FACTORIES = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet18_imagenet": resnet18_imagenet,
}


def _timestamp() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _import_ready(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _load_prep_state(data_root: Path) -> dict[str, Any]:
    path = data_root / PREP_STATE_FILE
    if not path.exists():
        return {"datasets": {}, "models": {}, "requirements": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"datasets": {}, "models": {}, "requirements": {}}


def _save_prep_state(data_root: Path, state: dict[str, Any]) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    path = data_root / PREP_STATE_FILE
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _record_state(data_root: Path, category: str, key: str, payload: dict[str, Any]) -> None:
    state = _load_prep_state(data_root)
    bucket = state.setdefault(category, {})
    bucket[key] = payload
    _save_prep_state(data_root, state)


def _requirements_status(project_root: Path) -> dict[str, Any]:
    items = []
    for package_name, import_name in _REQUIREMENT_IMPORTS.items():
        items.append(
            {
                "name": package_name,
                "importName": import_name,
                "ready": _import_ready(import_name),
            }
        )
    return {
        "ready": all(item["ready"] for item in items),
        "path": str(project_root / WEB_REQUIREMENTS_PATH),
        "items": items,
    }


def _check_text_dataset_cached(dataset_name: str, data_root: Path) -> dict[str, Any]:
    if not _import_ready("datasets"):
        return {
            "ready": False,
            "status": "dependency-missing",
            "detail": "Install the datasets package first.",
        }

    from datasets import DownloadConfig, load_dataset

    hf_name, hf_config = _HF_DATASET_MAP[normalize_dataset_key(dataset_name)]
    try:
        load_dataset(
            hf_name,
            hf_config,
            cache_dir=str(data_root),
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception as exc:
        return {
            "ready": False,
            "status": "missing",
            "detail": str(exc),
        }
    return {
        "ready": True,
        "status": "ready",
        "detail": "Cached locally.",
    }


def _check_image_dataset_cached(dataset_name: str, data_root: Path) -> dict[str, Any]:
    key = normalize_dataset_key(dataset_name)
    if key == "SYNTHETIC":
        return {
            "ready": True,
            "status": "built-in",
            "detail": "Synthetic data is generated at runtime.",
        }
    if key in ("TINYIMAGENET", "TINYIMAGENET200"):
        root = data_root / "tiny-imagenet-200"
        ready = (root / "train").exists() and (root / "val").exists()
        return {
            "ready": ready,
            "status": "manual" if not ready else "ready",
            "detail": "Extract tiny-imagenet-200 under the data root." if not ready else "Dataset folder detected.",
        }

    if key in ("CIFAR10", "CIFAR100") and not _import_ready("continuum"):
        return {
            "ready": False,
            "status": "dependency-missing",
            "detail": "Install continuum first.",
        }
    if key == "STL10" and not _import_ready("torchvision"):
        return {
            "ready": False,
            "status": "dependency-missing",
            "detail": "Install torchvision first.",
        }

    try:
        if key == "CIFAR10":
            from continuum.datasets import CIFAR10

            CIFAR10(data_path=str(data_root), train=True, download=False)
            CIFAR10(data_path=str(data_root), train=False, download=False)
        elif key == "CIFAR100":
            from continuum.datasets import CIFAR100

            CIFAR100(data_path=str(data_root), train=True, download=False)
            CIFAR100(data_path=str(data_root), train=False, download=False)
        elif key == "STL10":
            from torchvision.datasets import STL10

            STL10(str(data_root), split="train", download=False)
            STL10(str(data_root), split="test", download=False)
        else:
            return {
                "ready": False,
                "status": "unknown",
                "detail": "No readiness probe for this dataset.",
            }
    except Exception as exc:
        return {
            "ready": False,
            "status": "missing",
            "detail": str(exc),
        }

    return {
        "ready": True,
        "status": "ready",
        "detail": "Cached locally.",
    }


def _hf_model_cache_roots() -> list[Path]:
    roots: list[Path] = []
    if os.getenv("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.getenv("TRANSFORMERS_CACHE"):
        roots.append(Path(os.environ["TRANSFORMERS_CACHE"]))
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in roots:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _check_text_model_cached(model_name: str) -> dict[str, Any]:
    slug = f"models--{model_name.replace('/', '--')}"
    for root in _hf_model_cache_roots():
        if (root / slug).exists():
            return {
                "ready": True,
                "status": "ready",
                "detail": f"Found in {root}.",
            }
    return {
        "ready": False,
        "status": "missing",
        "detail": "Download required.",
    }


def _check_pretrained_image_weights() -> dict[str, Any]:
    if not _import_ready("torch"):
        return {
            "ready": False,
            "status": "dependency-missing",
            "detail": "Install torch first.",
        }

    import torch

    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    ready = any(checkpoints.glob("resnet18*.pth"))
    return {
        "ready": ready,
        "status": "ready" if ready else "missing",
        "detail": str(checkpoints),
    }


def inspect_workspace_readiness(project_root: Path, data_root: Path) -> dict[str, Any]:
    prep_state = _load_prep_state(data_root)
    datasets = []
    for option in DATASET_OPTIONS:
        if option.domain == "text":
            probe = _check_text_dataset_cached(option.id, data_root)
        else:
            probe = _check_image_dataset_cached(option.id, data_root)
        recorded = prep_state.get("datasets", {}).get(option.id)
        datasets.append(
            {
                "id": option.id,
                "label": option.label,
                "domain": option.domain,
                "autoDownload": option.auto_download,
                "ready": bool(probe["ready"] or (recorded and recorded.get("ready"))),
                "status": probe["status"],
                "detail": probe["detail"],
                "recorded": recorded,
            }
        )

    pretrained_status = _check_pretrained_image_weights()
    backbones = []
    for option in BACKBONE_OPTIONS:
        if option.id == "text_encoder":
            backbones.append(
                {
                    "id": option.id,
                    "label": option.label,
                    "ready": True,
                    "status": "code-ready",
                    "detail": option.note,
                }
            )
            continue
        ready = True
        status = "code-ready"
        detail = option.note
        if option.supports_pretrained:
            detail = pretrained_status["detail"]
            status = "pretrained-ready" if pretrained_status["ready"] else "pretrained-missing"
        backbones.append(
            {
                "id": option.id,
                "label": option.label,
                "ready": ready,
                "status": status,
                "detail": detail,
                "pretrainedReady": pretrained_status["ready"] if option.supports_pretrained else False,
            }
        )

    text_models = []
    for model_name in TextEncoderBackbone.supported_model_names():
        probe = _check_text_model_cached(model_name)
        recorded = prep_state.get("models", {}).get(f"text::{model_name}")
        text_models.append(
            {
                "id": model_name,
                "ready": bool(probe["ready"] or (recorded and recorded.get("ready"))),
                "status": probe["status"],
                "detail": probe["detail"],
                "recorded": recorded,
                "outDim": TextEncoderBackbone.SUPPORTED_MODELS[model_name],
            }
        )

    return {
        "requirements": _requirements_status(project_root),
        "datasets": datasets,
        "backbones": backbones,
        "textModels": text_models,
        "preparedState": prep_state,
    }


def _install_requirements(project_root: Path, data_root: Path) -> dict[str, Any]:
    requirements_path = project_root / WEB_REQUIREMENTS_PATH
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    completed = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
    payload = {
        "ready": completed.returncode == 0,
        "status": "ready" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "tail": lines[-20:],
        "finishedAt": _timestamp(),
    }
    _record_state(data_root, "requirements", str(WEB_REQUIREMENTS_PATH), payload)
    return payload


def _prepare_dataset(dataset_name: str, data_root: Path) -> dict[str, Any]:
    option = find_dataset_option(dataset_name)
    key = normalize_dataset_key(option.id)
    try:
        if key == "SYNTHETIC":
            payload = {
                "ready": True,
                "status": "built-in",
                "detail": "Synthetic data is always available.",
            }
        elif key == "CIFAR10":
            from continuum.datasets import CIFAR10

            CIFAR10(data_path=str(data_root), train=True, download=True)
            CIFAR10(data_path=str(data_root), train=False, download=True)
            payload = {"ready": True, "status": "ready", "detail": "CIFAR-10 downloaded."}
        elif key == "CIFAR100":
            from continuum.datasets import CIFAR100

            CIFAR100(data_path=str(data_root), train=True, download=True)
            CIFAR100(data_path=str(data_root), train=False, download=True)
            payload = {"ready": True, "status": "ready", "detail": "CIFAR-100 downloaded."}
        elif key == "STL10":
            from torchvision.datasets import STL10

            STL10(str(data_root), split="train", download=True)
            STL10(str(data_root), split="test", download=True)
            payload = {"ready": True, "status": "ready", "detail": "STL-10 downloaded."}
        elif key in ("TINYIMAGENET", "TINYIMAGENET200"):
            payload = {
                "ready": False,
                "status": "manual",
                "detail": "Tiny ImageNet still needs manual extraction.",
            }
        else:
            from datasets import load_dataset

            hf_name, hf_config = _HF_DATASET_MAP[key]
            raw = load_dataset(hf_name, hf_config, cache_dir=str(data_root))
            payload = {
                "ready": True,
                "status": "ready",
                "detail": f"Cached train/test splits ({len(raw['train'])}/{len(raw['test'])}).",
            }
    except Exception as exc:
        payload = {
            "ready": False,
            "status": "failed",
            "detail": str(exc),
        }

    payload["finishedAt"] = _timestamp()
    _record_state(data_root, "datasets", option.id, payload)
    return {"id": option.id, "label": option.label, **payload}


def _prepare_model_assets(
    *,
    backbone: str,
    pretrained_backbone: bool,
    text_models: list[str],
    data_root: Path,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if pretrained_backbone and backbone in _BACKBONE_FACTORIES:
        try:
            _BACKBONE_FACTORIES[backbone](pretrained=True)
            payload = {
                "id": f"backbone::{backbone}",
                "ready": True,
                "status": "ready",
                "detail": "Pretrained image weights are available.",
                "finishedAt": _timestamp(),
            }
        except Exception as exc:
            payload = {
                "id": f"backbone::{backbone}",
                "ready": False,
                "status": "failed",
                "detail": str(exc),
                "finishedAt": _timestamp(),
            }
        _record_state(data_root, "models", payload["id"], payload)
        reports.append(payload)

    for model_name in dict.fromkeys(text_models):
        try:
            payload = {
                "id": f"text::{model_name}",
                "ready": True,
                "status": "ready",
                "detail": TextEncoderBackbone(model_name=model_name).preload(),
                "finishedAt": _timestamp(),
            }
        except Exception as exc:
            payload = {
                "id": f"text::{model_name}",
                "ready": False,
                "status": "failed",
                "detail": str(exc),
                "finishedAt": _timestamp(),
            }
        _record_state(data_root, "models", payload["id"], payload)
        reports.append(payload)
    return reports


def prepare_workspace(
    *,
    project_root: Path,
    data_root: Path,
    install_requirements: bool,
    datasets: list[str],
    backbone: str,
    pretrained_backbone: bool,
    text_models: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "startedAt": _timestamp(),
        "requirements": None,
        "datasets": [],
        "models": [],
        "success": True,
    }

    if install_requirements:
        report["requirements"] = _install_requirements(project_root, data_root)
        report["success"] = bool(report["requirements"]["ready"])

    for dataset_name in dict.fromkeys(datasets):
        dataset_report = _prepare_dataset(dataset_name, data_root)
        report["datasets"].append(dataset_report)
        if dataset_report["status"] not in {"ready", "built-in", "manual"}:
            report["success"] = False

    model_reports = _prepare_model_assets(
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
        text_models=text_models,
        data_root=data_root,
    )
    report["models"].extend(model_reports)
    if any(not item["ready"] for item in model_reports):
        report["success"] = False

    report["finishedAt"] = _timestamp()
    return report
