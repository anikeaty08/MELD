"""Robustness evaluation helpers for MELD.

Currently implements a lightweight CIFAR-C evaluator using the common `.npy`
file layout:
  - `{corruption}.npy` with shape [50000, 32, 32, 3]
  - `labels.npy` with shape [10000]

If CIFAR-C files are not present locally, evaluation is skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _normalize_cifar(images: np.ndarray, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    """Convert uint8 CIFAR-C images to normalized torch tensor."""
    # CIFAR-C commonly stores uint8 in [0, 255] with shape [N, H, W, C].
    x = images.astype(np.float32)
    if x.max() > 2.0:
        x = x / 255.0
    x = np.transpose(x, (0, 3, 1, 2))  # [N, C, H, W]
    t = torch.from_numpy(x)
    mean_t = torch.tensor(mean, dtype=t.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype).view(1, 3, 1, 1)
    return (t - mean_t) / std_t


def evaluate_cifar_c(
    model: torch.nn.Module,
    *,
    dataset: str,
    data_root: Path,
    device: torch.device,
    corruptions: list[str] | None = None,
    severity: int = 5,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Evaluate a trained MELD model on local CIFAR-C .npy files.

    Returns a dict that can be directly embedded into `results.json`.
    """
    d = dataset.upper().replace("-", "")
    if d not in {"CIFAR10", "CIFAR100"}:
        return {"status": "skipped", "reason": f"Unsupported dataset for CIFAR-C: {dataset}"}

    corruption_types = corruptions or ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur"]
    cifar_c_dir = Path(data_root) / f"{d}-C"

    labels_path = cifar_c_dir / "labels.npy"
    if not labels_path.exists():
        return {
            "status": "skipped",
            "reason": f"CIFAR-C not found locally (missing {labels_path}).",
        }

    labels = np.load(labels_path).astype(np.int64)
    if labels.shape[0] < 10000:
        return {"status": "skipped", "reason": f"Unexpected labels.npy shape: {labels.shape}"}

    mean = _CIFAR10_MEAN if d == "CIFAR10" else _CIFAR100_MEAN
    std = _CIFAR10_STD if d == "CIFAR10" else _CIFAR100_STD

    model.eval()
    results: dict[str, Any] = {"status": "running", "severity": severity, "corruptions": {}}

    num_per_severity = 10000
    start = (severity - 1) * num_per_severity
    end = start + num_per_severity

    with torch.no_grad():
        for corr in corruption_types:
            path = cifar_c_dir / f"{corr}.npy"
            if not path.exists():
                results["corruptions"][corr] = {"status": "skipped", "reason": f"Missing {path}"}
                continue

            arr = np.load(path)
            if arr.shape[0] < end:
                results["corruptions"][corr] = {
                    "status": "skipped",
                    "reason": f"Corruption array too small: {arr.shape} (need index up to {end})",
                }
                continue

            images = arr[start:end]
            x = _normalize_cifar(images, mean=mean, std=std).to(device)
            y = torch.from_numpy(labels[:num_per_severity]).to(device)

            preds_top1: list[torch.Tensor] = []
            for i in range(0, x.size(0), batch_size):
                batch_x = x[i : i + batch_size]
                logits = model(batch_x)
                pred = logits.argmax(dim=1)
                preds_top1.append(pred)

            pred_all = torch.cat(preds_top1)
            acc = float((pred_all == y).float().mean().item())
            results["corruptions"][corr] = {"top1": acc}

    # Mean over successfully evaluated corruptions.
    accs = [
        v["top1"]
        for v in results["corruptions"].values()
        if isinstance(v, dict) and "top1" in v
    ]
    results["mean_top1"] = float(np.mean(accs)) if accs else None
    results["status"] = "completed"
    return results

