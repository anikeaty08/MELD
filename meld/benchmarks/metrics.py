"""Evaluation metrics for MELD benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        acc = float(accuracies[mask].mean())
        conf = float(confidences[mask].mean())
        ece += abs(acc - conf) * float(mask.mean())
    return float(ece)


def compute_ece_maybe(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float | None:
    """Compute ECE for a subset.

    Returns `None` if the subset is empty.
    """
    if labels.size == 0:
        return None
    return compute_ece(probs=probs, labels=labels, num_bins=num_bins)


def compute_classification_metrics(logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> dict[str, object]:
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results: dict[str, object] = {}
    for k in topk:
        actual_k = min(k, logits.size(1))
        correct_k = correct[:actual_k].reshape(-1).float().sum(0)
        results[f"top{k}"] = float(correct_k.item() / targets.size(0))

    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    y_pred = probabilities.argmax(axis=1)
    num_classes = logits.size(1)
    confusion = compute_confusion_matrix(y_true, y_pred, num_classes)
    per_class = {}
    for class_id in range(num_classes):
        denom = confusion[class_id].sum()
        per_class[class_id] = float(confusion[class_id, class_id] / denom) if denom else 0.0
    results["ece"] = compute_ece(probabilities, y_true)
    results["confusion_matrix"] = confusion
    results["per_class_accuracy"] = per_class
    return results


def _to_numpy_confusion(confusion: np.ndarray) -> np.ndarray:
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Confusion matrix must be square; got shape {matrix.shape}")
    return matrix


def confusion_cosine_similarity(confusion_a: np.ndarray, confusion_b: np.ndarray) -> float:
    matrix_a = _to_numpy_confusion(confusion_a)
    matrix_b = _to_numpy_confusion(confusion_b)
    if matrix_a.shape != matrix_b.shape:
        n = min(matrix_a.shape[0], matrix_b.shape[0])
        matrix_a = matrix_a[:n, :n]
        matrix_b = matrix_b[:n, :n]
    matrix_a = matrix_a / max(float(matrix_a.sum()), 1.0)
    matrix_b = matrix_b / max(float(matrix_b.sum()), 1.0)
    flat_a = matrix_a.reshape(-1)
    flat_b = matrix_b.reshape(-1)
    denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    if denom == 0.0:
        return 1.0
    return float(np.dot(flat_a, flat_b) / denom)


def compute_equivalence_gap(confusion_a: np.ndarray, confusion_b: np.ndarray) -> float:
    return float(1.0 - confusion_cosine_similarity(confusion_a, confusion_b))


def compute_compute_savings(delta_time: float, full_time: float) -> float:
    if full_time <= 0.0:
        return 0.0
    return float(((full_time - delta_time) / full_time) * 100.0)
