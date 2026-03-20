"""Optional Avalanche baseline integration for MELD results."""

from __future__ import annotations

from typing import Any


def run_avalanche_baselines(config: Any, device: Any) -> dict[str, Any]:
    """Run Avalanche baselines when dependency is installed.

    This function is intentionally best-effort: if Avalanche is unavailable,
    unsupported for the selected dataset, or runtime setup fails, MELD still
    completes and reports a structured skip reason.
    """
    try:
        import avalanche  # noqa: F401
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"avalanche-lib not installed ({exc})",
            "requested": ["EWC", "iCaRL", "DER"],
        }

    dataset = str(getattr(config, "dataset", "")).upper().replace("-", "")
    if dataset not in {"CIFAR10", "CIFAR100"}:
        return {
            "status": "skipped",
            "reason": f"Dataset '{getattr(config, 'dataset', '')}' not yet wired for Avalanche scenario builder",
            "requested": ["EWC", "iCaRL", "DER"],
        }

    # Placeholder for full Avalanche scenario/strategy execution.
    # Kept explicit so downstream tooling can distinguish between dependency
    # absence and partial integration.
    return {
        "status": "pending_implementation",
        "device": str(device),
        "dataset": getattr(config, "dataset", ""),
        "requested": ["EWC", "iCaRL", "DER"],
        "reason": "Dependency is available; complete strategy wiring can be added without changing results schema.",
    }
