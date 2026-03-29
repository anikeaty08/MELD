"""
Calibration Tracker — ECE measurement and temperature scaling.

Temperature scaling (Guo et al. 2017) is a post-hoc calibration
method that learns a single scalar T to divide logits by.
It does not change accuracy — only confidence calibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class CalibrationTracker:

    def __init__(self) -> None:
        self._temperature: float = 1.0

    @property
    def temperature(self) -> float:
        return self._temperature

    def compute_ece(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        n_bins: int = 10,
    ) -> float:
        all_confidences: list[float] = []
        all_correct: list[float] = []
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    continue
                if isinstance(inputs, dict):
                    inputs = {
                        k: v.to(device) if isinstance(v, Tensor) else v
                        for k, v in inputs.items()
                    }
                else:
                    inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=-1)
                confidence, predicted = probs.max(dim=-1)
                correct = predicted.eq(targets)
                all_confidences.extend(confidence.cpu().numpy().tolist())
                all_correct.extend(correct.cpu().numpy().tolist())
        if not all_confidences:
            return float("nan")
        confidences = np.array(all_confidences)
        correct_arr = np.array(all_correct, dtype=float)
        n = len(confidences)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = correct_arr[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_weight = mask.sum() / n
            ece += abs(bin_acc - bin_conf) * bin_weight
        return float(ece)

    def fit_temperature(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Learn optimal temperature T via NLL minimization on logits.

        Returns the fitted temperature value.
        """
        all_logits: list[Tensor] = []
        all_targets: list[Tensor] = []
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    continue
                if isinstance(inputs, dict):
                    inputs = {
                        k: v.to(device) if isinstance(v, Tensor) else v
                        for k, v in inputs.items()
                    }
                else:
                    inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
        if not all_logits:
            self._temperature = 1.0
            return 1.0

        logits_cat = torch.cat(all_logits)
        targets_cat = torch.cat(all_targets)

        # Optimize log(T) so T is always positive
        log_T = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            T = log_T.exp()
            scaled = logits_cat / T
            loss = torch.nn.functional.cross_entropy(scaled, targets_cat)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._temperature = float(log_T.exp().item())
        # Clamp to reasonable range
        self._temperature = max(0.1, min(self._temperature, 10.0))
        return self._temperature

    def apply_temperature(self, logits: Tensor) -> Tensor:
        """Scale logits by the fitted temperature."""
        return logits / max(self._temperature, 1e-6)
