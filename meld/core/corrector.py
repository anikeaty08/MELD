"""Bias correction for MELD classifier heads."""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from ..interfaces.base import BiasCorrector, TaskSnapshot


class AnalyticNormCorrector(BiasCorrector):
    def correct(self, model: nn.Module, snapshot: TaskSnapshot) -> nn.Module:
        if not snapshot.classifier_norms:
            return model
        target_norm = float(np.mean(list(snapshot.classifier_norms.values())))
        for class_id in model.classifier.class_to_head:
            weight = model.classifier.weight_vector(class_id)
            current_norm = float(weight.norm(p=2).item())
            if current_norm <= 0.0:
                continue
            gamma = target_norm / current_norm
            weight.data.mul_(gamma)
        return model
