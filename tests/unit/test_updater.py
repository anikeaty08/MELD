import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from meld.api import TrainConfig
from meld.core import updater as updater_module
from meld.core.updater import GeometryConstrainedUpdater
from meld.modeling import MELDModel
from meld.models.classifier import IncrementalClassifier


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(12, 4)
        self.out_dim = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.flatten(x))


def test_geometry_updater_uses_configured_mixup_alpha(monkeypatch):
    seen_alphas: list[float] = []

    def _fake_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
        seen_alphas.append(alpha)
        return x, y, y, 1.0

    monkeypatch.setattr(updater_module, "_mixup", _fake_mixup)

    classifier = IncrementalClassifier(4)
    classifier.adaption(2)
    model = MELDModel(_TinyBackbone(), classifier)
    loader = DataLoader(
        TensorDataset(
            torch.randn(4, 3, 2, 2),
            torch.tensor([0, 1, 0, 1], dtype=torch.long),
        ),
        batch_size=2,
        shuffle=False,
    )

    updater = GeometryConstrainedUpdater()
    updater.update(
        model,
        loader,
        None,
        TrainConfig(
            backbone="custom",
            epochs=1,
            batch_size=2,
            mixup_alpha=0.35,
            cutmix_alpha=0.9,
        ),
    )

    assert seen_alphas
    assert all(abs(alpha - 0.35) < 1e-6 for alpha in seen_alphas)
