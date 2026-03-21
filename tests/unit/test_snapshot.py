import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from meld.core.snapshot import FisherManifoldSnapshot
from meld.modeling import MELDModel
from meld.models.classifier import IncrementalClassifier


class _DummyTextBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 4)
        self.out_dim = 4

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.proj(input_ids.float() * attention_mask.float())


class _DummyTextDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        token = float(index + 1)
        return {
            "input_ids": torch.tensor([token, token, token], dtype=torch.float32),
            "attention_mask": torch.ones(3, dtype=torch.float32),
        }, index


def test_snapshot_capture_accepts_dict_batches():
    classifier = IncrementalClassifier(4)
    classifier.adaption(2)
    model = MELDModel(_DummyTextBackbone(), classifier)
    loader = DataLoader(_DummyTextDataset(), batch_size=2)

    snapshot = FisherManifoldSnapshot(fisher_samples=2, anchors_per_class=1).capture(
        model,
        loader,
        [0, 1],
        0,
    )

    assert snapshot.class_ids == [0, 1]
    assert snapshot.class_anchor_inputs[0].size == 0
    assert snapshot.input_feature_mean.shape == (4,)
