import sys
from types import ModuleType

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from meld.api import MELDConfig, TrainConfig
from meld.benchmarks.runner import BenchmarkRunner
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


class _FakeSplit:
    def __init__(self, rows: dict[str, list[object]]) -> None:
        self._rows = rows
        self.column_names = list(rows)

    def __getitem__(self, key: str):
        return self._rows[key]

    def __len__(self) -> int:
        return len(next(iter(self._rows.values())))


def test_runner_preserves_explicit_backbone_choice():
    runner = BenchmarkRunner(
        MELDConfig(
            dataset="CIFAR-100",
            train=TrainConfig(backbone="resnet32", epochs=1, batch_size=2),
        )
    )

    model = runner._build_model()

    assert len(model.backbone.stage_1) == 5


def test_runner_uses_configured_text_encoder_for_tokenization(monkeypatch):
    captured: dict[str, str] = {}

    class _FakeTextEncoder:
        def __init__(self, model_name: str, **_: object) -> None:
            captured["model_name"] = model_name

        def get_tokenizer(self):
            def _tokenizer(texts, **_: object):
                n = len(texts)
                return {
                    "input_ids": torch.ones((n, 4), dtype=torch.long),
                    "attention_mask": torch.ones((n, 4), dtype=torch.long),
                }

            return _tokenizer

    datasets_module = ModuleType("datasets")

    def _load_dataset(name: str, config: object, cache_dir: str):
        return {
            "train": _FakeSplit({"text": ["alpha", "beta"], "label": [0, 1]}),
            "test": _FakeSplit({"text": ["gamma", "delta"], "label": [0, 1]}),
        }

    datasets_module.load_dataset = _load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    import meld.core.text_encoder as text_encoder_module

    monkeypatch.setattr(text_encoder_module, "TextEncoderBackbone", _FakeTextEncoder)

    runner = BenchmarkRunner(
        MELDConfig(
            dataset="AGNews",
            num_tasks=1,
            classes_per_task=2,
            train=TrainConfig(
                backbone="text_encoder",
                text_encoder_model="sentence-transformers/all-mpnet-base-v2",
                batch_size=2,
            ),
        )
    )

    bundle = runner._nlp_bundle("AGNEWS")

    assert captured["model_name"] == "sentence-transformers/all-mpnet-base-v2"
    assert len(bundle) == 1


def test_runner_evaluate_accepts_dict_batches():
    runner = BenchmarkRunner(
        MELDConfig(
            dataset="synthetic",
            num_tasks=1,
            classes_per_task=2,
            train=TrainConfig(backbone="resnet20", batch_size=2, epochs=1),
        )
    )
    classifier = IncrementalClassifier(4)
    classifier.adaption(2)
    model = MELDModel(_DummyTextBackbone(), classifier)
    loader = DataLoader(_DummyTextDataset(), batch_size=2)

    metrics = runner._evaluate(model, [loader])

    assert metrics["top1"] >= 0.0
    assert metrics["confusion_matrix"].shape == (2, 2)
