from meld.api import TrainConfig
from meld.delta import DeltaModel


def test_delta_model_from_scratch_exposes_public_api():
    model = DeltaModel.from_scratch(
        num_classes=2,
        backbone="resnet20",
        prefer_cuda=False,
        train_config=TrainConfig(backbone="resnet20", epochs=1, batch_size=2),
    )

    summary = model.summary()

    assert "DeltaModel" in summary
    assert "classes=2" in summary
