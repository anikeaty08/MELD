"""Training strategies for continual learning."""

from .base import BaseStrategy
from .fisher_delta import FisherDeltaStrategy
from .full_retrain import FullRetrainStrategy
from .replay_delta import ReplayDeltaStrategy

DeltaStrategy = ReplayDeltaStrategy
ReplayStrategy = ReplayDeltaStrategy

__all__ = [
    "BaseStrategy",
    "DeltaStrategy",
    "FisherDeltaStrategy",
    "ReplayDeltaStrategy",
    "ReplayStrategy",
    "FullRetrainStrategy",
]
