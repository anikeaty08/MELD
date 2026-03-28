"""
delta — Provably Equivalent Incremental Learning Framework.

Structured like Avalanche/Mammoth: streams, strategies, plugins.

Usage (new API):
    from delta import (DeltaStream, FisherDeltaStrategy,
        EvaluationPlugin, accuracy_metrics, InteractiveLogger)

    stream = DeltaStream("synthetic", n_tasks=3)
    strategy = FisherDeltaStrategy(model, optimizer, criterion)
    for exp in stream.train_stream:
        strategy.train(exp)
        strategy.eval(stream.test_stream)

Legacy API (still works):
    from delta import DeltaTrainer
    trainer = DeltaTrainer(model)
    trainer.fit(loader)
    cert = trainer.fit_delta(new_loader)
"""

# Core mathematical components
from .core.state import DeltaState
from .core.certificate import EquivalenceCertificate
from .core.fisher import KFACComputer
from .core.shift import ShiftDetector
from .core.calibration import CalibrationTracker

# Benchmark streams
from .benchmarks.stream import DeltaStream, Experience
from .benchmarks.dataset_base import ContinualDataset, register_dataset

# Training strategies
from .training.base import BaseStrategy
from .training.fisher_delta import FisherDeltaStrategy
from .training.full_retrain import FullRetrainStrategy
from .training.replay_delta import ReplayDeltaStrategy
from .training import DeltaStrategy, ReplayStrategy

# Evaluation
from .evaluation.plugin import EvaluationPlugin
from .evaluation.metrics import (
    accuracy_metrics,
    equivalence_metrics,
    calibration_metrics,
    forgetting_metrics,
    compute_metrics,
)

# Logging
from .logging.interactive import InteractiveLogger
from .logging.csv_logger import CSVLogger

# Legacy API — backward compatible
from .trainer import DeltaTrainer

__all__ = [
    # Core
    "DeltaState",
    "EquivalenceCertificate",
    "KFACComputer",
    "ShiftDetector",
    "CalibrationTracker",
    # Streams
    "DeltaStream",
    "Experience",
    "ContinualDataset",
    "register_dataset",
    # Strategies
    "BaseStrategy",
    "DeltaStrategy",
    "FisherDeltaStrategy",
    "ReplayDeltaStrategy",
    "ReplayStrategy",
    "FullRetrainStrategy",
    # Evaluation
    "EvaluationPlugin",
    "accuracy_metrics",
    "equivalence_metrics",
    "calibration_metrics",
    "forgetting_metrics",
    "compute_metrics",
    # Logging
    "InteractiveLogger",
    "CSVLogger",
    # Legacy
    "DeltaTrainer",
]

__version__ = "0.3.0"

# Commit 13 - 2026-03-29 08:20:52

# Commit 17 - 2026-03-29 08:41:05

# Commit 31 - 2026-03-28 20:42:36

# Commit 33 - 2026-03-29 06:23:03

# Commit 34 - 2026-03-29 06:28:33

# Update 30 - 2026-03-28 20:08:18
# Update 34 - 2026-03-29 01:08:27
# Update 26 @ 2026-03-29 00:00:29
# Update 29 @ 2026-03-29 09:43:39
# Update 34 @ 2026-03-28 17:25:47
# Update 6 @ 2026-03-28 20:20:51
# Update 16 @ 2026-03-29 04:17:07
# Update 32 @ 2026-03-28 10:53:28
# Update 35 @ 2026-03-29 03:50:36
# Update 4 @ 2026-03-29 09:29:18
# Update 15 @ 2026-03-28 15:08:55
# Update 22 @ 2026-03-29 01:42:55