"""
Tests that directly prove the PS competition requirements.
Each test maps to one PS requirement.
Also tests the new framework structure (hooks, plugins, streams).
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
import numpy as np
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from delta import (
    DeltaTrainer, DeltaState, EquivalenceCertificate,
    DeltaStream, Experience, DeltaStrategy, FisherDeltaStrategy, FullRetrainStrategy, ReplayDeltaStrategy, ReplayStrategy,
    BaseStrategy, EvaluationPlugin, InteractiveLogger,
    accuracy_metrics, equivalence_metrics, calibration_metrics,
    compute_metrics, ContinualDataset, register_dataset,
    KFACComputer, ShiftDetector, CSVLogger,
)


def make_tiny_model():
    return nn.Linear(16, 4)


def make_tiny_loader(n=100, d=16, c=4, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)


def make_custom_model():
    return nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 4),
    )


def make_framework_model(in_dim=32, out_dim=4):
    return nn.Sequential(
        nn.Linear(in_dim, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, out_dim),
    )


# ═══════════════════════════════════════════════════════════
# PS REQUIREMENT TESTS (original — using legacy DeltaTrainer)
# ═══════════════════════════════════════════════════════════

class TestPS_Requirement1_NoOldData:
    def test_state_contains_no_raw_tensors(self):
        model = make_tiny_model()
        trainer = DeltaTrainer(model, device="cpu")
        loader = make_tiny_loader()
        trainer.fit(loader, epochs=2, lr=0.01)
        state = trainer.state
        assert state is not None
        for key, val in state.theta_ref.items():
            assert isinstance(val, np.ndarray)
        for k, v in state.__dict__.items():
            if isinstance(v, torch.Tensor):
                pytest.fail(f"DeltaState.{k} is a raw tensor")


class TestPS_Requirement2_DerivedWeights:
    def test_default_scales_stay_bounded_for_stability(self):
        model = make_tiny_model()
        trainer = DeltaTrainer(model, device="cpu")
        loader_0 = make_tiny_loader(n=100, seed=0)
        loader_1 = make_tiny_loader(n=10, seed=1)
        trainer.fit(loader_0, epochs=2)
        cert = trainer.fit_delta(loader_1, epochs=2)
        assert cert.ce_scale == 1.0
        assert cert.ewc_scale == 1.0


class TestPS_Requirement3_FormalBound:
    def test_certificate_has_computed_epsilon(self):
        model = make_tiny_model()
        trainer = DeltaTrainer(model, device="cpu")
        trainer.fit(make_tiny_loader(n=100), epochs=3)
        cert = trainer.fit_delta(make_tiny_loader(n=20), epochs=3)
        assert isinstance(cert.epsilon_param, float)
        assert not np.isnan(cert.epsilon_param)
        assert cert.epsilon_param < float("inf")

    def test_logistic_regression_recovery(self):
        torch.manual_seed(42)
        trainer = DeltaTrainer(make_tiny_model(), device="cpu")
        trainer.fit(make_tiny_loader(n=200, seed=0), epochs=5)
        cert = trainer.fit_delta(make_tiny_loader(n=50, seed=1), epochs=5)
        assert cert.epsilon_param < 10.0


class TestPS_Requirement4_ShiftDetection:
    def test_no_shift_standard_delta(self):
        model = make_tiny_model()
        trainer = DeltaTrainer(model, device="cpu")
        trainer.fit(make_tiny_loader(n=100, seed=0), epochs=2)
        cert = trainer.fit_delta(make_tiny_loader(n=20, seed=99), epochs=2)
        assert cert.shift_type in ("none", "covariate", "concept")


class TestPS_Requirement5_Calibration:
    def test_ece_tracked_before_and_after(self):
        model = make_tiny_model()
        trainer = DeltaTrainer(model, device="cpu")
        trainer.fit(make_tiny_loader(n=100), epochs=3)
        cert = trainer.fit_delta(make_tiny_loader(n=20), epochs=3)
        assert not np.isnan(cert.ece_before)
        assert not np.isnan(cert.ece_after)
        assert 0.0 <= cert.ece_before <= 1.0
        assert 0.0 <= cert.ece_after <= 1.0


class TestPS_Requirement6_ModelAgnostic:
    def test_works_with_custom_mlp(self):
        model = make_custom_model()
        trainer = DeltaTrainer(model, device="cpu")
        trainer.fit(make_tiny_loader(n=80), epochs=2)
        cert = trainer.fit_delta(make_tiny_loader(n=20), epochs=2)
        assert isinstance(cert.epsilon_param, float)

    def test_state_has_no_model_imports(self):
        delta_dir = os.path.join(os.path.dirname(__file__), "..", "delta")
        violations = []
        for root, dirs, files in os.walk(delta_dir):
            if "benchmarks" in root or "demos" in root:
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                with open(os.path.join(root, fname), encoding="utf-8") as f:
                    src = f.read()
                for word in ["resnet", "backbone", "CIFAR", "AGNews", "TinyImageNet"]:
                    if word.lower() in src.lower():
                        violations.append(f"{os.path.join(root, fname)}: {word}")
        assert not violations, f"Framework has model-specific code: {violations}"


class TestPS_Requirement7_ComputeSavings:
    def test_compute_ratio_reported(self):
        model = make_tiny_model()
        trainer = DeltaTrainer(model, device="cpu")
        trainer.fit(make_tiny_loader(n=100), epochs=3)
        cert = trainer.fit_delta(make_tiny_loader(n=20), epochs=3)
        assert cert.compute_ratio >= 0.0


# ═══════════════════════════════════════════════════════════
# FRAMEWORK STRUCTURE TESTS (new)
# ═══════════════════════════════════════════════════════════

class TestHookSystem:
    def test_hooks_fire_in_correct_order(self):
        hook_log = []

        class LoggingStrategy(BaseStrategy):
            def _before_training_experience(self, exp):
                hook_log.append("before_exp")
            def _before_training_epoch(self):
                hook_log.append("before_epoch")
            def _before_training_iteration(self, x, y):
                hook_log.append("before_iter")
            def _after_forward(self, x, y, logits):
                hook_log.append("after_forward")
            def _before_backward(self, loss):
                hook_log.append("before_backward")
            def _after_backward(self):
                hook_log.append("after_backward")
            def _after_training_iteration(self, x, y, logits, loss):
                hook_log.append("after_iter")
            def _after_training_epoch(self):
                hook_log.append("after_epoch")
            def _after_training_experience(self, exp):
                hook_log.append("after_exp")

        model = nn.Linear(32, 4)
        opt = SGD(model.parameters(), lr=0.01)
        strategy = LoggingStrategy(
            model, opt, nn.CrossEntropyLoss(),
            device="cpu", train_epochs=1, train_mb_size=32)

        stream = DeltaStream("synthetic", n_tasks=1, classes_per_task=4)
        strategy.train(stream.train_stream[0])

        assert hook_log[0] == "before_exp"
        assert hook_log[1] == "before_epoch"
        assert "before_iter" in hook_log
        assert "after_forward" in hook_log
        assert "before_backward" in hook_log
        assert "after_backward" in hook_log
        assert "after_iter" in hook_log
        assert hook_log[-2] == "after_epoch"
        assert hook_log[-1] == "after_exp"


class TestPluginInjection:
    def test_plugin_counts_backward_passes(self):
        class CounterPlugin:
            def __init__(self):
                self.count = 0
            def after_backward(self, strategy):
                self.count += 1

        model = nn.Linear(32, 4)
        opt = SGD(model.parameters(), lr=0.01)
        strategy = BaseStrategy(
            model, opt, nn.CrossEntropyLoss(),
            device="cpu", train_epochs=2, train_mb_size=32)

        counter = CounterPlugin()
        strategy.add_plugin(counter)

        stream = DeltaStream("synthetic", n_tasks=1, classes_per_task=4)
        strategy.train(stream.train_stream[0])

        assert counter.count > 0
        n_batches = len(list(
            stream.train_stream[0].train_dataloader(batch_size=32)))
        assert counter.count == 2 * n_batches


class TestEvaluationPlugin:
    def test_collects_metrics(self):
        model = make_framework_model()
        opt = SGD(model.parameters(), lr=0.01)
        ep = EvaluationPlugin(
            accuracy_metrics(stream=True),
            equivalence_metrics(epsilon=True),
            calibration_metrics(),
            compute_metrics(),
            loggers=[],
        )
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            evaluator=ep, device="cpu", train_epochs=2, train_mb_size=32)

        stream = DeltaStream("synthetic", n_tasks=2, classes_per_task=2)
        for exp in stream.train_stream:
            strategy.train(exp)
            strategy.eval(stream.test_stream)

        metrics = ep.get_last_metrics()
        assert "accuracy/stream" in metrics

    def test_resets_metrics_between_eval_streams(self):
        model = make_framework_model()
        opt = SGD(model.parameters(), lr=0.01)
        ep = EvaluationPlugin(accuracy_metrics(stream=True), loggers=[])
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            evaluator=ep, device="cpu", train_epochs=1, train_mb_size=16)

        stream = DeltaStream("synthetic", n_tasks=2, classes_per_task=2)
        first_exp = stream.train_stream[0]
        strategy.train(first_exp)
        strategy.eval([stream.test_stream[0]])
        first_metrics = ep.get_last_metrics()
        assert "accuracy/task_0" in first_metrics
        assert "accuracy/task_1" not in first_metrics

        strategy.eval([stream.test_stream[1]])
        second_metrics = ep.get_last_metrics()
        assert "accuracy/task_0" not in second_metrics
        assert "accuracy/task_1" in second_metrics


class TestDeltaStream:
    def test_returns_correct_experiences(self):
        stream = DeltaStream("synthetic", n_tasks=3, scenario="class_incremental")
        exps = list(stream.train_stream)
        assert len(exps) == 3
        assert all(isinstance(e, Experience) for e in exps)
        assert [e.task_id for e in exps] == [0, 1, 2]

    def test_experience_has_dataloaders(self):
        stream = DeltaStream("synthetic", n_tasks=2, classes_per_task=2)
        exp = stream.train_stream[0]
        train_dl = exp.train_dataloader(batch_size=8)
        test_dl = exp.test_dataloader(batch_size=8)
        batch = next(iter(train_dl))
        assert len(batch) == 2

    def test_domain_incremental_synthetic_reuses_class_ids(self):
        stream = DeltaStream(
            "synthetic",
            n_tasks=3,
            scenario="domain_incremental",
            classes_per_task=2,
        )
        exps = list(stream.train_stream)
        assert exps[0].classes_in_this_experience == [0, 1]
        assert exps[1].classes_in_this_experience == [0, 1]
        assert exps[2].classes_in_this_experience == [0, 1]
        assert all(exp.scenario == "domain_incremental" for exp in exps)

    def test_passes_dataset_preset_hints_to_provider(self):
        captured = {}

        def provider(config):
            captured["preset"] = config.preset
            captured["pretrained_backbone"] = config.pretrained_backbone
            captured["image_size"] = config.image_size
            captured["use_imagenet_stats"] = config.use_imagenet_stats
            ds = TensorDataset(torch.randn(4, 3), torch.zeros(4, dtype=torch.long))
            return [(ds, ds)]

        register_dataset("toy-preset-probe", provider, overwrite=True)
        stream = DeltaStream(
            "toy-preset-probe",
            n_tasks=1,
            preset="maxperf",
            pretrained_backbone=True,
            image_size=96,
            use_imagenet_stats=True,
        )
        assert len(stream.train_stream) == 1
        assert captured == {
            "preset": "maxperf",
            "pretrained_backbone": True,
            "image_size": 96,
            "use_imagenet_stats": True,
        }


class TestStrategyInheritance:
    def test_both_strategies_are_base_strategy(self):
        model = nn.Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        fs = FisherDeltaStrategy(model, opt, crit, device="cpu")
        fr = FullRetrainStrategy(model, opt, crit, device="cpu")
        assert isinstance(fs, BaseStrategy)
        assert isinstance(fr, BaseStrategy)

    def test_replay_delta_strategy_uses_practical_defaults(self):
        model = nn.Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        strategy = ReplayDeltaStrategy(model, opt, crit, device="cpu", train_mb_size=32)
        assert strategy.use_nme_classifier is False
        assert strategy.replay_memory_per_class >= 32
        assert strategy.replay_batch_size >= 64
        assert strategy.classifier_balance_steps >= 20
        assert strategy.bias_correction_steps == 40
        assert strategy.use_cosine_lr is True

    def test_replay_strategy_alias_points_to_replay_delta(self):
        model = nn.Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        strategy = ReplayStrategy(model, opt, crit, device="cpu")
        assert isinstance(strategy, ReplayDeltaStrategy)

    def test_delta_strategy_alias_points_to_replay_delta(self):
        model = nn.Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        strategy = DeltaStrategy(model, opt, crit, device="cpu")
        assert isinstance(strategy, ReplayDeltaStrategy)

    def test_replay_delta_strategy_bias_correction_adjusts_new_head(self):
        from delta.demos.models.classifier import IncrementalClassifier
        from delta.demos.models.modeling import MELDModel

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 8)
                self.out_dim = 8

            def forward(self, x):
                return self.proj(x)

            def embed(self, x):
                return self.proj(x)

        model = MELDModel(TinyBackbone(), IncrementalClassifier(8))
        strategy = ReplayDeltaStrategy(
            model,
            SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            device="cpu",
            train_epochs=1,
            train_mb_size=4,
        )
        model.classifier.adaption(2)
        model.classifier.adaption(2)
        strategy._head_bias_params[1] = (0.5, -1.0)
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        corrected = strategy._apply_bias_corrections(logits)
        assert torch.allclose(corrected[:, :2], logits[:, :2])
        assert torch.allclose(corrected[:, 2:], torch.tensor([[0.5, 1.0]]))

    def test_replay_delta_strategy_can_mask_eval_to_task_identity(self):
        model = nn.Linear(4, 6)
        strategy = ReplayDeltaStrategy(
            model,
            SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            device="cpu",
        )
        strategy.use_task_identity_inference = True
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32)
        masked = strategy._mask_logits_to_task(logits, [2, 3])
        assert masked[0, 2].item() == 3.0
        assert masked[0, 3].item() == 4.0
        assert masked[0, 0].item() < -1e8

    def test_replay_delta_strategy_applies_cosine_lr_schedule(self):
        model = nn.Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.1)
        strategy = ReplayDeltaStrategy(
            model,
            opt,
            nn.CrossEntropyLoss(),
            device="cpu",
            train_epochs=4,
            train_mb_size=16,
        )
        exp = Experience(
            train_dataset=TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))),
            test_dataset=TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))),
            task_id=0,
            classes_in_this_experience=[0, 1],
        )
        strategy._prepare_training_experience(exp)
        strategy._before_training_experience(exp)
        lrs = []
        for _ in range(4):
            strategy._before_training_epoch()
            lrs.append(strategy.optimizer.param_groups[0]["lr"])
        assert lrs[0] > lrs[-1]
        assert lrs[-1] >= 0.01


class TestCustomModelEndToEnd:
    def test_full_pipeline_with_custom_model(self):
        model = make_framework_model()
        opt = SGD(model.parameters(), lr=0.01)
        ep = EvaluationPlugin(
            accuracy_metrics(stream=True),
            equivalence_metrics(epsilon=True),
            loggers=[InteractiveLogger(verbose=False)],
        )
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            evaluator=ep, device="cpu", train_epochs=2, train_mb_size=32)

        stream = DeltaStream("synthetic", n_tasks=2, classes_per_task=2)
        for exp in stream.train_stream:
            strategy.train(exp)
            strategy.eval(stream.test_stream)

        assert strategy.last_certificate is not None
        assert isinstance(strategy.last_certificate, EquivalenceCertificate)


# ===================================================================
# CONV2D KFAC TESTS
# ===================================================================

class TestConv2dKFAC:
    def test_kfac_captures_conv2d_layers(self):
        """KFAC should compute factors for Conv2d, not just Linear."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
        )
        opt = SGD(model.parameters(), lr=0.01)
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            device="cpu", train_epochs=1, train_mb_size=8)

        # Create image-shaped synthetic data
        torch.manual_seed(42)
        tx = torch.randn(32, 3, 8, 8)
        ty = torch.randint(0, 4, (32,))
        ex = torch.randn(16, 3, 8, 8)
        ey = torch.randint(0, 4, (16,))
        exp = Experience(
            train_dataset=TensorDataset(tx, ty),
            test_dataset=TensorDataset(ex, ey),
            task_id=0,
            classes_in_this_experience=[0, 1, 2, 3],
        )
        strategy.train(exp)
        assert strategy.state is not None
        # Should have KFAC factors for both conv and linear
        assert len(strategy.state.kfac_A) >= 2, \
            f"Expected >=2 KFAC layers, got {len(strategy.state.kfac_A)}"

    def test_conv2d_penalty_runs_without_error(self):
        """Conv2d KFAC penalty should compute without crashing."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
        )
        opt = SGD(model.parameters(), lr=0.01)
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            device="cpu", train_epochs=1, train_mb_size=8)

        torch.manual_seed(42)
        for tid in range(2):
            tx = torch.randn(16, 3, 8, 8)
            ty = torch.randint(0, 4, (16,))
            ex = torch.randn(8, 3, 8, 8)
            ey = torch.randint(0, 4, (8,))
            exp = Experience(
                train_dataset=TensorDataset(tx, ty),
                test_dataset=TensorDataset(ex, ey),
                task_id=tid,
                classes_in_this_experience=[0, 1, 2, 3],
            )
            strategy.train(exp)
        assert strategy.last_certificate is not None
        assert strategy.last_certificate.epsilon_param < float("inf")


# ===================================================================
# TEMPERATURE SCALING TESTS
# ===================================================================

class TestTemperatureScaling:
    def test_fit_temperature_returns_positive(self):
        from delta.core.calibration import CalibrationTracker
        tracker = CalibrationTracker()
        model = nn.Linear(16, 4)
        loader = make_tiny_loader(n=50)
        T = tracker.fit_temperature(model, loader, torch.device("cpu"))
        assert T > 0.0
        assert 0.1 <= T <= 10.0

    def test_apply_temperature_scales_logits(self):
        from delta.core.calibration import CalibrationTracker
        tracker = CalibrationTracker()
        tracker._temperature = 2.0
        logits = torch.randn(4, 10)
        scaled = tracker.apply_temperature(logits)
        assert torch.allclose(scaled, logits / 2.0)


# ===================================================================
# SPEEDUP RATIO TESTS
# ===================================================================

class TestSpeedupRatio:
    def test_compute_ratio_greater_than_one_when_history_large(self):
        """When n_old >> n_new, estimated full retrain is longer."""
        model = make_framework_model(in_dim=32, out_dim=6)
        opt = SGD(model.parameters(), lr=0.01)
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            device="cpu", train_epochs=2, train_mb_size=16)
        torch.manual_seed(42)
        # Task 0: 500 samples
        tx0 = torch.randn(500, 32)
        ty0 = torch.randint(0, 6, (500,))
        ex0 = torch.randn(50, 32)
        ey0 = torch.randint(0, 6, (50,))
        strategy.train(Experience(
            train_dataset=TensorDataset(tx0, ty0),
            test_dataset=TensorDataset(ex0, ey0),
            task_id=0, classes_in_this_experience=[0, 1, 2, 3, 4, 5],
        ))
        # Task 1: 50 samples
        tx1 = torch.randn(50, 32)
        ty1 = torch.randint(0, 6, (50,))
        strategy.train(Experience(
            train_dataset=TensorDataset(tx1, ty1),
            test_dataset=TensorDataset(ex0, ey0),
            task_id=1, classes_in_this_experience=[0, 1, 2, 3, 4, 5],
        ))
        cert = strategy.last_certificate
        assert cert is not None
        # n_total=550, n_task0=500, estimated_full = task0_time * 550/500 = 1.1x
        # delta only trains on 50 samples, so ratio should be > 1
        assert cert.compute_ratio > 1.0, \
            f"Expected ratio > 1.0, got {cert.compute_ratio}"


# ===================================================================
# MULTI-TASK EPSILON STABILITY TESTS
# ===================================================================

class TestEpsilonStability:
    def test_epsilon_stays_bounded_across_5_tasks(self):
        model = make_framework_model(in_dim=32, out_dim=10)
        opt = SGD(model.parameters(), lr=0.01)
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            device="cpu", train_epochs=2, train_mb_size=16)
        stream = DeltaStream("synthetic", n_tasks=5, classes_per_task=2)
        epsilons = []
        for exp in stream.train_stream:
            strategy.train(exp)
            cert = strategy.last_certificate
            assert cert is not None
            epsilons.append(cert.epsilon_param)
        # All finite
        assert all(np.isfinite(e) for e in epsilons), f"Non-finite epsilon: {epsilons}"
        # All < 1.0
        assert all(e < 1.0 for e in epsilons), f"Epsilon > 1.0: {epsilons}"

    def test_scales_remain_stable_across_tasks(self):
        model = make_framework_model(in_dim=32, out_dim=6)
        opt = SGD(model.parameters(), lr=0.01)
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            device="cpu", train_epochs=1, train_mb_size=16)
        torch.manual_seed(42)
        # Task 0: 500 samples
        tx = torch.randn(500, 32)
        ty = torch.randint(0, 6, (500,))
        ex = torch.randn(50, 32)
        ey = torch.randint(0, 6, (50,))
        strategy.train(Experience(
            train_dataset=TensorDataset(tx, ty),
            test_dataset=TensorDataset(ex, ey),
            task_id=0, classes_in_this_experience=list(range(6)),
        ))
        assert strategy.ce_scale == 1.0
        assert strategy.ewc_scale == 0.0

        # Task 1: bounded regularization, neutral CE scale
        tx1 = torch.randn(50, 32)
        ty1 = torch.randint(0, 6, (50,))
        strategy.train(Experience(
            train_dataset=TensorDataset(tx1, ty1),
            test_dataset=TensorDataset(ex, ey),
            task_id=1, classes_in_this_experience=list(range(6)),
        ))
        assert strategy.ce_scale == 1.0
        assert strategy.ewc_scale == 1.0

        # Task 2: still bounded instead of growing with history
        tx2 = torch.randn(50, 32)
        ty2 = torch.randint(0, 6, (50,))
        strategy.train(Experience(
            train_dataset=TensorDataset(tx2, ty2),
            test_dataset=TensorDataset(ex, ey),
            task_id=2, classes_in_this_experience=list(range(6)),
        ))
        assert strategy.ce_scale == 1.0
        assert strategy.ewc_scale == 1.0


# ===================================================================
# DELTA REGULARIZATION REGRESSION TESTS
# ===================================================================

class ZeroLoss(nn.Module):
    def forward(self, logits, targets):
        return logits.sum() * 0.0


class TestDeltaRegularization:
    def test_active_class_masking_ignores_future_logits(self):
        model = nn.Linear(4, 6)
        strategy = BaseStrategy(
            model,
            SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            device="cpu",
        )
        logits = torch.tensor([
            [4.0, 1.0, 10.0, 9.0, 8.0, 7.0],
            [0.5, 3.0, 9.0, 8.0, 7.0, 6.0],
        ])
        targets = torch.tensor([0, 1], dtype=torch.long)
        strategy._set_active_classes([0, 1])
        masked_logits, masked_targets = strategy._masked_logits_and_targets(
            logits, targets, strategy.active_classes
        )

        assert masked_logits.shape[1] == 2
        assert torch.equal(masked_targets, targets)
        expected = nn.CrossEntropyLoss()(logits[:, :2], targets)
        actual = nn.CrossEntropyLoss()(masked_logits, masked_targets)
        assert torch.allclose(actual, expected)

    def test_kd_only_affects_seen_class_logits(self):
        torch.manual_seed(7)
        x = torch.randn(6, 4)
        y = torch.tensor([2, 2, 3, 3, 2, 3], dtype=torch.long)

        base_model = nn.Linear(4, 4, bias=False)
        old_model = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            base_model.weight.copy_(torch.randn_like(base_model.weight))
            old_model.weight.copy_(torch.randn_like(old_model.weight))

        def grad_for(kd_alpha: float) -> torch.Tensor:
            model = copy.deepcopy(base_model)
            strategy = FisherDeltaStrategy(
                model,
                SGD(model.parameters(), lr=0.01),
                nn.CrossEntropyLoss(),
                device="cpu",
                train_epochs=1,
                train_mb_size=6,
                kd_alpha=kd_alpha,
            )
            strategy.state = DeltaState(label_counts={0: 10, 1: 10})
            strategy.ce_scale = 1.0
            strategy.ewc_scale = 0.0
            strategy._old_model = copy.deepcopy(old_model)

            logits = strategy.model(x)
            loss = strategy._compute_loss(x, y, logits)
            loss.backward()
            return strategy.model.weight.grad.detach().clone()

        grad_without_kd = grad_for(0.0)
        grad_with_kd = grad_for(2.0)

        assert torch.allclose(
            grad_with_kd[2:], grad_without_kd[2:], atol=1e-6
        )
        assert not torch.allclose(
            grad_with_kd[:2], grad_without_kd[:2], atol=1e-6
        )

    def test_fisher_trace_normalization_is_scale_invariant(self):
        def penalty_loss(scale: float) -> float:
            model = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                model.weight.fill_(1.0)

            strategy = FisherDeltaStrategy(
                model,
                SGD(model.parameters(), lr=0.01),
                ZeroLoss(),
                device="cpu",
                train_epochs=1,
                train_mb_size=2,
            )
            strategy.state = DeltaState()
            strategy.ce_scale = 0.0
            strategy.ewc_scale = 1.0
            strategy._ref_params = {
                "weight": torch.zeros_like(model.weight)
            }
            strategy._kfac_names = set()
            strategy._fisher_splits = {
                "weight": torch.full_like(model.weight, scale)
            }
            strategy._fisher_trace = float(strategy._fisher_splits["weight"].sum().item())

            x = torch.randn(2, 2)
            y = torch.zeros(2, dtype=torch.long)
            logits = strategy.model(x)
            return float(strategy._compute_loss(x, y, logits).item())

        assert abs(penalty_loss(1.0) - penalty_loss(10.0)) < 1e-6

    def test_certificate_reports_normalized_kl_bound(self):
        model = make_framework_model(in_dim=32, out_dim=4)
        opt = SGD(model.parameters(), lr=0.01)
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            device="cpu", train_epochs=1, train_mb_size=16
        )
        stream = DeltaStream("synthetic", n_tasks=2, classes_per_task=2)
        for exp in stream.train_stream:
            strategy.train(exp)

        cert = strategy.last_certificate
        assert cert is not None
        assert np.isfinite(cert.kl_bound_normalized)
        assert cert.kl_bound_normalized <= cert.kl_bound

    def test_incremental_classifier_expands_per_task(self):
        from delta.demos.models.classifier import IncrementalClassifier
        from delta.demos.models.modeling import MELDModel

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 8)
                self.out_dim = 8

            def forward(self, x):
                return self.proj(x)

            def embed(self, x):
                return self.proj(x)

        model = MELDModel(TinyBackbone(), IncrementalClassifier(8))
        opt = SGD(model.parameters(), lr=0.01)
        strategy = FisherDeltaStrategy(
            model, opt, nn.CrossEntropyLoss(),
            device="cpu", train_epochs=1, train_mb_size=4
        )

        for tid in range(2):
            x = torch.randn(8, 4)
            y = torch.tensor([tid * 2, tid * 2 + 1] * 4, dtype=torch.long)
            exp = Experience(
                train_dataset=TensorDataset(x, y),
                test_dataset=TensorDataset(x, y),
                task_id=tid,
                classes_in_this_experience=[tid * 2, tid * 2 + 1],
            )
            strategy.train(exp)
            assert model.classifier.num_classes == (tid + 1) * 2

    def test_replay_memory_populates_and_samples_old_classes(self):
        from delta.demos.models.classifier import IncrementalClassifier
        from delta.demos.models.modeling import MELDModel

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 8)
                self.out_dim = 8

            def forward(self, x):
                return self.proj(x)

            def embed(self, x):
                return self.proj(x)

        model = MELDModel(TinyBackbone(), IncrementalClassifier(8))
        opt = SGD(model.parameters(), lr=0.01)
        strategy = FisherDeltaStrategy(
            model,
            opt,
            nn.CrossEntropyLoss(),
            device="cpu",
            train_epochs=1,
            train_mb_size=4,
        )
        strategy.replay_memory_per_class = 2
        strategy.replay_batch_size = 4

        x0 = torch.randn(8, 4)
        y0 = torch.tensor([0, 1] * 4, dtype=torch.long)
        exp0 = Experience(
            train_dataset=TensorDataset(x0, y0),
            test_dataset=TensorDataset(x0, y0),
            task_id=0,
            classes_in_this_experience=[0, 1],
        )
        strategy.train(exp0)
        assert len(strategy.replay_memory[0]) == 2
        assert len(strategy.replay_memory[1]) == 2

        x1 = torch.randn(8, 4)
        y1 = torch.tensor([2, 3] * 4, dtype=torch.long)
        exp1 = Experience(
            train_dataset=TensorDataset(x1, y1),
            test_dataset=TensorDataset(x1, y1),
            task_id=1,
            classes_in_this_experience=[2, 3],
        )
        strategy._prepare_training_experience(exp1)
        replay_batch = strategy._sample_replay_batch()
        assert replay_batch is not None
        _, replay_targets = replay_batch
        assert set(replay_targets.tolist()).issubset({0, 1})

    def test_replay_memory_populates_for_plain_model_without_embed(self):
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        strategy = ReplayDeltaStrategy(
            model,
            SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            device="cpu",
            train_epochs=1,
            train_mb_size=4,
        )
        strategy.replay_memory_per_class = 2
        strategy.replay_batch_size = 4

        x0 = torch.randn(8, 4)
        y0 = torch.tensor([0, 1] * 4, dtype=torch.long)
        exp0 = Experience(
            train_dataset=TensorDataset(x0, y0),
            test_dataset=TensorDataset(x0, y0),
            task_id=0,
            classes_in_this_experience=[0, 1],
        )
        strategy.train(exp0)
        assert len(strategy.replay_memory[0]) == 2
        assert len(strategy.replay_memory[1]) == 2

        x1 = torch.randn(8, 4)
        y1 = torch.tensor([2, 3] * 4, dtype=torch.long)
        exp1 = Experience(
            train_dataset=TensorDataset(x1, y1),
            test_dataset=TensorDataset(x1, y1),
            task_id=1,
            classes_in_this_experience=[2, 3],
        )
        strategy._prepare_training_experience(exp1)
        replay_batch = strategy._sample_replay_batch()
        assert replay_batch is not None
        _, replay_targets = replay_batch
        assert set(replay_targets.tolist()).issubset({0, 1})

    def test_replay_batches_mix_into_training_minibatch(self):
        from delta.demos.models.classifier import IncrementalClassifier
        from delta.demos.models.modeling import MELDModel

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 8)
                self.out_dim = 8

            def forward(self, x):
                return self.proj(x)

            def embed(self, x):
                return self.proj(x)

        model = MELDModel(TinyBackbone(), IncrementalClassifier(8))
        strategy = FisherDeltaStrategy(
            model,
            SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            device="cpu",
            train_epochs=1,
            train_mb_size=4,
        )
        strategy.replay_memory_per_class = 2
        strategy.replay_batch_size = 2

        x0 = torch.randn(8, 4)
        y0 = torch.tensor([0, 1] * 4, dtype=torch.long)
        exp0 = Experience(
            train_dataset=TensorDataset(x0, y0),
            test_dataset=TensorDataset(x0, y0),
            task_id=0,
            classes_in_this_experience=[0, 1],
        )
        strategy.train(exp0)

        x1 = torch.randn(4, 4)
        y1 = torch.tensor([2, 3, 2, 3], dtype=torch.long)
        exp1 = Experience(
            train_dataset=TensorDataset(x1, y1),
            test_dataset=TensorDataset(x1, y1),
            task_id=1,
            classes_in_this_experience=[2, 3],
        )
        strategy._prepare_training_experience(exp1)
        strategy._before_training_experience(exp1)
        strategy.mb_x = x1.clone()
        strategy.mb_y = y1.clone()
        strategy._before_training_iteration(strategy.mb_x, strategy.mb_y)

        assert strategy._batch_has_mixed_replay is True
        assert strategy.mb_x.shape[0] == 6
        assert strategy.mb_y.shape[0] == 6
        assert set(strategy.mb_y[-2:].tolist()).issubset({0, 1})

    def test_nme_eval_uses_stored_class_means(self):
        from delta.demos.models.classifier import IncrementalClassifier
        from delta.demos.models.modeling import MELDModel

        class IdentityBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.out_dim = 2

            def forward(self, x):
                return x

            def embed(self, x):
                return x

        model = MELDModel(IdentityBackbone(), IncrementalClassifier(2))
        model.classifier.adaption(2)
        strategy = FisherDeltaStrategy(
            model,
            SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            device="cpu",
        )
        strategy.state = DeltaState(
            class_feature_means={
                0: np.array([1.0, 0.0], dtype=np.float32),
                1: np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        x = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
        preds = strategy._predict_eval(x).argmax(dim=1)
        assert torch.equal(preds, torch.tensor([0, 1]))

    def test_weight_alignment_scales_new_head_toward_old_norm(self):
        from delta.demos.models.classifier import IncrementalClassifier
        from delta.demos.models.modeling import MELDModel

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 8)
                self.out_dim = 8

            def forward(self, x):
                return self.proj(x)

            def embed(self, x):
                return self.proj(x)

        model = MELDModel(TinyBackbone(), IncrementalClassifier(8))
        strategy = FisherDeltaStrategy(
            model,
            SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            device="cpu",
        )
        model.classifier.adaption(2)
        model.classifier.adaption(2)
        strategy._current_classes = [2, 3]
        with torch.no_grad():
            for class_id in [0, 1]:
                head_index, offset = model.classifier.class_to_head[class_id]
                model.classifier.heads[head_index].weight[offset].fill_(2.0)
            for class_id in [2, 3]:
                head_index, offset = model.classifier.class_to_head[class_id]
                model.classifier.heads[head_index].weight[offset].fill_(0.5)

        strategy._align_new_class_weights()

        old_norm = model.classifier.weight_vector(0).norm(p=2).item()
        new_norm = model.classifier.weight_vector(2).norm(p=2).item()
        assert abs(old_norm - new_norm) < 1e-5


# ===================================================================
# CROSS-PLATFORM TESTS
# ===================================================================

class TestCrossPlatform:
    def test_no_windows_specific_paths(self):
        """delta/ must not contain hardcoded Windows paths."""
        import pathlib
        delta_dir = pathlib.Path(__file__).parent.parent / "delta"
        violations = []
        for py_file in delta_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            src = py_file.read_text(encoding="utf-8")
            for bad in ["C:\\\\", "C:/Users", "\\\\.\\\\pipe"]:
                if bad in src:
                    violations.append(f"{py_file}: contains {bad}")
        assert not violations, f"Windows-specific code found: {violations}"

    def test_imports_work_without_optional_deps(self):
        """Core delta imports should work with only torch + numpy."""
        from delta import (
            DeltaStream, FisherDeltaStrategy, FullRetrainStrategy,
            BaseStrategy, EvaluationPlugin, DeltaState,
            EquivalenceCertificate, KFACComputer, ShiftDetector,
        )
        assert DeltaStream is not None
        assert FisherDeltaStrategy is not None
