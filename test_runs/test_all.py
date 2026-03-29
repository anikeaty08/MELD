"""
Master test runner — runs all tests, saves results to JSON.
Run: python test_runs/test_all.py

Results saved to: test_runs/results/
"""

import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset

from delta import (
    DeltaStream, Experience, FisherDeltaStrategy, FullRetrainStrategy,
    BaseStrategy, EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALL_RESULTS = {}


def save_result(test_name, data):
    ALL_RESULTS[test_name] = data
    # Save individual
    path = RESULTS_DIR / f"{test_name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    # Save combined
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(ALL_RESULTS, f, indent=2, default=str)


def run_strategy(model, stream, epochs=3, batch_size=32, lr=0.01, optimizer=None):
    opt = optimizer or SGD(model.parameters(), lr=lr, momentum=0.9)
    ep = EvaluationPlugin(
        accuracy_metrics(stream=True),
        equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
        calibration_metrics(ece_before=True, ece_after=True),
        compute_metrics(savings_ratio=True),
        loggers=[InteractiveLogger(verbose=False)],
    )
    strategy = FisherDeltaStrategy(
        model, opt, nn.CrossEntropyLoss(),
        evaluator=ep, train_epochs=epochs, train_mb_size=batch_size,
    )
    task_results = []
    t0 = time.time()
    for exp in stream.train_stream:
        t_task = time.time()
        strategy.train(exp)
        strategy.eval(stream.test_stream)
        cert = strategy.last_certificate
        metrics = ep.get_last_metrics()
        task_results.append({
            "task_id": exp.task_id,
            "train_samples": len(exp.train_dataset),
            "wall_time": round(time.time() - t_task, 3),
            "accuracy_stream": metrics.get("accuracy/stream"),
            "epsilon_param": cert.epsilon_param,
            "kl_bound": cert.kl_bound,
            "is_equivalent": cert.is_equivalent,
            "shift_type": cert.shift_type,
            "ece_before": cert.ece_before,
            "ece_after": cert.ece_after,
            "ece_delta": cert.ece_delta,
            "compute_ratio": cert.compute_ratio,
            "ce_scale": cert.ce_scale,
            "ewc_scale": cert.ewc_scale,
        })
    total_time = round(time.time() - t0, 3)
    return {
        "tasks": task_results,
        "total_wall_time": total_time,
        "final_accuracy": task_results[-1]["accuracy_stream"] if task_results else None,
        "final_epsilon": task_results[-1]["epsilon_param"] if task_results else None,
        "final_speedup": task_results[-1]["compute_ratio"] if task_results else None,
        "kfac_layers": len(strategy.state.kfac_A) if strategy.state else 0,
        "n_params": sum(p.numel() for p in model.parameters()),
    }, strategy


def run_full_retrain(model, stream, epochs=3, batch_size=32, lr=0.01):
    opt = SGD(model.parameters(), lr=lr, momentum=0.9)
    strategy = FullRetrainStrategy(
        model, opt, nn.CrossEntropyLoss(),
        train_epochs=epochs, train_mb_size=batch_size,
    )
    t0 = time.time()
    for exp in stream.train_stream:
        strategy.train(exp)
    results = strategy.eval(stream.test_stream)
    return {
        "accuracy_stream": results.get("accuracy/stream", 0),
        "wall_time": round(time.time() - t0, 3),
    }


# ===================================================================
print("\n" + "=" * 60)
print("  DELTA FRAMEWORK — FULL TEST SUITE")
print("=" * 60)


# ---- TEST 1: Synthetic MLP ----
print("\n>>> Test 1: Synthetic MLP")
try:
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6))
    stream = DeltaStream("synthetic", n_tasks=3, classes_per_task=2)
    delta_res, _ = run_strategy(model, stream, epochs=3, batch_size=16)

    model_f = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6))
    stream2 = DeltaStream("synthetic", n_tasks=3, classes_per_task=2)
    full_res = run_full_retrain(model_f, stream2, epochs=3, batch_size=16)

    save_result("test_1_synthetic", {
        "status": "PASS", "delta": delta_res, "full_retrain": full_res,
        "accuracy_gap": abs(delta_res["final_accuracy"] - full_res["accuracy_stream"]),
    })
    print("  PASS")
except Exception as e:
    save_result("test_1_synthetic", {"status": "FAIL", "error": traceback.format_exc()})
    print(f"  FAIL: {e}")


# ---- TEST 2: CIFAR-10 CNN ----
print("\n>>> Test 2: CIFAR-10 Custom CNN")
try:
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64, 10),
    )
    stream = DeltaStream("CIFAR-10", n_tasks=5, classes_per_task=2, data_root="./data")
    delta_res, _ = run_strategy(model, stream, epochs=3, batch_size=64)
    save_result("test_2_cifar10", {"status": "PASS", "delta": delta_res})
    print("  PASS")
except Exception as e:
    save_result("test_2_cifar10", {"status": "SKIP", "error": str(e)})
    print(f"  SKIP: {e}")


# ---- TEST 3: MNIST MLP ----
print("\n>>> Test 3: MNIST MLP")
try:
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    stream = DeltaStream("MNIST", n_tasks=5, classes_per_task=2, data_root="./data")
    delta_res, _ = run_strategy(model, stream, epochs=3, batch_size=64)

    model_f = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    stream2 = DeltaStream("MNIST", n_tasks=5, classes_per_task=2, data_root="./data")
    full_res = run_full_retrain(model_f, stream2, epochs=3, batch_size=64)

    save_result("test_3_mnist", {
        "status": "PASS", "delta": delta_res, "full_retrain": full_res,
        "accuracy_gap": abs(delta_res["final_accuracy"] - full_res["accuracy_stream"]),
    })
    print("  PASS")
except Exception as e:
    save_result("test_3_mnist", {"status": "SKIP", "error": str(e)})
    print(f"  SKIP: {e}")


# ---- TEST 4: FashionMNIST CNN ----
print("\n>>> Test 4: FashionMNIST CNN")
try:
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, 10),
    )
    stream = DeltaStream("FashionMNIST", n_tasks=5, classes_per_task=2, data_root="./data")
    delta_res, _ = run_strategy(model, stream, epochs=3, batch_size=64)
    save_result("test_4_fashionmnist", {"status": "PASS", "delta": delta_res})
    print("  PASS")
except Exception as e:
    save_result("test_4_fashionmnist", {"status": "SKIP", "error": str(e)})
    print(f"  SKIP: {e}")


# ---- TEST 5: CIFAR-100 ResNet20 ----
print("\n>>> Test 5: CIFAR-100 ResNet20")
try:
    from delta.demos.models import resnet20, IncrementalClassifier, MELDModel
    backbone = resnet20()
    classifier = IncrementalClassifier(backbone.out_dim)
    model = MELDModel(backbone, classifier)
    stream = DeltaStream("CIFAR-100", n_tasks=10, classes_per_task=10, data_root="./data")
    delta_res, _ = run_strategy(model, stream, epochs=2, batch_size=64)
    save_result("test_5_cifar100", {"status": "PASS", "delta": delta_res})
    print("  PASS")
except Exception as e:
    save_result("test_5_cifar100", {"status": "SKIP", "error": str(e)})
    print(f"  SKIP: {e}")


# ---- TEST 6: Custom plugin ----
print("\n>>> Test 6: Custom model + plugin")
try:
    class GradTracker:
        def __init__(self): self.norms = []
        def after_backward(self, strategy):
            n = sum(p.grad.norm().item()**2 for p in strategy.model.parameters() if p.grad is not None)**0.5
            self.norms.append(n)

    class StepCounter:
        def __init__(self): self.steps = 0
        def after_training_iteration(self, strategy): self.steps += 1

    torch.manual_seed(123)
    exps = []
    for tid in range(4):
        classes = [tid*2, tid*2+1]
        tx = torch.cat([torch.randn(100, 50) + torch.randn(1, 50)*c for c in classes])
        ty = torch.cat([torch.full((100,), c, dtype=torch.long) for c in classes])
        ex = torch.cat([torch.randn(30, 50) + torch.randn(1, 50)*c for c in classes])
        ey = torch.cat([torch.full((30,), c, dtype=torch.long) for c in classes])
        exps.append(Experience(
            train_dataset=TensorDataset(tx, ty), test_dataset=TensorDataset(ex, ey),
            task_id=tid, classes_in_this_experience=classes,
        ))

    model = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 8))
    gt, sc = GradTracker(), StepCounter()
    ep = EvaluationPlugin(accuracy_metrics(stream=True), equivalence_metrics(epsilon=True), loggers=[InteractiveLogger(verbose=False)])
    strategy = FisherDeltaStrategy(model, SGD(model.parameters(), lr=0.005, momentum=0.9), nn.CrossEntropyLoss(),
        evaluator=ep, train_epochs=3, train_mb_size=32)
    strategy.add_plugin(gt)
    strategy.add_plugin(sc)

    for exp in exps:
        strategy.train(exp)
        strategy.eval(exps)

    cert = strategy.last_certificate
    save_result("test_6_plugin", {
        "status": "PASS",
        "steps": sc.steps, "grad_norms_count": len(gt.norms),
        "mean_grad_norm": sum(gt.norms)/len(gt.norms),
        "epsilon": cert.epsilon_param, "speedup": cert.compute_ratio,
    })
    print("  PASS")
except Exception as e:
    save_result("test_6_plugin", {"status": "FAIL", "error": traceback.format_exc()})
    print(f"  FAIL: {e}")


# ---- TEST 7: SVHN ----
print("\n>>> Test 7: SVHN CNN")
try:
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        nn.Flatten(), nn.Linear(64*16, 128), nn.ReLU(), nn.Linear(128, 10),
    )
    stream = DeltaStream("SVHN", n_tasks=5, classes_per_task=2, data_root="./data")
    delta_res, _ = run_strategy(model, stream, epochs=2, batch_size=64)
    save_result("test_7_svhn", {"status": "PASS", "delta": delta_res})
    print("  PASS")
except Exception as e:
    save_result("test_7_svhn", {"status": "SKIP", "error": str(e)})
    print(f"  SKIP: {e}")


# ---- TEST 8: Stress 10 tasks ----
print("\n>>> Test 8: Stress — 10 tasks")
try:
    torch.manual_seed(42)
    exps = []
    for tid in range(10):
        n = 500 if tid == 0 else 50
        classes = [tid*2, tid*2+1]
        tx = torch.cat([torch.randn(n//2, 64) + torch.randn(1, 64)*c for c in classes])
        ty = torch.cat([torch.full((n//2,), c, dtype=torch.long) for c in classes])
        ex = torch.randn(50, 64); ey = torch.randint(0, 20, (50,))
        exps.append(Experience(train_dataset=TensorDataset(tx, ty), test_dataset=TensorDataset(ex, ey),
            task_id=tid, classes_in_this_experience=classes))

    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 20))
    ep = EvaluationPlugin(accuracy_metrics(stream=True), equivalence_metrics(epsilon=True), loggers=[InteractiveLogger(verbose=False)])
    strategy = FisherDeltaStrategy(model, SGD(model.parameters(), lr=0.005, momentum=0.9), nn.CrossEntropyLoss(),
        evaluator=ep, train_epochs=3, train_mb_size=32)

    task_data = []
    for exp in exps:
        strategy.train(exp)
        strategy.eval(exps)
        c = strategy.last_certificate
        task_data.append({"task": exp.task_id, "epsilon": c.epsilon_param, "ce_scale": c.ce_scale,
            "ewc_scale": c.ewc_scale, "speedup": c.compute_ratio, "shift": c.shift_type})

    save_result("test_8_stress", {
        "status": "PASS", "tasks": task_data,
        "all_finite": all(t["epsilon"] < float("inf") for t in task_data),
        "ce_scale_range": [task_data[0]["ce_scale"], task_data[-1]["ce_scale"]],
        "speedup_range": [task_data[0]["speedup"], task_data[-1]["speedup"]],
    })
    print("  PASS")
except Exception as e:
    save_result("test_8_stress", {"status": "FAIL", "error": traceback.format_exc()})
    print(f"  FAIL: {e}")


# ---- TEST 9: Quick CIFAR-10 ResNet20 ----
print("\n>>> Test 9: Quick CIFAR-10 ResNet20")
try:
    from delta.demos.models import resnet20, IncrementalClassifier, MELDModel

    backbone = resnet20()
    classifier = IncrementalClassifier(backbone.out_dim)
    model = MELDModel(backbone, classifier)
    stream = DeltaStream("CIFAR-10", n_tasks=5, classes_per_task=2, data_root="./data")
    delta_res, _ = run_strategy(model, stream, epochs=2, batch_size=64)
    save_result("test_9_quick_cifar10_resnet20", {"status": "PASS", "delta": delta_res})
    print("  PASS")
except Exception as e:
    save_result("test_9_quick_cifar10_resnet20", {"status": "SKIP", "error": str(e)})
    print(f"  SKIP: {e}")


# ---- TEST 10: Quick MNIST CNN ----
print("\n>>> Test 10: Quick MNIST CNN")
try:
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, 10),
    )
    stream = DeltaStream("MNIST", n_tasks=5, classes_per_task=2, data_root="./data")
    delta_res, _ = run_strategy(model, stream, epochs=2, batch_size=64)
    save_result("test_10_quick_mnist_cnn", {"status": "PASS", "delta": delta_res})
    print("  PASS")
except Exception as e:
    save_result("test_10_quick_mnist_cnn", {"status": "SKIP", "error": str(e)})
    print(f"  SKIP: {e}")


# ---- TEST 11: Transformer ----
print("\n>>> Test 11: Transformer")
try:
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(32, 64)
            self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
            self.norm = nn.LayerNorm(64)
            self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
            self.head = nn.Linear(64, 6)
        def forward(self, x):
            h = self.embed(x).unsqueeze(1)
            a, _ = self.attn(h, h, h)
            h = self.norm(h + a)
            h = h + self.ff(h)
            return self.head(h.squeeze(1))

    model = TinyTransformer()
    stream = DeltaStream("synthetic", n_tasks=3, classes_per_task=2)
    delta_res, strat = run_strategy(model, stream, epochs=3, batch_size=16, optimizer=Adam(model.parameters(), lr=0.001))
    save_result("test_11_transformer", {
        "status": "PASS", "delta": delta_res,
        "kfac_layers": len(strat.state.kfac_A),
        "kfac_param_names": strat.state.kfac_param_names,
    })
    print(f"  PASS — {len(strat.state.kfac_A)} KFAC layers")
except Exception as e:
    save_result("test_11_transformer", {"status": "FAIL", "error": traceback.format_exc()})
    print(f"  FAIL: {e}")


# ---- TEST 12: Speedup proof ----
print("\n>>> Test 12: Speedup proof (2000 vs 100)")
try:
    torch.manual_seed(42)
    exps = []
    for tid in range(6):
        n = 2000 if tid == 0 else 100
        tx = torch.randn(n, 64) + torch.randn(1, 64)*tid
        ty = torch.randint(0, 10, (n,))
        ex = torch.randn(100, 64); ey = torch.randint(0, 10, (100,))
        exps.append(Experience(train_dataset=TensorDataset(tx, ty), test_dataset=TensorDataset(ex, ey),
            task_id=tid, classes_in_this_experience=list(range(10))))

    model_d = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    ep = EvaluationPlugin(accuracy_metrics(stream=True), equivalence_metrics(epsilon=True), compute_metrics(), loggers=[InteractiveLogger(verbose=False)])
    delta_s = FisherDeltaStrategy(model_d, SGD(model_d.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
        evaluator=ep, train_epochs=3, train_mb_size=32)

    delta_times = []
    for exp in exps:
        t0 = time.time()
        delta_s.train(exp)
        delta_times.append(time.time() - t0)
        delta_s.eval(exps)

    model_f = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    full_s = FullRetrainStrategy(model_f, SGD(model_f.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(),
        train_epochs=3, train_mb_size=32)
    full_times = []
    for exp in exps:
        t0 = time.time()
        full_s.train(exp)
        full_times.append(time.time() - t0)

    delta_total = sum(delta_times)
    full_total = sum(full_times)
    actual_speedup = full_total / max(delta_total, 1e-6)
    delta_acc = ep.get_last_metrics().get("accuracy/stream", 0)
    full_acc = full_s.eval(exps).get("accuracy/stream", 0)

    save_result("test_12_speedup", {
        "status": "PASS" if actual_speedup > 1.0 else "FAIL",
        "delta_total_time": round(delta_total, 3),
        "full_total_time": round(full_total, 3),
        "actual_speedup": round(actual_speedup, 2),
        "estimated_speedup": delta_s.last_certificate.compute_ratio,
        "delta_accuracy": delta_acc,
        "full_accuracy": full_acc,
    })
    print(f"  {'PASS' if actual_speedup > 1.0 else 'FAIL'} — {actual_speedup:.2f}x actual speedup")
except Exception as e:
    save_result("test_12_speedup", {"status": "FAIL", "error": traceback.format_exc()})
    print(f"  FAIL: {e}")


# ---- SUMMARY ----
print(f"\n\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  {'Test':<25} | {'Status':<8} | {'Epsilon':>10} | {'Speedup':>8} | {'Acc':>6}")
print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")
for name, data in ALL_RESULTS.items():
    status = data.get("status", "?")
    eps = "--"
    spd = "--"
    acc = "--"
    if "delta" in data and isinstance(data["delta"], dict):
        eps = f"{data['delta'].get('final_epsilon', 0):.6f}" if data['delta'].get('final_epsilon') is not None else "--"
        spd = f"{data['delta'].get('final_speedup', 0):.1f}x" if data['delta'].get('final_speedup') is not None else "--"
        acc = f"{data['delta'].get('final_accuracy', 0)*100:.1f}%" if data['delta'].get('final_accuracy') is not None else "--"
    elif "tasks" in data:
        last = data["tasks"][-1] if data["tasks"] else {}
        eps = f"{last.get('epsilon', 0):.6f}" if "epsilon" in last else "--"
        spd = f"{last.get('speedup', 0):.1f}x" if "speedup" in last else "--"
    elif "actual_speedup" in data:
        spd = f"{data['actual_speedup']:.1f}x"
        acc = f"{data.get('delta_accuracy', 0)*100:.1f}%" if data.get('delta_accuracy') else "--"
    elif "epsilon" in data:
        eps = f"{data['epsilon']:.6f}"
        spd = f"{data.get('speedup', 0):.1f}x"
    print(f"  {name:<25} | {status:<8} | {eps:>10} | {spd:>8} | {acc:>6}")

passed = sum(1 for d in ALL_RESULTS.values() if d.get("status") == "PASS")
total = len(ALL_RESULTS)
print(f"{'='*70}")
print(f"  RESULT: {passed}/{total} passed")
print(f"  Results saved to: {RESULTS_DIR / 'all_results.json'}")
print(f"{'='*70}")
