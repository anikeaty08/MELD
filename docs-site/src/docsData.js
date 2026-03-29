export const siteMeta = {
  title: "delta-framework",
  packageName: "delta-framework",
  moduleName: "delta",
  version: "0.3.0",
  githubUrl: "https://github.com/anikeaty08/MELD",
  pypiUrl: "https://pypi.org/project/delta-framework/",
  license: "MIT",
};

const quickstartCode = `import torch.nn as nn
from torch.optim import SGD
from delta import DeltaStream, FisherDeltaStrategy

stream = DeltaStream("synthetic", n_tasks=3, classes_per_task=2)
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 6))
strategy = FisherDeltaStrategy(model, SGD(model.parameters(), lr=0.01), nn.CrossEntropyLoss())

for experience in stream.train_stream:
    strategy.train(experience)
    strategy.eval(stream.test_stream)

print(strategy.last_certificate.summary())`;

const certificateSummary = `=== Equivalence Certificate ===
  Epsilon (param bound):  0.004806
  KL bound:               94.896980
  KL bound (normalized):  0.182723
  Is equivalent:          False
  Shift type:             covariate
  ECE before:             0.8366
  ECE after:              0.6198
  ECE delta:              -0.2168
  Compute savings:        1.2x
  n_old / n_new:          40000 / 10000
  ce_scale (derived):     1.0000
  ewc_scale (derived):    0.2500
  Proof tier:             convex_approx
===============================`;

const metricTableOutput = `+-------------------------------------+
| Metric                 |        Value |
+-------------------------------------+
| accuracy/stream        |       0.5380 |
| accuracy/task_0        |       0.8330 |
| accuracy/task_1        |       0.3030 |
| accuracy/task_2        |       0.6030 |
| accuracy/task_3        |       0.6460 |
| accuracy/task_4        |       0.3050 |
| compute_ratio          |         1.2x |
| ece_after              |       0.6915 |
| ece_before             |       0.7466 |
| ece_delta              |      -0.0551 |
| epsilon_param          |       0.0039 |
| is_equivalent          |        False |
| kl_bound               |     186.2365 |
| kl_bound_normalized    |       0.3586 |
+-------------------------------------+`;

const pluginGradient = `class GradientNormTracker:
    """Tracks gradient norm after each backward pass."""
    def __init__(self):
        self.norms = []
    
    def after_backward(self, strategy):
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in strategy.model.parameters()
            if p.grad is not None
        ) ** 0.5
        self.norms.append(total_norm)
    
    def summary(self):
        if not self.norms:
            return "No gradients tracked."
        return f"Mean grad norm: {sum(self.norms)/len(self.norms):.4f}, Max: {max(self.norms):.4f}"

tracker = GradientNormTracker()
strategy.add_plugin(tracker)

for exp in stream.train_stream:
    strategy.train(exp)

print(tracker.summary())`;

const pluginEarlyStopping = `class EarlyStoppingPlugin:
    """Stops training if loss plateaus."""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self._recent_losses = []
    
    def after_training_epoch(self, strategy):
        if strategy.loss is None:
            return
        current_loss = strategy.loss.item()
        self._recent_losses.append(current_loss)
        if len(self._recent_losses) > self.patience:
            self._recent_losses.pop(0)
            improvement = self._recent_losses[0] - self._recent_losses[-1]
            if improvement < self.min_delta:
                strategy.train_epochs = strategy._epoch_index`;

const pluginCheckpoint = `import torch
from pathlib import Path

class CheckpointPlugin:
    """Saves model after each task."""
    def __init__(self, save_dir="./checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def after_training_experience(self, strategy, experience):
        path = self.save_dir / f"task_{experience.task_id}.pt"
        torch.save({
            "model_state": strategy.model.state_dict(),
            "task_id": experience.task_id,
            "certificate": strategy.last_certificate,
        }, path)
        print(f"Saved checkpoint: {path}")`;

const cifarExample = `import torch
import torch.nn as nn
from torch.optim import SGD
from delta import (
    DeltaStream, FisherDeltaStrategy, FullRetrainStrategy,
    EvaluationPlugin, accuracy_metrics, equivalence_metrics,
    calibration_metrics, compute_metrics, InteractiveLogger, CSVLogger,
)
from delta.demos.models import resnet20, IncrementalClassifier, MELDModel

backbone = resnet20()
classifier = IncrementalClassifier(backbone.out_dim)
model = MELDModel(backbone, classifier)

optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    equivalence_metrics(epsilon=True, is_equivalent=True),
    calibration_metrics(ece_before=True, ece_after=True),
    compute_metrics(savings_ratio=True),
    loggers=[InteractiveLogger(), CSVLogger("cifar10_results.csv")],
)

strategy = FisherDeltaStrategy(
    model, optimizer, nn.CrossEntropyLoss(),
    evaluator=eval_plugin,
    train_epochs=20,
    train_mb_size=64,
)

stream = DeltaStream("CIFAR-10", n_tasks=5, classes_per_task=2, data_root="./data")

for experience in stream.train_stream:
    print(f"\\n=== Task {experience.task_id} — classes {experience.classes_in_this_experience} ===")
    strategy.train(experience)
    strategy.eval(stream.test_stream)
    cert = strategy.last_certificate
    print(cert.summary())`;

const agnewsExample = `from delta import DeltaStream, FisherDeltaStrategy
import torch.nn as nn
from torch.optim import AdamW

stream = DeltaStream(
    "AGNews",
    n_tasks=4,
    classes_per_task=1,
    data_root="./data",
)

model = nn.Sequential(
    nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(256, 4),
)

strategy = FisherDeltaStrategy(
    model,
    AdamW(model.parameters(), lr=3e-4, weight_decay=0.01),
    nn.CrossEntropyLoss(),
    train_epochs=10,
    train_mb_size=128,
)

for experience in stream.train_stream:
    strategy.train(experience)
    strategy.eval(stream.test_stream)
    print(f"Task {experience.task_id}: {strategy.last_certificate.summary()}")`;

const pluginComboExample = `stream = DeltaStream("synthetic", n_tasks=3, classes_per_task=2)
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 6))
strategy = FisherDeltaStrategy(
    model,
    SGD(model.parameters(), lr=0.01),
    nn.CrossEntropyLoss(),
)

tracker = GradientNormTracker()
checkpoints = CheckpointPlugin("./synthetic-checkpoints")
strategy.add_plugin(tracker)
strategy.add_plugin(checkpoints)

for exp in stream.train_stream:
    strategy.train(exp)
    strategy.eval(stream.test_stream)

print(tracker.summary())`;

const registerDatasetExample = `from delta import register_dataset
from delta.benchmarks.dataset_base import TaskBundle
from torch.utils.data import TensorDataset
import torch

def my_provider(config) -> TaskBundle:
    tasks = []
    for task_id in range(config.num_tasks):
        train_x = torch.randn(1000, 128)
        train_y = torch.randint(0, 10, (1000,))
        test_x = torch.randn(200, 128)
        test_y = torch.randint(0, 10, (200,))
        tasks.append((
            TensorDataset(train_x, train_y),
            TensorDataset(test_x, test_y),
        ))
    return tasks

register_dataset("MyDataset", my_provider, aliases=["mydata"])

stream = DeltaStream("MyDataset", n_tasks=5)`;

export const pages = [
  {
    id: "home",
    path: "/",
    title: "Continual Learning with Formal Guarantees",
    description:
      "delta is a PyTorch framework for class-incremental learning. Train on new data only. Get a certificate proving your model is equivalent to full retraining.",
    hero: {
      headline: "Continual Learning with Formal Guarantees",
      subheadline:
        "delta is a PyTorch framework for class-incremental learning. Train on new data only. Get a certificate proving your model is equivalent to full retraining.",
      installCommand: "pip install delta-framework",
      cards: [
        {
          title: "K-FAC Fisher",
          body:
            "Kronecker-factored Fisher information. Tighter bounds than diagonal EWC and better layer-wise structure for modern PyTorch models.",
        },
        {
          title: "Equivalence Certificate",
          body:
            "Formal per-task diagnostics for parameter drift, KL deviation, calibration change, and compute savings against full retraining.",
        },
        {
          title: "12+ Datasets",
          body:
            "CIFAR-10/100, MNIST, FashionMNIST, STL-10, SVHN, AG News, DBpedia, IMDB, SST-2, TinyImageNet, and custom providers.",
        },
      ],
      quickstart: quickstartCode,
    },
    sections: [
      {
        id: "why-delta",
        level: 2,
        title: "Why delta?",
        blocks: [
          {
            type: "paragraph",
            text:
              "delta targets the painful middle ground between `naive fine-tuning` and `full retraining`. Fine-tuning is cheap but forgets. Full retraining is accurate but expensive. delta keeps the update local, computes structured Fisher state, tracks replay and calibration, and then tells you how close the resulting model is to the expensive baseline.",
          },
          {
            type: "table",
            columns: ["Approach", "Old data needed", "Compute cost", "Forgetting risk", "Formal diagnostics"],
            rows: [
              ["Naive fine-tuning", "No", "Low", "High", "None"],
              ["Full retrain", "Yes", "High", "Low", "Only empirical"],
              ["delta", "New data + compact state", "Medium", "Low to medium", "Certificate + calibration + speedup"],
            ],
          },
        ],
      },
      {
        id: "eight-line-quickstart",
        level: 2,
        title: "8-line quickstart",
        blocks: [
          { type: "code", language: "python", code: quickstartCode },
          {
            type: "paragraph",
            text:
              "This is the smallest end-to-end loop: build a `DeltaStream`, pick a `FisherDeltaStrategy`, iterate over experiences, and inspect `strategy.last_certificate` after every task.",
          },
        ],
      },
      {
        id: "landing-checklist",
        level: 2,
        title: "What you get on first install",
        blocks: [
          {
            type: "list",
            items: [
              "`DeltaStream` to turn a dataset into ordered continual-learning experiences.",
              "`FisherDeltaStrategy` for structured delta updates with Fisher/K-FAC state.",
              "`FullRetrainStrategy` as the expensive baseline for comparison.",
              "`EvaluationPlugin` and metric factories for accuracy, equivalence, calibration, forgetting, and compute savings.",
              "Dataset providers for vision, NLP, and custom benchmark registration.",
            ],
          },
        ],
      },
    ],
  },
  {
    id: "getting-started",
    path: "/getting-started",
    title: "Getting Started",
    description: "Install delta, run the eight-line example, and understand what each object returns.",
    sections: [
      {
        id: "installation",
        level: 2,
        title: "Installation",
        blocks: [
          {
            type: "code",
            language: "bash",
            code: `# Core (CPU only)
pip install delta-framework

# With vision datasets
pip install delta-framework[vision]

# With NLP datasets
pip install delta-framework[text]

# Everything
pip install delta-framework[full]`,
          },
          {
            type: "paragraph",
            text:
              "The `core` install gives you streams, strategies, certificates, and synthetic data. `vision` adds torchvision datasets. `text` adds HuggingFace datasets and sentence-transformer style embedding pipelines. `full` pulls everything used in the repo benchmarks.",
          },
        ],
      },
      {
        id: "quickstart",
        level: 2,
        title: "Quickstart",
        blocks: [{ type: "code", language: "python", code: quickstartCode }],
      },
      {
        id: "line-by-line",
        level: 2,
        title: "Line-by-line explanation",
        blocks: [
          {
            type: "ordered",
            items: [
              "`DeltaStream(\"synthetic\", n_tasks=3, classes_per_task=2)` creates three experiences. Each experience contains one train dataset, one test dataset, a task id, and the classes introduced in that task.",
              "`model = nn.Sequential(...)` is an ordinary PyTorch module. delta strategies work with plain `nn.Module` models and richer `MELDModel` wrappers.",
              "`FisherDeltaStrategy(...)` wraps `model + optimizer + criterion`. The strategy owns the continual-learning lifecycle.",
              "`for experience in stream.train_stream` iterates task by task, not batch by batch.",
              "`strategy.train(experience)` runs the full task update: snapshots old state, computes loss, updates replay/Fisher stats, and stores the new certificate.",
              "`strategy.eval(stream.test_stream)` evaluates the model over all seen tasks and returns a metrics dictionary.",
              "`strategy.last_certificate.summary()` prints the post-task certificate, including equivalence bounds, calibration drift, and compute savings.",
            ],
          },
        ],
      },
    ],
  },
  {
    id: "core-concepts",
    path: "/core-concepts",
    title: "Core Concepts",
    description: "Understand experiences, streams, strategies, certificates, and K-FAC Fisher in plain language.",
    sections: [
      {
        id: "experiences-and-streams",
        level: 2,
        title: "Experiences and Streams",
        blocks: [
          {
            type: "paragraph",
            text:
              "An `Experience` is one task: `(train_dataset, test_dataset, task_id, classes_in_this_experience)`. A `DeltaStream` is an ordered sequence of those experiences. In class-incremental learning, each task adds new classes while the model is still evaluated over all classes seen so far.",
          },
          {
            type: "code",
            language: "text",
            code: `time ─────────────────────────────────────────────────────────────▶

task 0: classes [0, 1]     train ---> evaluate on [0, 1]
task 1: classes [2, 3]     train ---> evaluate on [0, 1, 2, 3]
task 2: classes [4, 5]     train ---> evaluate on [0, 1, 2, 3, 4, 5]
task 3: classes [6, 7]     train ---> evaluate on all seen classes
task 4: classes [8, 9]     train ---> evaluate on all seen classes`,
          },
        ],
      },
      {
        id: "strategies",
        level: 2,
        title: "Strategies",
        blocks: [
          {
            type: "paragraph",
            text:
              "A strategy wraps a model, optimizer, and criterion, then exposes `train(experience)` and `eval(stream)`. Strategies own the hook system: `_before_training_experience`, `_before_training_epoch`, `_after_backward`, `_after_training_experience`, and the matching eval hooks.",
          },
          {
            type: "list",
            items: [
              "`FisherDeltaStrategy` is the research-oriented delta update engine.",
              "`FullRetrainStrategy` is the expensive baseline that accumulates all data and retrains from scratch.",
              "`DeltaStrategy` is the public balanced practical alias in the current repo, designed to keep stronger retention without the heaviest replay settings.",
            ],
          },
        ],
      },
      {
        id: "equivalence-certificate",
        level: 2,
        title: "The Equivalence Certificate",
        blocks: [
          {
            type: "paragraph",
            text:
              "After every task, `strategy.last_certificate` tells you whether the delta update stayed close to the estimated full-retrain solution and whether the update preserved calibration and compute efficiency.",
          },
          { type: "code", language: "text", code: certificateSummary },
          {
            type: "definition",
            items: [
              ["`epsilon_param`", "Parameter drift bound derived from the K-FAC approximation."],
              ["`kl_bound`", "Estimated KL divergence upper bound between the delta model and the full-retrain target."],
              ["`kl_bound_normalized`", "A practical KL-scaled value that is easier to compare across model sizes."],
              ["`is_equivalent`", "True when the configured epsilon, KL, and calibration thresholds are all satisfied."],
              ["`ece_before / ece_after / ece_delta`", "Expected calibration error before and after the update."],
              ["`compute_ratio`", "Estimated full-retrain time divided by the observed delta update time."],
              ["`ce_scale / ewc_scale`", "Derived regularization weights used during the task update."],
              ["`tier`", "Proof regime used for the bound, such as `initial`, `convex`, or `convex_approx`."],
            ],
          },
        ],
      },
      {
        id: "kfac-fisher",
        level: 2,
        title: "K-FAC Fisher",
        blocks: [
          {
            type: "paragraph",
            text:
              "Fisher information tells you how sensitive the model’s predictions are to each parameter. Diagonal EWC treats every weight independently. K-FAC is stronger because it captures layer-wise correlations through the Kronecker product `A ⊗ G`, so the penalty better matches how actual layers move.",
          },
          {
            type: "code",
            language: "text",
            code: `penalty(W) = trace(G × ΔW × A × ΔW^T)

A = activation covariance
G = gradient covariance
ΔW = current weight - reference weight`,
          },
          {
            type: "paragraph",
            text:
              "`kfac_layers` in benchmark output counts how many trainable layers were successfully captured in the K-FAC state. More captured layers usually means a tighter structured approximation of the original model.",
          },
        ],
      },
    ],
  },
  {
    id: "strategies",
    path: "/strategies",
    title: "Strategies",
    description: "Reference guide for FisherDeltaStrategy and FullRetrainStrategy, including lifecycle hooks and internal behavior.",
    sections: [
      {
        id: "fisher-reference",
        level: 2,
        title: "FisherDeltaStrategy",
        blocks: [
          {
            type: "code",
            language: "python",
            code: `FisherDeltaStrategy(
    model,              # nn.Module — your model
    optimizer,          # torch.optim.Optimizer
    criterion,          # nn.Module — loss function (CrossEntropyLoss recommended)
    evaluator=None,     # EvaluationPlugin — optional, for metrics
    device=None,        # auto-detected (CUDA > MPS > CPU)
    train_epochs=10,    # epochs per task
    train_mb_size=64,   # minibatch size
    kd_alpha=0.5,       # knowledge distillation weight
    kd_temperature=2.0, # KD temperature
)`,
          },
          {
            type: "paragraph",
            text:
              "Use `FisherDeltaStrategy` when you want the full delta-training story: shift detection, active-class masking, Fisher regularization, certificate computation, and the benchmark logic used throughout the repo.",
          },
        ],
      },
      {
        id: "hook-lifecycle",
        level: 2,
        title: "What happens internally",
        blocks: [
          {
            type: "definition",
            items: [
              ["`_before_training_experience`", "Computes shift type, snapshots the old model, prepares active classes, loads cached K-FAC tensors, and refreshes task-local state."],
              ["`_compute_loss`", "Builds the task loss from active-class CE, K-FAC EWC penalty, KD on seen classes, feature distillation, and replay when enabled."],
              ["`_after_training_experience`", "Computes the certificate, updates Fisher state, refreshes class feature means, updates replay memory, and rebalances the classifier."],
            ],
          },
          {
            type: "callout",
            tone: "info",
            title: "Practical note",
            body:
              "In the current repo, the public `DeltaStrategy` alias points to a balanced replay-first path. `FisherDeltaStrategy` remains available when you want the research-oriented baseline explicitly.",
          },
        ],
      },
      {
        id: "full-retrain",
        level: 2,
        title: "FullRetrainStrategy",
        blocks: [
          {
            type: "code",
            language: "python",
            code: `FullRetrainStrategy(
    model,
    optimizer,
    criterion,
    train_epochs=10,
    train_mb_size=64,
)`,
          },
          {
            type: "paragraph",
            text:
              "`FullRetrainStrategy` accumulates all datasets seen so far, resets the model to its initial weights, and trains from scratch on the full accumulated data. Use it as the gold-standard baseline for accuracy and for estimating compute ratio.",
          },
        ],
      },
    ],
  },
  {
    id: "evaluation",
    path: "/evaluation-and-metrics",
    title: "Evaluation and Metrics",
    description: "Use EvaluationPlugin and metric factories to log continual-learning metrics automatically.",
    sections: [
      {
        id: "evaluation-plugin",
        level: 2,
        title: "EvaluationPlugin",
        blocks: [
          {
            type: "code",
            language: "python",
            code: `from delta import EvaluationPlugin, accuracy_metrics, equivalence_metrics, \\
    calibration_metrics, compute_metrics, InteractiveLogger, CSVLogger

eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True),
    calibration_metrics(ece_before=True, ece_after=True),
    compute_metrics(savings_ratio=True),
    loggers=[
        InteractiveLogger(),           # prints table to terminal
        CSVLogger("results.csv"),      # saves to CSV
    ],
)`,
          },
          {
            type: "paragraph",
            text:
              "Attach one evaluation plugin to your strategy and delta will keep the metric lifecycle in sync with training and evaluation hooks. Loggers are independent sinks layered on top of the same metric state.",
          },
        ],
      },
      {
        id: "table-output",
        level: 2,
        title: "Reported metrics",
        blocks: [
          {
            type: "definition",
            items: [
              ["`accuracy/stream`", "Mean accuracy over the whole evaluation stream."],
              ["`accuracy/task_k`", "Per-task accuracy for each experience in the stream."],
              ["`epsilon_param`", "Certificate parameter bound for the current model update."],
              ["`kl_bound` / `kl_bound_normalized`", "Raw and normalized KL deviation diagnostics."],
              ["`ece_before` / `ece_after` / `ece_delta`", "Calibration quality before and after the task update."],
              ["`compute_ratio`", "Estimated speedup relative to full retraining."],
            ],
          },
        ],
      },
      {
        id: "metric-factories",
        level: 2,
        title: "Available metric factories",
        blocks: [
          {
            type: "list",
            items: [
              "`accuracy_metrics(experience=True, stream=True)` — per-task and mean accuracy.",
              "`equivalence_metrics(epsilon=True, kl_bound=True, is_equivalent=True)` — certificate metrics.",
              "`calibration_metrics(ece_before=True, ece_after=True)` — ECE tracking.",
              "`forgetting_metrics(per_experience=True)` — backward transfer / forgetting diagnostics.",
              "`compute_metrics(savings_ratio=True)` — speedup ratio against estimated full retrain cost.",
            ],
          },
        ],
      },
    ],
  },
  {
    id: "datasets",
    path: "/datasets",
    title: "Datasets",
    description: "Built-in dataset providers, install extras, and custom registration API.",
    sections: [
      {
        id: "supported-datasets",
        level: 2,
        title: "Supported datasets",
        blocks: [
          {
            type: "table",
            columns: ["Dataset", "Classes", "Input", "Install extra", "Auto-download"],
            rows: [
              ["synthetic", "custom", "32-dim vector", "none", "yes"],
              ["CIFAR-10", "10", "32×32 RGB", "vision", "yes"],
              ["CIFAR-100", "100", "32×32 RGB", "vision", "yes"],
              ["MNIST", "10", "28×28 gray", "vision", "yes"],
              ["FashionMNIST", "10", "28×28 gray", "vision", "yes"],
              ["STL-10", "10", "96×96 RGB", "vision", "yes"],
              ["SVHN", "10", "32×32 RGB", "vision", "yes"],
              ["TinyImageNet", "200", "64×64 RGB", "vision", "manual"],
              ["AG News", "4", "text", "text", "yes"],
              ["DBpedia", "14", "text", "text", "yes"],
              ["IMDB", "2", "text", "text", "yes"],
              ["SST-2", "2", "text", "text", "yes"],
            ],
          },
        ],
      },
      {
        id: "custom-datasets",
        level: 2,
        title: "Custom datasets",
        blocks: [
          { type: "code", language: "python", code: registerDatasetExample },
          {
            type: "paragraph",
            text:
              "A provider returns `TaskBundle = list[(train_dataset, test_dataset)]`. delta does the rest: wraps them into experiences, assigns task ids, and exposes the stream through the same API used by built-in datasets.",
          },
        ],
      },
    ],
  },
  {
    id: "plugins",
    path: "/plugins",
    title: "Plugins",
    description: "Attach hook-based plugins to any strategy for tracking, early stopping, checkpointing, and custom behaviors.",
    sections: [
      {
        id: "what-is-plugin",
        level: 2,
        title: "What is a plugin?",
        blocks: [
          {
            type: "paragraph",
            text:
              "A plugin is any object with methods named after strategy hooks. Attach it with `strategy.add_plugin(plugin)`. Plugins let you observe training, change behavior, or export data without subclassing the strategy.",
          },
          {
            type: "list",
            items: [
              "`before_training_experience(strategy, experience)`",
              "`before_training_epoch(strategy)`",
              "`before_training_iteration(strategy)`",
              "`after_forward(strategy)`",
              "`before_backward(strategy)`",
              "`after_backward(strategy)`",
              "`after_training_iteration(strategy)`",
              "`after_training_epoch(strategy)`",
              "`after_training_experience(strategy, experience)`",
              "`before_eval_experience(strategy, experience)`",
              "`after_eval_experience(strategy, experience)`",
              "`after_eval_stream(strategy, results)`",
            ],
          },
        ],
      },
      {
        id: "gradient-plugin",
        level: 2,
        title: "Example 1 — Gradient norm tracker",
        blocks: [{ type: "code", language: "python", code: pluginGradient }],
      },
      {
        id: "early-stopping-plugin",
        level: 2,
        title: "Example 2 — Early stopping",
        blocks: [{ type: "code", language: "python", code: pluginEarlyStopping }],
      },
      {
        id: "checkpoint-plugin",
        level: 2,
        title: "Example 3 — Checkpoint saver",
        blocks: [{ type: "code", language: "python", code: pluginCheckpoint }],
      },
    ],
  },
  {
    id: "examples",
    path: "/complete-examples",
    title: "Complete Examples",
    description: "Full runnable examples for CIFAR-10, AG News, and custom plugin combinations.",
    sections: [
      {
        id: "example-a",
        level: 2,
        title: "Example A — CIFAR-10 with ResNet20",
        blocks: [
          { type: "code", language: "python", code: cifarExample },
          {
            type: "list",
            items: [
              "`IncrementalClassifier` grows the output space at task-time, so future classes are not preallocated too early.",
              "`EvaluationPlugin` logs stream accuracy, equivalence diagnostics, calibration, and compute metrics at the end of every task.",
              "The loop prints task ids, class ids, and `cert.summary()` so you can inspect both learning quality and certificate quality side by side.",
            ],
          },
        ],
      },
      {
        id: "example-b",
        level: 2,
        title: "Example B — Text classification with AG News",
        blocks: [
          { type: "code", language: "python", code: agnewsExample },
          {
            type: "paragraph",
            text:
              "The AG News provider returns pre-computed L2-normalized text embeddings. You can treat them as fixed-size feature vectors and train a lightweight classifier on top, which keeps the CL loop identical to the vision case.",
          },
        ],
      },
      {
        id: "example-c",
        level: 2,
        title: "Example C — Custom plugin combination",
        blocks: [
          { type: "code", language: "python", code: pluginComboExample },
          {
            type: "paragraph",
            text:
              "This is the main pattern for extensions: compose several hook-aware plugins, attach them once, and keep your training script short.",
          },
        ],
      },
    ],
  },
  {
    id: "api-reference",
    path: "/api-reference",
    title: "API Reference",
    description: "Reference signatures and field-level descriptions for the most important public objects.",
    sections: [
      {
        id: "delta-stream-api",
        level: 2,
        title: "DeltaStream",
        blocks: [
          {
            type: "code",
            language: "python",
            code: `DeltaStream(dataset_name, n_tasks, scenario, classes_per_task, data_root, seed, batch_size)
  .train_stream -> list[Experience]
  .test_stream  -> list[Experience]`,
          },
          {
            type: "list",
            items: [
              "`dataset_name`: built-in provider name or registered alias.",
              "`scenario`: one of `class_incremental`, `task_incremental`, or `domain_incremental`.",
              "`classes_per_task`: number of labels added per task when the provider is class-split.",
            ],
          },
        ],
      },
      {
        id: "experience-api",
        level: 2,
        title: "Experience",
        blocks: [
          {
            type: "code",
            language: "python",
            code: `Experience(train_dataset, test_dataset, task_id, classes_in_this_experience, dataset_name)
  .train_dataloader(batch_size, num_workers, shuffle) -> DataLoader
  .test_dataloader(batch_size, num_workers) -> DataLoader`,
          },
        ],
      },
      {
        id: "strategy-api",
        level: 2,
        title: "FisherDeltaStrategy",
        blocks: [
          {
            type: "paragraph",
            text:
              "Public methods: `train(experience)`, `eval(stream)`, `add_plugin(plugin)`. Public state: `last_certificate`, `state`, `model`, `optimizer`, and the configured training hyperparameters.",
          },
          {
            type: "list",
            items: [
              "All init parameters shown on the Strategies page are part of the public constructor.",
              "Available hooks follow the same names documented on the Plugins page.",
            ],
          },
        ],
      },
      {
        id: "certificate-api",
        level: 2,
        title: "EquivalenceCertificate",
        blocks: [
          {
            type: "definition",
            items: [
              ["`epsilon_param: float`", "Parameter-drift bound."],
              ["`kl_bound: float`", "Raw KL deviation estimate."],
              ["`kl_bound_normalized: float`", "Model-size normalized KL estimate."],
              ["`is_equivalent: bool`", "Threshold-based equivalence decision."],
              ["`shift_type: str`", "Detected shift regime for the current experience."],
              ["`ece_before / ece_after / ece_delta: float`", "Calibration diagnostics."],
              ["`compute_ratio: float`", "Full-retrain estimate divided by delta update time."],
              ["`n_old / n_new: int`", "Historical vs new task sample counts."],
              ["`ce_scale / ewc_scale: float`", "Derived scaling factors used inside the loss."],
              ["`tier: str`", "Proof regime used for the bound."],
            ],
          },
        ],
      },
      {
        id: "evaluation-api",
        level: 2,
        title: "EvaluationPlugin and register_dataset",
        blocks: [
          {
            type: "code",
            language: "python",
            code: `EvaluationPlugin(*metric_groups, loggers=None)
  .get_last_metrics() -> dict[str, float]

register_dataset(name, provider, aliases=None, overwrite=False)`,
          },
          {
            type: "paragraph",
            text:
              "`EvaluationPlugin` owns metric state. `register_dataset` extends the dataset registry with custom providers returning `TaskBundle` objects.",
          },
        ],
      },
    ],
  },
  {
    id: "faq",
    path: "/faq",
    title: "FAQ",
    description: "Common questions about accuracy, certificates, compute ratio, transformers, datasets, and replay.",
    sections: [
      {
        id: "faq-items",
        level: 2,
        title: "Frequently asked questions",
        blocks: [
          {
            type: "faq",
            items: [
              [
                "Why is my accuracy low on early tasks?",
                "Class-incremental benchmarks are hard because the model must retain old classes while adding new ones. Start by checking whether you are in class-incremental or task-incremental mode, whether the classifier grows at task-time, and whether replay / KD are enabled with reasonable budgets.",
              ],
              [
                "What does `is_equivalent=False` mean — should I be worried?",
                "Not necessarily. It means at least one configured threshold was exceeded. Accuracy may still be good. Treat it as a diagnostic saying the delta update moved further away from the estimated full-retrain solution than the current thresholds allow.",
              ],
              [
                "When should I use `FullRetrainStrategy` vs `FisherDeltaStrategy`?",
                "Use `FullRetrainStrategy` as a baseline or when you need the strongest empirical reference. Use `FisherDeltaStrategy` when you want task-by-task updates, certificates, and better compute efficiency.",
              ],
              [
                "What does `compute_ratio=3.4x` mean?",
                "The framework estimated that full retraining would have taken 3.4 times longer than the observed delta update for that task.",
              ],
              [
                "How do I add a new dataset?",
                "Register a provider with `register_dataset(name, provider, aliases=...)` and return a `TaskBundle` of `(train_dataset, test_dataset)` pairs.",
              ],
              [
                "How do I add a new strategy?",
                "Subclass `BaseStrategy`, override the relevant hooks, and reuse the existing plugin and evaluation system instead of rewriting the training loop from scratch.",
              ],
              [
                "Does this work with transformers / BERT?",
                "Yes. The text providers in the repo already use fixed embedding pipelines, and you can also wrap your own transformer or embedding extractor in a normal PyTorch module.",
              ],
              [
                "What's the difference between `kl_bound` and `kl_bound_normalized`?",
                "The raw KL estimate grows with model size. The normalized version makes it easier to compare across architectures and task sizes.",
              ],
              [
                "How many exemplars does the framework store?",
                "The public balanced path stores a bounded replay memory per class. The exact number is strategy-configurable via `replay_memory_per_class`, and the high-performance benchmarks raise that value when they intentionally trade compute for retention.",
              ],
            ],
          },
        ],
      },
    ],
  },
];

export const pagesById = Object.fromEntries(pages.map((page) => [page.id, page]));
export const pagesByPath = Object.fromEntries(pages.map((page) => [page.path, page]));
