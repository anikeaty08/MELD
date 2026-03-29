"""CLI for the delta framework demos."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run delta framework experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--scenario", default="class_incremental",
                        choices=["class_incremental", "task_incremental", "domain_incremental"])
    parser.add_argument("--run-mode", default="compare",
                        choices=["compare", "compare_replay", "fisher_delta", "replay_delta", "full_retrain"])
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--classes-per-task", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--backbone", default="auto",
                        choices=["auto", "resnet20", "resnet32", "resnet44", "resnet56", "resnet18_imagenet"])
    parser.add_argument("--preset", default="standard", choices=["standard", "maxperf"])
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--use-imagenet-stats", action="store_true")
    parser.add_argument("--task-identity-inference", action="store_true")
    parser.add_argument("--replay-memory-per-class", type=int, default=None)
    parser.add_argument("--replay-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quiet:
        args.verbose = False

    config = {
        "dataset": args.dataset,
        "scenario": args.scenario,
        "run_mode": args.run_mode,
        "num_tasks": args.num_tasks,
        "classes_per_task": args.classes_per_task,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "backbone": args.backbone,
        "preset": args.preset,
        "pretrained_backbone": args.pretrained_backbone,
        "image_size": args.image_size,
        "use_imagenet_stats": args.use_imagenet_stats,
        "task_identity_inference": args.task_identity_inference,
        "num_workers": args.num_workers,
        "data_root": args.data_root,
        "seed": args.seed,
        "verbose": args.verbose,
    }
    if args.replay_memory_per_class is not None:
        config["replay_memory_per_class"] = args.replay_memory_per_class
    if args.replay_batch_size is not None:
        config["replay_batch_size"] = args.replay_batch_size

    from .runner import DemoRunner
    runner = DemoRunner(config)
    results = runner.run(results_path=args.results_path)

    summary = results.get("final_summary", {})
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
