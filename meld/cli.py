"""Command line interface for MELD."""

from __future__ import annotations

import argparse
import json

from .api import MELDConfig, TrainConfig, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MELD benchmarks.")
    parser.add_argument("--dataset", default="CIFAR-10")
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--classes-per-task", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--backbone", default="resnet32")
    parser.add_argument("--bound-tolerance", type=float, default=0.01)
    parser.add_argument("--shift-threshold", type=float, default=0.3)
    parser.add_argument("--lambda-geometry", type=float, default=1.0)
    parser.add_argument("--lambda-ewc", type=float, default=0.4)
    parser.add_argument("--geometry-decay", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefer-cuda", action="store_true")
    parser.add_argument("--results-path", default="results.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MELDConfig(
        dataset=args.dataset,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        prefer_cuda=args.prefer_cuda,
        bound_tolerance=args.bound_tolerance,
        shift_threshold=args.shift_threshold,
        train=TrainConfig(
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_geometry=args.lambda_geometry,
            lambda_ewc=args.lambda_ewc,
            geometry_decay=args.geometry_decay,
            num_workers=args.num_workers,
        ),
    )
    results = run(config, results_path=args.results_path)
    print(json.dumps(results["final_summary"], indent=2))


if __name__ == "__main__":
    main()
