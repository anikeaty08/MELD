"""Command line interface for MELD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import MELDConfig, TrainConfig, run
from .core.snapshot import FisherManifoldSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MELD benchmarks.")
    parser.add_argument(
        "--dataset",
        default="CIFAR-10",
        help=(
            "Dataset to use. Image: synthetic, CIFAR-10, CIFAR-100, TinyImageNet, STL-10. "
            "Text: AGNews (4 cls), DBpedia (14 cls)."
        ),
    )
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--classes-per-task", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--backbone",
        default="auto",
        choices=["auto", "resnet20", "resnet32", "resnet44", "resnet56", "resnet18_imagenet", "text_encoder"],
    )
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--incremental-strategy", default="geometry", choices=["geometry", "frozen_analytic"])
    parser.add_argument(
        "--text-encoder-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model for text backbone. Only used when --backbone text_encoder.",
    )
    parser.add_argument("--base-epochs", type=int, default=None)
    # Keep CLI default aligned with MELDConfig default (too-small values skip delta
    # training almost immediately and make comparisons misleading).
    parser.add_argument("--bound-tolerance", type=float, default=10.0)
    parser.add_argument("--pac-gate-tolerance", type=float, default=0.5)
    parser.add_argument("--shift-threshold", type=float, default=0.3)
    parser.add_argument("--lambda-geometry", type=float, default=5.0)
    parser.add_argument("--lambda-ewc", type=float, default=1.0)
    parser.add_argument("--geometry-decay", type=float, default=0.3)
    parser.add_argument("--analytic-ridge", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--fisher-samples", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefer-cuda", action="store_true")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--database-path", default="meld_results.db")
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
        pac_gate_tolerance=args.pac_gate_tolerance,
        shift_threshold=args.shift_threshold,
        data_root=Path(args.data_root),
        database_path=Path(args.database_path) if args.database_path else None,
        train=TrainConfig(
            backbone=args.backbone,
            pretrained_backbone=args.pretrained_backbone,
            incremental_strategy=args.incremental_strategy,
            epochs=args.epochs,
            base_epochs=args.base_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_geometry=args.lambda_geometry,
            lambda_ewc=args.lambda_ewc,
            geometry_decay=args.geometry_decay,
            analytic_ridge=args.analytic_ridge,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            max_grad_norm=args.max_grad_norm,
            num_workers=args.num_workers,
            text_encoder_model=args.text_encoder_model,
        ),
    )
    snapshot_strategy = FisherManifoldSnapshot(fisher_samples=int(args.fisher_samples))
    results = run(config, results_path=args.results_path, snapshot_strategy=snapshot_strategy)
    print(json.dumps(results["final_summary"], indent=2))


if __name__ == "__main__":
    main()
