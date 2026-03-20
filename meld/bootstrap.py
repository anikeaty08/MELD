"""Environment and dataset bootstrap helpers for MELD."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap MELD datasets and dependency checks.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["CIFAR-10", "CIFAR-100"],
        help="Datasets to validate/download. Supported: CIFAR-10, CIFAR-100, CIFAR-10-C.",
    )
    parser.add_argument("--data-root", default="./data")
    return parser


def _normalize_dataset_name(name: str) -> str:
    return name.upper().replace("-", "")


def _check_imports() -> None:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        from continuum.datasets import CIFAR10, CIFAR100  # noqa: F401
    except Exception as exc:  # pragma: no cover - bootstrap helper
        raise RuntimeError(
            "Missing MELD runtime dependencies. Run `pip install -r requirements.txt` first."
        ) from exc


def _download_dataset(name: str, data_root: Path) -> None:
    from continuum.datasets import CIFAR10, CIFAR100

    dataset_name = _normalize_dataset_name(name)
    if dataset_name == "CIFAR10":
        CIFAR10(data_path=str(data_root), train=True, download=True)
        CIFAR10(data_path=str(data_root), train=False, download=True)
        return
    if dataset_name == "CIFAR100":
        CIFAR100(data_path=str(data_root), train=True, download=True)
        CIFAR100(data_path=str(data_root), train=False, download=True)
        return
    if dataset_name == "CIFAR10C":
        _download_cifar_c(data_root)
        return
    raise ValueError(f"Unsupported dataset '{name}'.")


def _download_cifar_c(data_root: Path) -> None:
    """Download and extract CIFAR-10-C into data_root / 'CIFAR10-C'."""
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    archive_path = data_root / "CIFAR-10-C.tar"
    target_dir = data_root / "CIFAR10-C"
    labels_path = target_dir / "labels.npy"
    if labels_path.exists():
        return
    data_root.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, archive_path)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=data_root)
    extracted_dir = data_root / "CIFAR-10-C"
    if extracted_dir.exists() and not target_dir.exists():
        extracted_dir.rename(target_dir)
    if archive_path.exists():
        archive_path.unlink()


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    _check_imports()

    for dataset_name in args.datasets:
        _download_dataset(dataset_name, data_root)
        print(f"ready: {dataset_name} -> {data_root}")


if __name__ == "__main__":
    main()
