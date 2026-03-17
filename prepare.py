"""
One-time data preparation and fixed evaluation harness for MNIST autoresearch.

Usage:
    uv run prepare.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Fixed constants (do not modify during research)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60
TRAIN_SIZE = 55_000
VAL_SIZE = 5_000
SPLIT_SEED = 1337
IMAGE_SHAPE = (1, 28, 28)
INPUT_DIM = 28 * 28
NUM_CLASSES = 10

CACHE_DIR = Path.home() / ".cache" / "mnist-autoresearch"
DATA_DIR = CACHE_DIR / "data"
SPLIT_FILE = CACHE_DIR / "split_indices.pt"

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def ensure_cache() -> dict[str, int]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    transform = build_transform()
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    if SPLIT_FILE.exists():
        split_indices = torch.load(SPLIT_FILE, map_location="cpu")
    else:
        generator = torch.Generator().manual_seed(SPLIT_SEED)
        permutation = torch.randperm(len(train_dataset), generator=generator)
        split_indices = {
            "train": permutation[:TRAIN_SIZE],
            "val": permutation[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE],
        }
        torch.save(split_indices, SPLIT_FILE)

    return {
        "train_examples": len(split_indices["train"]),
        "val_examples": len(split_indices["val"]),
        "test_examples": len(test_dataset),
    }


def _load_split_indices() -> dict[str, torch.Tensor]:
    if not SPLIT_FILE.exists():
        ensure_cache()
    split_indices = torch.load(SPLIT_FILE, map_location="cpu")
    train_indices = split_indices["train"]
    val_indices = split_indices["val"]
    if len(train_indices) != TRAIN_SIZE or len(val_indices) != VAL_SIZE:
        raise RuntimeError("Cached split does not match expected train/val sizes.")
    return split_indices


def _num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(4, cpu_count)


def make_dataloaders(
    train_batch_size: int = 256,
    eval_batch_size: int = 1024,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    ensure_cache()
    split_indices = _load_split_indices()
    transform = build_transform()

    full_train = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=transform)

    train_dataset = Subset(full_train, split_indices["train"].tolist())
    val_dataset = Subset(full_train, split_indices["val"].tolist())

    worker_count = _num_workers() if num_workers is None else num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=worker_count,
        persistent_workers=worker_count > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=worker_count,
        persistent_workers=worker_count > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=worker_count,
        persistent_workers=worker_count > 0,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def evaluate_test(
    model: torch.nn.Module,
    device: torch.device,
    eval_batch_size: int = 1024,
    num_workers: int | None = None,
) -> dict[str, float]:
    _, _, test_loader = make_dataloaders(
        train_batch_size=256,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    return evaluate(model, test_loader, device)


def main() -> None:
    stats = ensure_cache()
    print(f"cache_dir:      {CACHE_DIR}")
    print(f"train_examples: {stats['train_examples']}")
    print(f"val_examples:   {stats['val_examples']}")
    print(f"test_examples:  {stats['test_examples']}")
    print(f"split_seed:     {SPLIT_SEED}")
    print(f"time_budget_s:  {TIME_BUDGET}")


if __name__ == "__main__":
    main()

