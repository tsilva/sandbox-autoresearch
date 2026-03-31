"""
One-time data preparation for autoresearch MNIST experiments.

Usage:
    python prepare.py

Data is stored in ~/.cache/autoresearch/mnist/.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300  # training time budget in seconds (5 minutes)
EVAL_BATCH_SIZE = 10000

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autoresearch" / "mnist"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
DATASET_NAME = "ylecun/mnist"
VALIDATION_SIZE = 5_000
SPLIT_SEED = 0
SPLIT_FILES = {
    "train": "train.pt",
    "validation": "validation.pt",
    "test": "test.pt",
}


def prepare_data() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, cache_dir=str(HF_CACHE_DIR))
    train_val_split = dataset["train"].train_test_split(
        test_size=VALIDATION_SIZE,
        seed=SPLIT_SEED,
        stratify_by_column="label",
    )
    prepared_splits = {
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": dataset["test"],
    }

    for split_name, split in prepared_splits.items():
        payload = _dataset_split_to_payload(split)
        torch.save(payload, CACHE_DIR / SPLIT_FILES[split_name])
        print(f"Prepared {split_name} split with {payload['labels'].numel()} examples")

    print(f"MNIST ready at {CACHE_DIR}")


def _dataset_split_to_payload(split: Dataset) -> dict[str, torch.Tensor]:
    images = np.empty((len(split), 28 * 28), dtype=np.float32)
    labels = np.empty((len(split),), dtype=np.int64)

    for index, example in enumerate(split):
        images[index] = np.asarray(example["image"], dtype=np.float32).reshape(-1)
        labels[index] = example["label"]

    image_tensor = torch.from_numpy(images)
    image_tensor.div_(255.0)
    image_tensor.sub_(0.1307).div_(0.3081)
    label_tensor = torch.from_numpy(labels)
    return {"images": image_tensor, "labels": label_tensor}


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    missing = [name for name in SPLIT_FILES.values() if not (CACHE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Prepared MNIST splits not found in {CACHE_DIR}. Run `python prepare.py` first."
        )

    train_payload = torch.load(CACHE_DIR / SPLIT_FILES["train"], weights_only=True)
    val_payload = torch.load(CACHE_DIR / SPLIT_FILES["validation"], weights_only=True)
    test_payload = torch.load(CACHE_DIR / SPLIT_FILES["test"], weights_only=True)
    return (
        train_payload["images"],
        train_payload["labels"],
        val_payload["images"],
        val_payload["labels"],
        test_payload["images"],
        test_payload["labels"],
    )


def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize(device)
    elif device.type == "mps": torch.mps.synchronize()


def get_peak_vram_mb(device):
    return (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = EVAL_BATCH_SIZE,
) -> float:
    model.eval()
    correct = 0
    total = labels.numel()
    for start in range(0, total, batch_size):
        end = start + batch_size
        batch_images = images[start:end].to(device)
        batch_labels = labels[start:end].to(device)
        predictions = model(batch_images).argmax(dim=1)
        correct += (predictions == batch_labels).sum().item()
    return correct / total


if __name__ == "__main__":
    prepare_data()
    print("Done! Ready to train.")
