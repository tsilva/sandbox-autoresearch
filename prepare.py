"""
One-time data preparation for autoresearch MNIST experiments.

Usage:
    python prepare.py

Data is stored in ~/.cache/autoresearch/mnist/.
"""

from __future__ import annotations

import gzip
import shutil
import struct
import urllib.request
from pathlib import Path

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
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "val_images": "t10k-images-idx3-ubyte.gz",
    "val_labels": "t10k-labels-idx1-ubyte.gz",
}
BASE_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]


def download_file(filename: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / filename
    if destination.exists():
        return destination

    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    last_error = None
    for base_url in BASE_URLS:
        try:
            with urllib.request.urlopen(base_url + filename) as response, open(temp_destination, "wb") as handle:
                shutil.copyfileobj(response, handle)
            temp_destination.replace(destination)
            print(f"Downloaded {filename}")
            return destination
        except Exception as exc:  # pragma: no cover - network failures are environment-specific
            last_error = exc
            if temp_destination.exists():
                temp_destination.unlink()
    raise RuntimeError(f"failed to download {filename}") from last_error


def prepare_data() -> None:
    for filename in FILES.values():
        download_file(filename)
    print(f"MNIST ready at {CACHE_DIR}")


def _load_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected image magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8).astype(np.float32).reshape(count, rows * cols)
    tensor = torch.from_numpy(data)
    tensor.div_(255.0)
    tensor.sub_(0.1307).div_(0.3081)
    return tensor


def _load_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8).astype(np.int64).reshape(count)
    return torch.from_numpy(data)


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    missing = [name for name in FILES.values() if not (CACHE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"MNIST files not found in {CACHE_DIR}. Run `python prepare.py` first."
        )

    train_images = _load_images(CACHE_DIR / FILES["train_images"])
    train_labels = _load_labels(CACHE_DIR / FILES["train_labels"])
    val_images = _load_images(CACHE_DIR / FILES["val_images"])
    val_labels = _load_labels(CACHE_DIR / FILES["val_labels"])
    return train_images, train_labels, val_images, val_labels


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
