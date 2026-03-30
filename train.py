"""
Train a simple linear classifier on MNIST.

Usage: python train.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn

from prepare import TIME_BUDGET, evaluate_accuracy, load_mnist

TRAIN_BATCH_SIZE = 4096
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    script_start = time.time()
    device = get_device()
    train_images, train_labels, val_images, val_labels = load_mnist()

    model = nn.Linear(28 * 28, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.time()
    best_val_acc = 0.0
    last_val_acc = 0.0
    last_loss = 0.0
    num_steps = 0
    epochs = 0
    examples_seen = 0

    while True:
        epochs += 1
        permutation = torch.randperm(train_labels.size(0))

        model.train()
        for start in range(0, train_labels.size(0), TRAIN_BATCH_SIZE):
            if num_steps > 0 and time.time() - train_start >= TIME_BUDGET:
                break

            batch_indices = permutation[start:start + TRAIN_BATCH_SIZE]
            batch_images = train_images[batch_indices].to(device)
            batch_labels = train_labels[batch_indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            num_steps += 1
            examples_seen += batch_labels.size(0)

        last_val_acc = evaluate_accuracy(model, val_images, val_labels, device)
        best_val_acc = max(best_val_acc, last_val_acc)
        elapsed = time.time() - train_start
        print(
            f"epoch {epochs:03d} step {num_steps:05d} "
            f"loss {last_loss:.4f} val_acc {last_val_acc:.4f} "
            f"best {best_val_acc:.4f} time {elapsed:.1f}s"
        )

        if time.time() - train_start >= TIME_BUDGET:
            break

    # Final eval after training completes, mirroring train_old.py.
    model.eval()
    final_val_acc = evaluate_accuracy(model, val_images, val_labels, device)
    best_val_acc = max(best_val_acc, final_val_acc)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    num_params = sum(parameter.numel() for parameter in model.parameters())

    print("---")
    print(f"val_acc:          {final_val_acc:.6f}")
    print(f"best_val_acc:     {best_val_acc:.6f}")
    print(f"last_val_acc:     {last_val_acc:.6f}")
    print(f"training_seconds: {time.time() - train_start:.1f}")
    print(f"total_seconds:    {time.time() - script_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"examples_seen_k:  {examples_seen / 1e3:.1f}")
    print(f"num_steps:        {num_steps}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"epochs:           {epochs}")


if __name__ == "__main__":
    main()
