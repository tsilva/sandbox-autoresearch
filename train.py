from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn

from prepare import (
    TIME_BUDGET,
    evaluate_accuracy,
    load_mnist,
    get_device,
    get_peak_vram_mb,
)


TRAIN_BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_MIN_DELTA = 1e-4


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.view(-1, 1, 28, 28)
        return self.classifier(self.features(images))


def main() -> None:
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()
    train_images, train_labels, val_images, val_labels = load_mnist()

    model = ConvNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_acc = 0.0
    best_state_dict = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    epochs_without_improvement = 0

    last_val_acc = 0.0
    last_loss = 0.0
    num_steps = 0
    epochs = 0
    examples_seen = 0
    start_t = time.time()
    stopped_early = False

    while True:
        epochs += 1
        permutation = torch.randperm(train_labels.size(0))

        model.train()
        for start in range(0, train_labels.size(0), TRAIN_BATCH_SIZE):
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

        improved = last_val_acc > (best_val_acc + EARLY_STOPPING_MIN_DELTA)
        if improved:
            best_val_acc = last_val_acc
            best_state_dict = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        elapsed_t = time.time() - start_t
        print(
            f"epoch {epochs:03d} step {num_steps:05d} "
            f"loss {last_loss:.4f} val_acc {last_val_acc:.4f} "
            f"best {best_val_acc:.4f} no_improve {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} "
            f"time {elapsed_t:.1f}s"
        )

        if elapsed_t >= TIME_BUDGET:
            break

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print(
                f"early stopping triggered after {epochs} epochs "
                f"with best_val_acc={best_val_acc:.4f}"
            )
            break

    model.load_state_dict(best_state_dict)
    final_val_acc = evaluate_accuracy(model, val_images, val_labels, device)
    elapsed_t = time.time() - start_t

    peak_vram_mb = get_peak_vram_mb(device)
    num_params = sum(parameter.numel() for parameter in model.parameters())

    print("---")
    print(f"val_acc:          {final_val_acc:.6f}")
    print(f"best_val_acc:     {best_val_acc:.6f}")
    print(f"last_val_acc:     {last_val_acc:.6f}")
    print(f"total_seconds:    {elapsed_t:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"examples_seen_k:  {examples_seen / 1e3:.1f}")
    print(f"num_steps:        {num_steps}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"epochs:           {epochs}")
    print(f"stopped_early:    {stopped_early}")


if __name__ == "__main__":
    main()
