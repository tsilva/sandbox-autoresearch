"""
MNIST autoresearch training script.

This file is the only intended research surface. Start with the baseline
linear model, then let the agent iterate on architecture and training choices.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn

from prepare import NUM_CLASSES, SPLIT_SEED, TIME_BUDGET, evaluate, evaluate_test, make_dataloaders

MODEL_NAME = "double_block_cnn_silu"
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.03
WARMUP_STEPS = 64
DECAY_STEPS = 1536
MIN_LR_SCALE = 0.2
START_LR_SCALE = 0.5


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()


class DoubleBlockCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._double_block(1, 32),
            self._double_block(32, 64),
            self._double_block(64, 96),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, NUM_CLASSES),
        )

    @staticmethod
    def _double_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model() -> nn.Module:
    return DoubleBlockCNN()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_model(model: nn.Module, device: torch.device) -> tuple[int, float]:
    train_loader, _, _ = make_dataloaders(
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()
    train_start = time.perf_counter()
    num_steps = 0

    while True:
        for images, labels in train_loader:
            synchronize(device)
            if time.perf_counter() - train_start >= TIME_BUDGET:
                training_seconds = time.perf_counter() - train_start
                return num_steps, training_seconds

            images = images.to(device)
            labels = labels.to(device)
            optimizer.param_groups[0]["lr"] = LEARNING_RATE * _learning_rate_scale(num_steps)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                label_smoothing=LABEL_SMOOTHING,
            )
            loss.backward()
            optimizer.step()

            num_steps += 1


def _learning_rate_scale(step: int) -> float:
    if step < WARMUP_STEPS:
        warmup_progress = float(step + 1) / float(WARMUP_STEPS)
        return START_LR_SCALE + (1.0 - START_LR_SCALE) * warmup_progress

    decay_step = min(step - WARMUP_STEPS, DECAY_STEPS)
    progress = float(decay_step) / float(max(1, DECAY_STEPS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_SCALE + (1.0 - MIN_LR_SCALE) * cosine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="After validation, also evaluate on the held-out MNIST test set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SPLIT_SEED)
    total_start = time.perf_counter()
    device = pick_device()
    model = build_model().to(device)

    num_steps, training_seconds = train_model(model, device)
    _, val_loader, _ = make_dataloaders(
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    )
    synchronize(device)
    val_metrics = evaluate(model, val_loader, device)
    synchronize(device)

    print("---")
    print(f"model_name:       {MODEL_NAME}")
    print(f"device:           {device.type}")
    print(f"val_accuracy:     {val_metrics['accuracy']:.6f}")
    print(f"val_loss:         {val_metrics['loss']:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {time.perf_counter() - total_start:.1f}")
    print(f"num_steps:        {num_steps}")
    print(f"num_params_k:     {count_parameters(model) / 1_000:.1f}")

    if args.final_test:
        test_metrics = evaluate_test(model, device, eval_batch_size=EVAL_BATCH_SIZE)
        synchronize(device)
        print(f"test_accuracy:    {test_metrics['accuracy']:.6f}")
        print(f"test_loss:        {test_metrics['loss']:.6f}")


if __name__ == "__main__":
    main()
