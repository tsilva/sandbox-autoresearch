"""
MNIST autoresearch training script.

This file is the only intended research surface. Start with the baseline
linear model, then let the agent iterate on architecture and training choices.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from prepare import INPUT_DIM, NUM_CLASSES, SPLIT_SEED, TIME_BUDGET, evaluate, evaluate_test, make_dataloaders

MODEL_NAME = "linear"
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 1024
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4


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


class LinearClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(INPUT_DIM, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images.view(images.size(0), -1))


def build_model() -> nn.Module:
    return LinearClassifier()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_model(model: nn.Module, device: torch.device) -> tuple[int, float]:
    train_loader, _, _ = make_dataloaders(
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
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

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            num_steps += 1


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
