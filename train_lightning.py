from __future__ import annotations

import lightning as L
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prepare import (
    EVAL_BATCH_SIZE,
    TIME_BUDGET,
    VALIDATION_SIZE,
    DATASET_NAME,
    TrainingSummaryCallback,
    build_wandb_logger,
    configure_runtime,
    evaluate_accuracy,
    finalize_wandb_run,
    get_device,
    get_peak_vram_mb,
    load_mnist,
    print_training_summary,
)


TRAIN_BATCH_SIZE = 4096
LEARNING_RATE = 3e-3
MAX_EPOCHS = 10_000

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4


class LitMNISTClassifier(L.LightningModule):
    def __init__(self, learning_rate: float = LEARNING_RATE) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = nn.Linear(28 * 28, 10)
        self._val_correct = 0
        self._val_total = 0
        self.last_val_acc = 0.0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_correct = 0
        self._val_total = 0

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, labels = batch
        logits = self(images)
        predictions = logits.argmax(dim=1)
        self._val_correct += int((predictions == labels).sum().item())
        self._val_total += labels.size(0)

    def on_validation_epoch_end(self) -> None:
        self.last_val_acc = self._val_correct / max(1, self._val_total)
        self.log("val/acc", self.last_val_acc, prog_bar=False, logger=True, sync_dist=False)
        self.log("val_acc", self.last_val_acc, prog_bar=False, logger=False, sync_dist=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def main() -> None:
    configure_runtime(use_lightning=True)

    device = get_device()
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist()

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(val_images, val_labels),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = LitMNISTClassifier()
    stats_callback = TrainingSummaryCallback(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    )
    wandb_logger = build_wandb_logger()

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        callbacks=[stats_callback],
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=wandb_logger,
        max_epochs=MAX_EPOCHS,
        num_sanity_val_steps=0,
    )
    if wandb_logger is not None:
        wandb_logger.log_hyperparams(
            {
                "train_batch_size": TRAIN_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                "time_budget_seconds": TIME_BUDGET,
                "model": "linear-784x10",
                "dataset": DATASET_NAME,
                "validation_size": VALIDATION_SIZE,
                "script": "train_lightning.py",
            }
        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    stats_callback.restore_best_weights(model)
    model.to(device)
    final_val_acc = evaluate_accuracy(model, val_images, val_labels, device)
    final_test_acc = evaluate_accuracy(model, test_images, test_labels, device)
    elapsed_t = time.time() - stats_callback.start_t

    peak_vram_mb = get_peak_vram_mb(device)
    num_params = sum(parameter.numel() for parameter in model.parameters())

    print_training_summary(
        final_val_acc=final_val_acc,
        final_test_acc=final_test_acc,
        best_val_acc=stats_callback.best_val_acc,
        last_val_acc=stats_callback.last_val_acc,
        elapsed_t=elapsed_t,
        peak_vram_mb=peak_vram_mb,
        examples_seen=stats_callback.examples_seen,
        num_steps=stats_callback.num_steps,
        num_params=num_params,
        epochs=stats_callback.epochs,
        stopped_early=stats_callback.stopped_early,
    )

    finalize_wandb_run(
        wandb_logger,
        {
            "summary/final_val_acc": final_val_acc,
            "summary/final_test_acc": final_test_acc,
            "summary/best_val_acc": stats_callback.best_val_acc,
            "summary/last_val_acc": stats_callback.last_val_acc,
            "summary/total_seconds": elapsed_t,
            "summary/peak_vram_mb": peak_vram_mb,
            "summary/examples_seen_k": stats_callback.examples_seen / 1e3,
            "summary/num_steps": float(stats_callback.num_steps),
            "summary/num_params_m": num_params / 1e6,
            "summary/epochs": float(stats_callback.epochs),
            "summary/stopped_early": float(stats_callback.stopped_early),
        },
        step=stats_callback.num_steps,
    )


if __name__ == "__main__":
    main()
