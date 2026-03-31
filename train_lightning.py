from __future__ import annotations

import os
import time
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prepare import (
    EVAL_BATCH_SIZE,
    TIME_BUDGET,
    evaluate_accuracy,
    get_device,
    get_peak_vram_mb,
    load_mnist,
)


TRAIN_BATCH_SIZE = 4096
LEARNING_RATE = 3e-3
MAX_EPOCHS = 10_000

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4
WANDB_PROJECT = "autoresearch-mnist"
WANDB_SAVE_DIR = Path("wandb")


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


class TrainingSummaryCallback(Callback):
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.start_t = 0.0
        self.best_val_acc = 0.0
        self.last_val_acc = 0.0
        self.last_loss = 0.0
        self.num_steps = 0
        self.epochs = 0
        self.examples_seen = 0
        self.epochs_without_improvement = 0
        self.stopped_early = False
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.start_t = time.time()
        device = trainer.strategy.root_device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self.best_state_dict = {
            key: value.detach().clone()
            for key, value in pl_module.state_dict().items()
        }

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | dict[str, torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.last_loss = float(loss.detach().item())
        _, labels = batch
        self.num_steps += 1
        self.examples_seen += labels.size(0)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        self.epochs = trainer.current_epoch + 1
        self.last_val_acc = pl_module._val_correct / max(1, pl_module._val_total)

        improved = self.last_val_acc > (self.best_val_acc + self.min_delta)
        if improved:
            self.best_val_acc = self.last_val_acc
            self.best_state_dict = {
                key: value.detach().clone()
                for key, value in pl_module.state_dict().items()
            }
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        elapsed_t = time.time() - self.start_t
        print(
            f"epoch {self.epochs:03d} step {self.num_steps:05d} "
            f"loss {self.last_loss:.4f} val_acc {self.last_val_acc:.4f} "
            f"best {self.best_val_acc:.4f} "
            f"no_improve {self.epochs_without_improvement}/{self.patience} "
            f"time {elapsed_t:.1f}s"
        )

        if elapsed_t >= TIME_BUDGET:
            trainer.should_stop = True
            return

        if self.epochs_without_improvement >= self.patience:
            self.stopped_early = True
            print(
                f"early stopping triggered after {self.epochs} epochs "
                f"with best_val_acc={self.best_val_acc:.4f}"
            )
            trainer.should_stop = True

        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "train/examples_seen": float(self.examples_seen),
                    "train/examples_seen_k": self.examples_seen / 1e3,
                    "train/epoch": float(self.epochs),
                    "train/no_improve_epochs": float(self.epochs_without_improvement),
                    "train/seconds_elapsed": elapsed_t,
                    "val/best_acc": self.best_val_acc,
                },
                step=self.num_steps,
            )

    def restore_best_weights(self, pl_module: L.LightningModule) -> None:
        if self.best_state_dict is not None:
            pl_module.load_state_dict(self.best_state_dict)


def main() -> None:
    L.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
                "dataset": "ylecun/mnist",
                "validation_size": 5_000,
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

    print("---")
    print(f"val_acc:          {final_val_acc:.6f}")
    print(f"test_acc:         {final_test_acc:.6f}")
    print(f"best_val_acc:     {stats_callback.best_val_acc:.6f}")
    print(f"last_val_acc:     {stats_callback.last_val_acc:.6f}")
    print(f"total_seconds:    {elapsed_t:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"examples_seen_k:  {stats_callback.examples_seen / 1e3:.1f}")
    print(f"num_steps:        {stats_callback.num_steps}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"epochs:           {stats_callback.epochs}")
    print(f"stopped_early:    {stats_callback.stopped_early}")

    if wandb_logger is not None:
        wandb_logger.log_metrics(
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
        wandb_logger.experiment.finish()


def build_wandb_logger() -> WandbLogger:
    WANDB_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    wandb_mode = os.getenv("WANDB_MODE", "").strip().lower()
    offline = wandb_mode == "offline" or (
        wandb_mode == "" and not os.getenv("WANDB_API_KEY")
    )
    project = os.getenv("WANDB_PROJECT", WANDB_PROJECT)
    name = os.getenv("WANDB_RUN_NAME")

    print(
        f"wandb: project={project} mode={'offline' if offline else 'online'} "
        f"dir={WANDB_SAVE_DIR}"
    )
    return WandbLogger(
        project=project,
        name=name,
        save_dir=str(WANDB_SAVE_DIR),
        offline=offline,
        log_model=False,
    )


if __name__ == "__main__":
    main()
