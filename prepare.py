"""
One-time data preparation for autoresearch MNIST experiments.

Usage:
    python prepare.py

Data is stored in ~/.cache/autoresearch/mnist/.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from datasets import Dataset, load_dataset
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
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
DEFAULT_SEED = 0
WANDB_PROJECT = "autoresearch-mnist"
WANDB_SAVE_DIR = Path("wandb")
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


def configure_runtime(seed: int = DEFAULT_SEED, *, use_lightning: bool = False) -> None:
    if use_lightning:
        import lightning as L

        L.seed_everything(seed, workers=True)
    else:
        torch.manual_seed(seed)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


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


def finalize_wandb_run(
    wandb_logger: WandbLogger | None,
    metrics: dict[str, float],
    *,
    step: int,
) -> None:
    if wandb_logger is None:
        return

    wandb_logger.log_metrics(metrics, step=step)
    wandb_logger.experiment.finish()


def print_training_summary(
    *,
    final_val_acc: float,
    final_test_acc: float,
    best_val_acc: float,
    last_val_acc: float,
    elapsed_t: float,
    peak_vram_mb: float,
    examples_seen: int,
    num_steps: int,
    num_params: int,
    epochs: int,
    stopped_early: bool,
) -> None:
    print("---")
    print(f"val_acc:          {final_val_acc:.6f}")
    print(f"test_acc:         {final_test_acc:.6f}")
    print(f"best_val_acc:     {best_val_acc:.6f}")
    print(f"last_val_acc:     {last_val_acc:.6f}")
    print(f"total_seconds:    {elapsed_t:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"examples_seen_k:  {examples_seen / 1e3:.1f}")
    print(f"num_steps:        {num_steps}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"epochs:           {epochs}")
    print(f"stopped_early:    {stopped_early}")


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

    def on_fit_start(self, trainer, pl_module) -> None:
        self.start_t = time.time()
        device = trainer.strategy.root_device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self.best_state_dict = clone_state_dict(pl_module)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: torch.Tensor | dict[str, torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.last_loss = float(loss.detach().item())
        _, labels = batch
        self.num_steps += 1
        self.examples_seen += labels.size(0)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        self.epochs = trainer.current_epoch + 1
        self.last_val_acc = pl_module._val_correct / max(1, pl_module._val_total)

        improved = self.last_val_acc > (self.best_val_acc + self.min_delta)
        if improved:
            self.best_val_acc = self.last_val_acc
            self.best_state_dict = clone_state_dict(pl_module)
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

    def restore_best_weights(self, pl_module: torch.nn.Module) -> None:
        if self.best_state_dict is not None:
            pl_module.load_state_dict(self.best_state_dict)


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
