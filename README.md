# mnist-autoresearch

Minimal MNIST adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The workflow stays intentionally small:

- `prepare.py` is the fixed harness for data, splits, and evaluation.
- `train.py` is the only file the autonomous researcher edits.
- `program.md` is the human-authored prompt that defines the experiment loop.

This repo targets Apple Silicon first via `mps`, with `cpu` fallback. Each experiment is capped at a fixed 60-second wall-clock training budget and ranked by validation accuracy, then validation loss, then simplicity.

## Quick start

```bash
uv sync
uv run prepare.py
uv run train.py
```

If you want a final held-out test report for a selected model:

```bash
uv run train.py --final-test
```

## Workflow

The setup mirrors the original autoresearch pattern, but for MNIST:

1. Read `README.md`, `prepare.py`, `train.py`, and `program.md`.
2. Ensure the cache exists by running `uv run prepare.py`.
3. Create a fresh branch named `codex/autoresearch/<tag>`.
4. Run the baseline linear classifier first.
5. Let the agent iterate only on `train.py`, keeping better runs and discarding worse ones.

The intended experiment log is an untracked `results.tsv` with this schema:

```text
commit	val_accuracy	val_loss	status	description
```

`status` must be one of `keep`, `discard`, or `crash`.

## Fixed harness

`prepare.py` owns all non-research state:

- downloads and caches MNIST under `~/.cache/mnist-autoresearch`
- creates a deterministic `55k / 5k` train/validation split from the standard 60k training set
- exposes the dataloader and evaluation helpers used by `train.py`

The MNIST test set is held out from optimization and only used for the final selected model.

## Device support

The default device order is:

1. `mps`
2. `cpu`

No CUDA-specific logic is included in this repo.

