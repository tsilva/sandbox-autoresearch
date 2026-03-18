# mnist-autoresearch

Minimal MNIST adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The workflow stays intentionally small:

- `prepare.py` is the fixed harness for data, splits, and evaluation.
- `train.py` is the only file the autonomous researcher edits.
- `modal_run.py` runs each experiment remotely on Modal while keeping the agent and git state local.
- `program.md` is the human-authored prompt that defines the experiment loop.

This repo targets Apple Silicon first via `mps`, with `cpu` fallback. Each experiment is capped at a fixed 60-second wall-clock training budget and ranked by validation accuracy, then validation loss, then simplicity.

## Quick start

```bash
uv sync
uv run modal setup
uv run modal run modal_run.py > run.log 2>&1
```

If you want a final held-out test report for a selected model:

```bash
uv run modal run modal_run.py --final-test > run.log 2>&1
```

## Workflow

The setup mirrors the original autoresearch pattern, but for MNIST. The agent still edits locally, commits locally, and maintains logs locally:

1. Read `README.md`, `prepare.py`, `train.py`, `modal_run.py`, and `program.md`.
2. Ensure Modal is authenticated by running `uv run modal setup`.
3. Create a fresh branch named `codex/autoresearch/<tag>`.
4. Run the baseline linear classifier first with `uv run modal run modal_run.py > run.log 2>&1`.
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

## Remote execution

`modal_run.py` sends the current repo contents to a Modal worker, runs `train.py` there, and writes the remote stdout and stderr back to the local process. That means this still works exactly as before:

```bash
uv run modal run modal_run.py > run.log 2>&1
grep "^val_accuracy:\|^val_loss:" run.log
tail -n 50 run.log
```

The MNIST cache is stored remotely in a named Modal Volume so repeated runs do not re-download the dataset each time. The remote function requests a single small GPU, preferring `T4` and falling back to `L4` when needed.

## Device support

The default device order is:

1. `cuda`
2. `mps`
3. `cpu`

Modal workers request a single GPU and should report `cuda` when scheduled on Modal.
