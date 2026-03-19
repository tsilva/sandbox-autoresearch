# mnist-autoresearch

Minimal MNIST adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The workflow stays intentionally small:

- `prepare.py` is the fixed harness for data, splits, and evaluation.
- `train.py` is the only file the autonomous researcher edits.
- `modal_run.py` runs each experiment remotely on Modal while keeping the agent and git state local.
- `experiment_tools.py` manages checksums, worktrees, run parsing, ranking, and `results.tsv`.
- `orchestrate.py` is the deterministic experiment runtime around agent-driven `train.py` edits.
- `run_exact_commit.py` exports one committed snapshot with `git archive` and runs that exact tree on Modal.
- `program.md` documents the policy boundary: Codex chooses experiments, the runtime handles mechanics.

This repo targets Apple Silicon first via `mps`, with `cpu` fallback. Each experiment is capped at a fixed 60-second wall-clock training budget and ranked by validation accuracy, then validation loss, then simplicity.

## Quick start

```bash
uv sync
uv run modal setup
uv run python experiment_tools.py ensure-results
uv run python run_exact_commit.py --commit HEAD > run.log 2>&1
```

If you want a final held-out test report for a selected model:

```bash
uv run python run_exact_commit.py --commit HEAD --final-test > run.log 2>&1
```

## Workflow

The setup mirrors the original autoresearch pattern, but now the coordinator uses exact committed snapshots and `train.py` checksums to avoid duplicate runs:

1. Read `README.md`, `prepare.py`, `train.py`, `modal_run.py`, `experiment_tools.py`, `run_exact_commit.py`, and `program.md`.
2. Ensure Modal is authenticated by running `uv run modal setup`.
3. Create a fresh branch named `codex/autoresearch/<tag>`.
4. Initialize or migrate `results.tsv` with `uv run python experiment_tools.py ensure-results`.
5. Create an ideas file with one experiment idea per line.
6. Start a round with `uv run python orchestrate.py start-round --ideas-file ideas.txt`.
7. Let Codex dispatch workers using the generated task and prompt artifacts.
8. Approve unique committed candidates with `uv run python orchestrate.py approve-candidates --round-id <id>`.
9. Execute approved candidates and record results with `uv run python orchestrate.py finalize-round --round-id <id>`.

The intended experiment log is an untracked `results.tsv` with this schema:

```text
commit	train_py_sha256	val_accuracy	val_loss	status	description
```

`status` must be one of `keep`, `discard`, or `crash`.

## Fixed harness

`prepare.py` owns all non-research state:

- downloads and caches MNIST under `~/.cache/mnist-autoresearch`
- creates a deterministic `55k / 5k` train/validation split from the standard 60k training set
- exposes the dataloader and evaluation helpers used by `train.py`

The MNIST test set is held out from optimization and only used for the final selected model.

## Remote execution

`modal_run.py` still sends the current working directory to a Modal worker, runs `train.py` there, and writes the remote stdout and stderr back to the local process. The intended entrypoint is now `run_exact_commit.py`, which first exports one immutable git revision with `git archive`, then runs `modal_run.py` from that exported directory:

```bash
uv run python run_exact_commit.py --commit HEAD > run.log 2>&1
grep "^val_accuracy:\|^val_loss:" run.log
tail -n 50 run.log
```

The MNIST cache is stored remotely in a named Modal Volume so repeated runs do not re-download the dataset each time. The remote function requests a single small GPU, preferring `T4` and falling back to `L4` when needed.

## Coordination helpers

`experiment_tools.py` is the deterministic helper surface:

```bash
uv run python experiment_tools.py ensure-results
uv run python experiment_tools.py checksum --commit HEAD --json
uv run python experiment_tools.py preflight --commit HEAD --reserved <sha256>
uv run python experiment_tools.py append-result \
  --commit HEAD \
  --val-accuracy 0.990000 \
  --val-loss 0.050000 \
  --status keep \
  --description "cnn variant"
```

`preflight` is the duplicate gate. If two commits have the same `train.py` SHA-256, only one of them should be allowed to run.

`orchestrate.py` owns round state and worktree lifecycle:

```bash
uv run python orchestrate.py start-round --ideas-file ideas.txt
uv run python orchestrate.py approve-candidates --round-id <id>
uv run python orchestrate.py finalize-round --round-id <id>
uv run python orchestrate.py resume --round-id <id>
```

Each round writes a JSON artifact under `artifacts/<round-id>/round.json` plus task, prompt, preflight, and run payload files. Codex remains responsible for proposing ideas and interpreting the resulting state; the runtime only manages the repeated mechanical steps.

## Device support

The default device order is:

1. `cuda`
2. `mps`
3. `cpu`

Modal workers request a single GPU and should report `cuda` when scheduled on Modal.
