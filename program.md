# mnist-autoresearch

This is an experiment to have the LLM do its own research on MNIST, constrained to CNN-like image models. The main Codex agent is the coordinator. It owns the current best commit, creates worker worktrees, assigns ideas, gates candidates by `train.py` checksum, and records results.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date. The branch `codex/autoresearch/<tag>` must not already exist. This is a fresh run.
2. **Create the branch**: `git checkout -b codex/autoresearch/<tag>` from the current default branch.
3. **Read the in-scope files**: the repo is small. Read these files for full context:
   - `README.md` - repository context.
   - `prepare.py` - fixed constants, data prep, data split, dataloaders, and evaluation.
   - `train.py` - the file you modify.
   - `modal_run.py` - the remote execution wrapper. Treat this as infrastructure, not a research surface.
   - `experiment_tools.py` - checksum, results, and duplicate-preflight helpers.
   - `run_exact_commit.py` - exact-commit Modal launcher.
   - `program.md` - the experiment protocol.
4. **Verify Modal is ready**: make sure local Modal auth is set up. If needed, tell the human to run `uv run modal setup`.
5. **Initialize `results.tsv`**: run `uv run python experiment_tools.py ensure-results`. This creates or migrates the TSV to the checksum-aware schema.

```text
commit	train_py_sha256	val_accuracy	val_loss	status	description
```

6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Coordinator loop

Each experiment runs on a single device. The training script runs for a **fixed time budget of 60 seconds** of wall clock training time. The coordinator does not run dirty worktrees on Modal. Every run goes through:

```bash
uv run python run_exact_commit.py --commit HEAD > run.log 2>&1
```

The coordinator owns all of this:

- Keep the main workspace on `codex/autoresearch/<tag>`.
- Treat `HEAD` as the current best commit `B`.
- Read prior checksums from `results.tsv`.
- Generate distinct worker tasks, defaulting to 3 per round.
- Create one worktree per worker from `B`.
- Spawn subagents and give each one:
  - its worktree path
  - the base commit `B`
  - one explicit experiment idea
  - the worker contract below
- Collect preflight payloads, approve only unique checksums, then collect final run payloads.
- Append rows to `results.tsv`.
- Select the winner and fast-forward the branch if it improved on `B`.
- Remove all worker worktrees before the next round.

The search must stay constrained to **CNN-like models** and closely related training decisions. Convolutions, depth/width changes, residual connections, pooling choices, activations, normalization, classifier heads, and similar ideas are all in scope. Only `train.py` is the research surface.

## Worker contract

Each worker subagent must:

- Work only inside its assigned worktree.
- Edit only `train.py`.
- Execute only the coordinator-assigned idea for that round.
- Keep any revision within the same idea if the initial implementation is invalid.
- Commit before checksum preflight.
- Compute `train_py_sha256` from the committed candidate with `uv run python experiment_tools.py checksum --commit HEAD --json`.
- Stop after preflight and report back. Do not run Modal until the coordinator approves the checksum.
- Once approved, run `uv run python run_exact_commit.py --commit HEAD > run.log 2>&1`.
- If the run crashes, retry only after a code change clearly intended to fix the crash.
- Make at most 2 fix-and-rerun attempts after the initial crash.
- Never write `results.tsv`.
- Never merge, rebase, or advance the main experiment branch.

Required worker preflight payload:

```text
worker_id
assigned_idea
base_commit
candidate_commit
train_py_sha256
description
```

Required worker final payload:

```text
worker_id
assigned_idea
base_commit
candidate_commit
train_py_sha256
run_status
val_accuracy
val_loss
description
run_summary
```

`run_status` must be `success` or `crash`. `run_summary` must be concise. On crash it must state exactly why the crash happened and what was attempted to fix it.

## Preflight stage

Every round is two-stage. In preflight:

1. The worker edits `train.py`.
2. The worker commits the candidate.
3. The worker computes the committed checksum.
4. The worker reports the preflight payload without running Modal.
5. The coordinator checks the checksum against:
   - all prior `train_py_sha256` values in `results.tsv`
   - all already-approved checksums in the current round
6. The coordinator approves only unique checksums.
7. If a worker proposes a duplicate checksum, the worker must revise within the same assigned idea and resubmit preflight.

Use this command to gate a candidate:

```bash
uv run python experiment_tools.py preflight --commit HEAD --reserved <sha256>
```

If two commits produce the same `train.py` checksum, they are treated as the same experiment code and must not both be run.

## Run stage

After the coordinator approves a worker checksum, the worker may run:

```bash
uv run python run_exact_commit.py --commit HEAD > run.log 2>&1
```

Once the script finishes it prints a summary like this:

```text
---
model_name:       linear
device:           cuda
val_accuracy:     0.923400
val_loss:         0.271100
training_seconds: 60.0
total_seconds:    61.4
num_steps:        812
num_params_k:     7.9
```

Extract metrics with:

```bash
grep "^val_accuracy:\|^val_loss:" run.log
```

If the grep output is empty, inspect the traceback with `tail -n 50 run.log`.

## Logging results

The coordinator is the only writer to `results.tsv`. Use:

```bash
uv run python experiment_tools.py append-result \
  --commit <sha> \
  --val-accuracy <acc> \
  --val-loss <loss> \
  --status <keep|discard|crash> \
  --description "<summary>"
```

Rules:

- Log every successful run with its commit and `train_py_sha256`.
- Log every final crash with its commit and `train_py_sha256`.
- Do not log blocked duplicates, because no experiment was run.
- Assign TSV status centrally:
  - `keep` only for the candidate that becomes the new best
  - `discard` for successful candidates that do not win
  - `crash` for candidates that never produce a successful run

## Ranking and constraints

The goal is simple: get the highest `val_accuracy`. If two runs tie at the printed precision, the lower `val_loss` wins. If they are still effectively tied, prefer the simpler implementation.

All else being equal, simpler is better. A tiny gain that adds ugly complexity is usually not worth it. Conversely, deleting code and getting equal or better results is a great outcome.

You may not:

- Modify `prepare.py`.
- Modify `modal_run.py` unless the user explicitly asks for infrastructure changes.
- Install new packages or add dependencies.
- Modify the validation split or evaluation harness.
- Optimize against the MNIST test set during the loop.

The held-out test set is only for the final chosen model:

```bash
uv run python run_exact_commit.py --commit HEAD --final-test > run.log 2>&1
```
