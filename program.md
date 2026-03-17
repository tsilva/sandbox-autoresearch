# mnist-autoresearch

This repo adapts the `autoresearch` workflow to a compact MNIST classifier search.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date.
2. Create a fresh branch named `codex/autoresearch/<tag>`.
3. Read the in-scope files for full context:
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `program.md`
4. Verify the cache exists under `~/.cache/mnist-autoresearch`. If not, run `uv run prepare.py`.
5. Initialize `results.tsv` with this header row and leave it untracked by git:

```text
commit	val_accuracy	val_loss	status	description
```

6. Confirm setup looks good and begin experimentation.

## Experimentation

Each experiment runs with a fixed 60-second wall-clock training budget:

```bash
uv run train.py
```

**What you CAN do:**

- Modify `train.py`. This is the only file you edit during the research loop.
- Change the model architecture after the baseline, but keep the search focused on compact CNN experiments and closely related training decisions.
- Change optimizer, LR schedule, weight decay, dropout, normalization, batch size, pooling, convolution widths, depths, and kernel sizes.

**What you CANNOT do:**

- Modify `prepare.py`.
- Add packages or dependencies.
- Change the validation split or evaluation harness.
- Optimize against the MNIST test set during the loop.

**The goal is simple: get the highest `val_accuracy`.** If two runs tie at the printed precision, keep the one with lower `val_loss`. If both are effectively tied, prefer the simpler implementation.

**The first run:** Always run the baseline linear classifier before trying CNN changes.

## Output format

`train.py` prints a summary like this:

```text
---
model_name:       linear
device:           mps
val_accuracy:     0.923400
val_loss:         0.271100
training_seconds: 60.0
total_seconds:    61.4
num_steps:        812
num_params_k:     7.9
```

Extract the ranking metrics with:

```bash
grep "^val_accuracy:\|^val_loss:" run.log
```

## Logging results

Log each experiment to `results.tsv` as tab-separated values:

```text
commit	val_accuracy	val_loss	status	description
```

Example:

```text
commit	val_accuracy	val_loss	status	description
a1b2c3d	0.917400	0.287100	keep	baseline linear classifier
b2c3d4e	0.986800	0.042300	keep	add 2-layer cnn with max pooling
c3d4e5f	0.985600	0.047900	discard	raise dropout to 0.5
d4e5f6g	0.000000	0.000000	crash	make cnn too wide for this machine
```

## Experiment loop

Loop continuously:

1. Check the current branch and commit.
2. Modify `train.py` with one experimental idea.
3. Commit the change.
4. Run `uv run train.py > run.log 2>&1`.
5. Read `val_accuracy` and `val_loss` from `run.log`.
6. If the metrics are missing, inspect the crash with `tail -n 50 run.log`, fix simple mistakes, or mark the attempt as `crash`.
7. Append the run to `results.tsv`.
8. If the run improved, keep the commit and continue from there.
9. If the run did not improve, reset back to the prior kept commit.

If a selected best model needs a final held-out report, rerun it with:

```bash
uv run train.py --final-test
```

That command is only for the final chosen model, not for day-to-day optimization.
