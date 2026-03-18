# mnist-autoresearch

This is an experiment to have the LLM do its own research on MNIST, constrained to CNN-like image models.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date. The branch `codex/autoresearch/<tag>` must not already exist. This is a fresh run.
2. **Create the branch**: `git checkout -b codex/autoresearch/<tag>` from the current default branch.
3. **Read the in-scope files**: the repo is small. Read these files for full context:
   - `README.md` - repository context.
   - `prepare.py` - fixed constants, data prep, data split, dataloaders, and evaluation.
   - `train.py` - the file you modify.
   - `modal_run.py` - the remote execution wrapper. Treat this as infrastructure, not a research surface.
   - `program.md` - the experiment protocol.
4. **Verify Modal is ready**: make sure local Modal auth is set up. If needed, tell the human to run `uv run modal setup`.
5. **Initialize `results.tsv`**: create `results.tsv` with just the header row. The baseline will be recorded after the first run.

```text
commit	val_accuracy	val_loss	status	description
```

6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single device. The training script runs for a **fixed time budget of 60 seconds** of wall clock training time. You launch it simply as:

```bash
uv run modal run modal_run.py > run.log 2>&1
```

**What you CAN do:**

- Modify `train.py` - this is the only file you edit. Everything inside it is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, regularization, normalization, pooling, scheduler, model size, and so on.
- The search should stay constrained to **CNN-like models** and closely related training decisions. Convolutions, depth/width changes, residual connections, pooling choices, activations, normalization, classifier heads, and similar ideas are all in scope.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It contains the fixed data preparation, train/validation split, dataloading, and evaluation harness.
- Modify `modal_run.py` unless the user explicitly asks for infrastructure changes.
- Install new packages or add dependencies. You can only use what is already in `pyproject.toml`.
- Modify the validation split or evaluation harness.
- Optimize against the MNIST test set during the loop. The held-out test set is only for the final chosen model via `uv run modal run modal_run.py --final-test > run.log 2>&1`.

**The goal is simple: get the highest `val_accuracy`.** If two runs tie at the printed precision, the lower `val_loss` wins. If they are still effectively tied, prefer the simpler implementation. Since the time budget is fixed, you do not need to worry much about training time beyond making sure the code actually runs and finishes within budget.

**Simplicity criterion**: all else being equal, simpler is better. A tiny gain that adds ugly complexity is usually not worth it. Conversely, deleting code and getting equal or better results is a great outcome. When deciding whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: your very first run should always be to establish the baseline, so run the training script as is before trying CNN ideas.

## Output format

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

You can extract the key metrics from the log file:

```bash
grep "^val_accuracy:\|^val_loss:" run.log
```

`run.log` is local. `modal_run.py` runs the training job remotely on Modal but forwards the remote stdout and stderr back to the local process, so redirecting `> run.log 2>&1` still captures the full run output for parsing and crash inspection. Modal requests one small GPU for each run, preferring `T4` and falling back to `L4`.

## Logging results

When an experiment is done, log it to `results.tsv` as tab-separated values:

```text
commit	val_accuracy	val_loss	status	description
```

1. git commit hash, short form, 7 chars
2. `val_accuracy` achieved (for crashes use `0.000000`)
3. `val_loss` achieved (for crashes use `0.000000`)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```text
commit	val_accuracy	val_loss	status	description
a1b2c3d	0.923400	0.271100	keep	baseline linear classifier
b2c3d4e	0.987200	0.040800	keep	add small 2-layer cnn with max pooling
c3d4e5f	0.986800	0.042500	discard	raise dropout to 0.5
d4e5f6g	0.000000	0.000000	crash	make cnn too wide and hit an error
```

## The experiment loop

The experiment runs on a dedicated branch such as `codex/autoresearch/mar18`. LOOP FOREVER:

1. Look at the git state: the current branch and commit you are on.
2. Tune `train.py` with one experimental idea by directly hacking the code.
3. `git commit`
4. Run the experiment: `uv run modal run modal_run.py > run.log 2>&1`
5. Read out the results: `grep "^val_accuracy:\|^val_loss:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you cannot get it to work after a few attempts, give up on that idea.
7. Record the results in the TSV. Do not commit `results.tsv`; leave it untracked by git.
8. If `val_accuracy` improved, or if accuracy tied and `val_loss` improved, you "advance" the branch, keeping the git commit.
9. If the run is worse, or tied without being simpler, reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep them. If they do not, discard them. Advance the branch only when the best known result improves.

**Timeout**: each experiment should take about 60 seconds of training time plus a small amount of startup and evaluation overhead. If a run gets stuck far beyond that, kill it and treat it as a failure.

**Crashes**: if a run crashes due to OOM, a bug, or anything else, use judgment. If it is something dumb and easy to fix, fix it and re-run. If the idea itself is fundamentally broken, log `crash` in the TSV and move on.

**NEVER STOP**: once the experiment loop has begun, do not pause to ask the human whether to continue. Do not ask if this is a good stopping point. Continue until the human manually interrupts you.

If you select a final best model and need a held-out report, run:

```bash
uv run modal run modal_run.py --final-test > run.log 2>&1
```

That command is only for the final chosen model, not for day-to-day optimization.
