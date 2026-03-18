from __future__ import annotations

import subprocess
import sys

import modal

APP_NAME = "mnist-autoresearch"
REMOTE_APP_DIR = "/root/app"
REMOTE_CACHE_DIR = "/root/.cache/mnist-autoresearch"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim()
    .uv_sync()
    .add_local_dir(
        ".",
        remote_path=REMOTE_APP_DIR,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            "*.pyc",
            "results.tsv",
            "run.log",
        ],
    )
)

cache_volume = modal.Volume.from_name(f"{APP_NAME}-cache", create_if_missing=True)


@app.function(
    image=image,
    volumes={REMOTE_CACHE_DIR: cache_volume},
    gpu=["T4", "L4"],
    timeout=15 * 60,
)
def run_train(final_test: bool = False) -> dict[str, str | int]:
    command = [sys.executable, "train.py"]
    if final_test:
        command.append("--final-test")

    result = subprocess.run(
        command,
        cwd=REMOTE_APP_DIR,
        capture_output=True,
        text=True,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def main(final_test: bool = False) -> None:
    result = run_train.remote(final_test=final_test)

    if result["stdout"]:
        sys.stdout.write(result["stdout"])
    if result["stderr"]:
        sys.stderr.write(result["stderr"])

    if result["returncode"] != 0:
        raise SystemExit(int(result["returncode"]))
