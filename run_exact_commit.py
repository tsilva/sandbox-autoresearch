"""
Run Modal from an archived export of one exact git commit.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from experiment_tools import ExperimentToolError, export_commit_tree


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modal_run.py from an archived git commit snapshot.")
    parser.add_argument("--commit", required=True, help="Commit-ish to export and run.")
    parser.add_argument(
        "--repo-dir",
        default=".",
        help="Git checkout or worktree to export from. Defaults to the current directory.",
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Forward --final-test to modal_run.py after validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_dir = Path(args.repo_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix="mnist-autoresearch-"))

    try:
        resolved = export_commit_tree(repo_dir, args.commit, temp_dir)
        print(f"resolved_commit: {resolved}", flush=True)
        command = ["uv", "run", "modal", "run", "modal_run.py"]
        if args.final_test:
            command.append("--final-test")
        result = subprocess.run(command, cwd=temp_dir, check=False)
        return result.returncode
    except ExperimentToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
