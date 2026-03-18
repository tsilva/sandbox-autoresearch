"""
Coordinator-side helpers for exact-commit experiment runs and checksum tracking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable

TRAIN_FILE = "train.py"
RESULTS_FILE = "results.tsv"
LEGACY_RESULTS_HEADER = ["commit", "val_accuracy", "val_loss", "status", "description"]
RESULTS_HEADER = ["commit", "train_py_sha256", "val_accuracy", "val_loss", "status", "description"]
VALID_RESULTS_STATUSES = {"keep", "discard", "crash"}


class ExperimentToolError(RuntimeError):
    """Raised when helper commands cannot complete."""


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=capture_output,
        text=text,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if isinstance(result.stderr, str) else (result.stderr or b"").decode().strip()
        raise ExperimentToolError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{stderr}".rstrip()
        )
    return result


def _sanitize_field(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").strip()


def _results_path(repo_dir: Path, results_path: str | Path) -> Path:
    path = Path(results_path)
    if not path.is_absolute():
        path = repo_dir / path
    return path


def resolve_commit(repo_dir: str | Path, commit: str) -> str:
    repo_path = Path(repo_dir)
    result = _run_command(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def short_commit(repo_dir: str | Path, commit: str) -> str:
    repo_path = Path(repo_dir)
    result = _run_command(
        ["git", "rev-parse", "--short=7", commit],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def committed_file_bytes(repo_dir: str | Path, commit: str, file_path: str = TRAIN_FILE) -> bytes:
    repo_path = Path(repo_dir)
    resolved = resolve_commit(repo_path, commit)
    result = _run_command(
        ["git", "show", f"{resolved}:{file_path}"],
        cwd=repo_path,
        capture_output=True,
        text=False,
    )
    stdout = result.stdout
    if not isinstance(stdout, bytes):
        raise ExperimentToolError("Expected byte output from git show.")
    return stdout


def train_py_sha256(repo_dir: str | Path, commit: str) -> str:
    return hashlib.sha256(committed_file_bytes(repo_dir, commit, TRAIN_FILE)).hexdigest()


def export_commit_tree(repo_dir: str | Path, commit: str, destination: str | Path) -> str:
    repo_path = Path(repo_dir)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    resolved = resolve_commit(repo_path, commit)
    archive = _run_command(
        ["git", "archive", "--format=tar", resolved],
        cwd=repo_path,
        capture_output=True,
        text=False,
    )
    stdout = archive.stdout
    if not isinstance(stdout, bytes):
        raise ExperimentToolError("Expected byte output from git archive.")
    with tarfile.open(fileobj=io.BytesIO(stdout), mode="r:") as tar:
        try:
            tar.extractall(path=destination_path, filter="data")
        except TypeError:
            tar.extractall(path=destination_path)
    return resolved


def ensure_results_tsv(repo_dir: str | Path, results_path: str | Path = RESULTS_FILE) -> str:
    repo_path = Path(repo_dir)
    results_file = _results_path(repo_path, results_path)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    if not results_file.exists() or results_file.stat().st_size == 0:
        with results_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(RESULTS_HEADER)
        return "created"

    with results_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = [row for row in reader if row]

    header = rows[0]
    body = rows[1:]

    if header == RESULTS_HEADER:
        return "unchanged"

    if header != LEGACY_RESULTS_HEADER:
        raise ExperimentToolError(
            f"Unsupported {results_file.name} header: expected {RESULTS_HEADER} or {LEGACY_RESULTS_HEADER}, got {header}"
        )

    migrated_rows: list[list[str]] = []
    for row in body:
        record = dict(zip(LEGACY_RESULTS_HEADER, row, strict=False))
        commit = record["commit"]
        migrated_rows.append(
            [
                commit,
                train_py_sha256(repo_path, commit),
                record["val_accuracy"],
                record["val_loss"],
                record["status"],
                record["description"],
            ]
        )

    with results_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(RESULTS_HEADER)
        writer.writerows(migrated_rows)
    return "migrated"


def load_results_rows(repo_dir: str | Path, results_path: str | Path = RESULTS_FILE) -> list[dict[str, str]]:
    repo_path = Path(repo_dir)
    ensure_results_tsv(repo_path, results_path)
    results_file = _results_path(repo_path, results_path)
    with results_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def load_recorded_checksums(repo_dir: str | Path, results_path: str | Path = RESULTS_FILE) -> set[str]:
    return {row["train_py_sha256"] for row in load_results_rows(repo_dir, results_path)}


def preflight_candidate(
    repo_dir: str | Path,
    commit: str,
    *,
    results_path: str | Path = RESULTS_FILE,
    reserved_checksums: Iterable[str] = (),
) -> dict[str, str | bool | None]:
    repo_path = Path(repo_dir)
    resolved = resolve_commit(repo_path, commit)
    checksum = train_py_sha256(repo_path, resolved)
    recorded = load_recorded_checksums(repo_path, results_path)
    reserved = set(reserved_checksums)

    duplicate_source: str | None = None
    if checksum in reserved:
        duplicate_source = "reserved"
    elif checksum in recorded:
        duplicate_source = results_path if isinstance(results_path, str) else str(results_path)

    return {
        "commit": resolved,
        "short_commit": short_commit(repo_path, resolved),
        "train_py_sha256": checksum,
        "duplicate": duplicate_source is not None,
        "duplicate_source": duplicate_source,
    }


def append_result(
    repo_dir: str | Path,
    *,
    commit: str,
    val_accuracy: float,
    val_loss: float,
    status: str,
    description: str,
    results_path: str | Path = RESULTS_FILE,
) -> dict[str, str]:
    repo_path = Path(repo_dir)
    if status not in VALID_RESULTS_STATUSES:
        raise ExperimentToolError(f"Invalid status: {status}")

    ensure_results_tsv(repo_path, results_path)
    resolved = resolve_commit(repo_path, commit)
    checksum = train_py_sha256(repo_path, resolved)
    rows = load_results_rows(repo_path, results_path)
    if checksum in {row["train_py_sha256"] for row in rows}:
        raise ExperimentToolError(f"Checksum {checksum} already exists in {_results_path(repo_path, results_path)}")

    results_file = _results_path(repo_path, results_path)
    row = {
        "commit": short_commit(repo_path, resolved),
        "train_py_sha256": checksum,
        "val_accuracy": f"{val_accuracy:.6f}",
        "val_loss": f"{val_loss:.6f}",
        "status": status,
        "description": _sanitize_field(description),
    }
    with results_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=RESULTS_HEADER)
        writer.writerow(row)
    return row


def _print_json(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helpers for checksum-aware autoresearch coordination.")
    parser.add_argument(
        "--repo-dir",
        default=".",
        help="Git checkout or worktree to operate on. Defaults to the current directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure_parser = subparsers.add_parser("ensure-results", help="Create or migrate results.tsv to the checksum schema.")
    ensure_parser.add_argument("--results", default=RESULTS_FILE, help="Path to results TSV relative to --repo-dir.")

    checksum_parser = subparsers.add_parser("checksum", help="Compute the committed SHA-256 for train.py at a commit.")
    checksum_parser.add_argument("--commit", required=True, help="Commit-ish that contains train.py.")
    checksum_parser.add_argument("--json", action="store_true", help="Print structured JSON.")

    preflight_parser = subparsers.add_parser("preflight", help="Check whether a candidate checksum is unique.")
    preflight_parser.add_argument("--commit", required=True, help="Candidate commit to validate.")
    preflight_parser.add_argument("--results", default=RESULTS_FILE, help="Path to results TSV relative to --repo-dir.")
    preflight_parser.add_argument(
        "--reserved",
        action="append",
        default=[],
        help="Checksum already reserved by another worker in the current round. May be repeated.",
    )

    append_parser = subparsers.add_parser("append-result", help="Append one coordinator-selected result row.")
    append_parser.add_argument("--commit", required=True, help="Candidate commit to log.")
    append_parser.add_argument("--val-accuracy", required=True, type=float, help="Validation accuracy.")
    append_parser.add_argument("--val-loss", required=True, type=float, help="Validation loss.")
    append_parser.add_argument("--status", required=True, choices=sorted(VALID_RESULTS_STATUSES), help="Coordinator-assigned result status.")
    append_parser.add_argument("--description", required=True, help="Short experiment description.")
    append_parser.add_argument("--results", default=RESULTS_FILE, help="Path to results TSV relative to --repo-dir.")
    append_parser.add_argument("--json", action="store_true", help="Print structured JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_dir = Path(args.repo_dir)

    try:
        if args.command == "ensure-results":
            status = ensure_results_tsv(repo_dir, args.results)
            print(status)
            return 0

        if args.command == "checksum":
            resolved = resolve_commit(repo_dir, args.commit)
            checksum = train_py_sha256(repo_dir, resolved)
            if args.json:
                _print_json(
                    {
                        "commit": resolved,
                        "short_commit": short_commit(repo_dir, resolved),
                        "train_py_sha256": checksum,
                    }
                )
            else:
                print(checksum)
            return 0

        if args.command == "preflight":
            _print_json(
                preflight_candidate(
                    repo_dir,
                    args.commit,
                    results_path=args.results,
                    reserved_checksums=args.reserved,
                )
            )
            return 0

        if args.command == "append-result":
            row = append_result(
                repo_dir,
                commit=args.commit,
                val_accuracy=args.val_accuracy,
                val_loss=args.val_loss,
                status=args.status,
                description=args.description,
                results_path=args.results,
            )
            if args.json:
                _print_json(row)
            else:
                print("\t".join(row[field] for field in RESULTS_HEADER))
            return 0
    except ExperimentToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
