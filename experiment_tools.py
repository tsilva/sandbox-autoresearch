"""
Coordinator-side helpers for exact-commit experiment runs and checksum tracking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

TRAIN_FILE = "train.py"
RESULTS_FILE = "results.tsv"
LEGACY_RESULTS_HEADER = ["commit", "val_accuracy", "val_loss", "status", "description"]
RESULTS_HEADER = ["commit", "train_py_sha256", "val_accuracy", "val_loss", "status", "description"]
VALID_RESULTS_STATUSES = {"keep", "discard", "crash"}

_SUMMARY_LINE_RE = re.compile(r"^(?P<key>[a-z_]+):\s+(?P<value>.+?)\s*$")


class ExperimentToolError(RuntimeError):
    """Raised when helper commands cannot complete."""


@dataclass(frozen=True)
class RunMetrics:
    model_name: str | None
    device: str | None
    val_accuracy: float
    val_loss: float
    training_seconds: float | None = None
    total_seconds: float | None = None
    num_steps: int | None = None
    num_params_k: float | None = None
    test_accuracy: float | None = None
    test_loss: float | None = None


@dataclass(frozen=True)
class RunParseResult:
    success: bool
    metrics: RunMetrics | None
    crash_reason: str | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


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


def _git_text_lines(repo_dir: Path, *command: str) -> list[str]:
    result = _run_command(["git", *command], cwd=repo_dir, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _last_non_empty_line(*chunks: str) -> str | None:
    for chunk in chunks:
        for line in reversed(chunk.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def repo_root(repo_dir: str | Path) -> Path:
    repo_path = Path(repo_dir)
    result = _run_command(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


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


def current_branch_name(repo_dir: str | Path) -> str | None:
    repo_path = Path(repo_dir)
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


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


def committed_file_size(repo_dir: str | Path, commit: str, file_path: str = TRAIN_FILE) -> int:
    return len(committed_file_bytes(repo_dir, commit, file_path))


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


def find_result_row_by_commit(
    repo_dir: str | Path,
    commit: str,
    results_path: str | Path = RESULTS_FILE,
) -> dict[str, str] | None:
    short = short_commit(repo_dir, commit)
    for row in load_results_rows(repo_dir, results_path):
        if row["commit"] == short:
            return row
    return None


def is_ancestor(repo_dir: str | Path, ancestor: str, descendant: str) -> bool:
    repo_path = Path(repo_dir)
    resolved_ancestor = resolve_commit(repo_path, ancestor)
    resolved_descendant = resolve_commit(repo_path, descendant)
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved_ancestor, resolved_descendant],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    stderr = result.stderr.strip()
    raise ExperimentToolError(
        f"Command failed ({result.returncode}): git merge-base --is-ancestor {resolved_ancestor} {resolved_descendant}\n{stderr}".rstrip()
    )


def changed_files_between(repo_dir: str | Path, base_commit: str, candidate_commit: str) -> list[str]:
    repo_path = Path(repo_dir)
    resolved_base = resolve_commit(repo_path, base_commit)
    resolved_candidate = resolve_commit(repo_path, candidate_commit)
    return _git_text_lines(repo_path, "diff", "--name-only", resolved_base, resolved_candidate)


def is_worktree_clean(repo_dir: str | Path) -> bool:
    repo_path = Path(repo_dir)
    result = _run_command(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == ""


def list_worktrees(repo_dir: str | Path) -> list[Path]:
    repo_path = Path(repo_dir)
    paths: list[Path] = []
    current: Path | None = None
    for line in _git_text_lines(repo_path, "worktree", "list", "--porcelain"):
        if line.startswith("worktree "):
            current = Path(line.removeprefix("worktree ").strip())
            paths.append(current)
    return paths


def create_worktree(repo_dir: str | Path, path: str | Path, commit: str) -> Path:
    repo_path = Path(repo_dir)
    destination = Path(path)
    resolved = resolve_commit(repo_path, commit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and any(destination.iterdir()):
        raise ExperimentToolError(f"Refusing to create worktree in non-empty directory: {destination}")
    _run_command(
        ["git", "worktree", "add", "--detach", str(destination), resolved],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return destination


def remove_worktree(repo_dir: str | Path, path: str | Path) -> bool:
    repo_path = Path(repo_dir)
    destination = Path(path)
    registered = {registered_path.resolve() for registered_path in list_worktrees(repo_path)}
    resolved_destination = destination.resolve()

    if resolved_destination in registered:
        _run_command(
            ["git", "worktree", "remove", "--force", str(destination)],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        _run_command(["git", "worktree", "prune"], cwd=repo_path, capture_output=True, text=True)
        return True

    if destination.exists():
        shutil.rmtree(destination)
        return True
    return False


def fast_forward_branch(repo_dir: str | Path, branch: str, commit: str) -> None:
    repo_path = Path(repo_dir)
    target = resolve_commit(repo_path, commit)
    branch_tip = resolve_commit(repo_path, branch)
    if not is_ancestor(repo_path, branch_tip, target):
        raise ExperimentToolError(f"Cannot fast-forward {branch} to {target}: target does not descend from current tip.")

    current_branch = current_branch_name(repo_path)
    if current_branch == branch:
        if not is_worktree_clean(repo_path):
            raise ExperimentToolError(f"Cannot fast-forward checked-out branch {branch} with a dirty worktree.")
        _run_command(["git", "merge", "--ff-only", target], cwd=repo_path, capture_output=True, text=True)
        return

    _run_command(["git", "branch", "-f", branch, target], cwd=repo_path, capture_output=True, text=True)


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


def parse_run_output(stdout: str, stderr: str = "", returncode: int = 0) -> RunParseResult:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        match = _SUMMARY_LINE_RE.match(line.strip())
        if match:
            parsed[match.group("key")] = match.group("value").strip()

    if returncode != 0:
        reason = _last_non_empty_line(stderr, stdout) or f"process exited with code {returncode}"
        return RunParseResult(
            success=False,
            metrics=None,
            crash_reason=reason,
            summary=f"crash: {reason}",
        )

    if "val_accuracy" not in parsed or "val_loss" not in parsed:
        reason = _last_non_empty_line(stderr, stdout) or "missing val_accuracy/val_loss in run output"
        return RunParseResult(
            success=False,
            metrics=None,
            crash_reason=reason,
            summary=f"crash: {reason}",
        )

    try:
        metrics = RunMetrics(
            model_name=parsed.get("model_name"),
            device=parsed.get("device"),
            val_accuracy=float(parsed["val_accuracy"]),
            val_loss=float(parsed["val_loss"]),
            training_seconds=float(parsed["training_seconds"]) if "training_seconds" in parsed else None,
            total_seconds=float(parsed["total_seconds"]) if "total_seconds" in parsed else None,
            num_steps=int(parsed["num_steps"]) if "num_steps" in parsed else None,
            num_params_k=float(parsed["num_params_k"]) if "num_params_k" in parsed else None,
            test_accuracy=float(parsed["test_accuracy"]) if "test_accuracy" in parsed else None,
            test_loss=float(parsed["test_loss"]) if "test_loss" in parsed else None,
        )
    except ValueError as exc:
        raise ExperimentToolError(f"Failed to parse run output: {exc}") from exc

    return RunParseResult(
        success=True,
        metrics=metrics,
        crash_reason=None,
        summary=f"val_accuracy={metrics.val_accuracy:.6f} val_loss={metrics.val_loss:.6f}",
    )


def rank_candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    num_params_k = candidate.get("num_params_k")
    train_py_bytes = candidate.get("train_py_bytes")
    checksum = str(candidate.get("train_py_sha256", ""))
    return (
        -float(candidate["val_accuracy"]),
        float(candidate["val_loss"]),
        float(num_params_k) if num_params_k is not None else float("inf"),
        int(train_py_bytes) if train_py_bytes is not None else sys.maxsize,
        checksum,
    )


def select_best_candidate(candidates: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    candidate_list = list(candidates)
    if not candidate_list:
        raise ExperimentToolError("No candidates provided for ranking.")
    return min(candidate_list, key=rank_candidate_key)


def candidate_beats_baseline(candidate: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> bool:
    if baseline is None:
        return True
    return rank_candidate_key(candidate) < rank_candidate_key(baseline)


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
