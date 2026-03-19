"""
Deterministic experiment runtime that manages round state around agent-driven train.py edits.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from experiment_tools import (
    RESULTS_FILE,
    TRAIN_FILE,
    ExperimentToolError,
    append_result,
    candidate_beats_baseline,
    changed_files_between,
    committed_file_size,
    create_worktree,
    current_branch_name,
    fast_forward_branch,
    find_result_row_by_commit,
    is_ancestor,
    load_recorded_checksums,
    load_results_rows,
    parse_run_output,
    preflight_candidate,
    remove_worktree,
    repo_root,
    resolve_commit,
    select_best_candidate,
)

SCHEMA_VERSION = 1
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_MAX_RETRIES = 2


@dataclass
class WorkerTask:
    worker_id: str
    assigned_idea: str
    worktree_path: str
    base_commit: str
    task_path: str
    prompt_path: str
    preflight_path: str
    run_path: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkerTask":
        return cls(**payload)


@dataclass
class PreflightPayload:
    worker_id: str
    assigned_idea: str
    base_commit: str
    candidate_commit: str
    train_py_sha256: str
    description: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreflightPayload":
        return cls(**payload)


@dataclass
class RunPayload:
    worker_id: str
    assigned_idea: str
    base_commit: str
    candidate_commit: str
    train_py_sha256: str
    run_status: str
    val_accuracy: float | None
    val_loss: float | None
    description: str
    run_summary: str
    attempt: int = 1
    model_name: str | None = None
    device: str | None = None
    num_params_k: float | None = None
    training_seconds: float | None = None
    total_seconds: float | None = None
    num_steps: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunPayload":
        return cls(**payload)


@dataclass
class CandidateValidationResult:
    valid: bool
    approval_status: str
    reason: str | None = None
    duplicate_source: str | None = None
    candidate_commit: str | None = None
    train_py_sha256: str | None = None
    changed_files: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateValidationResult":
        return cls(**payload)


@dataclass
class WorkerRecord:
    task: WorkerTask
    approval_status: str = "pending"
    approval_reason: str | None = None
    candidate_commit: str | None = None
    train_py_sha256: str | None = None
    description: str | None = None
    duplicate_source: str | None = None
    validation: CandidateValidationResult | None = None
    run: RunPayload | None = None
    results_logged: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkerRecord":
        validation = payload.get("validation")
        run_payload = payload.get("run")
        return cls(
            task=WorkerTask.from_dict(payload["task"]),
            approval_status=payload.get("approval_status", "pending"),
            approval_reason=payload.get("approval_reason"),
            candidate_commit=payload.get("candidate_commit"),
            train_py_sha256=payload.get("train_py_sha256"),
            description=payload.get("description"),
            duplicate_source=payload.get("duplicate_source"),
            validation=CandidateValidationResult.from_dict(validation) if validation else None,
            run=RunPayload.from_dict(run_payload) if run_payload else None,
            results_logged=payload.get("results_logged", False),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class RoundState:
    round_id: str
    status: str
    repo_dir: str
    branch: str | None
    base_commit: str
    current_best_commit: str
    current_best_result: dict[str, str] | None
    checksums_seen: list[str]
    reserved_checksums: list[str]
    workers: list[dict[str, Any]]
    history: list[dict[str, str]]
    winner_commit: str | None
    winner_worker_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoundArtifact:
    schema_version: int
    round_id: str
    repo_dir: str
    branch: str | None
    base_commit: str
    current_best_commit: str
    current_best_result: dict[str, str] | None
    results_path: str
    artifacts_dir: str
    worktree_root: str
    status: str
    max_retries: int
    created_at: str
    worker_records: list[WorkerRecord]
    winner_commit: str | None = None
    winner_worker_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoundArtifact":
        return cls(
            schema_version=payload["schema_version"],
            round_id=payload["round_id"],
            repo_dir=payload["repo_dir"],
            branch=payload.get("branch"),
            base_commit=payload["base_commit"],
            current_best_commit=payload["current_best_commit"],
            current_best_result=payload.get("current_best_result"),
            results_path=payload["results_path"],
            artifacts_dir=payload["artifacts_dir"],
            worktree_root=payload["worktree_root"],
            status=payload["status"],
            max_retries=payload["max_retries"],
            created_at=payload["created_at"],
            worker_records=[WorkerRecord.from_dict(item) for item in payload["worker_records"]],
            winner_commit=payload.get("winner_commit"),
            winner_worker_id=payload.get("winner_worker_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def to_state(self) -> RoundState:
        repo_dir = Path(self.repo_dir)
        history = load_results_rows(repo_dir, self.results_path)
        checksums_seen = sorted(load_recorded_checksums(repo_dir, self.results_path))
        reserved = sorted(
            worker.train_py_sha256
            for worker in self.worker_records
            if worker.approval_status == "approved" and worker.train_py_sha256 is not None
        )
        workers = [worker.to_dict() for worker in self.worker_records]
        return RoundState(
            round_id=self.round_id,
            status=self.status,
            repo_dir=self.repo_dir,
            branch=self.branch,
            base_commit=self.base_commit,
            current_best_commit=self.current_best_commit,
            current_best_result=self.current_best_result,
            checksums_seen=checksums_seen,
            reserved_checksums=reserved,
            workers=workers,
            history=history,
            winner_commit=self.winner_commit,
            winner_worker_id=self.winner_worker_id,
        )


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _artifact_root(repo_dir: Path, artifacts_dir: str | Path) -> Path:
    path = Path(artifacts_dir)
    if not path.is_absolute():
        path = repo_dir / path
    return path


def _default_worktree_root(repo_dir: Path) -> Path:
    return repo_dir.parent / f".{repo_dir.name}-worktrees"


def _round_dir(repo_dir: Path, artifacts_dir: str | Path, round_id: str) -> Path:
    return _artifact_root(repo_dir, artifacts_dir) / round_id


def _artifact_file(repo_dir: Path, artifacts_dir: str | Path, round_id: str) -> Path:
    return _round_dir(repo_dir, artifacts_dir, round_id) / "round.json"


def _load_ideas(ideas: Iterable[str] | None = None, ideas_file: str | Path | None = None) -> list[str]:
    if ideas is not None:
        values = [idea.strip() for idea in ideas if idea.strip()]
        if values:
            return values

    if ideas_file is None:
        raise ExperimentToolError("No experiment ideas supplied.")

    path = Path(ideas_file)
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(raw)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ExperimentToolError(f"Ideas file must contain a JSON string list: {path}")
        values = [item.strip() for item in payload if item.strip()]
    else:
        values = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]

    if not values:
        raise ExperimentToolError("No non-empty experiment ideas found.")
    return values


def _render_worker_prompt(task: WorkerTask) -> str:
    return (
        "Implement exactly one experiment idea inside train.py.\n\n"
        f"worker_id: {task.worker_id}\n"
        f"assigned_idea: {task.assigned_idea}\n"
        f"base_commit: {task.base_commit}\n"
        f"worktree_path: {task.worktree_path}\n\n"
        "Hard constraints:\n"
        "- Work only inside the assigned worktree.\n"
        f"- Edit only {TRAIN_FILE}.\n"
        "- Commit before preflight.\n"
        "- Stop after writing the preflight payload.\n"
        "- Do not run Modal until approval is recorded.\n\n"
        "Write the preflight payload as JSON to:\n"
        f"{task.preflight_path}\n\n"
        "The payload schema is:\n"
        "{\n"
        '  "worker_id": "...",\n'
        '  "assigned_idea": "...",\n'
        '  "base_commit": "...",\n'
        '  "candidate_commit": "...",\n'
        '  "train_py_sha256": "...",\n'
        '  "description": "..."\n'
        "}\n"
    )


def _save_artifact(artifact: RoundArtifact) -> None:
    artifact_path = _artifact_file(Path(artifact.repo_dir), artifact.artifacts_dir, artifact.round_id)
    _write_json(artifact_path, artifact.to_dict())


def load_round_artifact(
    repo_dir: str | Path,
    round_id: str,
    *,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> RoundArtifact:
    repo_path = repo_root(repo_dir)
    path = _artifact_file(repo_path, artifacts_dir, round_id)
    if not path.exists():
        raise ExperimentToolError(f"Round artifact not found: {path}")
    return RoundArtifact.from_dict(_read_json(path))


def start_round(
    repo_dir: str | Path,
    *,
    ideas: Iterable[str] | None = None,
    ideas_file: str | Path | None = None,
    round_id: str | None = None,
    base_commit: str = "HEAD",
    results_path: str | Path = RESULTS_FILE,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    worktree_root: str | Path | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> RoundArtifact:
    repo_path = repo_root(repo_dir)
    resolved_base = resolve_commit(repo_path, base_commit)
    branch = current_branch_name(repo_path)
    current_best = find_result_row_by_commit(repo_path, resolved_base, results_path)
    ideas_list = _load_ideas(ideas=ideas, ideas_file=ideas_file)

    if round_id is None:
        round_id = _timestamp_token()

    round_directory = _round_dir(repo_path, artifacts_dir, round_id)
    if round_directory.exists():
        raise ExperimentToolError(f"Round already exists: {round_directory}")

    resolved_worktree_root = Path(worktree_root) if worktree_root else _default_worktree_root(repo_path)
    worker_records: list[WorkerRecord] = []

    try:
        for index, idea in enumerate(ideas_list, start=1):
            worker_id = f"worker-{index:02d}"
            worktree_path = resolved_worktree_root / round_id / worker_id
            create_worktree(repo_path, worktree_path, resolved_base)

            task_dir = round_directory / "tasks"
            prompt_dir = round_directory / "prompts"
            preflight_dir = round_directory / "preflight"
            run_dir = round_directory / "runs"
            task = WorkerTask(
                worker_id=worker_id,
                assigned_idea=idea,
                worktree_path=str(worktree_path),
                base_commit=resolved_base,
                task_path=str(task_dir / f"{worker_id}.json"),
                prompt_path=str(prompt_dir / f"{worker_id}.md"),
                preflight_path=str(preflight_dir / f"{worker_id}.json"),
                run_path=str(run_dir / f"{worker_id}.json"),
            )
            worker = WorkerRecord(task=task)
            worker_records.append(worker)
            _write_json(Path(task.task_path), asdict(task))
            Path(task.prompt_path).parent.mkdir(parents=True, exist_ok=True)
            Path(task.prompt_path).write_text(_render_worker_prompt(task), encoding="utf-8")
    except Exception:
        for worker in worker_records:
            remove_worktree(repo_path, worker.task.worktree_path)
        raise

    artifact = RoundArtifact(
        schema_version=SCHEMA_VERSION,
        round_id=round_id,
        repo_dir=str(repo_path),
        branch=branch,
        base_commit=resolved_base,
        current_best_commit=resolved_base,
        current_best_result=current_best,
        results_path=str(results_path),
        artifacts_dir=str(artifacts_dir),
        worktree_root=str(resolved_worktree_root),
        status="awaiting_preflight",
        max_retries=max_retries,
        created_at=datetime.now(timezone.utc).isoformat(),
        worker_records=worker_records,
    )
    _save_artifact(artifact)
    return artifact


def validate_preflight_payload(
    repo_dir: str | Path,
    task: WorkerTask,
    payload: PreflightPayload,
    *,
    results_path: str | Path = RESULTS_FILE,
    reserved_checksums: Iterable[str] = (),
) -> CandidateValidationResult:
    repo_path = repo_root(repo_dir)

    if payload.worker_id != task.worker_id:
        return CandidateValidationResult(False, "rejected", reason="worker_id mismatch")
    if payload.assigned_idea != task.assigned_idea:
        return CandidateValidationResult(False, "rejected", reason="assigned_idea mismatch")
    if resolve_commit(repo_path, payload.base_commit) != resolve_commit(repo_path, task.base_commit):
        return CandidateValidationResult(False, "rejected", reason="base_commit mismatch")
    if not payload.description.strip():
        return CandidateValidationResult(False, "rejected", reason="description must be non-empty")

    resolved_candidate = resolve_commit(repo_path, payload.candidate_commit)
    if not is_ancestor(repo_path, task.base_commit, resolved_candidate):
        return CandidateValidationResult(False, "rejected", reason="candidate_commit does not descend from base_commit")

    changed_files = changed_files_between(repo_path, task.base_commit, resolved_candidate)
    if changed_files != [TRAIN_FILE]:
        return CandidateValidationResult(
            False,
            "rejected",
            reason=f"candidate must only change {TRAIN_FILE}",
            changed_files=changed_files,
        )

    preflight = preflight_candidate(
        repo_path,
        resolved_candidate,
        results_path=results_path,
        reserved_checksums=reserved_checksums,
    )
    checksum = str(preflight["train_py_sha256"])
    if payload.train_py_sha256 != checksum:
        return CandidateValidationResult(False, "rejected", reason="train_py_sha256 mismatch", changed_files=changed_files)
    if bool(preflight["duplicate"]):
        return CandidateValidationResult(
            False,
            "rejected",
            reason="duplicate checksum",
            duplicate_source=str(preflight["duplicate_source"]),
            candidate_commit=resolved_candidate,
            train_py_sha256=checksum,
            changed_files=changed_files,
        )

    return CandidateValidationResult(
        True,
        "approved",
        candidate_commit=resolved_candidate,
        train_py_sha256=checksum,
        changed_files=changed_files,
    )


def approve_candidates(
    repo_dir: str | Path,
    round_id: str,
    *,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> RoundArtifact:
    artifact = load_round_artifact(repo_dir, round_id, artifacts_dir=artifacts_dir)
    repo_path = Path(artifact.repo_dir)

    for worker in artifact.worker_records:
        preflight_path = Path(worker.task.preflight_path)
        if not preflight_path.exists():
            continue

        payload = PreflightPayload.from_dict(_read_json(preflight_path))
        current_reserved = {
            other.train_py_sha256
            for other in artifact.worker_records
            if other.task.worker_id != worker.task.worker_id
            and other.approval_status == "approved"
            and other.train_py_sha256 is not None
        }
        validation = validate_preflight_payload(
            repo_path,
            worker.task,
            payload,
            results_path=artifact.results_path,
            reserved_checksums=current_reserved,
        )
        worker.validation = validation
        worker.approval_status = validation.approval_status
        worker.approval_reason = validation.reason
        worker.duplicate_source = validation.duplicate_source
        worker.candidate_commit = validation.candidate_commit
        worker.train_py_sha256 = validation.train_py_sha256
        worker.description = payload.description.strip()
        if not validation.valid:
            worker.run = None

    artifact.status = (
        "awaiting_runs"
        if any(worker.approval_status == "approved" for worker in artifact.worker_records)
        else "awaiting_preflight"
    )
    _save_artifact(artifact)
    return artifact


def _default_runner(repo_dir: Path, commit: str) -> subprocess.CompletedProcess[str]:
    command = [
        "uv",
        "run",
        "python",
        "run_exact_commit.py",
        "--repo-dir",
        str(repo_dir),
        "--commit",
        commit,
    ]
    return subprocess.run(command, cwd=repo_dir, capture_output=True, text=True, check=False)


def _successful_candidate_map(repo_dir: Path, worker: WorkerRecord) -> dict[str, Any]:
    if worker.run is None or worker.run.val_accuracy is None or worker.run.val_loss is None or worker.train_py_sha256 is None:
        raise ExperimentToolError(f"Worker {worker.task.worker_id} is missing a successful run.")
    return {
        "worker_id": worker.task.worker_id,
        "commit": worker.candidate_commit,
        "train_py_sha256": worker.train_py_sha256,
        "val_accuracy": worker.run.val_accuracy,
        "val_loss": worker.run.val_loss,
        "num_params_k": worker.run.num_params_k,
        "train_py_bytes": committed_file_size(repo_dir, worker.candidate_commit or ""),
    }


def _baseline_candidate_map(artifact: RoundArtifact) -> dict[str, Any] | None:
    if artifact.current_best_result is None:
        return None
    row = artifact.current_best_result
    return {
        "commit": artifact.current_best_commit,
        "train_py_sha256": row["train_py_sha256"],
        "val_accuracy": float(row["val_accuracy"]),
        "val_loss": float(row["val_loss"]),
        "num_params_k": None,
        "train_py_bytes": None,
    }


def finalize_round(
    repo_dir: str | Path,
    round_id: str,
    *,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    cleanup_worktrees: bool = True,
    runner: Callable[[Path, str], subprocess.CompletedProcess[str]] | None = None,
) -> RoundArtifact:
    artifact = load_round_artifact(repo_dir, round_id, artifacts_dir=artifacts_dir)
    repo_path = Path(artifact.repo_dir)
    run_candidate = _default_runner if runner is None else runner

    for worker in artifact.worker_records:
        if worker.approval_status != "approved" or worker.candidate_commit is None or worker.train_py_sha256 is None:
            continue
        if worker.run is not None:
            continue

        completed = run_candidate(repo_path, worker.candidate_commit)
        parsed = parse_run_output(completed.stdout, completed.stderr, completed.returncode)
        run_payload = RunPayload(
            worker_id=worker.task.worker_id,
            assigned_idea=worker.task.assigned_idea,
            base_commit=worker.task.base_commit,
            candidate_commit=worker.candidate_commit,
            train_py_sha256=worker.train_py_sha256,
            run_status="success" if parsed.success else "crash",
            val_accuracy=parsed.metrics.val_accuracy if parsed.metrics else None,
            val_loss=parsed.metrics.val_loss if parsed.metrics else None,
            description=worker.description or worker.task.assigned_idea,
            run_summary=parsed.summary,
            attempt=1,
            model_name=parsed.metrics.model_name if parsed.metrics else None,
            device=parsed.metrics.device if parsed.metrics else None,
            num_params_k=parsed.metrics.num_params_k if parsed.metrics else None,
            training_seconds=parsed.metrics.training_seconds if parsed.metrics else None,
            total_seconds=parsed.metrics.total_seconds if parsed.metrics else None,
            num_steps=parsed.metrics.num_steps if parsed.metrics else None,
        )
        worker.run = run_payload
        _write_json(Path(worker.task.run_path), asdict(run_payload))

    successful_workers = [
        worker for worker in artifact.worker_records if worker.run is not None and worker.run.run_status == "success"
    ]
    baseline = _baseline_candidate_map(artifact)
    winner: WorkerRecord | None = None
    if successful_workers:
        best_candidate = select_best_candidate(_successful_candidate_map(repo_path, worker) for worker in successful_workers)
        winner = next(worker for worker in successful_workers if worker.task.worker_id == best_candidate["worker_id"])
        if not candidate_beats_baseline(best_candidate, baseline):
            winner = None

    if winner is not None:
        artifact.winner_worker_id = winner.task.worker_id
        artifact.winner_commit = winner.candidate_commit
    else:
        artifact.winner_worker_id = None
        artifact.winner_commit = None

    existing_checksums = load_recorded_checksums(repo_path, artifact.results_path)
    for worker in artifact.worker_records:
        if worker.run is None or worker.results_logged or worker.train_py_sha256 is None or worker.candidate_commit is None:
            continue
        if worker.train_py_sha256 in existing_checksums:
            worker.results_logged = True
            continue

        if worker.run.run_status == "success":
            status = "keep" if winner is not None and worker.task.worker_id == winner.task.worker_id else "discard"
            val_accuracy = float(worker.run.val_accuracy) if worker.run.val_accuracy is not None else math.nan
            val_loss = float(worker.run.val_loss) if worker.run.val_loss is not None else math.nan
        else:
            status = "crash"
            val_accuracy = math.nan
            val_loss = math.nan

        append_result(
            repo_path,
            commit=worker.candidate_commit,
            val_accuracy=val_accuracy,
            val_loss=val_loss,
            status=status,
            description=worker.description or worker.task.assigned_idea,
            results_path=artifact.results_path,
        )
        existing_checksums.add(worker.train_py_sha256)
        worker.results_logged = True

    if winner is not None and artifact.branch is not None and winner.candidate_commit is not None:
        fast_forward_branch(repo_path, artifact.branch, winner.candidate_commit)
        artifact.current_best_commit = winner.candidate_commit
        artifact.current_best_result = find_result_row_by_commit(repo_path, winner.candidate_commit, artifact.results_path)

    if cleanup_worktrees:
        for worker in artifact.worker_records:
            remove_worktree(repo_path, worker.task.worktree_path)

    artifact.status = "finalized"
    _save_artifact(artifact)
    return artifact


def resume_round(
    repo_dir: str | Path,
    round_id: str,
    *,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> RoundArtifact:
    return load_round_artifact(repo_dir, round_id, artifacts_dir=artifacts_dir)


def _print_state(artifact: RoundArtifact) -> None:
    print(json.dumps(artifact.to_state().to_dict(), indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic experiment round runtime.")
    parser.add_argument("--repo-dir", default=".", help="Repository root or worktree. Defaults to current directory.")
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR, help="Artifact directory relative to repo root.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start-round", help="Create worktrees and task artifacts for a new round.")
    start_parser.add_argument("--ideas-file", required=True, help="Text or JSON file containing experiment ideas.")
    start_parser.add_argument("--round-id", help="Optional explicit round identifier.")
    start_parser.add_argument("--base-commit", default="HEAD", help="Base commit for all worker worktrees.")
    start_parser.add_argument("--results", default=RESULTS_FILE, help="Results TSV path relative to repo root.")
    start_parser.add_argument("--worktree-root", help="Optional absolute or relative root for worker worktrees.")
    start_parser.add_argument("--max-retries", default=DEFAULT_MAX_RETRIES, type=int, help="Maximum allowed crash retries.")

    approve_parser = subparsers.add_parser("approve-candidates", help="Validate all submitted preflight payloads.")
    approve_parser.add_argument("--round-id", required=True, help="Round identifier.")

    finalize_parser = subparsers.add_parser("finalize-round", help="Run approved candidates and record results.")
    finalize_parser.add_argument("--round-id", required=True, help="Round identifier.")
    finalize_parser.add_argument(
        "--keep-worktrees",
        action="store_true",
        help="Leave worker worktrees on disk after finalization.",
    )

    resume_parser = subparsers.add_parser("resume", help="Load an existing round artifact and print the current state.")
    resume_parser.add_argument("--round-id", required=True, help="Round identifier.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "start-round":
            artifact = start_round(
                args.repo_dir,
                ideas_file=args.ideas_file,
                round_id=args.round_id,
                base_commit=args.base_commit,
                results_path=args.results,
                artifacts_dir=args.artifacts_dir,
                worktree_root=args.worktree_root,
                max_retries=args.max_retries,
            )
            _print_state(artifact)
            return 0

        if args.command == "approve-candidates":
            artifact = approve_candidates(args.repo_dir, args.round_id, artifacts_dir=args.artifacts_dir)
            _print_state(artifact)
            return 0

        if args.command == "finalize-round":
            artifact = finalize_round(
                args.repo_dir,
                args.round_id,
                artifacts_dir=args.artifacts_dir,
                cleanup_worktrees=not args.keep_worktrees,
            )
            _print_state(artifact)
            return 0

        if args.command == "resume":
            artifact = resume_round(args.repo_dir, args.round_id, artifacts_dir=args.artifacts_dir)
            _print_state(artifact)
            return 0
    except ExperimentToolError as exc:
        print(str(exc))
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
