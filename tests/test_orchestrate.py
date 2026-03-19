from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess, run

from experiment_tools import train_py_sha256
from orchestrate import approve_candidates, finalize_round, resume_round, start_round


class OrchestrateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.temp_dir.name)
        run(["git", "init"], cwd=self.repo_dir, check=True, capture_output=True)
        run(["git", "branch", "-m", "main"], cwd=self.repo_dir, check=True, capture_output=True)
        run(["git", "config", "user.name", "Test User"], cwd=self.repo_dir, check=True, capture_output=True)
        run(["git", "config", "user.email", "test@example.com"], cwd=self.repo_dir, check=True, capture_output=True)
        (self.repo_dir / "pyproject.toml").write_text("[project]\nname='tmp'\nversion='0.0.0'\n", encoding="utf-8")
        (self.repo_dir / "modal_run.py").write_text("print('placeholder')\n", encoding="utf-8")
        (self.repo_dir / "run_exact_commit.py").write_text("print('placeholder')\n", encoding="utf-8")
        (self.repo_dir / "train.py").write_text("print('baseline')\n", encoding="utf-8")
        run(
            ["git", "add", "train.py", "pyproject.toml", "modal_run.py", "run_exact_commit.py"],
            cwd=self.repo_dir,
            check=True,
            capture_output=True,
        )
        run(["git", "commit", "-m", "baseline"], cwd=self.repo_dir, check=True, capture_output=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _commit_worker_train(self, worktree_path: Path, contents: str, message: str) -> str:
        (worktree_path / "train.py").write_text(contents, encoding="utf-8")
        run(["git", "add", "train.py"], cwd=worktree_path, check=True, capture_output=True)
        run(["git", "commit", "-m", message], cwd=worktree_path, check=True, capture_output=True)
        return run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=worktree_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _write_preflight(self, artifact, worker_index: int, commit: str, description: str) -> None:
        worker = artifact.worker_records[worker_index]
        preflight_path = Path(worker.task.preflight_path)
        payload = {
            "worker_id": worker.task.worker_id,
            "assigned_idea": worker.task.assigned_idea,
            "base_commit": worker.task.base_commit,
            "candidate_commit": commit,
            "train_py_sha256": train_py_sha256(worker.task.worktree_path, "HEAD"),
            "description": description,
        }
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_start_round_writes_tasks_and_resume_state(self) -> None:
        artifact = start_round(self.repo_dir, ideas=["wider cnn", "residual block"], round_id="round-a")

        self.assertEqual(artifact.status, "awaiting_preflight")
        self.assertEqual(len(artifact.worker_records), 2)
        self.assertTrue(Path(artifact.worker_records[0].task.prompt_path).exists())

        resumed = resume_round(self.repo_dir, "round-a")

        self.assertEqual(resumed.round_id, "round-a")
        self.assertEqual(resumed.to_state().current_best_commit, artifact.base_commit)

    def test_approve_candidates_rejects_duplicate_checksums(self) -> None:
        artifact = start_round(self.repo_dir, ideas=["idea a", "idea b"], round_id="round-b")
        commit_a = self._commit_worker_train(Path(artifact.worker_records[0].task.worktree_path), "print('alpha')\n", "alpha")
        commit_b = self._commit_worker_train(Path(artifact.worker_records[1].task.worktree_path), "print('alpha')\n", "beta")
        self._write_preflight(artifact, 0, commit_a, "alpha")
        self._write_preflight(artifact, 1, commit_b, "beta")

        approved = approve_candidates(self.repo_dir, "round-b")

        self.assertEqual(approved.worker_records[0].approval_status, "approved")
        self.assertEqual(approved.worker_records[1].approval_status, "rejected")
        self.assertEqual(approved.worker_records[1].approval_reason, "duplicate checksum")

    def test_finalize_round_logs_once_and_advances_best_commit(self) -> None:
        artifact = start_round(self.repo_dir, ideas=["idea a"], round_id="round-c")
        commit_a = self._commit_worker_train(Path(artifact.worker_records[0].task.worktree_path), "print('alpha')\n", "alpha")
        self._write_preflight(artifact, 0, commit_a, "alpha")
        approve_candidates(self.repo_dir, "round-c")

        def fake_runner(_: Path, __: str) -> CompletedProcess[str]:
            stdout = "\n".join(
                [
                    "---",
                    "model_name:       cnn",
                    "device:           cuda",
                    "val_accuracy:     0.990100",
                    "val_loss:         0.041000",
                    "training_seconds: 60.0",
                    "total_seconds:    61.0",
                    "num_steps:        800",
                    "num_params_k:     12.0",
                ]
            )
            return CompletedProcess(args=["fake"], returncode=0, stdout=stdout, stderr="")

        finalized = finalize_round(self.repo_dir, "round-c", runner=fake_runner)
        finalized_again = finalize_round(self.repo_dir, "round-c", runner=fake_runner)

        self.assertEqual(finalized.status, "finalized")
        self.assertEqual(finalized.winner_commit, commit_a)
        self.assertEqual(finalized_again.winner_commit, commit_a)

        with (self.repo_dir / "results.tsv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "keep")

        head_commit = run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=self.repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(head_commit, commit_a)

    def test_finalize_round_records_crash(self) -> None:
        artifact = start_round(self.repo_dir, ideas=["idea a"], round_id="round-d")
        commit_a = self._commit_worker_train(Path(artifact.worker_records[0].task.worktree_path), "print('alpha')\n", "alpha")
        self._write_preflight(artifact, 0, commit_a, "alpha")
        approve_candidates(self.repo_dir, "round-d")

        def fake_runner(_: Path, __: str) -> CompletedProcess[str]:
            return CompletedProcess(args=["fake"], returncode=1, stdout="", stderr="RuntimeError: boom\n")

        finalized = finalize_round(self.repo_dir, "round-d", runner=fake_runner)

        self.assertEqual(finalized.worker_records[0].run.run_status, "crash")
        with (self.repo_dir / "results.tsv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(rows[0]["status"], "crash")


if __name__ == "__main__":
    unittest.main()
