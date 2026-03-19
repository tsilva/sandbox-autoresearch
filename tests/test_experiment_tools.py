from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from subprocess import run

from experiment_tools import (
    ExperimentToolError,
    append_result,
    create_worktree,
    parse_run_output,
    ensure_results_tsv,
    export_commit_tree,
    preflight_candidate,
    remove_worktree,
    resolve_commit,
    select_best_candidate,
    train_py_sha256,
)


class ExperimentToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.temp_dir.name)
        run(["git", "init"], cwd=self.repo_dir, check=True, capture_output=True)
        run(["git", "config", "user.name", "Test User"], cwd=self.repo_dir, check=True, capture_output=True)
        run(["git", "config", "user.email", "test@example.com"], cwd=self.repo_dir, check=True, capture_output=True)
        (self.repo_dir / "pyproject.toml").write_text("[project]\nname='tmp'\nversion='0.0.0'\n", encoding="utf-8")
        (self.repo_dir / "modal_run.py").write_text("print('placeholder')\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _commit_train(self, contents: str, message: str) -> str:
        (self.repo_dir / "train.py").write_text(contents, encoding="utf-8")
        run(["git", "add", "train.py", "pyproject.toml", "modal_run.py"], cwd=self.repo_dir, check=True, capture_output=True)
        run(["git", "commit", "-m", message], cwd=self.repo_dir, check=True, capture_output=True)
        return resolve_commit(self.repo_dir, "HEAD")

    def test_train_py_sha256_uses_committed_blob(self) -> None:
        commit = self._commit_train("print('baseline')\n", "baseline")
        (self.repo_dir / "train.py").write_text("print('dirty')\n", encoding="utf-8")

        checksum = train_py_sha256(self.repo_dir, commit)

        self.assertEqual(checksum, hashlib.sha256(b"print('baseline')\n").hexdigest())
        self.assertNotEqual(checksum, hashlib.sha256(b"print('dirty')\n").hexdigest())

    def test_export_commit_tree_excludes_dirty_state_and_untracked_files(self) -> None:
        commit = self._commit_train("print('baseline')\n", "baseline")
        (self.repo_dir / "train.py").write_text("print('dirty')\n", encoding="utf-8")
        (self.repo_dir / "scratch.txt").write_text("do not ship\n", encoding="utf-8")

        export_dir = self.repo_dir / "exported"
        export_commit_tree(self.repo_dir, commit, export_dir)

        self.assertEqual((export_dir / "train.py").read_text(encoding="utf-8"), "print('baseline')\n")
        self.assertFalse((export_dir / "scratch.txt").exists())

    def test_ensure_results_tsv_migrates_legacy_schema(self) -> None:
        commit = self._commit_train("print('baseline')\n", "baseline")
        results_path = self.repo_dir / "results.tsv"
        results_path.write_text(
            "commit\tval_accuracy\tval_loss\tstatus\tdescription\n"
            f"{commit[:7]}\t0.900000\t0.100000\tkeep\tbaseline\n",
            encoding="utf-8",
        )

        status = ensure_results_tsv(self.repo_dir, results_path)

        self.assertEqual(status, "migrated")
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(rows[0]["commit"], commit[:7])
        self.assertEqual(rows[0]["train_py_sha256"], hashlib.sha256(b"print('baseline')\n").hexdigest())

    def test_preflight_detects_recorded_and_reserved_duplicates(self) -> None:
        commit_a = self._commit_train("print('alpha')\n", "alpha")
        commit_b = self._commit_train("print('beta')\n", "beta")
        results_path = self.repo_dir / "results.tsv"
        append_result(
            self.repo_dir,
            commit=commit_a,
            val_accuracy=0.9,
            val_loss=0.1,
            status="keep",
            description="alpha",
            results_path=results_path,
        )

        duplicate_from_results = preflight_candidate(self.repo_dir, commit_a, results_path=results_path)
        duplicate_from_reserved = preflight_candidate(
            self.repo_dir,
            commit_b,
            results_path=results_path,
            reserved_checksums=[train_py_sha256(self.repo_dir, commit_b)],
        )

        self.assertTrue(duplicate_from_results["duplicate"])
        self.assertEqual(duplicate_from_results["duplicate_source"], str(results_path))
        self.assertTrue(duplicate_from_reserved["duplicate"])
        self.assertEqual(duplicate_from_reserved["duplicate_source"], "reserved")

    def test_append_result_rejects_duplicate_checksum(self) -> None:
        commit = self._commit_train("print('baseline')\n", "baseline")
        results_path = self.repo_dir / "results.tsv"
        append_result(
            self.repo_dir,
            commit=commit,
            val_accuracy=0.9,
            val_loss=0.1,
            status="keep",
            description="baseline",
            results_path=results_path,
        )

        with self.assertRaises(ExperimentToolError):
            append_result(
                self.repo_dir,
                commit=commit,
                val_accuracy=0.9,
                val_loss=0.1,
                status="discard",
                description="duplicate",
                results_path=results_path,
            )

    def test_parse_run_output_extracts_metrics(self) -> None:
        result = parse_run_output(
            "\n".join(
                [
                    "---",
                    "model_name:       cnn",
                    "device:           cuda",
                    "val_accuracy:     0.991200",
                    "val_loss:         0.031400",
                    "training_seconds: 60.0",
                    "total_seconds:    61.2",
                    "num_steps:        900",
                    "num_params_k:     42.5",
                ]
            )
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.metrics)
        assert result.metrics is not None
        self.assertEqual(result.metrics.model_name, "cnn")
        self.assertAlmostEqual(result.metrics.val_accuracy, 0.9912)
        self.assertAlmostEqual(result.metrics.num_params_k, 42.5)

    def test_parse_run_output_marks_missing_metrics_as_crash(self) -> None:
        result = parse_run_output("resolved_commit: deadbeef\n", "Traceback: boom\n", returncode=0)

        self.assertFalse(result.success)
        self.assertEqual(result.crash_reason, "Traceback: boom")

    def test_select_best_candidate_uses_accuracy_then_loss_then_simplicity(self) -> None:
        best = select_best_candidate(
            [
                {
                    "train_py_sha256": "b",
                    "val_accuracy": 0.99,
                    "val_loss": 0.04,
                    "num_params_k": 60.0,
                    "train_py_bytes": 300,
                },
                {
                    "train_py_sha256": "a",
                    "val_accuracy": 0.99,
                    "val_loss": 0.04,
                    "num_params_k": 40.0,
                    "train_py_bytes": 500,
                },
            ]
        )

        self.assertEqual(best["train_py_sha256"], "a")

    def test_create_and_remove_worktree_round_trip(self) -> None:
        commit = self._commit_train("print('baseline')\n", "baseline")
        worktree_path = self.repo_dir / "wt"

        create_worktree(self.repo_dir, worktree_path, commit)

        self.assertTrue((worktree_path / "train.py").exists())

        removed = remove_worktree(self.repo_dir, worktree_path)

        self.assertTrue(removed)
        self.assertFalse(worktree_path.exists())


if __name__ == "__main__":
    unittest.main()
