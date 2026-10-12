import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from forecasting import exploration_runner as runner


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.original_cwd = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(runner, "ROOT", self.root)
        self.root_patch.start()
        (self.root / "forecasting").mkdir()
        (self.root / "forecasting" / "models.py").write_text("version=1\n")
        (self.root / "config.yaml").write_text("training: {}\n")
        self.job = {"id": "model/2026/NSW1", "kind": "baseline", "baseline_kind": "xgboost",
                    "config": "config.yaml", "region": "NSW1", "seed": 2026,
                    "output": "results/NSW1", "dependencies": []}

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.root_patch.stop()
        self.temp.cleanup()

    def artifacts(self, output, finite=True):
        output.mkdir(parents=True, exist_ok=True)
        for split in ("validation", "test"):
            np.savez(output / f"{split}.npz", point=np.ones((2, 3)),
                     quantile=np.full((2, 3, 5), 1. if finite else np.nan),
                     actual=np.ones((2, 3)), origin_unix=[100, 200],
                     quantile_levels=[.05, .1, .5, .9, .95], seed=2026, region="NSW1")
        (output / "metrics.json").write_text("{}")

    def test_completion_requires_matching_source_and_untampered_artifacts(self):
        output = self.root / self.job["output"]
        self.artifacts(output)
        fingerprint = runner.source_hash()
        runner.atomic_json(output / "run_metadata.json", {
            "state": "completed", "identity": runner.job_identity(self.job, fingerprint),
            "artifact_hashes": {name: runner.file_hash(output / name)
                                for name in ("validation.npz", "test.npz", "metrics.json")}})
        self.assertTrue(runner.completed(self.job, fingerprint))
        self.assertFalse(runner.completed(self.job, "changed-source"))
        (output / "test.npz").write_bytes(b"corrupt")
        self.assertFalse(runner.completed(self.job, fingerprint))

    def test_nonfinite_predictions_fail(self):
        output = self.root / self.job["output"]
        self.artifacts(output, finite=False)
        with self.assertRaises(FloatingPointError):
            runner.validate_predictions(output, self.job)

    def test_event_controls_need_probabilities_not_fabricated_price_forecasts(self):
        output = self.root / self.job["output"]
        output.mkdir(parents=True)
        job = dict(self.job, baseline_kind="event_controls")
        for split in ("validation", "test"):
            np.savez(output / f"{split}.npz", actual=np.ones((2, 3)),
                     origin_unix=[100, 200], quantile_levels=[.05, .1, .5, .9, .95],
                     seed=2026, region="NSW1", **{f"{model}_prob_{event}": np.full((2, 3), .1)
                         for model in ("logistic", "persistence") for event in ("spike", "trough")})
        runner.validate_predictions(output, job)

    def test_worker_does_not_overwrite_unowned_output(self):
        output = self.root / self.job["output"]
        self.artifacts(output)
        before = runner.file_hash(output / "test.npz")
        with self.assertRaisesRegex(RuntimeError, "unowned"):
            runner.run_worker(self.job, runner.source_hash())
        self.assertEqual(runner.file_hash(output / "test.npz"), before)

    def test_manifest_rejects_dependency_cycle_and_duplicate_output(self):
        second = dict(self.job, id="second", output="results/other", dependencies=[self.job["id"]])
        first = dict(self.job, dependencies=["second"])
        with self.assertRaisesRegex(ValueError, "cycle"):
            runner.validate_manifest({"jobs": [first, second]})
        second["output"] = self.job["output"]
        with self.assertRaisesRegex(ValueError, "Duplicate job output"):
            runner.validate_manifest({"jobs": [self.job, second]})

    def test_source_change_is_detected_and_paths_cannot_escape(self):
        before = runner.source_hash()
        (self.root / "forecasting" / "models.py").write_text("version=2\n")
        self.assertNotEqual(before, runner.source_hash())
        with self.assertRaises(ValueError):
            runner.project_path("../outside")

    def test_declared_configuration_and_data_hashes_are_enforced(self):
        manifest = {"jobs": [self.job], "config_hashes": {
            "config.yaml": runner.file_hash(self.root / "config.yaml")},
            "data_hashes": {"config.yaml": "incorrect"}}
        runner.verify_manifest_hashes(manifest, include_data=False)
        with self.assertRaisesRegex(ValueError, "data hash mismatch"):
            runner.verify_manifest_hashes(manifest)
        (self.root / "config.yaml").write_text("changed")
        with self.assertRaisesRegex(ValueError, "configuration hash mismatch"):
            runner.verify_manifest_hashes(manifest, include_data=False)

    def test_scheduler_failed_dependency_is_blocked_not_executed(self):
        dependent = dict(self.job, id="dependent", output="results/dependent", dependencies=[self.job["id"]])
        manifest = {"jobs": [self.job, dependent]}
        path = self.root / "manifest.json"
        runner.atomic_json(path, manifest)
        class FailedProcess:
            pid = 12345
            def poll(self):
                return 1
        with patch("subprocess.Popen", return_value=FailedProcess()) as popen, patch.object(runner.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "failed or blocked"):
                runner.run_scheduler(path, manifest, ["6"], 1)
        self.assertEqual(popen.call_count, 1)
        status = json.loads(path.with_suffix(".status.json").read_text())
        self.assertEqual(status["jobs"][self.job["id"]]["state"], "failed")
        self.assertEqual(status["jobs"]["dependent"]["state"], "blocked")
        self.assertEqual(status["state"], "failed")


if __name__ == "__main__":
    unittest.main()
