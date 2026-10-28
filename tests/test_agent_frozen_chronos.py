"""Boundary tests with synthetic inputs only; importing this file loads no model."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from forecasting.agent_forecasting.contracts import FrozenForecastInput
from forecasting.agent_forecasting.frozen_chronos import (
    INPUT_FILES, load_development_input, predict_frozen, sha256_file,
    validate_output, validate_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agent_contract_runner", ROOT / "scripts/check_agent_forecast_contract.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FrozenToolBoundaryTests(unittest.TestCase):
    def _cache(self, directory):
        directory.mkdir()
        (directory / "metadata.json").write_text(json.dumps({"target_mean": -20, "target_std": 2}))
        np.save(directory / "origin_unix.npy", [360000])
        history = np.zeros((1, 72, 2), dtype=np.float32)
        history[:, :, 0] = -3
        np.save(directory / "input__history_values.npy", history)
        np.save(directory / "input__history_calendar.npy", np.zeros((1, 72, 10), dtype=np.float32))
        np.save(directory / "input__future_calendar.npy", np.zeros((1, 24, 10), dtype=np.float32))
        # Poison label files: any accidental np.load of these fails immediately.
        (directory / "actual.npy").write_bytes(b"forbidden label data")
        (directory / "target.npy").write_bytes(b"forbidden label data")
        return {"input": {"directory": str(directory), "row": 0, "origin_unix": 360000,
                          "region": "NSW1", "file_sha256": {n: sha256_file(directory / n) for n in INPUT_FILES}}}

    def test_allowlisted_loader_uses_inputs_and_decodes_negative_price_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cache(Path(tmp) / "cache")
            request, provenance = load_development_input(cfg)
            self.assertEqual(request.price_history, (-26.0,) * 72)
            self.assertEqual(request.target_time_unix, tuple(360000 + 3600 * h for h in range(24)))
            self.assertEqual(set(Path(p).name for p in provenance["files_read"]), set(INPUT_FILES))
            self.assertFalse(provenance["strict_live_asof_verified"])

    def test_identity_mismatch_and_label_allowlist_extension_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cache(Path(tmp) / "cache")
            cfg["input"]["file_sha256"]["metadata.json"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "identity"):
                load_development_input(cfg)
            cfg["input"]["file_sha256"]["actual.npy"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "allowlist"):
                load_development_input(cfg)

    def test_wrong_origin_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cache(Path(tmp) / "cache")
            cfg["input"]["origin_unix"] += 3600
            with self.assertRaisesRegex(ValueError, "origin"):
                load_development_input(cfg)

    def test_snapshot_hash_and_native_quantiles_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixed_revision"
            path.mkdir()
            (path / "model.safetensors").write_bytes(b"synthetic placeholder, never loaded")
            (path / "config.json").write_text(json.dumps({"chronos_config": {"quantiles": [0.05, 0.1, 0.5, 0.9, 0.95]}}))
            identity = {"directory": str(path), "revision": "fixed_revision",
                        "sha256": {n: sha256_file(path / n) for n in ("config.json", "model.safetensors")}}
            self.assertEqual(set(validate_snapshot(identity)), {"config.json", "model.safetensors"})
            (path / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_snapshot(identity)

    def test_output_rejects_nan_and_shape_and_preserves_negative_and_crossings(self):
        output = np.full((24, 5), -7.0)
        output[:, 0] = -1.0
        np.testing.assert_array_equal(validate_output(output), output)
        with self.assertRaises(ValueError):
            validate_output(np.zeros((24, 3)))
        output[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_output(output)

    def test_cpu_and_offline_guards_run_before_any_model_import(self):
        request = FrozenForecastInput(360000, tuple(360000 + 3600 * h for h in range(24)), (1.0,) * 72)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(RuntimeError, "CPU"):
                predict_frozen(request, {})
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "0"}):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                predict_frozen(request, {})

    def test_failed_or_interrupted_reservation_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "calls.jsonl"
            config = {"resources": {"max_model_calls": 3}}
            first = runner.reserve_smoke(ledger, config)
            self.assertEqual(first["attempt"], 1)
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                runner.reserve_smoke(ledger, config)
            self.assertEqual(len(ledger.read_text().splitlines()), 1)

    def test_output_artifacts_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            runner.write_json_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                runner.write_json_new(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_monitor_cleanup_terminates_worker_process_group(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            self.assertLess(runner.kill_group_and_wait(child), 0)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_private_worker_cannot_run_without_live_registered_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text("{}")
            with patch.dict(os.environ, {"G0_MONITOR_PID": "-1"}):
                with self.assertRaisesRegex(RuntimeError, "live budget monitor"):
                    runner.worker(config, Path(tmp))

    def test_partial_worker_report_cannot_prevent_parent_failure_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker_report.json"
            path.write_text('{"status":')
            payload, error = runner.read_worker_report(path)
            self.assertEqual(payload, {})
            self.assertEqual(error, "JSONDecodeError")


if __name__ == "__main__":
    unittest.main()
