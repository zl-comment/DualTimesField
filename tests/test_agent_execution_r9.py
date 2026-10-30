"""RED-first execution regressions: synthetic calls/files, no model or network.

The legacy entry identity is replaced only inside these narrow unit tests: the
fixtures exercise runtime effects, not the separate r5 branch identity gate.
The shared helper is imported inside tests so its absence is a per-test RED.
"""
from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from forecasting.agent_forecasting import m1_contract, m1_publication, m1_runtime
from forecasting.agent_forecasting.contracts import FrozenForecastInput

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/fixtures/agent_forecasting"))
import m1_scenario_harness as harness


def execution():
    try:
        return importlib.import_module("forecasting.agent_forecasting.execution")
    except ImportError as error:
        raise AssertionError("shared r9 execution implementation is missing") from error


def run_base(*, tool=None, publish_root=None, clock=None):
    snapshot = harness.build_snapshot(listing="EMPTY", new_outcome="COMPLETE_VALID",
                                      relation="BEFORE_ORIGIN",
                                      body_relation="DISTINCT_VINTAGES")
    base = FrozenForecastInput(**harness.synthetic_base_input_payload())
    with mock.patch.object(m1_contract, "verify_run_entry",
                           return_value={"contract_id": m1_contract.CONTRACT_ID}):
        return m1_runtime.run_arm("A", snapshot, base_input=base, tool=tool,
                                  publish_root=publish_root, clock=clock)


class RealDeadlineTests(unittest.TestCase):
    def test_shared_timeout_interrupts_a_real_blocking_call(self):
        module = execution()
        started = time.monotonic()
        with self.assertRaises(module.DeadlineExceeded):
            module.call_with_deadline(lambda: time.sleep(0.12), 0.025)
        self.assertLess(time.monotonic() - started, 0.095)

    def test_exhausted_budget_refuses_before_callback(self):
        module = execution()
        called = []
        with self.assertRaises(module.DeadlineExceeded):
            module.call_with_deadline(lambda: called.append(True), 0)
        self.assertEqual(called, [])

    def test_inner_timeout_never_extends_outer_deadline(self):
        module = execution()
        started = time.monotonic()
        with self.assertRaises(module.DeadlineExceeded):
            module.call_with_deadline(
                lambda: module.call_with_deadline(lambda: time.sleep(0.12), 0.1),
                0.025)
        self.assertLess(time.monotonic() - started, 0.095)

    def test_existing_alarm_and_handler_are_preserved(self):
        module = execution()
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        handler = lambda signum, frame: None
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, 1.0)
            self.assertEqual(module.call_with_deadline(lambda: 17, 0.1), 17)
            self.assertIs(signal.getsignal(signal.SIGALRM), handler)
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            self.assertLessEqual(signal.getitimer(signal.ITIMER_REAL)[0], 1.0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    def test_legacy_runtime_default_clock_times_out_blocking_forecast(self):
        class BlockingTool(m1_runtime.DeterministicForecastStub):
            called = False

            def predict(self, model_input, params=None):
                self.called = True
                time.sleep(0.12)
                return super().predict(model_input, params)

        tool = BlockingTool()
        started = time.monotonic()
        with mock.patch.object(m1_runtime, "DEADLINE_OFFSET", 0.025), \
                mock.patch.object(m1_runtime, "PUBLICATION_RESERVE", 0.0):
            result = run_base(tool=tool)
        self.assertLess(time.monotonic() - started, 0.095,
                        "default M1 runtime must interrupt, not check a logical clock afterwards")
        self.assertEqual(result.verdict, "SYSTEM_MISSING")
        self.assertTrue(tool.called, "the timeout must interrupt an entered effect")
        self.assertFalse(result.published)
        self.assertEqual(result.forecast_bytes, b"")


class SharedDirectoryCommitTests(unittest.TestCase):
    def _staging(self, root):
        staging = root / "staging"
        staging.mkdir()
        (staging / "payload.json").write_text('{"synthetic":true}')
        (staging / "COMMIT.json").write_text('{"run_id":"synthetic"}')
        return staging

    def test_commit_is_present_at_the_single_visible_rename(self):
        module = execution()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging, final = self._staging(root), root / "final"
            rename = os.rename
            calls = []
            def checked_rename(source, destination):
                self.assertTrue((Path(source) / "COMMIT.json").is_file())
                self.assertFalse(Path(destination).exists())
                calls.append((source, destination))
                return rename(source, destination)
            with mock.patch.object(module.os, "rename", side_effect=checked_rename):
                committed = module.publish_directory(staging, final)
            self.assertEqual(committed.path, final)
            self.assertTrue(committed.durable)
            self.assertEqual(len(calls), 1)
            self.assertTrue((final / "COMMIT.json").is_file())

    def test_deadline_expiry_refuses_promotion(self):
        module = execution()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging, final = self._staging(root), root / "final"
            with self.assertRaises(module.DeadlineExceeded):
                module.publish_directory(staging, final, deadline=10, clock=lambda: 11)
            self.assertFalse(final.exists())

    def test_post_commit_fsync_error_reports_committed_not_failed(self):
        module = execution()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging, final = self._staging(root), root / "final"
            fsync = os.fsync
            def fail_after_commit(fd):
                if final.exists():
                    raise OSError("synthetic post-rename durability failure")
                return fsync(fd)
            with mock.patch.object(module.os, "fsync", side_effect=fail_after_commit):
                committed = module.publish_directory(staging, final)
            self.assertEqual(committed.path, final)
            self.assertFalse(committed.durable)
            self.assertTrue((final / "COMMIT.json").is_file())


class LegacyPublicationTests(unittest.TestCase):
    def test_legacy_commit_marker_is_staged_before_directory_visibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rename = os.rename
            promotions = []
            def checked_rename(source, destination):
                if Path(destination).parent == root / "runs":
                    self.assertTrue((Path(source) / "COMMIT.json").is_file(),
                                    "legacy publisher exposes an incomplete final directory")
                    promotions.append(str(destination))
                return rename(source, destination)
            with mock.patch.object(m1_publication.os, "rename", side_effect=checked_rename):
                result = run_base(publish_root=root)
            self.assertTrue(result.published)
            self.assertEqual(len(promotions), 1)
            self.assertEqual(m1_publication.verify_run_closure(result.publish_dir)["status"], "CLOSED")

    def test_legacy_post_commit_error_cannot_return_failed_with_valid_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fsync = os.fsync
            def fail_after_commit(fd):
                if list((root / "runs").glob("*/COMMIT.json")):
                    raise OSError("synthetic failure after committed visibility")
                return fsync(fd)
            with mock.patch.object(m1_publication.os, "fsync", side_effect=fail_after_commit):
                result = run_base(publish_root=root)
            self.assertTrue(result.published,
                            "a valid visible commit must not also return FAILED_NO_COMMIT")
            self.assertNotEqual(result.verdict, "TERMINAL_FAILED_NO_COMMIT")
            self.assertEqual(m1_publication.verify_run_closure(result.publish_dir)["status"], "CLOSED")
            self.assertEqual(list((root / "failures").glob("*.json")), [])

    def test_legacy_rechecks_clock_after_staging_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = run_base(publish_root=root / "honest_source")
            self.assertTrue(source.published)
            archived = Path(source.publish_dir)
            manifest = json.loads((archived / "manifest.json").read_text())
            payloads = {name: (archived / name).read_bytes() for name in manifest["payloads"]}
            closure = manifest["closure"]
            now = [closure["origin_unix"]]
            write = m1_publication._write_exclusive
            def expire_while_staging(path, value):
                write(path, value)
                if Path(path).name == "manifest.json":
                    now[0] = closure["deadline_unix"] + 1
            destination = root / "destination"
            with mock.patch.object(m1_publication, "_write_exclusive", side_effect=expire_while_staging):
                with self.assertRaises(TimeoutError):
                    m1_publication.publish_run(destination, "late_run", payloads,
                        identity=manifest["identity"], closure=closure, clock=lambda: now[0])
            self.assertFalse((destination / "runs/late_run").exists())


if __name__ == "__main__":
    unittest.main()
