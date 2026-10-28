"""G0-A frozen-harness tests: budget guard + hard wall deadline, label
isolation, units, coverage with value-level fill verification, coverage-blind
selection, schema, single-process HEAD reading, and source/data binding.
All value-level tests use synthetic arrays; the only real files touched are
the frozen NPZ/config hash binding and the repo .git directory (read-only).

Gate-reliability regression tests (2026-10-28 repair round 2) additionally
pin the audit runner's failure semantics: a fault in the final
source_hashes/HEAD/budget phase must leave no ENGINEERING_PASS evidence, must
exit non-zero with an atomic ENGINEERING_FAIL record, success evidence must be
published once via a staging directory + atomic rename after ALL final checks,
budget timing must cover process start through evidence finalization, and no
failure or rerun path may overwrite already-published evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from forecasting.agent_forecasting import g0a_predispatch as g0a

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/agent_forecasting/g0a_predispatch_rrp_20261028.json"


def load_audit_runner():
    spec = importlib.util.spec_from_file_location(
        "audit_g0a_predispatch_rrp_under_test", REPO_ROOT / "scripts" / "audit_g0a_predispatch_rrp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_world(n=48):
    base = 1_700_002_800  # hour-aligned: 1_700_002_800 % 3600 == 0
    origins = np.arange(base, base + 3600 * n, 3600, dtype=np.int64)
    return {
        "forecast_origin_unix": origins,
        "predispatch_rrp": np.full((n, 24), 71.5, dtype=np.float32),
        "predispatch_covered": np.ones((n, 24), dtype=np.float32),
        "source_run_unix": origins - 5400,
        "source_last_changed_unix": origins - 5400,
        "source_run_seqno": np.arange(1, n + 1, dtype=np.int64),
    }


class BudgetGuardTests(unittest.TestCase):
    def test_refuses_when_expired(self):
        guard = g0a.BudgetGuard(0.001)
        time.sleep(0.003)
        with self.assertRaises(g0a.BudgetExceeded):
            guard.check()

    def test_report_fields_and_within_budget(self):
        guard = g0a.BudgetGuard(30)
        guard.check()
        report = guard.report()
        self.assertTrue(report["within_budget"])
        self.assertEqual(report["wall_seconds_budget"], 30)
        self.assertGreaterEqual(report["wall_seconds_elapsed"], 0.0)


@unittest.skipUnless(hasattr(signal, "setitimer"), "requires POSIX interval timers")
class WallDeadlineTests(unittest.TestCase):
    """External-audit finding 4: exceeding the budget must interrupt
    immediately (SIGALRM), not wait for the next cooperative check."""

    def test_deadline_interrupts_a_blocking_sleep(self):
        with self.assertRaises(g0a.BudgetExceeded):
            with g0a.wall_deadline(0.3):
                time.sleep(5.0)
        # timer disarmed: a long sleep after the context no longer raises
        time.sleep(0.05)

    def test_deadline_does_not_fire_within_budget(self):
        with g0a.wall_deadline(30.0):
            time.sleep(0.02)

    def test_previous_sigalrm_handler_restored(self):
        before = signal.getsignal(signal.SIGALRM)
        with g0a.wall_deadline(30.0):
            pass
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)

    def test_non_positive_deadline_rejected(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                with g0a.wall_deadline(bad):
                    pass


class LabelIsolationTests(unittest.TestCase):
    def test_path_guard_blocks_label_files_even_inside_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("actual.npy", "target.npy", "ACTUAL.NPY"):
                with self.assertRaises(g0a.PathIsolationError):
                    g0a.assert_allowed_path(root / name, [root])

    def test_path_guard_blocks_outside_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            root.mkdir()
            outside = Path(tmp) / "elsewhere" / "file.json"
            with self.assertRaises(g0a.PathIsolationError):
                g0a.assert_allowed_path(outside, [root])

    def test_path_guard_allows_inside_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            (root / "sub").mkdir(parents=True)
            resolved = g0a.assert_allowed_path(root / "sub" / "data.npz", [root])
            self.assertEqual(resolved.name, "data.npz")

    def test_static_scan_of_frozen_module_is_clean(self):
        source = (REPO_ROOT / "forecasting/agent_forecasting/g0a_predispatch.py").read_text()
        self.assertEqual(g0a.scan_source_for_forbidden(source), [])

    def test_static_scan_detects_dirty_source(self):
        dirty = "value = np.load('runs/r1/actual.npy')\nFORBIDDEN_BASENAMES = ('actual.npy', 'target.npy')\n"
        hits = g0a.scan_source_for_forbidden(dirty)
        self.assertEqual(len(hits), 1)
        self.assertIn("actual.npy", hits[0])


class NetworkKillTests(unittest.TestCase):
    def test_socket_blocked_inside_and_restored_after(self):
        with socket.socket():
            pass  # baseline: sockets work before the guard
        with g0a.network_disabled():
            with self.assertRaises(g0a.NetworkCallBlocked):
                socket.socket()
        with socket.socket():
            pass  # restored


class SelectionRuleTests(unittest.TestCase):
    def test_deterministic_endpoints_and_cap(self):
        for n in (1, 12, 13, 24, 100, 102263):
            indices = g0a.select_origin_indices(n)
            self.assertLessEqual(len(indices), g0a.MAX_ORIGINS)
            self.assertEqual(indices, sorted(set(indices)))
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], n - 1)
            self.assertEqual(indices, g0a.select_origin_indices(n))

    def test_formula_matches_frozen_definition(self):
        n = 102263
        expected = sorted({(k * (n - 1) + 5) // 11 for k in range(12)})
        self.assertEqual(g0a.select_origin_indices(n), expected)

    def test_selection_is_blind_to_coverage_and_values(self):
        n = 500
        origins = np.arange(1_700_000_000, 1_700_000_000 + 3600 * n, 3600, dtype=np.int64)
        indices = g0a.select_origin_indices(n)
        picked_all_covered = origins[indices]
        picked_none_covered = origins[indices]  # coverage/rrp are not inputs at all
        np.testing.assert_array_equal(picked_all_covered, picked_none_covered)
        # the frozen rule consumes the origin count only: no data argument exists
        params = list(inspect.signature(g0a.select_origin_indices).parameters)
        self.assertEqual(params, ["n"])


class UnitEnvelopeTests(unittest.TestCase):
    def test_plausible_values_pass(self):
        result = g0a.check_units(np.array([[-1000.0, 0.0], [65.5, 20000.0]]))
        self.assertEqual(result["status"], "pass")

    def test_out_of_envelope_and_nonfinite_fail(self):
        self.assertEqual(g0a.check_units(np.array([[20000.1]]))["status"], "fail")
        self.assertEqual(g0a.check_units(np.array([[-1000.1]]))["status"], "fail")
        self.assertEqual(g0a.check_units(np.array([[np.nan]]))["status"], "fail")
        self.assertEqual(g0a.check_units(np.array([[np.inf]]))["status"], "fail")


class VintageTests(unittest.TestCase):
    def test_clean_vintage_passes(self):
        world = synthetic_world()
        result = g0a.check_vintage(
            world["forecast_origin_unix"], world["source_run_unix"], world["source_run_seqno"]
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["veto_reasons"], [])
        self.assertTrue(all(not r["future_run_violation"] for r in result["rows"]))

    def test_future_run_vetoes(self):
        origins = np.array([1_700_000_000], dtype=np.int64)
        run = np.array([1_700_000_000 + 60], dtype=np.int64)
        result = g0a.check_vintage(origins, run, np.array([1], dtype=np.int64))
        self.assertEqual(result["status"], "fail")
        self.assertIn("future information", result["veto_reasons"][0])

    def test_stale_run_vetoes(self):
        origins = np.array([1_700_000_000], dtype=np.int64)
        run = np.array([1_700_000_000 - 7 * 3600], dtype=np.int64)
        result = g0a.check_vintage(origins, run, np.array([1], dtype=np.int64))
        self.assertEqual(result["status"], "fail")
        self.assertIn("median fill", result["veto_reasons"][0])


class CoverageNoFillTests(unittest.TestCase):
    def test_fully_covered_passes(self):
        origins = np.array([1_700_000_000], dtype=np.int64)
        run = np.array([1_700_000_000 - 3600], dtype=np.int64)
        covered = np.ones((1, 24), dtype=np.float32)
        values = np.full((1, 24), 71.5, dtype=np.float32)
        result = g0a.check_coverage(origins, covered, run, values)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["rows"][0]["fill_kind"], "none")
        self.assertIsNone(result["rows"][0]["tail_repeat_verified"])

    def test_tail_ffill_flagged_when_values_repeat_last_covered(self):
        origins = np.array([1_700_000_000], dtype=np.int64)
        run = np.array([1_700_000_000 - 3600], dtype=np.int64)
        covered = np.ones((1, 24), dtype=np.float32)
        covered[0, 22:] = 0.0
        values = np.full((1, 24), 71.5, dtype=np.float32)
        result = g0a.check_coverage(origins, covered, run, values)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["rows"][0]["uncovered_hours"], 2)
        self.assertEqual(result["rows"][0]["fill_kind"], "tail_ffill_repeat")
        self.assertTrue(result["rows"][0]["tail_repeat_verified"])
        self.assertEqual(result["rows"][0]["last_covered_hour"], 21)
        self.assertEqual(result["rows"][0]["last_covered_value"], 71.5)
        self.assertEqual(result["rows"][0]["tail_hours_after_last_covered"], [22, 23])
        self.assertIn("tail_repeat_verified=True", result["fill_reasons"][0])

    def test_stale_median_fill_flagged(self):
        origins = np.array([1_700_000_000], dtype=np.int64)
        run = np.array([1_700_000_000 - 8 * 3600], dtype=np.int64)
        covered = np.zeros((1, 24), dtype=np.float32)
        values = np.full((1, 24), 71.5, dtype=np.float32)
        result = g0a.check_coverage(origins, covered, run, values)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["rows"][0]["fill_kind"], "stale_median_fill")

    def test_inconsistent_stale_but_covered_flagged(self):
        origins = np.array([1_700_000_000], dtype=np.int64)
        run = np.array([1_700_000_000 - 8 * 3600], dtype=np.int64)
        covered = np.ones((1, 24), dtype=np.float32)
        values = np.full((1, 24), 71.5, dtype=np.float32)
        result = g0a.check_coverage(origins, covered, run, values)
        self.assertEqual(result["status"], "fail")
        self.assertIn("inconsistent", result["fill_reasons"][0])


class CoverageValueLevelTests(unittest.TestCase):
    """External-audit finding 2: the repeat label must be measured on values,
    never inferred from the covered flag alone."""

    def one_row_world(self):
        return (
            np.array([1_700_000_000], dtype=np.int64),
            np.array([1_700_000_000 - 3600], dtype=np.int64),
        )

    def test_uncovered_tail_with_different_value_is_not_labeled_repeat(self):
        origins, run = self.one_row_world()
        covered = np.ones((1, 24), dtype=np.float32)
        covered[0, 21:] = 0.0
        values = np.full((1, 24), 60.0, dtype=np.float32)
        values[0, 22] = 61.0  # uncovered hour differs from last covered value
        result = g0a.check_coverage(origins, covered, run, values)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["rows"][0]["fill_kind"], "uncovered_non_repeat")
        self.assertFalse(result["rows"][0]["tail_repeat_verified"])
        self.assertIn("tail_repeat_verified=False", result["fill_reasons"][0])

    def test_float32_stored_repeats_verify_exactly(self):
        origins, run = self.one_row_world()
        covered = np.ones((1, 24), dtype=np.float32)
        covered[0, 18:] = 0.0
        values = np.linspace(50.0, 75.0, 24).astype(np.float32).reshape(1, 24)
        values[0, 18:] = values[0, 17]  # exact stored-value repeat
        result = g0a.check_coverage(origins, covered, run, values)
        self.assertEqual(result["rows"][0]["fill_kind"], "tail_ffill_repeat")
        self.assertTrue(result["rows"][0]["tail_repeat_verified"])
        self.assertEqual(result["rows"][0]["last_covered_value"], float(values[0, 17]))
        self.assertEqual(result["rows"][0]["tail_hours_after_last_covered"], [18, 19, 20, 21, 22, 23])

    def test_gap_without_tail_is_not_labeled_repeat(self):
        origins, run = self.one_row_world()
        covered = np.ones((1, 24), dtype=np.float32)
        covered[0, 5] = 0.0  # uncovered positions exist but none after the last covered hour
        values = np.full((1, 24), 71.5, dtype=np.float32)
        result = g0a.check_coverage(origins, covered, run, values)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["rows"][0]["fill_kind"], "gap_uncovered")
        self.assertIsNone(result["rows"][0]["tail_repeat_verified"])

    def test_no_covered_anchor_is_reported_explicitly(self):
        origins, run = self.one_row_world()
        covered = np.zeros((1, 24), dtype=np.float32)
        values = np.full((1, 24), 71.5, dtype=np.float32)
        result = g0a.check_coverage(origins, covered, run, values)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["rows"][0]["fill_kind"], "uncovered_no_anchor")
        self.assertIsNone(result["rows"][0]["tail_repeat_verified"])

    def test_values_argument_is_required(self):
        origins, run = self.one_row_world()
        with self.assertRaises(TypeError):
            g0a.check_coverage(origins, np.ones((1, 24), dtype=np.float32), run)


class PublicationBasisTests(unittest.TestCase):
    def test_alias_equality_passes_as_archive_proxy(self):
        run = np.array([10, 20], dtype=np.int64)
        result = g0a.check_publication_basis(run, run)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["publication_basis"], "archive_proxy")
        self.assertFalse(result["first_seen_claimed"])

    def test_alias_mismatch_fails(self):
        result = g0a.check_publication_basis(
            np.array([10, 20], dtype=np.int64), np.array([10, 21], dtype=np.int64)
        )
        self.assertEqual(result["status"], "fail")


class SchemaTests(unittest.TestCase):
    def test_synthetic_world_passes(self):
        result = g0a.check_schema(synthetic_world())
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["n_origins"], 48)

    def test_missing_key_fails(self):
        world = synthetic_world()
        del world["source_run_seqno"]
        result = g0a.check_schema(world)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("source_run_seqno" in p for p in result["problems"]))

    def test_non_hourly_origins_fail(self):
        world = synthetic_world()
        origins = world["forecast_origin_unix"].copy()
        origins[5] += 1
        world["forecast_origin_unix"] = origins
        result = g0a.check_schema(world)
        self.assertEqual(result["status"], "fail")

    def test_bad_shape_fails(self):
        world = synthetic_world()
        world["predispatch_rrp"] = world["predispatch_rrp"][:, :23]
        result = g0a.check_schema(world)
        self.assertEqual(result["status"], "fail")

    def test_target_time_needs_hour_grid_and_24_columns(self):
        world = synthetic_world()
        ok = g0a.check_target_time(world["forecast_origin_unix"], 24)
        self.assertEqual(ok["status"], "pass")
        shifted = world["forecast_origin_unix"] + 17
        self.assertEqual(g0a.check_target_time(shifted, 24)["status"], "fail")
        self.assertEqual(g0a.check_target_time(world["forecast_origin_unix"], 23)["status"], "fail")


class VerdictTests(unittest.TestCase):
    def base_checks(self):
        world = synthetic_world()
        return [
            g0a.check_schema(world),
            g0a.check_units(world["predispatch_rrp"]),
            g0a.check_target_time(world["forecast_origin_unix"], 24),
            g0a.check_vintage(
                world["forecast_origin_unix"], world["source_run_unix"], world["source_run_seqno"]
            ),
            g0a.check_coverage(
                world["forecast_origin_unix"],
                world["predispatch_covered"],
                world["source_run_unix"],
                world["predispatch_rrp"],
            ),
            g0a.check_publication_basis(
                world["source_run_unix"], world["source_last_changed_unix"]
            ),
        ]

    def test_clean_world_is_engineering_pass_but_strict_blocked(self):
        verdict = g0a.assemble_verdict(
            checks=self.base_checks(),
            budget={"within_budget": True, "wall_seconds_budget": 300, "wall_seconds_elapsed": 0.1},
            preflight={"returncode": 0},
            guard_violations=[],
        )
        self.assertEqual(verdict["engineering"], "ENGINEERING_PASS")
        self.assertEqual(verdict["strict_asof"], "STRICT_ASOF_BLOCKED")
        self.assertEqual(verdict["publication_basis"], "archive_proxy")
        self.assertEqual(verdict["quick_veto"]["status"], "NOT_TRIGGERED")
        self.assertFalse(verdict["first_seen_claimed"])

    def test_failed_preflight_is_engineering_fail(self):
        verdict = g0a.assemble_verdict(
            checks=self.base_checks(),
            budget={"within_budget": True},
            preflight={"returncode": 1},
            guard_violations=[],
        )
        self.assertEqual(verdict["engineering"], "ENGINEERING_FAIL")

    def test_fill_triggers_quick_veto_without_engineering_fail(self):
        world = synthetic_world()
        covered = world["predispatch_covered"].copy()
        covered[7, 20:] = 0.0
        checks = self.base_checks()
        checks[4] = g0a.check_coverage(
            world["forecast_origin_unix"],
            covered,
            world["source_run_unix"],
            world["predispatch_rrp"],
        )
        verdict = g0a.assemble_verdict(
            checks=checks,
            budget={"within_budget": True},
            preflight={"returncode": 0},
            guard_violations=[],
        )
        self.assertEqual(verdict["engineering"], "ENGINEERING_PASS")
        self.assertEqual(verdict["quick_veto"]["status"], "TRIGGERED")
        self.assertEqual(verdict["strict_asof"], "STRICT_ASOF_BLOCKED")

    def test_strict_pass_is_never_grantable(self):
        verdict = g0a.assemble_verdict(
            checks=self.base_checks(),
            budget={"within_budget": True},
            preflight={"returncode": 0},
            guard_violations=[],
        )
        tampered = dict(verdict, strict_asof="STRICT_ASOF_PASS")
        with self.assertRaises(g0a.StrictPassNotGrantable):
            g0a.assert_never_strict_pass(tampered)


class GitHeadReaderTests(unittest.TestCase):
    """External-audit finding 4: HEAD must be read in-process (budget
    processes=1), never via a git subprocess in the audit path. The tests
    themselves may use git once, here, only to prove equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.runner = load_audit_runner()

    def test_real_repo_head_matches_git(self):
        head = self.runner.git_head(REPO_ROOT)
        self.assertRegex(head, r"^[0-9a-f]{40}$")
        expected = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        self.assertEqual(head, expected)

    def test_reads_ref_file_without_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            git_dir = Path(tmp) / ".git"
            (git_dir / "refs" / "heads" / "feature").mkdir(parents=True)
            (git_dir / "HEAD").write_text("ref: refs/heads/feature/x\n")
            (git_dir / "refs" / "heads" / "feature" / "x").write_text("a" * 40 + "\n")
            self.assertEqual(self.runner.git_head(Path(tmp)), "a" * 40)

    def test_reads_packed_refs_when_loose_ref_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            git_dir = Path(tmp) / ".git"
            (git_dir / "refs" / "heads").mkdir(parents=True)
            (git_dir / "HEAD").write_text("ref: refs/heads/packed\n")
            (git_dir / "packed-refs").write_text("# pack-refs with: peeled fully-peeled \n" + "b" * 40 + " refs/heads/packed\n")
            self.assertEqual(self.runner.git_head(Path(tmp)), "b" * 40)

    def test_detached_head_returns_raw_commit_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            git_dir = Path(tmp) / ".git"
            git_dir.mkdir()
            (git_dir / "HEAD").write_text("c" * 40 + "\n")
            self.assertEqual(self.runner.git_head(Path(tmp)), "c" * 40)

    def test_missing_ref_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            git_dir = Path(tmp) / ".git"
            git_dir.mkdir()
            (git_dir / "HEAD").write_text("ref: refs/heads/nowhere\n")
            with self.assertRaises(FileNotFoundError):
                self.runner.git_head(Path(tmp))


class FrozenBindingTests(unittest.TestCase):
    """Source-bound: the frozen config must still match the live NPZ and builder."""

    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text())

    def test_config_binds_live_npz_hash(self):
        npz = REPO_ROOT / self.config["npz_path"]
        if not npz.exists():
            self.skipTest(f"{npz} absent on this machine")
        self.assertEqual(sha256_file(npz), self.config["npz_sha256_at_freeze"])

    def test_config_binds_live_builder_hash(self):
        builder = REPO_ROOT / self.config["builder_path"]
        if not builder.exists():
            self.skipTest(f"{builder} absent on this machine")
        self.assertEqual(sha256_file(builder), self.config["builder_sha256_at_freeze"])

    def test_config_freeze_invariants(self):
        self.assertEqual(self.config["region"], "NSW1")
        self.assertEqual(self.config["field"], "predispatch_rrp")
        self.assertEqual(self.config["origin_selection"]["max_origins"], 12)
        self.assertEqual(self.config["budget"]["network"], 0)
        self.assertEqual(self.config["budget"]["gpu"], 0)
        self.assertEqual(self.config["budget"]["training"], 0)
        self.assertEqual(self.config["verdict_levels"]["strict_asof"], ["STRICT_ASOF_BLOCKED"])
        self.assertFalse(self.config["out_of_scope"]["g0b_g0c_g1_g2"])
        self.assertEqual(self.config["out_of_scope"]["live_collection_72h"], "not_started")

    def test_config_budget_matches_frozen_protocol_constant(self):
        self.assertEqual(
            float(self.config["budget"]["wall_seconds"]), g0a.WALL_SECONDS_BUDGET
        )
        self.assertEqual(self.config["budget"]["processes"], 1)


EVIDENCE_FILENAMES = (
    "g0a_verdict.json",
    "budget_report.json",
    "run_manifest.json",
    "selected_origins.json",
)

GATE_FAULT_DRIVER = '''
import importlib.util, os, sys

spec = importlib.util.spec_from_file_location("runner_under_test", os.environ["RUNNER_PATH"])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

fault = os.environ.get("GATE_FAULT")
if fault == "source_hashes_runtime":
    def _boom(*args, **kwargs):
        raise RuntimeError("injected final-phase source_hashes failure")
    runner.source_hashes = _boom
elif fault == "source_hashes_budget":
    def _boom(*args, **kwargs):
        raise runner.g0a.BudgetExceeded("injected final-phase BudgetExceeded")
    runner.source_hashes = _boom
elif fault == "git_head":
    def _boom(*args, **kwargs):
        raise RuntimeError("injected final-phase HEAD read failure")
    runner.git_head = _boom
elif fault == "deadline_at_entry":
    runner.g0a.WALL_SECONDS_BUDGET = 0.05
elif fault == "rename_publish":
    def _boom(src, dst):
        raise OSError("injected publication rename failure")
    runner.os.rename = _boom

sys.argv = ["audit_g0a_predispatch_rrp.py", "audit", "--config", os.environ["GATE_CONFIG_PATH"]]
runner.main()
'''


class GateRepairWorldMixin:
    """Builders for the gate-reliability tests: a complete synthetic frozen
    world (NPZ + config with real frozen-binding hashes) and a fault-injection
    subprocess driver so real process exit codes are observed."""

    def build_world(self, tmp, wall_seconds=300.0):
        tmp = Path(tmp)
        npz_dir = tmp / "merged"
        npz_dir.mkdir(parents=True, exist_ok=True)
        npz_path = npz_dir / "synthetic_predispatch.npz"
        np.savez(npz_path, **synthetic_world())
        run_dir = tmp / "run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "preflight_tests.json").write_text(
            json.dumps({"returncode": 0, "log_path": "preflight_tests.txt"})
        )
        config = {
            "status": "frozen_g0a",
            "execution_scope": "g0a_local_preaudit_only",
            "base_commit": "synthetic-base",
            "branch": "synthetic-gate-repair-test",
            "protocol": "docs/research/agent_forecasting/g0a_predispatch_rrp_protocol_20261028.txt",
            "harness_module": "forecasting/agent_forecasting/g0a_predispatch.py",
            "audit_script": "scripts/audit_g0a_predispatch_rrp.py",
            "tests": "tests/test_g0a_predispatch_rrp.py",
            "region": "NSW1",
            "field": "predispatch_rrp",
            "npz_path": str(npz_path),
            "npz_sha256_at_freeze": sha256_file(npz_path),
            "builder_path": "forecasting/build_predispatch_exogenous.py",
            "builder_sha256_at_freeze": sha256_file(
                REPO_ROOT / "forecasting/build_predispatch_exogenous.py"
            ),
            "unit_envelope_aud_per_mwh": {"min": -1000.0, "max": 20000.0},
            "stale_seconds": 21600,
            "origin_selection": {"rule": "even_index_stride_round_half_up", "max_origins": 12},
            "budget": {
                "wall_seconds": wall_seconds,
                "processes": 1,
                "network": 0,
                "gpu": 0,
                "training": 0,
                "model_calls": 0,
                "package_installs": 0,
            },
            "allow_roots": [str(npz_dir), str(run_dir)],
            "output_dir": str(run_dir),
            "archive_dir": str(tmp / "archive"),
        }
        config_path = tmp / "gate_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        return config_path, run_dir

    def run_audit_subprocess(self, config_path, tmp, fault=None):
        driver = Path(tmp) / "gate_fault_driver.py"
        driver.write_text(GATE_FAULT_DRIVER)
        env = dict(os.environ)
        env["RUNNER_PATH"] = str(REPO_ROOT / "scripts" / "audit_g0a_predispatch_rrp.py")
        env["GATE_CONFIG_PATH"] = str(config_path)
        if fault:
            env["GATE_FAULT"] = fault
        return subprocess.run(
            [sys.executable, str(driver)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=180,
        )

    def load_failure_record(self, run_dir):
        failures_dir = run_dir.parent / (run_dir.name + "_failures")
        records = sorted(failures_dir.glob("g0a_failure_*.json"))
        self.assertTrue(
            records,
            f"expected an ENGINEERING_FAIL failure record under {failures_dir}; "
            f"contents: {sorted(p.name for p in run_dir.parent.iterdir())}",
        )
        return json.loads(records[-1].read_text())


class GateFinalizationFaultTests(GateRepairWorldMixin, unittest.TestCase):
    """Repair requirement 1: a BudgetExceeded or any other exception in the
    final source_hashes/HEAD/budget phase must never leave ENGINEERING_PASS
    evidence, a success verdict, or a success budget/manifest behind; the
    process must exit non-zero and atomically write an explicit
    ENGINEERING_FAIL record."""

    def assert_clean_failure(self, proc, run_dir, expected_phase=None):
        self.assertNotEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
        for name in EVIDENCE_FILENAMES:
            self.assertFalse(
                (run_dir / name).exists(),
                f"{name} must not survive a failed audit (found in run dir)",
            )
        record = self.load_failure_record(run_dir)
        self.assertEqual(record["engineering"], "ENGINEERING_FAIL")
        self.assertIn("exception_type", record)
        if expected_phase is not None:
            self.assertEqual(record["failed_phase"], expected_phase)
        return record

    def test_source_hashes_runtime_error_leaves_no_success_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp)
            proc = self.run_audit_subprocess(config_path, tmp, fault="source_hashes_runtime")
            record = self.assert_clean_failure(proc, run_dir, expected_phase="final_hashes")
            self.assertEqual(record["exception_type"], "RuntimeError")

    def test_budget_exceeded_in_final_phase_leaves_no_success_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp)
            proc = self.run_audit_subprocess(config_path, tmp, fault="source_hashes_budget")
            record = self.assert_clean_failure(proc, run_dir, expected_phase="final_hashes")
            self.assertEqual(record["exception_type"], "BudgetExceeded")

    def test_git_head_failure_leaves_no_success_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp)
            proc = self.run_audit_subprocess(config_path, tmp, fault="git_head")
            record = self.assert_clean_failure(proc, run_dir, expected_phase="final_hashes")
            self.assertEqual(record["exception_type"], "RuntimeError")

    def test_exhausted_budget_from_process_start_fails_cleanly(self):
        """With a budget already consumed by process startup itself, the audit
        must fail before doing any work: no evidence, failure record, non-zero
        exit. This pins the process-start budget anchor (requirement 3)."""
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp, wall_seconds=0.05)
            proc = self.run_audit_subprocess(config_path, tmp, fault="deadline_at_entry")
            self.assertNotEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
            for name in EVIDENCE_FILENAMES:
                self.assertFalse((run_dir / name).exists())
            record = self.load_failure_record(run_dir)
            self.assertEqual(record["engineering"], "ENGINEERING_FAIL")

    def test_publication_rename_failure_leaves_no_partial_success_set(self):
        """Repair requirement 2 (write-path fault injection): if the atomic
        publication rename itself fails, no partial success set may appear in
        the run dir; the failure record must name the staging directory that
        holds the complete unpublished set."""
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp)
            proc = self.run_audit_subprocess(config_path, tmp, fault="rename_publish")
            record = self.assert_clean_failure(proc, run_dir, expected_phase="publication")
            staging = Path(record["staging_dir"])
            self.assertTrue(staging.exists(), "staged evidence must be retained for recovery")
            self.assertTrue((staging / "g0a_verdict.json").exists())


class GatePublicationAtomicityTests(GateRepairWorldMixin, unittest.TestCase):
    """Repair requirements 2 and 3: success evidence is published exactly once
    via a staging directory + atomic rename after ALL final checks, and the
    budget clock covers process start through evidence finalization."""

    def test_success_publishes_complete_set_with_process_start_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp)
            proc = self.run_audit_subprocess(config_path, tmp)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
            for name in EVIDENCE_FILENAMES:
                self.assertTrue((run_dir / name).exists(), f"{name} missing after success")
            # preflight inputs stay in the published run dir
            self.assertTrue((run_dir / "preflight_tests.json").exists())
            verdict = json.loads((run_dir / "g0a_verdict.json").read_text())
            self.assertEqual(verdict["engineering"], "ENGINEERING_PASS")
            budget_report = json.loads((run_dir / "budget_report.json").read_text())
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            # requirement 3: budget elapsed must not be smaller than the full
            # run elapsed recorded in the manifest (same process-start anchor)
            self.assertGreaterEqual(
                budget_report["wall_seconds_elapsed"],
                manifest["wall_seconds"],
                "budget accounting must cover the full run wall time",
            )
            self.assertEqual(manifest["wall_clock_anchor"], "process_start")
            phases = budget_report["phases"]
            self.assertGreater(
                phases["pre_audit_process_import_cli_s"],
                0.0,
                "budget timing must cover interpreter start, module imports and CLI parsing",
            )
            self.assertGreaterEqual(manifest["wall_seconds"], phases["pre_audit_process_import_cli_s"])
            # no staging leftovers next to the published run dir
            leftovers = list(run_dir.parent.glob(run_dir.name + ".publish-*"))
            self.assertEqual(leftovers, [])

    def test_evidence_not_published_before_final_checks(self):
        """Write interception: when the final source hashes are computed, the
        run dir must not yet contain any success evidence file — publication
        may only happen after all final checks complete."""
        runner = load_audit_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp)
            observed = {}
            real_source_hashes = runner.source_hashes

            def spy(*args, **kwargs):
                observed["occupants"] = sorted(p.name for p in run_dir.iterdir())
                return real_source_hashes(*args, **kwargs)

            runner.source_hashes = spy
            try:
                runner.cmd_audit(argparse.Namespace(config=str(config_path)))
            finally:
                runner.source_hashes = real_source_hashes
            leaked = [n for n in observed["occupants"] if n in EVIDENCE_FILENAMES]
            self.assertEqual(
                leaked,
                [],
                "evidence files must not exist in the run dir before the final source hashes",
            )


class GateEvidenceProtectionTests(GateRepairWorldMixin, unittest.TestCase):
    """Repair requirement 4: failure and rerun paths must never damage or
    overwrite already-published evidence directories."""

    def test_failed_audit_never_touches_published_evidence(self):
        runner = load_audit_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp)
            sentinel = {
                "audit": "g0a_predispatch_rrp_asof_preaudit",
                "engineering": "ENGINEERING_PASS",
                "sentinel": "previously published run, do not touch",
            }
            (run_dir / "g0a_verdict.json").write_text(json.dumps(sentinel, indent=2))
            verdict_before = sha256_file(run_dir / "g0a_verdict.json")

            real_source_hashes = runner.source_hashes

            def boom(*args, **kwargs):
                raise runner.g0a.BudgetExceeded("injected final-phase BudgetExceeded")

            runner.source_hashes = boom
            try:
                with self.assertRaises(SystemExit) as ctx:
                    runner.cmd_audit(argparse.Namespace(config=str(config_path)))
            finally:
                runner.source_hashes = real_source_hashes
            self.assertNotEqual(ctx.exception.code, 0)
            self.assertEqual(
                sha256_file(run_dir / "g0a_verdict.json"),
                verdict_before,
                "a failed audit must not modify published evidence",
            )
            record = self.load_failure_record(run_dir)
            self.assertEqual(record["engineering"], "ENGINEERING_FAIL")
            self.assertFalse(record["evidence_published"])

    def test_successful_rerun_refuses_to_overwrite_published_evidence(self):
        runner = load_audit_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp)
            sentinel = {"engineering": "ENGINEERING_PASS", "sentinel": "immutable"}
            (run_dir / "g0a_verdict.json").write_text(json.dumps(sentinel))
            verdict_before = sha256_file(run_dir / "g0a_verdict.json")
            with self.assertRaises(SystemExit) as ctx:
                runner.cmd_audit(argparse.Namespace(config=str(config_path)))
            self.assertNotEqual(ctx.exception.code, 0)
            self.assertEqual(sha256_file(run_dir / "g0a_verdict.json"), verdict_before)
            record = self.load_failure_record(run_dir)
            self.assertEqual(record["engineering"], "ENGINEERING_FAIL")
            self.assertEqual(record["failed_phase"], "publication")

    def test_record_tests_refuses_when_evidence_published(self):
        runner = load_audit_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config_path, run_dir = self.build_world(tmp)
            (run_dir / "preflight_tests.exit").write_text("0\n")
            (run_dir / "preflight_tests.txt").write_text("3 passed\n")
            sentinel_preflight = {"sentinel": "published preflight record"}
            (run_dir / "preflight_tests.json").write_text(json.dumps(sentinel_preflight))
            (run_dir / "g0a_verdict.json").write_text('{"engineering": "ENGINEERING_PASS"}')
            with self.assertRaises(SystemExit) as ctx:
                runner.cmd_record_tests(argparse.Namespace(config=str(config_path)))
            self.assertNotEqual(ctx.exception.code, 0)
            self.assertEqual(
                json.loads((run_dir / "preflight_tests.json").read_text()), sentinel_preflight
            )


if __name__ == "__main__":
    unittest.main()
