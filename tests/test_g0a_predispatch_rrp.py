"""G0-A frozen-harness tests: budget guard + hard wall deadline, label
isolation, units, coverage with value-level fill verification, coverage-blind
selection, schema, single-process HEAD reading, and source/data binding.
All value-level tests use synthetic arrays; the only real files touched are
the frozen NPZ/config hash binding and the repo .git directory (read-only)."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import signal
import socket
import subprocess
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


if __name__ == "__main__":
    unittest.main()
