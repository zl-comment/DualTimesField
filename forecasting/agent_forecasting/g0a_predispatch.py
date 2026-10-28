"""Frozen G0-A harness: local, label-free as-of pre-audit of NSW1 predispatch_rrp.

Companion protocol: docs/research/agent_forecasting/g0a_predispatch_rrp_protocol_20261028.txt
Companion config:   configs/agent_forecasting/g0a_predispatch_rrp_20261028.json

The archive NPZ can only quick-veto as-of use of ``predispatch_rrp``; it can
never, alone, support a strict point-in-time PASS. Every verdict therefore
carries ``publication_basis="archive_proxy"`` and
``strict_asof="STRICT_ASOF_BLOCKED"``; no code path may emit a strict PASS.

This module is frozen before the audit runs: budget guard, path/label
isolation, network kill, the coverage-blind origin-selection rule, and the
six pure checks. It reads no files itself and never modifies any value.
"""
from __future__ import annotations

import re
import signal
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

MAX_ORIGINS = 12
STALE_SECONDS = 6 * 3600
HOUR_SECONDS = 3600
WALL_SECONDS_BUDGET = 300  # protocol §5; the runner asserts config agreement
UNIT_ENVELOPE_AUD_PER_MWH = (-1000.0, 20000.0)
REQUIRED_KEYS = (
    "forecast_origin_unix",
    "predispatch_rrp",
    "predispatch_covered",
    "source_run_unix",
    "source_last_changed_unix",
    "source_run_seqno",
)
FORBIDDEN_BASENAMES = ("actual.npy", "target.npy")
_FORBIDDEN_LITERAL = re.compile(r"(actual\.npy|target\.npy)")
PUBLICATION_BASIS = "archive_proxy"
STRICT_ASOF_STATUS = "STRICT_ASOF_BLOCKED"
STRICT_BLOCK_REASONS = (
    "source_last_changed_unix is an archive proxy: MMSDM monthly-archive "
    "LASTCHANGED fetched at NPZ build time, not a live first-seen capture",
    "backdated late revisions with retroactive LASTCHANGED cannot be excluded "
    "from a month-end archive",
    "no prospective live capture was started in G0-A, so historical "
    "availability at each origin is unproven",
)
TARGET_TIME_MAPPING = (
    "row value at column h (h=0..23) is the mean of the two half hours of the "
    "trading hour ending at origin+(h+1)*3600 seconds (AEMO period-end "
    "convention, AEST=UTC+10); lineage bound to the frozen builder"
)


class BudgetExceeded(RuntimeError):
    """Wall-clock budget of the G0-A audit was exceeded."""


class PathIsolationError(RuntimeError):
    """A path outside the audit allow roots or a forbidden label file."""


class NetworkCallBlocked(RuntimeError):
    """G0-A forbids network access."""


class StrictPassNotGrantable(RuntimeError):
    """A strict as-of PASS cannot be granted from this artifact class."""


def select_origin_indices(n: int) -> list[int]:
    """Frozen coverage-blind selection: all origins if n<=12, else an even
    round-half-up index stride over the ascending origin array (k=0..11)."""
    if n <= 0:
        raise ValueError("no origins to select from")
    if n <= MAX_ORIGINS:
        return list(range(n))
    return sorted({(k * (n - 1) + 5) // 11 for k in range(MAX_ORIGINS)})


class BudgetGuard:
    """Monotonic wall-clock guard; refuses to continue once exceeded."""

    def __init__(self, wall_seconds: float):
        if wall_seconds <= 0:
            raise ValueError("wall_seconds must be positive")
        self.wall_seconds = float(wall_seconds)
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def check(self) -> None:
        if self.elapsed > self.wall_seconds:
            raise BudgetExceeded(
                f"wall budget {self.wall_seconds:.3f}s exceeded at {self.elapsed:.3f}s"
            )

    def report(self) -> dict:
        elapsed = self.elapsed
        return {
            "wall_seconds_budget": self.wall_seconds,
            "wall_seconds_elapsed": round(elapsed, 6),
            "within_budget": elapsed <= self.wall_seconds,
        }


@contextmanager
def wall_deadline(wall_seconds: float):
    """Hard wall-clock deadline (protocol §5 "超限立即终止"): arms an ITIMER_REAL
    timer so exceeding the budget interrupts the audit immediately, instead of
    waiting for the next cooperative ``BudgetGuard.check()`` call. Main thread
    only, as the audit is single-process."""
    wall_seconds = float(wall_seconds)
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    if not hasattr(signal, "setitimer"):  # pragma: no cover - non-POSIX fallback
        raise RuntimeError("wall_deadline requires signal.setitimer (POSIX)")
    previous = signal.getsignal(signal.SIGALRM)

    def _on_deadline(signum, frame):
        raise BudgetExceeded(
            f"wall deadline {wall_seconds:.3f}s reached: immediate interrupt"
        )

    signal.signal(signal.SIGALRM, _on_deadline)
    signal.setitimer(signal.ITIMER_REAL, wall_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def assert_allowed_path(path, allow_roots) -> Path:
    """Rejects forbidden label basenames anywhere and paths outside allow roots."""
    resolved = Path(path).resolve()
    if resolved.name.lower() in FORBIDDEN_BASENAMES:
        raise PathIsolationError(f"forbidden label file: {resolved.name}")
    for root in allow_roots:
        root_resolved = Path(root).resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            return resolved
    raise PathIsolationError(f"path outside audit allow roots: {resolved}")


@contextmanager
def network_disabled():
    """Replaces socket.socket so any connection attempt fails loudly."""
    original = socket.socket

    def _blocked(*args, **kwargs):
        raise NetworkCallBlocked("G0-A forbids network access")

    socket.socket = _blocked
    try:
        yield
    finally:
        socket.socket = original


def scan_source_for_forbidden(text: str) -> list[str]:
    """Static scan: forbidden label references outside the denylist declaration."""
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        if "FORBIDDEN_BASENAMES" in line:
            continue
        if _FORBIDDEN_LITERAL.search(line):
            hits.append(f"line {number}: {line.strip()}")
    return hits


def check_schema(arrays: dict) -> dict:
    problems = []
    for key in REQUIRED_KEYS:
        if key not in arrays:
            problems.append(f"missing key {key}")
    origins = np.asarray(arrays.get("forecast_origin_unix", []))
    n = len(origins) if origins.ndim == 1 else None
    if "forecast_origin_unix" in arrays:
        if origins.ndim != 1:
            problems.append("forecast_origin_unix is not 1-D")
        elif n == 0:
            problems.append("no origins")
        elif not np.all(np.diff(origins) == HOUR_SECONDS):
            problems.append("origins not strictly hourly at 3600s spacing")
    for key in ("predispatch_rrp", "predispatch_covered"):
        if key in arrays:
            array = np.asarray(arrays[key])
            if array.shape != (n, 24):
                problems.append(f"{key} shape {array.shape} != ({n}, 24)")
            elif not np.issubdtype(array.dtype, np.floating):
                problems.append(f"{key} dtype {array.dtype} is not floating")
    for key in ("source_run_unix", "source_last_changed_unix", "source_run_seqno"):
        if key in arrays:
            array = np.asarray(arrays[key])
            if array.shape != (n,):
                problems.append(f"{key} shape {array.shape} != ({n},)")
            elif not np.issubdtype(array.dtype, np.integer):
                problems.append(f"{key} dtype {array.dtype} is not integer")
    return {
        "check": "schema",
        "status": "fail" if problems else "pass",
        "problems": problems,
        "n_origins": n,
        "structural_no_splicing_evidence": "one scalar source_run_seqno per origin row",
    }


def check_units(values, envelope=UNIT_ENVELOPE_AUD_PER_MWH) -> dict:
    array = np.asarray(values, dtype=np.float64)
    low, high = envelope
    all_finite = bool(np.isfinite(array).all())
    within = bool(((array >= low) & (array <= high)).all()) if all_finite else False
    return {
        "check": "units_plausibility_aud_per_mwh",
        "status": "pass" if all_finite and within else "fail",
        "unit": "AUD/MWh",
        "basis": "PREDISPATCHPRICE.RRP lineage via frozen builder; plausibility envelope only, not a unit proof",
        "envelope": [low, high],
        "all_finite": all_finite,
        "all_within_envelope": within,
        "min": float(array.min()) if all_finite else None,
        "max": float(array.max()) if all_finite else None,
    }


def check_target_time(origins, n_columns: int) -> dict:
    origins = np.asarray(origins)
    on_hour_grid = bool(len(origins) and np.all(origins % HOUR_SECONDS == 0))
    ok = on_hour_grid and n_columns == 24
    return {
        "check": "target_time_alignment",
        "status": "pass" if ok else "fail",
        "origins_on_hour_grid": on_hour_grid,
        "n_columns": n_columns,
        "mapping": TARGET_TIME_MAPPING,
    }


def _stale_flags(origins, run_unix, stale_seconds=STALE_SECONDS) -> np.ndarray:
    return np.asarray(origins) - np.asarray(run_unix) > stale_seconds


def check_vintage(origins, run_unix, seqno, stale_seconds=STALE_SECONDS) -> dict:
    origins = np.asarray(origins)
    run_unix = np.asarray(run_unix)
    seqno = np.asarray(seqno)
    age = origins - run_unix
    rows, vetoes = [], []
    for i in range(len(origins)):
        future = bool(run_unix[i] > origins[i])
        stale = bool(age[i] > stale_seconds)
        rows.append(
            {
                "origin_unix": int(origins[i]),
                "source_run_unix": int(run_unix[i]),
                "run_age_seconds": int(age[i]),
                "run_age_hours": round(float(age[i]) / 3600.0, 4),
                "source_run_seqno": int(seqno[i]),
                "future_run_violation": future,
                "stale_median_fill": stale,
            }
        )
        if future:
            vetoes.append(
                f"origin {int(origins[i])}: run available at {int(run_unix[i])} after origin (future information)"
            )
        elif stale:
            vetoes.append(
                f"origin {int(origins[i])}: run age {int(age[i])}s exceeds {stale_seconds}s (median fill)"
            )
    return {
        "check": "run_vintage_asof",
        "status": "fail" if vetoes else "pass",
        "veto_reasons": vetoes,
        "rows": rows,
    }


def check_coverage(origins, covered, run_unix, values, stale_seconds=STALE_SECONDS) -> dict:
    """24-hour no-fill check. Beyond the covered flags, uncovered non-stale
    rows are verified at value level against the last covered hour: the
    ``tail_ffill_repeat`` label is only assigned when the uncovered values
    measurably repeat the last covered value (exact equality of the stored
    values), never inferred from the flag alone."""
    origins = np.asarray(origins)
    covered = np.asarray(covered, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    stale = _stale_flags(origins, run_unix, stale_seconds)
    rows, fills = [], []
    for i in range(len(origins)):
        covered_hours = int((covered[i] == 1.0).sum())
        row = {
            "origin_unix": int(origins[i]),
            "covered_hours": covered_hours,
            "uncovered_hours": 24 - covered_hours,
        }
        if covered_hours == 24:
            kind = "inconsistent_stale_but_fully_covered" if stale[i] else "none"
            row.update(fill_kind=kind, tail_repeat_verified=None)
        elif stale[i]:
            row.update(
                fill_kind="stale_median_fill",
                tail_repeat_verified=None,
                note="stale run: median-fill semantics, not value-verified here",
            )
            kind = "stale_median_fill"
        else:
            anchor = np.nonzero(covered[i] == 1.0)[0]
            if len(anchor) == 0:
                kind = "uncovered_no_anchor"
                row.update(fill_kind=kind, tail_repeat_verified=None)
            else:
                last = int(anchor[-1])
                uncovered = np.nonzero(covered[i] != 1.0)[0]
                tail = uncovered[uncovered > last]
                row.update(last_covered_hour=last, last_covered_value=float(values[i, last]))
                if len(tail) == 0:
                    kind = "gap_uncovered"
                    row.update(fill_kind=kind, tail_repeat_verified=None)
                else:
                    verified = bool(np.all(values[i, tail] == values[i, last]))
                    kind = "tail_ffill_repeat" if verified else "uncovered_non_repeat"
                    row.update(
                        fill_kind=kind,
                        tail_repeat_verified=verified,
                        tail_hours_after_last_covered=[int(t) for t in tail],
                    )
        rows.append(row)
        if covered_hours < 24:
            verified = row.get("tail_repeat_verified")
            suffix = "" if verified is None else f"; tail_repeat_verified={verified}"
            fills.append(
                f"origin {int(origins[i])}: covered {covered_hours}/24 ({kind}{suffix})"
            )
        elif kind == "inconsistent_stale_but_fully_covered":
            fills.append(
                f"origin {int(origins[i])}: stale run but covered=1 for all hours (inconsistent)"
            )
    return {
        "check": "coverage_24h_no_fill",
        "status": "fail" if fills else "pass",
        "fill_reasons": fills,
        "rows": rows,
    }


def check_publication_basis(run_unix, last_changed) -> dict:
    run_unix = np.asarray(run_unix)
    last_changed = np.asarray(last_changed)
    alias_equal = bool(np.array_equal(run_unix, last_changed))
    return {
        "check": "publication_basis_archive_proxy",
        "status": "pass" if alias_equal else "fail",
        "alias_elementwise_equal": alias_equal,
        "publication_basis": PUBLICATION_BASIS,
        "first_seen_claimed": False,
        "note": (
            "MMSDM monthly-archive LASTCHANGED fetched at NPZ build time "
            "(2026-10-10); cannot prove historical first-seen and cannot "
            "exclude backdated revisions"
        ),
    }


# Protocol S4: the engineering gate covers these structural checks only;
# coverage fills in the data feed quick_veto and never gate engineering.
ENGINEERING_GATE_CHECKS = (
    "schema",
    "units_plausibility_aud_per_mwh",
    "target_time_alignment",
    "run_vintage_asof",
    "publication_basis_archive_proxy",
)


def assemble_verdict(*, checks, budget, preflight, guard_violations) -> dict:
    statuses = {check["check"]: check["status"] for check in checks}
    tests_ok = preflight is not None and preflight.get("returncode") == 0
    budget_ok = budget is not None and budget.get("within_budget") is True
    guards_ok = guard_violations == []
    checks_ok = all(
        statuses[name] == "pass"
        for name in ENGINEERING_GATE_CHECKS
        if name in statuses
    ) and all(name in statuses for name in ENGINEERING_GATE_CHECKS)
    veto_reasons = []
    for check in checks:
        veto_reasons.extend(check.get("veto_reasons", []))
        veto_reasons.extend(check.get("fill_reasons", []))
    engineering_pass = tests_ok and budget_ok and guards_ok and checks_ok
    problems = []
    if not tests_ok:
        problems.append("preflight tests missing or failed (nonzero returncode)")
    if not budget_ok:
        problems.append("budget exceeded or missing")
    if not guards_ok:
        problems.extend(f"guard violation: {v}" for v in guard_violations)
    if not checks_ok:
        problems.extend(
            f"{name} failed"
            for name in ENGINEERING_GATE_CHECKS
            if statuses.get(name) != "pass"
        )
    verdict = {
        "audit": "g0a_predispatch_rrp_asof_preaudit",
        "engineering": "ENGINEERING_PASS" if engineering_pass else "ENGINEERING_FAIL",
        "engineering_problems": problems,
        "strict_asof": STRICT_ASOF_STATUS,
        "strict_block_reasons": list(STRICT_BLOCK_REASONS),
        "publication_basis": PUBLICATION_BASIS,
        "first_seen_claimed": False,
        "quick_veto": {
            "status": "TRIGGERED" if veto_reasons else "NOT_TRIGGERED",
            "reasons": veto_reasons,
        },
        "checks_status": statuses,
        "scope_note": (
            "NPZ alone can quick-veto only; strict G0 PASS is not grantable "
            "from this artifact; 72h live collection not started"
        ),
    }
    assert_never_strict_pass(verdict)
    return verdict


def assert_never_strict_pass(verdict: dict) -> None:
    if verdict.get("strict_asof") != STRICT_ASOF_STATUS:
        raise StrictPassNotGrantable(
            "strict as-of PASS cannot be granted from this artifact class"
        )
