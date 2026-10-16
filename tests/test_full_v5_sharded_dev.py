"""Round-7/9 tests: OS run locks, locked budget lifecycle, live budget
enforcement, trusted test hashes, and the region-sharded dev interface.

Round-9 additions/fixes on top of the Round-7 coverage:
- the budget lifecycle is owned by the execution locks: a refused duplicate
  CLI worker must not change one byte of the active owner's ledger (REAL
  two-process regression with a kill -9 owner and a successor worker);
- tests that spawn CLI children install the SAME small-window schema into the
  synthetic world and run every child through tests/_v5_child_entry.py, which
  verifies the world import path, effective windows, SCHEMA_HASH and library
  versions BEFORE executing, and dumps the actually-imported closure (incl.
  src/dualfield) afterwards;
- child output goes to independent realtime log files (never PIPEs), and every
  batch reaps/terminates/kills ALL children on failure or timeout (stub test
  with a high-output child and a deadline-exceeding child);
- serial/sharded equivalence is asserted per-marker integrity plus semantic
  content (predictions, metrics, gates, bundles minus created_at) — never on
  bundle hashes that merely encode execution timestamps;
- ONE merged real LP/LightGBM integration covers three concurrent regions,
  test-byte removal (no-read), duplicate refusal, finalize, and the serial
  comparison run in a single bounded world.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from forecasting.full_v5 import data, experiment, report, schema  # noqa: E402
from test_full_v5_protocol import (  # noqa: E402
    MINI_CUTOFFS, MINI_SEGMENTS, SMALL_CUTOFFS, SMALL_SEGMENTS,
    _apply_window_schema, _small_schema, _write_tests_passed, build_world)

REPO_ROOT = TESTS_DIR.parent
REGIONS = ("NSW1", "QLD1", "TAS1")
NSW1_CSV_SHA256 = "454ac6013d13f351c6c1edf5dde832b66a441639a9f35414c8de0d23a8c8c08f"
THREAD_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",  # child log files update in real time, not at flush
}
CHILD_ENTRY = TESTS_DIR / "_v5_child_entry.py"
DEP_MODULES = ("numpy", "scipy", "pandas", "lightgbm", "yaml")

_LOCK_CHILD = """
import sys, time
from pathlib import Path
from forecasting.full_v5 import experiment
run_root, barrier, result = sys.argv[1], sys.argv[2], sys.argv[3]
while not Path(barrier).exists():
    time.sleep(0.005)
try:
    lock = experiment._acquire_final_lock(experiment._run_paths(run_root))
except PermissionError:
    Path(result).write_text("refused")
    sys.exit(2)
Path(result).write_text("acquired")
time.sleep(1.0)
lock.release()
"""

# Round-9 fix 1 regression: a live region owner (real flock + real budget
# records) that is SIGKILLed mid-flight, leaving live usage behind.
_BUDGET_OWNER_CHILD = """
import os
import sys
import time
from pathlib import Path
from forecasting.full_v5 import experiment
run_root, region, flag_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
paths = experiment._run_paths(run_root)
lock = experiment._Flock(paths["development"] / "regions" / region / "region.lock",
                         f"dev-region {region}").acquire()
(flag_dir / "lock.flag").write_text("ok")
experiment._usage_begin(paths, "dev", region=region)
(flag_dir / "live.flag").write_text(str(os.getpid()))
time.sleep(1.2)
experiment._usage_checkpoint(paths)
budget = experiment._read_budget(Path(experiment._USAGE["path"]))
(flag_dir / "cpu.txt").write_text(str(float(budget["dev"].get("live_cpu_seconds", 0.0))))
time.sleep(600)  # parent SIGKILLs here: live keys survive, kernel frees the lock
"""

_BUDGET_SUCCESSOR_CHILD = """
import sys
from pathlib import Path
from forecasting.full_v5 import experiment
run_root, region = sys.argv[1], sys.argv[2]
paths = experiment._run_paths(run_root)
lock = experiment._Flock(paths["development"] / "regions" / region / "region.lock",
                         f"dev-region {region}").acquire()
try:
    experiment._usage_begin(paths, "dev", region=region)  # folds killed owner's live usage
    inherited = float(experiment._read_budget(Path(experiment._USAGE["path"]))["dev"]["cpu_seconds"])
    print(f"SUCCESSOR_INHERITED={inherited}", flush=True)
    experiment._usage_checkpoint(paths)
finally:
    experiment._usage_finish(paths)
    lock.release()
final = experiment._read_budget(paths["development"] / "regions" / region / "budget.json")["dev"]
print(f"SUCCESSOR_FINAL={float(final['cpu_seconds'])}", flush=True)
assert "live_pid" not in final, final
"""

_NOISY_CHILD = """
import time
for i in range(200000):
    print(f"{i}:" + "x" * 200, flush=True)
    time.sleep(0.001)
"""

_SLEEPER_CHILD = "import time; time.sleep(600)"


@pytest.fixture()
def patched_schema(monkeypatch):
    _apply_window_schema(monkeypatch, MINI_SEGMENTS, MINI_CUTOFFS)
    return schema


def _child_env(expect=None, snapshot=None):
    env = dict(os.environ)
    env.update(THREAD_ENV)
    if expect is not None:
        env["V5_CHILD_EXPECT"] = json.dumps(expect)
        env["V5_CHILD_SNAPSHOT"] = str(snapshot)
    return env


def _child_expect(cli_args, snapshot):
    """Everything a child must agree on with the parent process (Round-9
    fix 2): the installed window, the recomputed schema hash and the library
    versions the parent itself is running."""
    return dict(
        segments={k: list(v) for k, v in schema.SEGMENTS.items()},
        fit_cutoffs=dict(schema.FIT_CUTOFFS),
        schema_hash=schema.SCHEMA_HASH,
        dep_versions={name: __import__(name).__version__ for name in DEP_MODULES},
        cli_args=[str(a) for a in cli_args])


def _launch_logged(world, log_dir, name, cli_args, expect, snapshot):
    """Launch one CLI child through the verifying entry; all output goes to an
    independent line-buffered log file — never a PIPE — so a chatty child can
    never block and the log is readable in real time (Round-9 fix 3)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    handle = log_path.open("w", buffering=1)
    proc = subprocess.Popen([sys.executable, str(CHILD_ENTRY), *map(str, cli_args)],
                            cwd=str(world), env=_child_env(expect, snapshot),
                            stdout=handle, stderr=subprocess.STDOUT)
    proc.log_handle = handle
    proc.log_path = log_path
    return proc


def _log_tail(path, limit=3000):
    return Path(path).read_text(errors="replace")[-limit:]


def _reap_logged(procs, timeout):
    """Wait for every child under one shared deadline; on failure or timeout
    terminate -> kill -> wait ALL of them in finally so no child is left
    behind (Round-9 fix 3).  Raises with log tails if any child failed."""
    deadline = time.time() + timeout
    failures = []
    try:
        for name, proc in procs.items():
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                failures.append(f"{name}: deadline exceeded after {timeout}s (log {proc.log_path})")
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            proc.log_handle.close()
    for name, proc in procs.items():
        if proc.returncode != 0:
            failures.append(f"{name}: exit {proc.returncode}\n{_log_tail(proc.log_path)}")
    if failures:
        raise AssertionError("\n".join(failures))


def _run_cli(world, log_dir, name, cli_args, timeout=1800):
    """Run one CLI child to completion through the verifying entry; returns
    (returncode, log_path) without asserting on the code."""
    snapshot = log_dir / f"{name}.import_snapshot.json"
    proc = _launch_logged(world, log_dir, name, cli_args,
                          _child_expect(cli_args, snapshot), snapshot)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        proc.log_handle.close()
    return proc.returncode, proc.log_path


# --- Round-6 High: the run lock ---------------------------------------------
def test_final_lock_two_process_concurrent_start(tmp_path):
    """A REAL interleaving race: two processes spin on a start barrier and
    then both attempt to acquire the final-test run lock.  Exactly one
    proceeds; the other refuses with PermissionError."""
    run_root = tmp_path / "run"
    barrier = tmp_path / "barrier"
    results = [tmp_path / f"result_{i}" for i in range(2)]
    procs = [subprocess.Popen(
        [sys.executable, "-c", _LOCK_CHILD, str(run_root), str(barrier), str(result)],
        cwd=str(REPO_ROOT), env=_child_env(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for result in results]
    time.sleep(0.8)  # both children are now spinning on the barrier
    barrier.touch()  # simultaneous release: the race window is real
    for proc in procs:
        assert proc.wait(timeout=60) in (0, 2)
    texts = sorted(result.read_text() for result in results if result.exists())
    assert texts == ["acquired", "refused"], texts
    # the loser exited with the refusal code, the winner cleanly released
    assert sorted(p.returncode for p in procs) == [0, 2]


def test_region_lock_duplicate_worker_refuses(tmp_path):
    """dev-region refuses while another holder owns the region lock."""
    paths = experiment._run_paths(tmp_path / "run")
    holder = experiment._Flock(
        paths["development"] / "regions" / "NSW1" / "region.lock", "dev-region NSW1").acquire()
    try:
        with pytest.raises(PermissionError, match="held"):
            experiment._Flock(
                paths["development"] / "regions" / "NSW1" / "region.lock",
                "dev-region NSW1").acquire()
        # different regions use different lock files and never block each other
        other = experiment._Flock(
            paths["development"] / "regions" / "QLD1" / "region.lock", "dev-region QLD1").acquire()
        other.release()
    finally:
        holder.release()


# --- Round-9 fix 1: the budget lifecycle is owned by the execution locks -----
def test_budget_refused_worker_and_killed_owner_ledger(tmp_path, monkeypatch, patched_schema):
    """REAL two-process regression (no mocked bodies): while a live owner
    holds the region lock with an active budget record, a duplicate CLI worker
    is refused WITHOUT changing one byte of the ledger; after the owner is
    SIGKILLed (lock freed by the kernel, live keys left behind), the successor
    worker inherits the accumulated ledger and loses nothing."""
    _small_schema(monkeypatch)
    world = build_world(tmp_path, structured=True, segments=SMALL_SEGMENTS, cutoffs=SMALL_CUTOFFS)
    monkeypatch.chdir(world)
    run_root = world / "run"
    assert experiment.prepare(str(world / "configs" / "full_v5" / "experiment.yaml"), str(run_root)) == 0
    _write_tests_passed(run_root)
    ledger = run_root / "development" / "regions" / "NSW1" / "budget.json"
    flags = tmp_path / "flags"
    flags.mkdir(parents=True, exist_ok=True)
    owner_log = tmp_path / "owner.log"

    owner_handle = owner_log.open("w", buffering=1)
    owner = subprocess.Popen(
        [sys.executable, "-c", _BUDGET_OWNER_CHILD, str(run_root), "NSW1", str(flags)],
        cwd=str(world), env=_child_env(), stdout=owner_handle, stderr=subprocess.STDOUT)
    try:
        for flag in ("lock.flag", "live.flag", "cpu.txt"):
            for _ in range(600):  # up to ~30s
                if (flags / flag).exists():
                    break
                time.sleep(0.05)
            else:
                owner.kill()
                owner.wait(timeout=30)
                raise AssertionError(f"owner never reached {flag}; log: {_log_tail(owner_log)}")
        owner_cpu = float((flags / "cpu.txt").read_text())
        live_before = json.loads(ledger.read_text())["dev"]
        assert live_before["live_pid"] == owner.pid
        assert live_before["live_cpu_seconds"] >= owner_cpu

        # The duplicate worker runs the REAL CLI: identity checks pass, the
        # region lock refuses it — and the ledger must not change one byte.
        ledger_before = ledger.read_bytes()
        code, log = _run_cli(world, tmp_path / "logs", "dup_worker",
                             ["dev-region", "--run-root", str(run_root), "--region", "NSW1"],
                             timeout=600)
        assert code != 0, _log_tail(log)
        tail = _log_tail(log)
        assert "held" in tail or "refusing" in tail, tail
        assert ledger.read_bytes() == ledger_before, "refused worker modified the active owner's ledger"

        # SIGKILL the owner: kernel releases the flock; live usage survives.
        owner.kill()
        owner.wait(timeout=30)
        after_kill = json.loads(ledger.read_text())["dev"]
        assert after_kill["live_pid"] == owner.pid
        assert after_kill["live_cpu_seconds"] >= owner_cpu
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=30)
        owner_handle.close()

    # Successor startup folds the killed owner's live usage into the ledger
    # and finishes cleanly under the lock; nothing accumulated is lost.
    successor = subprocess.run(
        [sys.executable, "-c", _BUDGET_SUCCESSOR_CHILD, str(run_root), "NSW1"],
        cwd=str(world), env=_child_env(), capture_output=True, text=True, timeout=120)
    assert successor.returncode == 0, successor.stdout + successor.stderr
    inherited = float(next(line.split("=")[1] for line in successor.stdout.splitlines()
                           if line.startswith("SUCCESSOR_INHERITED=")))
    final = float(next(line.split("=")[1] for line in successor.stdout.splitlines()
                       if line.startswith("SUCCESSOR_FINAL=")))
    assert inherited >= owner_cpu, (inherited, owner_cpu)
    assert final >= inherited
    assert "live_pid" not in json.loads(ledger.read_text())["dev"]


# --- Round-9 fix 3: logged children are always reaped ------------------------
def test_child_logging_and_reclamation_high_output_and_timeout(tmp_path):
    """A high-output child and a deadline-exceeding child: output lands in
    independent realtime logs (never truncated by a full pipe), and the batch
    reaper terminates/kills/waits EVERY child on timeout — none left behind."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    scripts = {}
    for name, body in (("noisy", _NOISY_CHILD), ("sleeper", _SLEEPER_CHILD)):
        script = tmp_path / f"{name}.py"
        script.write_text(body)
        scripts[name] = script
    procs = {}
    for name, script in scripts.items():
        log_path = log_dir / f"{name}.log"
        handle = log_path.open("w", buffering=1)
        proc = subprocess.Popen([sys.executable, str(script)], cwd=str(tmp_path),
                                stdout=handle, stderr=subprocess.STDOUT)
        proc.log_handle, proc.log_path = handle, log_path
        procs[name] = proc
    started = time.time()
    with pytest.raises(AssertionError, match="deadline exceeded"):
        _reap_logged(procs, timeout=4)
    elapsed = time.time() - started
    for name, proc in procs.items():
        assert proc.poll() is not None, f"{name} left behind"
    noisy_lines = (log_dir / "noisy.log").read_text().splitlines()
    assert len(noisy_lines) > 500, len(noisy_lines)  # far beyond any pipe buffer
    assert elapsed < 30, elapsed


# --- Round-6 M1: test-read guards and trusted test hashes -------------------
def test_precomputed_baseline_test_load_guarded_before_file_access(tmp_path):
    """BaselineSource._load_precomputed refuses the test split BEFORE any file
    handle is created: the npz does not exist at all, so an unguarded loader
    would raise FileNotFoundError instead of PermissionError."""
    config = {"v5_calibrated_history": {
        "baseline": {"mode": "precomputed", "run_root": str(tmp_path / "baseline")}}}
    source = data.BaselineSource(config, tmp_path)
    data.set_stage("dev")
    try:
        with pytest.raises(PermissionError):
            source._load_precomputed("NSW1", "test")
        with pytest.raises(PermissionError):
            source.test_quantiles("NSW1")
    finally:
        data.set_stage("prepare")


def test_freeze_records_trusted_hashes_without_reading_test_bytes(
        tmp_path, monkeypatch, patched_schema):
    """freeze succeeds with EVERY test-side artefact removed: it records only
    the trusted pre-bound manifest hashes; actual bytes are verified by
    final-test (checked here via the recorded values matching on restore)."""
    world = build_world(tmp_path, structured=True)
    monkeypatch.chdir(world)
    run_root = world / "run"
    assert experiment.prepare(str(world / "configs" / "full_v5" / "experiment.yaml"), str(run_root)) == 0
    _write_tests_passed(run_root)
    # A manifest mutated after prepare must make freeze refuse.
    manifest = world / "baseline" / "test_source_hashes.json"
    original_manifest = manifest.read_bytes()
    manifest.write_bytes(original_manifest + b"\n")
    assert experiment.dev(str(run_root)) == 0
    assert json.loads((run_root / "development" / "gates.json").read_text())["passed"]
    with pytest.raises(ValueError, match="changed after prepare"):
        experiment.freeze(str(run_root))
    manifest.write_bytes(original_manifest)
    # Remove every test-side artefact: freeze must not open any of them.
    v4_test = world / "v4" / "NSW1_test.npz"
    baseline_test = world / "baseline" / "NSW1_test.npz"
    cache_test = world / "data_cache" / "NSW1" / "test"
    backups = {p: p.read_bytes() for p in (v4_test, baseline_test)}
    cache_backup = world / "cache_test_backup"
    shutil.copytree(cache_test, cache_backup)
    v4_test.unlink()
    baseline_test.unlink()
    shutil.rmtree(cache_test)
    assert experiment.freeze(str(run_root)) == 0
    freeze_payload = json.loads((run_root / "freeze.json").read_text())
    trusted = json.loads((world / "baseline" / "test_source_hashes.json").read_text())
    assert freeze_payload["test_sources"]["baseline_test"]["NSW1"] == trusted["NSW1"]
    assert freeze_payload["test_sources"]["v4_test"]["NSW1"] == \
        json.loads((world / "v4" / "test_source_hashes.json").read_text())["NSW1"]
    assert freeze_payload["test_sources"]["cache"]["NSW1"]
    # Restore the bytes; the recorded trusted hashes must match the actual
    # files (this exact comparison is what authorized final-test performs).
    v4_test.write_bytes(backups[v4_test])
    baseline_test.write_bytes(backups[baseline_test])
    shutil.move(str(cache_backup), str(cache_test))
    config = yaml.safe_load((run_root / "config.yaml").read_text())
    exp_cfg = config["v5_calibrated_history"]
    experiment._verify_test_sources(experiment._run_paths(run_root), world, config,
                                    exp_cfg, freeze_payload)


# --- Round-6 M4: live budget enforcement -------------------------------------
def test_budget_positive_midrun_overrun(tmp_path, monkeypatch, patched_schema):
    """A POSITIVE budget must stop the stage mid-run once ledger + current
    process consumption reach the ceiling; usage is persisted in the region
    record that owns the region lock (Round-9 fix 1)."""
    _small_schema(monkeypatch)
    world = build_world(tmp_path, structured=True)
    monkeypatch.chdir(world)
    config_path = world / "configs" / "full_v5" / "experiment.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["v5_calibrated_history"]["budget"]["dev_core_hours"] = 0.01  # 36 CPU s
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    run_root = world / "run"
    assert experiment.prepare(str(config_path), str(run_root)) == 0
    _write_tests_passed(run_root)
    calls = {"n": 0}

    def fake_process_cpu():
        calls["n"] += 1
        return 7200.0 * calls["n"]

    monkeypatch.setattr(experiment, "_process_cpu", fake_process_cpu)
    with pytest.raises(experiment.BudgetExceeded):
        experiment.main(["dev", "--run-root", str(run_root)])
    budget = json.loads((run_root / "development" / "regions" / "NSW1" / "budget.json").read_text())
    assert budget["dev"]["cpu_seconds"] >= 7200.0      # this attempt booked
    assert "live_pid" not in budget["dev"]             # live keys folded away
    state = json.loads((run_root / "state.json").read_text())
    assert state["stage"] == "prepared" and state["status"] == "interrupted"
    commands = (run_root / "development" / "regions" / "NSW1" / "commands.jsonl").read_text()
    assert "BudgetExceeded" in commands.strip().splitlines()[-1]
    assert not (run_root / "development" / "gates.json").exists()


def test_keyboard_interrupt_books_usage_and_state(tmp_path, monkeypatch, patched_schema):
    """Catchable BaseException interruptions (KeyboardInterrupt) are accounted:
    consumption lands in the region ledger, state is marked interrupted."""
    _small_schema(monkeypatch)
    world = build_world(tmp_path, structured=True)
    monkeypatch.chdir(world)
    run_root = world / "run"
    assert experiment.prepare(str(world / "configs" / "full_v5" / "experiment.yaml"), str(run_root)) == 0
    _write_tests_passed(run_root)
    original_marker = experiment._write_done_marker

    def interrupting_marker(path, identity, outputs):
        if Path(path).name.startswith("done_trees_T_roll"):
            raise KeyboardInterrupt("simulated operator interrupt")
        return original_marker(path, identity, outputs)

    monkeypatch.setattr(experiment, "_write_done_marker", interrupting_marker)
    with pytest.raises(KeyboardInterrupt):
        experiment.main(["dev", "--run-root", str(run_root)])
    budget = json.loads((run_root / "development" / "regions" / "NSW1" / "budget.json").read_text())
    assert budget["dev"]["cpu_seconds"] > 0.0
    assert "live_pid" not in budget["dev"]
    state = json.loads((run_root / "state.json").read_text())
    assert state["status"] == "interrupted" and "KeyboardInterrupt" in state["last_error"]
    commands = (run_root / "development" / "regions" / "NSW1" / "commands.jsonl").read_text()
    assert "KeyboardInterrupt" in commands.strip().splitlines()[-1]


# --- Round-6 L1/L2: conditional equivalence wording --------------------------
def test_equivalence_wording_conditional_on_numeric_threshold():
    within = {"NSW1": {"v5_vs_controls_max_abs_diff": {"C_roll": 3e-12, "C_recent": 0.0}}}
    lines = report._equivalence_lines(within)
    assert len(lines) == 1
    assert "数值精度内等价" in lines[0]
    assert "存在实质差异" not in lines[0]
    mixed = {"NSW1": {"v5_vs_controls_max_abs_diff": {"C_roll": 0.5, "C_recent": 2e-12}}}
    line = report._equivalence_lines(mixed)[0]
    assert "数值精度内等价" in line and "存在实质差异" in line
    assert f"{report.EQUIVALENCE_TOLERANCE:g}" in line  # threshold is explicit


def test_identity_doc_carries_verified_nsw1_hash():
    doc = (REPO_ROOT / "docs" / "full_v5_baseline_identity.md").read_text()
    assert NSW1_CSV_SHA256 in doc
    # the superseded wrong hash (…f351c6e1 variant) must be gone; note the
    # correct hash's own prefix is …c6c1, so the checked fragment is the
    # wrong one's distinguishing prefix, not the correct hash's prefix
    assert "454ac6013d13f351c6e1" not in doc


# --- Round-9 fix 5: ONE merged real integration ------------------------------
def test_sharded_three_regions_finalize_and_serial_equivalence(tmp_path, monkeypatch, patched_schema):
    """Merged real LP/LightGBM integration on one synthetic world (Round-9):
    duplicate-worker refusal with a byte-stable ledger, three concurrent
    region shards with test bytes REMOVED, dev-finalize, and a concurrent
    serial CLI comparison run — then per-marker integrity plus semantic
    equivalence (predictions/metrics/gates/bundles minus created_at)."""
    _small_schema(monkeypatch)
    world = build_world(tmp_path / "world", structured=True, regions=REGIONS,
                        segments=SMALL_SEGMENTS, cutoffs=SMALL_CUTOFFS)
    monkeypatch.chdir(world)
    log_dir = world / "run_child_logs"
    config_path = world / "configs" / "full_v5" / "experiment.yaml"
    run_sharded = world / "run_sharded"
    assert experiment.prepare(str(config_path), str(run_sharded)) == 0
    _write_tests_passed(run_sharded)
    run_serial = world / "run_serial"
    shutil.copytree(run_sharded, run_serial)  # same identity, same world
    # Prove no test access: remove every test-side artefact before any dev run.
    test_side = [world / "v4" / f"{r}_test.npz" for r in REGIONS]
    test_side += [world / "baseline" / f"{r}_test.npz" for r in REGIONS]
    backups = {p: p.read_bytes() for p in test_side}
    cache_backups = {}
    for region in REGIONS:
        cache_backups[region] = world / f"cache_test_backup_{region}"
        shutil.copytree(world / "data_cache" / region / "test", cache_backups[region])
        shutil.rmtree(world / "data_cache" / region / "test")
    for path in test_side:
        path.unlink()

    # Duplicate same-region worker: lock held; the CLI must refuse WITHOUT
    # touching the active owner's budget ledger (Round-9 fix 1, real CLI).
    ledger = run_sharded / "development" / "regions" / "NSW1" / "budget.json"
    holder = experiment._Flock(run_sharded / "development" / "regions" / "NSW1" / "region.lock",
                               "dev-region NSW1").acquire()
    try:
        ledger_before = ledger.read_bytes() if ledger.exists() else b""
        code, log = _run_cli(world, log_dir, "dup_nsw1",
                             ["dev-region", "--run-root", str(run_sharded), "--region", "NSW1"],
                             timeout=600)
        assert code != 0, _log_tail(log)
        tail = _log_tail(log)
        assert "held" in tail or "refusing" in tail, tail
        assert (ledger.read_bytes() if ledger.exists() else b"") == ledger_before
    finally:
        holder.release()

    # Three concurrent region shards AND the serial comparison run, all real
    # CLI children with independent realtime logs, reaped together (fix 3).
    batch = {}
    for region in REGIONS:
        cli = ["dev-region", "--run-root", str(run_sharded), "--region", region]
        snapshot = log_dir / f"shard_{region}.import_snapshot.json"
        batch[f"shard_{region}"] = _launch_logged(world, log_dir, f"shard_{region}", cli,
                                                  _child_expect(cli, snapshot), snapshot)
    serial_cli = ["dev", "--run-root", str(run_serial)]
    serial_snapshot = log_dir / "serial.import_snapshot.json"
    batch["serial"] = _launch_logged(world, log_dir, "serial", serial_cli,
                                     _child_expect(serial_cli, serial_snapshot), serial_snapshot)
    started = time.time()
    _reap_logged(batch, timeout=3600)
    dev_wall = time.time() - started
    # Child-side proof (fix 2): every child imported the world's own closure —
    # including src.dualfield — with the parent's dependency versions.
    parent_deps = _child_expect([], serial_snapshot)["dep_versions"]
    for name in batch:
        snapshot_payload = json.loads((log_dir / f"{name}.import_snapshot.json").read_text())
        world_imports = snapshot_payload["imported_from_world"]
        assert any(m.startswith("forecasting.full_v5.") for m in world_imports), name
        assert any(m.startswith("src.dualfield") for m in world_imports), name
        assert snapshot_payload["dep_versions"] == parent_deps, name
    # Shards wrote only region-scoped records: no shared run-root artefacts.
    development = run_sharded / "development"
    assert not (development / "metrics.json").exists()
    assert not (development / "gates.json").exists()
    assert not (run_sharded / "budget.json").exists()
    assert not (run_sharded / "commands.jsonl").exists()
    assert json.loads((run_sharded / "state.json").read_text())["stage"] == "prepared"
    for region in REGIONS:
        assert (development / f"done_{region}.json").exists()
        bundle = json.loads((development / "regions" / region / "bundle.json").read_text())
        assert bundle["region"] == region and bundle["identity_digest"]
        region_budget = json.loads((development / "regions" / region / "budget.json").read_text())
        assert region_budget["dev"]["cpu_seconds"] >= 0.0
        assert "live_pid" not in region_budget["dev"]  # every owner finished in-lock
        commands = (development / "regions" / region / "commands.jsonl").read_text()
        assert '"dev-region"' in commands

    finalize_code, finalize_log = _run_cli(world, log_dir, "finalize",
                                           ["dev-finalize", "--run-root", str(run_sharded)])
    assert finalize_code == 0, _log_tail(finalize_log)
    metrics = json.loads((development / "metrics.json").read_text())
    assert list(metrics["scored"].keys()) == list(REGIONS)  # canonical order
    gates = json.loads((development / "gates.json").read_text())
    assert gates["passed"] in (True, False)
    assert (run_sharded / "reports" / "dev_report.md").exists()
    state = json.loads((run_sharded / "state.json").read_text())
    assert state["stage"] in ("dev_passed", "dev_failed")
    budget = json.loads((run_sharded / "budget.json").read_text())
    assert sorted(budget["dev_regions"]) == sorted(REGIONS)
    assert budget["dev_total_cpu_seconds"] >= 0.0
    merged_commands = json.loads((development / "commands_by_region.json").read_text())
    assert list(merged_commands.keys()) == list(REGIONS)
    serial_state = json.loads((run_serial / "state.json").read_text())
    assert serial_state["stage"] in ("dev_passed", "dev_failed")
    # No test access happened: restoring the bytes keeps everything consistent.
    for path, payload in backups.items():
        path.write_bytes(payload)
    for region in REGIONS:
        shutil.move(str(cache_backups[region]), str(world / "data_cache" / region / "test"))

    # Round-9 fix 4: verify EACH run's markers independently (integrity binds
    # every output file hash), then compare semantic content — never bundle
    # hashes that merely encode created_at.
    for run_root in (run_sharded, run_serial):
        identity = data.read_json(run_root / "identity" / "identity.json")
        markers = sorted((run_root / "development").glob("done_*.json"))
        assert len(markers) >= len(REGIONS) + 1  # region + tree markers
        for marker in markers:
            assert experiment._verify_done_marker(marker, identity), marker
    for region in REGIONS:
        bundle_sharded = json.loads((run_sharded / "development" / "regions" / region / "bundle.json").read_text())
        bundle_serial = json.loads((run_serial / "development" / "regions" / region / "bundle.json").read_text())
        bundle_sharded.pop("created_at", None)
        bundle_serial.pop("created_at", None)
        assert bundle_sharded == bundle_serial, region
    serial_metrics = json.loads((run_serial / "development" / "metrics.json").read_text())
    assert metrics == serial_metrics
    serial_gates = json.loads((run_serial / "development" / "gates.json").read_text())
    gates.pop("evaluated_at", None)
    serial_gates.pop("evaluated_at", None)
    assert gates == serial_gates
    assert list(serial_metrics["scored"].keys()) == list(REGIONS)
    # Real numerical equivalence: every method's predictions match bit-exactly.
    for model_dir in sorted((development / "predictions").iterdir()):
        for npz in sorted(model_dir.glob("*.npz")):
            other = run_serial / "development" / "predictions" / model_dir.name / npz.name
            a = np.load(npz)
            b = np.load(other)
            assert np.array_equal(a["quantiles"], b["quantiles"]), npz.name
            assert np.array_equal(a["origin_unix"], b["origin_unix"])
            assert np.array_equal(a["actual"], b["actual"])
    print(f"concurrent shards + serial comparison wall time: {dev_wall:.1f}s")


# --- retained machinery coverage (unchanged logic; see round-9 report) -------
def test_dev_finalize_missing_region_refuses(tmp_path, monkeypatch, patched_schema):
    _small_schema(monkeypatch)
    world = build_world(tmp_path / "world", structured=True, regions=REGIONS)
    monkeypatch.chdir(world)
    run_root = world / "run"
    assert experiment.prepare(str(world / "configs" / "full_v5" / "experiment.yaml"), str(run_root)) == 0
    _write_tests_passed(run_root)
    assert experiment.dev_region(str(run_root), "NSW1") == 0
    assert experiment.dev_region(str(run_root), "QLD1") == 0
    with pytest.raises(ValueError, match="TAS1.*incomplete|incomplete.*TAS1"):
        experiment.dev_finalize(str(run_root))
    assert not (run_root / "development" / "gates.json").exists()
    assert not (run_root / "development" / "metrics.json").exists()
    state = json.loads((run_root / "state.json").read_text())
    assert state["stage"] == "prepared"  # not finalized


def test_dev_region_interrupted_resume_and_tamper(tmp_path, monkeypatch, patched_schema):
    """An interrupted region shard resumes and completes; a tampered persisted
    artefact is refused."""
    _small_schema(monkeypatch)
    world = build_world(tmp_path, structured=True)
    monkeypatch.chdir(world)
    run_root = world / "run"
    assert experiment.prepare(str(world / "configs" / "full_v5" / "experiment.yaml"), str(run_root)) == 0
    _write_tests_passed(run_root)
    original_marker = experiment._write_done_marker

    def crashing_marker(path, identity, outputs):
        if Path(path).name == "done_NSW1.json":
            raise RuntimeError("simulated shard interruption before marker")
        return original_marker(path, identity, outputs)

    monkeypatch.setattr(experiment, "_write_done_marker", crashing_marker)
    with pytest.raises(RuntimeError, match="simulated shard interruption"):
        experiment.dev_region(str(run_root), "NSW1")
    assert (run_root / "development" / "predictions" / "B" / "NSW1_v1.npz").exists()
    assert not (run_root / "development" / "done_NSW1.json").exists()
    monkeypatch.setattr(experiment, "_write_done_marker", original_marker)
    assert experiment.dev_region(str(run_root), "NSW1") == 0  # resume completes
    assert (run_root / "development" / "done_NSW1.json").exists()
    # Tampering a persisted development model must make resume refuse.
    model = run_root / "development" / "models" / "C_roll_NSW1.json"
    payload = json.loads(model.read_text())
    payload["params"]["0"][0][0] = 0.5
    model.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="outputs changed or missing"):
        experiment.dev_region(str(run_root), "NSW1")
