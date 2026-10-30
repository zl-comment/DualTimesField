"""Atomic append-only evidence publication for the M1 contract.

A result is visible ONLY when an immutable final directory contains a valid
COMMIT.json whose manifest hash and full bidirectional inventory reverify.  The
publisher: verifies the SEMANTIC closure (budget chain/fold/caps, deadline,
as-of origin, input/tool identity) BEFORE writing anything; takes an exclusive
per-run lock and refuses a committed identity; creates a same-filesystem O_EXCL
staging directory; writes every payload and fsyncs it; generates a per-payload
SHA-256/size manifest; re-verifies everything on disk; writes COMMIT.json in
staging; then atomically renames the complete directory. Post-commit durability
or lock-cleanup errors cannot turn an already visible commit into a failure.
Failure records use a separate O_EXCL directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Mapping

from . import m1_budget, m1_contract
from .execution import DirectoryCommit, call_with_deadline, publish_directory

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_CLOSURE_KEYS = frozenset({
    "origin_unix", "deadline_unix", "clock_final", "budget_fold", "budget_caps",
    "budget_chain_ok", "tool_identity_sha256", "input_hash",
    "canonical_model_input_sha256"})


class PublicationError(Exception):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_file(path: Path) -> None:
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _fsync_dir(path: Path) -> None:
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _write_exclusive(path: Path, payload: bytes) -> None:
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(handle, payload)
        os.fsync(handle)
    finally:
        os.close(handle)


def verify_publication_closure(closure: Mapping, payloads: Mapping[str, bytes]) -> dict:
    """Semantic gate BEFORE any write: budget chain/fold/caps, deadline, identity.

    Byte closure alone is not success: a run may only be published when its own
    archived budget ledger re-verifies verbatim, its fold respects every cap
    (the per-body byte cap is checked per largest single body) with the FROZEN
    ledger caps — never caller-supplied caps (audit-3/H4) — the final clock is
    within the deadline, and the as-of origin/input/tool identities are carried
    and consistent with the archived result record AND recomputed from the
    archived model ledger and forecast trace (audit-3/H4: a hex-shaped hash is
    not an identity; tool/input identity must recompute from the archives).
    """
    if not isinstance(closure, Mapping) or set(closure) != set(_CLOSURE_KEYS):
        raise PublicationError(
            "publication closure must carry exactly "
            f"{sorted(_CLOSURE_KEYS)}, got {sorted(closure) if isinstance(closure, Mapping) else type(closure)!r}")
    for key in ("tool_identity_sha256", "input_hash",
                "canonical_model_input_sha256"):
        if not _HEX64_RE.fullmatch(str(closure[key] or "")):
            raise PublicationError(f"closure {key} is not a sha256 hex digest")
    origin = closure["origin_unix"]
    deadline = closure["deadline_unix"]
    if not isinstance(origin, int) or not isinstance(deadline, int):
        raise PublicationError("closure origin/deadline must be integers")
    if deadline != origin + m1_contract.DEADLINE_OFFSET:
        raise PublicationError(
            f"closure deadline {deadline} is not origin+{m1_contract.DEADLINE_OFFSET}")
    if closure["clock_final"] > deadline:
        raise PublicationError(
            f"closure clock {closure['clock_final']} missed the deadline {deadline}")
    if closure["budget_chain_ok"] is not True:
        raise PublicationError("closure budget chain did not verify")
    ledger_payload = payloads.get("budget_ledger.jsonl")
    if ledger_payload is None:
        raise PublicationError("a publication must archive its budget ledger")
    try:
        entries = [json.loads(line) for line in ledger_payload.splitlines()
                   if line.strip()]
        ledger = m1_budget.BudgetLedger.from_entries(entries)
    except (ValueError, json.JSONDecodeError) as exc:
        raise PublicationError(f"archived budget ledger is invalid: {exc}") from exc
    fold = {cls.name: value for cls, value in ledger.fold().items()}
    if fold != {str(k): int(v) for k, v in closure["budget_fold"].items()}:
        raise PublicationError("closure budget fold disagrees with the archive")
    # the caps are the FROZEN contract caps, and the closure must carry exactly
    # them: a self-reported cap vector can never authorize an over-cap fold
    caps = {cls.name: value for cls, value in m1_budget.CAPS.items()}
    if {str(k): int(v) for k, v in closure["budget_caps"].items()} != caps:
        raise PublicationError(
            "closure budget_caps are not the frozen contract caps "
            "(self-reported caps cannot authorize publication)")
    if set(fold) != set(caps):
        raise PublicationError("closure budget class set drifted")
    per_body = m1_budget.BudgetClass.SOURCE_BYTES_ONE_BODY.name
    for name, value in sorted(fold.items()):
        cap = caps[name]
        if name == per_body:
            if ledger.max_reserved(
                    m1_budget.BudgetClass.SOURCE_BYTES_ONE_BODY) > cap:
                raise PublicationError(
                    "a single body exceeded the per-body byte cap")
        elif value > cap:
            raise PublicationError(
                f"budget fold for {name} ({value}) exceeds its cap ({cap})")
    record = payloads.get("result.json")
    if record is None:
        raise PublicationError("a publication must archive its result record")
    try:
        archived = json.loads(record)
    except json.JSONDecodeError as exc:
        raise PublicationError(f"result.json is not JSON: {exc}") from exc
    if archived.get("origin_unix") != origin:
        raise PublicationError("closure as-of origin disagrees with result.json")
    if archived.get("input_hash") not in (None, closure["input_hash"]):
        raise PublicationError("closure input identity disagrees with result.json")
    if closure["input_hash"] != closure["canonical_model_input_sha256"]:
        raise PublicationError(
            "the canonical model input hash and the input hash are the same "
            "canonical serialization and must be equal")
    _verify_publication_identities(closure, payloads)
    return {"closure": dict(closure), "budget_fold": fold}


def _verify_publication_identities(closure: Mapping,
                                   payloads: Mapping[str, bytes]) -> None:
    """Recompute tool/input identity from the archived model ledger and trace.

    The model ledger is mandatory: the tool identity hash must recompute from
    its recorded tool identity object.  When a forecast was published, its
    bytes must hash to exactly one archived forecast trace entry, and that
    entry's input hash must equal the closure's canonical input hash; when no
    forecast bytes are archived, the trace must be empty.  A well-formed hex
    digest supplied by the caller is never sufficient (audit-3/H4).
    """
    model_payload = payloads.get("model_ledger.jsonl")
    if model_payload is None:
        raise PublicationError("a publication must archive its model ledger")
    try:
        model_ledger = json.loads(model_payload)
    except json.JSONDecodeError as exc:
        raise PublicationError(f"model_ledger.jsonl is not JSON: {exc}") from exc
    if not isinstance(model_ledger, dict) or not isinstance(
            model_ledger.get("trace"), list) or not isinstance(
                model_ledger.get("tool"), dict):
        raise PublicationError("model ledger is not a tool/trace record")
    recomputed_tool = hashlib.sha256(json.dumps(
        model_ledger["tool"], sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if recomputed_tool != closure["tool_identity_sha256"]:
        raise PublicationError(
            "closure tool identity does not recompute from the archived model ledger")
    forecast = payloads.get("forecast.bin")
    if forecast is None:
        if model_ledger["trace"]:
            raise PublicationError(
                "no forecast bytes archived while the model ledger traces a call")
        return
    forecast_digest = hashlib.sha256(forecast).hexdigest()
    matches = [entry for entry in model_ledger["trace"]
               if isinstance(entry, dict)
               and entry.get("output_sha256") == forecast_digest]
    if len(matches) != 1:
        raise PublicationError(
            "the published forecast bytes must hash to exactly one archived "
            f"forecast trace entry (matched {len(matches)})")
    if matches[0].get("input_hash") != closure["canonical_model_input_sha256"]:
        raise PublicationError(
            "the canonical model input hash does not recompute from the "
            "archived forecast trace")


def publish_run(root: Path | str, run_id: str, payloads: Mapping[str, bytes], *,
                identity: Mapping, closure: Mapping, clock=None,
                return_commit: bool = False) -> Path | DirectoryCommit:
    """Publish once; optionally expose post-commit durability without retrying."""
    started = time.monotonic()
    if clock is None:
        clock = lambda: closure["clock_final"] + time.monotonic() - started
    root = Path(root)
    if not _RUN_ID_RE.fullmatch(run_id or ""):
        raise PublicationError(f"run id {run_id!r} is not a bounded identifier")
    if not payloads:
        raise PublicationError("a run must carry at least one payload")
    for name in payloads:
        if not _RUN_ID_RE.fullmatch(name) or "/" in name or name in (
                "manifest.json", "COMMIT.json", "COMMIT.json.tmp"):
            raise PublicationError(f"payload name {name!r} is not allowed")
    semantic = verify_publication_closure(closure, payloads)
    runs = root / "runs"
    locks = root / "locks"
    staging_root = root / "staging"
    for directory in (runs, locks, staging_root):
        directory.mkdir(parents=True, exist_ok=True)
    final = runs / run_id
    if final.exists():
        if (final / "COMMIT.json").exists():
            raise PublicationError(
                f"run identity {run_id!r} is already committed; amendment forbidden")
        raise PublicationError(
            f"run identity {run_id!r} exists without COMMIT; refusing to overwrite")
    lock = locks / f"{run_id}.lock"
    lock_handle = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    staging: Path | None = None
    committed: DirectoryCommit | None = None
    try:
        staging = staging_root / f"{run_id}.{os.getpid()}.{time.time_ns():x}"
        staging.mkdir(parents=True, exist_ok=False)

        def prepare():
            manifest_payloads = {}
            for name, payload in sorted(payloads.items()):
                _write_exclusive(staging / name, payload)
                manifest_payloads[name] = {"sha256": _sha256_bytes(payload),
                                           "size_bytes": len(payload)}
            manifest = {"run_id": run_id, "identity": dict(identity),
                        "payloads": manifest_payloads, **semantic}
            manifest_bytes = json.dumps(manifest, sort_keys=True,
                                        separators=(",", ":")).encode()
            _write_exclusive(staging / "manifest.json", manifest_bytes)
            for name, record in manifest_payloads.items():
                on_disk = (staging / name).read_bytes()
                if (len(on_disk) != record["size_bytes"]
                        or _sha256_bytes(on_disk) != record["sha256"]):
                    raise PublicationError(f"staged payload {name} failed reverification")
            commit = {"run_id": run_id,
                      "manifest_sha256": _sha256_bytes(manifest_bytes),
                      "committed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())}
            _write_exclusive(staging / "COMMIT.json", json.dumps(
                commit, sort_keys=True, separators=(",", ":")).encode())

        call_with_deadline(prepare, closure["deadline_unix"] - clock())
        committed = publish_directory(staging, final,
                                      deadline=closure["deadline_unix"], clock=clock)
    except BaseException:
        # a crash anywhere inside the publisher removes its own staging tree
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        try:
            os.close(lock_handle)
            try:
                os.unlink(lock)
            except FileNotFoundError:
                pass
        except BaseException:
            if committed is None:
                raise
            committed = DirectoryCommit(final, False)
    return committed if return_commit else committed.path


def verify_run_closure(final_dir: Path | str) -> dict:
    """Run OUTSIDE the evidence root: bidirectional inventory + hash closure."""
    final = Path(final_dir)
    if not final.is_dir() or final.name.startswith("staging"):
        raise PublicationError(f"{final} is not a published run directory")
    commit_path = final / "COMMIT.json"
    manifest_path = final / "manifest.json"
    if not commit_path.exists() or not manifest_path.exists():
        raise PublicationError(f"{final} carries no COMMIT/manifest: not a success")
    inventory = sorted(item.name for item in final.iterdir())
    manifest = json.loads(manifest_path.read_text())
    expected = sorted(list(manifest["payloads"]) + ["manifest.json", "COMMIT.json"])
    if inventory != expected:
        raise PublicationError(
            f"inventory drift in {final}: on-disk {inventory} != manifest {expected}")
    for name, record in sorted(manifest["payloads"].items()):
        payload = (final / name).read_bytes()
        if len(payload) != record["size_bytes"] or _sha256_bytes(payload) != \
                record["sha256"]:
            raise PublicationError(f"payload {name} failed hash/size closure")
    commit = json.loads(commit_path.read_text())
    if commit.get("manifest_sha256") != _sha256_bytes(manifest_path.read_bytes()):
        raise PublicationError("COMMIT manifest hash mismatch")
    if commit.get("run_id") != manifest.get("run_id") or manifest.get("run_id") \
            != final.name:
        raise PublicationError("run identity mismatch across COMMIT/manifest/dir")
    return {"status": "CLOSED", "run_id": final.name, "inventory": inventory,
            "payload_count": len(manifest["payloads"])}


def write_failure_record(root: Path | str, run_id: str, record: Mapping) -> Path:
    """Failures never touch the success tree: separate O_EXCL records."""
    failures = Path(root) / "failures"
    failures.mkdir(parents=True, exist_ok=True)
    for attempt in range(1000):
        target = failures / f"{run_id}.{attempt}.json"
        try:
            _write_exclusive(target, json.dumps(record, sort_keys=True,
                                                separators=(",", ":")).encode())
            return target
        except FileExistsError:
            continue
    raise PublicationError(f"failure record space exhausted for {run_id}")
