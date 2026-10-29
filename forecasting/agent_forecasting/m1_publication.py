"""Atomic append-only evidence publication for the M1 contract.

A result is visible ONLY when an immutable final directory contains a valid
COMMIT.json whose manifest hash and full bidirectional inventory reverify.  The
publisher: takes an exclusive per-run lock and refuses a committed identity;
creates a same-filesystem O_EXCL staging directory; writes every payload and
fsyncs it; generates a per-payload SHA-256/size manifest; re-verifies everything
on disk; atomically renames staging to the final directory; and finally renames
COMMIT.json.tmp -> COMMIT.json as the LAST evidence-root write.  Any crash before
that point leaves no success.  Failure records use a separate O_EXCL directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Mapping

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


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


def publish_run(root: Path | str, run_id: str, payloads: Mapping[str, bytes], *,
                identity: Mapping) -> Path:
    """Atomically publish one run; returns the immutable final directory."""
    root = Path(root)
    if not _RUN_ID_RE.fullmatch(run_id or ""):
        raise PublicationError(f"run id {run_id!r} is not a bounded identifier")
    if not payloads:
        raise PublicationError("a run must carry at least one payload")
    for name in payloads:
        if not _RUN_ID_RE.fullmatch(name) or "/" in name or name in (
                "manifest.json", "COMMIT.json", "COMMIT.json.tmp"):
            raise PublicationError(f"payload name {name!r} is not allowed")
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
    try:
        staging = staging_root / f"{run_id}.{os.getpid()}.{time.time_ns():x}"
        staging.mkdir(parents=True, exist_ok=False)
        manifest_payloads = {}
        for name, payload in sorted(payloads.items()):
            target = staging / name
            _write_exclusive(target, payload)
            manifest_payloads[name] = {"sha256": _sha256_bytes(payload),
                                       "size_bytes": len(payload)}
        manifest = {"run_id": run_id, "identity": dict(identity),
                    "payloads": manifest_payloads}
        _write_exclusive(staging / "manifest.json",
                         json.dumps(manifest, sort_keys=True,
                                    separators=(",", ":")).encode())
        _fsync_dir(staging)
        # re-verify every byte on disk against the manifest before promotion
        for name, record in manifest_payloads.items():
            on_disk = (staging / name).read_bytes()
            if (len(on_disk) != record["size_bytes"]
                    or _sha256_bytes(on_disk) != record["sha256"]):
                raise PublicationError(f"staged payload {name} failed reverification")
        os.rename(staging, final)
        _fsync_dir(runs)
        commit = {"run_id": run_id,
                  "manifest_sha256": _sha256_bytes(
                      (final / "manifest.json").read_bytes()),
                  "committed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime())}
        _write_exclusive(final / "COMMIT.json.tmp",
                         json.dumps(commit, sort_keys=True,
                                    separators=(",", ":")).encode())
        os.rename(final / "COMMIT.json.tmp", final / "COMMIT.json")
        _fsync_dir(final)
        _fsync_dir(runs)
        return final
    finally:
        os.close(lock_handle)


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
