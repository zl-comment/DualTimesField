"""Resolve audited foundation checkpoints locally, without following mutable refs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


def verified_checkpoint(model_file: str, hf_home: str | Path | None = None):
    """Return the exact snapshot and identity declared beside the adapter.

    The source records contain the download-time hashes from the archived run.
    Missing files or changed bytes fail before importing/loading the model. No
    fallback snapshot search, refs/main lookup, or network access is performed.
    """
    source = Path(model_file).with_name("source.json")
    weights = json.loads(source.read_text())["weights"]
    revision, expected = weights["revision"], weights["sha256"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Checkpoint revision must be a full commit: {source}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Checkpoint SHA256 is not pinned: {source}")
    if hf_home is None:
        hf_home = os.environ.get("HF_HOME", str(source.parents[2] / "weights" / "hf_home"))
    snapshot = (Path(hf_home) / "hub" / ("models--" + weights["repo"].replace("/", "--"))
                / "snapshots" / revision)
    checkpoint = snapshot / weights["file"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pinned checkpoint is unavailable locally: {checkpoint}")
    actual = file_sha256(checkpoint)
    if actual != expected or checkpoint.stat().st_size != weights["size_bytes"]:
        raise ValueError(f"Pinned checkpoint bytes changed: {checkpoint}")
    config_sha = weights.get("config_sha256")
    if config_sha is not None and file_sha256(snapshot / "config.json") != config_sha:
        raise ValueError(f"Pinned checkpoint config changed: {snapshot / 'config.json'}")
    identity = {"repo": weights["repo"], "revision": revision,
                "file": weights["file"], "sha256": actual,
                "size_bytes": checkpoint.stat().st_size,
                "checkpoint_policy": "audited_local_snapshot_v1"}
    if config_sha is not None:
        identity["config_sha256"] = config_sha
    return snapshot, identity


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
