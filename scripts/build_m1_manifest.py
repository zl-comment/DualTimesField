#!/usr/bin/env python3
"""Build the M1 artifact manifest: per-file SHA-256 + size over the evidence root.

manifest.json itself is the sole self-exclusion; run this as the LAST write into
the evidence root (the M0 audit-1 closure pattern: direct regeneration is the
final write and nothing is appended afterwards).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if root is None:
        print("usage: build_m1_manifest.py <evidence_root>", file=sys.stderr)
        return 2
    manifest_path = root / "artifact_manifest.json"
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest_path:
            files[str(path.relative_to(root))] = {
                "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "evidence_root": str(root),
        "self_exclusion": ["artifact_manifest.json"],
        "file_count": len(files),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {manifest_path} files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
