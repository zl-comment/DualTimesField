#!/usr/bin/env python3
"""Regenerate artifact_manifest.json as the FINAL write over the evidence root.

Audit-1 evidence-only repair: hashes every file under the root except the
manifest itself (self-hash closure is impossible).  Refuses to run while any
transient ``*.raw.*`` run_logged.sh temp file is present, and must be invoked
directly - never through run_logged.sh - so that no ledger append or temp
file follows the manifest write (closure rule).
"""
import argparse
import datetime
import hashlib
import json
import os


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = args.root.rstrip("/")

    rel_files = []
    for dirpath, _subdirs, names in os.walk(root):
        for name in names:
            rel_files.append(os.path.relpath(os.path.join(dirpath, name), root))
    rel_files.sort()

    transient = [r for r in rel_files if ".raw." in os.path.basename(r)]
    if transient:
        raise SystemExit(f"REFUSED: transient run_logged temp files present: {transient}")

    manifest_rel = "artifact_manifest.json"
    payload = [r for r in rel_files if r != manifest_rel]
    files = [
        {
            "path": rel,
            "sha256": sha256_file(os.path.join(root, rel)),
            "size_bytes": os.path.getsize(os.path.join(root, rel)),
        }
        for rel in payload
    ]

    manifest = {
        "evidence_root": root,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "file_count": len(rel_files),
        "hashed_payload_count": len(files),
        "files": files,
        "excluded": [
            {
                "path": manifest_rel,
                "reason": "self-exclusion: this manifest is the final write over the evidence root and cannot contain a stable hash of itself",
            }
        ],
        "closure_rule": "artifact_manifest.json is the last write under the evidence root; after its generation no run_logged.sh ledger append or .raw.$$ temp file may occur there",
        "note": "per-file sha256/size over every file under the evidence root except artifact_manifest.json itself",
    }

    manifest_path = os.path.join(root, manifest_rel)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"regenerated {manifest_path}")
    print(
        f"file_count={manifest['file_count']} "
        f"hashed_payload_count={manifest['hashed_payload_count']} "
        f"excluded=1 ({manifest_rel})"
    )
    print(f"generated_utc={manifest['generated_utc']}")


if __name__ == "__main__":
    main()
