#!/usr/bin/env python3
"""Read-only closure checker for the M0 c2 evidence artifact manifest.

Verifies that ``artifact_manifest.json`` under the evidence root is an exact
closure of the tree: every real file is listed once with matching sha256 and
size, the only exclusion is the manifest itself (a file cannot hold a stable
hash of itself), and no transient ``run_logged.sh`` temp files (``*.raw.*``)
remain.  Never writes anything.

Exit codes: 0 = closure holds, 1 = violations found, 2 = environment error.
"""
import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="evidence root directory")
    ap.add_argument(
        "--allow-raw-prefix",
        default=None,
        help="relative path prefix of the caller's own in-flight run_logged.sh "
        "temp file to ignore (only valid while a logged command is executing)",
    )
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"ENV_ERROR: evidence root not found: {root}")
        return 2
    manifest_rel = "artifact_manifest.json"
    manifest_path = os.path.join(root, manifest_rel)
    if not os.path.isfile(manifest_path):
        print("VIOLATION: artifact_manifest.json is missing from the evidence root")
        return 1

    rel_files = []
    for dirpath, _subdirs, names in os.walk(root):
        for name in names:
            rel_files.append(os.path.relpath(os.path.join(dirpath, name), root))
    rel_files.sort()

    problems = []
    transient = [r for r in rel_files if ".raw." in os.path.basename(r)]
    ignored = []
    for t in transient:
        if args.allow_raw_prefix and t.startswith(args.allow_raw_prefix):
            ignored.append(t)
        else:
            problems.append(f"transient run_logged temp file present: {t}")
    tree = [r for r in rel_files if r not in ignored]

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    print(f"evidence_root_field: {manifest.get('evidence_root')!r}")
    print(f"tree_total_files: {len(tree)} (ignored_transient: {len(ignored)})")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        print("VIOLATION: manifest has no files list")
        return 1

    excluded = manifest.get("excluded")
    if (
        not isinstance(excluded, list)
        or len(excluded) != 1
        or not isinstance(excluded[0], dict)
        or excluded[0].get("path") != manifest_rel
        or not str(excluded[0].get("reason", "")).strip()
    ):
        problems.append(
            "manifest must exclude exactly one file - artifact_manifest.json - "
            "with a stated reason (self-hash closure is impossible); "
            f"found: {json.dumps(excluded)[:200]}"
        )
        excluded_paths = []
    else:
        excluded_paths = [excluded[0]["path"]]

    entry_paths = [e.get("path") for e in entries]
    if manifest_rel in entry_paths:
        problems.append("manifest lists itself as a hashed payload; it must be self-excluded")

    dupes = sorted({p for p in entry_paths if entry_paths.count(p) > 1})
    if dupes:
        problems.append(f"duplicate manifest entries: {dupes}")

    covered = set(entry_paths) | set(excluded_paths)
    tree_set = set(tree)
    for p in sorted(tree_set - covered):
        problems.append(f"file in tree but not covered by manifest: {p}")
    for p in sorted(covered - tree_set):
        problems.append(f"manifest entry does not exist in tree: {p}")

    for e in entries:
        p = e.get("path")
        if p not in tree_set:
            continue
        full = os.path.join(root, p)
        size = os.path.getsize(full)
        if e.get("size_bytes") != size:
            problems.append(
                f"size mismatch for {p}: manifest {e.get('size_bytes')} vs actual {size}"
            )
        digest = sha256_file(full)
        if e.get("sha256") != digest:
            problems.append(
                f"sha256 mismatch for {p}: manifest {e.get('sha256')} vs actual {digest}"
            )

    file_count = manifest.get("file_count")
    if file_count != len(tree):
        problems.append(f"file_count {file_count} != actual tree total {len(tree)}")
    hashed_count = manifest.get("hashed_payload_count")
    if hashed_count != len(entries):
        problems.append(
            f"hashed_payload_count {hashed_count} != number of entries {len(entries)}"
        )
    if excluded_paths and len(entries) != len(tree) - len(excluded_paths):
        problems.append(
            f"entries {len(entries)} != tree {len(tree)} minus excluded {len(excluded_paths)}"
        )
    root_field = str(manifest.get("evidence_root", "")).rstrip("/")
    if root_field != root.rstrip("/"):
        problems.append(
            f"evidence_root field {root_field!r} != actual root {root.rstrip('/')!r}"
        )

    print(f"manifest_entries: {len(entries)}  excluded: {excluded_paths}")
    if problems:
        print(f"VIOLATIONS: {len(problems)}")
        for p in problems:
            print(f"  - {p}")
        print("CLOSURE_FAILED")
        return 1
    print(
        "CLOSURE_OK: manifest is an exact closure of the evidence root "
        "(sole exclusion: artifact_manifest.json)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
