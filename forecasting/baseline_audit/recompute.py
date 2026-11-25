"""Recompute corrected R25/R27 reports without fitting or model inference.

Run only on committed source. Never overwrite original or corrected artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "9cb437d11ff42e4001ec95127465524cdd31af22"
BASE_BRANCH = "research/r27-s3-confirmation"
REGIONS = ("NSW1", "QLD1", "TAS1")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_files():
    tracked = git("ls-files", "forecasting", "scripts/analyze_r25_ceiling.py",
                  "scripts/run_r27_confirmation.py").splitlines()
    return [ROOT / name for name in tracked if Path(name).suffix in (".py", ".json", ".yaml")]


def input_files():
    r25 = ROOT / "outputs/forecasting/r25_ceiling_20261119"
    files = sorted(r25.glob("*/*/*/selection.npz")) + sorted(r25.glob("*/*/*/evaluation.npz"))
    files += [r25 / "analysis_final.json",
              ROOT / "logs/forecasting/main_comparison_20261023/per_seed.csv",
              ROOT / "logs/forecasting/r27_s3_confirmation_20261125/results.json",
              ROOT / "logs/forecasting/r27_s3_confirmation_20261125/frozen_rules.json"]
    base = ROOT / "outputs/forecasting/aligned_baselines_20261012/runs"
    foundation = ROOT.parent / "DualTimesField-foundation-baselines/outputs/forecasting/foundation_baselines_20261023_r1/runs"
    for region in REGIONS:
        files.extend(base / "lightgbm" / f"seed{seed}" / region / "test.npz" for seed in (2026, 2027, 2028))
        files += [base / "xgboost/seed2026" / region / "test.npz",
                  foundation / "chronos2_zeroshot/seed2026" / region / "test.npz",
                  ROOT / "data/aemo_exogenous" / f"{region.lower()}_predispatch.npz"]
    if not files or any(not p.is_file() for p in files):
        raise FileNotFoundError("Required frozen input archives are missing")
    return sorted(set(p.resolve() for p in files))


def fingerprint(paths):
    return {str(path): digest(path) for path in paths}


def differences(output):
    old25 = read(ROOT / "outputs/forecasting/r25_ceiling_20261119/analysis_final.json")
    new25 = read(output / "r25_corrected.json")
    old27 = read(ROOT / "logs/forecasting/r27_s3_confirmation_20261125/results.json")
    new27 = read(output / "r27/results.json")
    old_rules = read(ROOT / "logs/forecasting/r27_s3_confirmation_20261125/frozen_rules.json")
    new_rules = read(output / "r27/frozen_rules.json")
    changed25 = {}
    for fold in new25["folds"]:
        changed25[fold] = {}
        for region in REGIONS:
            before, after = old25["folds"][fold]["regions"][region], new25["folds"][fold]["regions"][region]
            changed25[fold][region] = {
                "admission_changed": before["admitted_by_bucket"] != after["admitted_by_bucket"],
                "choice_counts_changed": before["s2_choices"] != after["s2_choices"],
                "new_selected_by_bucket": after["selected_by_bucket"],
                "note": "Old report retains choice counts, not per-origin choices; equality of counts alone is not proof of identical decisions.",
            }
    return {
        "r25": {"old_gate": old25["gate"], "new_gate": new25["gate"],
                "old_verdict": old25["verdict"], "new_verdict": new25["verdict"],
                "selection_changes": changed25},
        "r27": {"old": old27, "new": new27,
                "admission_changed": old_rules["admission"] != new_rules["admission"],
                "bucket_boundaries_changed": old_rules["boundaries"] != new_rules["boundaries"]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if git("status", "--porcelain"):
        raise RuntimeError("Commit the scoped repair before recomputation; the worktree must be clean")
    output.mkdir(parents=True, exist_ok=False)
    sources, inputs = source_files(), input_files()
    source_hashes, input_hashes = fingerprint(sources), fingerprint(inputs)
    manifest = {
        "baseline_commit": BASE_COMMIT, "baseline_branch": BASE_BRANCH,
        "execution_commit": git("rev-parse", "HEAD"), "execution_branch": git("branch", "--show-current"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "source_fingerprint": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "scope": "frozen prediction recomputation only; no fit, no model inference, no new tuning",
    }
    write(output / "manifest.json", manifest)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))
    commands = [
        [sys.executable, str(ROOT / "scripts/analyze_r25_ceiling.py"),
         "--output", str(output / "r25_corrected.json")],
        [sys.executable, str(ROOT / "scripts/run_r27_confirmation.py"),
         "--output-dir", str(output / "r27")],
    ]
    try:
        for index, command in enumerate(commands):
            print(f"Recompute {index + 1}/2: {Path(command[1]).name}", flush=True)
            with (output / f"step_{index + 1}.log").open("w") as stream:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        if fingerprint(sources) != source_hashes or fingerprint(inputs) != input_hashes:
            raise RuntimeError("Source or frozen inputs changed while recomputing")
        write(output / "comparison.json", differences(output))
        # Every new R27 archive must now round-trip through the common schema.
        from forecasting.aligned_scoring import load_prediction_artifact, score_predictions
        checks = {}
        for path in sorted((output / "r27").glob("*.npz")):
            checks[path.name] = score_predictions(load_prediction_artifact(path))["quantile_score_proxy"]
        if len(checks) != 18:
            raise ValueError(f"Expected 18 corrected R27 combination artifacts, got {len(checks)}")
        write(output / "verification.json", {"source_and_inputs_unchanged": True,
                                             "r27_artifact_roundtrip_qs": checks})
        write(output / "completed.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": fingerprint(sorted(p for p in output.rglob("*") if p.is_file()))})
        print(f"Completed; originals unchanged. Corrected reports: {output}", flush=True)
    except BaseException as exc:
        write(output / "failed.json", {"error": repr(exc), "original_inputs_unchanged": fingerprint(inputs) == input_hashes})
        raise


if __name__ == "__main__":
    main()
