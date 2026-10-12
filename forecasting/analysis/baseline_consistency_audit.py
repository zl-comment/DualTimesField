"""Read-only artifact audit; passing does not establish experimental fairness.

Run from the repository root:
    python -m forecasting.analysis.baseline_consistency_audit --output report.json

No training, checkpoint selection, calibration, or significance tests occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from forecasting.exploration_evaluate import REGIONS, SEEDS, load_forecast

MODELS = ("full", "xgboost", "linear", "dlinear", "patchtst", "gatets", "gatets_adapter")
DEFAULT_ROOT = Path("outputs/forecasting/exploration_20261011/runs/static")
SOURCE_HASH = "33e2346fb81bbc68a31b8819f0adcb7ebbc7001ddd7c556d87890dcf684b3923"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root: Path) -> dict:
    records = []
    for region in REGIONS:
        for seed in SEEDS:
            for split in ("validation", "test"):
                anchor = load_forecast(root / "full" / f"seed{seed}" / region / f"{split}.npz")
                for model in MODELS:
                    directory = root / model / f"seed{seed}" / region
                    path = directory / f"{split}.npz"
                    meta = json.loads((directory / "run_metadata.json").read_text())
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    require(meta["state"] == "completed", f"Incomplete job: {directory}")
                    require(meta["source_hash"] == SOURCE_HASH, f"Source mismatch: {directory}")
                    require(meta["artifact_hashes"][path.name] == digest, f"Hash mismatch: {path}")
                    job = meta["job"]
                    require((job["protocol"], job["variant"], job["region"], job["seed"])
                            == ("static", model, region, seed), f"Job identity mismatch: {directory}")
                    with np.load(path, allow_pickle=False) as archive:
                        require(int(archive["seed"]) == seed and str(archive["region"]) == region,
                                f"Prediction identity mismatch: {path}")
                    forecast = load_forecast(path)
                    require(np.array_equal(forecast.origin_unix, anchor.origin_unix),
                            f"Origin mismatch: {path}")
                    require(np.array_equal(forecast.actual, anchor.actual), f"Target mismatch: {path}")
                    median = forecast.quantile[..., 2]
                    records.append({
                        "model": model, "region": region, "seed": seed, "split": split,
                        "path": str(path), "sha256": digest,
                        "n_origins": len(forecast.origin_unix),
                        "point_mae": float(np.abs(forecast.point - forecast.actual).mean()),
                        "q50_mae": float(np.abs(median - forecast.actual).mean()),
                        "point_equals_sorted_q50": bool(np.array_equal(forecast.point, median)),
                        "crossing_rate_before_sort": forecast.crossing_rate,
                        "point_array_sha256": hashlib.sha256(forecast.point.tobytes()).hexdigest(),
                    })
    seed_effects = []
    for model in MODELS:
        for region in REGIONS:
            rows = [r for r in records if r["model"] == model
                    and r["region"] == region and r["split"] == "test"]
            seed_effects.append({
                "model": model, "region": region,
                "distinct_point_arrays": len({r["point_array_sha256"] for r in rows}),
                "mae_seed_sample_sd": float(np.std([r["point_mae"] for r in rows], ddof=1)),
            })
    return {
        "scope": "historical_static_artifact_integrity_not_fairness_certification",
        "training_source_hash": SOURCE_HASH,
        "verified_prediction_files": len(records),
        "all_origins_and_targets_match_full_within_cell": True,
        "fair_comparison_ready": False,
        "remaining_code_and_protocol_blockers": "configs/fair_sota/baseline_audit_v1.yaml",
        "records": records, "seed_effects": seed_effects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.root.resolve()),
            "Audit output must be outside the frozen predictions directory")
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "seed_effects")}, indent=2))


if __name__ == "__main__":
    main()
