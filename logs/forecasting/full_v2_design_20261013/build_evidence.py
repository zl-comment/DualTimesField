"""Read-only cache analysis and NumPy checks of the proposed transport algebra.

This does not implement, train, or evaluate Full v2. It never reads test labels.
Run from the repository with .venv/bin/python and this file's path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "outputs/forecasting/aligned_baselines_20261012/data_cache"


def episodes(times, values, *, region, split, metadata):
    result = []
    for name, mask in (("negative", values < 0), ("above_300", values > 300)):
        gaps = np.diff(times) != 3600
        starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1] | gaps])
        ends = np.flatnonzero(mask & np.r_[~mask[1:] | gaps, True])
        complete = (starts > 0) & (ends < len(times) - 1)
        complete &= times[starts] - times[np.maximum(starts - 1, 0)] == 3600
        complete &= times[np.minimum(ends + 1, len(times) - 1)] - times[ends] == 3600
        duration = (ends - starts + 1)[complete]
        result.append({
            "region": region, "split": split, "event": name,
            "unique_delivery_hours": len(times), "event_hours": int(mask.sum()),
            "event_rate": float(mask.mean()), "episodes": len(starts),
            "complete_episodes": int(complete.sum()),
            "median_duration_hours": float(np.median(duration)) if len(duration) else None,
            "p90_duration_hours": float(np.quantile(duration, .9)) if len(duration) else None,
            "max_duration_hours": int(duration.max()) if len(duration) else None,
            "one_hour_episode_fraction": float(np.mean(duration == 1)) if len(duration) else None,
            "source_sample_hash": metadata["sample_hash"],
            "source_preprocessing_hash": metadata["preprocessing_hash"],
        })
    return result


def transport(background, delta, raw_gap):
    gap = np.diff(background, axis=-1)
    r = 3.0 * np.tanh(raw_gap / 3.0)
    inc = gap * np.expm1(r)
    correction = np.stack((
        -inc[..., 1] - inc[..., 0], -inc[..., 1], np.zeros_like(delta),
        inc[..., 2], inc[..., 2] + inc[..., 3],
    ), axis=-1)
    return background + delta[..., None] + correction


def algebra_checks():
    rng = np.random.default_rng(20261013)
    background = np.sort(rng.normal(0, 1000, (10000, 5)), axis=-1)
    background[::7, 1] = background[::7, 0]
    background[::13, 4] = background[::13, 3]
    delta = rng.normal(0, 100, 10000)
    raw_gap = rng.normal(0, 10, (10000, 4))
    identity = transport(background, np.zeros(10000), np.zeros((10000, 4)))
    predicted = transport(background, delta, raw_gap)
    transformed = transport(2.5 * background + 71, 2.5 * delta, raw_gap)
    affine_error = float(np.max(np.abs(transformed - (2.5 * predicted + 71))))
    identity_ok = bool(np.array_equal(identity, background))
    finite_ok = bool(np.isfinite(predicted).all())
    minimum_gap = float(np.diff(predicted, axis=-1).min())
    monotone_ok = minimum_gap >= -1e-9
    affine_ok = affine_error < 1e-8
    assert identity_ok and finite_ok and monotone_ok and affine_ok
    return {
        "scope": "proposed transport algebra only; not a Full v2 implementation or model test",
        "dtype": "float64", "random_vectors": len(background),
        "identity_elementwise_exact": identity_ok, "all_finite": finite_ok,
        "monotonicity_within_1e_minus9": monotone_ok,
        "minimum_adjacent_gap": minimum_gap,
        "positive_affine_consistency": affine_ok,
        "max_affine_absolute_error": affine_error,
        "includes_original_zero_gaps": True,
    }


def main():
    rows = []
    oof_start = np.datetime64("2017-12-31T14:00:00", "s").astype(np.int64)
    for region in ("NSW1", "QLD1", "TAS1"):
        for split in ("train", "selection"):
            directory = CACHE / region / split
            actual = np.load(directory / "actual.npy", mmap_mode="r")
            target_times = np.load(directory / "target_time_unix.npy", mmap_mode="r")
            metadata = json.loads((directory / "metadata.json").read_text())
            times, indices = np.unique(target_times.reshape(-1), return_index=True)
            values = np.asarray(actual).reshape(-1)[indices]
            assert np.isfinite(values).all()
            rows.extend(episodes(times, values, region=region, split=split, metadata=metadata))
            if split == "train":
                keep = times >= oof_start
                rows.extend(episodes(times[keep], values[keep], region=region,
                                     split="train_labels_2018_2021_subset", metadata=metadata))
    output = {
        "design_date": "2026-10-13", "source_run": "aligned_baselines_20261012",
        "reads_test_labels": False, "trains_models": False,
        "counting": "Unique target timestamps; consecutive hourly threshold exceedances; boundary-censored episodes excluded from duration summaries",
        "independence": "Episode counts are not claims of statistical independence across regions or time",
        "oof_subset": "2018-2021 label coverage only, not a completed OOF cache or exact fold-origin coverage",
        "rows": rows, "transport_algebra_checks": algebra_checks(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination = Path(__file__).with_name("evidence.json")
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "rows": len(rows),
                      "algebra": output["transport_algebra_checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
