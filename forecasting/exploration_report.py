"""Complete-matrix forecasting report with validation-only floor controls."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from . import exploration_evaluate as evaluate
from .exploration_runner import ROOT, atomic_json, completed, source_hash

REGIONS, SEEDS = evaluate.REGIONS, evaluate.SEEDS
FLOOR_VARIANTS = {"full", "concat", "pd_calibrator", "pd_nofields", "no_tail", "xgboost"}
FLOOR_QUANTILES = (None, .0005, .001, .002, .005, .01)
SENSITIVITY_VARIANTS = {"no_tail", "concat", "no_ic", "xgboost"}


def save_forecast(path, forecast, seed, region):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, point=forecast.point, quantile=forecast.quantile,
                            actual=forecast.actual, origin_unix=forecast.origin_unix,
                            quantile_levels=evaluate.LEVELS, seed=seed, region=region)
    os.replace(temporary, path)


def stitch_forecasts(paths, output, seed, region, reference=None, expected_origins=17521):
    if len(paths) != 8:
        raise ValueError("Rolling protocol requires exactly eight quarterly forecasts")
    pieces = [evaluate.load_forecast(path) for path in paths]
    origins = np.concatenate([part.origin_unix for part in pieces])
    if len(origins) != expected_origins or not np.all(np.diff(origins) == 3600):
        raise ValueError("Rolling origins have gaps/duplicates or incorrect total origin count")
    joined = evaluate.Forecast(
        np.concatenate([part.point for part in pieces]),
        np.concatenate([part.quantile for part in pieces]),
        np.concatenate([part.actual for part in pieces]), origins,
        sum(part.crossing_rate * len(part.actual) for part in pieces) / len(origins), str(output))
    if reference is not None:
        evaluate.check_alignment(reference, joined)
    save_forecast(output, joined, seed, region)
    return joined


def crps(quantile, actual):
    error = actual[..., None] - quantile
    return float(2 * np.maximum(evaluate.LEVELS * error, (evaluate.LEVELS - 1) * error).mean())


def select_floors(validation, training_prices):
    training_prices = np.asarray(training_prices, dtype=np.float64)
    if not len(training_prices) or not np.isfinite(training_prices).all():
        raise ValueError("Floor fitting needs finite training-only prices")
    candidates = [None if quantile is None else min(0., float(np.quantile(training_prices, quantile)))
                  for quantile in FLOOR_QUANTILES]
    point_scores, quantile_scores = [], []
    for floor in candidates:
        point = validation.point if floor is None else np.maximum(validation.point, floor)
        quantile = validation.quantile if floor is None else np.maximum(validation.quantile, floor)
        point_scores.append(float(np.abs(point - validation.actual).mean()))
        quantile_scores.append(crps(quantile, validation.actual))
    point_index, quantile_index = int(np.argmin(point_scores)), int(np.argmin(quantile_scores))
    return {"point_floor": candidates[point_index], "quantile_floor": candidates[quantile_index],
            "point_training_quantile": FLOOR_QUANTILES[point_index],
            "quantile_training_quantile": FLOOR_QUANTILES[quantile_index],
            "candidate_training_quantiles": list(FLOOR_QUANTILES), "candidate_floors": candidates,
            "validation_mae": point_scores, "validation_crps": quantile_scores,
            "tie_rule": "first candidate; no clipping wins ties", "selection_scope": "one region, seed, and refit"}


def apply_floors(forecast, choice):
    point_floor, quantile_floor = choice["point_floor"], choice["quantile_floor"]
    point = forecast.point.copy() if point_floor is None else np.maximum(forecast.point, point_floor)
    quantile = forecast.quantile.copy() if quantile_floor is None else np.maximum(forecast.quantile, quantile_floor)
    return evaluate.Forecast(point, quantile, forecast.actual, forecast.origin_unix,
                             forecast.crossing_rate, forecast.path + "+floor")


@lru_cache(maxsize=128)
def _training_prices(csv_path, timestamp_column, target_column, timezone, start, end, cap):
    frame = pd.read_csv(csv_path, usecols=[timestamp_column, target_column])
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    lower = pd.Timestamp(start, tz=timezone).tz_convert("UTC")
    upper = pd.Timestamp(end, tz=timezone).tz_convert("UTC")
    values = frame.loc[(timestamps >= lower) & (timestamps < upper), target_column].to_numpy(dtype=float)
    return np.minimum(values, cap) if cap is not None else values


def floor_job(job, root):
    from .datasets import load_forecast_config
    config = load_forecast_config(ROOT / job["config"])
    data = config["data"]
    train = data["split_boundaries"]["train"]
    training_prices = _training_prices(str((ROOT / data["region_files"][job["region"]]).resolve()),
        data["delivery_column"], data["target_column"], data["timezone"],
        train["start"], train["end_exclusive"], data.get("price_cap"))
    source = ROOT / job["output"]
    validation = evaluate.load_forecast(source / "validation.npz")
    choice = select_floors(validation, training_prices)
    destination = root / "floored" / job["protocol"] / job["variant"] / f"seed{job['seed']}" / job["region"]
    for split in ("validation", "test"):
        forecast = validation if split == "validation" else evaluate.load_forecast(source / "test.npz")
        save_forecast(destination / f"{split}.npz", apply_floors(forecast, choice), job["seed"], job["region"])
    atomic_json(destination / "floor_selection.json", {"job": job["id"], "training_boundaries": train, **choice})
    return destination / "test.npz", choice


def matrix_groups(jobs):
    groups = defaultdict(list)
    for job in jobs:
        if job.get("baseline_kind") == "event_controls":
            continue
        protocol = "rolling" if job["protocol"].startswith("rolling/") else job["protocol"]
        groups[(protocol, job["variant"])].append(job)
    expected = {(r, s) for r in REGIONS for s in SEEDS}
    for (protocol, variant), rows in groups.items():
        quarterly = defaultdict(list)
        for job in rows:
            quarterly[job["protocol"]].append((job["region"], job["seed"]))
        if protocol == "rolling" and set(quarterly) != {f"rolling/q{i}" for i in range(8)}:
            raise ValueError(f"Incomplete rolling quarters for {variant}")
        for quarter, cells in quarterly.items():
            if len(cells) != 9 or set(cells) != expected:
                raise ValueError(f"Incomplete or duplicate region/seed cells: {quarter}/{variant}")
    return groups


def infer_output_root(jobs):
    roots = set()
    for job in jobs:
        output = ROOT / job["output"]
        parts = output.parts
        if "runs" not in parts:
            raise ValueError("Manifest output paths must contain the runs directory")
        roots.add(Path(*parts[:parts.index("runs")]))
    if len(roots) != 1:
        raise ValueError("Manifest jobs must share a common experiment output root")
    return roots.pop()


def source_pattern(paths):
    first = paths[(REGIONS[0], SEEDS[0])]
    pattern = str(first).replace(f"seed{SEEDS[0]}", "seed{seed}").replace(REGIONS[0], "{region}")
    if any(Path(pattern.format(seed=seed, region=region)) != path for (region, seed), path in paths.items()):
        raise ValueError("Forecast outputs do not follow the declared region/seed pattern")
    return pattern


def prepare_sources(groups, output_root, with_floor=True):
    sources = defaultdict(dict)
    floor_choices = {}
    static_reference = {}
    if ("static", "full") in groups:
        static_reference = {(job["region"], job["seed"]): ROOT / job["output"] / "test.npz"
                            for job in groups[("static", "full")]}
    for (protocol, variant), jobs in groups.items():
        if protocol != "rolling":
            paths = {(job["region"], job["seed"]): ROOT / job["output"] / "test.npz" for job in jobs}
            sources[protocol][variant] = source_pattern(paths)
        floor_paths = {}
        if with_floor and variant in FLOOR_VARIANTS:
            for job in jobs:
                path, choice = floor_job(job, output_root)
                floor_paths[job["id"]] = path
                floor_choices[job["id"]] = choice
        if protocol == "rolling":
            if not static_reference:
                raise ValueError("Rolling report requires static/full forecasts for exact grid and actual checks")
            for use_floor in ([False, True] if floor_paths else [False]):
                name = variant + ("+floor" if use_floor else "")
                paths = {}
                for region in REGIONS:
                    for seed in SEEDS:
                        quarters = sorted((job for job in jobs if job["region"] == region and job["seed"] == seed),
                                          key=lambda job: int(job["protocol"].split("q")[-1]))
                        parts = [floor_paths[job["id"]] if use_floor else ROOT / job["output"] / "test.npz" for job in quarters]
                        destination = output_root / "stitched" / "rolling" / name / f"seed{seed}" / region / "test.npz"
                        stitch_forecasts(parts, destination, seed, region,
                            reference=evaluate.load_forecast(static_reference[(region, seed)]))
                        paths[(region, seed)] = destination
                sources[protocol][name] = source_pattern(paths)
        elif floor_paths:
            paths = {(job["region"], job["seed"]): floor_paths[job["id"]] for job in jobs}
            sources[protocol][variant + "+floor"] = source_pattern(paths)
    return dict(sources), floor_choices


def cost_summary(jobs):
    rows = []
    for job in jobs:
        output = ROOT / job["output"]
        metadata = json.loads((output / "run_metadata.json").read_text())
        rows.append({"job": job["id"], "protocol": job["protocol"], "variant": job["variant"],
                     "region": job["region"], "seed": job["seed"], "seconds": metadata.get("seconds"),
                     "gpu": metadata.get("cuda_visible_devices"), "kind": job["kind"]})
    return {"jobs": rows, "sum_worker_seconds": sum(row["seconds"] or 0 for row in rows),
            "note": "Sum of per-worker elapsed times including data preparation and export; not wall time or exclusive GPU time."}


def routing_diagnostics(jobs):
    result = {}
    for job in jobs:
        with np.load(ROOT / job["output"] / "test.npz", allow_pickle=False) as data:
            keys = [key for key in ("dgf_fusion_weight", "dgf_quantile_fusion_weight") if key in data]
            if not keys:
                continue
            actual = data["actual"]
            masks = {"negative": actual < 0, "normal": (actual >= 0) & (actual <= 300), "spike": actual > 300}
            result[job["id"]] = {}
            for key in keys:
                values = data[key]
                if values.shape[:2] != actual.shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid routing diagnostics: {job['id']}/{key}")
                result[job["id"]][key] = {band: {"n": int(mask.sum()),
                    "mean_by_output": values[mask].mean(axis=0).tolist() if mask.any() else None}
                    for band, mask in masks.items()}
    return {"interpretation": "Descriptive gate activity conditioned on realized price bands; not causal evidence or future information supplied to the model.",
            "per_job": result}


def number(row):
    if row["mean"] is None:
        return "NA"
    return f"{row['mean']:.3f} ± {row['std']:.3f}" if row.get("std") is not None else f"{row['mean']:.3f}"


def markdown_report(result):
    lines = ["# Forecasting exploration results", "", result["interpretation"], "",
             "Every table reports three seeds independently; macro results average regions within each seed before computing sample standard deviation.", "",
             "CRPS~ is twice the equal-weight mean of five nonuniform quantile pinball losses, not an exact CRPS integral. Intervals, tails, monthly and horizon details are retained in JSON.", ""]
    for protocol, report in result["protocols"].items():
        lines += [f"## {protocol}", "", "Rows retain manifest order; no automatic performance ranking or success claim is made.", ""]
        main_variants = {"full", "linear", "xgboost", "dlinear", "patchtst", "gatets", "gatets_adapter", "pd_calibrator", "pd_nofields"}
        table_names = {"Main comparisons": [], "Ablations": [], "Frozen-event price forecasts": []}
        for name in report["summaries"]:
            category = "Frozen-event price forecasts" if name.startswith("event_frozen") else (
                "Main comparisons" if name.removesuffix("+floor") in main_variants else "Ablations")
            table_names[category].append(name)
        for title, names in table_names.items():
            if not names:
                continue
            lines += [f"### {title}", "", "| Variant | MAE | Window RMSE | CRPS~ | CRPS~ negative | CRPS~ spike | Coverage 90 | AIS 90 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name in names:
                row = report["summaries"][name]["macro_region_seed_summary"]
                lines.append("| " + name + " | " + " | ".join(number(row[key]) for key in
                    ("mae", "rmse_window_mean", "crps_quantile_approx", "crps_negative", "crps_spike", "coverage_90", "ais_90")) + " |")
            lines.append("")
        lines += ["", "### Regional seed means and standard deviations", "",
                  "| Variant | Region | MAE | CRPS~ | q05 negative pinball | q95 spike pinball |", "|---|---|---:|---:|---:|---:|"]
        for name, summary in report["summaries"].items():
            for region in REGIONS:
                row = summary["region_seed_summary"][region]["overall"]
                lines.append(f"| {name} | {region} | " + " | ".join(number(row[key]) for key in
                    ("mae", "crps_quantile_approx", "pinball_q05_negative_hours", "pinball_q95_spike_hours")) + " |")
        lines += ["", "### Paired differences against full", "", "Candidate minus full; negative favors the candidate. Holm adjustment spans all comparisons, metrics and regions within this protocol and block length.", "",
                  "| Candidate | Block hours | Metric | Difference | 95% CI | Holm p |", "|---|---:|---|---:|---|---:|"]
        for hours in ("168", "24"):
            for name, comparison in report["comparisons"][hours].items():
                for metric in ("mae", "crps_quantile_approx"):
                    row = comparison["metrics"][metric]["macro_regions"]
                    lines.append(f"| {name} | {hours} | {metric} | {row['difference']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['p_value_holm']:.4g} |")
        lines += ["", "Frozen-event forecasts are listed separately and excluded from paired price-forecast comparisons because their frozen trunks generate the same price forecasts.", ""]
        lines += ["### Equal-floor comparisons against full+floor", "",
                  "The same training candidate quantiles and validation selection rule are applied independently to every model, seed and region.", "",
                  "| Candidate | Metric | Difference | 95% CI | Holm p |", "|---|---|---:|---|---:|"]
        for name, comparison in report["comparisons_floor"]["168"].items():
            for metric in ("mae", "crps_quantile_approx"):
                row = comparison["metrics"][metric]["macro_regions"]
                lines.append(f"| {name} | {metric} | {row['difference']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['p_value_holm']:.4g} |")
        lines += ["", f"Case figures: `{protocol}/figures/`; cases selected from forecasts by a fixed rule, not by observed outcomes.", ""]
    lines += ["## Computational cost", "", f"Sum of worker elapsed time: {result['cost']['sum_worker_seconds']/3600:.2f} hours.", "", result["cost"]["note"], "",
              "| Job | GPU | Worker seconds |", "|---|---|---:|"]
    for row in result["cost"]["jobs"]:
        seconds = "NA" if row["seconds"] is None else f"{row['seconds']:.1f}"
        lines.append(f"| {row['job']} | {row['gpu']} | {seconds} |")
    return "\n".join(lines) + "\n"


def report(manifest_path, protocol=None, replicates=19999):
    manifest = json.loads(manifest_path.read_text())
    jobs = [job for job in manifest["jobs"] if protocol is None or job["protocol"] == protocol]
    if not jobs:
        raise ValueError("No jobs in selected protocol")
    run_sources = set()
    for job in jobs:
        marker = ROOT / job["output"] / "run_metadata.json"
        if not marker.exists():
            raise ValueError(f"Missing completed run: {job['id']}")
        metadata = json.loads(marker.read_text())
        fingerprint = metadata.get("source_hash", "")
        if not completed(job, fingerprint):
            raise ValueError(f"Incomplete or changed experiment artifacts: {job['id']}")
        run_sources.add(fingerprint)
    if len(run_sources) != 1:
        raise ValueError("Experiments were produced by different source snapshots")
    groups = matrix_groups(jobs)
    output_root = infer_output_root(jobs)
    patterns, floors = prepare_sources(groups, output_root)
    destination = manifest_path.parent / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    result = {"manifest": str(manifest_path), "protocol_filter": protocol,
              "training_source_hash": next(iter(run_sources)), "report_source_hash": source_hash(),
              "regions": list(REGIONS), "seeds": list(SEEDS), "bootstrap_replicates": replicates,
              "interpretation": manifest.get("test_role", "Retrospective exploration; test results are not used to select checkpoints or floors."),
              "floor_rule": "Independent per seed/region/refit; candidates from training-only prices; point selected by validation MAE and quantiles by unweighted validation CRPS~.",
              "floor_choices": floors, "cost": cost_summary(jobs),
              "routing_diagnostics": routing_diagnostics(jobs), "protocols": {}}
    for name, sources in patterns.items():
        if "full" not in sources:
            raise ValueError(f"Protocol {name} lacks full reference")
        reference = evaluate.load_source(sources["full"])
        floor_reference = evaluate.load_source(sources["full+floor"]) if "full+floor" in sources else None
        section = {"source_patterns": sources, "reference": "full", "summaries": {},
                   "frozen_event_variants": [], "comparisons": {"168": {}, "24": {}},
                   "floor_reference": "full+floor", "comparisons_floor": {"168": {}, "24": {}}}
        for variant, pattern in sources.items():
            candidate = reference if variant == "full" else evaluate.load_source(pattern)
            evaluate.check_sources(reference, candidate)
            section["summaries"][variant] = evaluate.summarize_source(candidate)
            if variant.startswith("event_frozen"):
                section["frozen_event_variants"].append(variant)
            elif variant != "full":
                section["comparisons"]["168"][variant] = evaluate.compare_sources(reference, candidate, replicates=replicates, block_hours=168)
                if variant.removesuffix("+floor") in SENSITIVITY_VARIANTS:
                    section["comparisons"]["24"][variant] = evaluate.compare_sources(reference, candidate, replicates=replicates, block_hours=24)
                if variant.endswith("+floor") and variant != "full+floor":
                    section["comparisons_floor"]["168"][variant] = evaluate.compare_sources(floor_reference, candidate, replicates=replicates, block_hours=168)
                    if variant.removesuffix("+floor") in SENSITIVITY_VARIANTS:
                        section["comparisons_floor"]["24"][variant] = evaluate.compare_sources(floor_reference, candidate, replicates=replicates, block_hours=24)
            print(json.dumps({"report_protocol": name, "variant": variant, "state": "scored"}), flush=True)
        for hours in ("168", "24"):
            combined = dict(section["comparisons"][hours])
            combined.update({name + " vs full+floor": value for name, value in section["comparisons_floor"][hours].items()})
            evaluate.holm_adjust(combined)
        result["protocols"][name] = section
        atomic_json(destination / f"{name}.json", section)
        from .exploration_figures import make_figures
        make_figures(sources, destination / name / "figures")
    stem = "report" if protocol is None else f"report_{protocol}"
    atomic_json(destination / f"{stem}.json", result)
    (destination / f"{stem}.md").write_text(markdown_report(result))
    command = [sys.executable, "-m", "forecasting.exploration_events", "summarize", "--manifest", str(manifest_path)]
    if protocol:
        command += ["--protocol", protocol]
    if any(job.get("baseline_kind") == "event_controls" or "event_" in job["variant"] for job in jobs):
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"Complete report: {destination / (stem + '.md')}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", choices=["static"])
    parser.add_argument("--replicates", type=int, default=19999)
    args = parser.parse_args()
    report(args.manifest.resolve(), args.protocol, args.replicates)


if __name__ == "__main__":
    main()
