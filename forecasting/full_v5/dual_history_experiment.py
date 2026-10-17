"""Frozen three-region contribution check for dual-field historical features."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import yaml

from ..aligned_scoring import assert_comparable, save_prediction_artifact
from . import data, diagnostics, schema
from .dual_history_model import DualHistoryConfig, fit_historical_dualfield
from .dual_history_memory import build_corrections
from .history_correction import apply_history_correction, fit_history_weights
from .history_quickcheck import load_frozen_region

REGIONS = ("NSW1", "QLD1", "TAS1")
VARIANTS = {"H_recent": "recentraw", "H_raw": "rawmatched", "H_smooth": "fixedsmooth",
            "H_dual_trend": "dualtrend", "H_dual_full": "dualfull"}
MODELS = ("B", "C_roll", *VARIANTS, "H_dual_event_off")
METRICS = ("mae", "quantile_score_proxy")


def price_lags(arrays, metadata, baseline_config):
    """Recover existing 72h price data with its original train-only affine map."""
    if metadata["price_transform"] != "identity":
        raise ValueError("This experiment requires original affine raw-price inputs")
    columns = baseline_config["data"]["history_columns"]
    column = columns.index(baseline_config["data"]["target_column"])
    hist = np.asarray(arrays["history_values"], dtype=np.float64)
    if hist.shape[1:] != (72, len(columns)):
        raise ValueError("72h input schema changed")
    price = hist[:, :, column] * metadata["target_std"] + metadata["target_mean"]
    return np.stack((price[:, :24], price[:, 24:48], price[:, 48:72]), axis=-1)


def run_region(project, source, output, region, identity, config, baseline_config):
    start = time.monotonic()
    arrays, meta, bq, z, anchor, rows, frozen_hashes, gap_count = load_frozen_region(
        project, source, region, identity)
    origins = arrays["origin_unix"]
    lags = price_lags(arrays, meta, baseline_config)
    print(f"{region}: training historical dual field on pre-October records", flush=True)
    model, train_audit = fit_historical_dualfield(origins, arrays["actual"], z, lags,
                                                seed=2026, config=config)
    model.save_checkpoint(output / "models" / f"{region}.pt", train_audit)
    print(f"{region}: frozen encoder; releasing matured historical curves", flush=True)
    memory = build_corrections(origins, arrays["actual"], anchor, z, lags, model)
    predictions = {"B": bq, "C_roll": anchor}
    fits = {}
    for model_id, key in VARIANTS.items():
        correction = memory["corrections"][key]
        fits[model_id] = fit_history_weights(origins, arrays["actual"], anchor, correction)
        predictions[model_id] = apply_history_correction(anchor, correction, fits[model_id]["alpha"])
    # Same fitted full-model weights and neighbor IDs: switch off only E.
    predictions["H_dual_event_off"] = apply_history_correction(
        anchor, memory["corrections"]["dualtrend"], fits["H_dual_full"]["alpha"])
    idx = rows["v2"]
    subset = {k: arrays[k][idx] for k in ("origin_unix", "target_time_unix", "actual")}
    scored, changes, losses = {}, {}, {}
    reference = None
    for model_id, quantiles in predictions.items():
        artifact = diagnostics.build_artifact(model_id, quantiles[idx], subset, meta, "dual_history_v2")
        if reference is None:
            reference = artifact["artifact"]
        else:
            assert_comparable(reference, artifact["artifact"], allow_different_seed=False)
        save_prediction_artifact(output / "predictions" / f"{region}_{model_id}.npz", artifact["artifact"])
        scored[model_id] = dict(metrics=artifact["scores"],
            bands=diagnostics.band_decomposition(quantiles[idx, :, 2], subset["actual"]),
            spike=diagnostics.spike_alarm_diagnostics(quantiles[idx], subset["actual"]))
        delta = quantiles[idx, :, 2] - anchor[idx, :, 2]
        changes[model_id] = dict(fraction=float(np.mean(np.abs(delta) > 1e-9)),
                                 mean_abs=float(np.abs(delta).mean()), max_abs=float(np.abs(delta).max()))
        losses[model_id] = diagnostics.per_origin_losses(quantiles[idx], subset["actual"])
    encoded = memory["encoded_curves"]
    ei = encoded["indices"]
    reconstruction = {}
    for name in ("fixedsmooth", "trend", "reconstruction"):
        reconstructed = encoded[name]
        reconstruction[name] = diagnostics.band_decomposition(reconstructed, arrays["actual"][ei])
    reconstruction["event_mean_abs"] = float(np.abs(encoded["event"]).mean())
    reconstruction["event_max_abs"] = float(np.abs(encoded["event"]).max())
    data.atomic_npz(output / "features" / f"{region}_encoded.npz", **encoded)
    data.atomic_npz(output / "features" / f"{region}_corrections.npz", origin_unix=origins,
                    matched_indices=memory["matched_indices"], recent_indices=memory["recent_indices"],
                    **memory["corrections"])
    for path, digest in frozen_hashes.items():
        if data.sha256_file(source / path) != digest:
            raise ValueError("Frozen parent artifact changed during contribution experiment")
    result = dict(region=region, seed=2026, scores=scored, changes=changes, fits=fits,
                  encoder=train_audit, reconstruction=reconstruction, history_audit=memory["audit"],
                  frozen_hashes=frozen_hashes, reconstructed_boundary_origins=gap_count,
                  checkpoint_sha=data.sha256_file(output / "models" / f"{region}.pt"),
                  wall_seconds=time.monotonic() - start)
    data.atomic_json(output / f"{region}.json", result)
    print(f"{region}: " + ", ".join(f"{m}={scored[m]['metrics']['mae']:.4f}" for m in MODELS), flush=True)
    return result, origins[idx], losses


def summarize(results):
    macro = {m: {k: float(np.mean([r["scores"][m]["metrics"][k] for r in results]))
                 for k in METRICS} for m in MODELS}
    comparisons = {m: {k: macro[m][k] - macro["H_dual_full"][k] for k in METRICS}
                   for m in MODELS if m != "H_dual_full"}
    gain = all(v > 1e-9 for m in ("C_roll", "H_raw", "H_smooth", "H_recent")
               for v in comparisons[m].values())
    regions_ok = all(r["scores"]["H_dual_full"]["metrics"][k] <=
                     r["scores"]["C_roll"]["metrics"][k] + 1e-9 for r in results for k in METRICS)
    return dict(macro=macro, reference_minus_dual_full=comparisons,
                field_activation=all(r["reconstruction"]["event_mean_abs"] > 1e-9 for r in results),
                forecast_activation=all(r["changes"]["H_dual_full"]["fraction"] > 0 for r in results),
                incremental_value_pass=gain and regions_ok,
                per_region_non_regression=regions_ok,
                beats_original_baseline=all(v > 1e-9 for v in comparisons["B"].values()),
                event_increment_pass=all(v > 1e-9 for v in comparisons["H_dual_trend"].values()),
                event_fixed_head_pass=all(v > 1e-9 for v in comparisons["H_dual_event_off"].values()),
                scope="Retrospective December2022 development; all settings fixed before scoring; no final test")


def report(output, results, summary):
    lines = ["# Dual-field historical features: contribution results", "",
             "Three regions, seed2026. Historical encoder: Jul–Sep; correction weights: Oct–Nov; scores: December2022.",
             "This is retrospective development. Actual first publication timestamps remain unaudited; causality is under archive proxies.", "",
             "| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        values = [r["scores"][m]["metrics"]["mae"] for r in results] + [summary["macro"][m][k] for k in METRICS]
        lines.append(f"| {m} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "## Contribution checks", ""]
    for k in ("field_activation", "forecast_activation", "incremental_value_pass", "per_region_non_regression", "event_increment_pass", "event_fixed_head_pass", "beats_original_baseline"):
        lines.append(f"- {k}: **{summary[k]}**")
    for r in results:
        lines += ["", f"## {r['region']}", "",
                  f"- Encoder parameters: {r['encoder']['trainable_parameters']}; fit rows: {r['encoder']['fit_rows']}; latest training-label availability: {r['encoder']['latest_fit_label_available']}.",
                  f"- Full history weights: {r['fits']['H_dual_full']['alpha']}",
                  f"- Trend-only weights: {r['fits']['H_dual_trend']['alpha']}",
                  f"- Full forecast changes: {r['changes']['H_dual_full']}",
                  f"- Historical reconstruction MAE: {r['reconstruction']['reconstruction']['mae_total']:.6f}; fixed-smoothing MAE: {r['reconstruction']['fixedsmooth']['mae_total']:.6f}.",
                  f"- Wall time: {r['wall_seconds']:.1f}s."]
    lines += ["", "All matched variants share candidate IDs and four downstream parameters. Nonzero fields or better reconstruction alone do not establish forecast improvement.",
              "Bootstrap intervals and all price-band contributions are in bootstrap.json and regional JSON. No automatic architecture selection or final-test launch follows."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/forecasting/v5_calibrated_history_20261015_r3")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    source, output = (project / args.source).resolve(), (project / args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; preserve previous trials")
    config_path = project / "configs/full_v5/dual_history.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["seed"] != 2026 or tuple(config["regions"]) != REGIONS or config["encoder"] != asdict(DualHistoryConfig()):
        raise ValueError("Fixed experiment configuration changed")
    identity = data.read_json(source / "identity/identity.json")
    for block in ("sources", "tests", "reused_sources"):
        for path, digest in identity[block].items():
            if data.sha256_file(project / path) != digest:
                raise ValueError(f"Existing predictive implementation changed: {path}")
    baseline_config_path = project / "outputs/forecasting/aligned_baselines_20261012/config.yaml"
    baseline_config = yaml.safe_load(baseline_config_path.read_text())
    if baseline_config["data"]["history_columns"] != ["rrp_aud_per_mwh", "total_demand_mw"]:
        raise ValueError("Original baseline history schema changed")
    output.mkdir(parents=True)
    for folder in ("models", "predictions", "features"):
        (output / folder).mkdir()
    files = [p for p in (project / "forecasting/full_v5").glob("*.py")]
    files += list((project / "tests").glob("test_full_v5_dual_history*.py"))
    files += [config_path, project / "docs/full_v5_dual_history_20261017.md",
              project / "docs/full_v5_dual_history_audit_20261017.md", project / "src/dualfield/core.py"]
    hashes = {}
    for path in files:
        rel = str(path.relative_to(project))
        hashes[rel] = data.sha256_file(path)
        target = output / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    data.atomic_json(output / "identity.json", dict(git=data.git_identity(project),
                     parent_identity_sha=data.sha256_file(source / "identity/identity.json"),
                     baseline_config_sha=data.sha256_file(baseline_config_path), source_hashes=hashes))
    completed = [run_region(project, source, output, r, identity, DualHistoryConfig(), baseline_config) for r in REGIONS]
    results = [r[0] for r in completed]
    summary = summarize(results)
    origins = {r[0]["region"]: r[1] for r in completed}
    uncertainty = {}
    for reference in ("C_roll", "H_raw", "H_smooth", "H_recent", "H_dual_trend", "H_dual_event_off"):
        ref = {r[0]["region"]: r[2][reference] for r in completed}
        cand = {r[0]["region"]: r[2]["H_dual_full"] for r in completed}
        uncertainty[reference] = {str(block): diagnostics.paired_block_bootstrap(
            origins, ref, cand, block_hours=block, n_boot=999, seed=2026) for block in (168, 24)}
    data.atomic_json(output / "bootstrap.json", uncertainty)
    data.atomic_json(output / "summary.json", summary)
    report(output, results, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
