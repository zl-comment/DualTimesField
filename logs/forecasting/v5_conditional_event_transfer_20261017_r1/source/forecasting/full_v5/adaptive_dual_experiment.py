"""Representation-only adaptive historical dual-field contribution experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml

from ..aligned_scoring import assert_comparable, save_prediction_artifact
from . import data, diagnostics, schema
from .adaptive_dual_history import AdaptiveDualConfig, FIT_CUTOFF, fit_adaptive_dualfield
from .dual_history_experiment import price_lags, REGIONS, METRICS
from .dual_history_model import DualHistoryConfig, fit_historical_dualfield
from .dual_history_memory import build_corrections
from .history_correction import apply_history_correction, fit_history_weights
from .history_quickcheck import load_frozen_region

OCTOBER = schema.unix_timestamp("2022-10-01")
MODELS = ("B", "C_roll", "H_recent", "H_raw", "H_smooth", "H_old_dual",
          "H_fixed_pulses", "H_adaptive_trend", "H_adaptive_full", "H_adaptive_event_off")


class FixedGeometry:
    def __init__(self, model):
        self.model = model

    def encode_matured(self, *args, **kwargs):
        return self.model.encode_matured(*args, **kwargs, fixed=True)


def reconstruction_metrics(prediction, actual, event=None):
    result = diagnostics.band_decomposition(prediction, actual)
    result["mean_level_abs_error"] = float(np.abs((prediction - actual).mean(axis=1)).mean())
    spike = actual.max(axis=1) > 300
    result["spike_curves"] = int(spike.sum())
    if spike.any():
        truth_peak = actual[spike].argmax(axis=1)
        predicted_peak = prediction[spike].argmax(axis=1)
        result["spike_peak_hour_mae"] = float(np.abs(predicted_peak - truth_peak).mean())
        result["spike_peak_exact_fraction"] = float(np.mean(predicted_peak == truth_peak))
        result["spike_max_price_mae"] = float(np.abs(prediction[spike].max(axis=1) - actual[spike].max(axis=1)).mean())
        if event is not None:
            result["event_peak_exact_fraction"] = float(np.mean(event[spike].argmax(axis=1) == truth_peak))
    if event is not None:
        result["event_mean_abs"] = float(np.abs(event).mean())
    return result


def assess_reconstruction(origins, actual, z, lags, old, adaptive):
    """Fixed September diagnostics; never used for selection or training."""
    idx = np.flatnonzero((origins >= FIT_CUTOFF) & (origins + 24 * 3600 < OCTOBER))
    if not len(idx):
        raise ValueError("No mature September assessment curves")
    args = [torch.as_tensor(np.asarray(a[idx], dtype=np.float32)) for a in (actual, z, lags)]
    old.eval()
    adaptive.eval()
    with torch.no_grad():
        full = {"old_dual": old(*args), "fixed_pulses": adaptive(*args, fixed=True),
                "adaptive": adaptive(*args)}
        # Identical fixed mask for all three representations; no hidden labels
        # go to adaptive proposals, geometry features or the coefficient solve.
        torch.manual_seed(2026)
        masked_old = old(*args, mask_probability=.2)
        visible = ~masked_old["hidden_mask"]
        masked = {"old_dual": masked_old,
                  "fixed_pulses": adaptive(*args, observed_mask=visible, fixed=True),
                  "adaptive": adaptive(*args, observed_mask=visible)}
    truth = np.asarray(actual[idx], dtype=np.float64)
    hidden = ~visible.numpy()
    result = dict(rows=len(idx), first_origin=int(origins[idx[0]]),
                  latest_label_available=int(origins[idx[-1]] + 24 * 3600),
                  selection_used=False, full_visible={}, hidden_only={}, hidden_cells=int(hidden.sum()))
    from .dual_history_memory import fixed_smooth
    result["full_visible"]["fixed_smooth"] = reconstruction_metrics(fixed_smooth(truth), truth)
    for name, fields in full.items():
        prediction, event = (fields[k].numpy() * 300 for k in ("reconstruction", "event"))
        result["full_visible"][name] = reconstruction_metrics(prediction, truth, event)
        masked_prediction = masked[name]["reconstruction"].numpy() * 300
        error = np.abs(masked_prediction - truth)
        tail = hidden & (truth > 300)
        result["hidden_only"][name] = dict(mae=float(error[hidden].mean()),
            spike_mae=float(error[tail].mean()) if tail.any() else None,
            spike_cells=int(tail.sum()))
    return result


def run_region(project, source, output, region, identity, config, baseline_config):
    started = time.monotonic()
    arrays, meta, bq, z, anchor, rows, frozen_hashes, gaps = load_frozen_region(project, source, region, identity)
    origins, actual = arrays["origin_unix"], arrays["actual"]
    lags = price_lags(arrays, meta, baseline_config)
    print(f"{region}: fitting old and adaptive geometry on July-August only", flush=True)
    old, old_audit = fit_historical_dualfield(origins, actual, z, lags, fit_cutoff=FIT_CUTOFF,
                                             seed=2026, config=DualHistoryConfig())
    adaptive, audit = fit_adaptive_dualfield(origins, actual, z, lags, seed=2026, config=config)
    for name, model, model_audit in (("old", old, old_audit), ("adaptive", adaptive, audit)):
        model.save_checkpoint(output / "models" / f"{region}_{name}.pt", model_audit)
    reconstruction = assess_reconstruction(origins, actual, z, lags, old, adaptive)
    data.atomic_json(output / f"{region}_reconstruction.json", reconstruction)
    print(f"{region}: reconstruction frozen; building causal historical corrections", flush=True)
    memories = {name: build_corrections(origins, actual, anchor, z, lags, model)
                for name, model in (("adaptive", adaptive), ("fixed", FixedGeometry(adaptive)), ("old", old))}
    reference_memory = memories["adaptive"]
    for name, memory in memories.items():
        if not np.array_equal(memory["matched_indices"], reference_memory["matched_indices"]):
            raise ValueError("Representation comparisons changed historical candidate IDs")
        for key in ("rawmatched", "recentraw", "fixedsmooth"):
            if not np.array_equal(memory["corrections"][key], reference_memory["corrections"][key]):
                raise ValueError("Representation comparisons changed a frozen control")
        data.atomic_npz(output / "features" / f"{region}_{name}_encoded.npz", **memory["encoded_curves"])
        data.atomic_npz(output / "features" / f"{region}_{name}_corrections.npz", origin_unix=origins,
                        matched_indices=memory["matched_indices"], recent_indices=memory["recent_indices"],
                        **memory["corrections"])
    mappings = {"H_recent": ("adaptive", "recentraw"), "H_raw": ("adaptive", "rawmatched"),
                "H_smooth": ("adaptive", "fixedsmooth"), "H_old_dual": ("old", "dualfull"),
                "H_fixed_pulses": ("fixed", "dualfull"), "H_adaptive_trend": ("adaptive", "dualtrend"),
                "H_adaptive_full": ("adaptive", "dualfull")}
    predictions, fits = {"B": bq, "C_roll": anchor}, {}
    for name, (memory_name, key) in mappings.items():
        correction = memories[memory_name]["corrections"][key]
        fits[name] = fit_history_weights(origins, actual, anchor, correction)
        predictions[name] = apply_history_correction(anchor, correction, fits[name]["alpha"])
    predictions["H_adaptive_event_off"] = apply_history_correction(anchor,
        reference_memory["corrections"]["dualtrend"], fits["H_adaptive_full"]["alpha"])
    idx = rows["v2"]
    subset = {k: arrays[k][idx] for k in ("origin_unix", "target_time_unix", "actual")}
    scored, changes, losses = {}, {}, {}
    reference = None
    for name, prediction in predictions.items():
        artifact = diagnostics.build_artifact(name, prediction[idx], subset, meta, "adaptive_dual_v2")
        if reference is not None:
            assert_comparable(reference, artifact["artifact"], allow_different_seed=False)
        reference = artifact["artifact"]
        save_prediction_artifact(output / "predictions" / f"{region}_{name}.npz", artifact["artifact"])
        scored[name] = dict(metrics=artifact["scores"], bands=diagnostics.band_decomposition(prediction[idx, :, 2], actual[idx]),
                            spike=diagnostics.spike_alarm_diagnostics(prediction[idx], actual[idx]))
        delta = prediction[idx, :, 2] - anchor[idx, :, 2]
        changes[name] = dict(fraction=float(np.mean(np.abs(delta) > 1e-9)), mean_abs=float(np.abs(delta).mean()))
        losses[name] = diagnostics.per_origin_losses(prediction[idx], actual[idx])
    for path, digest in frozen_hashes.items():
        if data.sha256_file(source / path) != digest:
            raise ValueError("Frozen parent artifact changed")
    result = dict(region=region, seed=2026, scores=scored, changes=changes, fits=fits,
                  encoder=audit, old_encoder=old_audit, reconstruction=reconstruction,
                  history_audits={k: v["audit"] for k, v in memories.items()}, frozen_hashes=frozen_hashes,
                  reconstructed_boundary_origins=gaps,
                  checkpoint_hashes={name: data.sha256_file(output / "models" / f"{region}_{name}.pt") for name in ("old", "adaptive")},
                  unique_test_spike_hours=len(np.unique(subset["target_time_unix"][subset["actual"] > 300])),
                  wall_seconds=time.monotonic() - started)
    data.atomic_json(output / f"{region}.json", result)
    print(f"{region}: " + ", ".join(f"{m}={scored[m]['metrics']['mae']:.4f}" for m in MODELS), flush=True)
    return result, origins[idx], losses


def summarize(results):
    if tuple(r["region"] for r in results) != REGIONS:
        raise ValueError("All three ordered region cells required")
    macro = {m: {k: float(np.mean([r["scores"][m]["metrics"][k] for r in results])) for k in METRICS} for m in MODELS}
    improvement = {m: {k: macro[m][k] - macro["H_adaptive_full"][k] for k in METRICS} for m in MODELS if m != "H_adaptive_full"}
    wins = {m: all(v > 1e-9 for v in delta.values()) for m, delta in improvement.items()}
    region_ok = all(r["scores"]["H_adaptive_full"]["metrics"][k] <= r["scores"]["C_roll"]["metrics"][k] + 1e-9
                    for r in results for k in METRICS)
    return dict(macro=macro, reference_minus_adaptive=improvement, macro_mae_and_qs_wins=wins,
                per_region_non_regression=region_ok,
                incremental_value_pass=region_ok and all(wins[m] for m in ("C_roll", "H_old_dual", "H_fixed_pulses", "H_recent", "H_raw", "H_smooth")),
                scope="Retrospective December2022 development; fixed settings, no final test or blind-test claim")


def report(output, results, summary):
    lines = ["# Adaptive historical dual fields: fixed contribution check", "",
             "Seed2026, three regions. Both encoders fit July-August, September reconstruction diagnosis, October-November correction fitting, December2022 retrospective scoring.",
             "No final test accessed. Historical reconstruction uses already-known prices; it is not future spike prediction. First-publication timestamps remain independently unaudited.", "",
             "| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |", "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        values = [r["scores"][model]["metrics"]["mae"] for r in results] + [summary["macro"][model][k] for k in METRICS]
        lines.append(f"| {model} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", f"Incremental value gate: **{summary['incremental_value_pass']}**; all-region non-regression vs C_roll: **{summary['per_region_non_regression']}**.", "",
              "| September full-visible reconstruction | NSW1 MAE | QLD1 MAE | TAS1 MAE |", "|---|---:|---:|---:|"]
    for name in ("fixed_smooth", "old_dual", "fixed_pulses", "adaptive"):
        lines.append(f"| {name} | " + " | ".join(f"{r['reconstruction']['full_visible'][name]['mae_total']:.6f}" for r in results) + " |")
    lines += ["", "| September hidden-only reconstruction (same 20% mask) | NSW1 MAE | QLD1 MAE | TAS1 MAE |", "|---|---:|---:|---:|"]
    for name in ("old_dual", "fixed_pulses", "adaptive"):
        lines.append(f"| {name} | " + " | ".join(f"{r['reconstruction']['hidden_only'][name]['mae']:.6f}" for r in results) + " |")
    for r in results:
        lines += ["", f"## {r['region']}", f"- Learned parameters: {r['encoder']['trainable_parameters']}; fit rows: {r['encoder']['fit_rows']}; latest fit label: {r['encoder']['latest_fit_label_available']}.",
                  f"- Adaptive correction weights: {r['fits']['H_adaptive_full']['alpha']}",
                  f"- Changed predictions vs C_roll: {r['changes']['H_adaptive_full']}",
                  f"- Unique December actual spike hours: {r['unique_test_spike_hours']}; wall time: {r['wall_seconds']:.1f}s."]
    lines += ["", "All matched variants use identical historical neighbor IDs and four downstream weights. Event-off reuses the full head. Fixed pulses share the analytic solver and peak proposals, isolating geometry learning.",
              "Regional JSON includes tail errors, event localization, hidden-only diagnostics and maturity audits. Bootstrap intervals are descriptive: December supplies few independent weekly blocks."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/forecasting/v5_calibrated_history_20261015_r3")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    source, output = (project / args.source).resolve(), (project / args.output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous trials; use a new output path")
    config_path = project / "configs/full_v5/adaptive_dual_history.yaml"
    config = yaml.safe_load(config_path.read_text())
    expected = dict(protocol="adaptive-historical-dual-fields-20261017", seed=2026, regions=list(REGIONS),
                    encoder=asdict(AdaptiveDualConfig()), encoder_fit_before="2022-09-01",
                    reconstruction_assessment_before="2022-10-01", correction_fit_before="2022-12-01",
                    assessment="retrospective_December2022")
    if config != expected:
        raise ValueError("Fixed experiment configuration changed")
    identity = data.read_json(source / "identity/identity.json")
    for block in ("sources", "tests", "reused_sources"):
        for path, digest in identity[block].items():
            if data.sha256_file(project / path) != digest:
                raise ValueError(f"Frozen predictive implementation changed: {path}")
    baseline_config_path = project / "outputs/forecasting/aligned_baselines_20261012/config.yaml"
    if data.sha256_file(baseline_config_path) != identity["baseline_run"]["original_config_hash"]:
        raise ValueError("Original baseline input configuration changed")
    baseline_config = yaml.safe_load(baseline_config_path.read_text())
    output.mkdir(parents=True)
    for folder in ("models", "predictions", "features"):
        (output / folder).mkdir()
    files = list((project / "forecasting/full_v5").glob("*.py"))
    files += list((project / "tests").glob("test_full_v5_*history*.py"))
    files += [project / "tests/test_full_v5_adaptive_dual_experiment.py"]
    files += list((project / "docs").glob("full_v5_adaptive_dual*20261017.md"))
    files += [config_path, baseline_config_path, project / "src/dualfield/core.py", project / "forecasting/aligned_scoring.py"]
    hashes = {}
    for path in files:
        rel = str(path.relative_to(project))
        hashes[rel] = data.sha256_file(path)
        target = output / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    data.atomic_json(output / "identity.json", dict(git=data.git_identity(project), source_hashes=hashes,
        parent_identity_sha=data.sha256_file(source / "identity/identity.json"), baseline_config_sha=data.sha256_file(baseline_config_path)))
    completed = [run_region(project, source, output, region, identity, AdaptiveDualConfig(), baseline_config) for region in REGIONS]
    for path, digest in hashes.items():
        if data.sha256_file(project / path) != digest:
            raise ValueError(f"Experiment source changed during run: {path}")
    results = [item[0] for item in completed]
    summary = summarize(results)
    origins = {item[0]["region"]: item[1] for item in completed}
    candidate = {item[0]["region"]: item[2]["H_adaptive_full"] for item in completed}
    uncertainty = {reference: {str(block): diagnostics.paired_block_bootstrap(origins,
        {item[0]["region"]: item[2][reference] for item in completed}, candidate,
        block_hours=block, n_boot=999, seed=2026) for block in (168, 24)}
        for reference in ("B", "C_roll", "H_old_dual", "H_fixed_pulses", "H_adaptive_trend", "H_adaptive_event_off", "H_recent")}
    data.atomic_json(output / "summary.json", summary)
    data.atomic_json(output / "bootstrap.json", uncertainty)
    report(output, results, summary)
    print(f"Complete: incremental_value_pass={summary['incremental_value_pass']}", flush=True)


if __name__ == "__main__":
    main()
