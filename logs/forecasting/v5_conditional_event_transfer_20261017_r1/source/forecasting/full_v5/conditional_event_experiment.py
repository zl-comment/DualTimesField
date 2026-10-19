"""Frozen October-fit / November-check conditional history event transport."""
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
from .adaptive_dual_history import AdaptiveHistoricalDualField
from .conditional_event_memory import build_transfer_memory
from .conditional_transfer_gate import GateConfig, ConditionalTransferGate, fit_transfer_gate, FIT_CUTOFF
from .dual_history_experiment import price_lags, REGIONS, METRICS
from .history_correction import apply_history_correction, fit_history_weights
from .history_quickcheck import load_frozen_region

END = schema.unix_timestamp("2022-12-01")
MODELS = ("B", "C_roll", "H_recent", "H_previous_global", "T_only", "TE_unshifted", "TE_full",
          "TE_fixed_pulses", "TE_event_off", "TE_time_shift_off")


def check_shared_candidates(memories):
    ref = memories["full"]
    for name, memory in memories.items():
        for key in ("trend_ids", "event_record_indices", "event_source_h", "event_proposed_target_h", "event_weights", "event_distances"):
            if not np.array_equal(ref[key], memory[key]):
                raise ValueError(f"Candidate control changed {name}: {key}")
    for key in ("event_amplitudes", "event_widths", "trend"):
        if not np.array_equal(ref[key], memories["unshifted"][key]):
            raise ValueError(f"Timing control changed more than center: {key}")


def run_region(project, source, encoder_source, output, region, identity, baseline_config):
    start = time.monotonic()
    arrays, meta, bq, z, anchor, _, frozen, gaps = load_frozen_region(project, source, region, identity)
    lags = price_lags(arrays, meta, baseline_config)
    # Exclude December and all windows that have labels unavailable at END,
    # before building any new library or gate fitting/assessment arrays.
    keep = arrays["origin_unix"] + 24 * 3600 < END
    arrays = {k: v[keep] for k, v in arrays.items() if isinstance(v, np.ndarray) and v.ndim and len(v) == len(keep)}
    bq, z, anchor, lags = (a[keep] for a in (bq, z, anchor, lags))
    origins, actual = arrays["origin_unix"], arrays["actual"]
    parent = data.read_json(encoder_source / f"{region}.json")
    checkpoint = encoder_source / "models" / f"{region}_adaptive.pt"
    if data.sha256_file(checkpoint) != parent["checkpoint_hashes"]["adaptive"]:
        raise ValueError("Frozen historical encoder changed")
    encoder, encoder_audit = AdaptiveHistoricalDualField.load_checkpoint(checkpoint)
    if encoder_audit["latest_fit_label_available"] >= encoder_audit["fit_cutoff"] or encoder.fit_cutoff > FIT_CUTOFF:
        raise ValueError("Historical encoder fit overlaps transfer assessment")
    torch.set_num_threads(2)
    print(f"{region}: frozen encoder; constructing trend and state-aligned event proposals", flush=True)
    memories = {"full": build_transfer_memory(origins, actual, anchor, z, lags, encoder),
                "unshifted": build_transfer_memory(origins, actual, anchor, z, lags, encoder, shift=False),
                "fixed": build_transfer_memory(origins, actual, anchor, z, lags, encoder, fixed=True)}
    check_shared_candidates(memories)
    for name, memory in memories.items():
        data.atomic_npz(output / "features" / f"{region}_{name}.npz", origin_unix=origins,
                        **{key: value for key, value in memory.items() if isinstance(value, np.ndarray)})
    # Saved corrections were constructed causally; only their October weights
    # are refitted. No previous October-November-fitted weights are reused.
    old_path = encoder_source / "features" / f"{region}_adaptive_corrections.npz"
    with np.load(old_path) as raw:
        idx = np.searchsorted(raw["origin_unix"], origins)
        if not np.array_equal(raw["origin_unix"][idx], origins):
            raise ValueError("Previous correction origin mismatch")
        controls = {"H_recent": raw["recentraw"][idx], "H_previous_global": raw["dualfull"][idx]}
    predictions, fits, heads, parts = {"B": bq, "C_roll": anchor}, {}, {}, {}
    for name, correction in controls.items():
        fit = fit_history_weights(origins, actual, anchor, correction, fit_cutoff=FIT_CUTOFF)
        fits[name] = fit
        predictions[name] = apply_history_correction(anchor, correction, fit["alpha"])
    print(f"{region}: fitting four 60-parameter gates on October only", flush=True)
    mapping = {"T_only": ("full", "trend"), "TE_unshifted": ("unshifted", "full"),
               "TE_full": ("full", "full"), "TE_fixed_pulses": ("fixed", "full")}
    for name, (memory_name, mode) in mapping.items():
        memory = memories[memory_name]
        head, audit = fit_transfer_gate(origins, actual, anchor, memory["features"], memory["trend"], memory["event"],
                                       cutoff=FIT_CUTOFF, seed=2026, config=GateConfig(), mode=mode)
        head.save_checkpoint(output / "models" / f"{region}_{name}.json", audit)
        heads[name], fits[name] = head, audit
        parts[name] = head.predict(memory["features"], memory["trend"], memory["event"], anchor)
        predictions[name] = parts[name]["quantiles"]
    full, head = memories["full"], heads["TE_full"]
    parts["TE_event_off"] = head.predict(full["features"], full["trend"], full["event"], anchor, event_off=True)
    parts["TE_time_shift_off"] = head.predict(full["features"], full["trend"], memories["unshifted"]["event"], anchor)
    for name in ("TE_event_off", "TE_time_shift_off"):
        predictions[name] = parts[name]["quantiles"]
        for key in ("gate_trend", "gate_event"):
            if not np.array_equal(parts["TE_full"][key], parts[name][key]):
                raise ValueError("Fixed-head intervention changed adoption gates")
    assessment = origins >= FIT_CUTOFF
    subset = {k: arrays[k][assessment] for k in ("origin_unix", "actual", "target_time_unix")}
    scored, losses, activation = {}, {}, {}
    ref = None
    for name, q in predictions.items():
        artifact = diagnostics.build_artifact(name, q[assessment], subset, meta, "conditional_event_november")
        if ref is not None:
            assert_comparable(ref, artifact["artifact"], allow_different_seed=False)
        ref = artifact["artifact"]
        save_prediction_artifact(output / "predictions" / f"{region}_{name}.npz", artifact["artifact"])
        scored[name] = dict(metrics=artifact["scores"], bands=diagnostics.band_decomposition(q[assessment, :, 2], actual[assessment]),
                            spike=diagnostics.spike_alarm_diagnostics(q[assessment], actual[assessment]))
        losses[name] = diagnostics.per_origin_losses(q[assessment], actual[assessment])
    for name, fields in parts.items():
        data.atomic_npz(output / "gates" / f"{region}_{name}.npz", origin_unix=origins[assessment],
                        **{k: v[assessment] for k, v in fields.items() if k != "quantiles"})
        activation[name] = {key: dict(mean=float(fields[key][assessment].mean()),
            positive_fraction=float(np.mean(fields[key][assessment] > 1e-9))) for key in ("gate_trend", "gate_event")}
        activation[name]["mean_abs_shift"] = float(np.abs(fields["delta"][assessment]).mean())
    for path, digest in frozen.items():
        if data.sha256_file(source / path) != digest:
            raise ValueError("Frozen parent artifact changed")
    result = dict(region=region, seed=2026, scores=scored, fits=fits, activation=activation,
                  feature_names=full["feature_names"], memory_audits={k: v["audit"] for k, v in memories.items()},
                  encoder_checkpoint_sha256=data.sha256_file(checkpoint), encoder_audit=encoder_audit,
                  reused_corrections_sha256=data.sha256_file(old_path), frozen_hashes=frozen,
                  reconstructed_boundary_origins=gaps, assessment_rows=int(assessment.sum()),
                  first_assessment_origin=int(origins[assessment][0]),
                  latest_assessment_label_available=int(origins[assessment][-1] + 24 * 3600),
                  unique_spike_hours=len(np.unique(subset["target_time_unix"][subset["actual"] > 300])),
                  gate_checkpoint_hashes={name: data.sha256_file(output / "models" / f"{region}_{name}.json") for name in mapping},
                  shared_event_candidates=True, shared_timing_control_gates=True, wall_seconds=time.monotonic() - start)
    data.atomic_json(output / f"{region}.json", result)
    print(f"{region}: " + ", ".join(f"{m}={scored[m]['metrics']['mae']:.4f}" for m in MODELS), flush=True)
    return result, origins[assessment], losses


def summarize(results):
    if tuple(r["region"] for r in results) != REGIONS:
        raise ValueError("Three ordered region results required")
    macro = {name: {metric: float(np.mean([r["scores"][name]["metrics"][metric] for r in results]))
                   for metric in METRICS} for name in MODELS}
    wins = {name: all(macro["TE_full"][metric] < macro[name][metric] - 1e-9 for metric in METRICS)
            for name in MODELS if name != "TE_full"}
    region_ok = all(r["scores"]["TE_full"]["metrics"][metric] <= r["scores"]["C_roll"]["metrics"][metric] + 1e-9
                    for r in results for metric in METRICS)
    return dict(macro=macro, macro_wins_mae_and_qs=wins, per_region_non_regression=region_ok,
                incremental_value_pass=region_ok and all(wins[name] for name in ("C_roll", "H_recent", "H_previous_global", "TE_fixed_pulses", "T_only")),
                scope="Retrospective November2022; fixed October fitting; no December scoring or final test")


def report(output, results, summary):
    lines = ["# Conditional historical event transfer", "", "Frozen July-August encoder; October-only gate fitting; November2022 retrospective development scores, seed2026. No December assessment or final test.", "",
             "| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |", "|---|---:|---:|---:|---:|---:|"]
    for name in MODELS:
        values = [r["scores"][name]["metrics"]["mae"] for r in results] + [summary["macro"][name][m] for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", f"Incremental-value gate: **{summary['incremental_value_pass']}**. All-region non-regression: **{summary['per_region_non_regression']}**.",
              "State-aligned event transport is a conditional forecasting hypothesis, not identification of marginal generators or a causal market mechanism."]
    for r in results:
        lines += ["", f"## {r['region']}",
                  f"- Assessment origins: {r['assessment_rows']}; unique actual spike hours: {r['unique_spike_hours']}.",
                  f"- Full gate fit: {r['fits']['TE_full']['fit_rows']} mature origins, parameters {r['fits']['TE_full']['trainable_parameters']}, selected null={r['fits']['TE_full']['selected_null']}.",
                  f"- Full gate activation: {r['activation']['TE_full']}.",
                  f"- Shared historical candidates across geometry/timing controls: {r['shared_event_candidates']}; all memory maturity violations zero.",
                  f"- Wall seconds: {r['wall_seconds']:.1f}."]
    lines += ["", "T_only and TE_unshifted refit their own gates. T_only retains event-related inputs but removes the direct event term; it is not an event-information-free ablation. TE_event_off and TE_time_shift_off reuse the exact full model gates and features. Candidate identities, distances and weights are shared across timing and geometry controls.",
              "All quantiles shift together, retain q50 point scoring, and corrections stay within +/-150. Actual first-publication timestamp audits remain limited to existing archive proxies. Results cannot be compared numerically to the prior December table as though the evaluation period were unchanged."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    output = (project / args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    config_path = project / "configs/full_v5/conditional_event_transfer.yaml"
    config = yaml.safe_load(config_path.read_text())
    expected = dict(protocol="conditional-event-transfer-20261017", regions=list(REGIONS), seed=2026,
                    encoder_source="outputs/forecasting/v5_adaptive_dual_20261017_r2", gate=asdict(GateConfig()),
                    gate_fit_cutoff="2022-11-01", assessment_end="2022-12-01", assessment="retrospective_November2022")
    if config != expected:
        raise ValueError("Fixed experiment configuration changed")
    source = project / "outputs/forecasting/v5_calibrated_history_20261015_r3"
    encoder_source = project / config["encoder_source"]
    identity = data.read_json(source / "identity/identity.json")
    encoder_identity = data.read_json(encoder_source / "identity.json")
    for path, digest in encoder_identity["source_hashes"].items():
        if data.sha256_file(project / path) != digest:
            raise ValueError(f"Frozen parent implementation changed: {path}")
    for block in ("sources", "tests", "reused_sources"):
        for path, digest in identity[block].items():
            if data.sha256_file(project / path) != digest:
                raise ValueError(f"Frozen predictive implementation changed: {path}")
    baseline_path = project / "outputs/forecasting/aligned_baselines_20261012/config.yaml"
    if data.sha256_file(baseline_path) != identity["baseline_run"]["original_config_hash"]:
        raise ValueError("Frozen baseline configuration changed")
    baseline_config = yaml.safe_load(baseline_path.read_text())
    # Verify archived reused corrections before reading them, not just hash
    # whatever happens to be present during this run.
    manifest = data.read_json(project / "logs/forecasting/v5_adaptive_dual_20261017_r2/artifact_manifest.json")["artifacts"]
    for region in REGIONS:
        for path in (f"features/{region}_adaptive_corrections.npz", f"models/{region}_adaptive.pt"):
            if data.sha256_file(encoder_source / path) != manifest[path]["sha256"]:
                raise ValueError("Reused causal corrections or encoder changed")
    output.mkdir(parents=True)
    for folder in ("models", "predictions", "features", "gates"):
        (output / folder).mkdir()
    files = list((project / "forecasting/full_v5").glob("*.py"))
    files += list((project / "tests").glob("test_full_v5_conditional*.py"))
    files += [config_path, baseline_path, project / "docs/full_v5_conditional_event_transfer_20261017.md",
              project / "forecasting/aligned_scoring.py", project / "src/dualfield/core.py"]
    hashes = {}
    for path in files:
        relative = str(path.relative_to(project))
        hashes[relative] = data.sha256_file(path)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    data.atomic_json(output / "identity.json", dict(git=data.git_identity(project), source_hashes=hashes,
        parent_identity_sha=data.sha256_file(source / "identity/identity.json"),
        encoder_identity_sha=data.sha256_file(encoder_source / "identity.json")))
    completed = [run_region(project, source, encoder_source, output, region, identity, baseline_config) for region in REGIONS]
    for path, digest in hashes.items():
        if data.sha256_file(project / path) != digest:
            raise ValueError(f"Source changed during run: {path}")
    results = [v[0] for v in completed]
    summary = summarize(results)
    origins = {v[0]["region"]: v[1] for v in completed}
    candidate = {v[0]["region"]: v[2]["TE_full"] for v in completed}
    uncertainty = {name: {str(block): diagnostics.paired_block_bootstrap(origins,
        {v[0]["region"]: v[2][name] for v in completed}, candidate, block_hours=block, n_boot=999, seed=2026)
        for block in (168, 24)} for name in ("C_roll", "H_recent", "T_only", "TE_fixed_pulses", "TE_event_off", "TE_time_shift_off")}
    data.atomic_json(output / "summary.json", summary)
    data.atomic_json(output / "bootstrap.json", uncertainty)
    report(output, results, summary)
    print(f"Complete: incremental_value_pass={summary['incremental_value_pass']}", flush=True)


if __name__ == "__main__":
    main()
