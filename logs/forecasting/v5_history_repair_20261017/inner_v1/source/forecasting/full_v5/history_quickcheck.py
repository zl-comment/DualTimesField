"""Retrospective V1-fit / V2-check for an opt-in history-only correction.

Run with ``python -m forecasting.full_v5.history_quickcheck --output ...``.
Never loads the final test split, retrains C_roll, or changes its saved files.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..aligned_scoring import assert_comparable, save_prediction_artifact
from . import calibrator, data, diagnostics, history, schema
from .history_correction import (
    apply_history_correction, compute_history_corrections, fit_history_weights,
)

REGIONS = ("NSW1", "QLD1", "TAS1")
MODELS = ("B", "C_roll", "H_recent", "H_matched")


def checked_file(root, relative, manifest):
    path = root / relative
    if relative not in manifest or data.sha256_file(path) != manifest[relative]:
        raise ValueError(f"Source artifact changed: {path}")
    return path


def load_frozen_region(project, source, region, identity):
    """Reuse exact scored predictions; reconstruct only unscored boundary rows."""
    arrays, meta = data.load_cache_split(
        project / "outputs/forecasting/aligned_baselines_20261012/data_cache",
        region, "calibration", identity["cache_hashes"][region])
    origins = arrays["origin_unix"]
    baseline_rel = f"baseline_predictions/{region}_calibration.npz"
    baseline_digest = identity["baseline_predictions"][region]["file_sha256"]
    if data.sha256_file(source / baseline_rel) != baseline_digest:
        raise ValueError("Frozen baseline file changed")
    bq = data.load_prepared_baseline(source, region, "calibration")
    with np.load(source / "baseline_predictions" / f"{region}_calibration.npz") as a:
        if not np.array_equal(a["origin_unix"], origins):
            raise ValueError("Frozen baseline origin grid mismatch")
        if not np.array_equal(a["target_time_unix"], arrays["target_time_unix"]):
            raise ValueError("Frozen baseline target grid mismatch")
    z = data.build_z(bq, arrays)
    marker = data.read_json(source / "development" / f"done_{region}.json")
    manifest = marker["outputs"]
    model_rel = f"development/models/C_roll_{region}.json"
    model_path = checked_file(source, model_rel, manifest)
    model = calibrator.CalibratorModel.from_json(data.read_json(model_path), schema.CONTROL_SPECS)
    # No earlier C_roll predictions can enter the new memory. Fill them with
    # frozen B only to maintain finite full-H2 arrays accepted by the engine.
    anchor = bq.copy()
    seen = np.zeros(len(origins), dtype=bool)
    hashes = {model_rel: data.sha256_file(model_path), baseline_rel: baseline_digest}
    rows = {}
    for segment in ("v1", "v2"):
        rel = f"development/predictions/C_roll/{region}_{segment}.npz"
        path = checked_file(source, rel, manifest)
        hashes[rel] = data.sha256_file(path)
        with np.load(path, allow_pickle=False) as a:
            idx = np.searchsorted(origins, a["origin_unix"])
            if not np.array_equal(origins[idx], a["origin_unix"]):
                raise ValueError("Anchor sample grid mismatch")
            if not np.array_equal(arrays["actual"][idx], a["actual"]):
                raise ValueError("Anchor label mismatch")
            anchor[idx] = a["quantiles"]
            seen[idx] = True
            rows[segment] = idx
    start = schema.segment_bounds()["v1"][0]
    gap = np.flatnonzero((origins >= start) & ~seen)
    library = history.HistoryLibrary.from_segment(origins, arrays["actual"], bq, z)
    # C_roll uses only z and m, so there is no need to rerun original matching.
    for idx in gap:
        m, _ = history.compute_m(library, int(origins[idx]),
                                 np.flatnonzero(library.matured_mask(int(origins[idx]))))
        f = np.concatenate((z[idx], m, np.zeros((24, 2))), axis=-1)[None]
        anchor[idx] = model.predict(f, bq[idx:idx + 1])["quantiles"][0]
    return arrays, meta, bq, z, anchor, rows, hashes, len(gap)


def run_region(project, source, output, region, identity, policy="state"):
    started = time.monotonic()
    arrays, meta, bq, z, anchor, rows, hashes, gap_count = load_frozen_region(
        project, source, region, identity)
    origins = arrays["origin_unix"]
    corrections = compute_history_corrections(origins, arrays["actual"], anchor, z, policy=policy)
    predictions = {"B": bq, "C_roll": anchor}
    fits = {}
    for label, key in (("H_recent", "recent"), ("H_matched", "matched")):
        fits[label] = fit_history_weights(origins, arrays["actual"], anchor, corrections[key])
        predictions[label] = apply_history_correction(anchor, corrections[key], fits[label]["alpha"])
    scores, changes = {}, {}
    v2 = rows["v2"]
    subset = {k: arrays[k][v2] for k in ("actual", "origin_unix", "target_time_unix")}
    reference = None
    for model_id, q in predictions.items():
        scored = diagnostics.build_artifact(model_id, q[v2], subset, meta, "history_repair_v2")
        artifact = scored["artifact"]
        if reference is None:
            reference = artifact
        else:
            assert_comparable(reference, artifact, allow_different_seed=False)
        save_prediction_artifact(output / "predictions" / f"{region}_{model_id}.npz", artifact)
        scores[model_id] = dict(metrics=scored["scores"],
                               bands=diagnostics.band_decomposition(q[v2, :, 2], subset["actual"]))
        diff = q[v2, :, 2] - anchor[v2, :, 2]
        changes[model_id] = dict(fraction=float(np.mean(np.abs(diff) > 1e-9)),
                                 mean_abs=float(np.abs(diff).mean()), max_abs=float(np.abs(diff).max()))
    # Preserve the complete causal feature trace, including zero/warm-up rows.
    keep = origins >= schema.segment_bounds()["v1"][0]
    data.atomic_npz(output / "features" / f"{region}.npz", origin_unix=origins[keep],
                    matched=corrections["matched"][keep], recent=corrections["recent"][keep])
    for rel, digest in hashes.items():
        if data.sha256_file(source / rel) != digest:
            raise ValueError("Frozen artifact mutated during check")
    result = dict(region=region, seed=2026, scores=scores, changes=changes, fits=fits,
                  history_audit=corrections["audit"], frozen_hashes=hashes,
                  reconstructed_boundary_origins=gap_count,
                  elapsed_seconds=time.monotonic() - started)
    data.atomic_json(output / f"{region}.json", result)
    print(f"{region}: " + ", ".join(f"{m} MAE={scores[m]['metrics']['mae']:.4f}" for m in MODELS), flush=True)
    return result


def summarize(results):
    macro = {m: {metric: float(np.mean([r["scores"][m]["metrics"][metric] for r in results]))
                 for metric in ("mae", "quantile_score_proxy")} for m in MODELS}
    metrics = ("mae", "quantile_score_proxy")
    activation = all(r["changes"]["H_matched"]["fraction"] > 0 for r in results)
    improves_anchor = all(macro["H_matched"][m] < macro["C_roll"][m] - 1e-9 for m in metrics)
    regions_ok = all(r["scores"]["H_matched"]["metrics"][m] <=
                     r["scores"]["C_roll"]["metrics"][m] + 1e-9 for r in results for m in metrics)
    improves_recent = all(macro["H_matched"][m] < macro["H_recent"][m] - 1e-9 for m in metrics)
    return dict(macro=macro, mechanism_pass=activation,
                incremental_value_pass=activation and improves_anchor and regions_ok,
                matching_value_pass=improves_recent,
                per_region_non_regression=regions_ok,
                scope="retrospective development only; V1 fit, V2 evaluation; no final test")


def write_report(output, results, summary):
    lines = ["# V5 history-only repair — retrospective development", "",
             "Frozen C_roll; history weights fit October–November 2022; evaluated December 2022.",
             "December has been previously inspected. This is not a blind test or the original V5 gate protocol.", "",
             f"Policy: `{results[0]['history_audit']['policy']}`.", "",
             "| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        values = [r["scores"][m]["metrics"]["mae"] for r in results]
        values += [summary["macro"][m]["mae"], summary["macro"][m]["quantile_score_proxy"]]
        lines.append("| " + m + " | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "## Acceptance", ""]
    for k in ("mechanism_pass", "incremental_value_pass", "matching_value_pass", "per_region_non_regression"):
        lines.append(f"- {k}: **{summary[k]}**")
    for r in results:
        lines += ["", f"## {r['region']}", "",
                  f"- Matched weights: {r['fits']['H_matched']['alpha']}",
                  f"- Recent weights: {r['fits']['H_recent']['alpha']}",
                  f"- Matched forecast change: {r['changes']['H_matched']}",
                  f"- Reconstructed boundary origins: {r['reconstructed_boundary_origins']}",
                  f"- Wall time: {r['elapsed_seconds']:.1f} seconds."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_inner_selection(project, source, output, identity):
    """Only October fit / November scoring may select the follow-up policy."""
    nov1, dec1 = schema.unix_timestamp("2022-11-01"), schema.unix_timestamp("2022-12-01")
    candidates = ("fresh28", "recent_consistent")
    results = {}
    for region in REGIONS:
        arrays, meta, _, z, anchor, _, _, _ = load_frozen_region(project, source, region, identity)
        # All candidate construction, training and scores are restricted to V1.
        keep = arrays["origin_unix"] < dec1
        origins, y = arrays["origin_unix"][keep], arrays["actual"][keep]
        anchor, z = anchor[keep], z[keep]
        rows = data.window_in_segment(origins, nov1, dec1)
        subset = dict(actual=y[rows], origin_unix=origins[rows],
                      target_time_unix=arrays["target_time_unix"][keep][rows])
        scores = {"C_roll": diagnostics.build_artifact("C_roll", anchor[rows], subset,
                                                     meta, "history_inner_november")["scores"]}
        fits = {}
        for policy in candidates:
            corrections = compute_history_corrections(origins, y, anchor, z, policy=policy)
            fits[policy] = fit_history_weights(origins, y, anchor, corrections["matched"], fit_cutoff=nov1)
            q = apply_history_correction(anchor, corrections["matched"], fits[policy]["alpha"])
            scores[policy] = diagnostics.build_artifact(policy, q[rows], subset,
                                                       meta, "history_inner_november")["scores"]
        results[region] = dict(scores=scores, fits=fits)
        print(f"inner V1 {region}: " + ", ".join(f"{p} MAE={scores[p]['mae']:.4f}"
                                                     for p in ("C_roll",) + candidates), flush=True)
    macro = {p: {metric: float(np.mean([results[r]["scores"][p][metric] for r in REGIONS]))
                 for metric in ("mae", "quantile_score_proxy")} for p in ("C_roll",) + candidates}
    eligible = [p for p in candidates if macro[p]["quantile_score_proxy"] <= macro["C_roll"]["quantile_score_proxy"]]
    selected = min(eligible, key=lambda p: macro[p]["mae"]) if eligible else None
    selection = dict(selected_policy=selected, macro=macro, regions=results,
                     parent_identity_sha=data.sha256_file(source / "identity/identity.json"),
                     fit_cutoff=nov1, score_end_exclusive=dec1,
                     scope="October fit, November scoring; December excluded from all candidate calculations")
    data.atomic_json(output / "selection.json", selection)
    print(json.dumps(dict(selected_policy=selected, macro=macro), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/forecasting/v5_calibrated_history_20261015_r3")
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--select-policy", action="store_true")
    mode.add_argument("--selection", help="Frozen inner-V1 selection.json for the single follow-up")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    source, output = (project / args.source).resolve(), (project / args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output path; no silent reruns")
    identity = data.read_json(source / "identity/identity.json")
    policy, selection_digest = "state", None
    if args.selection:
        selection_path = project / args.selection
        selection = data.read_json(selection_path)
        policy = selection["selected_policy"]
        if policy not in ("fresh28", "recent_consistent"):
            raise ValueError("Inner V1 selection did not qualify a follow-up policy")
        if selection["parent_identity_sha"] != data.sha256_file(source / "identity/identity.json"):
            raise ValueError("Selection came from a different parent experiment")
        selection_digest = data.sha256_file(selection_path)
    # All parent inputs/implementation remain identical, including calibration
    # and scoring. Newly added opt-in modules are bound separately below.
    for block in ("sources", "tests", "reused_sources"):
        for path, digest in identity[block].items():
            if data.sha256_file(project / path) != digest:
                raise ValueError(f"Parent implementation changed: {path}")
    output.mkdir(parents=True)
    (output / "predictions").mkdir()
    (output / "features").mkdir()
    extra = ("forecasting/full_v5/history_correction.py", "forecasting/full_v5/history_quickcheck.py",
             "tests/test_full_v5_history_correction.py", "docs/full_v5_history_repair_20261017.md")
    for path in extra:
        target = output / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / path, target)
    data.atomic_json(output / "identity.json", dict(parent_identity_sha=data.sha256_file(source / "identity/identity.json"),
                     parent_run=str(source), git=data.git_identity(project),
                     policy=policy, inner_selection_sha=selection_digest,
                     added_source_hashes={p: data.sha256_file(project / p) for p in extra}))
    if args.select_policy:
        run_inner_selection(project, source, output, identity)
        return
    results = [run_region(project, source, output, r, identity, policy=policy) for r in REGIONS]
    summary = summarize(results)
    data.atomic_json(output / "summary.json", summary)
    write_report(output, results, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
