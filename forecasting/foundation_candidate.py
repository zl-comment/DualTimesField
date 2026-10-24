"""Candidate comparison report: foundation batch vs the current main table.

Reads the completed foundation run (verified by build_aligned_report) and the
existing 21-output main comparison CSVs strictly read-only, then writes a
candidate report with explicit grouping labels:

  table            Q / P            output capability
  input_scope      full_public / restricted_* (per registry declaration)
  pretraining      unaudited_overlap flag on every foundation row
  joint_methods    Full_joint & co. keep their wider-information label

Outputs (candidate/): main_candidate.csv, regional.csv, macro_per_seed.csv,
per_seed.csv, provenance.json. The existing main table directory is never
written to; nothing is re-selected or overwritten — this is a candidate for
Codex review, not a new main table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from forecasting.aligned_experiment import read_json, sha
from forecasting.aligned_report import build_aligned_report
from forecasting.common.foundation_registry import FOUNDATION_REGISTRY

REGIONS = ("NSW1", "QLD1", "TAS1")
SEEDS = (2026, 2027, 2028)
METRICS = ("mae", "rmse_pooled", "quantile_score_proxy", "coverage_80", "coverage_90",
           "interval_width_80", "interval_width_90", "crossing_rate")
DEFAULT_MAIN = Path("/root/workspace/Code/DualTimesField/logs/forecasting/main_comparison_20261023")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{path.stat().st_size if path.exists() else 0}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _foundation_rows(report: dict, run_root: Path, manifest: dict) -> tuple[list, list, list, list]:
    main, regional, macro, per_seed = [], [], [], []
    for model, row in report["tables"]["Q"].items():
        _append_model(main, regional, macro, per_seed, model, row, "Q", run_root, manifest)
    for model, row in report["tables"]["P"].items():
        _append_model(main, regional, macro, per_seed, model, row, "P", run_root, manifest)
    return main, regional, macro, per_seed


def _append_model(main, regional, macro, per_seed, model, row, table, run_root, manifest):
    spec = FOUNDATION_REGISTRY[model]
    identity = dict(
        model=model, role="foundation_zero_shot_or_frozen" if model != "tabpfn_v3"
        else "foundation_tabular_label_context",
        table=table, input_scope=spec.input_scope,
        output_capability=("native_five_quantiles" if table == "Q" else "point_only_native_quantile_subset"),
        pretraining_overlap="unaudited",
        seed_scope=spec.seed_scope,
        source_run=manifest["run_id"], source_hash=manifest["source_hash"],
        config_hash=manifest["config_hash"])
    region_scores = [row["regions"][region]["scores"] for region in REGIONS]
    for region in REGIONS:
        scores = row["regions"][region]["scores"]
        seed_summary = scores["seed_summary"]
        regional.append(dict(identity, region=region,
                             **{m: seed_summary[m]["mean"] for m in METRICS if m in seed_summary},
                             **{m + "_std": seed_summary[m]["std"] for m in METRICS if m in seed_summary},
                             seed_evidence=row["regions"][region]["seed_evidence"]["interpretation"]))
        for seed, per_seed_scores in scores["per_seed"].items():
            per_seed.append(dict(identity, region=region, seed=int(seed),
                                 **{m: per_seed_scores[m] for m in METRICS if m in per_seed_scores}))
    available = [m for m in METRICS if all(m in scores["seed_summary"] for scores in region_scores)]
    main.append(dict(identity,
        **{m + "_mean": float(np.mean([scores["seed_summary"][m]["mean"] for scores in region_scores]))
           for m in available},
        **{m + "_std": float(np.std([scores["seed_summary"][m]["mean"] for scores in region_scores], ddof=1))
           for m in available}))
    for seed in SEEDS:
        macro.append(dict(identity, seed=seed,
            **{m: float(np.mean([scores["per_seed"][str(seed)][m] for scores in region_scores]))
               for m in available
               if all(str(seed) in scores["per_seed"] and m in scores["per_seed"][str(seed)]
                      for scores in region_scores)}))


def _existing_rows(directory: Path):
    main = _rows(directory / "main.csv")
    regional = _rows(directory / "regional.csv")
    macro = _rows(directory / "macro_per_seed.csv")
    per_seed = _rows(directory / "per_seed.csv")
    for rows in (main, regional, macro, per_seed):
        for row in rows:
            row["source_run"] = "main_comparison_20261023 (read-only)"
            row["pretraining_overlap"] = row.get("pretraining_overlap", "")
            row.setdefault("table", "Q" if row.get("model", "").startswith(("baseline", "Full", "Anchor", "B_", "event", "time")) else "")
    return main, regional, macro, per_seed


def build(run_root: Path, main_directory: Path, output: Path) -> dict:
    run_root = run_root.resolve()
    manifest = read_json(run_root / "jobs.json")
    report = build_aligned_report(run_root)
    if not report["comparison_available"]:
        raise ValueError("Foundation run is incomplete; a partial matrix publishes no ranking")
    foundation = _foundation_rows(report, run_root, manifest)
    existing = _existing_rows(main_directory)
    files = {}
    for name, rows in zip(("main_candidate.csv", "regional.csv", "macro_per_seed.csv", "per_seed.csv"),
                          (foundation[0] + existing[0], foundation[1] + existing[1],
                           foundation[2] + existing[2], foundation[3] + existing[3])):
        _write_csv(output / name, rows)
        files[name] = sha(output / name)
    provenance = {
        "candidate_for": "Codex review; not a new main table",
        "foundation_run": {
            "run_root": str(run_root), "run_id": manifest["run_id"],
            "source_hash": manifest["source_hash"], "config_hash": manifest["config_hash"],
            "parent": manifest["foundation_from"], "scope": manifest["scope"],
            "declared_scopes": manifest["foundation"]["declared_scopes"],
            "interpreters": manifest["interpreters"], "git_head": manifest["git_head"]},
        "existing_main_table": {
            "directory": str(main_directory), "read_only": True,
            "verification": sha(main_directory / "verification.json")
            if (main_directory / "verification.json").exists() else None},
        "grouping_notes": {
            "equal_input_Q5": "only tabpfn_v3 consumes all 1465 public columns; "
                              "chronos2_zeroshot enters Q5 with restricted covariates (declared)",
            "point_table": "moirai2_zeroshot and timesfm25_xreg have no native q05/q95; "
                           "no tails were extrapolated, copied or fabricated",
            "pretraining": "all four checkpoints post-date the 2023-2024 test window and their "
                           "training-data overlap with NEM is unaudited; not a clean blind test",
            "full_joint_label": "Full_joint/Anchor rows use mature historical errors and keep "
                                "their wider-information label; they are not equal-input "
                                "comparisons to any static or foundation row"},
        "files": files,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--main-table", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = build(args.run_root, args.main_table, args.output)
    print(json.dumps({"output": str(args.output), "files": sorted(provenance["files"])}, indent=2))


if __name__ == "__main__":
    main()
