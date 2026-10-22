"""Combine the parent, TimeXer/PriceFM extension, and expansion runs into one report.

Generalizes the audited two-run extension reporter to three frozen runs. Every
cell is revalidated by ``build_aligned_report`` (source/config snapshots,
artifact hashes, rescoring), artifacts are cross-checked per region with
``assert_comparable``, and each row keeps its own run provenance. No partial
leaderboard: unless all three matrices are complete, only progress and audit
state are published and stale CSVs are removed.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from .aligned_experiment import atomic_json, read_json, sha
from .aligned_report import build_aligned_report
from .aligned_scoring import assert_comparable, load_prediction_artifact

RUN_KEYS = ("run_id", "source_hash", "config_hash", "cache_hashes", "models")


def build_multi_run_report(run_roots: list[Path], *, replacements: dict[str, Path] | None = None) -> dict:
    """Audit every source, optionally replace a complete model grid explicitly.

    Old artifacts remain audited and referenced as superseded. Replacement
    selects a source, never the best score. Without an explicit mapping,
    duplicate models are an error as before.
    """
    if len(run_roots) < 2:
        raise ValueError("Need at least two runs to combine")
    roots = [Path(root).resolve() for root in run_roots]
    if len(set(roots)) != len(roots):
        raise ValueError('Duplicate run paths')
    replacements = {model: str(Path(root).resolve()) for model, root in (replacements or {}).items()}
    sources, reports, manifests, expected = {}, {}, {}, 0
    owners = {}
    for root in roots:
        manifest = read_json(root / "jobs.json")
        report = build_aligned_report(root)
        manifests[str(root)] = manifest
        sources[str(root)] = {key: manifest[key] for key in RUN_KEYS}
        sources[str(root)]["jobs_hash"] = sha(root / "jobs.json")
        sources[str(root)]["scope"] = manifest.get("scope")
        reports[str(root)] = report
        expected += report["progress"]["expected_cells"]
        for model in manifest['models']:
            owners.setdefault(model, []).append(str(root))
    for model, root in replacements.items():
        if model not in owners or len(owners[model]) < 2:
            raise ValueError(f'Replacement requires a duplicate model: {model}')
        if root not in owners[model]:
            raise ValueError(f'Replacement source does not provide {model}: {root}')
    chosen, superseded, excluded = {}, [], set()
    for model, providers in owners.items():
        if len(providers) > 1 and model not in replacements:
            raise ValueError(f'Duplicate model requires explicit replacement: {model}')
        chosen[model] = replacements.get(model, providers[0])
        if len(providers) == 1:
            continue
        grids = [{(j['region'], j['seed']) for j in manifests[root]['jobs'] if j['model'] == model}
                 for root in providers]
        if any(grid != grids[0] for grid in grids[1:]):
            raise ValueError(f'Replacement changes region/seed coverage: {model}')
        for root in providers:
            if root == chosen[model]:
                continue
            job_ids = [j['id'] for j in manifests[root]['jobs'] if j['model'] == model]
            excluded.update((root, job_id) for job_id in job_ids)
            superseded.append(dict(model=model, previous_source=sources[root],
                previous_run_root=root, replacement_run_root=chosen[model],
                replacement_source=sources[chosen[model]], excluded_job_ids=job_ids,
                reason='explicit_model_source_replacement_not_score_selection'))
    excluded_completed = sum((root, job['id']) in excluded and job['state'] == 'completed'
                             for root, item in reports.items() for job in item['jobs'])
    source_completed = sum(item['progress']['completed_cells'] for item in reports.values())
    complete = all(report["comparison_available"] for report in reports.values())
    report = {
        "run_id": "+".join(root.name for root in roots),
        "state": "complete" if complete else "incomplete",
        "stage": "fixed_configuration_comparison_not_tuned_sota",
        "comparison_available": complete,
        "sources": sources,
        "progress": {
            "expected_cells": expected - len(excluded),
            "completed_cells": source_completed - excluded_completed,
            "source_expected_cells": expected,
            "source_completed_cells": source_completed,
            "superseded_cells": len(excluded),
            "per_run": {name: r["progress"] for name, r in reports.items()},
        },
        "completed_artifact_validation": "passed",
        "model_source_selection": chosen,
        "superseded": superseded,
    }
    if not complete:
        return report
    references = {}
    for root in roots:
        manifest = manifests[str(root)]
        for job in manifest["jobs"]:
            artifact = load_prediction_artifact(root / job["output"] / "test.npz")
            if job["region"] in references:
                assert_comparable(references[job["region"]], artifact, allow_different_seed=True)
            else:
                references[job["region"]] = artifact
    report["cross_run_alignment_validation"] = \
        f"passed_all_{expected}_artifacts"
    report['active_comparison_cells'] = expected - len(excluded)
    report["tables"] = {"P": {}, "Q": {}}
    for name, run_report in reports.items():
        for table, models in run_report["tables"].items():
            for model, row in models.items():
                if chosen[model] != name:
                    continue
                if model in report["tables"][table]:
                    raise ValueError(f"Duplicate model across source runs: {model}")
                report["tables"][table][model] = dict(row, provenance=sources[name])
                previous = [entry for entry in superseded if entry['model'] == model]
                if previous:
                    report['tables'][table][model]['supersedes'] = previous
    return report


def write_multi_run_report(output: Path, report: dict) -> None:
    output = Path(output)
    atomic_json(output, report)
    directory = output.parent / (output.stem + "_tables")
    filenames = ("main.csv", "regional.csv", "per_seed.csv")
    if not report["comparison_available"]:
        for name in filenames:
            (directory / name).unlink(missing_ok=True)
        return

    def flatten(values, prefix=""):
        result = {}
        for key, value in values.items():
            name = prefix + key
            if isinstance(value, dict):
                result.update(flatten(value, name + "."))
            else:
                result[name] = value
        return result

    main, regional, per_seed = [], [], []
    for table, models in report["tables"].items():
        for model, row in models.items():
            identity = dict(table=table, model=model, **{
                key: row["provenance"][key] for key in ("run_id", "source_hash", "config_hash")})
            main.append(dict(identity, **flatten(row["macro_regions"])))
            for region, region_row in row["regions"].items():
                regional.append(dict(identity, region=region,
                                     **flatten(region_row["scores"]["seed_summary"])))
                for seed, scores in region_row["scores"]["per_seed"].items():
                    per_seed.append(dict(identity, region=region, seed=seed, **flatten(scores)))
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(filenames, (main, regional, per_seed)):
        path = directory / name
        temp = path.with_name(path.name + f".{os.getpid()}.tmp")
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with temp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="Two or more frozen run roots, each with a complete matrix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--replace-model', action='append', default=[], metavar='MODEL=RUN_ROOT',
                        help='Explicit replacement source for a duplicate model; old sources remain audited')
    args = parser.parse_args()
    replacements = {}
    for assignment in args.replace_model:
        if '=' not in assignment:
            parser.error('--replace-model requires MODEL=RUN_ROOT')
        model, root = assignment.split('=', 1)
        if not model or not root or model in replacements:
            parser.error('Each replacement requires one unique model and nonempty run root')
        replacements[model] = Path(root)
    write_multi_run_report(args.output, build_multi_run_report(args.runs, replacements=replacements))


if __name__ == "__main__":
    main()
