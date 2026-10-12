"""Independent event-probability controls and strict seed-wise event reports.

Controls follow event_metrics.py: training-fitted event persistence and
LogisticRegression(C=1, solver='lbfgs', max_iter=2000, no class weights).
The logistic inputs use its hour-level known drivers (including PD/IC in
future_exogenous), past-day event indicators/extremes, and known origin gas
context. No validation or test labels participate in fitting either control.
These are event classifiers; they do not manufacture price/quantile forecasts.

    python -m forecasting.exploration_events run --config CONFIG \
        --region NSW1 --seed 2026 --output OUTPUT
    python -m forecasting.exploration_events summarize --manifest MANIFEST \
        --output OUTPUT_DIRECTORY
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from .datasets import build_region_datasets, load_forecast_config
from .event_metrics import _ece, _hour_features, _labels
from .exploration_evaluate import LEVELS, REGIONS, SEEDS

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ("spike", "trough")
SPLITS = ("validation", "test")


def event_scores(probability: np.ndarray, outcome: np.ndarray) -> dict:
    p, y = np.asarray(probability), np.asarray(outcome)
    if p.shape != y.shape or p.ndim != 2 or not p.size:
        raise ValueError("Event probabilities/outcomes must have the same nonempty [N,H] shape")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Event probabilities must be finite and in [0,1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Event outcomes must be binary")
    p, y = p.ravel(), y.ravel().astype(int)
    positives = int(y.sum())
    return {
        "n_observations": int(y.size), "n_positive": positives, "n_negative": int(y.size - positives),
        "rate": float(y.mean()), "mean_probability": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < positives < y.size else None,
        "pr_auc": float(average_precision_score(y, p)) if positives else None,
        "ece": _ece(p, y),
    }


def _features(dataset, labels: dict) -> np.ndarray:
    features = _hour_features(dataset, labels)
    context = getattr(dataset, "origin_context_values", None)
    if context is not None:
        context = np.asarray(context)[np.asarray(dataset.origin_indices)].reshape(len(features), -1)
        context = np.broadcast_to(context[:, None, :], (*features.shape[:2], context.shape[-1]))
        features = np.concatenate([features, context], axis=-1)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite event classifier features")
    return features.reshape(-1, features.shape[-1])


def fit_persistence(labels: dict, event: str) -> dict:
    outcomes = labels[event]
    past = np.broadcast_to(labels[f"past_{event}"][:, None], outcomes.shape)
    overall = float(outcomes.mean())
    return {str(int(flag)): float(outcomes[past == flag].mean()) if (past == flag).any() else overall
            for flag in (False, True)}


def predict_persistence(labels: dict, event: str, rates: dict) -> np.ndarray:
    past = np.broadcast_to(labels[f"past_{event}"][:, None], labels[event].shape)
    return np.where(past, rates["1"], rates["0"])


def _atomic_json(path: Path, content: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(config_path, region: str, seed: int, output) -> dict:
    if region not in REGIONS or seed not in SEEDS:
        raise ValueError(f"Expected region in {REGIONS} and seed in {SEEDS}")
    config_path, output = Path(config_path), Path(output)
    config = load_forecast_config(config_path)
    if config.get("loss", {}).get("spike_threshold", 300.) != 300. or config.get("loss", {}).get("trough_threshold", 0.) != 0.:
        raise ValueError("Event controls require spike >300 and trough <0 AUD/MWh")
    start = time.monotonic()
    datasets = build_region_datasets(config_path, region)
    labels = {split: _labels(dataset, 300., 0.) for split, dataset in datasets.items()}
    x_train = _features(datasets["train"], labels["train"])
    models, persistence, metadata, parameters = {}, {}, {}, {}
    # Fit each event independently in each seed job. lbfgs is deterministic, so
    # independent fits may correctly have zero between-seed variance.
    with threadpool_limits(limits=2):
        for event in EVENTS:
            target = labels["train"][event].ravel().astype(int)
            persistence[event] = fit_persistence(labels["train"], event)
            unique = np.unique(target)
            if len(unique) == 1:
                models[event] = float(unique[0])
                details = {"constant_probability": models[event], "reason": "single training class"}
            else:
                classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier.fit(x_train, target)
                convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                models[event] = classifier
                details = {"iterations": classifier.n_iter_.tolist(), "convergence_warnings": convergence}
                parameters[f"{event}_coef"] = classifier.coef_
                parameters[f"{event}_intercept"] = classifier.intercept_
                parameters[f"{event}_classes"] = classifier.classes_
            metadata[event] = {"training_positive": int(target.sum()), "training_observations": int(len(target)),
                               "persistence_rates": persistence[event], **details}
            print(f"event={event} seed={seed} region={region} fitted {details}", flush=True)
    del x_train
    output.mkdir(parents=True, exist_ok=True)
    scores = {}
    for split in SPLITS:
        dataset = datasets[split]
        x = _features(dataset, labels[split])
        origins = np.asarray(dataset.origin_indices)
        actual = np.stack([dataset.target_values_raw[i:i + dataset.output_hours] for i in origins])
        arrays = {"actual": actual, "origin_unix": dataset.delivery_unix_seconds[origins],
                  "quantile_levels": LEVELS, "seed": seed, "region": region}
        scores[split] = {}
        with threadpool_limits(limits=2):
            for event in EVENTS:
                classifier = models[event]
                probability = (np.full(actual.shape, classifier) if isinstance(classifier, float)
                               else classifier.predict_proba(x)[:, 1].reshape(actual.shape))
                reference = predict_persistence(labels[split], event, persistence[event])
                arrays[f"logistic_prob_{event}"] = probability
                arrays[f"persistence_prob_{event}"] = reference
                scores[split][event] = {"logistic": event_scores(probability, labels[split][event]),
                                       "persistence": event_scores(reference, labels[split][event])}
        if not np.isfinite(actual).all():
            raise ValueError("Non-finite actual prices in event controls")
        path = output / f"{split}.npz"
        with path.with_suffix(".npz.tmp").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        path.with_suffix(".npz.tmp").replace(path)
        del x
    np.savez_compressed(output / "event_models.npz", **parameters)
    result = {"kind": "event_controls", "seed": seed, "region": region,
              "config": str(config_path), "configuration": config, "seconds": time.monotonic() - start,
              "thresholds": {"spike_gt": 300., "trough_lt": 0.},
              "feature_definition": "event_metrics._hour_features (known PD/IC, scarcity, netload, calendar, past-day event indicators/extremes) plus origin context",
              "logistic_parameters": {"C": 1., "solver": "lbfgs", "max_iter": 2000, "class_weight": None,
                                      "random_state": seed, "fit_split": "train"},
              "ece_definition": "10 equal-frequency probability bins; ties kept together",
              "seed_note": "independently fitted deterministic controls may have identical seed results",
              "models": metadata, "scores": scores}
    _atomic_json(output / "metrics.json", result)
    _atomic_json(output / "events.json", {key: result[key] for key in ("seed", "region", "thresholds", "scores")})
    return result


@dataclass
class EventForecast:
    actual: np.ndarray
    origins: np.ndarray
    probabilities: dict
    path: str


def load_event_forecast(path, region: str, seed: int, *, controls: bool = False) -> EventForecast:
    path = Path(path)
    prefixes = ("logistic", "persistence") if controls else ("model",)
    with np.load(path, allow_pickle=False) as data:
        required = {"actual", "origin_unix", "quantile_levels", "seed", "region"}
        required.update(f"{method}_prob_{event}" if controls else f"prob_{event}" for method in prefixes for event in EVENTS)
        if not required.issubset(data.files):
            raise ValueError(f"{path}: missing arrays {sorted(required.difference(data.files))}")
        actual, origins = np.asarray(data["actual"], dtype=float), np.asarray(data["origin_unix"])
        if actual.ndim != 2 or not all(actual.shape) or origins.shape != (len(actual),):
            raise ValueError(f"{path}: expected nonempty actual[N,H] and origin_unix[N]")
        if not np.isfinite(actual).all() or not np.isfinite(origins).all() or not np.equal(origins, np.floor(origins)).all():
            raise ValueError(f"{path}: non-finite actual or invalid origin seconds")
        if len(origins) > 1 and not np.all(np.diff(origins) == 3600):
            raise ValueError(f"{path}: origins must be a complete increasing hourly grid")
        levels = np.asarray(data["quantile_levels"])
        if levels.shape != LEVELS.shape or not np.allclose(levels, LEVELS, rtol=0, atol=1e-7):
            raise ValueError(f"{path}: wrong quantile level convention")
        if np.asarray(data["seed"]).shape != () or int(data["seed"]) != seed or str(data["region"]) != region:
            raise ValueError(f"{path}: wrong seed/region metadata")
        probabilities = {}
        for method in prefixes:
            probabilities[method] = {}
            for event in EVENTS:
                key = f"{method}_prob_{event}" if controls else f"prob_{event}"
                p = np.asarray(data[key], dtype=float)
                if p.shape != actual.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError(f"{path}: invalid {key} probabilities")
                probabilities[method][event] = p
    return EventForecast(actual, origins, probabilities, str(path))


def _aligned(reference: EventForecast, candidate: EventForecast, *, same_region=True):
    if reference.actual.shape != candidate.actual.shape or not np.array_equal(reference.origins, candidate.origins):
        raise ValueError(f"Event origins/shapes differ: {reference.path} vs {candidate.path}")
    if same_region and not np.array_equal(reference.actual, candidate.actual):
        raise ValueError(f"Event actual prices differ: {reference.path} vs {candidate.path}")


def _seed_summary(rows: list[dict]) -> dict:
    return {metric: {"mean": float(np.mean(values)) if values else None,
                     "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                     "n_seeds": len(values)}
            for metric in rows[0] for values in [[row[metric] for row in rows if row[metric] is not None]]}


def _selected(job):
    return (job.get("baseline_kind") == "event_controls"
            or job.get("variant", "").startswith(("event_joint", "event_frozen"))
            or job.get("kind") == "frozen_event")


def summarize(manifest_path, output=None, *, project_root=ROOT, protocols=None) -> dict:
    manifest_path, project_root = Path(manifest_path), Path(project_root)
    output = Path(output) if output is not None else manifest_path.parent / "reports"
    manifest = json.loads(manifest_path.read_text())
    if set(manifest.get("regions", REGIONS)) != set(REGIONS) or set(manifest.get("seeds", SEEDS)) != set(SEEDS):
        raise ValueError("Event summary requires NSW1/QLD1/TAS1 and seeds 2026/2027/2028")
    groups = {}
    for job in manifest["jobs"]:
        if not _selected(job) or (protocols is not None and job.get("protocol") not in protocols):
            continue
        key = (job["protocol"], job["variant"])
        cell = (job["region"], job["seed"])
        if cell in groups.setdefault(key, {}):
            raise ValueError(f"Duplicate event run: {key}/{cell}")
        groups[key][cell] = job
    if not groups:
        raise ValueError("Manifest has no event joint/frozen models or event_controls")
    expected = {(region, seed) for region in REGIONS for seed in SEEDS}
    for key, jobs in groups.items():
        if set(jobs) != expected:
            raise ValueError(f"Incomplete 3-region x 3-seed event group {key}: missing {expected.difference(jobs)}")
    result = {"schema_version": 1, "manifest": str(manifest_path), "regions": list(REGIONS), "seeds": list(SEEDS),
              "thresholds": {"spike_gt": 300., "trough_lt": 0.},
              "aggregation": "score each seed independently, then arithmetic mean and sample SD (ddof=1); no probability ensemble",
              "pr_auc_definition": "average precision", "ece_definition": "10 equal-frequency probability bins; ties kept together",
              "protocols": {}}
    alignment, region_alignment = {}, {}
    for (protocol, variant), jobs in sorted(groups.items()):
        protocol_result = result["protocols"].setdefault(protocol, {})
        for split in SPLITS:
            methods = protocol_result.setdefault(split, {})
            for (region, seed), job in jobs.items():
                directory = Path(job["output"])
                if not directory.is_absolute():
                    directory = project_root / directory
                controls = job.get("baseline_kind") == "event_controls"
                forecast = load_event_forecast(directory / f"{split}.npz", region, seed, controls=controls)
                align_key = (protocol, split, region)
                if align_key in alignment:
                    _aligned(alignment[align_key], forecast)
                else:
                    # Keep only truth/timestamps for alignment, not all probabilities.
                    alignment[align_key] = EventForecast(forecast.actual, forecast.origins, {}, forecast.path)
                joint_key = (protocol, split)
                if joint_key in region_alignment:
                    _aligned(region_alignment[joint_key], forecast, same_region=False)
                else:
                    region_alignment[joint_key] = alignment[align_key]
                for method, probabilities in forecast.probabilities.items():
                    name = f"{variant}/{method}" if controls else variant
                    entry = methods.setdefault(name, {"per_seed": {}, "region_seed_summary": {}})
                    rows = entry["per_seed"].setdefault(region, {})
                    rows[str(seed)] = {event: event_scores(p, forecast.actual > 300 if event == "spike" else forecast.actual < 0)
                                       for event, p in probabilities.items()}
            # Finalize just-completed variants, after all nine independent runs.
            for name, entry in methods.items():
                if name != variant and not (controls and name.startswith(variant + "/")):
                    continue
                for region, seeds in entry["per_seed"].items():
                    entry["region_seed_summary"][region] = {event: _seed_summary([row[event] for row in seeds.values()]) for event in EVENTS}
                macro_seed = {}
                for seed in SEEDS:
                    macro_seed[str(seed)] = {}
                    for event in EVENTS:
                        rows = [entry["per_seed"][region][str(seed)][event] for region in REGIONS]
                        macro_seed[str(seed)][event] = {
                            metric: float(np.mean([row[metric] for row in rows])) if all(row[metric] is not None for row in rows) else None
                            for metric in rows[0]
                        }
                entry["macro_region_per_seed"] = macro_seed
                entry["macro_region_seed_summary"] = {event: _seed_summary([row[event] for row in macro_seed.values()]) for event in EVENTS}
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "events_report.json", result)
    lines = ["# Event probability report", "", "Each cell is the mean ± sample SD across three separately scored seeds. "
             "PR-AUC is average precision; ECE uses 10 equal-frequency probability bins. Empty positive-event support gives null PR-AUC; "
             "ROC-AUC requires both classes. Thresholds: spike > 300 and trough < 0 AUD/MWh.", "",
             "| Protocol | Split | Model | Region | Event | Seeds | Brier ↓ | PR-AUC ↑ | ROC-AUC ↑ | ECE ↓ |", 
             "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    def cell(value):
        if value["mean"] is None:
            return "null"
        return f"{value['mean']:.5f} ± {value['std']:.5f}" if value["std"] is not None else f"{value['mean']:.5f}"
    for protocol, splits in result["protocols"].items():
        for split, methods in splits.items():
            for name, entry in methods.items():
                rows = {**entry["region_seed_summary"], "macro_regions": entry["macro_region_seed_summary"]}
                for region, events in rows.items():
                    for event, values in events.items():
                        scores = " | ".join(cell(values[k]) for k in ("brier", "pr_auc", "roc_auc", "ece"))
                        lines.append(f"| {protocol} | {split} | {name} | {region} | {event} | 3 | {scores} |")
    (output / "events_report.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("run")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--region", required=True, choices=REGIONS)
    train.add_argument("--seed", required=True, choices=SEEDS, type=int)
    train.add_argument("--output", required=True, type=Path)
    report = commands.add_parser("summarize")
    report.add_argument("--manifest", required=True, type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--protocol", action="append", help="Repeat to select protocol groups")
    args = parser.parse_args()
    if args.command == "run":
        run(args.config, args.region, args.seed, args.output)
    else:
        output = args.output or args.manifest.parent / "reports"
        result = summarize(args.manifest, output, protocols=args.protocol)
        print(f"Wrote {output}/events_report.json and events_report.md: {len(result['protocols'])} protocols")


if __name__ == "__main__":
    main()
