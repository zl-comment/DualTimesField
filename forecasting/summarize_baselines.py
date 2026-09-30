"""Tabulate the re-implemented RE-Price baselines against the dual-field trunk.

Reads the per-run summaries written by ``forecasting.baselines`` and the
RE-Price style metrics of the trunk for seeds 2026-2028, and prints markdown
tables of the three-seed mean (and standard deviation of MAE) per region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
SEEDS = (2026, 2027, 2028)
METRICS = (
    ("mae", "MAE", 1.0),
    ("rmse_window_mean", "Window RMSE", 1.0),
    ("sde_window_mean", "Window SDE", 1.0),
    ("picp_90", "90% PICP", 100.0),
    ("piaw_90", "90% PIAW", 1.0),
    ("ais_90", "90% AIS", 1.0),
    ("crps_quantile_approx", "CRPS~", 1.0),
)
BASELINES = (("xgboost", "XGBoost"), ("gru", "GRU"), ("deepar", "DeepAR"))
# RE-Price Table 2 (MAE, RMSE, SDE) and Table 3 (CRPS); reported, not reproduced.
RE_PRICE = {
    "RE-Price": {"NSW1": (23.48, 34.15, 34.14, 17.36), "QLD1": (25.85, 37.15, 36.89, 20.58), "TAS1": (18.96, 22.81, 21.57, 13.13)},
    "XGBoost": {"NSW1": (38.63, 51.99, 50.61, 28.79), "QLD1": (44.70, 64.05, 61.36, 29.17), "TAS1": (37.56, 40.33, 38.96, 23.43)},
    "GRU": {"NSW1": (33.24, 48.98, 45.58, 25.12), "QLD1": (43.24, 52.70, 43.06, 28.48), "TAS1": (30.40, 41.80, 33.94, 19.66)},
    "DeepAR": {"NSW1": (33.77, 50.20, 47.87, 26.25), "QLD1": (36.67, 42.32, 36.73, 25.94), "TAS1": (26.51, 41.90, 35.87, 20.18)},
}


def _trunk_paths(protocol: str, region: str) -> list[Path]:
    root = Path("logs/forecasting")
    if protocol == "raw":
        return [
            root / "paper_metrics/25_pdpasa_netload_softclip_ctf" / f"{region}.json",
            *(root / f"seed_variance/paper_metrics/pdpasa_netload_softclip_ctf_seed{s}" / f"{region}.json" for s in SEEDS[1:]),
        ]
    return [root / f"protocol_alignment/capped650_trunk/paper_metrics/seed{s}" / f"{region}.json" for s in SEEDS]


def load(protocol: str, baseline_root: Path) -> dict:
    """Returns {model: {region: [metrics per seed]}}."""
    table = {"Dual-field trunk": {}}
    for region in REGIONS:
        table["Dual-field trunk"][region] = [
            json.loads(path.read_text())["test"]["model"] for path in _trunk_paths(protocol, region)
        ]
    for key, name in BASELINES:
        table[name] = {
            region: [
                json.loads((baseline_root / protocol / key / f"seed{s}" / f"{region}.json").read_text())["test"]
                for s in SEEDS
            ]
            for region in REGIONS
        }
    return table


def markdown(protocol: str, table: dict) -> str:
    lines = []
    for region in (*REGIONS, "Mean"):
        lines += [f"#### {region}", "", "| Model | " + " | ".join(label for _, label, _ in METRICS) + " |",
                  "|---|" + "---:|" * len(METRICS)]
        for model, per_region in table.items():
            cells = []
            for key, _, scale in METRICS:
                if region == "Mean":
                    seeds = np.mean([[m[key] for m in per_region[r]] for r in REGIONS], axis=0)
                else:
                    seeds = np.asarray([m[key] for m in per_region[region]])
                seeds = seeds * scale
                cell = f"{seeds.mean():.2f}"
                if key == "mae":
                    cell += f" ± {seeds.std(ddof=1):.2f}"
                cells.append(cell)
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        if region != "Mean":
            for model, values in RE_PRICE.items():
                mae, rmse, sde, crps = values[region]
                lines.append(f"| {model} (RE-Price paper, with news) | {mae:.2f} | {rmse:.2f} | {sde:.2f} | - | - | - | {crps:.2f} |")
        lines.append("")
    return f"### Protocol: {protocol}\n\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=Path("logs/forecasting/baselines"))
    args = parser.parse_args()
    for protocol in ("raw", "capped650"):
        print(markdown(protocol, load(protocol, args.baseline_root)))


if __name__ == "__main__":
    main()
