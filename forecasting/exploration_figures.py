"""Deterministically selected forecast cases and descriptive routing diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .exploration_evaluate import REGIONS, SEEDS


def make_figures(patterns: dict[str, str], output: Path):
    """Cases use seed 2026 and the first issued forecast crossing fixed thresholds.

    Actual test outcomes never choose the case. These are illustrative forecasts,
    not additional independent tests or a selected best seed.
    """
    if "full" not in patterns:
        return
    output.mkdir(parents=True, exist_ok=True)
    selection, routing = {}, {}
    for region in REGIONS:
        routing[region] = {}
        for seed in SEEDS:
            with np.load(patterns["full"].format(region=region, seed=seed)) as a:
                if "dgf_fusion_weight" not in a:
                    continue
                y = a["actual"]
                masks = {"all": np.ones(y.shape, dtype=bool), "negative": y < 0,
                         "ordinary": (y >= 0) & (y <= 300), "spike": y > 300}
                routing[region][str(seed)] = {
                    key: {band: {"count": int(mask.sum()),
                                 "mean": a[key][mask].mean(axis=0).tolist() if mask.any() else None}
                          for band, mask in masks.items()}
                    for key in ("dgf_fusion_weight", "dgf_quantile_fusion_weight") if key in a
                }
        with np.load(patterns["full"].format(region=region, seed=2026)) as a:
            point, q, y, origins = [a[k].copy() for k in ("point", "quantile", "actual", "origin_unix")]
        choices = {"first_origin": 0}
        for name, mask in (("first_spike_forecast", (q[..., -1] > 300).any(axis=1)),
                           ("first_negative_forecast", (q[..., 0] < 0).any(axis=1))):
            indices = np.flatnonzero(mask)
            if len(indices):
                choices[name] = int(indices[0])
        selection[region] = {}
        for name, i in choices.items():
            fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
            h = np.arange(1, y.shape[1] + 1)
            ax.fill_between(h, q[i, :, 0], q[i, :, -1], alpha=.2, label="Full 90% interval")
            ax.plot(h, y[i], color="black", label="Observed")
            ax.plot(h, point[i], label="Full point")
            if "xgboost" in patterns:
                with np.load(patterns["xgboost"].format(region=region, seed=2026)) as b:
                    if not np.array_equal(b["origin_unix"], origins) or not np.array_equal(b["actual"], y):
                        raise ValueError("Case-study comparator is not aligned")
                    ax.plot(h, b["point"][i], linestyle="--", label="XGBoost median")
            stamp = pd.Timestamp(origins[i], unit="s", tz="UTC").tz_convert("Australia/Brisbane")
            ax.set(title=f"{region} | {stamp} | seed 2026", xlabel="Forecast horizon (hours)", ylabel="AUD/MWh")
            ax.legend(loc="best", fontsize=8)
            path = output / f"{region}_{name}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            selection[region][name] = {"origin_unix": int(origins[i]), "seed": 2026, "image": str(path)}
    (output / "case_selection.json").write_text(json.dumps({"rule": "First issued full-model forecast in chronological order with q95>300 or q05<0 at any horizon; plus first origin. Outcomes do not select cases.", "cases": selection}, indent=2))
    (output / "routing.json").write_text(json.dumps({"interpretation": "Descriptive gate activity by realized band; not evidence of causal event identification.", "per_seed": routing}, indent=2, allow_nan=False))
