"""Build the paper's figures and tables from the recorded results.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python paper/make_materials.py

The main line is the quarterly-refit setting with AEMO predispatch inputs; the
static-split development ladder, the structural ablations of the detected-event
DGF, and the field-separation figure are supplementary.

Figures go to ``paper/figures`` (PDF and PNG) and tables to ``paper/tables``
(LaTeX, booktabs). Every number is read from ``logs/forecasting`` or recomputed
from the saved checkpoints; nothing is typed in by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "forecasting"
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"
REGIONS = ("NSW1", "QLD1", "TAS1")
SEEDS = ("seed2026", "seed2027", "seed2028")

# Validated categorical slots (dataviz reference palette, light surface).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, CONTEXT = "#0b0b0b", "#52514e", "#e1e0d9", "#b4b2a9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": CONTEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def _metrics(directory: Path, region: str) -> dict:
    test = json.loads((directory / f"{region}.json").read_text())["test"]
    return test.get("model", test)


def _three_seed(paths: dict[str, Path], key: str) -> tuple[dict, float, float]:
    per_region = {r: float(np.mean([_metrics(paths[s], r)[key] for s in SEEDS])) for r in REGIONS}
    per_seed = [np.mean([_metrics(paths[s], r)[key] for r in REGIONS]) for s in SEEDS]
    return per_region, float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))


def _seed_dirs(seed2026: str, other: str) -> dict[str, Path]:
    return {
        "seed2026": LOGS / seed2026,
        "seed2027": LOGS / other.format(seed=2027),
        "seed2028": LOGS / other.format(seed=2028),
    }


def _nested(root: str) -> dict[str, Path]:
    return {s: LOGS / root / s for s in SEEDS}


def _baseline_json(protocol: str, kind: str) -> dict[str, Path]:
    return {s: None for s in SEEDS} | {"kind": kind, "protocol": protocol}


def _baseline(protocol: str, kind: str, key: str) -> tuple[dict, float, float]:
    def value(seed: str, region: str) -> float:
        return json.loads((LOGS / "baselines" / protocol / kind / seed / f"{region}.json").read_text())["test"][key]
    per_region = {r: float(np.mean([value(s, r) for s in SEEDS])) for r in REGIONS}
    per_seed = [np.mean([value(s, r) for r in REGIONS]) for s in SEEDS]
    return per_region, float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))


MODELS = {
    "gabor_trunk": _seed_dirs("paper_metrics/28_scarcity_quantile_inputs",
                              "seed_variance/paper_metrics/scarcity_quantile_inputs_seed{seed}"),
    "raw_history": _seed_dirs("paper_metrics/29_ablation_raw_history",
                              "seed_variance/paper_metrics/ablation_raw_history_seed{seed}"),
    "detected": _nested("detected_dgf/paper_metrics/detected_dgf"),
    "detected_capped": _nested("step5/paper_metrics/capped650_detected_dgf"),
    "adaptive": _nested("adaptive_dgf/paper_metrics/adaptive_dgf_echo_raw"),
    "detected_echo": _nested("detected_dgf/paper_metrics/detected_dgf_echo_raw"),
}
for variant in ("k4", "k16", "sep1", "sep6", "nothreshold", "mean"):
    MODELS[f"detected_{variant}"] = _nested(f"step5/paper_metrics/detected_dgf_{variant}")

# Main line: every model refit each test quarter (forecasting.rolling).
# "{pre}" is "" for raw prices and "capped650_" for prices capped at 650.
PROTOCOLS = (("raw", ""), ("capped650", "capped650_"))
BASELINES = (("XGBoost", "xgboost"), ("PatchTST", "patchtst"), ("DLinear", "dlinear"), ("Linear", "linear_mse"))
SETTINGS = {  # setting -> model -> run root (seed directories below it)
    "Static split": {
        "Ours": "linear_base/paper_metrics/{pre}linear_base",
        "Linear base only": "linear_base/paper_metrics/{pre}linear_base_only",
        **{name: "baselines/{p}/" + kind for name, kind in BASELINES},
    },
    "Quarterly refit": {
        "Ours": "rolling/paper_metrics/{pre}linear_base",
        "Linear base only": "rolling/paper_metrics/{pre}linear_base_only",
        **{name: "baselines_rolling/{p}/" + kind for name, kind in BASELINES},
    },
    "Quarterly refit + predispatch": {
        "Ours": "price_floor/paper_metrics/{pre}pd_calibrator_floor",
        "Linear base only": "predispatch_rolling/paper_metrics/{pre}pd_linear_base_only",
        **{name: "baselines_rolling_pd/{p}/" + kind for name, kind in BASELINES},
    },
}
FINAL = "price_floor/paper_metrics/{pre}pd_calibrator_floor"


def _run(root: str, protocol: str, prefix: str) -> dict[str, Path]:
    return _nested(root.format(p=protocol, pre=prefix))


# Development ladder on the static split (supplementary): three-region mean
# test MAE and CRPS~ (three seeds from step 4 on).
LADDER = [
    ("Static linear baseline", "paper_metrics/01_static_normalization_baseline", None),
    ("asinh price target", "paper_metrics/16_asinh_additive_trigonometric_fusion", None),
    ("Price-space quantiles", "paper_metrics/18_asinh_price_space_quantiles", None),
    ("Residual path", "paper_metrics/19_residual_skip_path", "seed_variance/paper_metrics/residual_skip_path_seed{seed}"),
    ("Gas price -> CTF", "paper_metrics/20_gas_price_ctf", "seed_variance/paper_metrics/gas_price_ctf_seed{seed}"),
    ("Spare capacity -> DGF", "paper_metrics/21_pdpasa_dgf", "seed_variance/paper_metrics/pdpasa_dgf_seed{seed}"),
    ("Net load -> CTF", "paper_metrics/22_pdpasa_netload_ctf", "seed_variance/paper_metrics/pdpasa_netload_ctf_seed{seed}"),
    ("Soft saturation", "paper_metrics/25_pdpasa_netload_softclip_ctf", "seed_variance/paper_metrics/pdpasa_netload_softclip_ctf_seed{seed}"),
    ("Quantile gate + shortfalls", "paper_metrics/28_scarcity_quantile_inputs", "seed_variance/paper_metrics/scarcity_quantile_inputs_seed{seed}"),
    ("Detected-event DGF", None, None),
]


def ladder_values() -> list[tuple[str, float, float, bool]]:
    rows = []
    for label, first, other in LADDER:
        if first is None:
            paths, three = MODELS["detected"], True
        elif other is None:
            paths, three = {"seed2026": LOGS / first}, False
        else:
            paths, three = _seed_dirs(first, other), True
        seeds = SEEDS if three else ("seed2026",)
        mae = np.mean([np.mean([_metrics(paths[s], r)["mae"] for r in REGIONS]) for s in seeds])
        crps = np.mean([np.mean([_metrics(paths[s], r)["crps_quantile_approx"] for r in REGIONS]) for s in seeds])
        rows.append((label, float(mae), float(crps), three))
    return rows


def figure_ladder() -> None:
    rows = ladder_values()
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, index, title in ((axes[0], 1, "Test MAE (AUD/MWh)"), (axes[1], 2, "CRPS~ (AUD/MWh)")):
        values = np.array([r[index] for r in rows])
        ax.plot(values, y, color=CONTEXT, lw=1.0, zorder=1)
        for i, (row, value) in enumerate(zip(rows, values)):
            final = i == len(rows) - 1
            color = BLUE if final else ("#8fb8ea" if row[3] else "#d3d1c7")
            ax.scatter(value, y[i], s=42 if final else 30, color=color, edgecolor="white", linewidth=1.2, zorder=3)
            ax.text(value + (values.max() - values.min()) * 0.04, y[i], f"{value:.2f}", va="center", fontsize=6.5, color=INK)
        span = values.max() - values.min()
        ax.set_xlim(values.min() - span * 0.08, values.max() + span * 0.22)
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", length=0)
    fig.text(0.01, 0.01, "Points: grey = seed 2026 only; light blue = mean of seeds 2026-2028; blue = final model (three seeds). "
             "Axes do not start at zero.", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "ablation_ladder")


def figure_settings() -> None:
    """Three-region mean MAE of every model in the three settings."""
    colors = {"Ours": BLUE, "XGBoost": ORANGE, "PatchTST": AQUA, "Linear base only": "#8fb8ea",
              "DLinear": "#8a8983", "Linear": "#b4b2a9"}
    settings = list(SETTINGS)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (protocol, prefix), title in zip(axes, PROTOCOLS, ("Raw prices", "Prices capped at 650 AUD/MWh")):
        x = np.arange(len(settings))
        ends = []
        for name, color in colors.items():
            values = [_three_seed(_run(SETTINGS[s][name], protocol, prefix), "mae")[1] for s in settings]
            final = name == "Ours"
            ax.plot(x, values, color=color, lw=2.0 if final else 1.2, zorder=3 if final else 2)
            ax.scatter(x, values, s=30 if final else 18, color=color, edgecolor="white", linewidth=1.0, zorder=4)
            ends.append((values[-1], name, color))
        # Direct labels at the right end, spread so they do not overlap.
        ends.sort()
        span = max(v for v, _, _ in ends) - min(v for v, _, _ in ends)
        placed = []
        for value, name, color in ends:
            y = value if not placed else max(value, placed[-1] + span * 0.11 + 0.1)
            placed.append(y)
            ax.text(x[-1] + 0.08, y, f"{name} {value:.2f}", va="center", fontsize=6.5, color=INK)
        ax.set_xticks(x, ["Static\nsplit", "Quarterly\nrefit", "Quarterly refit\n+ predispatch"])
        ax.set_xlim(-0.2, len(settings) - 1 + 0.95)
        ax.set_title(f"{title}: test MAE (AUD/MWh)", loc="left", color=INK)
        ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.0, "Three-region means over seeds 2026-2028. Ours in the last setting includes the calibrator and "
             "the validation-chosen price floor. Axes do not start at zero.", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "settings")


def figure_comparison() -> None:
    models = [("Ours", FINAL)] + [(name, "baselines_rolling_pd/{p}/" + kind) for name, kind in BASELINES]
    shades = [BLUE, ORANGE, "#8a8983", "#b4b2a9", "#d3d1c7"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, (protocol, prefix), title in zip(axes, PROTOCOLS, ("Raw prices", "Prices capped at 650 AUD/MWh")):
        width = 0.16
        x = np.arange(len(REGIONS))
        for i, ((name, root), color) in enumerate(zip(models, shades)):
            per_region = _three_seed(_run(root, protocol, prefix), "mae")[0]
            ax.bar(x + (i - 2) * width, [per_region[r] for r in REGIONS], width=width - 0.02, color=color, label=name)
        ax.set_xticks(x, [r[:-1] for r in REGIONS])
        ax.set_title(f"{title}: test MAE (AUD/MWh)", loc="left", color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=5, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "same_data_comparison")


def figure_architecture() -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(-6, 72)
    ax.axis("off")
    styles = {
        "input": ("#f1efe8", "#888780"),
        "ctf": ("#e6f1fb", BLUE),
        "dgf": ("#fdeee7", ORANGE),
        "base": ("#e8f6f0", AQUA),
        "neutral": ("#ffffff", "#888780"),
    }

    def box(x, y, w, h, title, subtitle, kind):
        face, edge = styles[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, linewidth=0.9))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=7.5, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center", fontsize=6.2, color=MUTED)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#888780", lw=0.8, shrinkA=0, shrinkB=0,
                                    mutation_scale=8))

    # Inputs
    box(2, 63, 28, 7, "72-h history", "price (asinh), demand", "input")
    box(36, 63, 28, 7, "AEMO predispatch", "price, demand, generation, flows", "input")
    box(70, 63, 28, 7, "PD PASA, gas", "spare capacity, net load, gas", "input")
    # Dual field (left half)
    box(2, 49, 22, 7.5, "Event detection", "median departures", "dgf")
    box(2, 36.5, 22, 7.5, "DGF: event field", "sparse bumps", "dgf")
    box(26, 36.5, 22, 7.5, "CTF: trend field", "remainder", "ctf")
    ax.text(25, 32.6, "loss: history = CTF + DGF", ha="center", fontsize=6.0, color=MUTED)
    # Known inputs (right half)
    box(52, 42.5, 46, 7.5, "Known inputs and history", "predispatch, spare capacity, net load, gas, calendar", "input")
    # Heads
    box(2, 22, 22, 7, "DGF heads", "events + scarcity", "dgf")
    box(26, 22, 22, 7, "CTF heads", "level + net load", "ctf")
    box(52, 22, 22, 7, "Linear base", "linear map", "base")
    box(76, 22, 22, 7, "Calibrator", "per-hour MLP", "neutral")
    box(10, 9, 80, 7, "Fusion", "base + CTF + sin\u00b2\u03b8\u00b7DGF + calibrator, per-quantile gates", "neutral")
    box(30, -1.8, 40, 7, "Price floor", "level chosen on the validation quarter", "neutral")

    arrow((13, 63), (13, 56.8))                 # history -> detection
    arrow((24, 52.5), (34, 44.3))               # detection -> CTF (remainder)
    arrow((13, 49), (13, 44.3))                 # detection -> DGF
    arrow((28, 63), (60, 50.3))                 # history -> known inputs
    arrow((50, 63), (70, 50.3))                 # predispatch -> known inputs
    arrow((84, 63), (84, 50.3))                 # PD PASA, gas -> known inputs
    arrow((7, 36.5), (7, 29.3))                 # DGF -> heads
    arrow((43, 36.5), (43, 29.3))               # CTF -> heads
    arrow((63, 42.5), (63, 29.3))               # known inputs -> base
    arrow((87, 42.5), (87, 29.3))               # known inputs -> calibrator
    for x in (13, 37, 63, 87):                  # heads -> fusion
        arrow((x, 22), (min(max(x, 14), 86), 16.3))
    arrow((50, 9), (50, 5.5))
    ax.text(50, -5.4, "Refit every test quarter. Output: 24-hour point forecast and 0.05/0.10/0.50/0.90/0.95 quantiles",
            ha="center", fontsize=6.2, color=MUTED)
    _save(fig, "architecture")


def _load_model(config: str, checkpoint: Path, device):
    from forecasting.datasets import load_forecast_config
    from forecasting.train import build_model

    model = build_model(load_forecast_config(ROOT / config)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.set_epoch(state.get("model_epoch", state["best_epoch"]))
    model.eval()
    return model


def mechanism_data() -> dict:
    from forecasting.datasets import build_region_datasets
    from forecasting.train import DeviceBatches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {
        "gabor": ("configs/aemo_forecast_scarcity_quantile_inputs.yaml", "outputs/forecasting/scarcity_quantile_inputs"),
        "detected": ("configs/aemo_forecast_detected_dgf.yaml", "outputs/forecasting/detected_dgf"),
    }
    data = {}
    for region in REGIONS:
        data[region] = {}
        for name, (config, root) in runs.items():
            dataset = build_region_datasets(ROOT / config, region)["test"]
            model = _load_model(config, ROOT / root / region / "best_model.pt", device)
            price, ctf, event = [], [], []
            with torch.no_grad():
                for batch in DeviceBatches(dataset, 4096, False, device):
                    out = model(batch["history_values"], batch["future_calendar"], batch.get("future_exogenous"),
                                batch.get("origin_context"), batch.get("ctf_exogenous"), batch.get("quantile_exogenous"))
                    price.append(batch["history_values"][..., 0].cpu())
                    ctf.append(out["ctf_signal"][..., 0].cpu())
                    event.append(out["event_signal"][..., 0].cpu())
            origins = np.asarray(dataset.origin_indices)
            raw = np.asarray(dataset.target_values_raw, dtype=np.float64)
            data[region][name] = {
                "price": torch.cat(price).numpy(), "ctf": torch.cat(ctf).numpy(), "event": torch.cat(event).numpy(),
                "spike": np.stack([raw[i - 72:i] for i in origins]) > 300,
            }
    return data


def figure_mechanism(data: dict) -> dict:
    nsw = data["NSW1"]
    price = nsw["gabor"]["price"]
    # A window with its largest price 12-36 hours before the origin, chosen as the highest such peak.
    peak_position = price.argmax(axis=1)
    candidates = np.flatnonzero((peak_position >= 36) & (peak_position <= 60))
    window = int(candidates[np.argmax(price[candidates].max(axis=1))])
    hours = np.arange(-72, 0)
    fig = plt.figure(figsize=(7.0, 4.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0), hspace=0.55, wspace=0.28)
    top = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for ax, name, title in ((top[0], "gabor", "Transplanted dual field (Gabor DGF)"),
                            (top[1], "detected", "Detected-event DGF (ours)")):
        run = nsw[name]
        ax.plot(hours, run["price"][window], color=CONTEXT, lw=2.0, label="History")
        ax.plot(hours, run["ctf"][window], color=BLUE, lw=1.6, label="CTF (trend)")
        ax.plot(hours, run["event"][window], color=ORANGE, lw=1.6, label="DGF (events)")
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Hours before forecast origin")
        ax.set_xlim(-72, 0)
    top[0].set_ylabel("Standardized asinh price")
    low = min(a.get_ylim()[0] for a in top)
    high = max(a.get_ylim()[1] for a in top)
    for ax in top:
        ax.set_ylim(low, high)
    top[1].legend(frameon=False, loc="upper left", ncols=1)

    bottom = fig.add_subplot(grid[1, :])
    x = np.arange(len(REGIONS))
    summary = {}
    for offset, name, color, label in ((-0.19, "gabor", CONTEXT, "Transplanted (Gabor DGF)"),
                                       (0.19, "detected", BLUE, "Detected-event DGF")):
        shares = []
        for region in REGIONS:
            run = data[region][name]
            mask = run["spike"]
            shares.append(float(run["event"][mask].mean() / run["price"][mask].mean()))
        summary[name] = dict(zip(REGIONS, shares))
        bars = bottom.bar(x + offset, shares, width=0.36, color=color, label=label)
        for bar, v in zip(bars, shares):
            bottom.text(bar.get_x() + bar.get_width() / 2, v + (0.01 if v >= 0 else -0.04), f"{v:.2f}",
                        ha="center", fontsize=6.5, color=INK)
    bottom.axhline(0, color=CONTEXT, lw=0.8)
    bottom.set_xticks(x, [r[:-1] for r in REGIONS])
    bottom.set_title("Share of past spike hours (price > 300 AUD/MWh) carried by the event field, test split",
                     loc="left", color=INK)
    bottom.grid(axis="x", visible=False)
    bottom.set_ylim(-0.1, 0.5)
    bottom.legend(frameon=False, loc="upper left", ncols=2)
    _save(fig, "field_separation")
    return summary


def table_cost() -> dict:
    """Parameters and wall-clock time of one quarterly refit (Q3 2023, NSW1, seed 2026)."""
    from forecasting.datasets import load_forecast_config
    from forecasting.train import build_model

    seconds = {}
    for line in (LOGS / "cost" / "seconds.tsv").read_text().splitlines():
        name, value, status = line.split("\t")
        if status == "0":
            seconds[name] = float(value)
    rows = []
    for label, key in (("Ours", "pd_calibrator"), ("Linear base only", "pd_linear_base_only")):
        model = build_model(load_forecast_config(ROOT / f"configs/aemo_forecast_rolling_{key}_q3.yaml"))
        rows.append((label, f"{sum(p.numel() for p in model.parameters()):,}", seconds.get(key)))
    for label, key in (("XGBoost", "xgboost"), ("PatchTST", "patchtst"), ("DLinear", "dlinear"), ("Linear", "linear_mse")):
        summary = json.loads((ROOT / f"outputs/forecasting/cost/{key}/NSW1.json").read_text())
        size = (f"{sum(summary['rounds_per_horizon']):,} rounds" if key == "xgboost"
                else f"{summary['parameters']:,}")
        rows.append((label, size, seconds.get(key)))
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrr}", "\\toprule",
             "Model & Parameters & Seconds per refit \\\\", "\\midrule"]
    lines += [f"{label} & {size} & {'--' if t is None else f'{t:.0f}'} \\\\" for label, size, t in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("cost", "\n".join(lines) + "\n")
    return {label: {"size": size, "seconds": t} for label, size, t in rows}


def _case_origins() -> dict[str, int]:
    """Case days, chosen by fixed rules on the test split (origins at 00:00 AEST)."""
    S = ROOT / "outputs/forecasting/significance_inputs"
    cases = {}
    for region, rule in (("QLD1", "deepest"), ("NSW1", "phantom")):
        with np.load(S / f"rolling_pd_calibrator_floor/{region}.npz") as a:
            origins, actual = a["origin_unix"], a["actual"]
        with np.load(ROOT / f"data/aemo_exogenous/{region.lower()}_predispatch.npz") as a:
            pd_rrp = a["predispatch_rrp"][np.searchsorted(a["forecast_origin_unix"], origins)]
        midnight = (origins + 10 * 3600) % 86400 == 0
        if rule == "deepest":  # lowest mean actual price over the day
            score = np.where(midnight, -actual.mean(axis=1), -np.inf)
        else:  # predispatch overstated the peak: > 1000 forecast, < 300 actual
            ok = midnight & (pd_rrp.max(axis=1) > 1000) & (actual.max(axis=1) < 300)
            score = np.where(ok, pd_rrp.max(axis=1) - actual.max(axis=1), -np.inf)
        cases[region] = int(origins[int(np.argmax(score))])
    return cases


def _stages(region: str, origin: int) -> dict:
    """Forecast stages of the seed-2026 refit covering ``origin``: base, + fields, + calibrator, + floor."""
    import pandas as pd
    from forecasting.datasets import build_region_datasets, load_forecast_config
    from forecasting.rolling import QUARTERS, quarter_config_path
    from forecasting.train import build_model, move_inputs

    day = pd.to_datetime(origin, unit="s") + pd.Timedelta(hours=10)
    k = int(np.searchsorted(QUARTERS, day, side="right") - 1)
    config_path = ROOT / quarter_config_path("pd_calibrator", k)
    dataset = build_region_datasets(config_path, region)["test"]
    index = int(np.flatnonzero(dataset.delivery_unix_seconds[np.asarray(dataset.origin_indices)] == origin)[0])
    batch = {key: (value[None] if torch.is_tensor(value) else value) for key, value in dataset[index].items()}
    device = torch.device("cpu")
    model = _load_model(str(config_path.relative_to(ROOT)), ROOT / f"outputs/forecasting/rolling/pd_calibrator/q{k}/{region}/best_model.pt", device)
    *inputs, _ = move_inputs(batch, device)
    with torch.no_grad():
        out = model(*inputs)
    price = lambda z: dataset.denormalize_target(z)[0, :, 0].numpy()
    choice = json.loads((LOGS / f"price_floor/paper_metrics/pd_calibrator_floor/choices_{region}.json").read_text())[k]
    floor = choice["point_floor"]
    quantile = np.sort(dataset.denormalize_quantiles(out["quantile_forecast"])[0].numpy(), axis=-1)
    if choice["quantile_floor"] is not None:
        quantile = np.sort(np.maximum(quantile, choice["quantile_floor"]), axis=-1)
    calibrated = price(out["point_forecast"])
    return {
        "base": price(out["base_point"]),
        "fields": price(out["base_point"] + out["field_point"]),
        "calibrator": calibrated,
        "final": calibrated if floor is None else np.maximum(calibrated, floor),
        "q05": quantile[:, 0], "q95": quantile[:, -1], "floor": floor,
    }


def figure_cases() -> dict:
    import pandas as pd

    S = ROOT / "outputs/forecasting/significance_inputs"
    cases = _case_origins()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    titles = {"QLD1": "QLD, deepest negative-price day", "NSW1": "NSW, predispatch overstates the peak"}
    summary = {}
    for row, (region, origin) in enumerate(cases.items()):
        with np.load(S / f"rolling_pd_calibrator_floor/{region}.npz") as a:
            i = int(np.flatnonzero(a["origin_unix"] == origin)[0])
            actual = a["actual"][i]
        with np.load(ROOT / f"outputs/forecasting/baselines_rolling_pd/raw/xgboost/seed2026/{region}.npz") as a:
            xgb = a["point"][int(np.flatnonzero(a["origin_unix"] == origin)[0])]
        with np.load(ROOT / f"data/aemo_exogenous/{region.lower()}_predispatch.npz") as a:
            pdp = a["predispatch_rrp"][int(np.searchsorted(a["forecast_origin_unix"], origin))]
        stages = _stages(region, origin)
        hours = np.arange(24)
        day = (pd.to_datetime(origin, unit="s") + pd.Timedelta(hours=10)).strftime("%d %b %Y")
        left, right = axes[row]
        left.fill_between(hours, stages["q05"], stages["q95"], color=BLUE, alpha=0.15, lw=0, label="Ours 90% interval")
        left.plot(hours, pdp, color=CONTEXT, lw=1.2, ls="--", label="AEMO predispatch")
        left.plot(hours, xgb, color=ORANGE, lw=1.4, label="XGBoost")
        left.plot(hours, stages["final"], color=BLUE, lw=1.8, label="Ours")
        left.plot(hours, actual, color=INK, lw=1.4, label="Actual")
        right.plot(hours, stages["base"], color="#8fb8ea", lw=1.3, label="Linear base")
        right.plot(hours, stages["fields"], color=AQUA, lw=1.3, label="+ dual fields")
        right.plot(hours, stages["calibrator"], color=ORANGE, lw=1.3, label="+ calibrator")
        right.plot(hours, stages["final"], color=BLUE, lw=1.8, label="+ price floor (final)")
        right.plot(hours, actual, color=INK, lw=1.4, label="Actual")
        for ax in (left, right):
            ax.set_xlim(0, 23)
            ax.axhline(0, color=CONTEXT, lw=0.6)
            ax.set_xticks([0, 6, 12, 18, 23])
        if region == "NSW1":  # keep the actual price visible next to the predispatch spike
            top = max(actual.max(), stages["final"].max(), xgb.max()) * 1.6
            left.set_ylim(min(actual.min(), stages["final"].min()) - 20, top)
            right.set_ylim(left.get_ylim())
            peak = int(np.argmax(pdp))
            left.annotate(f"predispatch {pdp.max():,.0f}", xy=(peak, top * 0.97), xytext=(max(peak - 10, 0.5), top * 0.85),
                          fontsize=6.2, color=MUTED, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=6))
        left.set_title(f"{titles[region]} ({day})", loc="left", color=INK)
        right.set_title("Our forecast, stage by stage (seed 2026)", loc="left", color=INK)
        left.set_ylabel("AUD/MWh")
        summary[region] = {"origin": day, "mae": {name: float(np.abs(v - actual).mean()) for name, v in
                           (("ours", stages["final"]), ("xgboost", xgb), ("predispatch", pdp), ("base", stages["base"]),
                            ("fields", stages["fields"]), ("calibrator", stages["calibrator"]))},
                           "floor": stages["floor"]}
    for ax in axes[1]:
        ax.set_xlabel("Hour of the day (AEST)")
    for column, anchor in ((0, (0.27, 0.0)), (1, (0.76, 0.0))):
        handles, labels = axes[0, column].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncols=3, loc="lower center", bbox_to_anchor=anchor, fontsize=6.2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, "cases")
    return summary


def figure_framework() -> None:
    """Overview in the style of ML venues: data thumbnails along the pipeline (QLD case window)."""
    import pandas as pd
    from matplotlib.patches import FancyBboxPatch
    from forecasting.datasets import build_region_datasets
    from forecasting.rolling import QUARTERS, quarter_config_path
    from forecasting.train import move_inputs

    region, origin = "QLD1", _case_origins()["QLD1"]
    day = pd.to_datetime(origin, unit="s") + pd.Timedelta(hours=10)
    k = int(np.searchsorted(QUARTERS, day, side="right") - 1)
    config_path = ROOT / quarter_config_path("pd_calibrator", k)
    dataset = build_region_datasets(config_path, region)["test"]
    index = int(np.flatnonzero(dataset.delivery_unix_seconds[np.asarray(dataset.origin_indices)] == origin)[0])
    sample = dataset[index]
    batch = {key: (value[None] if torch.is_tensor(value) else value) for key, value in sample.items()}
    model = _load_model(str(config_path.relative_to(ROOT)),
                        ROOT / f"outputs/forecasting/rolling/pd_calibrator/q{k}/{region}/best_model.pt", torch.device("cpu"))
    *inputs, _ = move_inputs(batch, torch.device("cpu"))
    with torch.no_grad():
        out = model(*inputs)
    history = sample["history_values"][:, 0].numpy()
    events = out["event_signal"][0, :, 0].numpy()
    trend = out["ctf_signal"][0, :, 0].numpy()
    stages = _stages(region, origin)
    raw = np.asarray(dataset.target_values_raw)
    start = int(dataset.origin_indices[index])
    actual = raw[start:start + 24]
    past = raw[start - 72:start]
    with np.load(ROOT / "data/aemo_exogenous/qld1_predispatch.npz") as a:
        pdp = a["predispatch_rrp"][int(np.searchsorted(a["forecast_origin_unix"], origin))]
    with np.load(ROOT / "data/aemo_exogenous/qld1_pdpasa.npz") as a:
        row = int(np.searchsorted(a["forecast_origin_unix"], origin))
        net_load = a["demand50_mw"][row] - a["uigf_mw"][row]

    fig = plt.figure(figsize=(7.0, 3.55))
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    def panel(x, y, w, h, title, face, edge):
        canvas.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                        facecolor=face, edgecolor=edge, linewidth=0.8))
        canvas.text(x + 0.008, y + h - 0.012, title, ha="left", va="top", fontsize=7, color=INK, weight="bold")

    def thumb(x, y, w, h, label):
        ax = fig.add_axes((x, y, w, h))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(CONTEXT)
            ax.spines[side].set_linewidth(0.6)
        ax.set_title(label, fontsize=5.8, color=MUTED, loc="left", pad=1.5)
        return ax

    def block(x, y, w, h, title, subtitle, face, edge):
        canvas.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.008",
                                        facecolor=face, edgecolor=edge, linewidth=0.8))
        canvas.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=6.4, color=INK, weight="bold")
        canvas.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center", fontsize=5.4, color=MUTED)

    def arrow(start, end, color="#888780"):
        canvas.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))

    top, bottom = 0.95, 0.20
    # (a) Inputs
    panel(0.008, bottom, 0.215, top - bottom, "(a) Inputs at origin $t$", "#f6f5f0", "#c9c7bd")
    ax = thumb(0.03, 0.66, 0.175, 0.13, "72-h price history")
    ax.plot(np.arange(-72, 0), past, color=INK, lw=0.8)
    ax.axhline(0, color=CONTEXT, lw=0.4)
    ax = thumb(0.03, 0.45, 0.175, 0.13, "AEMO predispatch price, next 24 h")
    ax.plot(np.arange(24), pdp, color=MUTED, lw=0.9, ls="--")
    ax.axhline(0, color=CONTEXT, lw=0.4)
    ax = thumb(0.03, 0.24, 0.175, 0.13, "PD PASA net load, next 24 h")
    ax.plot(np.arange(24), net_load, color=MUTED, lw=0.9)
    # (b) Dual-field decomposition
    panel(0.245, bottom, 0.235, top - bottom, "(b) Dual-field decomposition", "#fbf3ee", "#efc3ad")
    ax = thumb(0.265, 0.66, 0.195, 0.13, "detected events $\\to$ DGF")
    ax.plot(np.arange(-72, 0), history, color=CONTEXT, lw=0.8)
    ax.plot(np.arange(-72, 0), events, color=ORANGE, lw=1.0)
    ax = thumb(0.265, 0.45, 0.195, 0.13, "remainder $\\to$ CTF (trend)")
    ax.plot(np.arange(-72, 0), history - events, color=CONTEXT, lw=0.8)
    ax.plot(np.arange(-72, 0), trend, color=BLUE, lw=1.0)
    block(0.265, 0.25, 0.09, 0.11, "DGF heads", "events, scarcity", "#fdeee7", ORANGE)
    block(0.37, 0.25, 0.09, 0.11, "CTF heads", "level, net load", "#e6f1fb", BLUE)
    # (c) Known-input terms and fusion
    panel(0.502, bottom, 0.20, top - bottom, "(c) Base, calibrator, fusion", "#eef7f3", "#a9dcc6")
    block(0.517, 0.56, 0.082, 0.17, "Linear\nbase", "all inputs", "#e8f6f0", AQUA)
    block(0.607, 0.56, 0.082, 0.17, "Cali-\nbrator", "per-hour MLP", "#ffffff", "#888780")
    block(0.517, 0.26, 0.172, 0.15, "Fusion", "base + CTF + sin$^2\\theta\\cdot$DGF + cal.", "#ffffff", "#888780")
    # (d) Output
    panel(0.724, bottom, 0.268, top - bottom, "(d) Floor and forecast", "#f6f5f0", "#c9c7bd")
    ax = thumb(0.75, 0.27, 0.22, 0.52, "24-h forecast (QLD, %s)" % day.strftime("%d %b %Y"))
    hours = np.arange(24)
    ax.fill_between(hours, stages["q05"], stages["q95"], color=BLUE, alpha=0.18, lw=0, label="90% interval")
    ax.plot(hours, stages["final"], color=BLUE, lw=1.2, label="forecast")
    ax.plot(hours, actual, color=INK, lw=0.9, label="actual")
    if stages["floor"] is not None:
        ax.axhline(stages["floor"], color=ORANGE, lw=0.7, ls=":")
        ax.text(23, stages["floor"], "price floor", fontsize=5.2, color=ORANGE, ha="right", va="top")
    ax.axhline(0, color=CONTEXT, lw=0.4)
    ax.legend(frameon=False, fontsize=5.2, loc="upper left", handlelength=1.4)

    def route(points, color="#888780"):
        xs, ys = zip(*points)
        canvas.plot(xs[:-1], ys[:-1], color=color, lw=0.8, solid_capstyle="butt")
        arrow(points[-2], points[-1], color)

    # history -> decomposition
    arrow((0.223, 0.725), (0.245, 0.725))
    # decomposition -> field heads
    arrow((0.3125, 0.45), (0.3125, 0.36))
    arrow((0.415, 0.45), (0.415, 0.36))
    # known inputs: bus above the panels into (c)
    route([(0.115, 0.95), (0.115, 0.975), (0.491, 0.975), (0.491, 0.80), (0.648, 0.80), (0.648, 0.73)])
    canvas.text(0.36, 0.982, "known inputs: predispatch, PD PASA, gas, calendar, history", ha="center", va="bottom",
                fontsize=5.6, color=MUTED)
    route([(0.558, 0.80), (0.558, 0.73)])
    # field heads: bus below the heads into fusion
    canvas.plot([0.31, 0.31, 0.49, 0.49], [0.25, 0.225, 0.225, 0.335], color="#888780", lw=0.8)
    canvas.plot([0.415, 0.415], [0.25, 0.225], color="#888780", lw=0.8)
    arrow((0.49, 0.335), (0.517, 0.335))
    # base, calibrator -> fusion; fusion -> forecast
    arrow((0.558, 0.56), (0.558, 0.41))
    arrow((0.648, 0.56), (0.648, 0.41))
    arrow((0.689, 0.335), (0.75, 0.45))
    # Quarterly recalibration timeline
    y = 0.075
    canvas.text(0.008, 0.155, "(e) Refit before every test quarter $k$ (2023Q1-2024Q4)", fontsize=7, color=INK,
                weight="bold", va="center")
    segments = [(0.03, 0.70, "train: 2015 to three months before quarter $k$", "#e6f1fb", BLUE),
                (0.70, 0.80, "validate", "#fdeee7", ORANGE),
                (0.80, 0.90, "test $k$", "#e8f6f0", AQUA)]
    for x0, x1, label, face, edge in segments:
        canvas.add_patch(FancyBboxPatch((x0, y - 0.022), x1 - x0 - 0.004, 0.044, boxstyle="round,pad=0,rounding_size=0.006",
                                        facecolor=face, edgecolor=edge, linewidth=0.7))
        canvas.text((x0 + x1) / 2, y, label, ha="center", va="center", fontsize=5.8, color=INK)
    canvas.text(0.95, y, "$k{+}1$ $\\to$", ha="center", va="center", fontsize=6.2, color=MUTED)
    _save(fig, "framework")


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _write(name: str, text: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(text)


def _fmt(value: float, best: bool = False, digits: int = 2) -> str:
    text = f"{value:.{digits}f}"
    return f"\\textbf{{{text}}}" if best else text


def _stars(p: float) -> str:
    return "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")


def table_main() -> None:
    rows = []
    models = [("Ours", FINAL), ("Linear base only", SETTINGS["Quarterly refit + predispatch"]["Linear base only"])]
    models += [(name, "baselines_rolling_pd/{p}/" + kind) for name, kind in BASELINES]
    for protocol, prefix in PROTOCOLS:
        entries = []
        for name, root in models:
            paths = _run(root, protocol, prefix)
            region, mean, std = _three_seed(paths, "mae")
            entries.append((name, region, mean, std, _three_seed(paths, "rmse_window_mean")[1],
                            _three_seed(paths, "ais_90")[1], _three_seed(paths, "crps_quantile_approx")[1]))
        best = {k: min(e[i] for e in entries) for i, k in ((2, "mean"), (4, "rmse"), (5, "ais"), (6, "crps"))}
        best_region = {r: min(e[1][r] for e in entries) for r in REGIONS}
        label = "Raw prices" if protocol == "raw" else "Capped at 650"
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for name, region, mean, std, rmse, ais, crps in entries:
            cells = [_fmt(region[r], region[r] == best_region[r]) for r in REGIONS]
            cells.append(f"{_fmt(mean, mean == best['mean'])} $\\pm$ {std:.2f}")
            cells += [_fmt(rmse, rmse == best["rmse"]), _fmt(ais, ais == best["ais"], 1), _fmt(crps, crps == best["crps"])]
            rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows.pop()
    _write("same_data_comparison", "\n".join([
        "% Generated by paper/make_materials.py",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "& \\multicolumn{3}{c}{MAE} & & & & \\\\",
        "\\cmidrule(lr){2-4}",
        "Model & NSW & QLD & TAS & Mean MAE & Window RMSE & 90\\% AIS & CRPS \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]) + "\n")


def table_significance() -> None:
    result = json.loads((LOGS / "significance" / "paper_final.json").read_text())["final_vs"]
    names = {"base_only": "Linear base only", "xgboost": "XGBoost", "patchtst": "PatchTST",
             "dlinear": "DLinear", "linear_mse": "Linear"}
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "& \\multicolumn{5}{c}{Raw prices} & \\multicolumn{2}{c}{Capped at 650} \\\\",
             "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}",
             "Comparator & $\\Delta$MAE & NSW & QLD & TAS & $\\Delta$CRPS & $\\Delta$MAE & $\\Delta$CRPS \\\\", "\\midrule"]

    def cell(entry: dict) -> str:
        return f"{entry['mean_difference']:+.2f}{_stars(entry['p_value'])}"

    for key, name in names.items():
        raw_mae, raw_crps = result["raw"]["mae"][key], result["raw"]["crps"][key]
        cap_mae, cap_crps = result["capped650"]["mae"][key], result["capped650"]["crps"][key]
        cells = [cell(raw_mae["three_region_mean"]), *(cell(raw_mae[r]) for r in REGIONS),
                 cell(raw_crps["three_region_mean"]), cell(cap_mae["three_region_mean"]), cell(cap_crps["three_region_mean"])]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("significance", "\n".join(lines) + "\n")


def table_ablations_static(summary: dict) -> None:
    """Supplementary: structure and detector variants of the detected-event DGF, static split."""
    structural = [
        ("Transplanted dual field (Gabor DGF)", "gabor_trunk"),
        ("No decomposition (heads read raw history)", "raw_history"),
        ("Adaptive attention atoms + echo", "adaptive"),
        ("Detected events + daily echo", "detected_echo"),
        ("Detected-event DGF", "detected"),
    ]
    detector = [
        ("4 events", "detected_k4"), ("16 events", "detected_k16"),
        ("No suppression (1-hour separation)", "detected_sep1"), ("6-hour separation", "detected_sep6"),
        ("No threshold", "detected_nothreshold"), ("Mean baseline", "detected_mean"),
    ]
    base = [np.mean([_metrics(MODELS["detected"][s], r)["mae"] for r in REGIONS]) for s in SEEDS]
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrrrr}", "\\toprule",
             "Variant & MAE & Window RMSE & CRPS & $\\Delta$MAE vs detected (seeds) \\\\", "\\midrule",
             "\\multicolumn{5}{l}{\\emph{Structure}} \\\\"]
    for group, entries in (("structure", structural), ("detector", detector)):
        if group == "detector":
            lines += ["\\midrule", "\\multicolumn{5}{l}{\\emph{Detector settings (8 events, 3-hour separation, learned threshold, median)}} \\\\"]
        for label, key in entries:
            _, mae, std = _three_seed(MODELS[key], "mae")
            rmse = _three_seed(MODELS[key], "rmse_window_mean")[1]
            crps = _three_seed(MODELS[key], "crps_quantile_approx")[1]
            seeds = [np.mean([_metrics(MODELS[key][s], r)["mae"] for r in REGIONS]) for s in SEEDS]
            delta = "--" if key == "detected" else " / ".join(f"{a - b:+.2f}" for a, b in zip(seeds, base))
            lines.append(f"{label} & {mae:.2f} $\\pm$ {std:.2f} & {rmse:.2f} & {crps:.2f} & {delta} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("ablations_static", "\n".join(lines) + "\n")
    _write("field_separation", "\n".join([
        "% Generated by paper/make_materials.py",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "Event field & NSW & QLD & TAS \\\\", "\\midrule",
        *(f"{label} & " + " & ".join(f"{summary[key][r]:.2f}" for r in REGIONS) + " \\\\"
          for label, key in (("Transplanted (Gabor)", "gabor"), ("Detected events (ours)", "detected"))),
        "\\bottomrule", "\\end{tabular}",
    ]) + "\n")


def table_ablations() -> None:
    """When the fields help, and what each later component adds (Diebold-Mariano, pooled)."""
    pairs = json.loads((LOGS / "significance" / "paper_final.json").read_text())["pairs"]
    rows = [
        ("\\emph{Fields minus linear base}", None),
        ("Static split", "fields_static"),
        ("Quarterly refit", "fields_rolling"),
        ("Quarterly refit + predispatch", "fields_rolling_predispatch"),
        ("\\emph{Added to fields + predispatch}", None),
        ("Hinge features on predispatch price", "hinge"),
        ("Calibrator", "calibrator"),
        ("Price floor (after calibrator)", "floor"),
        ("Interconnector inputs (after floor)", "interconnector"),
    ]
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrrrr}", "\\toprule",
             "& \\multicolumn{2}{c}{Raw prices} & \\multicolumn{2}{c}{Capped at 650} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Change & $\\Delta$MAE & $\\Delta$CRPS & $\\Delta$MAE & $\\Delta$CRPS \\\\", "\\midrule"]
    for label, key in rows:
        if key is None:
            lines.append(f"\\multicolumn{{5}}{{l}}{{{label}}} \\\\")
            continue
        cells = []
        for protocol in ("raw", "capped650"):
            for loss in ("mae", "crps"):
                entry = pairs[protocol][loss][key]["three_region_mean"]
                cells.append(f"{entry['mean_difference']:+.2f}{_stars(entry['p_value'])}")
        lines.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("ablations", "\n".join(lines) + "\n")


def table_settings() -> None:
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrrrrrr}", "\\toprule",
             "& \\multicolumn{3}{c}{Raw prices} & \\multicolumn{3}{c}{Capped at 650} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Model & Static & Refit & Refit + PD & Static & Refit & Refit + PD \\\\", "\\midrule"]
    models = list(SETTINGS["Static split"])
    values = {m: [_three_seed(_run(SETTINGS[s][m], p, pre), "mae")[1] for p, pre in PROTOCOLS for s in SETTINGS]
              for m in models}
    best = [min(values[m][i] for m in models) for i in range(6)]
    for m in models:
        lines.append(f"{m} & " + " & ".join(_fmt(v, v == best[i]) for i, v in enumerate(values[m])) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("settings", "\n".join(lines) + "\n")


def table_tails() -> None:
    names = [("Ours", None), ("Linear base only", "base_only"), ("XGBoost", "xgboost"), ("PatchTST", "patchtst"),
             ("DLinear", "dlinear"), ("Linear", "linear_mse")]
    keys = (("crps_by_band", "<0"), ("pinball_q05_negative_hours", None), ("crps_by_band", ">300"),
            ("pinball_q95_spike_hours", None), ("coverage_90", None))
    tests = ("crps_negative", "q05_negative", "crps_spike", "q95_spike")
    significance = json.loads((LOGS / "significance" / "paper_tails.json").read_text())["results"]
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrrrrr}", "\\toprule",
             "& \\multicolumn{2}{c}{Negative-price hours} & \\multicolumn{2}{c}{Spike hours ($>$300)} & \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Model & CRPS & $q_{0.05}$ pinball & CRPS & $q_{0.95}$ pinball & 90\\% coverage \\\\", "\\midrule"]
    for protocol, _ in PROTOCOLS:
        tails = json.loads((LOGS / "predispatch_rolling" / f"{protocol}_tail_metrics.json").read_text())
        final = json.loads((LOGS / "price_floor" / f"{protocol}_tail_metrics.json").read_text())["floor"]
        table = {name: (final if key is None else tails[key])["mean"] for name, key in names}
        values = {name: [table[name][k][sub] if sub else table[name][k] for k, sub in keys] for name, _ in names}
        best = [min(values[n][i] for n, _ in names) for i in range(4)]
        label = "Raw prices" if protocol == "raw" else "Capped at 650"
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for name, key in names:
            v = values[name]
            cells = []
            for i in range(4):
                mark = ""
                if key is not None:
                    test = significance[protocol][tests[i]][key]["three_region_mean"]
                    if test["p_value"] < 0.05:
                        mark = "$^{\\dagger}$" if test["mean_difference"] < 0 else "$^{\\ddagger}$"
                cells.append(_fmt(v[i], v[i] == best[i], 1 if i in (2, 3) else 2) + mark)
            cells.append(f"{100 * v[4]:.1f}\\%")
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write("tails", "\n".join(lines) + "\n")


def main() -> None:
    figure_architecture()
    figure_framework()
    figure_settings()
    figure_comparison()
    figure_ladder()
    cases = figure_cases()
    (LOGS / "paper_cases.json").write_text(json.dumps(cases, indent=2))
    summary = figure_mechanism(mechanism_data())
    table_main()
    table_significance()
    table_ablations()
    table_settings()
    table_tails()
    print(json.dumps({"cost": table_cost()}, indent=2))
    table_ablations_static(summary)
    print(json.dumps({"field_separation_share": summary}, indent=2))


if __name__ == "__main__":
    main()
