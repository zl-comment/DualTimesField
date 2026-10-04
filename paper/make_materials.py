"""Build the paper's figures and tables from the recorded results.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python paper/make_materials.py

The main line is the quarterly-refit setting with AEMO predispatch inputs; the
static-split development ladder, the structural ablations of the detected-event
DGF, and the field-separation figure are supplementary.

Figures go to ``paper/figures`` (PDF and PNG) and tables to ``paper/tables``
(LaTeX, booktabs). Every number is read from ``logs/forecasting`` or recomputed
from the saved checkpoints; nothing is typed in by hand except the RE-Price
values, which are copied from its Tables 2 and 3.
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
        for name, _ in names:
            v = values[name]
            cells = [_fmt(v[i], v[i] == best[i], 1 if i in (2, 3) else 2) for i in range(4)] + [f"{100 * v[4]:.1f}\\%"]
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write("tails", "\n".join(lines) + "\n")


def table_re_price() -> None:
    reported = {  # RE-Price Tables 2 and 3: MAE, CRPS (with news).
        "RE-Price": ((23.48, 25.85, 18.96), (17.36, 20.58, 13.13)),
        "GPT4TS": ((28.12, 29.80, 22.88), (21.70, 24.51, 16.75)),
        "Informer": ((29.25, 31.80, 30.85), (22.68, 21.78, 20.55)),
        "DeepAR": ((33.77, 36.67, 26.51), (26.25, 25.94, 20.18)),
        "GRU": ((33.24, 43.24, 30.40), (25.12, 28.48, 19.66)),
        "XGBoost": ((38.63, 44.70, 37.56), (28.79, 29.17, 23.43)),
    }
    ours = _run(FINAL, "capped650", "capped650_")
    region, mean, _ = _three_seed(ours, "mae")
    crps_region, crps_mean, _ = _three_seed(ours, "crps_quantile_approx")
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             "& \\multicolumn{4}{c}{MAE} & \\multicolumn{4}{c}{CRPS} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             "Model & NSW & QLD & TAS & Mean & NSW & QLD & TAS & Mean \\\\", "\\midrule",
             f"Ours (no news, predispatch, capped at 650) & {' & '.join(f'{region[r]:.2f}' for r in REGIONS)} & {mean:.2f} & "
             f"{' & '.join(f'{crps_region[r]:.2f}' for r in REGIONS)} & {crps_mean:.2f} \\\\", "\\midrule"]
    for name, (mae, crps) in reported.items():
        lines.append(f"{name} (reported, with news) & {' & '.join(f'{v:.2f}' for v in mae)} & {np.mean(mae):.2f} & "
                     f"{' & '.join(f'{v:.2f}' for v in crps)} & {np.mean(crps):.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("re_price_reference", "\n".join(lines) + "\n")


def main() -> None:
    figure_architecture()
    figure_settings()
    figure_comparison()
    figure_ladder()
    summary = figure_mechanism(mechanism_data())
    table_main()
    table_significance()
    table_ablations()
    table_settings()
    table_tails()
    table_ablations_static(summary)
    table_re_price()
    print(json.dumps({"field_separation_share": summary}, indent=2))


if __name__ == "__main__":
    main()
