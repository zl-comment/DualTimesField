"""Build the paper's figures and tables from the recorded results.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python paper/make_materials.py

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
    return json.loads((directory / f"{region}.json").read_text())["test"]["model"]


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

# Ablation ladder: three-region mean test MAE and CRPS~ (three seeds from step 4 on).
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


def figure_comparison() -> None:
    models = [("Ours", None), ("XGBoost", "xgboost"), ("GRU", "gru"), ("DeepAR", "deepar")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, protocol, ours_key, title in (
        (axes[0], "raw", "detected", "Raw prices"),
        (axes[1], "capped650", "detected_capped", "Prices capped at 650 AUD/MWh"),
    ):
        width = 0.2
        x = np.arange(len(REGIONS))
        for i, (name, kind) in enumerate(models):
            per_region = (_three_seed(MODELS[ours_key], "mae")[0] if kind is None
                          else _baseline(protocol, kind, "mae")[0])
            values = [per_region[r] for r in REGIONS]
            color = BLUE if kind is None else ["#8a8983", "#b4b2a9", "#d3d1c7"][i - 1]
            ax.bar(x + (i - 1.5) * width, values, width=width - 0.02, color=color, label=name)
        ax.set_xticks(x, [r[:-1] for r in REGIONS])
        ax.set_title(f"{title}: test MAE (AUD/MWh)", loc="left", color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, None)
    axes[0].legend(ncols=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    _save(fig, "same_data_comparison")


def figure_architecture() -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    styles = {
        "input": ("#f1efe8", "#888780"),
        "ctf": ("#e6f1fb", BLUE),
        "dgf": ("#fdeee7", ORANGE),
        "neutral": ("#ffffff", "#888780"),
    }

    def box(x, y, w, h, title, subtitle, kind):
        face, edge = styles[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, linewidth=0.9))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=8, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=6.5, color=MUTED)
        return x, y, w, h

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#888780", lw=0.8, shrinkA=0, shrinkB=0,
                                    mutation_scale=8))

    box(34, 58, 32, 7, "72-h history", "price (asinh, standardized), demand", "input")
    box(56, 46.5, 36, 7.5, "Event detection", "median departures, top-K, suppression", "dgf")
    box(56, 35.5, 36, 7.5, "DGF: event field", "sparse bumps, learned threshold, width", "dgf")
    box(8, 35.5, 36, 7.5, "CTF: trend field", "band-limited Fourier INR", "ctf")
    ax.text(50, 31.4, "decomposition loss: history = CTF + DGF", ha="center", fontsize=6.5, color=MUTED)
    box(20, 21.5, 24, 7, "Fundamentals", "gas price, net load forecast", "input")
    box(56, 21.5, 24, 7, "Scarcity", "spare capacity, shortfalls", "input")
    box(8, 11.5, 36, 7, "CTF linear heads", "level: point and quantiles", "ctf")
    box(56, 11.5, 36, 7, "DGF linear heads", "events: point and quantiles", "dgf")
    box(30, 1.2, 40, 7.5, "Trigonometric fusion", "CTF + sin\u00b2\u03b8\u00b7DGF, per-quantile gates", "neutral")

    arrow((50, 58), (50, 56.3))
    arrow((50, 56.3), (74, 54.3))
    arrow((50, 56.3), (26, 43.3))
    arrow((74, 46.5), (74, 43.3))
    arrow((14, 35.5), (14, 18.7))
    arrow((86, 35.5), (86, 18.7))
    arrow((32, 21.5), (32, 18.7))
    arrow((68, 21.5), (68, 18.7))
    arrow((26, 11.5), (38, 8.9))
    arrow((74, 11.5), (62, 8.9))
    ax.text(50, -1.8, "Output: 24-hour point forecast and 0.05/0.10/0.50/0.90/0.95 quantiles",
            ha="center", fontsize=6.5, color=MUTED)
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


def table_main() -> None:
    rows = []
    for protocol, ours in (("raw", "detected"), ("capped650", "detected_capped")):
        entries = []
        region, mean, std = _three_seed(MODELS[ours], "mae")
        entries.append(("Ours", region, mean, std,
                        _three_seed(MODELS[ours], "rmse_window_mean")[1],
                        _three_seed(MODELS[ours], "ais_90")[1],
                        _three_seed(MODELS[ours], "crps_quantile_approx")[1]))
        for name, kind in (("XGBoost", "xgboost"), ("GRU", "gru"), ("DeepAR", "deepar")):
            region, mean, std = _baseline(protocol, kind, "mae")
            entries.append((name, region, mean, std, _baseline(protocol, kind, "rmse_window_mean")[1],
                            _baseline(protocol, kind, "ais_90")[1], _baseline(protocol, kind, "crps_quantile_approx")[1]))
        best = {k: min(e[i] if k != "region" else 0 for e in entries) for i, k in ((2, "mean"), (4, "rmse"), (5, "ais"), (6, "crps"))}
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
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Comparator & Pooled $\\Delta$MAE & DM & NSW & QLD & TAS \\\\", "\\midrule"]
    names = {"gabor_trunk": "Transplanted dual field", "softclip_trunk": "Previous trunk", "raw_history_ablation": "No decomposition",
             "gbdt": "GBDT (same inputs)", "xgboost": "XGBoost", "gru": "GRU", "deepar": "DeepAR", "naive": "Seasonal naive"}

    def star(p: float) -> str:
        return "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")

    for protocol, file in (("Raw prices", "raw_detected_dgf.json"), ("Capped at 650", "capped650_detected_dgf.json")):
        result = json.loads((LOGS / "significance" / file).read_text())
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{protocol}}}}} \\\\")
        for name, comparison in result["comparisons"].items():
            pooled = comparison["three_region_mean"]
            cells = [f"{pooled['mean_difference']:+.2f}{star(pooled['p_value'])}", f"{pooled['statistic']:+.2f}"]
            cells += [f"{comparison[r]['mean_difference']:+.2f}{star(comparison[r]['p_value'])}" for r in REGIONS]
            lines.append(f"{names.get(name, name)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write("significance", "\n".join(lines) + "\n")


def table_ablations(summary: dict) -> None:
    structural = [
        ("Transplanted dual field (Gabor DGF)", "gabor_trunk"),
        ("No decomposition (heads read raw history)", "raw_history"),
        ("Adaptive attention atoms + echo", "adaptive"),
        ("Detected events + daily echo", "detected_echo"),
        ("Detected-event DGF (ours)", "detected"),
    ]
    detector = [
        ("4 events", "detected_k4"), ("16 events", "detected_k16"),
        ("No suppression (1-hour separation)", "detected_sep1"), ("6-hour separation", "detected_sep6"),
        ("No threshold", "detected_nothreshold"), ("Mean baseline", "detected_mean"),
    ]
    base = [np.mean([_metrics(MODELS["detected"][s], r)["mae"] for r in REGIONS]) for s in SEEDS]
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrrrr}", "\\toprule",
             "Variant & MAE & Window RMSE & CRPS & $\\Delta$MAE vs ours (seeds) \\\\", "\\midrule",
             "\\multicolumn{5}{l}{\\emph{Structure}} \\\\"]
    for group, entries in (("structure", structural), ("detector", detector)):
        if group == "detector":
            lines += ["\\midrule", "\\multicolumn{5}{l}{\\emph{Detector settings (ours: 8 events, 3-hour separation, learned threshold, median)}} \\\\"]
        for label, key in entries:
            _, mae, std = _three_seed(MODELS[key], "mae")
            rmse = _three_seed(MODELS[key], "rmse_window_mean")[1]
            crps = _three_seed(MODELS[key], "crps_quantile_approx")[1]
            seeds = [np.mean([_metrics(MODELS[key][s], r)["mae"] for r in REGIONS]) for s in SEEDS]
            delta = "--" if key == "detected" else " / ".join(f"{a - b:+.2f}" for a, b in zip(seeds, base))
            lines.append(f"{label} & {mae:.2f} $\\pm$ {std:.2f} & {rmse:.2f} & {crps:.2f} & {delta} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("ablations", "\n".join(lines) + "\n")
    shares = " & ".join(f"{summary['gabor'][r]:.2f} & {summary['detected'][r]:.2f}" for r in REGIONS)
    _write("field_separation", "\n".join([
        "% Generated by paper/make_materials.py",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "& \\multicolumn{2}{c}{NSW} & \\multicolumn{2}{c}{QLD} & \\multicolumn{2}{c}{TAS} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "& Gabor & Ours & Gabor & Ours & Gabor & Ours \\\\", "\\midrule",
        f"Event-field share of past spike hours & {shares} \\\\",
        "\\bottomrule", "\\end{tabular}",
    ]) + "\n")


def table_re_price() -> None:
    reported = {  # RE-Price Tables 2 and 3: MAE, CRPS (with news).
        "RE-Price": ((23.48, 25.85, 18.96), (17.36, 20.58, 13.13)),
        "GPT4TS": ((28.12, 29.80, 22.88), (21.70, 24.51, 16.75)),
        "Informer": ((29.25, 31.80, 30.85), (22.68, 21.78, 20.55)),
        "DeepAR": ((33.77, 36.67, 26.51), (26.25, 25.94, 20.18)),
        "GRU": ((33.24, 43.24, 30.40), (25.12, 28.48, 19.66)),
        "XGBoost": ((38.63, 44.70, 37.56), (28.79, 29.17, 23.43)),
    }
    region, mean, _ = _three_seed(MODELS["detected_capped"], "mae")
    crps_region, crps_mean, _ = _three_seed(MODELS["detected_capped"], "crps_quantile_approx")
    lines = ["% Generated by paper/make_materials.py", "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             "& \\multicolumn{4}{c}{MAE} & \\multicolumn{4}{c}{CRPS} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             "Model & NSW & QLD & TAS & Mean & NSW & QLD & TAS & Mean \\\\", "\\midrule",
             f"Ours (no news, capped at 650) & {' & '.join(f'{region[r]:.2f}' for r in REGIONS)} & {mean:.2f} & "
             f"{' & '.join(f'{crps_region[r]:.2f}' for r in REGIONS)} & {crps_mean:.2f} \\\\", "\\midrule"]
    for name, (mae, crps) in reported.items():
        lines.append(f"{name} (reported, with news) & {' & '.join(f'{v:.2f}' for v in mae)} & {np.mean(mae):.2f} & "
                     f"{' & '.join(f'{v:.2f}' for v in crps)} & {np.mean(crps):.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("re_price_reference", "\n".join(lines) + "\n")


def main() -> None:
    figure_architecture()
    figure_ladder()
    figure_comparison()
    summary = figure_mechanism(mechanism_data())
    table_main()
    table_significance()
    table_ablations(summary)
    table_re_price()
    print(json.dumps({"field_separation_share": summary}, indent=2))


if __name__ == "__main__":
    main()
