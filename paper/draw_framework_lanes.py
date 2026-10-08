"""Overview figure in a lane layout: no connecting wires.

Each additive term of the forecast has its own horizontal lane; the inputs a
lane reads are shown as small x/z tags at its start, and the lane outputs are
stacked at the right so that they read as one sum. Stages are separated by
white space and pale tints instead of outlines and arrows. Thumbnails are real
data from one QLD1 test day (the case-study day), drawn from the seed-2026
quarterly refit.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python paper/draw_framework_lanes.py

Writes ``paper/figures/framework_lanes.{pdf,png}`` (vector PDF).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_materials as mm  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, MUTED, FAINT = "#1f2328", "#5f6368", "#9aa0a6"
NEUTRAL_TINT = "#f1f0ec"
BASE, TREND, EVENT, CAL = "#2f8577", "#3a6ea5", "#c2643c", "#6f6e69"


def case_data() -> dict:
    import pandas as pd
    from forecasting.datasets import build_region_datasets
    from forecasting.rolling import QUARTERS, quarter_config_path
    from forecasting.train import move_inputs

    region, origin = "QLD1", mm._case_origins()["QLD1"]
    day = pd.to_datetime(origin, unit="s") + pd.Timedelta(hours=10)
    k = int(np.searchsorted(QUARTERS, day, side="right") - 1)
    config_path = mm.ROOT / quarter_config_path("pd_calibrator", k)
    dataset = build_region_datasets(config_path, region)["test"]
    index = int(np.flatnonzero(dataset.delivery_unix_seconds[np.asarray(dataset.origin_indices)] == origin)[0])
    sample = dataset[index]
    batch = {key: (value[None] if torch.is_tensor(value) else value) for key, value in sample.items()}
    model = mm._load_model(str(config_path.relative_to(mm.ROOT)),
                           mm.ROOT / f"outputs/forecasting/rolling/pd_calibrator/q{k}/{region}/best_model.pt",
                           torch.device("cpu"))
    *inputs, _ = move_inputs(batch, torch.device("cpu"))
    with torch.no_grad():
        out = model(*inputs)
    raw = np.asarray(dataset.target_values_raw)
    start = int(dataset.origin_indices[index])
    with np.load(mm.ROOT / "data/aemo_exogenous/qld1_predispatch.npz") as a:
        predispatch = a["predispatch_rrp"][int(np.searchsorted(a["forecast_origin_unix"], origin))]
    return {
        "day": day.strftime("%d %b %Y"),
        "history": sample["history_values"][:, 0].numpy(),
        "demand": sample["history_values"][:, 1].numpy(),
        "events": out["event_signal"][0, :, 0].numpy(),
        "trend": out["ctf_signal"][0, :, 0].numpy(),
        "past": raw[start - 72:start],
        "actual": raw[start:start + 24],
        "predispatch": predispatch,
        "stages": mm._stages(region, origin),
    }


def draw(data: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.16, 3.3))
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, 100)
    canvas.set_ylim(0, 46)
    canvas.axis("off")
    to_fig = lambda x, y: (x / 100, y / 46)  # canvas units -> figure fraction

    def tint(x, y, w, h, color, alpha=1.0, radius=1.0):
        canvas.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                        facecolor=color, edgecolor="none", alpha=alpha, zorder=0))

    def text(x, y, s, size=6.6, color=INK, weight="normal", ha="left", va="center", **kw):
        canvas.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, **kw)

    def tag(x, y, s, color):
        canvas.add_patch(FancyBboxPatch((x, y - 0.95), 2.3, 1.9, boxstyle="round,pad=0,rounding_size=0.45",
                                        facecolor=color, edgecolor="none", zorder=2))
        canvas.text(x + 1.15, y, s, fontsize=6.6, color="white", ha="center", va="center", zorder=3,
                    style="italic", weight="bold")

    def thumb(x, y, w, h):
        ax = fig.add_axes((*to_fig(x, y), w / 100, h / 46))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.patch.set_alpha(0)
        for side in ax.spines.values():
            side.set_visible(False)
        return ax

    def chevron(x, y):
        text(x, y, "›", size=17, color="#c9c7bf", ha="center", va="center")

    # Stage headings: a number and a title, no boxes.
    for x, number, title in ((1.0, "1", "Inputs at origin $t$"), (20.5, "2", "Additive forecaster"),
                             (83.0, "3", "Forecast")):
        text(x, 43.6, number, size=7.4, color=TREND, weight="bold")
        text(x + 1.7, 43.6, title, size=7.4, weight="bold")

    # 1. Inputs: two neutral tiles with their tags.
    tint(1.0, 22.2, 16.5, 18.6, NEUTRAL_TINT)
    tag(2.0, 38.9, "x", INK)
    text(5.0, 38.9, "History, 72 h", weight="bold")
    ax = thumb(2.0, 27.0, 14.5, 10.0)
    hours = np.arange(-72, 0)
    ax.plot(hours, data["past"], color=INK, lw=0.8)
    ax.axhline(0, color=FAINT, lw=0.4)
    text(2.0, 24.6, "price (shown), demand", size=5.8, color=MUTED)
    text(2.0, 23.2, "observed up to $t-1$", size=5.8, color=MUTED)

    tint(1.0, 5.6, 16.5, 15.0, NEUTRAL_TINT)
    tag(2.0, 18.7, "z", MUTED)
    text(5.0, 18.7, "Known inputs", weight="bold")
    for i, line in enumerate(("AEMO predispatch", "PD PASA spare, net load", "gas price, calendar")):
        text(2.4, 15.4 - 2.4 * i, "•  " + line, size=6.0, color=INK)
    text(2.0, 7.4, "published by $t$", size=5.8, color=MUTED)

    chevron(19.0, 23.0)

    # 2. Four additive lanes. Each: tint, title, input tags, content, output term at the right.
    lanes = [  # centre, height, colour, title, inputs, note, output term
        (37.6, 5.2, BASE, "Linear base", ("x", "z"), "one linear map of the history and all known inputs", r"$B_h$"),
        (28.5, 8.0, TREND, "Trend field (CTF)", ("x",), None, r"$C_h$"),
        (18.3, 8.0, EVENT, "Event field (DGF)", ("x",), None, r"$g_h D_h$"),
        (9.2, 5.2, CAL, "Calibrator", ("z",), "per-hour MLP on $z_h$, horizon and recent price", r"$\delta_h$"),
    ]
    for y, h, color, title, inputs, note, term in lanes:
        tint(20.5, y - h / 2, 51.6, h, color, alpha=0.085)
        top = y + h / 2 - 1.5
        text(21.5, top, title, size=6.8, color=color, weight="bold")
        tx = 21.5
        for s_ in inputs:
            tag(tx, top - 2.4, s_, INK if s_ == "x" else MUTED)
            tx += 2.7
        if note:
            text(35.5, y, note, size=6.1, color=INK)
        text(75.7, y, term, size=8.4, color=color, ha="center")

    # Field lanes: real decomposition of the history, then what the heads read.
    ax = thumb(35.5, 25.1, 12.5, 6.8)
    ax.plot(hours, data["history"], color="#c9c7bf", lw=0.8)
    ax.plot(hours, data["trend"], color=TREND, lw=1.3)
    text(49.6, 30.1, "trend $c$: smooth part of the history", size=6.0)
    tag(49.6, 26.9, "z", MUTED)
    text(52.6, 26.9, "heads: $c$, $r$, net load, gas", size=5.8, color=MUTED)

    ax = thumb(35.5, 14.9, 12.5, 6.8)
    ax.plot(hours, data["history"], color="#c9c7bf", lw=0.8)
    ax.plot(hours, data["events"], color=EVENT, lw=1.3)
    text(49.6, 19.9, "events $e$: largest departures", size=6.0)
    tag(49.6, 16.7, "z", MUTED)
    text(52.6, 16.7, "heads: $e$, spare, predispatch", size=5.8, color=MUTED)
    text(46.3, 23.4, "history $= c + e +$ remainder $r$ (decomposition loss)", size=5.6, color=MUTED, ha="center")

    # The sum: lane outputs stacked in one pale column, joined by plus signs.
    tint(72.7, 5.6, 6.0, 35.6, NEUTRAL_TINT, alpha=1.0)
    for y in (33.05, 23.4, 13.75):
        text(75.7, y, "+", size=8.0, color=MUTED, ha="center")
    text(75.7, 3.5, r"$=\hat{y}_h$", size=8.4, color=INK, ha="center")

    chevron(81.0, 23.0)

    # 3. Forecast: transform, floor, and the real forecast for the case day.
    text(83.0, 39.2, r"$\hat{p}_h=\max\{\mathrm{asinh}^{-1}(\hat{y}_h),\ \varphi\}$", size=6.6)
    text(83.0, 37.0, r"floor $\varphi$ chosen on validation", size=5.8, color=MUTED)
    ax = thumb(83.0, 11.0, 16.0, 24.0)
    stages = data["stages"]
    h24 = np.arange(24)
    ax.fill_between(h24, stages["q05"], stages["q95"], color=TREND, alpha=0.14, lw=0)
    ax.plot(h24, data["actual"], color=INK, lw=0.9)
    ax.plot(h24, stages["final"], color=TREND, lw=1.3)
    if stages["floor"] is not None:
        ax.axhline(stages["floor"], color=EVENT, lw=0.7, ls=(0, (1.5, 1.5)))
    ax.axhline(0, color=FAINT, lw=0.4)
    text(83.0, 8.6, "forecast, 90% interval", size=5.8, color=TREND)
    text(83.0, 6.8, "actual", size=5.8, color=INK)
    text(91.0, 6.8, "floor $\\varphi$", size=5.8, color=EVENT)
    text(83.0, 4.4, f"QLD1, {data['day']}", size=5.6, color=MUTED)

    # Refit schedule: a slim segmented bar, no outlines.
    y = 0.9
    for x0, x1, label, color in ((20.5, 56.0, "train: 2015 to three months before quarter $k$", TREND),
                                 (56.4, 63.5, "validate", EVENT), (63.9, 70.5, "test $k$", BASE)):
        tint(x0, y, x1 - x0, 2.3, color, alpha=0.12, radius=0.6)
        text((x0 + x1) / 2, y + 1.15, label, size=5.6, color=INK, ha="center")
    text(1.0, y + 1.15, "Quarterly refits, 2023Q1-2024Q4", size=5.8, color=MUTED)

    for suffix in ("pdf", "png"):
        fig.savefig(mm.FIGURES / f"framework_lanes.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    draw(case_data())
