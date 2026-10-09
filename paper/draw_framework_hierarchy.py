"""Vector method figure: a compact overview and two expanded field mechanisms.

Run ``PYTHONPATH=. .venv/bin/python paper/draw_framework_hierarchy.py``.
Thumbnails and atoms come from the existing QLD1 case and seed-2026 checkpoint.
Fourier thumbnails show three actual learned basis functions, not measurements.
No training is performed. Previous framework versions are left untouched.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
CACHE = ROOT / "paper/figure_data/framework_hierarchy.npz"
INK, MUTED, LINE = "#25313c", "#68727b", "#d6dce0"
BLUE, CORAL, TEAL = "#477a9d", "#b96e59", "#527e72"
PALE_BLUE, PALE_CORAL = "#f0f5f8", "#fbf3ef"


def load_case(refresh=False):
    """Cache only figure data; invalidate when the relevant checkpoint changes."""
    checkpoint = ROOT / "outputs/forecasting/rolling/pd_calibrator/q3/QLD1/best_model.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Include extraction code, configuration and price-floor selection in cache
    # validation, so changing a dependency cannot silently retain old thumbnails.
    dependencies = [checkpoint,
        ROOT / "paper/draw_framework_lanes.py", ROOT / "paper/make_materials.py",
        ROOT / "forecasting/models.py", ROOT / "src/dualfield/core.py",
        ROOT / "configs/aemo_forecast_rolling_pd_calibrator_q3.yaml",
        ROOT / "logs/forecasting/price_floor/paper_metrics/pd_calibrator_floor/choices_QLD1.json",
        ROOT / "outputs/forecasting/significance_inputs/rolling_pd_calibrator_floor/QLD1.npz",
        ROOT / "data/aemo_exogenous/qld1_predispatch.npz"]
    state = hashlib.sha256()
    for path in dependencies:
        state.update(str(path.relative_to(ROOT)).encode())
        state.update(hashlib.sha256(path.read_bytes()).digest())
    extraction_digest = state.hexdigest()
    if CACHE.is_file() and not refresh:
        with np.load(CACHE, allow_pickle=False) as stored:
            if (str(stored["checkpoint_sha256"]) == digest
                    and "extraction_sha256" in stored
                    and str(stored["extraction_sha256"]) == extraction_digest):
                return {key: stored[key] for key in stored.files}
    import torch
    from paper.draw_framework_lanes import case_data
    from paper.make_materials import _load_model

    torch.set_num_threads(4)
    data = case_data()
    model = _load_model("configs/aemo_forecast_rolling_pd_calibrator_q3.yaml",
                        checkpoint, torch.device("cpu"))
    history = torch.from_numpy(np.stack([data["history"], data["demand"]], axis=-1))[None]
    with torch.no_grad():
        ctf, dgf = model.dual_field.ctf, model.dual_field.dgf
        positions, departures = dgf.detect(history[..., 0])
        time = model._history_time(history)
        event, amplitudes, _ = dgf.extract_events(history, time)
        lowpass = ctf._lowpass_filter(history)[0, :, 0].numpy()
        width = float(torch.nn.functional.softplus(dgf.raw_width) + 0.25)
        threshold = float(torch.nn.functional.softplus(dgf.raw_threshold))
        centres = positions[0].numpy()
        atoms = amplitudes[0, :, 0].numpy()[:, None] * np.exp(
            -(np.arange(72)[None] - centres[:, None]) ** 2 / (2 * width**2))
        frequencies = ctf.get_frequencies().numpy()
    np.testing.assert_allclose(atoms.sum(axis=0), data["events"], atol=2e-6)
    data_out = {key: np.asarray(data[key]) for key in
                ("day", "history", "demand", "past", "actual", "trend", "events")}
    data_out.update(lowpass=lowpass, centres=centres, atoms=atoms,
                    departures=departures[0].numpy(), frequencies=frequencies,
                    width=np.asarray(width), threshold=np.asarray(threshold),
                    point=data["stages"]["final"], q05=data["stages"]["q05"],
                    q95=data["stages"]["q95"], checkpoint_sha256=np.asarray(digest),
                    checkpoint=np.asarray(str(checkpoint.relative_to(ROOT))),
                    extraction_sha256=np.asarray(extraction_digest))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **data_out)
    return data_out


def draw(data):
    # Use installed TrueType outlines; some PDF readers warn about the CFF
    # Nimbus Sans exports produced by this Matplotlib version.
    for path in Path("/usr/share/fonts/truetype/freefont").glob("FreeSans*.ttf"):
        font_manager.fontManager.addfont(str(path))
    style = {"font.family": "FreeSans", "font.size": 7,
             "mathtext.fontset": "stix", "pdf.fonttype": 42,
             "svg.fonttype": "none", "axes.grid": False}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.5, 5.45), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, 100), ylim=(0, 72))
        ax.axis("off")

        def text(x, y, s, size=7, color=INK, weight="normal", ha="left", **kw):
            # FreeSans advertises its bold face as semibold (600).
            weight = 600 if weight == "bold" else weight
            return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                           ha=ha, va="center", zorder=10, **kw)

        def box(x, y, w, h, s="", color=INK, fill="white", size=7, edge=None):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0,rounding_size=0.6", linewidth=.65,
                         facecolor=fill, edgecolor=edge or color, zorder=3))
            if s:
                text(x+w/2, y+h/2, s, size, color, ha="center")

        def arrow(a, b, color=INK, rad=0, lw=.85, dashed=False):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                         mutation_scale=7, linewidth=lw, color=color,
                         connectionstyle=f"arc3,rad={rad}",
                         linestyle=(0, (2, 2)) if dashed else "-",
                         shrinkA=0, shrinkB=0, zorder=4))

        def route(points, color=INK):
            xs, ys = zip(*points[:-1])
            ax.plot(xs, ys, color=color, linewidth=.8, solid_joinstyle="round", zorder=4)
            arrow(points[-2], points[-1], color)

        def section(y, letter, title, subtitle, color=INK):
            text(2, y, f"({letter})", 9, color, "bold")
            text(7, y, title, 9, color, "bold")
            text(98, y, subtitle, 6.5, MUTED, ha="right")

        def plot(x, y, w, h):
            p = fig.add_axes((x/100, y/72, w/100, h/72), facecolor="none")
            p.set_xticks([]); p.set_yticks([])
            p.grid(False)
            for spine in p.spines.values():
                spine.set_visible(False)
            p.margins(x=.02, y=.12)
            return p

        hours = np.arange(-72, 0)
        # Overview: only the principal architecture, not all implementation details.
        section(69.5, "a", "Rolling forecast", "72 h history  /  24 h horizon")
        text(2, 65.3, r"History $x$", 7.8, weight="bold")
        p = plot(2, 58.7, 16, 5.2)
        p.plot(hours, data["past"], color=INK, lw=.8)
        text(2, 57.3, r"$t-72$", 6.5, MUTED)
        text(18, 57.3, r"$t-1$", 6.5, MUTED, ha="right")
        text(2, 54.4, r"Known at origin $z$", 7, TEAL)
        text(2, 52.2, "Predispatch · PD PASA · calendar · gas", 6, MUTED)
        arrow((19, 61), (25, 61))
        box(25, 57.9, 20, 6.2, edge=LINE, fill="#f7f9fa")
        text(35, 62, "Dual-field residual", 7.6, weight="bold", ha="center")
        text(35, 59.7, "CTF (b)  +  gated DGF (c)", 6.6, ha="center")
        arrow((45, 61), (65.5, 61))
        text(53.5, 62.6, r"$F_h=C_h+g_hD_h$", 8, ha="center")
        box(48, 65.3, 14, 3.2, "Linear base", MUTED, "#fafafa", edge=LINE)
        text(46.5, 66.9, r"$x,z$", 7, MUTED, ha="right")
        route([(62, 66.9), (67, 66.9), (67, 62.3)], MUTED)
        text(68.5, 65.5, r"$B_h$", 7, MUTED)
        box(48, 53.4, 14, 3.2, "Calibrator", MUTED, "#fafafa", edge=LINE)
        text(46.5, 55, "context, horizon,\nrecent price", 6, MUTED, ha="right", linespacing=1.15)
        route([(62, 55), (67, 55), (67, 59.7)], MUTED)
        text(68.5, 56.2, r"$\delta_h$", 7, MUTED)
        ax.add_patch(Circle((67, 61), 1.3, facecolor="white", edgecolor=INK, lw=.85, zorder=6))
        text(67, 61, "+", 12, ha="center")
        arrow((68.3, 61), (76.8, 61))
        text(72.3, 63.4, "Output", 6.7, ha="center")
        text(72.3, 58.5, "inverse / sort\n+ floors", 6, MUTED, ha="center", linespacing=1.15)
        p = plot(79, 56.8, 19, 8.8)
        h = np.arange(24)
        p.fill_between(h, data["q05"], data["q95"], color=BLUE, alpha=.17, lw=0)
        p.plot(h, data["actual"], color="#939ba2", lw=.7)
        p.plot(h, data["point"], color=BLUE, lw=1.05)
        text(79, 67.1, "Price forecast", 7.8, weight="bold")
        text(79, 55.5, r"$t$", 6.5, MUTED)
        text(98, 55.5, r"$t+23$", 6.5, MUTED, ha="right")
        text(79, 53.3, "Point + 90% interval", 6.3, BLUE)
        text(98, 51.3, "Gray: observed price", 6, MUTED, ha="right")
        ax.plot([2, 98], [50, 50], color=LINE, lw=.65)

        # CTF: data and continuous-time features are independent encoder inputs.
        section(47.5, "b", "Continuous trend field", "CTF  /  full history, not event-subtracted", BLUE)
        text(2, 43.9, "Low-pass history", 7.2, weight="bold")
        p = plot(2, 36.7, 18, 6)
        p.plot(hours, data["history"], color="#c8ced3", lw=.65)
        p.plot(hours, data["lowpass"], color=BLUE, lw=1.1)
        text(11, 35.5, r"$\widetilde{x}(\tau)$", 8, BLUE, ha="center")
        arrow((21, 39.8), (26, 39.8), BLUE)
        box(26, 37.7, 15, 4.2, "Data encoder", BLUE, PALE_BLUE)
        text(2, 32.7, "Fourier time features", 7.2, weight="bold")
        p = plot(2, 26.8, 18, 4.7)
        tau = np.linspace(0, 1, 160)
        for i, idx in enumerate((0, 7, 15)):
            p.plot(tau, .3*np.sin(2*np.pi*data["frequencies"][idx]*tau)+i,
                   color=BLUE, lw=.75, alpha=.55+.2*i)
        text(21, 29.3, r"$\gamma(\tau)$", 8, BLUE)
        arrow((21, 31.2), (26, 31.2), BLUE)
        box(26, 29.1, 15, 4.2, "Time encoder", BLUE, PALE_BLUE)
        route([(41, 39.8), (46, 39.8), (46, 36.5)], BLUE)
        route([(41, 31.2), (46, 31.2), (46, 34.5)], BLUE)
        box(43, 34.5, 6, 2, "concat", MUTED, "white", 5.8, LINE)
        arrow((49, 35.5), (53, 35.5), BLUE)
        box(53, 32.6, 12, 5.8, "Fusion\nMLP", BLUE, PALE_BLUE, 7.4)
        arrow((65, 35.5), (69, 35.5), BLUE)
        p = plot(70, 32, 11, 7)
        p.plot(hours, data["trend"], color=BLUE, lw=1.2)
        text(75.5, 40.7, r"Trend $c(\tau)$", 7.3, BLUE, ha="center")
        arrow((82, 35.5), (86, 35.5), BLUE)
        box(86, 33.3, 12, 4.4, "Trend head", BLUE, PALE_BLUE, 6.8)
        text(92, 40.6, r"$C_h$", 10, BLUE, ha="center")
        arrow((92, 37.7), (92, 39.2), BLUE)
        text(92, 28.8, r"$r,\ z^{\mathrm{C}}$", 8, TEAL, ha="center")
        arrow((92, 30.3), (92, 33.3), TEAL)
        text(53, 28, r"$c(\tau)=f_\theta(\widetilde{x}(\tau),\gamma(\tau))$", 8, BLUE)
        ax.plot([2, 98], [24.5, 24.5], color=LINE, lw=.65)

        # DGF: exact detector centres and weighted atoms from the saved model.
        section(22, "c", "Detected event field", "DGF  /  price channel only", CORAL)
        for x, label in [(2, "1  Detect departures"), (31, "2  Form Gaussian atoms"), (61, "3  Sum events")]:
            text(x, 18.4, label, 7.2, weight="bold")
        p = plot(2, 9.7, 21, 7.2)
        hist = data["history"]
        median = float(np.sort(hist)[(len(hist)-1)//2])
        p.plot(hours, hist, color="#a7afb6", lw=.75)
        p.axhline(median, color=MUTED, lw=.6, ls=(0, (3, 2)))
        centres = data["centres"].astype(int)
        p.scatter(hours[centres], hist[centres], s=8, color=CORAL, zorder=4)
        for centre in centres:
            p.plot([hours[centre]]*2, [median, hist[centre]], color=CORAL, lw=.65)
        text(2, 8.5, "Top 8 deviations · 3 h separation", 6.4, MUTED)
        text(2, 6.8, "Dashed: window median", 6, MUTED)
        arrow((24, 13), (29, 13), CORAL)
        p = plot(31, 9.7, 23, 7.2)
        for atom in data["atoms"]:
            p.plot(hours, atom, color=CORAL, lw=.85, alpha=.7)
        p.axhline(0, color=LINE, lw=.5)
        text(31, 8.5, r"Learned threshold $\kappa$ and width $w$", 6.4, MUTED)
        arrow((55, 13), (59, 13), CORAL)
        p = plot(61, 9.7, 13, 7.2)
        p.plot(hours, data["events"], color=CORAL, lw=1.15)
        p.axhline(0, color=LINE, lw=.5)
        text(67.5, 8.5, r"$e=\sum_k a_kG_k$", 8, CORAL, ha="center")
        arrow((75, 13), (79, 13), CORAL)
        box(79, 10.9, 11, 4.2, "Event head", CORAL, PALE_CORAL, 6.8)
        text(84.5, 7.6, r"$z^{\mathrm{D}}$", 8, TEAL, ha="center")
        arrow((84.5, 9), (84.5, 10.9), TEAL)
        arrow((90, 13), (93.4, 13), CORAL)
        ax.add_patch(Circle((94.5, 13), 1.1, facecolor="white", edgecolor=CORAL, lw=.8, zorder=6))
        text(94.5, 13, r"$\times$", 10, CORAL, ha="center")
        text(94.5, 17.2, r"$g_h=\sin^2\theta_h$", 7.4, CORAL, ha="center")
        arrow((94.5, 15.7), (94.5, 14.1), CORAL)
        text(98, 8.2, r"$g_hD_h$", 8, CORAL, ha="right")
        route([(95.6, 13), (98, 13), (98, 10.2)], CORAL)
        ax.plot([2, 98], [5, 5], color=LINE, lw=.65)
        text(2, 2.8, r"Shared reconstruction: $x=c+e+r$", 7)
        text(43, 2.8, r"$r$ also enters the trend head", 6.5, MUTED)
        text(98, 2.8, "QLD1 · 1 Oct 2023 · seed 2026", 6.4, MUTED, ha="right")
        return fig


def save_framework(directory=None, refresh=False):
    directory = Path(directory) if directory is not None else OUT
    directory.mkdir(parents=True, exist_ok=True)
    data = load_case(refresh=refresh)
    fig = draw(data)
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(directory / f"framework_hierarchy.{suffix}", dpi=300,
                        facecolor="white", metadata={"Creator": "DualTimesField vector figure builder"})
    plt.close(fig)
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-extract thumbnails from the checkpoint")
    save_framework(refresh=parser.parse_args().refresh)
