"""Original DualTimesField artwork with TimeMixer++ Figure 2's visual grammar.

Central bottom-to-top architecture, four mechanism callouts, pastel modules,
rounded frames, smooth routing and dashed detail links. No reference pixels are
reused. All signal thumbnails use the existing, tested QLD1 checkpoint cache.
Style reference: https://arxiv.org/html/2410.16032v5#S3.F2
Inter is bundled under its SIL Open Font License in paper/assets/fonts.
Run: PYTHONPATH=. .venv/bin/python paper/draw_framework_timemixer.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
import numpy as np

from paper.draw_framework_hierarchy import load_case

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
INK, GRAY = "#283342", "#77828c"
BLUE, CORAL, GREEN, GOLD = "#5689ad", "#c5817d", "#6f9e8d", "#b39a58"
PALE_BLUE, PALE_CORAL = "#e8f2f9", "#f9e9e6"
PALE_GREEN, PALE_GOLD = "#eaf4ed", "#faf3da"


def draw(data):
    for path in (ROOT / "paper/assets/fonts").glob("Inter-*.ttf"):
        font_manager.fontManager.addfont(str(path))
    with plt.rc_context({"font.family": "Inter", "font.size": 7.4,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(8.4, 6.05), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, 100), ylim=(0, 72))
        ax.axis("off")
        labels = []

        def text(x, y, s, size=7.4, color=INK, ha="center", weight="normal", **kw):
            t = ax.text(x, y, s, ha=ha, va="center", fontsize=size,
                        color=color, fontweight=600 if weight == "bold" else weight,
                        zorder=20, **kw)
            labels.append(t)
            return t

        def box(x, y, w, h, label=None, fill="white", edge=GRAY, radius=1.1,
                lw=.8, size=7.3, dashed=False, zorder=4):
            p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge, linewidth=lw,
                              linestyle=(0, (3, 2.5)) if dashed else "-", zorder=zorder)
            ax.add_patch(p)
            if label:
                text(x+w/2, y+h/2, label, size)
            return p

        def panel(x, y, w, h, title, color):
            # A quiet offset edge echoes the reference's layered frames.
            box(x+.35, y-.3, w, h, fill="#f0f2f4", edge="none", radius=1.8, zorder=0)
            box(x, y, w, h, edge=color, radius=1.8, lw=1.15, zorder=1)
            text(x+1.6, y+h-2.3, title, 8.6, INK, ha="left", weight="bold")

        def curve(a, b, color=INK, c1=None, c2=None, dashed=False,
                  head=True, lw=1, zorder=3):
            if c1 is None:
                c1 = (a[0], (a[1]+b[1])/2)
                c2 = (b[0], (a[1]+b[1])/2)
            path = MPath([a, c1, c2, b], [MPath.MOVETO, MPath.CURVE4,
                                         MPath.CURVE4, MPath.CURVE4])
            if head:
                p = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=7.8,
                                    linewidth=lw, color=color, zorder=zorder,
                                    linestyle=(0, (3, 3)) if dashed else "-")
            else:
                p = PathPatch(path, fill=False, color=color, linewidth=lw,
                              linestyle=(0, (3, 3)) if dashed else "-", zorder=zorder)
            ax.add_patch(p)
            return p

        def node(x, y, s, color=INK, radius=1):
            ax.add_patch(Circle((x, y), radius, facecolor="white", edgecolor=color,
                                linewidth=1, zorder=8))
            text(x, y, s, 10, color)

        def plot(x, y, w, h):
            p = fig.add_axes((x/100, y/72, w/100, h/72), facecolor="none", zorder=5)
            p.axis("off")
            p.margins(x=.025, y=.12)
            return p

        def trace(x, y, w, h, values, color, lw=1.05):
            p = plot(x, y, w, h)
            p.plot(values, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round")
            return p

        # Four satellites and one continuous central spine, as in the reference.
        panel(1.5, 2.6, 32.5, 20.2, "(a)  Inputs at the forecast origin", GREEN)
        panel(1.5, 26, 32.5, 43.5, "(b)  Continuous trend field", BLUE)
        panel(66, 31.2, 32.5, 38.3, "(c)  Detected event field", CORAL)
        panel(66, 2.6, 32.5, 25.4, "(d)  Probabilistic forecast", GOLD)

        # (a) Distinct price/demand channels, not fictitious multi-scale data.
        text(10, 17.9, "Historical signals", 7.1)
        box(4.5, 12.2, 11, 3.7, fill="#f5f8fa", edge="#d6e0e6", radius=.7, lw=.5)
        trace(5, 12.7, 10, 2.7, data["history"], BLUE, .75)
        box(5.4, 7.7, 11, 3.7, fill="#f4f8f5", edge="#d6e0da", radius=.7, lw=.5)
        trace(5.9, 8.2, 10, 2.7, data["demand"], GREEN, .75)
        text(25, 17.9, "Known covariates", 7.1)
        for y, label in [(13.5, "Predispatch"), (9.8, "PD PASA"), (6.1, "Calendar · Gas")]:
            box(18.2, y, 13.3, 2.9, label, PALE_GREEN, "#a5bfb2", radius=1.25, size=6.8)
        text(10.5, 5.4, "Price · Demand", 6.5, GRAY)

        # (b) Full-history lowpass and Fourier features are parallel inputs.
        text(10, 62.6, "Low-pass history", 7.5, BLUE)
        text(25.7, 62.6, "Fourier features", 7.5, BLUE)
        p = trace(4.2, 54.8, 12.2, 6, data["history"], "#ccd4db", .65)
        p.plot(data["lowpass"], color=BLUE, lw=1.15, solid_capstyle="round")
        p = plot(19.2, 54.8, 12.5, 6)
        tau = np.linspace(0, 1, 200)
        for j, index in enumerate((0, 7, 15)):
            p.plot(tau, .29*np.sin(2*np.pi*data["frequencies"][index]*tau)+j,
                   color=BLUE, alpha=.55+.2*j, lw=.85, solid_capstyle="round")
        text(10.3, 53.9, r"$\widetilde{x}(\tau)$", 8.6, BLUE)
        text(25.6, 53.9, r"$\gamma(\tau)$", 8.6, BLUE)
        curve((10.3, 52.8), (10.3, 51.3), BLUE)
        curve((25.6, 52.8), (25.6, 51.3), BLUE)
        box(4, 47.2, 12.6, 4.1, "Data encoder", PALE_BLUE, BLUE, radius=1.7)
        box(19.3, 47.2, 12.6, 4.1, "Time encoder", PALE_BLUE, BLUE, radius=1.7)
        curve((10.3, 47.2), (16.8, 43.3), BLUE,
              c1=(10.3, 44), c2=(12.5, 43.3))
        curve((25.6, 47.2), (19.2, 43.3), BLUE,
              c1=(25.6, 44), c2=(23, 43.3))
        node(18, 43.3, r"$\Vert$", BLUE, .95)
        text(22, 42.1, "concat", 6.5, GRAY)
        curve((18, 42.35), (18, 40.5), BLUE)
        box(10, 36.3, 16, 4.2, "Fusion MLP", PALE_BLUE, BLUE, radius=1.8)
        curve((18, 36.3), (18, 34.7), BLUE)
        trace(9.3, 29.8, 17.4, 4.7, data["trend"], BLUE, 1.25)
        text(18, 28.1, r"$c(\tau)=f_\theta(\widetilde{x}(\tau),\gamma(\tau))$", 8, BLUE)

        # (c) The actual detector and learned Gaussian atoms.
        text(82.3, 62.7, "Median-relative departures", 7.5, CORAL)
        p = trace(69.3, 54.2, 26, 7, data["history"], "#b9c1c7", .8)
        median = np.sort(data["history"])[35]
        centres = data["centres"].astype(int)
        p.axhline(median, color=GRAY, lw=.6, ls=(0, (3, 2)))
        p.vlines(centres, median, data["history"][centres], color=CORAL, lw=.8)
        p.scatter(centres, data["history"][centres], color=CORAL, s=7, zorder=6)
        text(82.3, 52.6, "Top 8 events · 3 h separation", 6.9, GRAY)
        curve((82.3, 51.3), (82.3, 49.5), CORAL)
        box(71.8, 45.4, 21, 4.1, "Soft-threshold amplitudes", PALE_CORAL, CORAL,
            radius=1.7, size=7.2)
        curve((82.3, 45.4), (82.3, 43.6), CORAL)
        p = plot(69.8, 35.9, 25, 7.6)
        p.axhline(0, color="#e7ddd9", lw=.5)
        # Evaluate the learned analytic kernels between hourly samples. This
        # prevents narrow Gaussians looking like triangles; no data are altered.
        fine_time = np.linspace(0, 71, 1600)
        amplitudes = data["atoms"][np.arange(len(centres)), centres]
        fine_atoms = amplitudes[:, None] * np.exp(
            -(fine_time[None]-centres[:, None])**2 / (2*data["width"]**2))
        for atom in fine_atoms:
            p.plot(fine_time, atom, color=CORAL, lw=.95, alpha=.75, solid_capstyle="round")
        text(82.3, 34.1, r"$e(\tau)=\sum_k a_kG_k(\tau)$", 9, CORAL)

        # Center: a genuine parallel dual-field architecture, not MixerBlocks.
        text(50, 3.3, "Input", 9.2, weight="bold")
        curve((50, 4.9), (50, 8), INK)
        box(39.8, 8, 20.4, 4.4, r"History $x$ + context $z$", PALE_GREEN, GREEN,
            radius=1.8, size=7.7)
        curve((50, 12.4), (50, 16.5), INK)
        box(37.8, 18.2, 24.4, 32.1, fill="#fafbfd", edge="#8996a2", radius=2.3, lw=1.2, zorder=1)
        text(38.2, 52.1, "Dual-field model", 6.9, ha="left", weight="bold")
        curve((50, 16.5), (44.5, 24.2), BLUE, c1=(50, 21.3), c2=(44.5, 20.5))
        curve((50, 16.5), (55.5, 24.2), CORAL, c1=(50, 21.3), c2=(55.5, 20.5))
        box(39.9, 24.2, 9.2, 4, "CTF", PALE_BLUE, BLUE, radius=1.6, size=8)
        box(50.9, 24.2, 9.2, 4, "DGF", PALE_CORAL, CORAL, radius=1.6, size=8)
        curve((44.5, 28.2), (44.5, 29.1), BLUE)
        curve((55.5, 28.2), (55.5, 29.1), CORAL)
        trace(40.6, 29.3, 7.8, 3.7, data["trend"], BLUE, .9)
        trace(51.6, 29.3, 7.8, 3.7, fine_atoms.sum(axis=0), CORAL, .9)
        curve((44.5, 33.2), (44.5, 34.6), BLUE)
        curve((55.5, 33.2), (55.5, 34.6), CORAL)
        box(39.8, 34.6, 9.4, 4, "Trend head", PALE_BLUE, BLUE, radius=1.6, size=6.9)
        box(50.8, 34.6, 9.4, 4, "Event head", PALE_CORAL, CORAL, radius=1.6, size=6.9)
        text(42.4, 40.7, r"$C_h$", 9, BLUE)
        text(57.6, 40.7, r"$D_h$", 9, CORAL)
        curve((44.5, 38.6), (46.9, 44), BLUE, c1=(44.5, 43), c2=(45.5, 44))
        curve((55.5, 38.6), (53.1, 44), CORAL, c1=(55.5, 43), c2=(54.5, 44))
        box(40.2, 44, 19.6, 4.6, r"$C_h+\sin^2\theta_h\,D_h$", PALE_GOLD, GOLD,
            radius=1.8, size=9)
        curve((50, 48.6), (50, 54), INK)
        text(51.9, 52.1, r"$F_h$", 9, GOLD)
        node(50, 55.1, "+", INK, 1.1)
        box(37.2, 57.5, 10.9, 4, "Linear base", PALE_GREEN, GREEN, radius=1.7, size=7.1)
        box(51.9, 57.5, 10.9, 4, "Calibrator", "#f0edf8", "#9a8cb5", radius=1.7, size=7.1)
        curve((42.65, 57.5), (48.9, 55.1), GREEN, c1=(42.65, 54), c2=(46.5, 55.1))
        curve((57.35, 57.5), (51.1, 55.1), "#9a8cb5", c1=(57.35, 54), c2=(53.5, 55.1))
        text(40, 55.1, r"$B_h$", 8.2, GREEN)
        text(60, 55.1, r"$\delta_h$", 8.2, "#9a8cb5")
        curve((50, 56.2), (50, 64), INK)
        box(39.8, 64, 20.4, 4.1, "Output mapping", PALE_GOLD, GOLD, radius=1.7, size=7.7)
        curve((50, 68.1), (50, 69.7), INK)
        text(50, 71, "Forecast", 9.2, weight="bold")

        # Dashed, arrowless links mean magnification, never a data dependency.
        curve((34, 12.2), (39.8, 10.2), GREEN, c1=(36, 12.2), c2=(37.5, 10.2),
              dashed=True, head=False, lw=.9)
        curve((34, 41), (39.9, 26.2), BLUE, c1=(36, 41), c2=(35.9, 26.2),
              dashed=True, head=False, lw=.9)
        curve((66, 41), (60.1, 26.2), CORAL, c1=(64, 41), c2=(64.1, 26.2),
              dashed=True, head=False, lw=.9)
        curve((60.2, 66.05), (66, 24), GOLD, c1=(65.3, 66.05), c2=(62.8, 24),
              dashed=True, head=False, lw=.9)

        # (d) Real 24-hour point forecast and calibrated 90% interval.
        text(82.3, 21.2, r"$\widehat{y}_h=B_h+F_h+\delta_h$", 9.4)
        box(69, 15.7, 26.5, 3.6, "Inverse / sort + price floors", PALE_GOLD, GOLD,
            radius=1.5, size=7.1)
        curve((82.3, 15.7), (82.3, 14.4), GOLD)
        p = plot(69.4, 6.2, 25.5, 8)
        steps = np.arange(24)
        p.fill_between(steps, data["q05"], data["q95"], color=BLUE, alpha=.2, linewidth=0)
        p.plot(steps, data["actual"], color="#9aa1a7", lw=.8, solid_capstyle="round")
        p.plot(steps, data["point"], color=BLUE, lw=1.1, solid_capstyle="round")
        text(82.3, 4.6, "Point + 90% interval   ·   Observed", 6.5, GRAY)
        text(2, .9, r"$r=x-c-e$ enters the trend head; heads also read their context inputs.",
             6.7, GRAY, ha="left")
        text(98, .9, "Dashed links: expanded views", 6.5, GRAY, ha="right")
        fig._diagram_labels = labels
        return fig


def save_framework(directory=None):
    directory = Path(directory) if directory is not None else OUT
    directory.mkdir(parents=True, exist_ok=True)
    fig = draw(load_case())
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(directory / f"framework_timemixer.{suffix}", dpi=300,
                        facecolor="white", metadata={"Creator": "DualTimesField vector figure builder"})
    plt.close(fig)


if __name__ == "__main__":
    save_framework()
