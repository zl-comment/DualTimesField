"""Editable vector overview; no illustrative traces or checkpoint dependency.

Run ``.venv/bin/python paper/draw_framework.py`` from the repository root.
The same function is called by ``paper/make_materials.py``.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def draw_framework():
    ink, muted, line = "#172435", "#536273", "#aeb8c4"
    blue, orange, green = "#356da8", "#ba6846", "#418375"
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(10.6, 5.05))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        fig.patch.set_facecolor("white")
        ax.set(xlim=(0, 12.4), ylim=(0, 5.9))
        ax.axis("off")

        def label(x, y, s, size=9, color=ink, weight="normal", ha="left", **kwargs):
            return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                           ha=ha, va="center", **kwargs)

        def box(x, y, w, h, edge=line, face="white", lw=0.9):
            patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                  linewidth=lw, edgecolor=edge, facecolor=face, zorder=3)
            ax.add_patch(patch)
            return patch

        def arrow(a, b, color=muted, style="-", lw=1.0):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                        lw=lw, color=color, linestyle=style,
                                        shrinkA=0, shrinkB=0, zorder=2))

        # Stage labels and generous whitespace replace the old enclosing panels.
        for x, n, title in [(0.18, "01", "INFORMATION AT ORIGIN"),
                            (3.13, "02", "ADDITIVE FORECASTER"),
                            (9.14, "03", "PRICE FORECASTS")]:
            label(x, 5.62, n, 9, blue, "bold")
            label(x + 0.36, 5.62, title, 8.8, ink, "bold")

        # Inputs: history is available only through t-1.
        box(0.18, 3.36, 2.23, 1.12)
        label(0.36, 4.19, "Observed history", 10, weight="bold")
        label(0.36, 3.86, r"$x_{t-72:t-1}$", 12)
        label(0.36, 3.57, "72 h · price + demand", 8, muted)
        box(0.18, 1.39, 2.23, 1.54)
        label(0.36, 2.64, "Known information", 10, weight="bold")
        label(0.36, 2.31, "Predispatch · PD PASA", 8.3, muted)
        label(0.36, 2.01, "Calendar · gas context", 8.3, muted)
        label(0.36, 1.65, r"Published by $t$", 8.5)

        # An information bus feeds the applicable inputs of each branch.
        ax.plot([2.75, 2.75], [1.365, 4.88], lw=1, color=line, zorder=1)
        for y in (3.92, 2.15):
            ax.plot([2.41, 2.75], [y, y], lw=1, color=line, zorder=1)
            ax.add_patch(Circle((2.75, y), 0.026, color=line, zorder=2))

        # Four additive paths. Each dual-field row shows field -> forecast head.
        box(3.13, 4.48, 4.62, 0.78, edge=green, face="#f5faf8")
        label(3.32, 4.96, "Linear base", 10, green, "bold")
        label(3.32, 4.66, "Flattened history + known inputs", 8.2, muted)
        label(7.5, 4.86, r"$B_h$", 13, green, ha="right")

        box(3.13, 3.30, 2.46, 0.88, edge=blue, face="#f3f7fc")
        label(3.32, 3.94, "Trend field  CTF", 9.6, blue, "bold")
        label(3.32, 3.64, "Low-pass history + Fourier INR", 8.1, muted)
        box(6.04, 3.30, 1.71, 0.88, edge=blue, face="#f3f7fc")
        label(6.895, 3.93, "Trend head", 9.3, blue, "bold", ha="center")
        label(6.895, 3.60, r"$C_h(c,r,z^{\rm C})$", 10.1, ha="center")
        arrow((5.59, 3.74), (6.04, 3.74), blue)

        box(3.13, 2.12, 2.46, 0.88, edge=orange, face="#fcf6f2")
        label(3.32, 2.76, "Event field  DGF", 9.6, orange, "bold")
        label(3.32, 2.46, "Detected Gaussian events", 8.1, muted)
        box(6.04, 2.12, 1.71, 0.88, edge=orange, face="#fcf6f2")
        label(6.895, 2.76, "Gated event head", 8.9, orange, "bold", ha="center")
        label(6.895, 2.43, r"$\sin^2\!\theta_h\,D_h(e,z^{\rm D})$", 9.6, ha="center")
        arrow((5.59, 2.56), (6.04, 2.56), orange)

        box(3.13, 0.97, 4.62, 0.79, edge=green, face="#f5faf8")
        label(3.32, 1.46, "Shared per-hour calibrator", 9.8, green, "bold")
        label(3.32, 1.16, "Known inputs + horizon embedding + recent price level", 7.9, muted)
        label(7.5, 1.43, r"$\delta_h$", 13, green, ha="right")

        for y in (4.87, 3.74, 2.56, 1.365):
            arrow((2.75, y), (3.13, y), line)
        # The field remainder and decomposition objective are explicit but not
        # drawn as extra crossing edges through the forecast branches.
        label(5.45, 1.91, r"$r=x-c-e$     ·     decomposition loss on $c+e$", 8.1, muted, ha="center")

        # Distinct lanes join at a single sum, keeping edge crossings out of boxes.
        sum_x, sum_y = 8.49, 3.18
        for y, color in [(4.87, green), (3.74, blue), (2.56, orange), (1.365, green)]:
            arrow((7.75, y), (8.35, sum_y + (0.07 if y > sum_y else -0.07)), color)
        ax.add_patch(Circle((sum_x, sum_y), 0.18, facecolor="white", edgecolor=ink, lw=1.1, zorder=4))
        label(sum_x, sum_y + 0.01, "+", 17, ha="center", zorder=5)
        arrow((8.67, sum_y), (9.14, sum_y), ink)

        box(9.14, 1.63, 3.00, 3.20, edge=line)
        label(9.34, 4.48, "Point + five quantiles", 10, weight="bold")
        label(9.34, 4.04, "Point: inverse asinh", 9, muted)
        label(9.34, 3.68, "Quantiles: price space + sorting", 8.3, muted)
        ax.plot([9.35, 11.93], [3.43, 3.43], color="#dce2e8", lw=0.8, zorder=5)
        label(9.34, 3.09, "Validation-selected bounds", 9.0, weight="bold")
        label(9.34, 2.70, r"$\widehat p_h\leftarrow\max(\widehat p_h,\phi)$", 12)
        label(9.34, 2.27, "Separate point / quantile bounds", 8.0, muted)
        label(9.34, 1.91, r"24 hours: $p_t,\ldots,p_{t+23}$", 9.3)

        # A compact time strip gives the evaluation protocol without a fifth panel.
        ax.plot([0.18, 12.14], [0.72, 0.72], color="#dce2e8", lw=0.8)
        label(0.18, 0.40, "QUARTERLY REFIT", 8.5, weight="bold")
        for x, w, text, color, face in [
            (3.13, 4.20, "Expanding train window from 2015", blue, "#f3f7fc"),
            (7.44, 2.18, "Previous quarter: validate", orange, "#fcf6f2"),
            (9.73, 2.08, r"Quarter $k$: test", green, "#f5faf8"),
        ]:
            box(x, 0.18, w, 0.43, edge=color, face=face, lw=0.7)
            label(x + w / 2, 0.395, text, 7.8, ha="center")
        arrow((11.91, 0.395), (12.14, 0.395), muted)
        return fig


def save_framework(directory=None):
    """Use identical export settings in standalone and full-materials builds."""
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    figure = draw_framework()
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for extension in ("pdf", "svg", "png"):
            figure.savefig(directory / f"framework.{extension}", dpi=300,
                           bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


if __name__ == "__main__":
    save_framework()
