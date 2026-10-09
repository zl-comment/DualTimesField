"""Plot the per-target-hour error series written by ``forecasting.error_profile``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIONS = ("NSW1", "QLD1", "TAS1")
COLORS = ("#2a78d6", "#eb6834")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = np.load(args.input / "error_profile_series.npz")
    names = sorted({key.split("|")[1] for key in data.files}, key=lambda n: (not n.startswith("C"), n))
    fig, axes = plt.subplots(3, 2, figsize=(13, 9.5), facecolor=SURFACE, gridspec_kw={"width_ratios": [3.2, 1]})
    for row, region in enumerate(REGIONS):
        ax_day, ax_hour = axes[row]
        style(ax_day); style(ax_hour)
        peaks = None
        for color, name in zip(COLORS, names):
            time, error, price = (data[f"{region}|{name}|{k}"] for k in ("time", "error", "price"))
            stamp = pd.to_datetime(time + 10 * 3600, unit="s")
            frame = pd.DataFrame({"error": error, "price": price}, index=stamp)
            daily = frame.error.resample("D").mean()
            ax_day.plot(daily.index, daily.values, color=color, linewidth=1.2, label=name)
            by_hour = frame.groupby(frame.index.hour).error.mean()
            ax_hour.plot(by_hour.index, by_hour.values, color=color, linewidth=1.8, marker="o", markersize=3.5)
            if peaks is None:
                top = daily.nlargest(3)
                peaks = top
                maxprice = frame.price.resample("D").max()
                for day, value in top.items():
                    ax_day.annotate(f"{day:%d %b %y}\nmax price {maxprice[day]:,.0f}", (day, value), xytext=(6, -2),
                                    textcoords="offset points", fontsize=8, color=INK, va="top")
        ax_day.set_ylabel(f"{region}\nmean abs. error per day (AUD/MWh)", color=INK, fontsize=9)
        ax_hour.set_ylabel("mean abs. error by hour of day", color=INK, fontsize=9)
        if row == 0:
            ax_day.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
            ax_day.set_title("Daily mean of the per-target-hour absolute error, 2023-2024", loc="left", color=INK, fontsize=10.5)
            ax_hour.set_title("By hour of day (AEST)", loc="left", color=INK, fontsize=10.5)
        if row == 2:
            ax_hour.set_xlabel("hour of day", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160, facecolor=SURFACE)
    print(args.output)


if __name__ == "__main__":
    main()
