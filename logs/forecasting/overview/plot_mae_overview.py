"""Bar chart of the three-region mean test MAE of every variant (raw prices, quarterly refits, predispatch)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O, B, V, G = "#eb6834", "#2a78d6", "#6250d6", "#898781"
ROWS = [
    ("XGBoost + predispatch-error history (probe, seed 2026)", 35.46, O),
    ("XGBoost (baseline)", 35.52, O),
    ("XGBoost point, fields correct quantiles", 35.53, O),
    ("C-mae: base + calibrator + floor, no fields", 36.08, B),
    ("Base + driver channels, no fields", 36.12, B),
    ("Same as C-mae, total-loss checkpoint", 36.34, B),
    ("Final model, MAE checkpoint", 36.36, V),
    ("Final model (manuscript)", 36.64, V),
    ("Driver space, same inputs without fields", 36.72, V),
    ("Driver space, event field", 36.84, V),
    ("XGBoost + raw-history residual", 37.06, O),
    ("XGBoost + fields residual", 37.48, O),
    ("Linear base only", 37.60, G),
    ("Linear", 37.92, G),
    ("DLinear", 38.20, G),
    ("PatchTST", 38.85, G),
]
fig, ax = plt.subplots(figsize=(9, 6.2), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
y = range(len(ROWS))[::-1]
ax.barh(list(y), [r[1] - 34 for r in ROWS], left=34, color=[r[2] for r in ROWS], height=0.62)
for yi, (_, value, _) in zip(y, ROWS):
    ax.text(value + 0.05, yi, f"{value:.2f}", va="center", fontsize=9, color="#52514e")
ax.set_yticks(list(y))
ax.set_yticklabels([r[0] for r in ROWS], fontsize=9, color="#0b0b0b")
ax.set_xlim(34, 39.4)
ax.set_xlabel("Mean test MAE, AUD/MWh (lower is better; axis starts at 34)", fontsize=9, color="#52514e")
ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (O, B, V, G)]
ax.legend(handles, ["XGBoost family", "Ours, no dual field", "Ours, with dual field", "Linear and neural baselines"],
          loc="upper right", frameon=False, fontsize=9)
ax.set_title("Raw prices, quarterly refits with predispatch, three-region mean", loc="left", fontsize=10, color="#0b0b0b")
fig.tight_layout()
fig.savefig(Path(__file__).with_name("mae_overview.png"), dpi=160, facecolor="#fcfcfb")
