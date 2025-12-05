import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------- Poster styling ----------------
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]
rcParams["mathtext.fontset"] = "dejavusans"  # matches sans-serif look
rcParams["font.size"] = 16            # base
rcParams["axes.titlesize"] = 22
rcParams["axes.labelsize"] = 18
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["legend.fontsize"] = 14
rcParams["figure.titlesize"] = 22

INK  = "#111827"
BLUE = "#4169E1"   # R^2 line (use "#5A6FD3" if you want the Zn blue)
RED  = "#E74C3C"   # MAE line
GRID = "#E5E7EB"

# ---------------- Data ----------------
df = pd.DataFrame({
    "top_features": [5,10,15,20,25,30,35,40,45,50],
    "MAE_avg": [0.675454,0.618046,0.575659,0.564437,0.555306,0.542463,0.539533,0.538908,0.536589,0.534742],
    "R2_avg":  [0.383469,0.467108,0.527647,0.543122,0.551695,0.571460,0.574307,0.574516,0.577989,0.580038],
})

# ---------------- Plot ----------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# X ticks: exactly the tested counts
ax1.set_xlabel("Number of top descriptors")
ax1.set_xticks(df["top_features"])

# MAE (left Y)
l1, = ax1.plot(df["top_features"], df["MAE_avg"],
               color=RED, marker="o", markersize=6, linewidth=2.0, alpha=0.9,
               label="MAE")
ax1.set_ylabel("MAE (eV)", color=INK)

# Nicely bounded left axis
mae_min, mae_max = df["MAE_avg"].min(), df["MAE_avg"].max()
pad = 0.02
ax1.set_ylim(mae_min - pad, mae_max + pad)
ax1.grid(axis="y", linestyle=":", color=GRID)

# R^2 (right Y)
ax2 = ax1.twinx()
l2, = ax2.plot(df["top_features"], df["R2_avg"],
               color=BLUE, marker="s", markersize=6, linewidth=2.0, alpha=0.9,
               label=r"$R^2$")
ax2.set_ylabel(r"$R^2$", color=INK)
r2_min, r2_max = df["R2_avg"].min(), df["R2_avg"].max()
ax2.set_ylim(r2_min - 0.02, r2_max + 0.02)

# Vertical marker at 30 features
ax1.axvline(x=30, color="#5E6D80", linestyle="--", linewidth=1.0)
ax1.text(30, ax1.get_ylim()[1]-0.01, "  selected = 30", va="bottom", ha="left", color="#5E6D80")

# Title & legend
ax1.set_title("Model performance vs. number of top descriptors")
fig.legend(handles=[l1, l2], loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)

# Clean frame
for sp in ["top", "right"]:
    ax1.spines[sp].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("F_perf_vs_top_features.pdf")
plt.savefig("F_perf_vs_top_features.png", dpi=300)
plt.show()