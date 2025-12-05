import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from math import ceil

# ---------- Poster styling ----------
from matplotlib import rcParams
rcParams["figure.facecolor"] = "none"
rcParams["axes.facecolor"] = "none"
rcParams["savefig.transparent"] = True
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]
rcParams["mathtext.fontset"] = "dejavusans"  # matches sans-serif look

# Sizes tuned for posters (adjust if needed)
rcParams["font.size"] = 16            # base
rcParams["axes.titlesize"] = 22
rcParams["axes.labelsize"] = 18
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["legend.fontsize"] = 14
rcParams["figure.titlesize"] = 22

# Palette (consistent)
BLUE = "#4169E1"   # main points
RED  = "#E74C3C"   # low-gap highlight
INK  = "#111827"   # lines/text

LOWGAP_TH = 1.5  # eV

# ---------- Load ----------
df = pd.read_csv("/Users/minhyukkang/VScode/results/cgcnn/test_results_default_2.csv",
                 header=None, names=["bandgap", "predicted_bandgap"]).dropna()

y_true = df["bandgap"].values
y_pred = df["predicted_bandgap"].values

# Masks
m_low  = y_true < LOWGAP_TH
m_high = ~m_low

# Metrics
r2      = r2_score(y_true, y_pred)
mae     = mean_absolute_error(y_true, y_pred)
mae_low = mean_absolute_error(y_true[m_low], y_pred[m_low]) if m_low.any() else np.nan

# Common axes
xmin = 0.0
xmax = float(np.nanmax([y_true.max(), y_pred.max()]))
xmax = ceil(xmax * 1.02)  # headroom

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(10, 6))

# High-gap points
ax.scatter(y_true[m_high], y_pred[m_high],
           s=18, alpha=0.35, color=BLUE, edgecolors="none",
           label=f"≥ {LOWGAP_TH:.1f} eV  (N={m_high.sum()})")

# Low-gap points
ax.scatter(y_true[m_low], y_pred[m_low],
           s=28, alpha=0.55, color=RED, edgecolors="#000000", linewidths=0.3,
           label=f"< {LOWGAP_TH:.1f} eV  (N={m_low.sum()})")

# y = x line
ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", color=INK, linewidth=1.2, label="Ideal  y = x")

# Axes & labels
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
ax.set_aspect('auto')
ax.set_xlabel(r"Calculated Band Gap  $E_g$  (eV)")
ax.set_ylabel(r"Predicted Band Gap  $\hat{E}_g$  (eV)")
ax.set_title("CGCNN Band Gap Prediction")

# Metrics box
txt = f"R² = {r2:.3f}\nMAE = {mae:.3f} eV"
if np.isfinite(mae_low):
    txt += f"\nMAE (<{LOWGAP_TH:.1f} eV) = {mae_low:.3f} eV"
ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, lw=0.5))

plt.tight_layout()

# Saves (vector + 300 dpi raster)
plt.savefig("parity_cgcnn_default_2.pdf")
plt.savefig("parity_cgcnn_default_2.png", dpi=300)
plt.show()