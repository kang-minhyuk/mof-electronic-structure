import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Poster fonts (drop-in) ---
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

# ---- load ----
df = pd.read_csv("df_with_labels.csv")
COL_HSE, COL_PBE, COL_LABEL = "EgHSE06", "EgPBE", "label"

eg_hse = pd.to_numeric(df[COL_HSE], errors="coerce")
eg_pbe = pd.to_numeric(df[COL_PBE], errors="coerce")

mask_calc = (df[COL_LABEL] == 0) & eg_hse.notna()   # newly calculated
mask_qmof = (df[COL_LABEL] == 1) & eg_hse.notna()   # existing QMOF

eg_hse_calc = eg_hse[mask_calc].dropna()
eg_hse_qmof = eg_hse[mask_qmof].dropna()
eg_hse_all  = eg_hse.dropna()
eg_pbe_all  = eg_pbe.dropna()

# ---- common bins ----
emax = float(np.nanmax([eg_hse_all.max(), eg_pbe_all.max()]))
bins = np.linspace(0, np.ceil(emax), 40)

# ---- counts for shading (HSE06_all minus HSE06_QMOF) ----
hse_all_cnt, edges = np.histogram(eg_hse_all, bins=bins)
hse_qmof_cnt, _    = np.histogram(eg_hse_qmof, bins=edges)

# area to shade = newly calculated = label 0
hse_calc_cnt = np.clip(hse_all_cnt - hse_qmof_cnt, 0, None)

# ---- plot ----
BLUE = "#2D53C5"   # HSE06
RED  = "#E74C3C"   # QMOF outline (subset)
ORNG = "#F4A261"   # PBE
SHADE= "#31A05F"   # newly calculated fill
BLACK = "#000000"
GREY = "#7F8C8D"

fig, ax = plt.subplots(figsize=(10, 6))

# bases
ax.hist(eg_pbe_all, bins=edges, color=GREY, alpha=0.50, label=f"PBE")
ax.hist(eg_hse_all, bins=edges, color=BLUE, alpha=0.70, label=f"HSE06 (qmof)")

# QMOF subset as dashed step (outline only)
ax.hist(eg_hse_qmof, bins=edges, histtype="step", linewidth=1, linestyle="--", color = BLACK)
ax.hist(eg_hse_all, bins=edges, histtype="step", linewidth=1, linestyle="-", color = BLACK)

# shade the *difference* between HSE06(all) and QMOF subset = newly calculated
# convert binned counts to step-shaped area
x = edges
y1 = hse_qmof_cnt
y2 = hse_qmof_cnt + hse_calc_cnt   # equals hse_all_cnt
ax.fill_between(x, np.r_[y1, y1[-1]], np.r_[y2, y2[-1]],
                step="post", alpha=0.5, color=SHADE,
                label=f"HSE06 (calculated)")

# cosmetics
ax.set_title("Band-gap Distributions (HSE06 vs PBE)", pad=8)
ax.set_xlabel("Band gap  $E_g$  (eV)")
ax.set_ylabel("Count")
ax.set_xlim(0, edges[-1])
ax.legend(frameon=False)
ax.tick_params(labelsize=12)
plt.tight_layout()

plt.savefig("F2_bandgap_distribution_shaded.pdf")   # vector for poster
plt.savefig("F2_bandgap_distribution_shaded.png", dpi=300)
plt.show()