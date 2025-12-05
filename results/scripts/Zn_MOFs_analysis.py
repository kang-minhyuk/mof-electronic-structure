import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Poster styling ----------------
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["font.size"] = 14
rcParams["axes.titlesize"] = 18
rcParams["axes.labelsize"] = 16
rcParams["xtick.labelsize"] = 13
rcParams["ytick.labelsize"] = 13
rcParams["legend.fontsize"] = 12
rcParams["figure.facecolor"] = "white"
rcParams["axes.facecolor"] = "white"
rcParams["savefig.transparent"] = False

# Palette (consistent)
INK   = "#111827"   # lines/text
C_ZN  = "#4169E1"   # Zn-oxo (blue)
C_CL  = "#31A05F"   # Metal cluster (green)
C_SM  = "#E74C3C"   # Single metal-centered (red)

# ---------------- Load ----------------
df_Zn = pd.read_csv("/Users/minhyukkang/VScode/results/df_zn.csv")
df_Zn["qmof_id"] = df_Zn["qmof_id"].astype(str)
for col in ["actual_bandgap", "predicted_bandgap"]:
    df_Zn[col] = pd.to_numeric(df_Zn[col], errors="coerce")

zn_mofs = pd.read_csv("/Users/minhyukkang/VScode/results/cgcnn/Zn_MOFs.csv")
zn_mofs["qmof_id"] = zn_mofs["qmof_id"].astype(str)

zn_1 = zn_mofs[zn_mofs["type 1"] == 1]["qmof_id"].values
zn_2 = zn_mofs[zn_mofs["type 2"] == 1]["qmof_id"].values
zn_3 = zn_mofs[zn_mofs["type 3"] == 1]["qmof_id"].values
zn_4 = zn_mofs[zn_mofs["type 4"] == 1]["qmof_id"].values
zn_5 = zn_mofs[zn_mofs["abnormal"] == 1]["qmof_id"].values

# Groups (all circles)
df_br = df_Zn[df_Zn["qmof_id"].isin(np.r_[zn_1, zn_2])].dropna(subset=["actual_bandgap","predicted_bandgap"])  # Zn-oxo
df_cl = df_Zn[df_Zn["qmof_id"].isin(np.r_[zn_3, zn_4])].dropna(subset=["actual_bandgap","predicted_bandgap"])  # Metal cluster
df_sm = df_Zn[df_Zn["qmof_id"].isin(zn_5)].dropna(subset=["actual_bandgap","predicted_bandgap"])               # Single metal-centered

# ---------------- Plot ----------------
LOW_MAX = 1.5
fig, ax = plt.subplots(figsize=(8.8, 6))

# Darker look: higher alpha, a touch larger, slightly thicker white edge
ax.scatter(df_br["actual_bandgap"], df_br["predicted_bandgap"],
           s=36, alpha=0.80, color=C_ZN, marker="o",
           edgecolors="white", linewidths=0.6,
           label=f"Zn-oxo (N={len(df_br)})", zorder=3)

ax.scatter(df_cl["actual_bandgap"], df_cl["predicted_bandgap"],
           s=36, alpha=0.80, color=C_CL, marker="o",
           edgecolors="white", linewidths=0.6,
           label=f"Metal cluster (N={len(df_cl)})", zorder=3)

ax.scatter(df_sm["actual_bandgap"], df_sm["predicted_bandgap"],
           s=40, alpha=0.85, color=C_SM, marker="o",
           edgecolors="white", linewidths=0.6,
           label=f"Single metal-centered (N={len(df_sm)})", zorder=3)

# y = x within window
ax.plot([0, LOW_MAX], [0, LOW_MAX], lw=1.1, color=INK, linestyle="--", zorder=2)

# Axes & labels
ax.set_xlim(0, LOW_MAX)
pred_max = pd.concat(
    [g["predicted_bandgap"] for g in [df_br, df_cl, df_sm] if not g.empty],
    ignore_index=True
).max() if any([not g.empty for g in [df_br, df_cl, df_sm]]) else LOW_MAX
ax.set_ylim(0, max(LOW_MAX, float(pred_max) * 1.05))

ax.set_xlabel(r"Calculated Band Gap  $E_g$  (eV)")
ax.set_ylabel(r"Predicted Band Gap  $\hat{E}_g$  (eV)")
ax.set_title("Zn MOFs — Near-Conductive Region (≤ 1.5 eV)")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.18)
ax.legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig("F_Zn_near_conductive.pdf")
plt.savefig("F_Zn_near_conductive.png", dpi=300)
plt.show()