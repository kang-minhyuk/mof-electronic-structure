import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- Poster styling ----------
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

# Palette (consistent with other figures)
BLUE = "#4169E1"
INK  = "#111827"

# ---------- Load & prepare ----------
df = pd.read_csv("/Users/minhyukkang/VScode/results/grouped_feature_importance.csv")
df = df.sort_values("importance", ascending=False).copy()

# Top-k
TOPK = 10
df_top = df.head(TOPK).copy()

# Optional: pick an axis label to match your metric
IMPORTANCE_LABEL = "Importance (XGBoost gain)"  # change to "mean |SHAP|" if using SHAP

# If you have variability columns, wire them in automatically
err_col = None
for c in ("std", "stderr", "se"):  # auto-detect a reasonable error column name
    if c in df_top.columns:
        err_col = c
        break

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(df_top))
vals = df_top["importance"].values
labels = df_top["group"].astype(str).values
errors = df_top[err_col].values if err_col else None

bars = ax.barh(
    y_pos, vals, xerr=errors, color=BLUE, edgecolor=INK, linewidth=0.6,
    alpha=0.85, capsize=4 if err_col is not None else 0
)

# Highest importance at the top
ax.invert_yaxis()
ax.set_yticks(y_pos, labels)

ax.set_xlabel(IMPORTANCE_LABEL)
ax.set_title("Top Feature Groups — XGBoost")

# Add value labels at the end of bars
for i, b in enumerate(bars):
    x = b.get_width()
    ax.text(
        x + (0.01 * max(vals)), b.get_y() + b.get_height() / 2,
        f"{x:.3f}", va="center", ha="left", color=INK
    )

# Clean look
ax.grid(axis="x", linestyle=":", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("top10_feature_groups.pdf")        # vector for poster
plt.savefig("top10_feature_groups.png", dpi=300)
plt.show()