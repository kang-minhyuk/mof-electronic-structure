# XGBoost Feature-Based Band Gap Prediction

This directory contains the full ML pipeline for feature-driven prediction of MOF band gaps using classical models (primary entry point: `main.py`). It integrates descriptors from Matminer, mofdscribe, and additional MOF structural/porosity feature sources to benchmark classical ML vs CGCNN and to identify physically meaningful descriptors for predicting HSE06 band gaps.

---

## Directory structure

xgb/
├── main.py                          # Primary entry point: orchestrates feature merging, training, evaluation, and SHAP steps
├── xgb_shap_analysis.py             # Optional/legacy SHAP analysis utilities (can be invoked separately)
├── README.md                        # This file
└── results/                         # Output metrics, plots, and analyses

---

## Data requirements

Raw datasets are not included. To run the pipeline, provide precomputed feature tables (recommended paths shown):

- data/matminer_features.csv          (matminer descriptors)
- data/mof_features_combined.csv      (geometric/porosity, custom MOF features)
- data/mofdscribe_results.csv         (mofdscribe descriptors)

Update paths passed to `main.py` (or the individual scripts) if your files are located elsewhere.

---

## Pipeline overview

Stage 1 — Feature extraction (external)
- Compute descriptors externally using Matminer, mofdscribe, and any custom scripts.

Stage 2 — Feature merging
- Merge descriptor tables on the primary key (qmof_id) to produce a single dataframe for modeling.

Stage 3 — Feature selection & model evaluation (invoked via `main.py`)
- `main.py` serves as the orchestrator for training XGBoost models across multiple random seeds (default seeds are configurable).
- Computes average feature importances and evaluates model performance as a function of top-k features.
- Typical outputs:
    - results/xgb/avg_feature_importances.csv
    - results/xgb/model_performance_by_top_features.csv
    - results/xgb/performance_vs_features.png
    - results/xgb/best_model_pred_vs_actual.png

Stage 4 — SHAP analysis
- SHAP computations can be run through `main.py` or `xgb_shap_analysis.py` to produce interpretability plots for a chosen trained model or top-n features.
- Typical outputs:
    - results/xgb/shap/shap_summary_top30.png
    - results/xgb/shap/shap_beeswarm_top30.png (if enabled)

---

## Feature groups used

The merged descriptor table typically includes:
- Bond length statistics
- Neighbor distance variations
- Structural complexity measures
- Packing metrics, density, porosity (ASA, VSA, accessible pore volumes)
- RDF histogram features
- Crystallographic attributes (space group, symmetry, dimensionality)
- Pore geometry and related ratios

The pipeline selects relevant columns automatically; adjust selection logic or configuration passed to `main.py` if you change descriptor names or prefixes.