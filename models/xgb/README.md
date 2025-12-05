# XGBoost Feature-Based Band Gap Prediction

This directory contains the full machine-learning pipeline for **feature-driven prediction of MOF band gaps** using classical models (primarily XGBoost).  
The workflow integrates descriptors from **Matminer**, **mofdscribe**, and additional MOF structural/porosity feature sources.

The goal of this pipeline is to benchmark classical ML methods against CGCNN and to identify the most physically meaningful descriptors for predicting hybrid-DFT (HSE06) band gaps of MOFs.

---

## Directory Structure

xgb/
├── xgb_feature_selection.py        # Full feature selection + model evaluation script
├── xgb_shap_analysis.py            # SHAP-based interpretability analysis
├── README.md                       # This file
└── results/                        # Output metrics, plots, and analyses

---

## 1. Data Requirements

Due to size limitations, **raw datasets are not included in the repository**.

To reproduce the XGBoost pipeline, download or generate the following feature tables:

### 1. Matminer Features  
Documentation:  
https://hackingmaterials.lbl.gov/matminer/

Example file:  

matminer_features.csv

### 2. MOF Feature Bundle  
Includes geometric and chemical descriptors (e.g., bond length statistics, neighbor distances, structural complexity, pore metrics).

Example file:  

mof_features_combined.csv

### 3. mofdscribe Descriptors  
Documentation:  
https://mofdscribe.readthedocs.io/en/latest/

Example file:  

mofdscribe_results.csv

or update paths in your script accordingly.

---

## 2. Pipeline Overview

The full pipeline proceeds in four stages:

### **Stage 1 — Feature Extraction (external)**
Descriptors are computed externally using Matminer, mofdscribe, and custom scripts.  
This repository does not compute them automatically, but provides instructions.

### **Stage 2 — Feature Merging**
All descriptors are merged into a single dataframe keyed by:

qmof_id

### **Stage 3 — Feature Selection with XGBoost**
- Train models over random seeds 0–9  
- Compute average feature importances  
- Evaluate top-k features for k = 50 → 5  
- Identify optimal feature subset  
- Save:
  - `avg_feature_importances.csv`
  - `model_performance_by_top_features.csv`
  - `performance_vs_features.png`
  - `best_model_pred_vs_actual.png`

## 4. Running SHAP Analysis

python xgb_SHAP_analysis.py 
–matminer-features data/raw/matminer_features.csv 
–mof-features data/raw/mof_features_combined.csv 
–mofdscribe-results data/raw/mofdscribe_results.csv 
–avg-importances results/xgb/avg_feature_importances.csv 
–top-n 30 
–outdir results/xgb/shap

Outputs:

shap_summary_top30.png

---

## 5. Feature Groups Used

The pipeline uses the following descriptor categories:

- **Bond length statistics**  
- **Neighbor distance variations**  
- **Structural complexity measures**  
- **Packing metrics, density, porosity**  
- **RDF histogram features**  
- **Crystallographic attributes (space group, symmetry, dimensionality)**  
- **Pore features (ASA, VSA, A/V ratios, accessible pore volumes)**  

The merged descriptor table includes all of these categories, and the pipeline automatically selects relevant columns.