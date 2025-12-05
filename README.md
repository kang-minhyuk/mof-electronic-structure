# Data-Driven Electronic Structure Analysis of Metal-Organic Frameworks

This repository contains code, workflows, and analysis used in:

**“Data-Driven Electronic Structure Analysis of Metal-Organic Frameworks”**  
**Minhyuk Kang, Seung-Jae Shin, Tianshu Li, Aron Walsh**

The goal of this repository is to provide *fully reproducible tooling* for preparing datasets, training machine-learning models, and evaluating predictions of HSE06 band gaps for metal–organic frameworks (MOFs).  
Both graph neural network (CGCNN-style) models and XGBoost baselines are included.

## Highlights
- High-fidelity **HSE06 band gap labels** (not included here due to size restrictions).  
- Modular implementations of:
  - CGCNN and extended variants.  
  - XGBoost feature-based baselines with feature selection and SHAP interpretation.  
- End-to-end scripts for:
  - Structure preprocessing  
  - Graph construction  
  - Model training & hyperparameter search  
  - Dataset splitting and reproducibility  
  - Analysis and figure generation  

---

## Repository Layout
- data/
  - metadata/
  - README.md
- models/
  - cgcnn/
    - charges/
    - data/
    - data.py
    - main.py
    - model.py
    - predict.py
    - README.md
  - xgb/
    - data/
    - main.py
    - README.md
- results/
  - figures/
  - scripts/
- README.md


