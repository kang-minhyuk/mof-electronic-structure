# Data-Driven Electronic Structure Analysis of Metal-Organic Frameworks  

This repository contains all code, workflows, and analysis scripts used in:

**“Data-Driven Electronic Structure Analysis of Metal-Organic Frameworks”**  
**Minhyuk Kang, Seung-Jae Shin, Tianshu Li, Aron Walsh**

This work investigates machine-learning approaches for accurate prediction of **HSE06 band gaps** in ~15,000 MOFs, focusing especially on the conductivity-related applications. The repository includes:

- Modified **CGCNN** with physics-informed atomic embeddings (partial atomic charges, electronic configuration features, GS properties).
- Full **XGBoost baseline pipeline**, including feature extraction, feature selection, and SHAP interpretability.
- Utilities for dataset reconstruction, preprocessing, graph construction, model training, and evaluation.
- Scripts for generating figures and reproducing results presented in poster/manuscript form.

---

## Overview

Accurate prediction of **MOFs** remains a difficult challenge for both traditional ML models and existing graph neural networks. Our findings highlight:

- Classical descriptors (RDF peaks, ordering parameters, pore metrics) help identify structural signatures of near-conductive MOFs.
- Baseline ML models (RF, SVR, LASSO, XGBoost) achieve good global accuracy but **fail systematically for Eg < 1.5 eV**.
- Standard CGCNN also overpredicts in this regime due to insufficient long-range electronic information.
- Introducing **partial charge embeddings** (PACMAN-predicted charges) can improve prediction.
- Case studies of **Zn-based MOFs** demonstrate structural categories (Zn–O bridges, clusters, single-metal units) where model performance differs due to charge-transport topology.

---

# Poster Summary

A full research poster describing this project is available here:

[**📄 MRS Poster — Data-Driven Electronic Structure Analysis of MOFs**](results/MRSposter.pdf)

The poster contains:
- Research motivation  
- MOF dataset sources  
- HSE06 band gap calculations  
- CGCNN architecture used  
- XGBoost baseline comparison  
- Key results and conclusions  

---

# Repository Structure

```
├── data/
│   ├── metadata/
│   └── README.md
│
├── models/
│   ├── cgcnn/
│   │   ├── charges/
│   │   ├── data/
│   │   ├── data.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── predict.py
│   │   └── README.md
│   │
│   └── xgb/
│       ├── data/
│       ├── main.py
│       └── README.md
│
├── results/
│   ├── figures/
│   └── scripts/
│
└── README.md
```

---

# Dataset

### Summary
- **Total MOFs:** 14,673  
  - **10,458** from the **QMOF database**  
  - **4,215** newly calculated using **HSE06** within the research group  

### Raw data is *not included* due to size and licensing constraints.

1. **QMOF dataset:**  
   https://github.com/Andrew-S-Rosen/QMOF  
2. **New HSE06 dataset (4,215 MOFs):**  
   Zenodo DOI: https://zenodo.org/records/15078750  
3. **Feature extraction tools:**  
   - MOFdscribe: https://mofdscribe.readthedocs.io  
   - Matminer: https://hackingmaterials.lbl.gov/matminer/  
   - PACMAN charges: https://github.com/Chung-Research-Group/PACMAN-charge

If required, **the full processed dataset can be provided upon request**.

---

# Models

# 1. CGCNN Pipeline

We extend the Crystal Graph Convolutional Neural Network to incorporate **additional atomic embeddings**, including:

- Valence electron configuration (Ns, Np, Nd, Nf unfilled)
- Ground-state properties (GS band gap, GS magnetic moment, GS volume/atom)
- **Partial atomic charges** predicted by PACMAN

# 2. XGBoost Pipeline

The XGBoost pipeline integrates ~300 descriptors including:
- RDF peaks
- Neighbor distance metrics
- Bond-length distortion features
- Packing fraction, density, pore properties
- Ordering parameters
- Space group + symmetry features

---
# Case Study: Zn-based MOFs

Using 165 Zn-based MOF structures
- Zn-O bridging networks: most predictable (delocalized paths)
- Metal clusters: intermediate difficulty
- Single-metal-centered MOFs: largest errors (localized electrons)