# Data-Driven Electronic Structure Analysis of Metal-Organic Frameworks

This repository contains code and workflows for the project:

> **“Data-Driven Electronic Structure Analysis of Metal-Organic Frameworks”**  
> Minhyuk Kang, Seung-Jae Shin, Tianshu Li, Aron Walsh

The goal is to build and benchmark machine learning models for **HSE06-based band gap prediction** of metal-organic frameworks (MOFs).

---

## 1. Project Overview

### 1.1 Motivation

Most prior ML models for MOF band gaps are trained on **PBE** labels, which systematically **underestimate band gaps** and distort screening of semiconductive or conductive MOFs.  
To address this, we assembled a **high-fidelity HSE06 dataset (~14.7k MOFs)** and trained models directly on hybrid-functional labels.

Key points:

- High-quality electronic structure labels from **HSE06**.
- Balanced coverage over the full band-gap spectrum, including:
  - Near-conductive: 472 structures (Eg < 1.5 eV)  
  - Insulating: 468 structures (Eg > 6 eV)
- Benchmarking of **graph neural networks (CGCNN and variants)** and **gradient-boosted trees (XGBoost)**.

### 1.2 Dataset

The working dataset is **not shipped** in this repository due to size and licensing constraints. Instead, we provide:

- Scripts and metadata to reconstruct the splits and features.
- Interfaces for reading:
  - QMOF entries (≈10,458 structures)
  - Newly computed HSE06 entries (≈4,215 structures)

Data directories:

- `data/raw/`  
  Contains raw input data (e.g. CIFs, raw labels).
- `data/processed/`  
  Preprocessed tables and graph objects used for ML.
- `data/metadata/`  
  Data source information, IDs, provenance, and mapping files.

See `data/README.md` (to be created) for instructions on how to obtain and prepare data.

---

## 2. Repository Structure

```text
mof-electronic-structure/
│
├── data/                  # Raw / processed data (not in version control)
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── src/                   # Python source code (installable package)
│   ├── models/            # GNNs, XGBoost wrappers, etc.
│   ├── data_processing/   # Graph building, feature engineering, charges
│   ├── training/          # Training loops, schedulers, logging
│   ├── evaluation/        # Metrics, error analysis, motif studies
│   └── utils/             # Shared helpers
│
├── notebooks/             # Exploratory data analysis, plotting, debugging
│
├── results/               # Generated figures and metrics
│   ├── figures/
│   └── metrics/
│
├── scripts/               # CLI entry points (preprocess, train, evaluate)
│
├── configs/               # YAML configs for experiments
│
├── README.md
├── requirements.txt or environment.yml
├── LICENSE
└── CITATION.cff