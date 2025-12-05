# CGCNN for MOF Band Gap Prediction

This directory contains a reproducible implementation of a Crystal Graph Convolutional Neural Network (CGCNN) for predicting electronic properties of metal–organic frameworks (MOFs), focused on HSE06 band gap prediction.

Key features:
- Modified CGCNN with residual skip connections
- Optional Global Context Module for long-range information
- Support for custom atom embeddings (atom_init.json and variants)
- Optional partial-charge augmentation
- Full training pipeline with Weights & Biases (WandB) logging
- Simple inference pipeline and reproducible dataset construction

---

## Directory structure

cgcnn/
```
├── main.py        # Training pipeline
├── model.py       # CGCNN architecture (residual + global context)
├── data.py        # CIFData loader, embeddings, partial charges
├── predict.py     # Inference script
└── README.md      # This file
```

---

## 1. Requirements

Recommended environment:
- Python 3.9+
- PyTorch >= 2.0
- pymatgen
- scikit-learn
- wandb
- numpy

Install:
```
pip install torch pymatgen scikit-learn wandb numpy
```

---

## 2. Dataset layout

Place your processed dataset under a dataset root:

```
data/
├── id_prop.csv
├── atom_init.json
├── atom_init_<variant>.json    # optional embedding variants
└── <qmof_id>.cif
```

id_prop.csv (example):
```
qmof_id,target
QMOF_0001,2.14
QMOF_0002,0.83
```

Atom embedding files:
- atom_init.json — default atom embedding
- atom_init_<variant>.json — alternate embeddings; selected via the --embedding flag

Partial charges (optional):
```
dataset_root/charges/charges_dict.json
```
Example format:
```json
{
    "QMOF_0001": [0.02, -0.13, 0.51],
    "QMOF_0002": [-0.01, 0.22]
}
```
Enable with --use-charge.

---

## 3. Training

Basic usage:
```
python main.py dataset_root \
        --embedding default \
        --use-charge false \
        --epochs 200 \
        --batch-size 256 \
        --lr 0.01 \
        --wandb-project cgcnn_mof
```

Important arguments:
- dataset_root: Directory containing CIFs + id_prop.csv
- --embedding: Atom embedding name (default / variant)
- --use-charge: true | false — append partial charges to atom features
- --task: regression | classification
- --random: random seed
- --n-conv: number of convolution layers
- --n-h: number of fully connected layers
- --atom-fea-len: atom feature dimension
- --h-fea-len: hidden layer dimension
- --resume: path to checkpoint to resume
- --lr-milestones: scheduler milestones

Output files:
- checkpoint.pth.tar
- model_best_charge_<embedding>_<seed>.pth.tar

---

## 4. Inference / Prediction

Run standalone prediction:
```
python predict.py \
        model_best_charge_default_123.pth.tar \
        dataset_root \
        --embedding-name default \
        --random-seed 123
```
Produces test_results_pred.csv with columns:
```
qmof_id,target,predicted
```

---

## 5. Model architecture (model.py)

Highlights:
- Residual skip connections in convolution layers
- Optional GlobalContextModule for long-range structural context
- Configurable depth and hidden sizes
- Supports charge-augmented atom features
- Flexible pooling and fully connected head

Controlled via:
--n-conv, --n-h, --atom-fea-len, --h-fea-len

---

## 6. Data pipeline (data.py)

CIFData performs:
- Reading structures with pymatgen
- Building neighbor lists and Gaussian distance expansion
- Creating atom features via AtomCustomJSONInitializer (atom_init.json)
- Adding partial charges when enabled
- Returning:
    - atom_fea        (N_atoms, feature_dim)
    - nbr_fea         (N_atoms, max_neighbors, dist_features)
    - nbr_fea_idx     (neighbor indices)
    - crystal_atom_idx (mapping atom index → crystal index)

All randomness is controlled with --random <seed>.

---

## 7. Checkpoints

Each checkpoint includes:
- state_dict
- optimizer state
- normalizer
- args
- epoch
- best_mae_error

predict.py will load these automatically.

---

## 8. Example training command

```
python main.py data/qmof_hse \
        --task regression \
        --embedding default \
        --use-charge true \
        --n-conv 3 \
        --n-h 1 \
        --atom-fea-len 64 \
        --h-fea-len 128 \
        --batch-size 256 \
        --lr 0.01 \
        --epochs 200 \
        --random 42 \
        --wandb-project mof_cgcnn_hse
```

---

## 9. Example prediction command

```
python predict.py \
        model_best_charge_default_42.pth.tar \
        data/qmof_hse \
        --embedding-name default \
        --random-seed 42
```
Output: test_results_pred.csv

---

## 10. Notes

- Compatible with QMOF and custom MOF datasets
- Supports custom atom embeddings and charge-augmented feature sets
- Ensure atom_init.json variants match the atom types in your CIFs
