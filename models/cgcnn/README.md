# CGCNN for MOF Band Gap Prediction

This directory contains a full, reproducible implementation of a **Crystal Graph Convolutional Neural Network (CGCNN)** for predicting electronic properties of metal–organic frameworks (MOFs), with a focus on HSE06 band gap prediction.

It includes:

- A modified CGCNN architecture with residual connections  
- Optional Global Context Module  
- Support for custom atom embeddings (`atom_init.json` + variants)  
- Optional partial charge augmentation  
- Full training pipeline with WandB logging  
- A clean inference pipeline for prediction  
- Reproducible dataset construction and splitting  

---

## Directory Structure

cgcnn/
├── main.py                # Training pipeline  
├── model.py               # CGCNN architecture (residual + global context)  
├── data.py                # CIFData loader + embeddings + partial charges  
├── predict.py             # Inference script  
└── README.md              # This file  

---

## 1. Requirements

Recommended environment:

- Python 3.9+
- PyTorch >= 2.0
- pymatgen
- scikit-learn
- wandb

Install dependencies:

pip install torch pymatgen scikit-learn wandb numpy

---

## 2. Dataset Structure

Place your processed dataset in the following format:

data/
├── id_prop.csv
├── atom_init.json
├── atom_init_<variant>.json        # optional embedding variants
└── <qmof_id>.cif

### id_prop.csv (example)

qmof_id,target
QMOF_0001,2.14
QMOF_0002,0.83

### Atom Embedding Files

- `atom_init.json` — default embedding  
- `atom_init_<variant>.json` — alternate versions selected via:

–embedding 

### Partial Charges (Optional)

If available:

dataset_root/charges/charges_dict.json

Format:

```json
{
  "QMOF_0001": [0.02, -0.13, 0.51],
  "QMOF_0002": [-0.01, 0.22]
}

Enable with:

--use-charge

⸻

3. Training

Run the training pipeline:

python main.py dataset_root \
    --embedding default \
    --use-charge false \
    --epochs 200 \
    --batch-size 256 \
    --lr 0.01 \
    --wandb-project cgcnn_mof

Important Arguments

Argument	Description
dataset_root	Directory containing CIFs + id_prop.csv
–embedding	Choose atom_init variant
–use-charge	Append partial charges
–task	regression or classification
–random	Random seed
–n-conv	Number of convolution layers
–n-h	Number of fully connected layers
–atom-fea-len	Atom feature dimension
–h-fea-len	Hidden layer dimension
–resume	Resume from checkpoint
–lr-milestones	Learning rate scheduler

Output Files
	•	checkpoint.pth.tar
	•	model_best_charge_<embedding>_<seed>.pth.tar

⸻

4. Inference / Prediction

Use the standalone prediction script:

python predict.py \
    model_best_charge_default_123.pth.tar \
    dataset_root \
    --embedding-name default \
    --random-seed 123

Outputs:

test_results_pred.csv

Format:

qmof_id,target,predicted

⸻

5. Model Architecture (model.py)

Key features:
	•	Residual skip connections in every convolution layer
	•	Optional GlobalContextModule for long-range structural information
	•	Configurable model depth and hidden dimensions
	•	Supports charge-augmented atom features
	•	Flexible pooling and fully connected layers

Hyperparameters are controlled via:

--n-conv
--n-h
--atom-fea-len
--h-fea-len

⸻

6. Data Pipeline (data.py)

CIFData handles:
	•	Reading structures via pymatgen
	•	Building neighbor lists with Gaussian distance expansion
	•	Creating atom features using AtomCustomJSONInitializer
	•	Adding partial charges if enabled
	•	Returning:

atom_fea              # (N_atoms, feature_dim)
nbr_fea               # (N_atoms, max_neighbors, dist_features)
nbr_fea_idx           # neighbor indices
crystal_atom_idx      # mapping atom index → crystal index

All steps are reproducible via:

--random <seed>

⸻

7. Checkpoints

Each checkpoint contains:
	•	state_dict
	•	optimizer
	•	normalizer
	•	args
	•	epoch
	•	best_mae_error

predict.py loads these automatically.

⸻

8. Example Training Command

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

⸻

9. Example Prediction Command

python predict.py \
    model_best_charge_default_42.pth.tar \
    data/qmof_hse \
    --embedding-name default \
    --random-seed 42

Produces:

test_results_pred.csv

⸻

10. Notes
	The implementation is compatible with:
	•	QMOF dataset
	•	New or custom MOF datasets
	•	Custom atom embeddings
	•	Charge-augmented feature sets
