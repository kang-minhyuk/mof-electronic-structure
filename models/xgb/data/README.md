# Dataset Construction

Raw CSV feature files are **not uploaded** to this repository due to size limitations and the heavy external dependencies required to generate them.  
To reproduce the dataset used in this project, please obtain the following feature sets from the official sources listed below.
---

## 1. mofdscribe Features

MOF-specific descriptors including structural complexity, symmetry information, crystal system, space-group metrics, packing fraction, density, and other structure-derived features.

**Documentation:**  
https://mofdscribe.readthedocs.io/en/latest/

---

## 2. Matminer Features

General-purpose materials descriptors including RDFs, bond-length statistics, structural metrics, chemical/atomic environment descriptors, and many others.

**Documentation:**  
https://hackingmaterials.lbl.gov/matminer/

---

## 3. MOF Porosity & Geometry Features

Descriptors related to pore structure, accessible surface area, pore size, void fraction, LIS/LIFS/LIFSP, and adsorption-accessible geometry.  
These are typically generated using tools such as Zeo++ or equivalent MOF geometry analysis packages.

---

## Notes

- These datasets require external preprocessing and are **not** included directly in this repository.  
- Follow the Matminer and mofdscribe documentation to regenerate descriptor files if needed.  
- The machine-learning scripts in this repository automatically merge these raw feature files into a unified dataset for training and evaluation.  
- **The full processed dataset can be provided upon request.**