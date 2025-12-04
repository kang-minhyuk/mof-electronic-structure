# QMOF Database (Original MOF Dataset)

**Database name**: QMOF — Quantum MOF Database  
**Repository**: https://github.com/Andrew-S-Rosen/QMOF  
**Primary dataset download**: Provided via Figshare; follow instructions in the QMOF README.  
**License**: CC BY 4.0  

## Citation
If you use QMOF data, please cite:
- Rosen A. S., Iyer S. M., Ray D., Yao Z., Aspuru-Guzik A., Gagliardi L., Notestein J. M., Snurr R. Q. “Machine Learning the Quantum-Chemical Properties of Metal–Organic Frameworks for Accelerated Materials Discovery”, *Matter*, 4, 1578–1597 (2021). DOI: 10.1016/j.matt.2021.02.015  [GitHub](https://github.com/Andrew-S-Rosen/QMOF)  
- Rosen A. S., Fung V., Huck P., O’Donnell C. T., Horton M. K., Truhlar D. G., Persson K. A., Notestein J. M., Snurr R. Q. “High-Throughput Predictions of Metal–Organic Framework Electronic Properties: Theoretical Challenges, Graph Neural Networks, and Data Exploration”, *npj Comput. Mater.*, 8, 112 (2022). DOI: 10.1038/s41524-022-00796-6  [GitHub](https://github.com/Andrew-S-Rosen/QMOF)

## Usage in this project
We treat QMOF as the “core structure database + PBE reference data.”

- A script `scripts/download_qmof_data.py` will download the QMOF dataset to a local data directory
