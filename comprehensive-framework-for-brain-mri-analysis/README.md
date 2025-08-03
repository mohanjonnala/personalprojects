# Comprehensive Framework for Brain MRI Analysis 🧠

**Based on:** *A Comprehensive Framework for Brain MRI Analysis: Classification, Segmentation, and Survival Prediction* — presented in *FICTA 2024* and published March 29, 2025 ([Springer Link](https://link.springer.com/chapter/10.1007/978-981-96-0143-1_48))

---

## 📋 Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Evaluation & Results](#evaluation--results)
* [Citation](#citation)

---

## Overview

This repository implements a unified framework for **brain tumor MRI analysis**, combining:

1. **Classification** of MRI scans into *meningioma*, *pituitary*, *glioma*, or *non-tumor* using a CNN-based classifier.
2. **Segmentation** of tumor regions with a U‑Net architecture, enabling quantification of tumor size and localization.
3. **Survival prediction** using a CNN-based regression model.

This end‑to‑end approach achieved high performance:

* Classification accuracy: **98.12%**
* Segmentation accuracy (Dice): **92%**
* Survival prediction accuracy: **96%**

---

## Features

* **Pre‑processing pipeline**: skull-stripping, normalization, augmentation
* **Deep learning classifiers**: CNN for tumor type prediction
* **Segmentation**: U‑Net for tumor mask generation
* **Survival models**: CNN‑based model trained to predict patient survival times
* Modular and extendable architecture

---

## Project Structure

```
comprehensive-framework-for-brain-mri-analysis/
│
├── data/                   # Datasets, preprocessing scripts  
├── preprocessing/          # Image normalization, augmentation  
├── models/
│   ├── classification/     # CNN architecture and training scripts  
│   ├── segmentation/       # U‑Net model and mask generation  
│   └── survival/           # Regression model  
├── notebooks/              # Jupyter notebooks for experimental analysis  
├── results/                # Output masks, metrics, visualizations  
├── requirements.txt        # Python dependencies  
└── README.md               # This documentation
```

---

## Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/mohanjonnala/personalprojects.git
   cd personalprojects/comprehensive-framework-for-brain-mri-analysis
   ```

2. **Set up the environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Prepare your dataset** in the `data/` folder. Organize according to BIDS standard for neuroimaging data.

4. **Run preprocessing**:

   ```bash
   python preprocessing/prepare_data.py --input data/raw --output data/processed
   ```

---

## Usage

### 🔍 Train / Evaluate Models

* **Classification**:

  ```bash
  python models/classification/train.py --data data/processed --epochs 50 --batch-size 32
  ```
* **Segmentation**:

  ```bash
  python models/segmentation/train_unet.py --data data/processed
  ```
* **Survival prediction**:

  ```bash
  python models/survival/train_survival.py --data data/processed
  ```

### 🎯 Inference

```bash
python inference/predict_all.py \
  --image input_mri.nii.gz \
  --out_dir predictions/
```

Output:

* `tumor_type_pred.txt`
* `segmentation_mask.nii.gz`
* `survival_prediction.csv`

---

## Evaluation & Results

Based on experiments documented in the accompanying publication:

* **Classification:** 98.12% accuracy on test set
* **Segmentation:** Dice similarity score 0.92
* **Survival Prediction:** R² ≈ 0.96

Performance was validated across multiple MRI datasets and robust across tumor types.

---

## 📚 Citation

If you use this framework in your work, please cite:

> Raju, B. V. S. R. K., Jonnala, M. S. D., Dasari, J., Annem, R. S., & Challa, S. S. (2025). *A Comprehensive Framework for Brain MRI Analysis: Classification, Segmentation, and Survival Prediction.* In *Intelligent Computing and Automation* (Vol. 421, pp. 593–607). Springer Nature. DOI: [10.1007/978-981-96-0143-1\_48](https://link.springer.com/chapter/10.1007/978-981-96-0143-1_48)

---

## 🔧 Further Enhancements

* Add **multi-class explainability** using CAM/Grad‑CAM modules
* Incorporate **nnU‑Net** or **FastSurfer**‑style pipelines for enhanced segmentation
* Evaluate integration with **FSL**, **CAT12**, or **FreeSurfer** for tissue-specific morphometry
