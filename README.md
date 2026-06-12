# Medical Diagnosis for Incomplete and Imbalanced Data

Deep learning model for disease risk prediction on healthcare datasets with missing values and class imbalance.

> Published: **Springer, Intelligent Data Engineering and Analytics** — Feb 28, 2022

## Problem

Real-world clinical datasets suffer from two compounding issues:
- **Missing data** — patients with incomplete lab results or records
- **Class imbalance** — disease-positive cases are a small minority

## Solution

| Challenge | Technique |
|-----------|-----------|
| Missing values | KNN Imputation (`n_neighbors=5`) |
| Class imbalance | SMOTE oversampling |
| Generalisation | Dropout + BatchNorm + Early Stopping |

## Architecture

```
Input Features
     ↓
Dense(128) → BatchNorm → Dropout(0.3)
     ↓
Dense(64)  → BatchNorm → Dropout(0.3)
     ↓
Dense(32)  → ReLU
     ↓
Dense(1)   → Sigmoid (binary) / Softmax (multi-class)
```

## Setup

```bash
pip install tensorflow scikit-learn imbalanced-learn pandas matplotlib
```

## Usage

1. Place your dataset CSV in `data/healthcare_dataset.csv`
2. Set `TARGET_COL` in `model.py` to your label column name
3. Run:

```bash
python model.py
```

Outputs:
- `best_model.keras` — best checkpoint
- `training_curves.png` — accuracy & loss plots

## Evaluation Metrics

- Precision / Recall / F1-score
- ROC-AUC
- Confusion Matrix

These metrics are chosen to handle imbalanced classes where plain accuracy is misleading.
