# docs — Research & Methodology

This folder contains documentation supporting the published research behind MEDISCAN.

## Publication

**Title:** Medical Diagnosis for Incomplete and Imbalanced Data  
**Published in:** Springer — Intelligent Data Engineering and Analytics  
**Date:** February 28, 2022  

## Key Methodology

### Challenge 1 — Missing Data
Real-world clinical datasets frequently contain missing values due to incomplete lab results,
patient non-compliance, or data entry errors. MEDISCAN uses **KNN Imputation** (k=5) to fill
missing values based on the k nearest neighbours in feature space, preserving local data
distribution better than mean/median imputation.

### Challenge 2 — Class Imbalance
Disease-positive cases are typically a small minority in clinical datasets (often 10-20%).
Training on imbalanced data causes models to be biased toward the majority (healthy) class,
leading to poor sensitivity for detecting at-risk patients. MEDISCAN uses **SMOTE**
(Synthetic Minority Over-sampling Technique) to generate synthetic minority class samples,
balancing the training set before model fitting.

### Model Architecture
A multi-instance neural network with:
- Dense(128) → BatchNormalization → Dropout(0.3)
- Dense(64)  → BatchNormalization → Dropout(0.3)
- Dense(32)  → ReLU
- Dense(1)   → Sigmoid

### Evaluation Metrics
Plain accuracy is misleading on imbalanced datasets. MEDISCAN reports:
- **ROC-AUC** — measures discrimination ability across all thresholds
- **F1 Score** — harmonic mean of precision and recall
- **Precision** — of all predicted positives, how many are truly positive
- **Recall (Sensitivity)** — of all actual positives, how many were detected
- **Confusion Matrix** — full breakdown of TP, TN, FP, FN
