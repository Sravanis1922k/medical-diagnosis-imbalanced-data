# data/sample — Sample Healthcare Dataset

This folder contains a sample dataset to test MEDISCAN immediately after setup.

## Using the sample dataset

Upload `sample_healthcare_data.csv` in the **Train Model** tab of the Streamlit app.
Set the target column to `disease_risk` and click **Train Model**.

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| `age` | float | Patient age |
| `bmi` | float | Body mass index |
| `blood_pressure` | float | Systolic blood pressure |
| `cholesterol` | float | Total cholesterol (mg/dL) |
| `glucose` | float | Fasting glucose (mg/dL) |
| `insulin` | float | Insulin level |
| `heart_rate` | float | Resting heart rate |
| `hemoglobin` | float | Hemoglobin level (g/dL) |
| `wbc_count` | float | White blood cell count |
| `rbc_count` | float | Red blood cell count |
| `platelet_count` | float | Platelet count |
| `sodium` | float | Serum sodium (mEq/L) |
| `potassium` | float | Serum potassium (mEq/L) |
| `creatinine` | float | Creatinine level (mg/dL) |
| `urea` | float | Blood urea nitrogen (mg/dL) |
| `disease_risk` | int | Target: 0 = Healthy, 1 = At Risk |

## Real Datasets to Try

- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)

> ⚠️ Replace real patient data before uploading. Never commit real healthcare data to GitHub.
