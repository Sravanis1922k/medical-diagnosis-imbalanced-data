# 🏥 MEDISCAN — Medical Diagnosis for Incomplete & Imbalanced Data

A production-grade deep learning system for disease risk prediction on real-world healthcare datasets with missing values and class imbalance. Upload a CSV dataset, train the model, and get instant risk predictions with confidence scores.

> 📄 **Published:** Springer — Intelligent Data Engineering and Analytics, Feb 28, 2022

---

## Architecture

```
Healthcare CSV Dataset
           ↓
  Preprocessing Agent  ← KNN Imputation (missing values) + SMOTE (class imbalance)
           ↓
  Training Agent       ← Multi-instance neural network (BatchNorm + Dropout)
           ↓
  Evaluation Agent     ← ROC-AUC, F1, Precision, Recall, Confusion Matrix
           ↓
  Prediction Agent     ← Single-patient risk scoring with confidence
           ↓
  FastAPI Backend      ← /upload, /train, /predict, /evaluate
           ↓
  Streamlit UI         ← Train → Evaluate → Predict dashboard
```

---

## Quickstart

### Step 1 — Clone the repo
```bash
git clone https://github.com/Sravanis1922k/medical-diagnosis-imbalanced-data.git
cd medical-diagnosis-imbalanced-data
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment
```bash
cp .env.example .env
```

### Step 4 — Start the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Step 5 — Start the frontend (new terminal)
```bash
cd frontend
streamlit run app.py
```

Open **http://localhost:8501** — upload your dataset and start training!

---

## Docker (run everything at once)
```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend (Streamlit) | http://localhost:8501 |
| Backend (FastAPI) | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## Common Errors & Fixes

**❌ TensorFlow not found**
```bash
pip install tensorflow>=2.16
```

**❌ imbalanced-learn not found**
```bash
pip install imbalanced-learn
```

**❌ Target column not found**
Make sure the target column name in the UI exactly matches the column name in your CSV (case-sensitive).

**❌ Model not trained yet — call /train first**
Go to the **Train Model** tab, upload a dataset, select the target column, and click Train.

**❌ FP16 is not supported on CPU; using FP32 instead**
This is a warning, not an error. Ignore it — runs fine on CPU.

**❌ SMOTE error — not enough samples in minority class**
Your dataset may have very few positive cases. Lower `SMOTE_STRATEGY` or add more data.

**❌ Backend shows stale config after editing .env**
```bash
uvicorn main:app --reload --port 8000
```

---

## Startup Checklist
```
□ requirements.txt installed
□ .env configured
□ Backend running on port 8000
□ Frontend running on port 8501
□ CSV dataset ready (see data/sample/README.md for schema)
```

**Start everything (copy-paste):**
```bash
# Terminal 1 — Backend
cd backend && uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && streamlit run app.py
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `KNN_NEIGHBORS` | `5` | Neighbours for KNN imputation |
| `SMOTE_STRATEGY` | `auto` | SMOTE sampling strategy |
| `BATCH_SIZE` | `32` | Training batch size |
| `LEARNING_RATE` | `0.001` | Adam optimizer learning rate |
| `DROPOUT_RATE` | `0.3` | Dropout for regularisation |
| `EARLY_STOP_PATIENCE` | `10` | Early stopping patience |
| `CONFIDENCE_THRESHOLD` | `0.5` | Risk classification threshold |

---

## Model Architecture

```
Input Features
     ↓
Dense(128) → BatchNormalization → Dropout(0.3)
     ↓
Dense(64)  → BatchNormalization → Dropout(0.3)
     ↓
Dense(32)  → ReLU
     ↓
Dense(1)   → Sigmoid
```

**Why this architecture?**
- BatchNormalization stabilises training on heterogeneous clinical features
- Dropout prevents overfitting on small healthcare datasets
- Early stopping + ReduceLROnPlateau ensures best weights are preserved

---

## Evaluation Metrics

Plain accuracy is misleading on imbalanced datasets. MEDISCAN reports:

| Metric | Why it matters |
|---|---|
| **ROC-AUC** | Measures discrimination across all thresholds |
| **F1 Score** | Balances precision and recall for imbalanced classes |
| **Precision** | Of predicted positives, how many are truly positive |
| **Recall** | Of actual positives, how many were detected |
| **Confusion Matrix** | Full TP / TN / FP / FN breakdown |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status + model state |
| `POST` | `/upload` | Upload and preview a CSV dataset |
| `POST` | `/train` | Train model on uploaded dataset |
| `POST` | `/predict` | Predict risk for a single patient |
| `GET` | `/evaluate` | Get latest evaluation metrics |

---

## Project Structure

```
medical-diagnosis-imbalanced-data/
├── backend/
│   ├── main.py                     # FastAPI — all endpoints
│   ├── config.py                   # Central config from .env
│   ├── agents/
│   │   ├── preprocessing_agent.py  # KNN Imputation + SMOTE + scaling
│   │   ├── training_agent.py       # Neural network training
│   │   ├── evaluation_agent.py     # ROC-AUC, F1, confusion matrix
│   │   └── prediction_agent.py     # Single-patient inference
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                      # Streamlit UI — train/evaluate/predict
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   └── Medical_Diagnosis_Notebook.ipynb   # Full EDA + pipeline walkthrough
├── data/
│   └── sample/                     # Sample dataset + schema docs
├── docs/                           # Research methodology documentation
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Results

| Metric | Value |
|---|---|
| Missing value handling | KNN Imputation (k=5) |
| Class imbalance handling | SMOTE oversampling |
| ROC-AUC  | 0.92 (held-out test set) |
| F1 Score | 0.89 (held-out test set) |
| Minority class recall improvement | 38% after SMOTE |
| Missing values handled | 15% missing rate → 0 after KNN imputation|
| Publication | Springer — Feb 2022 |

---

## License
MIT
