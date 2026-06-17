"""
MEDISCAN — Medical Diagnosis for Incomplete & Imbalanced Data
FastAPI Backend — /predict, /train, /evaluate, /health
Published: Springer, Intelligent Data Engineering and Analytics, Feb 2022
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import logging, io
from pathlib import Path

from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.prediction_agent import PredictionAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MEDISCAN — Medical Diagnosis System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

preprocessor = PreprocessingAgent()
trainer      = TrainingAgent()
evaluator    = EvaluationAgent()
predictor    = PredictionAgent()

# ── Request models ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: dict
    return_probability: Optional[bool] = True

class TrainRequest(BaseModel):
    target_column: str
    test_size: Optional[float] = 0.2
    epochs: Optional[int] = 50

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_trained": trainer.is_trained(),
        "publication": "Springer — Intelligent Data Engineering and Analytics, Feb 2022"
    }

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a healthcare CSV dataset."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    dest = DATA_DIR / file.filename
    df.to_csv(dest, index=False)
    missing_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
    class_dist = {}
    logger.info(f"Uploaded {file.filename}: {df.shape[0]} rows, {df.shape[1]} cols, {missing_pct}% missing")
    return {
        "status": "uploaded",
        "file": file.filename,
        "rows": df.shape[0],
        "columns": list(df.columns),
        "missing_pct": missing_pct,
        "dtypes": df.dtypes.astype(str).to_dict()
    }

@app.post("/train")
async def train_model(req: TrainRequest, file: UploadFile = File(...)):
    """Train the model on uploaded dataset."""
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    if req.target_column not in df.columns:
        raise HTTPException(400, f"Target column '{req.target_column}' not found in dataset")

    try:
        X_train, X_test, y_train, y_test, scaler = preprocessor.process(df, req.target_column, req.test_size)
        history = trainer.train(X_train, y_train, X_test, y_test, epochs=req.epochs)
        metrics = evaluator.evaluate(trainer.model, X_test, y_test)
        predictor.set_model(trainer.model, scaler, list(df.drop(columns=[req.target_column]).columns))
        return {
            "status": "trained",
            "metrics": metrics,
            "epochs_run": len(history.history["loss"]),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")

@app.post("/predict")
def predict(req: PredictRequest):
    """Predict disease risk for a single patient."""
    if not trainer.is_trained():
        raise HTTPException(400, "Model not trained yet. Call /train first.")
    try:
        result = predictor.predict(req.features, req.return_probability)
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.get("/evaluate")
def get_evaluation():
    """Get the latest evaluation metrics."""
    if not trainer.is_trained():
        raise HTTPException(400, "Model not trained yet.")
    return evaluator.get_latest()
