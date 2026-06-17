"""
Prediction Agent — single patient risk prediction
"""
import numpy as np
import config, logging

logger = logging.getLogger(__name__)

class PredictionAgent:
    def __init__(self):
        self.model    = None
        self.scaler   = None
        self.features = []

    def set_model(self, model, scaler, features: list):
        self.model    = model
        self.scaler   = scaler
        self.features = features

    def predict(self, feature_dict: dict, return_probability: bool = True) -> dict:
        values = [float(feature_dict.get(f, 0)) for f in self.features]
        X = np.array(values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prob = float(self.model.predict(X_scaled).ravel()[0])
        label = "At Risk" if prob >= config.CONFIDENCE_THRESHOLD else "Healthy"
        result = {
            "prediction": label,
            "risk_score": round(prob, 4),
            "confidence": round(prob if label == "At Risk" else 1 - prob, 4),
            "threshold_used": config.CONFIDENCE_THRESHOLD
        }
        if return_probability:
            result["probability_healthy"] = round(1 - prob, 4)
            result["probability_at_risk"]  = round(prob, 4)
        logger.info(f"Prediction: {label} (score={prob:.4f})")
        return result
