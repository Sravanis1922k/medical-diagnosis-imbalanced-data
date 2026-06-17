"""
Evaluation Agent
ROC-AUC, F1, Precision, Recall, Confusion Matrix
"""
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_fscore_support
)
import config, logging

logger = logging.getLogger(__name__)

class EvaluationAgent:
    def __init__(self):
        self._latest = {}

    def evaluate(self, model, X_test, y_test) -> dict:
        y_prob = model.predict(X_test).ravel()
        y_pred = (y_prob >= config.CONFIDENCE_THRESHOLD).astype(int)

        roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
        cm      = confusion_matrix(y_test, y_pred).tolist()
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

        metrics = {
            "roc_auc":   roc_auc,
            "precision": round(float(p), 4),
            "recall":    round(float(r), 4),
            "f1_score":  round(float(f1), 4),
            "confusion_matrix": cm,
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=["Healthy", "At Risk"],
                output_dict=True
            )
        }
        self._latest = metrics
        logger.info(f"Evaluation — ROC-AUC: {roc_auc} | F1: {round(float(f1),4)}")
        return metrics

    def get_latest(self) -> dict:
        return self._latest
