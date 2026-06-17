import os
from dotenv import load_dotenv
load_dotenv()

# Imputation
KNN_NEIGHBORS       = int(os.getenv("KNN_NEIGHBORS", "5"))

# SMOTE
SMOTE_STRATEGY      = os.getenv("SMOTE_STRATEGY", "auto")
RANDOM_STATE        = int(os.getenv("RANDOM_STATE", "42"))

# Model
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE       = float(os.getenv("LEARNING_RATE", "0.001"))
DROPOUT_RATE        = float(os.getenv("DROPOUT_RATE", "0.3"))
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "10"))

# Evaluation
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Paths
MODEL_SAVE_PATH     = os.getenv("MODEL_SAVE_PATH", "models/best_model.keras")
SCALER_SAVE_PATH    = os.getenv("SCALER_SAVE_PATH", "models/scaler.pkl")
