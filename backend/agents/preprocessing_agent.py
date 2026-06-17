"""
Preprocessing Agent
- KNN Imputation for missing values
- SMOTE for class imbalance
- StandardScaler normalisation
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import config, logging

logger = logging.getLogger(__name__)

class PreprocessingAgent:
    def process(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2):
        X = df.drop(columns=[target_col])
        y = df[target_col].values

        # Step 1 — KNN Imputation
        missing_before = int(X.isnull().sum().sum())
        imputer = KNNImputer(n_neighbors=config.KNN_NEIGHBORS)
        X_imputed = imputer.fit_transform(X)
        logger.info(f"KNN Imputation: {missing_before} missing values filled")

        # Step 2 — SMOTE
        counts_before = dict(zip(*np.unique(y, return_counts=True)))
        smote = SMOTE(random_state=config.RANDOM_STATE, sampling_strategy=config.SMOTE_STRATEGY)
        X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
        counts_after = dict(zip(*np.unique(y_resampled, return_counts=True)))
        logger.info(f"SMOTE: {counts_before} → {counts_after}")

        # Step 3 — Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled,
            test_size=test_size,
            random_state=config.RANDOM_STATE,
            stratify=y_resampled
        )

        # Step 4 — Standardise
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test, scaler
