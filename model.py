"""
Medical Diagnosis for Incomplete and Imbalanced Data
Stack: TensorFlow/Keras, Scikit-learn, imbalanced-learn
Published: Springer, Intelligent Data Engineering and Analytics, Feb 2022
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(csv_path: str, target_col: str, test_size: float = 0.2):
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Missing values:\n{df.isnull().sum()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- KNN Imputation for missing values ---
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    # --- SMOTE for class imbalance ---
    logger.info(f"Class distribution before SMOTE: {dict(zip(*np.unique(y, return_counts=True)))}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
    logger.info(f"Class distribution after SMOTE:  {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=42, stratify=y_resampled
    )

    # --- Standardisation ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.shape[1]


# ---------------------------------------------------------------------------
# 2. Model: multi-instance neural network
# ---------------------------------------------------------------------------

def build_model(input_dim: int, n_classes: int = 2) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")

    # Instance-level feature extraction
    x = keras.layers.Dense(128, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(32, activation="relu")(x)

    # Output
    activation = "sigmoid" if n_classes == 2 else "softmax"
    outputs = keras.layers.Dense(1 if n_classes == 2 else n_classes, activation=activation)(x)

    model = keras.Model(inputs, outputs)
    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=loss, metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train(model: keras.Model, X_train, y_train, X_val, y_val, epochs=50):
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: keras.Model, X_test, y_test):
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "At Risk"]))

    roc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    return {"roc_auc": roc, "confusion_matrix": cm}


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history.history[metric],      label="Train")
        ax.plot(history.history[f"val_{metric}"], label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CSV_PATH   = "data/healthcare_dataset.csv"   # replace with your dataset
    TARGET_COL = "disease_risk"                  # replace with your target column

    X_train, X_test, y_train, y_test, n_features = load_and_preprocess(CSV_PATH, TARGET_COL)

    # 80/20 further split for validation during training
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model = build_model(input_dim=n_features)
    model.summary()

    history = train(model, X_tr, y_tr, X_val, y_val)
    metrics = evaluate(model, X_test, y_test)
    plot_history(history)

    model.save("medical_diagnosis_model.keras")
    logger.info("Model saved to medical_diagnosis_model.keras")
