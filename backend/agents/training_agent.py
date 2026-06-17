"""
Training Agent
Multi-instance neural network with BatchNorm + Dropout
"""
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import config, logging

logger = logging.getLogger(__name__)

class TrainingAgent:
    def __init__(self):
        self.model   = None
        self._trained = False

    def is_trained(self) -> bool:
        return self._trained

    def _build(self, input_dim: int) -> keras.Model:
        inputs = keras.Input(shape=(input_dim,), name="features")
        x = keras.layers.Dense(128, activation="relu")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(config.DROPOUT_RATE)(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(config.DROPOUT_RATE)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 50):
        self.model = self._build(X_train.shape[1])
        Path("models").mkdir(exist_ok=True)

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=config.EARLY_STOP_PATIENCE,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
            keras.callbacks.ModelCheckpoint(
                config.MODEL_SAVE_PATH, save_best_only=True, verbose=0
            ),
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        self._trained = True
        logger.info(f"Training complete — {len(history.history['loss'])} epochs")
        return history
