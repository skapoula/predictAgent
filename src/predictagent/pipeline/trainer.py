"""TensorFlow LSTM + GradientBoosting ensemble trainer and predictor."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from predictagent.config import Settings
from predictagent.exceptions import TrainingError
from predictagent.schemas import TrainingMetrics

logger = logging.getLogger(__name__)


def build_lstm_model(timesteps: int, n_features: int, lr: float) -> tf.keras.Model:
    """Build the 3-layer LSTM regressor with L1L2 regularisation.

    Args:
        timesteps: Lookback window length.
        n_features: Number of input features.
        lr: Adam learning rate.

    Returns:
        Compiled Keras model.
    """
    reg = tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, n_features)),
            tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, kernel_regularizer=reg),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae"],
    )
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict with "rmse", "mae", "mape" keys.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_pred - y_true) / denom)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def _compute_blend_weight(
    lstm_pred: np.ndarray, gbr_pred: np.ndarray, y_true: np.ndarray
) -> float:
    """Compute optimal LSTM blend weight alpha on the validation set.

    alpha minimises ||y - (alpha * lstm + (1-alpha) * gbr)||^2.

    Returns:
        Alpha clipped to [0, 1].
    """
    denom_vec = lstm_pred - gbr_pred
    numerator = float(np.dot(y_true - gbr_pred, denom_vec))
    denominator = float(np.dot(denom_vec, denom_vec))
    if denominator == 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def predict(
    lstm_model: tf.keras.Model,
    gbr_model: GradientBoostingRegressor,
    X: np.ndarray,
    alpha: float,
) -> float:
    """Run blend prediction for a single input window.

    Args:
        lstm_model: Trained Keras model.
        gbr_model: Trained GradientBoostingRegressor.
        X: Input array of shape (1, lookback, n_features).
        alpha: LSTM blend weight (stored in registry metadata).

    Returns:
        Scalar blend prediction.
    """
    lstm_pred = float(lstm_model.predict(X, verbose=0).ravel()[0])
    gbr_pred = float(gbr_model.predict(X.reshape(1, -1))[0])
    return alpha * lstm_pred + (1.0 - alpha) * gbr_pred


def train_cell(
    cell_name: str,
    tensor_dir: Path,
    settings: Settings,
) -> tuple[tf.keras.Model, GradientBoostingRegressor, float, TrainingMetrics]:
    """Train LSTM + GBR ensemble for a single cell.

    Args:
        cell_name: Viavi cell identifier (used for logging).
        tensor_dir: Directory containing train/val/test joblib tensors.
        settings: Application settings.

    Returns:
        Tuple of (lstm_model, gbr_model, alpha_blend_weight, TrainingMetrics).

    Raises:
        TrainingError: If tensors are missing or training fails.
    """
    tf.random.set_seed(settings.training.seed)
    np.random.seed(settings.training.seed)

    required = ["train_inputs", "train_targets", "val_inputs", "val_targets",
                "test_inputs", "test_targets"]
    for name in required:
        if not (tensor_dir / f"{name}.joblib").exists():
            raise TrainingError(f"Missing tensor file '{name}.joblib' in {tensor_dir}")

    train_x = joblib.load(tensor_dir / "train_inputs.joblib")
    train_y = joblib.load(tensor_dir / "train_targets.joblib")
    val_x = joblib.load(tensor_dir / "val_inputs.joblib")
    val_y = joblib.load(tensor_dir / "val_targets.joblib")
    test_x = joblib.load(tensor_dir / "test_inputs.joblib")
    test_y = joblib.load(tensor_dir / "test_targets.joblib")

    timesteps, n_features = train_x.shape[1], train_x.shape[2]
    logger.info(
        "Training %s: %d train, %d val, %d test samples; shape=(%d, %d)",
        cell_name, len(train_x), len(val_x), len(test_x), timesteps, n_features,
    )

    model = build_lstm_model(timesteps, n_features, settings.training.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=settings.training.patience,
            restore_best_weights=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
    ]
    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=settings.training.epochs,
        batch_size=settings.training.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )

    val_lstm = model.predict(val_x, verbose=0).ravel()
    test_lstm = model.predict(test_x, verbose=0).ravel()

    gbr = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        random_state=settings.training.seed,
    )
    gbr.fit(train_x.reshape(len(train_x), -1), train_y)
    val_gbr = gbr.predict(val_x.reshape(len(val_x), -1))
    test_gbr = gbr.predict(test_x.reshape(len(test_x), -1))

    alpha = _compute_blend_weight(val_lstm, val_gbr, val_y)
    test_pred = alpha * test_lstm + (1.0 - alpha) * test_gbr

    test_metrics = compute_metrics(test_y, test_pred)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metrics = TrainingMetrics(
        cell_name=cell_name,
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        mape=test_metrics["mape"],
        trained_at=datetime.now(timezone.utc),
        model_version=version,
    )
    logger.info(
        "Cell %s test metrics: MAE=%.4f RMSE=%.4f MAPE=%.2f%%",
        cell_name, metrics.mae, metrics.rmse, metrics.mape,
    )
    return model, gbr, alpha, metrics


def run_training(settings: Settings, tensors_dir: Path) -> list[TrainingMetrics]:
    """Train all cells found in tensors_dir and save to the registry.

    Args:
        settings: Application settings.
        tensors_dir: Output of run_sequencing().

    Returns:
        List of TrainingMetrics, one per cell.
    """
    from predictagent.registry.model_registry import ModelRegistry

    registry = ModelRegistry(settings.registry.model_dir)
    all_metrics: list[TrainingMetrics] = []

    for cell_dir in sorted(tensors_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        meta_path = cell_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cell_name = meta.get("cell_name", cell_dir.name)

        try:
            lstm_model, gbr_model, alpha, metrics = train_cell(cell_name, cell_dir, settings)
        except TrainingError as exc:
            logger.error("Training failed for %s: %s", cell_name, exc)
            continue

        feature_scaler = joblib.load(cell_dir / "feature_scaler.joblib")
        registry.save(
            cell_name=cell_name,
            lstm_model=lstm_model,
            gbr_model=gbr_model,
            feature_scaler=feature_scaler,
            metrics=metrics,
            alpha=alpha,
            feature_columns=meta.get("feature_columns", []),
        )
        all_metrics.append(metrics)

    return all_metrics
