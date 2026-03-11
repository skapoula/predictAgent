"""Filesystem-based model registry with versioned directories."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from predictagent.exceptions import ModelNotFoundError, RegistryError
from predictagent.schemas import TrainingMetrics

logger = logging.getLogger(__name__)

_LATEST_FILE = "latest.json"


def _safe_name(cell_name: str) -> str:
    """Convert cell name to a filesystem-safe directory name."""
    return cell_name.replace("/", "_")


class ModelRegistry:
    """Save and load versioned LSTM+GBR ensemble models from the filesystem.

    Directory layout::

        model_dir/
        └── S1_B2_C1/
            ├── 20260311_091500/
            │   ├── model.keras
            │   ├── gbr.joblib
            │   ├── feature_scaler.joblib
            │   └── metadata.json
            └── latest.json          ← {"version": "20260311_091500"}
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        cell_name: str,
        lstm_model: tf.keras.Model,
        gbr_model: GradientBoostingRegressor,
        feature_scaler: StandardScaler,
        metrics: TrainingMetrics,
        alpha: float,
        feature_columns: list[str],
    ) -> str:
        """Persist model artefacts and return the version string.

        Args:
            cell_name: Viavi cell identifier.
            lstm_model: Trained Keras model.
            gbr_model: Trained GradientBoostingRegressor.
            feature_scaler: Fitted StandardScaler.
            metrics: Training metrics to embed in metadata.
            alpha: LSTM blend weight.
            feature_columns: Ordered list of feature column names.

        Returns:
            Version string (UTC timestamp).
        """
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe = _safe_name(cell_name)
        version_dir = self.model_dir / safe / version
        version_dir.mkdir(parents=True, exist_ok=True)

        try:
            lstm_model.save(version_dir / "model.keras")
            joblib.dump(gbr_model, version_dir / "gbr.joblib")
            joblib.dump(feature_scaler, version_dir / "feature_scaler.joblib")

            metadata = {
                "cell_name": cell_name,
                "version": version,
                "alpha": alpha,
                "feature_columns": feature_columns,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "mape": metrics.mape,
                "trained_at": metrics.trained_at.isoformat(),
            }
            (version_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            # Update latest pointer
            latest_path = self.model_dir / safe / _LATEST_FILE
            latest_path.write_text(
                json.dumps({"version": version}), encoding="utf-8"
            )
        except Exception as exc:
            raise RegistryError(f"Failed to save model for {cell_name}: {exc}") from exc

        logger.info("Saved model for %s → version %s", cell_name, version)
        return version

    def load(
        self, cell_name: str, version: str = "latest"
    ) -> dict:
        """Load model artefacts for a cell.

        Args:
            cell_name: Viavi cell identifier.
            version: Version string or "latest".

        Returns:
            Dict with keys: lstm_model, gbr_model, feature_scaler, alpha,
            feature_columns, metadata.

        Raises:
            ModelNotFoundError: If no model exists for cell_name or version.
        """
        safe = _safe_name(cell_name)
        cell_dir = self.model_dir / safe

        if not cell_dir.exists():
            raise ModelNotFoundError(f"No model found for cell '{cell_name}'")

        if version == "latest":
            latest_path = cell_dir / _LATEST_FILE
            if not latest_path.exists():
                raise ModelNotFoundError(f"No 'latest' version found for cell '{cell_name}'")
            version = json.loads(latest_path.read_text(encoding="utf-8"))["version"]

        version_dir = cell_dir / version
        if not version_dir.exists():
            raise ModelNotFoundError(
                f"Version '{version}' not found for cell '{cell_name}'"
            )

        try:
            lstm_model = tf.keras.models.load_model(version_dir / "model.keras")
            gbr_model = joblib.load(version_dir / "gbr.joblib")
            feature_scaler = joblib.load(version_dir / "feature_scaler.joblib")
            metadata = json.loads(
                (version_dir / "metadata.json").read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RegistryError(
                f"Failed to load model for {cell_name} v{version}: {exc}"
            ) from exc

        return {
            "lstm_model": lstm_model,
            "gbr_model": gbr_model,
            "feature_scaler": feature_scaler,
            "alpha": metadata["alpha"],
            "feature_columns": metadata["feature_columns"],
            "metadata": metadata,
        }

    def list_versions(self, cell_name: str) -> list[str]:
        """Return all available versions for a cell, sorted ascending.

        Args:
            cell_name: Viavi cell identifier.

        Returns:
            List of version strings.
        """
        safe = _safe_name(cell_name)
        cell_dir = self.model_dir / safe
        if not cell_dir.exists():
            return []
        return sorted(
            d.name for d in cell_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
