"""Sliding-window sequence builder, train/val/test splitter, and feature scaler."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from predictagent.config import Settings
from predictagent.exceptions import SequencerError

logger = logging.getLogger(__name__)

# Type alias: one sample is (X_window, y_target, meta_dict)
Sample = tuple[np.ndarray, float, dict[str, Any]]


def _infer_step(timestamps: np.ndarray) -> float:
    """Return median inter-sample step in seconds."""
    diffs = np.diff(timestamps)
    return float(np.median(diffs)) if len(diffs) > 0 else np.nan


def build_sequences(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    lookback: int,
    horizon: int,
) -> list[Sample]:
    """Build sliding-window samples from a single-cell feature DataFrame.

    Sequences that span timestamp gaps (> inferred step) are discarded.

    Args:
        df: Single-cell feature DataFrame sorted by timestamp.
        target_column: Column to predict.
        feature_columns: Ordered list of input feature columns.
        lookback: Number of time steps in the input window.
        horizon: Number of steps ahead to forecast.

    Returns:
        List of (X_window, y_target, meta) tuples sorted by target_timestamp.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].to_numpy()
    step = _infer_step(timestamps)
    if not np.isfinite(step) or step <= 0:
        logger.warning("Could not infer step size — returning empty sequence list")
        return []

    features = df[feature_columns].to_numpy(dtype=np.float32)
    targets = df[target_column].to_numpy(dtype=np.float32)
    dt_list = df["timestamp_dt"].tolist()

    total = len(df)
    window_span = lookback + horizon
    if total < window_span:
        return []

    samples: list[Sample] = []
    for start in range(total - window_span + 1):
        end = start + lookback
        target_idx = end + horizon - 1

        input_ts = timestamps[start:end]
        target_ts = timestamps[end - 1: target_idx + 1]

        if np.any(np.isnan(features[start:end])) or np.isnan(targets[target_idx]):
            continue
        if not np.all(np.diff(input_ts) == step):
            continue
        if target_ts[-1] - input_ts[-1] != step * horizon:
            continue

        meta: dict[str, Any] = {
            "entity": df.loc[0, "Viavi.Cell.Name"],
            "input_start_ts": int(input_ts[0]),
            "input_end_ts": int(input_ts[-1]),
            "target_timestamp": int(target_ts[-1]),
            "input_start_dt": str(dt_list[start]),
            "target_dt": str(dt_list[target_idx]),
        }
        samples.append((features[start:end].copy(), float(targets[target_idx]), meta))

    return samples


def split_sequences(
    samples: list[Sample],
    val_fraction: float,
    test_fraction: float,
) -> dict[str, dict[str, Any]]:
    """Chronological train/val/test split.

    Args:
        samples: List of (X, y, meta) tuples.
        val_fraction: Fraction of data for validation.
        test_fraction: Fraction of data for test.

    Returns:
        Dict with keys "train", "val", "test"; each maps to
        {"X": ndarray, "y": ndarray, "meta": list[dict]}.

    Raises:
        SequencerError: If no samples are provided or splits are degenerate.
    """
    if not samples:
        raise SequencerError("Cannot split an empty sample list")

    order = np.argsort([m["target_timestamp"] for _, _, m in samples])
    sorted_samples = [samples[i] for i in order]

    total = len(sorted_samples)
    train_end = int(total * (1 - val_fraction - test_fraction))
    val_end = int(total * (1 - test_fraction))

    if train_end <= 0 or val_end <= train_end or val_end >= total:
        raise SequencerError(
            f"Dataset of {total} samples is too small for "
            f"val_fraction={val_fraction}, test_fraction={test_fraction}"
        )

    def _stack(subset: list[Sample]) -> dict[str, Any]:
        xs, ys, metas = zip(*subset)
        return {
            "X": np.stack(xs, axis=0),
            "y": np.array(ys, dtype=np.float32),
            "meta": list(metas),
        }

    return {
        "train": _stack(sorted_samples[:train_end]),
        "val": _stack(sorted_samples[train_end:val_end]),
        "test": _stack(sorted_samples[val_end:]),
    }


def scale_splits(
    splits: dict[str, dict[str, Any]],
    scale_target: bool,
) -> tuple[dict[str, dict[str, Any]], StandardScaler]:
    """Fit StandardScaler on train features only; transform all splits.

    Args:
        splits: Output of split_sequences().
        scale_target: If True, also scale y values.

    Returns:
        Tuple of (scaled_splits, feature_scaler). scaled_splits has the same
        structure as splits but with scaled X (and optionally y) arrays.
    """
    n_features = splits["train"]["X"].shape[-1]
    feature_scaler = StandardScaler()
    feature_scaler.fit(splits["train"]["X"].reshape(-1, n_features))

    def _transform_x(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        return feature_scaler.transform(arr.reshape(-1, n_features)).reshape(shape)

    target_scaler: StandardScaler | None = None
    if scale_target:
        target_scaler = StandardScaler()
        target_scaler.fit(splits["train"]["y"].reshape(-1, 1))

    scaled: dict[str, dict[str, Any]] = {}
    for split_name, data in splits.items():
        y = data["y"]
        if scale_target and target_scaler is not None:
            y = target_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()
        scaled[split_name] = {"X": _transform_x(data["X"]), "y": y, "meta": data["meta"]}

    return scaled, feature_scaler


def run_sequencing(settings: Settings, cell_features_dir: Path) -> Path:
    """Build sequences, split, scale, and save tensors for all cells.

    Scaler is fitted on the train split only — val/test use the train scaler.

    Args:
        settings: Application settings.
        cell_features_dir: Directory of per-cell feature CSVs (from run_feature_engineering).

    Returns:
        Path to the directory containing per-cell joblib tensor files.
    """
    tensors_dir = settings.data.processed_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(cell_features_dir.glob("*_features.csv")):
        cell_df = pd.read_csv(csv_path)
        cell_df["timestamp_dt"] = pd.to_datetime(cell_df["timestamp"], unit="s", utc=True)
        cell_name = cell_df["Viavi.Cell.Name"].iloc[0]
        safe = cell_name.replace("/", "_")

        samples = build_sequences(
            cell_df,
            settings.features.target_column,
            settings.features.feature_columns,
            settings.features.lookback_steps,
            settings.features.forecast_horizon,
        )
        if not samples:
            logger.warning("No sequences for cell %s — skipping", cell_name)
            continue

        try:
            splits = split_sequences(
                samples, settings.features.val_fraction, settings.features.test_fraction
            )
        except SequencerError as exc:
            logger.warning("Skipping cell %s: %s", cell_name, exc)
            continue

        scaled, feature_scaler = scale_splits(splits, settings.features.scale_target)

        cell_tensor_dir = tensors_dir / safe
        cell_tensor_dir.mkdir(parents=True, exist_ok=True)

        for split_name, data in scaled.items():
            joblib.dump(data["X"], cell_tensor_dir / f"{split_name}_inputs.joblib")
            joblib.dump(data["y"], cell_tensor_dir / f"{split_name}_targets.joblib")
            joblib.dump(data["meta"], cell_tensor_dir / f"{split_name}_meta.joblib")
        joblib.dump(feature_scaler, cell_tensor_dir / "feature_scaler.joblib")

        metadata = {
            "cell_name": cell_name,
            "target_column": settings.features.target_column,
            "feature_columns": settings.features.feature_columns,
            "lookback_steps": settings.features.lookback_steps,
            "forecast_horizon": settings.features.forecast_horizon,
            "scale_target": settings.features.scale_target,
            "train_samples": int(scaled["train"]["X"].shape[0]),
            "val_samples": int(scaled["val"]["X"].shape[0]),
            "test_samples": int(scaled["test"]["X"].shape[0]),
        }
        (cell_tensor_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        logger.info(
            "Sequencer: %s → train=%d, val=%d, test=%d",
            cell_name,
            metadata["train_samples"],
            metadata["val_samples"],
            metadata["test_samples"],
        )

    return tensors_dir
