"""Feature engineering: rolling statistics, EMAs, and lag features."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from predictagent.config import Settings
from predictagent.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

# Patterns for auto-generating features from column name conventions
_ROLL_MEAN_RE = re.compile(r"^(.+)_roll_mean_(\d+)$")
_ROLL_STD_RE = re.compile(r"^(.+)_roll_std_(\d+)$")
_EMA_RE = re.compile(r"^(.+)_ema_(\d+)$")
_LAG_RE = re.compile(r"^(.+)_lag_(\d+)$")


def _add_derived_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compute a single derived column in-place and return the DataFrame.

    Args:
        df: Input DataFrame.
        col: Column name encoding the transformation, e.g. "PRB.Util.DL_roll_mean_4".

    Returns:
        DataFrame with the new column added.

    Raises:
        FeatureEngineeringError: If the column name pattern is unrecognised
            or the source column does not exist.
    """
    if col in df.columns:
        return df

    for pattern, transform in [
        (_ROLL_MEAN_RE, "roll_mean"),
        (_ROLL_STD_RE, "roll_std"),
        (_EMA_RE, "ema"),
        (_LAG_RE, "lag"),
    ]:
        m = pattern.match(col)
        if not m:
            continue
        src, n = m.group(1), int(m.group(2))
        if src not in df.columns:
            raise FeatureEngineeringError(
                f"Source column '{src}' required for '{col}' not found in DataFrame"
            )
        if transform == "roll_mean":
            df[col] = df[src].rolling(n).mean()
        elif transform == "roll_std":
            df[col] = df[src].rolling(n).std()
        elif transform == "ema":
            df[col] = df[src].ewm(span=n, adjust=False).mean()
        elif transform == "lag":
            df[col] = df[src].shift(n)
        return df

    raise FeatureEngineeringError(
        f"Column '{col}' is not a base column and does not match any known "
        "pattern (roll_mean_N, roll_std_N, ema_N, lag_N)"
    )


def engineer_features(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Add all requested feature columns and drop rows with NaN values.

    All rolling/lag/EMA features are computed per the column naming convention.
    Base columns (e.g. PRB.Util.UL) must already exist in df.
    Derived columns (e.g. PRB.Util.DL_roll_mean_4) are computed on demand.

    Args:
        df: Cell-level DataFrame sorted by timestamp (single cell).
        target_column: Name of the prediction target column.
        feature_columns: List of feature column names to produce.

    Returns:
        DataFrame with all feature columns present, NaN rows dropped.

    Raises:
        FeatureEngineeringError: If a column cannot be produced.
    """
    df = df.copy()
    for col in feature_columns:
        df = _add_derived_column(df, col)

    required_cols = feature_columns + [target_column]
    before = len(df)
    df = df.dropna(subset=required_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN after feature engineering", dropped)

    return df


def run_feature_engineering(settings: Settings, processed_csv: Path) -> Path:
    """Apply feature engineering to the processed CSV and write per-cell CSVs.

    Args:
        settings: Application settings (feature_columns, target_column, processed_dir).
        processed_csv: Path to the rolled-up CSV written by run_ingestion().

    Returns:
        Path to the directory containing per-cell feature CSVs.

    Raises:
        FeatureEngineeringError: If feature computation fails.
    """
    df = pd.read_csv(processed_csv)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    cell_dir = settings.data.processed_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    cells = sorted(df["Viavi.Cell.Name"].unique())
    for cell in cells:
        cell_df = df[df["Viavi.Cell.Name"] == cell].copy()
        cell_df = cell_df.sort_values("timestamp").reset_index(drop=True)
        try:
            cell_feat = engineer_features(
                cell_df,
                settings.features.target_column,
                settings.features.feature_columns,
            )
        except FeatureEngineeringError:
            logger.warning("Skipping cell %s: feature engineering failed", cell)
            continue

        safe_name = cell.replace("/", "_")
        out_path = cell_dir / f"{safe_name}_features.csv"
        cell_feat.to_csv(out_path, index=False)
        logger.info("Features written for %s → %s (%d rows)", cell, out_path, len(cell_feat))

    return cell_dir
