"""FastAPI router for the /forecast and /health endpoints."""
from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from predictagent.config import Settings
from predictagent.exceptions import ModelNotFoundError
from predictagent.pipeline.features import engineer_features
from predictagent.pipeline.trainer import predict
from predictagent.registry.model_registry import ModelRegistry
from predictagent.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_settings: Settings | None = None
_registry_cache: dict = {}


def init_router(settings: Settings) -> None:
    """Inject settings into the router at app startup."""
    global _settings
    _settings = settings


def get_registry() -> ModelRegistry:
    """Return the ModelRegistry singleton."""
    assert _settings is not None, "Router not initialised"
    return ModelRegistry(_settings.registry.model_dir)


@router.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "models_cached": len(_registry_cache)}


@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Predict DL PRB utilisation for a cell.

    The caller must supply at least `lookback_steps` telemetry rows.

    Raises:
        422: Fewer rows than lookback_steps.
        404: No trained model for the requested cell.
        500: Internal inference error.
    """
    assert _settings is not None
    min_rows = _settings.features.lookback_steps
    if len(request.rows) < min_rows:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {min_rows} rows; got {len(request.rows)}",
        )

    try:
        artifacts = get_registry().load(request.cell_name)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for cell '{request.cell_name}'",
        )

    import pandas as pd

    rows_data = [
        {
            "timestamp": r.timestamp,
            "timestamp_dt": pd.Timestamp(r.timestamp, unit="s", tz="UTC"),
            "Viavi.Cell.Name": r.cell_name,
            "RRU.PrbUsedDl": r.prb_used_dl,
            "RRU.PrbAvailDl": r.prb_avail_dl,
            "RRU.PrbUsedUl": r.prb_used_ul,
            "RRU.PrbAvailUl": r.prb_avail_ul,
            "RRC.ConnMean": r.rrc_conn_mean if r.rrc_conn_mean is not None else 0.0,
            "DRB.UEThpDl": r.drb_ue_thp_dl if r.drb_ue_thp_dl is not None else 0.0,
            "PEE.AvgPower": r.pee_avg_power if r.pee_avg_power is not None else 0.0,
        }
        for r in request.rows
    ]
    df = pd.DataFrame(rows_data)

    feature_cols = artifacts["feature_columns"]
    target_col = _settings.features.target_column

    # Compute PRB utilisation ratios from raw columns if not already present
    if target_col not in df.columns:
        avail = df["RRU.PrbAvailDl"].where(df["RRU.PrbAvailDl"] != 0)
        df[target_col] = (df["RRU.PrbUsedDl"] / avail).clip(0, 1)
    if "PRB.Util.UL" not in df.columns:
        avail_ul = df["RRU.PrbAvailUl"].where(df["RRU.PrbAvailUl"] != 0)
        df["PRB.Util.UL"] = (df["RRU.PrbUsedUl"] / avail_ul).clip(0, 1)

    try:
        df_feat = engineer_features(df, target_col, feature_cols)
    except Exception as exc:
        logger.error("Feature engineering failed for %s: %s", request.cell_name, exc)
        raise HTTPException(status_code=500, detail="Feature engineering failed")

    if len(df_feat) < _settings.features.lookback_steps:
        raise HTTPException(
            status_code=422,
            detail=f"After feature engineering only {len(df_feat)} rows remain; need {_settings.features.lookback_steps}",
        )

    # Take the last lookback_steps rows
    window_df = df_feat.tail(_settings.features.lookback_steps).reset_index(drop=True)
    feature_scaler = artifacts["feature_scaler"]
    X_raw = window_df[feature_cols].to_numpy(dtype=np.float32)
    X_scaled = feature_scaler.transform(X_raw).reshape(
        1, _settings.features.lookback_steps, len(feature_cols)
    ).astype(np.float32)

    version = artifacts["metadata"].get("version", "unknown")
    pred = predict(artifacts["lstm_model"], artifacts["gbr_model"], X_scaled, artifacts["alpha"])

    logger.info(
        "Forecast for %s: pred=%.4f, model_version=%s",
        request.cell_name, pred, version,
    )

    return ForecastResponse(
        cell_name=request.cell_name,
        forecast_horizon_minutes=_settings.features.forecast_horizon * _settings.data.rollup_minutes,
        predicted_prb_util_dl=float(pred),
        model_version=version,
    )
