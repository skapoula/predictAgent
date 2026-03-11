"""Pydantic schemas for API I/O and internal data contracts."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator


class TelemetryRow(BaseModel):
    """One 15-minute telemetry sample from a cell."""

    timestamp: int
    cell_name: str
    prb_used_dl: float
    prb_avail_dl: float
    prb_used_ul: float
    prb_avail_ul: float
    rrc_conn_mean: float | None = None
    drb_ue_thp_dl: float | None = None
    pee_avg_power: float | None = None


class ForecastRequest(BaseModel):
    """Inference request: cell identifier + recent telemetry window."""

    cell_name: str
    rows: list[TelemetryRow]

    @field_validator("rows")
    @classmethod
    def rows_not_empty(cls, v: list[TelemetryRow]) -> list[TelemetryRow]:
        if not v:
            raise ValueError("rows must contain at least one TelemetryRow")
        return v


class ForecastResponse(BaseModel):
    """Inference response with predicted PRB DL utilisation."""

    cell_name: str
    forecast_horizon_minutes: int
    predicted_prb_util_dl: float
    model_version: str


class TrainingMetrics(BaseModel):
    """Per-cell training outcome stored in the model registry."""

    cell_name: str
    mae: float
    rmse: float
    mape: float
    trained_at: datetime
    model_version: str
