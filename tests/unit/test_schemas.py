import pytest
from datetime import datetime, timezone


@pytest.mark.unit
def test_forecast_request_valid():
    from predictagent.schemas import ForecastRequest, TelemetryRow

    rows = [
        TelemetryRow(
            timestamp=1672502400 + i * 900,
            cell_name="S1/B2/C1",
            prb_used_dl=70.0,
            prb_avail_dl=100.0,
            prb_used_ul=40.0,
            prb_avail_ul=100.0,
        )
        for i in range(10)
    ]
    req = ForecastRequest(cell_name="S1/B2/C1", rows=rows)
    assert req.cell_name == "S1/B2/C1"
    assert len(req.rows) == 10


@pytest.mark.unit
def test_forecast_request_empty_rows_raises():
    from predictagent.schemas import ForecastRequest
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ForecastRequest(cell_name="S1/B2/C1", rows=[])


@pytest.mark.unit
def test_forecast_response_fields():
    from predictagent.schemas import ForecastResponse

    resp = ForecastResponse(
        cell_name="S1/B2/C1",
        forecast_horizon_minutes=15,
        predicted_prb_util_dl=0.72,
        model_version="20260311_091500",
    )
    assert resp.predicted_prb_util_dl == pytest.approx(0.72)


@pytest.mark.unit
def test_training_metrics_fields():
    from predictagent.schemas import TrainingMetrics

    m = TrainingMetrics(
        cell_name="S1/B2/C1",
        mae=0.05,
        rmse=0.07,
        mape=8.3,
        trained_at=datetime.now(timezone.utc),
        model_version="20260311_091500",
    )
    assert m.mae == pytest.approx(0.05)
