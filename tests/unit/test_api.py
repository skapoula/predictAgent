import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def app(minimal_settings):
    from predictagent.api.app import create_app
    return create_app(minimal_settings)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def valid_payload(minimal_settings):
    """Payload with enough rows for feature engineering + lookback window.

    Feature engineering requires extra historical rows (e.g. roll_mean_8 needs 8
    rows before the first valid output). We send lookback_steps * 3 to be safe.
    """
    base_ts = 1672502400
    n_rows = minimal_settings.features.lookback_steps * 3
    rows = [
        {
            "timestamp": base_ts + i * 900,
            "cell_name": "S1/B2/C1",
            "prb_used_dl": 70.0,
            "prb_avail_dl": 100.0,
            "prb_used_ul": 40.0,
            "prb_avail_ul": 100.0,
        }
        for i in range(n_rows)
    ]
    return {"cell_name": "S1/B2/C1", "rows": rows}


@pytest.mark.unit
def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.unit
def test_forecast_too_few_rows_returns_422(client, valid_payload):
    payload = dict(valid_payload)
    payload["rows"] = valid_payload["rows"][:2]  # below lookback_steps
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 422


@pytest.mark.unit
def test_forecast_unknown_cell_returns_404(client, valid_payload, minimal_settings):
    payload = dict(valid_payload)
    payload["cell_name"] = "unknown/cell/99"
    payload["rows"] = [dict(r, cell_name="unknown/cell/99") for r in valid_payload["rows"]]
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 404


@pytest.mark.unit
def test_forecast_valid_request_returns_200(client, valid_payload, minimal_settings, tmp_path):
    """Mock the registry so we don't need a real model."""
    from unittest.mock import patch, MagicMock
    import numpy as np

    mock_artifacts = {
        "lstm_model": MagicMock(**{"predict.return_value": np.array([[0.72]])}),
        "gbr_model": MagicMock(**{"predict.return_value": np.array([0.68])}),
        "feature_scaler": MagicMock(**{"transform.return_value": np.zeros((minimal_settings.features.lookback_steps, 8))}),
        "alpha": 0.7,
        "feature_columns": minimal_settings.features.feature_columns,
        "metadata": {"version": "20260311_091500"},
    }

    with patch("predictagent.api.routers.forecast.get_registry") as mock_reg:
        mock_registry = MagicMock()
        mock_registry.load.return_value = mock_artifacts
        mock_reg.return_value = mock_registry
        resp = client.post("/forecast", json=valid_payload)

    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_prb_util_dl" in body
    assert body["cell_name"] == "S1/B2/C1"
