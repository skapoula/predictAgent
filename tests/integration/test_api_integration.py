import pytest
import numpy as np


@pytest.mark.integration
def test_api_forecast_after_training(minimal_settings):
    """POST /forecast returns 200 after a real training run."""
    from predictagent.pipeline.ingestor import run_ingestion
    from predictagent.pipeline.features import run_feature_engineering
    from predictagent.pipeline.sequencer import run_sequencing
    from predictagent.pipeline.trainer import run_training
    from predictagent.api.app import create_app
    from fastapi.testclient import TestClient

    # Run training pipeline
    processed_csv = run_ingestion(minimal_settings)
    cell_features_dir = run_feature_engineering(minimal_settings, processed_csv)
    tensors_dir = run_sequencing(minimal_settings, cell_features_dir)
    run_training(minimal_settings, tensors_dir)

    client = TestClient(create_app(minimal_settings))

    # Build a valid payload: enough rows for feature engineering + lookback_steps
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
            "rrc_conn_mean": 30.0,
            "drb_ue_thp_dl": 0.2,
            "pee_avg_power": 150.0,
        }
        for i in range(n_rows)
    ]
    resp = client.post("/forecast", json={"cell_name": "S1/B2/C1", "rows": rows})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["predicted_prb_util_dl"] <= 1.5
