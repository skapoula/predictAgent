import numpy as np
import pytest


@pytest.mark.unit
def test_build_model_output_shape():
    from predictagent.pipeline.trainer import build_lstm_model

    model = build_lstm_model(timesteps=8, n_features=3, lr=1e-3)
    import numpy as np
    x = np.random.rand(5, 8, 3).astype(np.float32)
    pred = model.predict(x, verbose=0)
    assert pred.shape == (5, 1)


@pytest.mark.unit
def test_compute_metrics_known_values():
    from predictagent.pipeline.trainer import compute_metrics

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_predict_returns_float(tmp_path):
    """predict() wraps model + GBR blend and returns a scalar float."""
    from predictagent.pipeline.trainer import build_lstm_model, predict
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib

    lookback, n_feat = 4, 2
    model = build_lstm_model(timesteps=lookback, n_features=n_feat, lr=1e-3)
    gbr = GradientBoostingRegressor(n_estimators=10, random_state=42)
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0, 1, (50, lookback, n_feat)).astype(np.float32)
    y_train = rng.uniform(0, 1, 50).astype(np.float32)
    gbr.fit(X_train.reshape(50, -1), y_train)

    X_input = rng.uniform(0, 1, (1, lookback, n_feat)).astype(np.float32)
    result = predict(model, gbr, X_input, alpha=0.7)
    assert isinstance(result, float)
    assert 0.0 <= result <= 2.0  # rough sanity on scale
