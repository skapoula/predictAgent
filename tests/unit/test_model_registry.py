import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path


@pytest.fixture
def registry(tmp_path):
    from predictagent.registry.model_registry import ModelRegistry
    return ModelRegistry(tmp_path / "models")


@pytest.fixture
def dummy_artifacts(tmp_path):
    """Minimal model + GBR + scaler for registry tests."""
    import tensorflow as tf
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from predictagent.pipeline.trainer import build_lstm_model
    from predictagent.schemas import TrainingMetrics

    model = build_lstm_model(timesteps=4, n_features=2, lr=1e-3)
    gbr = GradientBoostingRegressor(n_estimators=5, random_state=0)
    rng = np.random.default_rng(0)
    gbr.fit(rng.uniform(0, 1, (20, 8)), rng.uniform(0, 1, 20))
    scaler = StandardScaler()
    scaler.fit(rng.uniform(0, 1, (20, 2)))
    metrics = TrainingMetrics(
        cell_name="S1/B2/C1",
        mae=0.05,
        rmse=0.07,
        mape=8.0,
        trained_at=datetime.now(timezone.utc),
        model_version="test",
    )
    return model, gbr, scaler, metrics


@pytest.mark.unit
def test_save_creates_versioned_directory(registry, dummy_artifacts):
    model, gbr, scaler, metrics = dummy_artifacts
    version = registry.save("S1/B2/C1", model, gbr, scaler, metrics, alpha=0.7, feature_columns=["a"])
    safe = "S1_B2_C1"
    assert (registry.model_dir / safe / version / "model.keras").exists()
    assert (registry.model_dir / safe / version / "gbr.joblib").exists()
    assert (registry.model_dir / safe / version / "feature_scaler.joblib").exists()
    assert (registry.model_dir / safe / version / "metadata.json").exists()


@pytest.mark.unit
def test_load_latest_returns_artifacts(registry, dummy_artifacts):
    model, gbr, scaler, metrics = dummy_artifacts
    registry.save("S1/B2/C1", model, gbr, scaler, metrics, alpha=0.7, feature_columns=["a"])
    loaded = registry.load("S1/B2/C1")
    assert loaded["alpha"] == pytest.approx(0.7)
    assert loaded["feature_columns"] == ["a"]


@pytest.mark.unit
def test_load_unknown_cell_raises(registry):
    from predictagent.exceptions import ModelNotFoundError
    with pytest.raises(ModelNotFoundError):
        registry.load("unknown/cell")


@pytest.mark.unit
def test_list_versions(registry, dummy_artifacts):
    model, gbr, scaler, metrics = dummy_artifacts
    registry.save("S1/B2/C1", model, gbr, scaler, metrics, alpha=0.7, feature_columns=[])
    registry.save("S1/B2/C1", model, gbr, scaler, metrics, alpha=0.6, feature_columns=[])
    versions = registry.list_versions("S1/B2/C1")
    assert len(versions) == 2
