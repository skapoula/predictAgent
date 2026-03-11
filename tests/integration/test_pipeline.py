import pytest
import numpy as np


@pytest.mark.integration
def test_full_training_pipeline(minimal_settings, tmp_path):
    """ingest → features → sequence → train produces at least one trained model."""
    from predictagent.pipeline.ingestor import run_ingestion
    from predictagent.pipeline.features import run_feature_engineering
    from predictagent.pipeline.sequencer import run_sequencing
    from predictagent.pipeline.trainer import run_training

    processed_csv = run_ingestion(minimal_settings)
    assert processed_csv.exists()

    cell_features_dir = run_feature_engineering(minimal_settings, processed_csv)
    assert any(cell_features_dir.glob("*_features.csv"))

    tensors_dir = run_sequencing(minimal_settings, cell_features_dir)
    assert any(tensors_dir.iterdir())

    all_metrics = run_training(minimal_settings, tensors_dir)
    assert len(all_metrics) >= 1
    for m in all_metrics:
        assert m.mae >= 0
        assert m.rmse >= 0


@pytest.mark.integration
def test_registry_save_load_roundtrip(minimal_settings, tmp_path):
    """A model saved to the registry can be loaded and used to predict."""
    from predictagent.pipeline.ingestor import run_ingestion
    from predictagent.pipeline.features import run_feature_engineering
    from predictagent.pipeline.sequencer import run_sequencing
    from predictagent.pipeline.trainer import run_training, predict
    from predictagent.registry.model_registry import ModelRegistry
    import numpy as np

    processed_csv = run_ingestion(minimal_settings)
    cell_features_dir = run_feature_engineering(minimal_settings, processed_csv)
    tensors_dir = run_sequencing(minimal_settings, cell_features_dir)
    run_training(minimal_settings, tensors_dir)

    registry = ModelRegistry(minimal_settings.registry.model_dir)
    artifacts = registry.load("S1/B2/C1")

    n_feat = len(minimal_settings.features.feature_columns)
    X = np.zeros((1, minimal_settings.features.lookback_steps, n_feat), dtype=np.float32)
    result = predict(artifacts["lstm_model"], artifacts["gbr_model"], X, artifacts["alpha"])
    assert isinstance(result, float)
