from pathlib import Path
import pytest
import yaml


@pytest.mark.unit
def test_load_settings_from_yaml(tmp_path):
    from predictagent.config import load_settings

    cfg_data = {
        "data": {
            "raw_path": "viavi-dataset/raw/CellReports.csv",
            "processed_dir": "viavi-dataset/processed/",
            "site_filter": "S1/",
            "rollup_minutes": 15,
        },
        "features": {
            "target_column": "PRB.Util.DL",
            "feature_columns": ["PRB.Util.UL", "RRC.ConnMean"],
            "lookback_steps": 48,
            "forecast_horizon": 1,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "scale_target": False,
        },
        "training": {
            "batch_size": 256,
            "epochs": 30,
            "learning_rate": 0.001,
            "patience": 5,
            "seed": 42,
        },
        "registry": {"model_dir": "models/"},
        "api": {"host": "0.0.0.0", "port": 8000},
    }
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))

    settings = load_settings(cfg_path)
    assert settings.features.target_column == "PRB.Util.DL"
    assert settings.features.lookback_steps == 48
    assert settings.training.seed == 42


@pytest.mark.unit
def test_missing_required_section_raises(tmp_path):
    from predictagent.config import load_settings
    import pydantic

    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.dump({"data": {}}))  # missing features, training etc.

    with pytest.raises((pydantic.ValidationError, KeyError)):
        load_settings(cfg_path)


@pytest.mark.unit
def test_val_test_fraction_sum_validates(tmp_path):
    from predictagent.config import load_settings
    import pydantic

    cfg_data = {
        "data": {
            "raw_path": "x.csv",
            "processed_dir": "out/",
            "site_filter": "S1/",
            "rollup_minutes": 15,
        },
        "features": {
            "target_column": "PRB.Util.DL",
            "feature_columns": [],
            "lookback_steps": 48,
            "forecast_horizon": 1,
            "val_fraction": 0.6,
            "test_fraction": 0.6,  # sum > 1
            "scale_target": False,
        },
        "training": {"batch_size": 256, "epochs": 30, "learning_rate": 0.001, "patience": 5, "seed": 42},
        "registry": {"model_dir": "models/"},
        "api": {"host": "0.0.0.0", "port": 8000},
    }
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))

    with pytest.raises((pydantic.ValidationError, ValueError)):
        load_settings(cfg_path)
