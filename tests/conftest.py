"""Shared pytest fixtures for all test tiers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SAMPLE_CELL = "S1/B2/C1"
SAMPLE_TS_START = 1672502400  # 2023-01-01 00:00:00 UTC
INTERVAL_SECONDS = 900        # 15 minutes


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """60-row single-cell CSV at 15-min resolution — enough for lookback_steps=48."""
    n = 60
    rng = np.random.default_rng(42)
    timestamps = [SAMPLE_TS_START + i * INTERVAL_SECONDS for i in range(n)]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "Viavi.Cell.Name": [SAMPLE_CELL] * n,
            "RRU.PrbUsedDl": rng.uniform(50, 90, n),
            "RRU.PrbAvailDl": [100.0] * n,
            "RRU.PrbUsedUl": rng.uniform(30, 70, n),
            "RRU.PrbAvailUl": [100.0] * n,
            "RRC.ConnMean": rng.uniform(10, 50, n),
            "DRB.UEThpDl": rng.uniform(0.1, 0.5, n),
            "PEE.AvgPower": rng.uniform(100, 200, n),
        }
    )
    path = tmp_path / "sample_cell_reports.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def minimal_settings(tmp_path: Path, sample_csv: Path):
    """Settings instance pointing at tmp fixture data."""
    from predictagent.config import Settings, DataConfig, FeaturesConfig, TrainingConfig, RegistryConfig, ApiConfig

    return Settings(
        data=DataConfig(
            raw_path=sample_csv,
            processed_dir=tmp_path / "processed",
            site_filter="S1/",
            rollup_minutes=15,
        ),
        features=FeaturesConfig(
            target_column="PRB.Util.DL",
            feature_columns=[
                "PRB.Util.UL",
                "RRC.ConnMean",
                "DRB.UEThpDl",
                "PEE.AvgPower",
                "PRB.Util.DL_roll_mean_4",
                "PRB.Util.DL_roll_mean_8",
                "PRB.Util.DL_ema_4",
                "PRB.Util.DL_lag_5",
            ],
            lookback_steps=8,
            forecast_horizon=1,
            val_fraction=0.2,
            test_fraction=0.2,
            scale_target=False,
        ),
        training=TrainingConfig(
            batch_size=16,
            epochs=2,
            learning_rate=0.001,
            patience=2,
            seed=42,
        ),
        registry=RegistryConfig(model_dir=tmp_path / "models"),
        api=ApiConfig(host="127.0.0.1", port=8000),
    )
