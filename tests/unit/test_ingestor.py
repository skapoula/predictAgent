import pandas as pd
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def raw_df():
    """Minimal valid raw DataFrame (1-minute resolution, 2 cells)."""
    rng = np.random.default_rng(0)
    n = 30
    base_ts = 1672502400
    rows = []
    for cell in ["S1/B2/C1", "S1/B13/C1"]:
        for i in range(n):
            rows.append(
                {
                    "timestamp": base_ts + i * 60,
                    "Viavi.Cell.Name": cell,
                    "RRU.PrbUsedDl": rng.uniform(50, 90),
                    "RRU.PrbAvailDl": 100.0,
                    "RRU.PrbUsedUl": rng.uniform(30, 70),
                    "RRU.PrbAvailUl": 100.0,
                    "RRC.ConnMean": rng.uniform(10, 50),
                    "DRB.UEThpDl": rng.uniform(0.1, 0.5),
                    "PEE.AvgPower": rng.uniform(100, 200),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_validate_schema_passes_on_valid_df(raw_df):
    from predictagent.pipeline.ingestor import validate_schema
    validate_schema(raw_df)  # must not raise


@pytest.mark.unit
def test_validate_schema_raises_on_missing_column(raw_df):
    from predictagent.pipeline.ingestor import validate_schema
    from predictagent.exceptions import SchemaValidationError

    bad_df = raw_df.drop(columns=["RRU.PrbUsedDl"])
    with pytest.raises(SchemaValidationError, match="RRU.PrbUsedDl"):
        validate_schema(bad_df)


@pytest.mark.unit
def test_compute_prb_utilisation_range(raw_df):
    from predictagent.pipeline.ingestor import compute_prb_utilisation

    result = compute_prb_utilisation(raw_df)
    assert "PRB.Util.DL" in result.columns
    assert "PRB.Util.UL" in result.columns
    assert result["PRB.Util.DL"].between(0, 1).all()


@pytest.mark.unit
def test_derive_site_sector_parses_correctly():
    from predictagent.pipeline.ingestor import derive_site_sector

    assert derive_site_sector("S1/B2/C1") == "S1/Sect1"
    assert derive_site_sector("S7/N77/C3") == "S7/Sect3"
    assert derive_site_sector("bad") is None


@pytest.mark.unit
def test_filter_by_site_keeps_prefix(raw_df):
    from predictagent.pipeline.ingestor import filter_by_site

    filtered = filter_by_site(raw_df, "S1/")
    assert filtered["Viavi.Cell.Name"].str.startswith("S1/").all()


@pytest.mark.unit
def test_rollup_reduces_row_count(raw_df):
    from predictagent.pipeline.ingestor import rollup_to_interval

    # add SiteSector needed by rollup
    raw_df = raw_df.copy()
    from predictagent.pipeline.ingestor import derive_site_sector
    raw_df["SiteSector"] = raw_df["Viavi.Cell.Name"].apply(derive_site_sector)
    rolled = rollup_to_interval(raw_df, interval_minutes=15)
    assert len(rolled) < len(raw_df)


@pytest.mark.unit
def test_run_ingestion_writes_output(tmp_path, sample_csv):
    from predictagent.pipeline.ingestor import run_ingestion
    from predictagent.config import DataConfig, FeaturesConfig, TrainingConfig, RegistryConfig, ApiConfig, Settings

    settings = Settings(
        data=DataConfig(
            raw_path=sample_csv,
            processed_dir=tmp_path / "processed",
            site_filter="S1/",
            rollup_minutes=15,
        ),
        features=FeaturesConfig(
            target_column="PRB.Util.DL",
            feature_columns=[],
            lookback_steps=48,
            forecast_horizon=1,
            val_fraction=0.2,
            test_fraction=0.2,
            scale_target=False,
        ),
        training=TrainingConfig(batch_size=256, epochs=1, learning_rate=0.001, patience=2, seed=42),
        registry=RegistryConfig(model_dir=tmp_path / "models"),
        api=ApiConfig(host="127.0.0.1", port=8000),
    )
    output_path = run_ingestion(settings)
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert "PRB.Util.DL" in df.columns
    assert "SiteSector" in df.columns
