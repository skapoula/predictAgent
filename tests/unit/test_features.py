import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cell_df():
    """Single-cell DataFrame with PRB.Util.DL and supporting columns, 60 rows."""
    n = 60
    rng = np.random.default_rng(7)
    base_ts = 1672502400
    return pd.DataFrame(
        {
            "timestamp": [base_ts + i * 900 for i in range(n)],
            "timestamp_dt": pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC"),
            "Viavi.Cell.Name": ["S1/B2/C1"] * n,
            "SiteSector": ["S1/Sect1"] * n,
            "PRB.Util.DL": rng.uniform(0.3, 0.9, n),
            "PRB.Util.UL": rng.uniform(0.1, 0.6, n),
            "RRC.ConnMean": rng.uniform(10, 50, n),
            "DRB.UEThpDl": rng.uniform(0.1, 0.5, n),
            "PEE.AvgPower": rng.uniform(100, 200, n),
        }
    )


FEATURE_COLS = [
    "PRB.Util.UL",
    "RRC.ConnMean",
    "DRB.UEThpDl",
    "PEE.AvgPower",
    "PRB.Util.DL_roll_mean_4",
    "PRB.Util.DL_roll_mean_8",
    "PRB.Util.DL_ema_4",
    "PRB.Util.DL_lag_5",
]


@pytest.mark.unit
def test_engineer_features_returns_all_requested_columns(cell_df):
    from predictagent.pipeline.features import engineer_features

    result = engineer_features(cell_df, "PRB.Util.DL", FEATURE_COLS)
    for col in FEATURE_COLS:
        assert col in result.columns, f"Missing feature column: {col}"


@pytest.mark.unit
def test_engineer_features_drops_nan_rows(cell_df):
    from predictagent.pipeline.features import engineer_features

    result = engineer_features(cell_df, "PRB.Util.DL", FEATURE_COLS)
    assert result[FEATURE_COLS + ["PRB.Util.DL"]].isna().sum().sum() == 0


@pytest.mark.unit
def test_roll_mean_values_are_correct(cell_df):
    from predictagent.pipeline.features import engineer_features

    result = engineer_features(cell_df, "PRB.Util.DL", ["PRB.Util.DL_roll_mean_4"])
    # After dropna, the roll_mean_4 at each row should equal the mean of the prior 4 values
    expected = cell_df["PRB.Util.DL"].rolling(4).mean().dropna().values
    np.testing.assert_allclose(
        result["PRB.Util.DL_roll_mean_4"].values, expected, rtol=1e-5
    )


@pytest.mark.unit
def test_lag_feature_shifts_correctly(cell_df):
    from predictagent.pipeline.features import engineer_features

    result = engineer_features(cell_df, "PRB.Util.DL", ["PRB.Util.DL_lag_5"])
    original = cell_df["PRB.Util.DL"].values
    result_reset = result.reset_index(drop=True)
    first_lag_val = result_reset["PRB.Util.DL_lag_5"].iloc[0]
    # find the matching original index
    orig_idx = result.index[0] - 5
    if orig_idx >= 0:
        assert abs(first_lag_val - original[orig_idx]) < 1e-6


@pytest.mark.unit
def test_unknown_feature_column_raises(cell_df):
    from predictagent.pipeline.features import engineer_features
    from predictagent.exceptions import FeatureEngineeringError

    with pytest.raises(FeatureEngineeringError, match="not_a_feature"):
        engineer_features(cell_df, "PRB.Util.DL", ["not_a_feature"])
