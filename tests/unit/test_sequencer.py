import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def feature_df():
    """60-row single-cell feature DataFrame at 15-min resolution."""
    rng = np.random.default_rng(99)
    n = 60
    base_ts = 1672502400
    step = 900
    return pd.DataFrame(
        {
            "timestamp": [base_ts + i * step for i in range(n)],
            "timestamp_dt": pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC"),
            "Viavi.Cell.Name": ["S1/B2/C1"] * n,
            "PRB.Util.DL": rng.uniform(0.3, 0.9, n).astype(np.float32),
            "PRB.Util.UL": rng.uniform(0.1, 0.6, n).astype(np.float32),
            "RRC.ConnMean": rng.uniform(10, 50, n).astype(np.float32),
        }
    )


FEAT_COLS = ["PRB.Util.UL", "RRC.ConnMean"]
TARGET = "PRB.Util.DL"
LOOKBACK = 8
HORIZON = 1


@pytest.mark.unit
def test_build_sequences_output_shape(feature_df):
    from predictagent.pipeline.sequencer import build_sequences

    samples = build_sequences(feature_df, TARGET, FEAT_COLS, LOOKBACK, HORIZON)
    assert len(samples) > 0
    x, y, _ = samples[0]
    assert x.shape == (LOOKBACK, len(FEAT_COLS))
    assert isinstance(y, float)


@pytest.mark.unit
def test_build_sequences_temporal_order(feature_df):
    from predictagent.pipeline.sequencer import build_sequences

    samples = build_sequences(feature_df, TARGET, FEAT_COLS, LOOKBACK, HORIZON)
    timestamps = [meta["target_timestamp"] for _, _, meta in samples]
    assert timestamps == sorted(timestamps)


@pytest.mark.unit
def test_split_sequences_chronological_and_sizes(feature_df):
    from predictagent.pipeline.sequencer import build_sequences, split_sequences

    samples = build_sequences(feature_df, TARGET, FEAT_COLS, LOOKBACK, HORIZON)
    splits = split_sequences(samples, val_fraction=0.2, test_fraction=0.2)
    total = sum(len(splits[k]["y"]) for k in ("train", "val", "test"))
    assert total == len(samples)
    # train targets must come before val, val before test
    assert splits["train"]["meta"][-1]["target_timestamp"] <= splits["val"]["meta"][0]["target_timestamp"]


@pytest.mark.unit
def test_scale_features_fitted_on_train_only(feature_df):
    from predictagent.pipeline.sequencer import build_sequences, split_sequences, scale_splits

    samples = build_sequences(feature_df, TARGET, FEAT_COLS, LOOKBACK, HORIZON)
    splits = split_sequences(samples, val_fraction=0.2, test_fraction=0.2)
    scaled, scaler = scale_splits(splits, scale_target=False)

    # Scaler was fitted on train — train features should be ~N(0,1)
    train_x = scaled["train"]["X"]
    flat = train_x.reshape(-1, train_x.shape[-1])
    np.testing.assert_allclose(flat.mean(axis=0), np.zeros(len(FEAT_COLS)), atol=0.1)


@pytest.mark.unit
def test_gap_detection_splits_segments():
    """A gap larger than the step size should produce separate segments."""
    from predictagent.pipeline.sequencer import build_sequences

    base_ts = 1672502400
    step = 900
    n = 20
    # Introduce a 2-step gap at position 10
    timestamps = [base_ts + i * step for i in range(10)] + [
        base_ts + (10 + 2) * step + i * step for i in range(10)
    ]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "timestamp_dt": pd.to_datetime(timestamps, unit="s", utc=True),
            "Viavi.Cell.Name": ["S1/B2/C1"] * n,
            "PRB.Util.DL": rng.uniform(0.3, 0.9, n).astype(np.float32),
            "PRB.Util.UL": rng.uniform(0.1, 0.6, n).astype(np.float32),
        }
    )
    # With lookback=8 and the gap, no sequence should span the gap
    samples = build_sequences(df, "PRB.Util.DL", ["PRB.Util.UL"], lookback=8, horizon=1)
    for _, _, meta in samples:
        gap_start = base_ts + 9 * step
        gap_end = base_ts + 12 * step
        assert not (meta["input_start_ts"] <= gap_start and meta["target_timestamp"] >= gap_end)
