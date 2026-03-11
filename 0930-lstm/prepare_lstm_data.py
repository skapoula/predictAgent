#!/usr/bin/env python3
"""Prepare per-cell sliding-window tensors for LSTM forecasting on CellReports data."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "CellReports_15_S1_top_features.csv"
DEFAULT_CONFIG = BASE_DIR / "data_prep_config.yaml"
DEFAULT_OUTPUT = BASE_DIR / "prepared_lstm"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class Config:
    target_column: str
    feature_columns: List[str]
    lookback_steps: int
    forecast_horizon: int
    val_fraction: float
    test_fraction: float
    scale_target: bool


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    required_keys = [
        "target_column",
        "feature_columns",
        "lookback_steps",
        "forecast_horizon",
        "val_fraction",
        "test_fraction",
    ]
    missing = [key for key in required_keys if key not in raw]
    if missing:
        raise KeyError(f"Config missing keys: {', '.join(missing)}")

    return Config(
        target_column=raw["target_column"],
        feature_columns=list(raw["feature_columns"]),
        lookback_steps=int(raw["lookback_steps"]),
        forecast_horizon=int(raw["forecast_horizon"]),
        val_fraction=float(raw["val_fraction"]),
        test_fraction=float(raw["test_fraction"]),
        scale_target=bool(raw.get("scale_target", False)),
    )


def validate_schema(df: pd.DataFrame, cfg: Config) -> None:
    required = {"timestamp", "Viavi.Cell.Name", cfg.target_column, *cfg.feature_columns}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Input data missing required columns: {', '.join(sorted(missing))}")
    if cfg.lookback_steps <= 0 or cfg.forecast_horizon <= 0:
        raise ValueError("lookback_steps and forecast_horizon must be positive")
    if not (0 < cfg.val_fraction < 1) or not (0 < cfg.test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be between 0 and 1")
    if cfg.val_fraction + cfg.test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "Viavi.Cell.Name"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    if "timestamp_dt" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], errors="coerce", utc=True)
    else:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def sanitize_cell_name(cell_name: str) -> str:
    return cell_name.replace("/", "_")


def sanitize_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def cell_dataset_prefix(dataset_id: str, cell_name: str) -> str:
    return f"{dataset_id}_{sanitize_cell_name(cell_name)}"


@dataclass
class SequenceSample:
    inputs: np.ndarray
    target: float
    meta: dict


def infer_step_median(group: pd.DataFrame) -> float:
    diffs = group["timestamp"].diff().dropna()
    if diffs.empty:
        return np.nan
    return float(diffs.median())


def build_sequences(cell_df: pd.DataFrame, cfg: Config, feature_cols: List[str]) -> List[SequenceSample]:
    samples: List[SequenceSample] = []
    target_col = cfg.target_column

    cell_df = cell_df.sort_values("timestamp").reset_index(drop=True)
    step = infer_step_median(cell_df)
    if not np.isfinite(step) or step <= 0:
        return samples

    timestamps = cell_df["timestamp"].to_numpy()
    features = cell_df[feature_cols].to_numpy(dtype=np.float32)
    targets = cell_df[target_col].to_numpy(dtype=np.float32)
    dt_series = cell_df["timestamp_dt"].to_list()

    total = len(cell_df)
    window_span = cfg.lookback_steps + cfg.forecast_horizon
    if total < window_span:
        return samples

    for start in range(0, total - window_span + 1):
        end = start + cfg.lookback_steps
        target_idx = end + cfg.forecast_horizon - 1

        input_slice = slice(start, end)
        target_slice = slice(end - 1, target_idx + 1)

        input_ts = timestamps[input_slice]
        target_ts = timestamps[target_slice]

        if np.any(np.isnan(features[input_slice])) or np.isnan(targets[target_idx]):
            continue

        if not np.all(np.diff(input_ts) == step):
            continue
        if target_ts[-1] - input_ts[-1] != step * cfg.forecast_horizon:
            continue

        x_window = features[input_slice]
        y_value = float(targets[target_idx])
        meta = {
            "entity": cell_df.loc[0, "Viavi.Cell.Name"],
            "input_start_ts": int(input_ts[0]),
            "input_end_ts": int(input_ts[-1]),
            "target_timestamp": int(target_ts[-1]),
            "input_start_dt": dt_series[input_slice.start],
            "target_dt": dt_series[target_idx],
        }
        samples.append(SequenceSample(inputs=x_window.copy(), target=y_value, meta=meta))

    return samples


def split_sequences(
    samples: Sequence[SequenceSample],
    val_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], List[dict], List[dict]]:
    if not samples:
        raise RuntimeError("No valid sequences generated. Check lookback, horizon, or data continuity.")

    order = np.argsort([s.meta["target_timestamp"] for s in samples])
    sorted_samples = [samples[idx] for idx in order]

    total = len(sorted_samples)
    train_end = int(total * (1 - val_fraction - test_fraction))
    val_end = int(total * (1 - test_fraction))
    if train_end <= 0 or val_end <= train_end or val_end >= total:
        raise ValueError("Dataset too small for requested split ratios.")

    def stack(subset: Sequence[SequenceSample]) -> tuple[np.ndarray, np.ndarray, List[dict]]:
        x = np.stack([s.inputs for s in subset], axis=0)
        y = np.array([s.target for s in subset], dtype=np.float32)
        meta = [s.meta for s in subset]
        return x, y, meta

    train_samples = sorted_samples[:train_end]
    val_samples = sorted_samples[train_end:val_end]
    test_samples = sorted_samples[val_end:]

    train_x, train_y, train_meta = stack(train_samples)
    val_x, val_y, val_meta = stack(val_samples)
    test_x, test_y, test_meta = stack(test_samples)
    return train_x, train_y, val_x, val_y, test_x, test_y, train_meta, val_meta, test_meta


def scale_features(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    scale_target: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler | None]:
    num_features = train_x.shape[-1]
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_x.reshape(-1, num_features))

    def transform_inputs(arr: np.ndarray) -> np.ndarray:
        original_shape = arr.shape
        reshaped = arr.reshape(-1, num_features)
        scaled = feature_scaler.transform(reshaped)
        return scaled.reshape(original_shape)

    train_x_scaled = transform_inputs(train_x)
    val_x_scaled = transform_inputs(val_x)
    test_x_scaled = transform_inputs(test_x)

    target_scaler: StandardScaler | None = None
    if scale_target:
        target_scaler = StandardScaler()
        target_scaler.fit(train_y.reshape(-1, 1))

        def transform_targets(arr: np.ndarray) -> np.ndarray:
            return target_scaler.transform(arr.reshape(-1, 1)).astype(np.float32).ravel()

        train_y_scaled = transform_targets(train_y)
        val_y_scaled = transform_targets(val_y)
        test_y_scaled = transform_targets(test_y)
    else:
        train_y_scaled, val_y_scaled, test_y_scaled = train_y, val_y, test_y

    return (
        train_x_scaled,
        val_x_scaled,
        test_x_scaled,
        train_y_scaled,
        val_y_scaled,
        test_y_scaled,
        feature_scaler,
        target_scaler,
    )


def save_outputs(
    output_dir: Path,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    train_meta: List[dict],
    val_meta: List[dict],
    test_meta: List[dict],
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler | None,
    cfg: Config,
    cell_name: str,
    dataset_id: str,
    feature_columns: List[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(train_x, output_dir / "train_inputs.joblib")
    joblib.dump(val_x, output_dir / "val_inputs.joblib")
    joblib.dump(test_x, output_dir / "test_inputs.joblib")
    joblib.dump(train_y, output_dir / "train_targets.joblib")
    joblib.dump(val_y, output_dir / "val_targets.joblib")
    joblib.dump(test_y, output_dir / "test_targets.joblib")
    joblib.dump(train_meta, output_dir / "train_meta.joblib")
    joblib.dump(val_meta, output_dir / "val_meta.joblib")
    joblib.dump(test_meta, output_dir / "test_meta.joblib")
    joblib.dump(feature_scaler, output_dir / "feature_scaler.joblib")
    if target_scaler is not None:
        joblib.dump(target_scaler, output_dir / "target_scaler.joblib")

    metadata = {
        "target_column": cfg.target_column,
        "feature_columns": feature_columns,
        "lookback_steps": cfg.lookback_steps,
        "forecast_horizon": cfg.forecast_horizon,
        "val_fraction": cfg.val_fraction,
        "test_fraction": cfg.test_fraction,
        "scale_target": cfg.scale_target,
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "test_samples": int(test_x.shape[0]),
        "input_shape": list(train_x.shape[1:]),
        "cell_name": cell_name,
        "dataset": dataset_id,
        "feature_scaler_path": str((output_dir / "feature_scaler.joblib").resolve()),
        "target_scaler_path": str((output_dir / "target_scaler.joblib").resolve()) if target_scaler else None,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CellReports time-series for LSTM forecasting.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to features CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML configuration file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory to store processed tensors (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Optional Viavi.Cell.Name to process (default: process all cells)",
    )
    parser.add_argument(
        "--cells",
        type=str,
        nargs="*",
        help="Optional list of cell names to process; overrides --cell when provided.",
    )
    parser.add_argument(
        "--lookback-steps",
        type=int,
        default=None,
        help="Override lookback window in steps (e.g., 96 for 24h @15min, 1440 for 24h @1min)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[INFO] Loading config from {args.config}")
    cfg = load_config(args.config)
    if args.lookback_steps:
        cfg.lookback_steps = args.lookback_steps
    print(f"[INFO] Loading data from {args.input}")
    df = load_data(args.input)
    validate_schema(df, cfg)

    dataset_id_raw = sanitize_dataset_name(Path(args.input).stem)
    dataset_id = dataset_id_raw.removesuffix("_top_features")
    if args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = BASE_DIR / f"prepared_lstm_{dataset_id}"

    available_cells = sorted(df["Viavi.Cell.Name"].unique())
    if args.cells:
        missing = [cell for cell in args.cells if cell not in available_cells]
        if missing:
            raise ValueError(f"Cells not found in dataset: {', '.join(missing)}")
        cells = args.cells
    elif args.cell:
        if args.cell not in available_cells:
            raise ValueError(f"Cell '{args.cell}' not found in dataset")
        cells = [args.cell]
    else:
        cells = available_cells

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for cell in cells:
        cell_df = df[df["Viavi.Cell.Name"] == cell].copy()
        print(f"[INFO] Processing cell {cell} with {len(cell_df)} rows")
        top_features_path = args.output_dir / cell_dataset_prefix(dataset_id, cell) / f"{dataset_id}_{sanitize_cell_name(cell)}_top_features.csv"
        if not top_features_path.exists():
            # fallback to any top-feature csv in directory
            candidates = list((args.output_dir / cell_dataset_prefix(dataset_id, cell)).glob("*_top_features.csv"))
            if candidates:
                top_features_path = candidates[0]

        if top_features_path.exists():
            header = pd.read_csv(top_features_path, nrows=0).columns.tolist()
            cell_feature_columns = [col for col in cfg.feature_columns if col in header]
            if cfg.target_column not in header:
                header.append(cfg.target_column)
        else:
            cell_feature_columns = cfg.feature_columns

        samples = build_sequences(cell_df, cfg, cell_feature_columns)
        if not samples:
            print(f"[WARN] Skipping {cell}: insufficient data for sequence generation")
            continue
        try:
            (
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
                train_meta,
                val_meta,
                test_meta,
            ) = split_sequences(samples, cfg.val_fraction, cfg.test_fraction)
        except ValueError as exc:
            print(f"[WARN] Skipping {cell}: {exc}")
            continue

        print(
            f"[INFO] {cell}: split counts -> train: {len(train_x)}, val: {len(val_x)}, test: {len(test_x)}"
        )

        (
            train_x_scaled,
            val_x_scaled,
            test_x_scaled,
            train_y_scaled,
            val_y_scaled,
            test_y_scaled,
            feature_scaler,
            target_scaler,
        ) = scale_features(train_x, val_x, test_x, train_y, val_y, test_y, cfg.scale_target)

        cell_dir = args.output_dir / cell_dataset_prefix(dataset_id, cell)
        save_outputs(
            cell_dir,
            train_x_scaled,
            val_x_scaled,
            test_x_scaled,
            train_y_scaled,
            val_y_scaled,
            test_y_scaled,
            train_meta,
            val_meta,
            test_meta,
            feature_scaler,
            target_scaler,
            cfg,
            cell_name=cell,
            dataset_id=dataset_id,
            feature_columns=cell_feature_columns,
        )
        summary.append(
            {
                "cell": cell,
                "output_dir": str(cell_dir.resolve()),
                "train_samples": len(train_x),
                "val_samples": len(val_x),
                "test_samples": len(test_x),
            }
        )

    if summary:
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Data preparation finished for {len(summary)} cells. Summary saved to {summary_path}")
    else:
        print("[WARN] No cell produced usable sequences. No outputs generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
