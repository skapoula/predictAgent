#!/usr/bin/env python3
"""Train per-cell LSTM regressors on preprocessed CellReports tensors."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
tf.random.set_seed(42)

# Fully utilise available CPU cores for this workload.
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-cell LSTM models on prepared tensors.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Directory containing prepared outputs (base directory or single cell directory).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset identifier or path (e.g., CellReports_1_S1.csv) to infer prepared directory automatically.",
    )
    parser.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Optional Viavi.Cell.Name or sanitised directory name to train (default: train all cells).",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="model.keras",
        help="Filename for saved model (per cell directory).",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    return parser.parse_args(list(argv) if argv is not None else None)


def load_array(base_dir: Path, prefix: str) -> np.ndarray:
    npy_path = base_dir / f"{prefix}.npy"
    joblib_path = base_dir / f"{prefix}.joblib"
    if npy_path.exists():
        return np.load(npy_path)
    if joblib_path.exists():
        data = joblib.load(joblib_path)
        return np.asarray(data)
    raise FileNotFoundError(f"Required tensor file missing: {npy_path} or {joblib_path}")


def load_dataset(base_dir: Path) -> Tuple[np.ndarray, ...]:
    tensors: Dict[str, np.ndarray] = {}
    metas: Dict[str, List[dict] | None] = {}
    for split in ["train", "val", "test"]:
        tensors[f"X_{split}"] = load_array(base_dir, f"{split}_inputs")
        tensors[f"y_{split}"] = load_array(base_dir, f"{split}_targets")
        meta_path = base_dir / f"{split}_meta.joblib"
        metas[f"{split}_meta"] = joblib.load(meta_path) if meta_path.exists() else None

    return (
        tensors["X_train"],
        tensors["y_train"],
        tensors["X_val"],
        tensors["y_val"],
        tensors["X_test"],
        tensors["y_test"],
        metas.get("train_meta"),
        metas.get("val_meta"),
        metas.get("test_meta"),
    )


def ensure_shapes(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> Tuple[int, int]:
    for arr, name in [
        (train_x, "X_train"),
        (train_y, "y_train"),
        (val_x, "X_val"),
        (val_y, "y_val"),
        (test_x, "X_test"),
        (test_y, "y_test"),
    ]:
        if arr.ndim == 0:
            raise ValueError(f"Tensor {name} is empty or scalar")
    if train_x.ndim != 3:
        raise ValueError(f"Expected train inputs to be 3D (samples, T, F); got {train_x.shape}")
    if train_y.ndim != 1:
        raise ValueError(f"Expected train targets to be 1D; got {train_y.shape}")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("Mismatch between train inputs and targets")
    if val_x.shape[0] != val_y.shape[0] or test_x.shape[0] != test_y.shape[0]:
        raise ValueError("Mismatch between validation/test inputs and targets")
    if val_x.shape[1:] != train_x.shape[1:] or test_x.shape[1:] != train_x.shape[1:]:
        raise ValueError("Inconsistent temporal dimensions between splits")
    return train_x.shape[1], train_x.shape[2]


def build_model(timesteps: int, features: int, lr: float) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, features)),
            tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, kernel_regularizer=regularizer),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=0.05), metrics=["mae"])
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_pred - y_true) / denom)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def naive_baseline(inputs: np.ndarray, metadata: dict, feature_scaler, target_name: str) -> np.ndarray | None:
    feat_cols = metadata.get("feature_columns", [])
    if target_name in feat_cols and feature_scaler is not None:
        idx = feat_cols.index(target_name)
        last_step_scaled = inputs[:, -1, :]
        last_step_raw = feature_scaler.inverse_transform(last_step_scaled)
        return last_step_raw[:, idx]
    return None


def inverse_target_values(arr: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return arr
    return scaler.inverse_transform(arr.reshape(-1, 1)).ravel()


def build_chronological_indices(meta_list: List[dict] | None, length: int) -> np.ndarray:
    if meta_list and len(meta_list) > 0 and isinstance(meta_list[0], dict):
        timestamps = np.array(
            [m.get("target_timestamp", idx) for idx, m in enumerate(meta_list)],
            dtype=np.float64,
        )
        return np.argsort(timestamps)
    return np.arange(length)


def build_time_axis(meta_list: List[dict] | None, indices: np.ndarray) -> pd.Series:
    times = []
    for idx in indices:
        meta = meta_list[idx] if meta_list and idx < len(meta_list) else None
        dt_value = None
        if isinstance(meta, dict):
            dt_value = meta.get("target_dt") or meta.get("input_end_dt")
        times.append(pd.to_datetime(dt_value) if dt_value is not None else idx)
    return pd.Series(times)


def extract_band(meta: dict | None, default: str = "Unknown") -> str:
    if not meta or "entity" not in meta:
        return default
    entity = meta.get("entity")
    if not isinstance(entity, str):
        return default
    parts = entity.split("/")
    if len(parts) >= 2:
        return parts[1]
    return default


def render_dashboard_schema() -> str:
    return """
      <section>
        <h2>Dashboard Schema for LSTM (TensorFlow)</h2>
        <p>Panels grouped by layers of observability:</p>
        <div class="dashboard-schema">
          <h3>A. Model Training Performance</h3>
          <ul>
            <li><strong>Loss curves</strong>
              <ul>
                <li><code>train_loss</code> vs <code>val_loss</code> per epoch</li>
                <li>Rolling average of training loss to smooth epoch-to-epoch noise</li>
              </ul>
            </li>
            <li><strong>Accuracy metrics</strong>
              <ul>
                <li>Classification: accuracy, precision, recall, F1</li>
                <li>Regression: RMSE, MAE</li>
              </ul>
            </li>
            <li><strong>Learning dynamics</strong>
              <ul>
                <li>Gradient norms per layer to monitor vanishing/exploding trends</li>
                <li>Weight histograms per epoch (<code>model.layers[i].kernel</code>)</li>
                <li>Activation histograms from LSTM cell outputs before/after nonlinearities</li>
              </ul>
            </li>
          </ul>
          <p><em>TensorFlow tip:</em> enable <code>tf.keras.callbacks.TensorBoard(histogram_freq=1)</code> to log weights, activations, and gradient histograms for these panels.</p>
        </div>
      </section>
    """


def resolve_data_dir(raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path
    cwd_path = (Path.cwd() / raw_path).resolve()
    if cwd_path.exists():
        return cwd_path
    script_path = (SCRIPT_DIR / raw_path).resolve()
    if script_path.exists():
        return script_path
    parts = raw_path.parts
    if parts and parts[0] == SCRIPT_DIR.name:
        trimmed = Path(*parts[1:])
        trimmed_cwd = (Path.cwd() / trimmed).resolve()
        if trimmed_cwd.exists():
            return trimmed_cwd
        trimmed_script = (SCRIPT_DIR / trimmed).resolve()
        if trimmed_script.exists():
            return trimmed_script
    return cwd_path


def is_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "train_inputs.joblib").exists()


def discover_datasets(base_dir: Path) -> List[Tuple[Path, dict]]:
    if is_dataset_dir(base_dir):
        meta_path = base_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if meta.get("cell_name"):
            return [(base_dir, meta)]

    datasets: List[Tuple[Path, dict]] = []
    for child in sorted(base_dir.iterdir()):
        if is_dataset_dir(child):
            meta_path = child / "metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            datasets.append((child, meta))
    return datasets


def build_report(
    report_path: Path,
    history: dict,
    metrics: dict,
    loss_fig_html: str,
    scatter_fig_html: str,
    series_fig_html: str,
    series_title: str,
) -> None:
    def metrics_table(metrics_dict: dict) -> str:
        rows = []

        def fmt(value: float | None, precision: int) -> str:
            if value is None or np.isnan(value):
                return "N/A"
            return f"{value:.{precision}f}"

        display_order = [
            ("Blend", "blend"),
            ("LSTM", "lstm"),
            ("Tree", "tree"),
            ("Baseline", "baseline"),
        ]

        for label, key in display_order:
            entry = metrics_dict.get(key)
            if not entry:
                continue
            val_metrics = entry.get("val", {})
            test_metrics = entry.get("test", {})
            rows.append(
                f"<tr><td>{label} (Val)</td>"
                f"<td>{fmt(val_metrics.get('rmse'), 4)}</td>"
                f"<td>{fmt(val_metrics.get('mae'), 4)}</td>"
                f"<td>{fmt(val_metrics.get('mape'), 2)}</td>"
                "</tr>"
            )
            rows.append(
                f"<tr><td>{label} (Test)</td>"
                f"<td>{fmt(test_metrics.get('rmse'), 4)}</td>"
                f"<td>{fmt(test_metrics.get('mae'), 4)}</td>"
                f"<td>{fmt(test_metrics.get('mape'), 2)}</td>"
                "</tr>"
            )
        return "".join(rows)

    history_frame = pd.DataFrame(history)
    history_html = history_frame.to_html(index=True, classes="table", float_format=lambda x: f"{x:.6f}")

    schema_html = render_dashboard_schema()

    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <title>LSTM Training Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; background: #f7fafc; color: #243b53; margin: 2rem; }}
        section {{ background: #ffffff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
        h1, h2 {{ color: #102a43; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
        th, td {{ border: 1px solid #d9e2ec; padding: 0.5rem 0.75rem; text-align: center; }}
        th {{ background: #d8e2ff; }}
        .table {{ margin-top: 1rem; }}
        .dashboard-schema ul {{ margin-left: 1.25rem; list-style-type: disc; }}
        .dashboard-schema ul ul {{ list-style-type: circle; margin-top: 0.25rem; }}
        .dashboard-schema li {{ margin-bottom: 0.35rem; }}
        .dashboard-schema strong {{ color: #243b53; }}
      </style>
    </head>
    <body>
      <h1>LSTM Training Summary</h1>
      {schema_html}
      <section>
        <h2>Performance Metrics</h2>
        <table>
          <thead>
            <tr>
              <th>Model / Split</th>
              <th>RMSE</th>
              <th>MAE</th>
              <th>MAPE (%)</th>
            </tr>
          </thead>
          <tbody>
            {metrics_table(metrics)}
          </tbody>
        </table>
      </section>
      <section>
        <h2>Training History</h2>
        {history_html}
        {loss_fig_html}
      </section>
      <section>
        <h2>Test Predictions vs Actuals</h2>
        {scatter_fig_html}
      </section>
      <section>
        <h2>{series_title}</h2>
        {series_fig_html}
      </section>
    </body>
    </html>
    """

    report_path.write_text(html, encoding="utf-8")


def train_single_cell(dataset_dir: Path, meta: dict, args: argparse.Namespace) -> dict:
    cell_name = meta.get("cell_name", dataset_dir.name)
    dataset_id = meta.get("dataset") or sanitize_dataset_name(dataset_dir.parent.name.replace("prepared_lstm_", ""))
    cell_key = cell_dataset_key(dataset_id, cell_name)
    print(f"[INFO] Training {dataset_id} — {cell_name} from {dataset_dir}")

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
    ) = load_dataset(dataset_dir)

    timesteps, features = ensure_shapes(train_x, train_y, val_x, val_y, test_x, test_y)

    feature_scaler = joblib.load(dataset_dir / "feature_scaler.joblib") if (dataset_dir / "feature_scaler.joblib").exists() else None
    target_scaler = joblib.load(dataset_dir / "target_scaler.joblib") if (dataset_dir / "target_scaler.joblib").exists() else None

    metadata_cfg = meta
    target_name = metadata_cfg.get("target_column", "PRB.Util.DL")

    baseline_val_pred = naive_baseline(val_x, metadata_cfg, feature_scaler, target_name)
    baseline_test_pred = naive_baseline(test_x, metadata_cfg, feature_scaler, target_name)

    model = build_model(timesteps, features, args.lr)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(dataset_dir / "model_best.keras"), save_best_only=True, monitor="val_loss"),
    ]

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    val_pred = model.predict(val_x, batch_size=args.batch_size).ravel()
    test_pred = model.predict(test_x, batch_size=args.batch_size).ravel()

    val_y_eval = inverse_target_values(val_y, target_scaler)
    test_y_eval = inverse_target_values(test_y, target_scaler)
    val_pred_eval = inverse_target_values(val_pred, target_scaler)
    test_pred_eval = inverse_target_values(test_pred, target_scaler)

    val_metrics = compute_metrics(val_y_eval, val_pred_eval)
    test_metrics = compute_metrics(test_y_eval, test_pred_eval)

    results = {
        "cell": cell_name,
        "dataset": dataset_id,
        "cell_key": cell_key,
        "model": {"val": val_metrics, "test": test_metrics},
    }

    train_flat = train_x.reshape(train_x.shape[0], -1)
    val_flat = val_x.reshape(val_x.shape[0], -1)
    test_flat = test_x.reshape(test_x.shape[0], -1)

    gbr = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gbr.fit(train_flat, train_y)
    val_pred_tree = gbr.predict(val_flat)
    test_pred_tree = gbr.predict(test_flat)

    val_pred_tree_eval = inverse_target_values(val_pred_tree, target_scaler)
    test_pred_tree_eval = inverse_target_values(test_pred_tree, target_scaler)

    val_metrics_tree = compute_metrics(val_y_eval, val_pred_tree_eval)
    test_metrics_tree = compute_metrics(test_y_eval, test_pred_tree_eval)

    denom = val_pred_eval - val_pred_tree_eval
    numerator = np.dot(val_y_eval - val_pred_tree_eval, denom)
    denominator = np.dot(denom, denom)
    if denominator == 0:
        alpha = 1.0
    else:
        alpha = np.clip(numerator / denominator, 0.0, 1.0)

    val_pred_blend_eval = alpha * val_pred_eval + (1 - alpha) * val_pred_tree_eval
    test_pred_blend_eval = alpha * test_pred_eval + (1 - alpha) * test_pred_tree_eval

    val_metrics_blend = compute_metrics(val_y_eval, val_pred_blend_eval)
    test_metrics_blend = compute_metrics(test_y_eval, test_pred_blend_eval)

    if baseline_val_pred is not None and baseline_test_pred is not None:
        base_val_metrics = compute_metrics(val_y_eval, baseline_val_pred)
        base_test_metrics = compute_metrics(test_y_eval, baseline_test_pred)
        results["baseline"] = {"val": base_val_metrics, "test": base_test_metrics}

    model_path = dataset_dir / args.model_out
    model.save(model_path)
    joblib.dump(gbr, dataset_dir / "tree_model.joblib")

    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    (dataset_dir / "training_history.json").write_text(json.dumps(history_dict, indent=2), encoding="utf-8")

    results["lstm"] = {"val": val_metrics, "test": test_metrics}
    results["tree"] = {"val": val_metrics_tree, "test": test_metrics_tree}
    results["blend"] = {"val": val_metrics_blend, "test": test_metrics_blend, "weight": alpha}
    results["model"] = results["blend"]

    (dataset_dir / "metrics_val_test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    samples = min(1000, len(test_y_eval))
    indices = np.random.choice(len(test_y_eval), size=samples, replace=False)

    chronological_indices = build_chronological_indices(test_meta, len(test_y_eval))
    sample_idx = chronological_indices[:samples]
    time_values = build_time_axis(test_meta, sample_idx)
    bands = [extract_band(test_meta[idx] if test_meta else None, default="Band") for idx in sample_idx]
    series_df = pd.DataFrame(
        {
            "time": time_values.tolist(),
            "band": bands,
            "actual": test_y_eval[sample_idx],
            "pred_lstm": test_pred_eval[sample_idx],
            "pred_tree": test_pred_tree_eval[sample_idx],
            "pred": test_pred_blend_eval[sample_idx],
        }
    )
    focused_df = series_df[series_df["band"].isin(["B2", "B13"])].copy()
    if focused_df.empty:
        focused_df = series_df.copy()

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(y=history.history.get("loss", []), mode="lines", name="Train Loss"))
    loss_fig.add_trace(go.Scatter(y=history.history.get("val_loss", []), mode="lines", name="Val Loss"))
    loss_fig.update_layout(title="Training vs Validation Loss", xaxis_title="Epoch", yaxis_title="MSE Loss")
    loss_fig_html = pio.to_html(loss_fig, full_html=False, include_plotlyjs="cdn")

    scatter_fig = go.Figure()
    scatter_fig.add_trace(
        go.Scatter(
            x=test_y_eval[indices],
            y=test_pred_blend_eval[indices],
            mode="markers",
            opacity=0.5,
            name="Blend",
        )
    )
    lims = [
        min(test_y_eval[indices].min(), test_pred_blend_eval[indices].min()),
        max(test_y_eval[indices].max(), test_pred_blend_eval[indices].max()),
    ]
    scatter_fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Ideal", line=dict(dash="dash", color="red")))
    scatter_fig.update_layout(title="Prediction vs Actual (Test subset)", xaxis_title="Actual", yaxis_title="Predicted")
    scatter_fig_html = pio.to_html(scatter_fig, full_html=False, include_plotlyjs=False)

    series_fig = go.Figure()
    for band in sorted(focused_df["band"].unique()):
        band_df = focused_df[focused_df["band"] == band]
        series_fig.add_trace(go.Scatter(x=band_df["time"], y=band_df["actual"], mode="lines", name=f"{band} Actual", line=dict(width=2)))
        series_fig.add_trace(
            go.Scatter(
                x=band_df["time"],
                y=band_df["pred"],
                mode="lines",
                name=f"{band} Blend",
                line=dict(width=1.5, dash="dash"),
            )
        )
        series_fig.add_trace(
            go.Scatter(
                x=band_df["time"],
                y=band_df["pred_lstm"],
                mode="lines",
                name=f"{band} LSTM",
                line=dict(width=1, dash="dot"),
                opacity=0.4,
            )
        )
    series_title = "Chronological Comparison – " + ", ".join(sorted(focused_df["band"].unique()))
    series_fig.update_layout(title=series_title, xaxis_title="Timestamp", yaxis_title="Target Value", autosize=True, margin=dict(t=60, r=20, l=20, b=40))
    series_fig_html = pio.to_html(series_fig, full_html=False, include_plotlyjs=False, config={"responsive": True})

    metrics_records = []
    for label, key in [("Blend", "blend"), ("LSTM", "lstm"), ("Tree", "tree"), ("Baseline", "baseline")]:
        entry = results.get(key)
        if not entry:
            continue
        for split in ("val", "test"):
            metrics_split = entry.get(split)
            if not metrics_split:
                continue
            metrics_records.append({"Model": f"{label} ({split.title()})", "Metric": "RMSE", "Value": metrics_split.get("rmse")})
            metrics_records.append({"Model": f"{label} ({split.title()})", "Metric": "MAE", "Value": metrics_split.get("mae")})
            metrics_records.append({"Model": f"{label} ({split.title()})", "Metric": "MAPE", "Value": metrics_split.get("mape")})

    if metrics_records:
        metrics_df = pd.DataFrame(metrics_records)
        metrics_fig = px.bar(
            metrics_df,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group",
            title="Model Metrics",
        )
        metrics_fig.update_layout(autosize=True, margin=dict(t=60, r=20, l=20, b=40))
        metrics_fig_html = pio.to_html(metrics_fig, full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    else:
        metrics_fig_html = "<p>No metrics available.</p>"

    feature_report_name = f"{dataset_id}_{sanitize_cell_name(cell_name)}_feature_report.html"
    feature_report_path = dataset_dir / feature_report_name
    if not feature_report_path.exists():
        candidates = list(dataset_dir.glob("*_feature_report.html"))
        if candidates:
            feature_report_path = candidates[0]
    if feature_report_path.exists():
        feature_html_raw = feature_report_path.read_text(encoding="utf-8", errors="ignore")
        feature_html = f"<iframe srcdoc='{html.escape(feature_html_raw)}' title='Preprocessing Report' style='width:100%;min-height:800px;border:none;'></iframe>"
    else:
        feature_html = "<p>No preprocessing report found for this cell.</p>"

    build_report(
        dataset_dir / "training_report.html",
        history_dict,
        results,
        loss_fig_html,
        scatter_fig_html,
        series_fig_html,
        series_title,
    )

    print(f"[INFO] Cell {dataset_id} — {cell_name}: model saved to {model_path}")
    return {
        "cell": cell_name,
        "dataset": dataset_id,
        "cell_key": cell_key,
        "metrics": results,
        "metrics_fig_html": metrics_fig_html,
        "loss_html": loss_fig_html,
        "scatter_html": scatter_fig_html,
        "series_html": series_fig_html,
        "history_html": pd.DataFrame(history_dict).to_html(index=True, classes="table", float_format=lambda x: f"{x:.6f}"),
        "series_title": series_title,
        "feature_html": feature_html,
    }


def sanitize_cell_name(cell_name: str | None) -> str | None:
    if not cell_name:
        return None
    return cell_name.replace("/", "_")


def sanitize_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def cell_dataset_key(dataset_id: str, cell_name: str | None) -> str:
    cell_part = sanitize_cell_name(cell_name) if cell_name else ""
    return f"{dataset_id}_{cell_part}".strip("_")


def apply_cell_filter(datasets: List[Tuple[Path, dict]], cell_filter: str | None) -> List[Tuple[Path, dict]]:
    if not cell_filter:
        return datasets

    targets = set(cell_filter.split(",")) if isinstance(cell_filter, str) else set(cell_filter)

    def matches(path: Path, meta: dict) -> bool:
        cell_name = meta.get("cell_name")
        dataset_id = meta.get("dataset")
        candidates = {
            path.name,
            sanitize_cell_name(cell_name) if cell_name else None,
            cell_name,
            cell_dataset_key(dataset_id, cell_name) if dataset_id and cell_name else None,
        }
        return any(target in candidates for target in targets)

    filtered = [entry for entry in datasets if matches(*entry)]
    if not filtered:
        raise ValueError(f"Requested cell(s) '{cell_filter}' not found in prepared datasets")
    return filtered


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dataset and args.data_dir:
        raise ValueError("Specify either --dataset or --data_dir, not both.")

    if args.dataset:
        dataset_path = Path(args.dataset)
        dataset_id = sanitize_dataset_name(dataset_path.stem if dataset_path.suffix else args.dataset)
        base_dir = SCRIPT_DIR / f"prepared_lstm_{dataset_id}"
    elif args.data_dir is not None:
        base_dir = resolve_data_dir(args.data_dir)
    else:
        candidates = sorted(
            SCRIPT_DIR.glob("prepared_lstm_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No prepared_lstm_* directories found; specify --dataset or --data_dir")
        base_dir = candidates[0]
    datasets = discover_datasets(base_dir)
    if not datasets:
        raise FileNotFoundError(f"No prepared datasets found in {base_dir}")

    datasets = apply_cell_filter(datasets, args.cell)

    summary: Dict[str, dict] = {}
    cell_results: List[dict] = []
    for dataset_dir, meta in datasets:
        result = train_single_cell(dataset_dir, meta, args)
        summary[result["cell_key"]] = result["metrics"]
        cell_results.append(result)

    if len(datasets) >= 1:
        summary_path = base_dir / "metrics_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Summary metrics saved to {summary_path}")

        master_report_path = base_dir / "training_report.html"
        build_master_report(master_report_path, cell_results)
        print(f"[INFO] Master training report saved to {master_report_path}")

    return 0


def build_master_report(report_path: Path, cell_results: List[dict]) -> None:
    if not cell_results:
        return

    def cell_id(key: str) -> str:
        return key.replace("/", "_").replace(" ", "_")

    options_html = "".join(
        f"<option value='{cell_id(res['cell_key'])}'>{res['dataset']} — {res['cell']}</option>" for res in cell_results
    )

    sections_html = []
    for idx, res in enumerate(cell_results):
        cid = cell_id(res["cell_key"])
        display = "block" if idx == 0 else "none"
        metrics_json = json.dumps(res["metrics"], indent=2)
        sections_html.append(
            f"""
            <div class="cell-section" id="cell-{cid}" style="display:{display}">
              <h2>{res['dataset']} — {res['cell']}</h2>
              <div class="cell-layout">
                <nav class="cell-nav">
                  <button class="tab-button" onclick="showCellPanel('{cid}', 'metrics')">Metrics</button>
                  <button class="tab-button" onclick="showCellPanel('{cid}', 'training')">Training</button>
                  <button class="tab-button" onclick="showCellPanel('{cid}', 'preprocessing')">Preprocessing</button>
                </nav>
                <div class="cell-content">
                  <div class="cell-panel" id="cell-{cid}-metrics" style="display:block">
                    <section>
                      <h3>Model Metrics</h3>
                      <div class="plotly-container">{res['metrics_fig_html']}</div>
                      <details><summary>Metrics JSON</summary><pre>{metrics_json}</pre></details>
                    </section>
                  </div>
                  <div class="cell-panel" id="cell-{cid}-training" style="display:none">
                    <section>
                      <h3>Training History</h3>
                      {res['history_html']}
                      <div class="plotly-container">{res['loss_html']}</div>
                    </section>
                    <section>
                      <h3>Test Predictions vs Actuals</h3>
                      <div class="plotly-container">{res['scatter_html']}</div>
                    </section>
                    <section>
                      <h3>{res['series_title']}</h3>
                      <div class="plotly-container">{res['series_html']}</div>
                    </section>
                  </div>
                  <div class="cell-panel" id="cell-{cid}-preprocessing" style="display:none">
                    <section>
                      <h3>Preprocessing Feature Report</h3>
                      {res['feature_html']}
                    </section>
                  </div>
                </div>
              </div>
            </div>
            """
        )

    schema_html = render_dashboard_schema()

    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <title>Per-Cell LSTM Training Summary</title>
      <style>
        body {{ font-family: Arial, sans-serif; background: #f7fafc; color: #243b53; margin: 2rem; }}
        section {{ background: #ffffff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
        h1, h2, h3 {{ color: #102a43; }}
        select {{ padding: 0.6rem; font-size: 1rem; margin-bottom: 1.5rem; }}
        details {{ margin-top: 1rem; }}
        pre {{ background: #e2e8f0; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
        .cell-layout {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
        .cell-nav {{ display: flex; flex-direction: column; gap: 0.75rem; min-width: 180px; }}
        .tab-button {{ padding: 0.65rem 1rem; border: none; background: #486581; color: #fff; border-radius: 6px; cursor: pointer; text-align: left; }}
        .tab-button:hover {{ background: #334e68; }}
        .cell-content {{ flex: 1; min-width: 280px; }}
        .cell-panel section {{ margin-bottom: 1.5rem; }}
        .plotly-container {{ width: 100%; }}
        iframe {{ width: 100%; min-height: 800px; border: none; }}
        .dashboard-schema ul {{ margin-left: 1.25rem; list-style-type: disc; }}
        .dashboard-schema ul ul {{ list-style-type: circle; margin-top: 0.25rem; }}
        .dashboard-schema li {{ margin-bottom: 0.35rem; }}
        .dashboard-schema strong {{ color: #243b53; }}
      </style>
      <script>
        function onCellChange(sel) {{
          document.querySelectorAll('.cell-section').forEach(sec => sec.style.display = 'none');
          const target = document.getElementById('cell-' + sel.value);
          if (target) {{
            target.style.display = 'block';
            showCellPanel(sel.value, 'metrics');
          }}
        }}
        function showCellPanel(cellId, panel) {{
          document.querySelectorAll('#cell-' + cellId + ' .cell-panel').forEach(sec => sec.style.display = 'none');
          const target = document.getElementById('cell-' + cellId + '-' + panel);
          if (target) target.style.display = 'block';
        }}
        window.addEventListener('load', function() {{
          const selector = document.getElementById('cell-select');
          if (selector && selector.options.length) onCellChange(selector);
        }});
      </script>
    </head>
    <body>
      <h1>Per-Cell LSTM Training Summary</h1>
      <label for="cell-select">Select Cell:</label>
      <select id="cell-select" onchange="onCellChange(this)">
        {options_html}
      </select>
      {schema_html}
      {''.join(sections_html)}
    </body>
    </html>
    """

    report_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
