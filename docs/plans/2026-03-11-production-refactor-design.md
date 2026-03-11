# Production Refactor Design — predictagent

**Date:** 2026-03-11
**Scope:** Refactor `0930-lstm/` pipeline into a production-grade Python package with a batch training pipeline and FastAPI inference service. Delete all root-level research scripts.

---

## Decisions

| Question | Decision |
|---|---|
| Canonical implementation | `0930-lstm/` (TF LSTM + GBR ensemble) |
| Root-level scripts | Deleted entirely (`cell_load_lstm.py`, `cell_load_inference.py`, `preprocessing.py`) |
| Production mode | Batch retraining pipeline + FastAPI REST API |
| API input | Cell identifier + recent telemetry rows in request payload |
| Model storage | Filesystem (versioned directories, `latest` symlink) |
| Retraining trigger | Manual CLI (`predictagent-train`) — simplest to start |
| Web framework | FastAPI |
| Test coverage | Unit + integration + regression |

---

## Approach

Single Python package (`predictagent`), `uv`-managed, with clear submodule boundaries: `pipeline/`, `registry/`, `api/`. One `pyproject.toml`. CLI entry points for ingest, train, and serve.

---

## Section 1: Package Structure

```
/workspace/predictagent/
├── pyproject.toml
├── config/
│   └── default.yaml
├── src/
│   └── predictagent/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── exceptions.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── ingestor.py
│       │   ├── features.py
│       │   ├── sequencer.py
│       │   └── trainer.py
│       ├── registry/
│       │   ├── __init__.py
│       │   └── model_registry.py
│       └── api/
│           ├── __init__.py
│           ├── app.py
│           └── routers/
│               └── forecast.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_ingestor.py
│   │   ├── test_features.py
│   │   └── test_sequencer.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_api.py
│   └── regression/
│       ├── baseline.json
│       └── test_metrics_baseline.py
└── viavi-dataset/
    ├── raw/                         ← all current data files
    │   ├── CellReports.csv
    │   ├── UEReports-flow.csv
    │   ├── README.md
    │   └── README_Viavi_Dataset.md
    └── processed/                   ← empty; pipeline writes outputs here
```

**Migration from `0930-lstm/`:**

| Old file | New home |
|---|---|
| `prepare_cell_reports.py` | `pipeline/ingestor.py` |
| `prepare_lstm_data.py` | `pipeline/features.py` + `pipeline/sequencer.py` |
| `train_lstm.py` | `pipeline/trainer.py` |
| `data_prep_config.yaml` | `config/default.yaml` (extended) |
| `feature_selection_prb.py` | absorbed into `pipeline/features.py` |
| `analyze_*.py` | deleted (research artefacts) |

---

## Section 2: Configuration & Schemas

**`config/default.yaml`** — single source of truth for all hyperparameters and column names:

```yaml
data:
  raw_path: viavi-dataset/raw/CellReports.csv
  processed_dir: viavi-dataset/processed/
  site_filter: "S1/"
  rollup_minutes: 15

features:
  target_column: PRB.Util.DL
  feature_columns:
    - PRB.Util.UL
    - RRC.ConnMean
    - DRB.UEThpDl
    - PEE.AvgPower
    - PRB.Util.DL_roll_mean_4
    - PRB.Util.DL_roll_mean_8
    - PRB.Util.DL_roll_mean_16
    - PRB.Util.DL_roll_mean_24
    - PRB.Util.DL_roll_std_4
    - PRB.Util.DL_roll_std_8
    - PRB.Util.DL_roll_std_16
    - PRB.Util.DL_roll_std_24
    - PRB.Util.DL_ema_4
    - PRB.Util.DL_ema_8
    - PRB.Util.DL_ema_16
    - PRB.Util.DL_ema_24
    - PRB.Util.DL_lag_5
    - PRB.Util.DL_lag_6
    - PRB.Util.DL_lag_7
    - PRB.Util.DL_lag_10
    - PRB.Util.DL_lag_11
    - PRB.Util.DL_lag_12
  lookback_steps: 48
  forecast_horizon: 1
  val_fraction: 0.2
  test_fraction: 0.2
  scale_target: false

training:
  batch_size: 256
  epochs: 30
  learning_rate: 0.001
  patience: 5
  seed: 42

registry:
  model_dir: models/

api:
  host: 0.0.0.0
  port: 8000
```

**Rules:**
- Config loaded once at startup; invalid/missing fields raise at boot
- Column names referenced in code via `settings.features.target_column` — never as string literals
- `ForecastRequest.rows` validated to have at least `lookback_steps` rows

**Pydantic schemas (`schemas.py`):**

```python
class TelemetryRow(BaseModel):
    timestamp: int
    cell_name: str
    prb_used_dl: float
    prb_avail_dl: float
    # ... other raw columns

class ForecastRequest(BaseModel):
    cell_name: str
    rows: list[TelemetryRow]          # caller provides last N minutes of data

class ForecastResponse(BaseModel):
    cell_name: str
    forecast_horizon_minutes: int
    predicted_prb_util_dl: float
    model_version: str

class TrainingMetrics(BaseModel):
    cell_name: str
    mae: float
    rmse: float
    mape: float
    trained_at: datetime
    model_version: str
```

---

## Section 3: Data Pipeline & Feature Engineering

**Training data flow:**

```
raw/CellReports.csv
    ↓ ingestor.py
    • validate schema (required columns, no null timestamps)
    • parse timestamps (Unix epoch → datetime)
    • filter to site_filter prefix
    • compute PRB.Util.DL = PrbUsedDl / PrbAvailDl
    • derive SiteSector metadata column
    • rollup to rollup_minutes intervals (mean per cell per interval)
    • write processed/CellReports_15_S1.csv
    ↓ features.py
    • rolling means/stds at configured window sizes
    • EMAs at configured spans
    • lag features at configured offsets
    • validate no NaN in feature columns after engineering
    ↓ sequencer.py
    • split per cell into contiguous segments (detect gaps > rollup_minutes)
    • chronological train/val/test split
    • fit StandardScaler on train split only  ← fixes data leakage bug
    • build sliding windows: (n_samples, lookback_steps, n_features)
    • write processed/<cell_name>/train.joblib, val.joblib, test.joblib, scaler.joblib
    ↓ trainer.py
    • train TF LSTM + GBR ensemble
    • evaluate → TrainingMetrics
    • save to registry
```

**Inference data flow:**

```
POST /forecast  {cell_name, rows: [...telemetry...]}
    ↓ validate ForecastRequest (Pydantic, ≥ lookback_steps rows)
    ↓ features.py  — apply feature engineering (no fitting)
    ↓ sequencer.py — build single sequence using saved scaler
    ↓ registry     — load latest model for cell_name
    ↓ trainer.py   — predict (LSTM + GBR blend)
    ↓ return ForecastResponse
```

**Key bug fixes vs current code:**

| Bug | Fix |
|---|---|
| Scaler fitted on full dataset | Scaler fitted on train split only |
| Column names hardcoded in 5+ places | All column names from `settings.features` |
| `print()` throughout | `logging.getLogger(__name__)` per module |
| Silent NaN drops | Log count of dropped rows at WARNING |
| No gap detection logging | Log segment count and rows discarded per cell |

---

## Section 4: Model Registry & API

**Filesystem registry layout:**

```
models/
└── S1_B2_C1/
    ├── 20260311_091500/
    │   ├── model.keras
    │   ├── gbr.joblib
    │   ├── scaler.joblib
    │   └── metadata.json
    └── latest -> 20260311_091500/
```

**Registry interface:**
```python
registry.save(cell_name, model, scaler, metrics)  → version str
registry.load(cell_name, version="latest")        → (model, scaler, metadata)
registry.list(cell_name)                          → list[str]
```

**API endpoints:**

```
POST /forecast
    Body:    ForecastRequest
    Returns: ForecastResponse
    Errors:  404 if no model for cell_name
             422 if rows < lookback_steps
             500 structured error (never raw exception)

GET /health
    Returns: {"status": "ok", "models_loaded": int}
```

- Models loaded on demand, cached per cell name in module-level dict
- All exceptions caught at router level; tracebacks never returned to caller
- Every request logs: `cell_name`, `n_rows_received`, `prediction`, `model_version`, latency

**CLI entry points:**
```bash
predictagent-ingest   --config config/default.yaml
predictagent-train    --config config/default.yaml
predictagent-serve    --config config/default.yaml
```

---

## Section 5: Testing

**pytest markers:**

| Marker | When | Default run |
|---|---|---|
| `unit` | Pure functions, no I/O | Yes |
| `integration` | Real filesystem, sample fixture data | Yes |
| `regression` | Full training run, metric assertion | No (`-m regression`) |

**Unit tests (`tests/unit/`):**

| File | Covers |
|---|---|
| `test_ingestor.py` | Schema validation, timestamp parsing, PRB ratio, site filter, rollup |
| `test_features.py` | Rolling stats shape, lag shifts, NaN logging |
| `test_sequencer.py` | Window shape, gap detection, train-only scaler fit, chronological order |

**Integration tests (`tests/integration/`):**

| File | Covers |
|---|---|
| `test_pipeline.py` | Full ingest → features → sequence → train → predict round-trip |
| `test_api.py` | POST /forecast 200; too few rows 422; unknown cell 404 |

**Regression tests (`tests/regression/`):**

| File | Covers |
|---|---|
| `test_metrics_baseline.py` | `MAE < baseline_mae` and `RMSE < baseline_rmse` from `baseline.json` |
