# predictagent Production Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the `0930-lstm/` pipeline into a production Python package with a batch training CLI (`predictagent-train`) and a FastAPI inference service (`predictagent-serve`), deleting all root-level research scripts.

**Architecture:** Single `src/predictagent` package with three sub-namespaces — `pipeline/` (ingest, features, sequencer, trainer), `registry/` (filesystem model store), `api/` (FastAPI app + routers). Config is YAML-driven via Pydantic. Three CLI entry points wire everything together.

**Tech Stack:** Python 3.11+, TensorFlow 2.x, scikit-learn, pandas, Pydantic v2, FastAPI, uvicorn, pytest, uv

**Source references:** Logic migrates from `0930-lstm/prepare_cell_reports.py`, `prepare_lstm_data.py`, `train_lstm.py`. Read those files before implementing each task.

---

## Task 1: Scaffold package, move data, delete research files

**Files:**
- Create: `src/predictagent/__init__.py`
- Create: `src/predictagent/pipeline/__init__.py`
- Create: `src/predictagent/registry/__init__.py`
- Create: `src/predictagent/api/__init__.py`
- Create: `src/predictagent/api/routers/__init__.py`
- Create: `pyproject.toml`
- Create: `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `tests/regression/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/` directory (empty, add `.gitkeep`)
- Create: `viavi-dataset/raw/` (move existing data files here)
- Create: `viavi-dataset/processed/` (empty)
- Delete: `cell_load_lstm.py`, `cell_load_inference.py`, `preprocessing.py`
- Delete: `0930-lstm/analyze_*.py`, `0930-lstm/feature_selection_prb.py`
- Delete: `0930-lstm/prepared_lstm/`, `0930-lstm/prepared_lstm_CellReports_*/`

**Step 1: Move data files into raw subfolder**

```bash
cd /workspace/predictagent
mkdir -p viavi-dataset/raw viavi-dataset/processed
mv viavi-dataset/CellReports.csv viavi-dataset/raw/
mv viavi-dataset/UEReports-flow.csv viavi-dataset/raw/ 2>/dev/null || true
mv viavi-dataset/README*.md viavi-dataset/raw/ 2>/dev/null || true
mv viavi-dataset/Sample* viavi-dataset/raw/ 2>/dev/null || true
```

**Step 2: Delete root-level research scripts**

```bash
rm -f cell_load_lstm.py cell_load_inference.py preprocessing.py
rm -f 0930-lstm/analyze_*.py 0930-lstm/feature_selection_prb.py
rm -rf 0930-lstm/prepared_lstm/ 0930-lstm/prepared_lstm_CellReports_*/
```

**Step 3: Create package directory structure**

```bash
mkdir -p src/predictagent/pipeline
mkdir -p src/predictagent/registry
mkdir -p src/predictagent/api/routers
mkdir -p tests/unit tests/integration tests/regression tests/fixtures
touch src/predictagent/__init__.py
touch src/predictagent/pipeline/__init__.py
touch src/predictagent/registry/__init__.py
touch src/predictagent/api/__init__.py
touch src/predictagent/api/routers/__init__.py
touch tests/__init__.py tests/unit/__init__.py
touch tests/integration/__init__.py tests/regression/__init__.py
touch tests/fixtures/.gitkeep
mkdir -p config
```

**Step 4: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "predictagent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "tensorflow>=2.13",
    "joblib>=1.3",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "fastapi>=0.104",
    "uvicorn[standard]>=0.24",
    "httpx>=0.25",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
]

[project.scripts]
predictagent-ingest = "predictagent.cli:ingest"
predictagent-train = "predictagent.cli:train"
predictagent-serve = "predictagent.cli:serve"

[tool.hatch.build.targets.wheel]
packages = ["src/predictagent"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not regression'"
markers = [
    "unit: pure function tests — no I/O",
    "integration: writes to tmp_path, uses fixture data",
    "regression: full training run; excluded from default suite",
]
```

**Step 5: Create `tests/conftest.py` with shared fixtures**

```python
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
```

**Step 6: Install the package in editable mode**

```bash
cd /workspace/predictagent
uv pip install -e ".[dev]"
```

Expected output: `Successfully installed predictagent-0.1.0`

**Step 7: Verify pytest discovers tests (all skip/pass — no tests yet)**

```bash
uv run pytest --collect-only 2>&1 | head -20
```

Expected: `no tests ran` or empty collection — no errors.

**Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold package structure, move data to raw/, delete research scripts"
```

---

## Task 2: Exceptions module

**Files:**
- Create: `src/predictagent/exceptions.py`
- Create: `tests/unit/test_exceptions.py`

**Step 1: Write failing test**

```python
# tests/unit/test_exceptions.py
import pytest
from predictagent.exceptions import (
    PredictAgentError,
    SchemaValidationError,
    IngestionError,
    FeatureEngineeringError,
    SequencerError,
    TrainingError,
    RegistryError,
    ModelNotFoundError,
    InferenceError,
    InsufficientDataError,
)


@pytest.mark.unit
def test_all_exceptions_are_subclasses_of_base():
    for exc_cls in [
        SchemaValidationError,
        IngestionError,
        FeatureEngineeringError,
        SequencerError,
        TrainingError,
        RegistryError,
        InferenceError,
    ]:
        assert issubclass(exc_cls, PredictAgentError)


@pytest.mark.unit
def test_model_not_found_is_registry_error():
    assert issubclass(ModelNotFoundError, RegistryError)


@pytest.mark.unit
def test_insufficient_data_is_inference_error():
    assert issubclass(InsufficientDataError, InferenceError)


@pytest.mark.unit
def test_raise_and_catch_as_base():
    with pytest.raises(PredictAgentError):
        raise IngestionError("bad data")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_exceptions.py -v
```

Expected: `ImportError` — module does not exist yet.

**Step 3: Implement `src/predictagent/exceptions.py`**

```python
"""Domain exceptions for predictagent."""


class PredictAgentError(Exception):
    """Base class for all predictagent errors."""


class SchemaValidationError(PredictAgentError):
    """Input data is missing required columns or has invalid types."""


class IngestionError(PredictAgentError):
    """Failure during raw data loading or rollup."""


class FeatureEngineeringError(PredictAgentError):
    """Failure during feature computation."""


class SequencerError(PredictAgentError):
    """Failure during sequence building or splitting."""


class TrainingError(PredictAgentError):
    """Failure during model training."""


class RegistryError(PredictAgentError):
    """Failure reading from or writing to the model registry."""


class ModelNotFoundError(RegistryError):
    """No model found for the requested cell name and version."""


class InferenceError(PredictAgentError):
    """Failure during model inference."""


class InsufficientDataError(InferenceError):
    """Caller provided fewer rows than lookback_steps requires."""
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_exceptions.py -v
```

Expected: `4 passed`

**Step 5: Commit**

```bash
git add src/predictagent/exceptions.py tests/unit/test_exceptions.py
git commit -m "feat(exceptions): add domain exception hierarchy"
```

---

## Task 3: Config module + default YAML

**Files:**
- Create: `config/default.yaml`
- Create: `src/predictagent/config.py`
- Create: `tests/unit/test_config.py`

**Step 1: Write failing test**

```python
# tests/unit/test_config.py
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: `ImportError` — module does not exist yet.

**Step 3: Create `config/default.yaml`**

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

**Step 4: Implement `src/predictagent/config.py`**

```python
"""Configuration loading and validation for predictagent."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    raw_path: Path
    processed_dir: Path
    site_filter: str
    rollup_minutes: int


class FeaturesConfig(BaseModel):
    target_column: str
    feature_columns: list[str]
    lookback_steps: int
    forecast_horizon: int
    val_fraction: float
    test_fraction: float
    scale_target: bool

    @model_validator(mode="after")
    def _fractions_sum_below_one(self) -> "FeaturesConfig":
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction ({self.val_fraction}) + test_fraction ({self.test_fraction}) must be < 1"
            )
        return self


class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    seed: int


class RegistryConfig(BaseModel):
    model_dir: Path


class ApiConfig(BaseModel):
    host: str
    port: int


class Settings(BaseModel):
    data: DataConfig
    features: FeaturesConfig
    training: TrainingConfig
    registry: RegistryConfig
    api: ApiConfig


def load_settings(config_path: Path) -> Settings:
    """Load and validate settings from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated Settings instance.

    Raises:
        FileNotFoundError: If config_path does not exist.
        pydantic.ValidationError: If any required field is missing or invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    settings = Settings.model_validate(raw)
    logger.info("Settings loaded from %s", config_path)
    return settings
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: `3 passed`

**Step 6: Commit**

```bash
git add config/default.yaml src/predictagent/config.py tests/unit/test_config.py
git commit -m "feat(config): add YAML-driven Pydantic settings with validation"
```

---

## Task 4: Schemas module

**Files:**
- Create: `src/predictagent/schemas.py`
- Create: `tests/unit/test_schemas.py`

**Step 1: Write failing test**

```python
# tests/unit/test_schemas.py
import pytest
from datetime import datetime, timezone


@pytest.mark.unit
def test_forecast_request_valid():
    from predictagent.schemas import ForecastRequest, TelemetryRow

    rows = [
        TelemetryRow(
            timestamp=1672502400 + i * 900,
            cell_name="S1/B2/C1",
            prb_used_dl=70.0,
            prb_avail_dl=100.0,
            prb_used_ul=40.0,
            prb_avail_ul=100.0,
        )
        for i in range(10)
    ]
    req = ForecastRequest(cell_name="S1/B2/C1", rows=rows)
    assert req.cell_name == "S1/B2/C1"
    assert len(req.rows) == 10


@pytest.mark.unit
def test_forecast_request_empty_rows_raises():
    from predictagent.schemas import ForecastRequest
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ForecastRequest(cell_name="S1/B2/C1", rows=[])


@pytest.mark.unit
def test_forecast_response_fields():
    from predictagent.schemas import ForecastResponse

    resp = ForecastResponse(
        cell_name="S1/B2/C1",
        forecast_horizon_minutes=15,
        predicted_prb_util_dl=0.72,
        model_version="20260311_091500",
    )
    assert resp.predicted_prb_util_dl == pytest.approx(0.72)


@pytest.mark.unit
def test_training_metrics_fields():
    from predictagent.schemas import TrainingMetrics

    m = TrainingMetrics(
        cell_name="S1/B2/C1",
        mae=0.05,
        rmse=0.07,
        mape=8.3,
        trained_at=datetime.now(timezone.utc),
        model_version="20260311_091500",
    )
    assert m.mae == pytest.approx(0.05)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_schemas.py -v
```

Expected: `ImportError`

**Step 3: Implement `src/predictagent/schemas.py`**

```python
"""Pydantic schemas for API I/O and internal data contracts."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator


class TelemetryRow(BaseModel):
    """One 15-minute telemetry sample from a cell."""

    timestamp: int
    cell_name: str
    prb_used_dl: float
    prb_avail_dl: float
    prb_used_ul: float
    prb_avail_ul: float
    rrc_conn_mean: float | None = None
    drb_ue_thp_dl: float | None = None
    pee_avg_power: float | None = None


class ForecastRequest(BaseModel):
    """Inference request: cell identifier + recent telemetry window."""

    cell_name: str
    rows: list[TelemetryRow]

    @field_validator("rows")
    @classmethod
    def rows_not_empty(cls, v: list[TelemetryRow]) -> list[TelemetryRow]:
        if not v:
            raise ValueError("rows must contain at least one TelemetryRow")
        return v


class ForecastResponse(BaseModel):
    """Inference response with predicted PRB DL utilisation."""

    cell_name: str
    forecast_horizon_minutes: int
    predicted_prb_util_dl: float
    model_version: str


class TrainingMetrics(BaseModel):
    """Per-cell training outcome stored in the model registry."""

    cell_name: str
    mae: float
    rmse: float
    mape: float
    trained_at: datetime
    model_version: str
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_schemas.py -v
```

Expected: `4 passed`

**Step 5: Commit**

```bash
git add src/predictagent/schemas.py tests/unit/test_schemas.py
git commit -m "feat(schemas): add Pydantic I/O and metrics schemas"
```

---

## Task 5: pipeline/ingestor.py

Migrates logic from `0930-lstm/prepare_cell_reports.py`. Reads raw CSV → validates schema → filters by site → computes PRB utilisation ratios → derives SiteSector → rollups to configured interval → writes `processed/CellReports_{N}_{site}.csv`.

**Files:**
- Create: `src/predictagent/pipeline/ingestor.py`
- Create: `tests/unit/test_ingestor.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_ingestor.py
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_ingestor.py -v
```

Expected: `ImportError`

**Step 3: Implement `src/predictagent/pipeline/ingestor.py`**

Key logic source: `0930-lstm/prepare_cell_reports.py` — read that file now.

```python
"""Raw data ingestion: load, validate, enrich, rollup CellReports CSV."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from predictagent.config import Settings
from predictagent.exceptions import IngestionError, SchemaValidationError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "timestamp",
        "Viavi.Cell.Name",
        "RRU.PrbUsedDl",
        "RRU.PrbAvailDl",
        "RRU.PrbUsedUl",
        "RRU.PrbAvailUl",
    }
)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise SchemaValidationError if any required column is missing.

    Args:
        df: Raw input DataFrame.

    Raises:
        SchemaValidationError: Lists all missing column names.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Input data missing required columns: {', '.join(sorted(missing))}"
        )


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Unix-epoch timestamp column to tz-naive datetime."""
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    null_count = df["timestamp"].isna().sum()
    if null_count > 0:
        logger.warning("Dropping %d rows with unparseable timestamps", null_count)
    df = df.dropna(subset=["timestamp", "Viavi.Cell.Name"])
    df["timestamp"] = df["timestamp"].astype("int64")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def compute_prb_utilisation(df: pd.DataFrame) -> pd.DataFrame:
    """Add PRB.Util.DL and PRB.Util.UL columns (used/avail ratio).

    Args:
        df: DataFrame with RRU.Prb* columns.

    Returns:
        New DataFrame with PRB.Util.DL and PRB.Util.UL columns appended.
    """
    df = df.copy()
    for used, avail, col in [
        ("RRU.PrbUsedDl", "RRU.PrbAvailDl", "PRB.Util.DL"),
        ("RRU.PrbUsedUl", "RRU.PrbAvailUl", "PRB.Util.UL"),
    ]:
        used_s = pd.to_numeric(df[used], errors="coerce")
        avail_s = pd.to_numeric(df[avail], errors="coerce")
        df[col] = used_s.divide(avail_s.where(avail_s != 0)).clip(lower=0, upper=1)
    return df


def derive_site_sector(cell_name: str) -> str | None:
    """Return Site/SectN identifier from a Viavi cell name.

    Args:
        cell_name: e.g. "S1/B2/C1"

    Returns:
        e.g. "S1/Sect1", or None if the name cannot be parsed.
    """
    if not isinstance(cell_name, str):
        return None
    parts = [s.strip() for s in cell_name.split("/") if s.strip()]
    if len(parts) < 3:
        return None
    site = parts[0]
    cell_component = parts[-1]
    digits = "".join(ch for ch in cell_component if ch.isdigit())
    if not digits:
        return None
    return f"{site}/Sect{digits[-1]}"


def filter_by_site(df: pd.DataFrame, site_filter: str) -> pd.DataFrame:
    """Keep only rows whose Viavi.Cell.Name starts with site_filter.

    Args:
        df: Input DataFrame.
        site_filter: Prefix string, e.g. "S1/".

    Returns:
        Filtered DataFrame.
    """
    mask = df["Viavi.Cell.Name"].astype(str).str.startswith(site_filter, na=False)
    filtered = df[mask].copy()
    logger.info(
        "Site filter '%s': kept %d / %d rows", site_filter, len(filtered), len(df)
    )
    return filtered


def rollup_to_interval(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Aggregate metrics to a fixed time interval by mean per cell.

    Args:
        df: DataFrame with timestamp and Viavi.Cell.Name columns.
        interval_minutes: Target interval in minutes.

    Returns:
        Rolled-up DataFrame sorted by cell name and timestamp.
    """
    df = df.copy()
    dt_index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["interval_start"] = dt_index.dt.floor(f"{interval_minutes}min")

    non_metric = {"timestamp", "Viavi.Cell.Name", "SiteSector", "interval_start", "timestamp_dt"}
    metric_cols = [c for c in df.columns if c not in non_metric]

    grouped = (
        df.groupby(["Viavi.Cell.Name", "SiteSector", "interval_start"], dropna=False)[
            metric_cols
        ]
        .mean()
        .reset_index()
    )
    grouped["timestamp"] = (
        grouped["interval_start"].astype("int64") // 10**9
    )
    grouped["timestamp_dt"] = grouped["interval_start"]
    ordered = ["timestamp", "timestamp_dt", "Viavi.Cell.Name", "SiteSector", *metric_cols]
    grouped = grouped[ordered].sort_values(
        ["Viavi.Cell.Name", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)

    logger.info(
        "Rolled up to %d-min intervals: %d rows → %d rows",
        interval_minutes,
        len(df),
        len(grouped),
    )
    return grouped


def run_ingestion(settings: Settings) -> Path:
    """Run the full ingestion pipeline: load → validate → enrich → rollup → save.

    Args:
        settings: Validated application settings.

    Returns:
        Path to the written processed CSV file.

    Raises:
        IngestionError: If loading or processing fails.
        SchemaValidationError: If required columns are missing.
    """
    raw_path = settings.data.raw_path.resolve()
    if not raw_path.exists():
        raise IngestionError(f"Raw data file not found: {raw_path}")

    logger.info("Loading raw data from %s", raw_path)
    try:
        df = pd.read_csv(raw_path)
    except Exception as exc:
        raise IngestionError(f"Failed to read CSV: {exc}") from exc

    validate_schema(df)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    df = parse_timestamps(df)
    df = compute_prb_utilisation(df)
    df["SiteSector"] = df["Viavi.Cell.Name"].apply(derive_site_sector)

    df = filter_by_site(df, settings.data.site_filter)
    if df.empty:
        raise IngestionError(
            f"No rows remain after filtering for site '{settings.data.site_filter}'"
        )

    df = rollup_to_interval(df, settings.data.rollup_minutes)

    processed_dir = settings.data.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    site_slug = settings.data.site_filter.rstrip("/").replace("/", "_")
    output_path = processed_dir / f"CellReports_{settings.data.rollup_minutes}_{site_slug}.csv"
    df.to_csv(output_path, index=False)
    logger.info("Ingestion complete → %s (%d rows)", output_path, len(df))
    return output_path
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_ingestor.py -v
```

Expected: `8 passed`

**Step 5: Commit**

```bash
git add src/predictagent/pipeline/ingestor.py tests/unit/test_ingestor.py
git commit -m "feat(pipeline/ingestor): add raw data loading, PRB util, rollup pipeline"
```

---

## Task 6: pipeline/features.py

Migrates rolling-stats, EMA, and lag feature engineering from `0930-lstm/prepare_lstm_data.py`. Reads the config's `feature_columns` list and generates only the listed features. No fitting — purely deterministic transformations.

**Files:**
- Create: `src/predictagent/pipeline/features.py`
- Create: `tests/unit/test_features.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_features.py
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
    lagged = result["PRB.Util.DL_lag_5"].values
    # After dropna the lag_5 values should equal original values shifted by 5
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_features.py -v
```

Expected: `ImportError`

**Step 3: Implement `src/predictagent/pipeline/features.py`**

Feature naming convention: `{target_col}_roll_mean_{N}`, `{target_col}_roll_std_{N}`, `{target_col}_ema_{N}`, `{target_col}_lag_{N}`.

```python
"""Feature engineering: rolling statistics, EMAs, and lag features."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from predictagent.config import Settings
from predictagent.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

# Patterns for auto-generating features from column name conventions
_ROLL_MEAN_RE = re.compile(r"^(.+)_roll_mean_(\d+)$")
_ROLL_STD_RE = re.compile(r"^(.+)_roll_std_(\d+)$")
_EMA_RE = re.compile(r"^(.+)_ema_(\d+)$")
_LAG_RE = re.compile(r"^(.+)_lag_(\d+)$")


def _add_derived_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compute a single derived column in-place and return the DataFrame.

    Args:
        df: Input DataFrame.
        col: Column name encoding the transformation, e.g. "PRB.Util.DL_roll_mean_4".

    Returns:
        DataFrame with the new column added.

    Raises:
        FeatureEngineeringError: If the column name pattern is unrecognised
            or the source column does not exist.
    """
    if col in df.columns:
        return df

    for pattern, transform in [
        (_ROLL_MEAN_RE, "roll_mean"),
        (_ROLL_STD_RE, "roll_std"),
        (_EMA_RE, "ema"),
        (_LAG_RE, "lag"),
    ]:
        m = pattern.match(col)
        if not m:
            continue
        src, n = m.group(1), int(m.group(2))
        if src not in df.columns:
            raise FeatureEngineeringError(
                f"Source column '{src}' required for '{col}' not found in DataFrame"
            )
        if transform == "roll_mean":
            df[col] = df[src].rolling(n).mean()
        elif transform == "roll_std":
            df[col] = df[src].rolling(n).std()
        elif transform == "ema":
            df[col] = df[src].ewm(span=n, adjust=False).mean()
        elif transform == "lag":
            df[col] = df[src].shift(n)
        return df

    raise FeatureEngineeringError(
        f"Column '{col}' is not a base column and does not match any known "
        "pattern (roll_mean_N, roll_std_N, ema_N, lag_N)"
    )


def engineer_features(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Add all requested feature columns and drop rows with NaN values.

    All rolling/lag/EMA features are computed per the column naming convention.
    Base columns (e.g. PRB.Util.UL) must already exist in df.
    Derived columns (e.g. PRB.Util.DL_roll_mean_4) are computed on demand.

    Args:
        df: Cell-level DataFrame sorted by timestamp (single cell).
        target_column: Name of the prediction target column.
        feature_columns: List of feature column names to produce.

    Returns:
        DataFrame with all feature columns present, NaN rows dropped.

    Raises:
        FeatureEngineeringError: If a column cannot be produced.
    """
    df = df.copy()
    for col in feature_columns:
        df = _add_derived_column(df, col)

    required_cols = feature_columns + [target_column]
    before = len(df)
    df = df.dropna(subset=required_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN after feature engineering", dropped)

    return df


def run_feature_engineering(settings: Settings, processed_csv: Path) -> Path:
    """Apply feature engineering to the processed CSV and write per-cell CSVs.

    Args:
        settings: Application settings (feature_columns, target_column, processed_dir).
        processed_csv: Path to the rolled-up CSV written by run_ingestion().

    Returns:
        Path to the directory containing per-cell feature CSVs.

    Raises:
        FeatureEngineeringError: If feature computation fails.
    """
    import pandas as pd

    df = pd.read_csv(processed_csv)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    cell_dir = settings.data.processed_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    cells = sorted(df["Viavi.Cell.Name"].unique())
    for cell in cells:
        cell_df = df[df["Viavi.Cell.Name"] == cell].copy()
        cell_df = cell_df.sort_values("timestamp").reset_index(drop=True)
        try:
            cell_feat = engineer_features(
                cell_df,
                settings.features.target_column,
                settings.features.feature_columns,
            )
        except FeatureEngineeringError:
            logger.warning("Skipping cell %s: feature engineering failed", cell)
            continue

        safe_name = cell.replace("/", "_")
        out_path = cell_dir / f"{safe_name}_features.csv"
        cell_feat.to_csv(out_path, index=False)
        logger.info("Features written for %s → %s (%d rows)", cell, out_path, len(cell_feat))

    return cell_dir
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_features.py -v
```

Expected: `5 passed`

**Step 5: Commit**

```bash
git add src/predictagent/pipeline/features.py tests/unit/test_features.py
git commit -m "feat(pipeline/features): add rolling, EMA, lag feature engineering"
```

---

## Task 7: pipeline/sequencer.py

Migrates `build_sequences`, `split_sequences`, `scale_features` from `0930-lstm/prepare_lstm_data.py`. Key correctness fix: scaler fitted on train split only.

**Files:**
- Create: `src/predictagent/pipeline/sequencer.py`
- Create: `tests/unit/test_sequencer.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_sequencer.py
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
    total = sum(len(splits[k][1]) for k in ("train", "val", "test"))
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_sequencer.py -v
```

Expected: `ImportError`

**Step 3: Implement `src/predictagent/pipeline/sequencer.py`**

Core logic from `0930-lstm/prepare_lstm_data.py` — `build_sequences`, `split_sequences`, `scale_features`. Read that file now.

```python
"""Sliding-window sequence builder, train/val/test splitter, and feature scaler."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from predictagent.config import Settings
from predictagent.exceptions import SequencerError

logger = logging.getLogger(__name__)

# Type alias: one sample is (X_window, y_target, meta_dict)
Sample = tuple[np.ndarray, float, dict[str, Any]]


def _infer_step(timestamps: np.ndarray) -> float:
    """Return median inter-sample step in seconds."""
    diffs = np.diff(timestamps)
    return float(np.median(diffs)) if len(diffs) > 0 else np.nan


def build_sequences(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    lookback: int,
    horizon: int,
) -> list[Sample]:
    """Build sliding-window samples from a single-cell feature DataFrame.

    Sequences that span timestamp gaps (> inferred step) are discarded.

    Args:
        df: Single-cell feature DataFrame sorted by timestamp.
        target_column: Column to predict.
        feature_columns: Ordered list of input feature columns.
        lookback: Number of time steps in the input window.
        horizon: Number of steps ahead to forecast.

    Returns:
        List of (X_window, y_target, meta) tuples sorted by target_timestamp.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].to_numpy()
    step = _infer_step(timestamps)
    if not np.isfinite(step) or step <= 0:
        logger.warning("Could not infer step size — returning empty sequence list")
        return []

    features = df[feature_columns].to_numpy(dtype=np.float32)
    targets = df[target_column].to_numpy(dtype=np.float32)
    dt_list = df["timestamp_dt"].tolist()

    total = len(df)
    window_span = lookback + horizon
    if total < window_span:
        return []

    samples: list[Sample] = []
    for start in range(total - window_span + 1):
        end = start + lookback
        target_idx = end + horizon - 1

        input_ts = timestamps[start:end]
        target_ts = timestamps[end - 1 : target_idx + 1]

        if np.any(np.isnan(features[start:end])) or np.isnan(targets[target_idx]):
            continue
        if not np.all(np.diff(input_ts) == step):
            continue
        if target_ts[-1] - input_ts[-1] != step * horizon:
            continue

        meta: dict[str, Any] = {
            "entity": df.loc[0, "Viavi.Cell.Name"],
            "input_start_ts": int(input_ts[0]),
            "input_end_ts": int(input_ts[-1]),
            "target_timestamp": int(target_ts[-1]),
            "input_start_dt": str(dt_list[start]),
            "target_dt": str(dt_list[target_idx]),
        }
        samples.append((features[start:end].copy(), float(targets[target_idx]), meta))

    return samples


def split_sequences(
    samples: list[Sample],
    val_fraction: float,
    test_fraction: float,
) -> dict[str, dict[str, Any]]:
    """Chronological train/val/test split.

    Args:
        samples: List of (X, y, meta) tuples.
        val_fraction: Fraction of data for validation.
        test_fraction: Fraction of data for test.

    Returns:
        Dict with keys "train", "val", "test"; each maps to
        {"X": ndarray, "y": ndarray, "meta": list[dict]}.

    Raises:
        SequencerError: If no samples are provided or splits are degenerate.
    """
    if not samples:
        raise SequencerError("Cannot split an empty sample list")

    order = np.argsort([m["target_timestamp"] for _, _, m in samples])
    sorted_samples = [samples[i] for i in order]

    total = len(sorted_samples)
    train_end = int(total * (1 - val_fraction - test_fraction))
    val_end = int(total * (1 - test_fraction))

    if train_end <= 0 or val_end <= train_end or val_end >= total:
        raise SequencerError(
            f"Dataset of {total} samples is too small for "
            f"val_fraction={val_fraction}, test_fraction={test_fraction}"
        )

    def _stack(subset: list[Sample]) -> dict[str, Any]:
        xs, ys, metas = zip(*subset)
        return {
            "X": np.stack(xs, axis=0),
            "y": np.array(ys, dtype=np.float32),
            "meta": list(metas),
        }

    return {
        "train": _stack(sorted_samples[:train_end]),
        "val": _stack(sorted_samples[train_end:val_end]),
        "test": _stack(sorted_samples[val_end:]),
    }


def scale_splits(
    splits: dict[str, dict[str, Any]],
    scale_target: bool,
) -> tuple[dict[str, dict[str, Any]], StandardScaler]:
    """Fit StandardScaler on train features only; transform all splits.

    Args:
        splits: Output of split_sequences().
        scale_target: If True, also scale y values.

    Returns:
        Tuple of (scaled_splits, feature_scaler). scaled_splits has the same
        structure as splits but with scaled X (and optionally y) arrays.
    """
    n_features = splits["train"]["X"].shape[-1]
    feature_scaler = StandardScaler()
    feature_scaler.fit(splits["train"]["X"].reshape(-1, n_features))

    def _transform_x(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        return feature_scaler.transform(arr.reshape(-1, n_features)).reshape(shape)

    target_scaler: StandardScaler | None = None
    if scale_target:
        target_scaler = StandardScaler()
        target_scaler.fit(splits["train"]["y"].reshape(-1, 1))

    scaled: dict[str, dict[str, Any]] = {}
    for split_name, data in splits.items():
        y = data["y"]
        if scale_target and target_scaler is not None:
            y = target_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()
        scaled[split_name] = {"X": _transform_x(data["X"]), "y": y, "meta": data["meta"]}

    return scaled, feature_scaler


def run_sequencing(settings: Settings, cell_features_dir: Path) -> Path:
    """Build sequences, split, scale, and save tensors for all cells.

    Scaler is fitted on the train split only — val/test use the train scaler.

    Args:
        settings: Application settings.
        cell_features_dir: Directory of per-cell feature CSVs (from run_feature_engineering).

    Returns:
        Path to the directory containing per-cell joblib tensor files.
    """
    tensors_dir = settings.data.processed_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(cell_features_dir.glob("*_features.csv")):
        cell_df = pd.read_csv(csv_path)
        cell_df["timestamp_dt"] = pd.to_datetime(cell_df["timestamp"], unit="s", utc=True)
        cell_name = cell_df["Viavi.Cell.Name"].iloc[0]
        safe = cell_name.replace("/", "_")

        samples = build_sequences(
            cell_df,
            settings.features.target_column,
            settings.features.feature_columns,
            settings.features.lookback_steps,
            settings.features.forecast_horizon,
        )
        if not samples:
            logger.warning("No sequences for cell %s — skipping", cell_name)
            continue

        try:
            splits = split_sequences(
                samples, settings.features.val_fraction, settings.features.test_fraction
            )
        except SequencerError as exc:
            logger.warning("Skipping cell %s: %s", cell_name, exc)
            continue

        scaled, feature_scaler = scale_splits(splits, settings.features.scale_target)

        cell_tensor_dir = tensors_dir / safe
        cell_tensor_dir.mkdir(parents=True, exist_ok=True)

        for split_name, data in scaled.items():
            joblib.dump(data["X"], cell_tensor_dir / f"{split_name}_inputs.joblib")
            joblib.dump(data["y"], cell_tensor_dir / f"{split_name}_targets.joblib")
            joblib.dump(data["meta"], cell_tensor_dir / f"{split_name}_meta.joblib")
        joblib.dump(feature_scaler, cell_tensor_dir / "feature_scaler.joblib")

        import json
        metadata = {
            "cell_name": cell_name,
            "target_column": settings.features.target_column,
            "feature_columns": settings.features.feature_columns,
            "lookback_steps": settings.features.lookback_steps,
            "forecast_horizon": settings.features.forecast_horizon,
            "scale_target": settings.features.scale_target,
            "train_samples": int(scaled["train"]["X"].shape[0]),
            "val_samples": int(scaled["val"]["X"].shape[0]),
            "test_samples": int(scaled["test"]["X"].shape[0]),
        }
        (cell_tensor_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        logger.info(
            "Sequencer: %s → train=%d, val=%d, test=%d",
            cell_name,
            metadata["train_samples"],
            metadata["val_samples"],
            metadata["test_samples"],
        )

    return tensors_dir
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_sequencer.py -v
```

Expected: `5 passed`

**Step 5: Commit**

```bash
git add src/predictagent/pipeline/sequencer.py tests/unit/test_sequencer.py
git commit -m "feat(pipeline/sequencer): add sequence builder, splitter, train-only scaler"
```

---

## Task 8: pipeline/trainer.py

Migrates `build_model`, `train_single_cell`, `compute_metrics`, `inverse_target_values`, blend logic from `0930-lstm/train_lstm.py`. Adds `predict()` function used by the API. Read `train_lstm.py` carefully before implementing.

**Files:**
- Create: `src/predictagent/pipeline/trainer.py`
- Create: `tests/unit/test_trainer.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_trainer.py
import numpy as np
import pytest


@pytest.mark.unit
def test_build_model_output_shape():
    from predictagent.pipeline.trainer import build_lstm_model

    model = build_lstm_model(timesteps=8, n_features=3, lr=1e-3)
    import numpy as np
    x = np.random.rand(5, 8, 3).astype(np.float32)
    pred = model.predict(x, verbose=0)
    assert pred.shape == (5, 1)


@pytest.mark.unit
def test_compute_metrics_known_values():
    from predictagent.pipeline.trainer import compute_metrics

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_predict_returns_float(tmp_path):
    """predict() wraps model + GBR blend and returns a scalar float."""
    from predictagent.pipeline.trainer import build_lstm_model, predict
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib

    lookback, n_feat = 4, 2
    model = build_lstm_model(timesteps=lookback, n_features=n_feat, lr=1e-3)
    gbr = GradientBoostingRegressor(n_estimators=10, random_state=42)
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0, 1, (50, lookback, n_feat)).astype(np.float32)
    y_train = rng.uniform(0, 1, 50).astype(np.float32)
    gbr.fit(X_train.reshape(50, -1), y_train)

    X_input = rng.uniform(0, 1, (1, lookback, n_feat)).astype(np.float32)
    result = predict(model, gbr, X_input, alpha=0.7)
    assert isinstance(result, float)
    assert 0.0 <= result <= 2.0  # rough sanity on scale
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_trainer.py -v
```

Expected: `ImportError`

**Step 3: Implement `src/predictagent/pipeline/trainer.py`**

```python
"""TensorFlow LSTM + GradientBoosting ensemble trainer and predictor."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from predictagent.config import Settings
from predictagent.exceptions import TrainingError
from predictagent.schemas import TrainingMetrics

logger = logging.getLogger(__name__)


def build_lstm_model(timesteps: int, n_features: int, lr: float) -> tf.keras.Model:
    """Build the 3-layer LSTM regressor with L1L2 regularisation.

    Args:
        timesteps: Lookback window length.
        n_features: Number of input features.
        lr: Adam learning rate.

    Returns:
        Compiled Keras model.
    """
    reg = tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, n_features)),
            tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, kernel_regularizer=reg),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae"],
    )
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict with "rmse", "mae", "mape" keys.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_pred - y_true) / denom)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def _compute_blend_weight(
    lstm_pred: np.ndarray, gbr_pred: np.ndarray, y_true: np.ndarray
) -> float:
    """Compute optimal LSTM blend weight alpha on the validation set.

    alpha minimises ||y - (alpha * lstm + (1-alpha) * gbr)||^2.

    Returns:
        Alpha clipped to [0, 1].
    """
    denom_vec = lstm_pred - gbr_pred
    numerator = float(np.dot(y_true - gbr_pred, denom_vec))
    denominator = float(np.dot(denom_vec, denom_vec))
    if denominator == 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def predict(
    lstm_model: tf.keras.Model,
    gbr_model: GradientBoostingRegressor,
    X: np.ndarray,
    alpha: float,
) -> float:
    """Run blend prediction for a single input window.

    Args:
        lstm_model: Trained Keras model.
        gbr_model: Trained GradientBoostingRegressor.
        X: Input array of shape (1, lookback, n_features).
        alpha: LSTM blend weight (stored in registry metadata).

    Returns:
        Scalar blend prediction.
    """
    lstm_pred = float(lstm_model.predict(X, verbose=0).ravel()[0])
    gbr_pred = float(gbr_model.predict(X.reshape(1, -1))[0])
    return alpha * lstm_pred + (1.0 - alpha) * gbr_pred


def train_cell(
    cell_name: str,
    tensor_dir: Path,
    settings: Settings,
) -> tuple[tf.keras.Model, GradientBoostingRegressor, float, TrainingMetrics]:
    """Train LSTM + GBR ensemble for a single cell.

    Args:
        cell_name: Viavi cell identifier (used for logging).
        tensor_dir: Directory containing train/val/test joblib tensors.
        settings: Application settings.

    Returns:
        Tuple of (lstm_model, gbr_model, alpha_blend_weight, TrainingMetrics).

    Raises:
        TrainingError: If tensors are missing or training fails.
    """
    tf.random.set_seed(settings.training.seed)
    np.random.seed(settings.training.seed)

    required = ["train_inputs", "train_targets", "val_inputs", "val_targets",
                "test_inputs", "test_targets"]
    for name in required:
        if not (tensor_dir / f"{name}.joblib").exists():
            raise TrainingError(f"Missing tensor file '{name}.joblib' in {tensor_dir}")

    train_x = joblib.load(tensor_dir / "train_inputs.joblib")
    train_y = joblib.load(tensor_dir / "train_targets.joblib")
    val_x = joblib.load(tensor_dir / "val_inputs.joblib")
    val_y = joblib.load(tensor_dir / "val_targets.joblib")
    test_x = joblib.load(tensor_dir / "test_inputs.joblib")
    test_y = joblib.load(tensor_dir / "test_targets.joblib")

    timesteps, n_features = train_x.shape[1], train_x.shape[2]
    logger.info(
        "Training %s: %d train, %d val, %d test samples; shape=(%d, %d)",
        cell_name, len(train_x), len(val_x), len(test_x), timesteps, n_features,
    )

    model = build_lstm_model(timesteps, n_features, settings.training.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=settings.training.patience,
            restore_best_weights=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
    ]
    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=settings.training.epochs,
        batch_size=settings.training.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )

    val_lstm = model.predict(val_x, verbose=0).ravel()
    test_lstm = model.predict(test_x, verbose=0).ravel()

    gbr = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        random_state=settings.training.seed,
    )
    gbr.fit(train_x.reshape(len(train_x), -1), train_y)
    val_gbr = gbr.predict(val_x.reshape(len(val_x), -1))
    test_gbr = gbr.predict(test_x.reshape(len(test_x), -1))

    alpha = _compute_blend_weight(val_lstm, val_gbr, val_y)
    val_pred = alpha * val_lstm + (1.0 - alpha) * val_gbr
    test_pred = alpha * test_lstm + (1.0 - alpha) * test_gbr

    test_metrics = compute_metrics(test_y, test_pred)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metrics = TrainingMetrics(
        cell_name=cell_name,
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        mape=test_metrics["mape"],
        trained_at=datetime.now(timezone.utc),
        model_version=version,
    )
    logger.info(
        "Cell %s test metrics: MAE=%.4f RMSE=%.4f MAPE=%.2f%%",
        cell_name, metrics.mae, metrics.rmse, metrics.mape,
    )
    return model, gbr, alpha, metrics


def run_training(settings: Settings, tensors_dir: Path) -> list[TrainingMetrics]:
    """Train all cells found in tensors_dir and save to the registry.

    Args:
        settings: Application settings.
        tensors_dir: Output of run_sequencing().

    Returns:
        List of TrainingMetrics, one per cell.
    """
    from predictagent.registry.model_registry import ModelRegistry

    registry = ModelRegistry(settings.registry.model_dir)
    all_metrics: list[TrainingMetrics] = []

    for cell_dir in sorted(tensors_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        meta_path = cell_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cell_name = meta.get("cell_name", cell_dir.name)

        try:
            lstm_model, gbr_model, alpha, metrics = train_cell(cell_name, cell_dir, settings)
        except TrainingError as exc:
            logger.error("Training failed for %s: %s", cell_name, exc)
            continue

        feature_scaler = joblib.load(cell_dir / "feature_scaler.joblib")
        registry.save(
            cell_name=cell_name,
            lstm_model=lstm_model,
            gbr_model=gbr_model,
            feature_scaler=feature_scaler,
            metrics=metrics,
            alpha=alpha,
            feature_columns=meta.get("feature_columns", []),
        )
        all_metrics.append(metrics)

    return all_metrics
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_trainer.py -v
```

Expected: `3 passed`

**Step 5: Commit**

```bash
git add src/predictagent/pipeline/trainer.py tests/unit/test_trainer.py
git commit -m "feat(pipeline/trainer): add LSTM+GBR ensemble training and predict()"
```

---

## Task 9: registry/model_registry.py

Filesystem-based model store. Version = UTC timestamp string. `latest` symlink always points to newest successful version.

**Files:**
- Create: `src/predictagent/registry/model_registry.py`
- Create: `tests/unit/test_model_registry.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_model_registry.py
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_model_registry.py -v
```

Expected: `ImportError`

**Step 3: Implement `src/predictagent/registry/model_registry.py`**

```python
"""Filesystem-based model registry with versioned directories."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from predictagent.exceptions import ModelNotFoundError, RegistryError
from predictagent.schemas import TrainingMetrics

logger = logging.getLogger(__name__)

_LATEST_FILE = "latest.json"


def _safe_name(cell_name: str) -> str:
    """Convert cell name to a filesystem-safe directory name."""
    return cell_name.replace("/", "_")


class ModelRegistry:
    """Save and load versioned LSTM+GBR ensemble models from the filesystem.

    Directory layout::

        model_dir/
        └── S1_B2_C1/
            ├── 20260311_091500/
            │   ├── model.keras
            │   ├── gbr.joblib
            │   ├── feature_scaler.joblib
            │   └── metadata.json
            └── latest.json          ← {"version": "20260311_091500"}
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        cell_name: str,
        lstm_model: tf.keras.Model,
        gbr_model: GradientBoostingRegressor,
        feature_scaler: StandardScaler,
        metrics: TrainingMetrics,
        alpha: float,
        feature_columns: list[str],
    ) -> str:
        """Persist model artefacts and return the version string.

        Args:
            cell_name: Viavi cell identifier.
            lstm_model: Trained Keras model.
            gbr_model: Trained GradientBoostingRegressor.
            feature_scaler: Fitted StandardScaler.
            metrics: Training metrics to embed in metadata.
            alpha: LSTM blend weight.
            feature_columns: Ordered list of feature column names.

        Returns:
            Version string (UTC timestamp).
        """
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe = _safe_name(cell_name)
        version_dir = self.model_dir / safe / version
        version_dir.mkdir(parents=True, exist_ok=True)

        try:
            lstm_model.save(version_dir / "model.keras")
            joblib.dump(gbr_model, version_dir / "gbr.joblib")
            joblib.dump(feature_scaler, version_dir / "feature_scaler.joblib")

            metadata = {
                "cell_name": cell_name,
                "version": version,
                "alpha": alpha,
                "feature_columns": feature_columns,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "mape": metrics.mape,
                "trained_at": metrics.trained_at.isoformat(),
            }
            (version_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            # Update latest pointer
            latest_path = self.model_dir / safe / _LATEST_FILE
            latest_path.write_text(
                json.dumps({"version": version}), encoding="utf-8"
            )
        except Exception as exc:
            raise RegistryError(f"Failed to save model for {cell_name}: {exc}") from exc

        logger.info("Saved model for %s → version %s", cell_name, version)
        return version

    def load(
        self, cell_name: str, version: str = "latest"
    ) -> dict:
        """Load model artefacts for a cell.

        Args:
            cell_name: Viavi cell identifier.
            version: Version string or "latest".

        Returns:
            Dict with keys: lstm_model, gbr_model, feature_scaler, alpha,
            feature_columns, metadata.

        Raises:
            ModelNotFoundError: If no model exists for cell_name or version.
        """
        safe = _safe_name(cell_name)
        cell_dir = self.model_dir / safe

        if not cell_dir.exists():
            raise ModelNotFoundError(f"No model found for cell '{cell_name}'")

        if version == "latest":
            latest_path = cell_dir / _LATEST_FILE
            if not latest_path.exists():
                raise ModelNotFoundError(f"No 'latest' version found for cell '{cell_name}'")
            version = json.loads(latest_path.read_text(encoding="utf-8"))["version"]

        version_dir = cell_dir / version
        if not version_dir.exists():
            raise ModelNotFoundError(
                f"Version '{version}' not found for cell '{cell_name}'"
            )

        try:
            lstm_model = tf.keras.models.load_model(version_dir / "model.keras")
            gbr_model = joblib.load(version_dir / "gbr.joblib")
            feature_scaler = joblib.load(version_dir / "feature_scaler.joblib")
            metadata = json.loads(
                (version_dir / "metadata.json").read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RegistryError(
                f"Failed to load model for {cell_name} v{version}: {exc}"
            ) from exc

        return {
            "lstm_model": lstm_model,
            "gbr_model": gbr_model,
            "feature_scaler": feature_scaler,
            "alpha": metadata["alpha"],
            "feature_columns": metadata["feature_columns"],
            "metadata": metadata,
        }

    def list_versions(self, cell_name: str) -> list[str]:
        """Return all available versions for a cell, sorted ascending.

        Args:
            cell_name: Viavi cell identifier.

        Returns:
            List of version strings.
        """
        safe = _safe_name(cell_name)
        cell_dir = self.model_dir / safe
        if not cell_dir.exists():
            return []
        return sorted(
            d.name for d in cell_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_model_registry.py -v
```

Expected: `4 passed`

**Step 5: Commit**

```bash
git add src/predictagent/registry/model_registry.py tests/unit/test_model_registry.py
git commit -m "feat(registry): add filesystem model registry with versioned saves"
```

---

## Task 10: FastAPI app + /forecast endpoint

**Files:**
- Create: `src/predictagent/api/app.py`
- Create: `src/predictagent/api/routers/forecast.py`
- Create: `tests/unit/test_api.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_api.py
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def app(minimal_settings):
    from predictagent.api.app import create_app
    return create_app(minimal_settings)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def valid_payload(minimal_settings):
    """Payload with exactly lookback_steps rows."""
    import time
    base_ts = 1672502400
    rows = [
        {
            "timestamp": base_ts + i * 900,
            "cell_name": "S1/B2/C1",
            "prb_used_dl": 70.0,
            "prb_avail_dl": 100.0,
            "prb_used_ul": 40.0,
            "prb_avail_ul": 100.0,
        }
        for i in range(minimal_settings.features.lookback_steps)
    ]
    return {"cell_name": "S1/B2/C1", "rows": rows}


@pytest.mark.unit
def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.unit
def test_forecast_too_few_rows_returns_422(client, valid_payload):
    payload = dict(valid_payload)
    payload["rows"] = valid_payload["rows"][:2]  # below lookback_steps
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 422


@pytest.mark.unit
def test_forecast_unknown_cell_returns_404(client, valid_payload, minimal_settings):
    payload = dict(valid_payload)
    payload["cell_name"] = "unknown/cell/99"
    payload["rows"] = [dict(r, cell_name="unknown/cell/99") for r in valid_payload["rows"]]
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 404


@pytest.mark.unit
def test_forecast_valid_request_returns_200(client, valid_payload, minimal_settings, tmp_path):
    """Mock the registry so we don't need a real model."""
    from unittest.mock import patch, MagicMock
    import numpy as np

    mock_artifacts = {
        "lstm_model": MagicMock(**{"predict.return_value": np.array([[0.72]])}),
        "gbr_model": MagicMock(**{"predict.return_value": np.array([0.68])}),
        "feature_scaler": MagicMock(**{"transform.return_value": np.zeros((minimal_settings.features.lookback_steps, 8))}),
        "alpha": 0.7,
        "feature_columns": minimal_settings.features.feature_columns,
        "metadata": {"version": "20260311_091500"},
    }

    with patch("predictagent.api.routers.forecast.get_registry") as mock_reg:
        mock_registry = MagicMock()
        mock_registry.load.return_value = mock_artifacts
        mock_reg.return_value = mock_registry
        resp = client.post("/forecast", json=valid_payload)

    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_prb_util_dl" in body
    assert body["cell_name"] == "S1/B2/C1"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_api.py -v
```

Expected: `ImportError`

**Step 3: Implement `src/predictagent/api/routers/forecast.py`**

```python
"""FastAPI router for the /forecast and /health endpoints."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException

from predictagent.config import Settings
from predictagent.exceptions import InsufficientDataError, ModelNotFoundError
from predictagent.pipeline.features import engineer_features
from predictagent.pipeline.sequencer import build_sequences
from predictagent.pipeline.trainer import predict
from predictagent.registry.model_registry import ModelRegistry
from predictagent.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_settings: Settings | None = None
_registry_cache: dict = {}


def init_router(settings: Settings) -> None:
    """Inject settings into the router at app startup."""
    global _settings
    _settings = settings


def get_registry() -> ModelRegistry:
    """Return the ModelRegistry singleton."""
    assert _settings is not None, "Router not initialised"
    return ModelRegistry(_settings.registry.model_dir)


@router.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "models_cached": len(_registry_cache)}


@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Predict DL PRB utilisation for a cell.

    The caller must supply at least `lookback_steps` telemetry rows.

    Raises:
        422: Fewer rows than lookback_steps.
        404: No trained model for the requested cell.
        500: Internal inference error.
    """
    assert _settings is not None
    min_rows = _settings.features.lookback_steps
    if len(request.rows) < min_rows:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {min_rows} rows; got {len(request.rows)}",
        )

    try:
        artifacts = get_registry().load(request.cell_name)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for cell '{request.cell_name}'",
        )

    import pandas as pd

    rows_data = [
        {
            "timestamp": r.timestamp,
            "timestamp_dt": pd.Timestamp(r.timestamp, unit="s", tz="UTC"),
            "Viavi.Cell.Name": r.cell_name,
            "RRU.PrbUsedDl": r.prb_used_dl,
            "RRU.PrbAvailDl": r.prb_avail_dl,
            "RRU.PrbUsedUl": r.prb_used_ul,
            "RRU.PrbAvailUl": r.prb_avail_ul,
            "RRC.ConnMean": r.rrc_conn_mean if r.rrc_conn_mean is not None else 0.0,
            "DRB.UEThpDl": r.drb_ue_thp_dl if r.drb_ue_thp_dl is not None else 0.0,
            "PEE.AvgPower": r.pee_avg_power if r.pee_avg_power is not None else 0.0,
        }
        for r in request.rows
    ]
    df = pd.DataFrame(rows_data)

    feature_cols = artifacts["feature_columns"]
    target_col = _settings.features.target_column

    # Compute PRB.Util.DL from raw columns if not already present
    if target_col not in df.columns:
        avail = df["RRU.PrbAvailDl"].where(df["RRU.PrbAvailDl"] != 0)
        df[target_col] = (df["RRU.PrbUsedDl"] / avail).clip(0, 1)

    try:
        df_feat = engineer_features(df, target_col, feature_cols)
    except Exception as exc:
        logger.error("Feature engineering failed for %s: %s", request.cell_name, exc)
        raise HTTPException(status_code=500, detail="Feature engineering failed")

    if len(df_feat) < _settings.features.lookback_steps:
        raise HTTPException(
            status_code=422,
            detail=f"After feature engineering only {len(df_feat)} rows remain; need {_settings.features.lookback_steps}",
        )

    # Take the last lookback_steps rows
    window_df = df_feat.tail(_settings.features.lookback_steps).reset_index(drop=True)
    feature_scaler = artifacts["feature_scaler"]
    X_raw = window_df[feature_cols].to_numpy(dtype=np.float32)
    X_scaled = feature_scaler.transform(X_raw).reshape(
        1, _settings.features.lookback_steps, len(feature_cols)
    ).astype(np.float32)

    version = artifacts["metadata"].get("version", "unknown")
    pred = predict(artifacts["lstm_model"], artifacts["gbr_model"], X_scaled, artifacts["alpha"])

    logger.info(
        "Forecast for %s: pred=%.4f, model_version=%s",
        request.cell_name, pred, version,
    )

    return ForecastResponse(
        cell_name=request.cell_name,
        forecast_horizon_minutes=_settings.features.forecast_horizon * _settings.data.rollup_minutes,
        predicted_prb_util_dl=float(pred),
        model_version=version,
    )
```

**Step 4: Implement `src/predictagent/api/app.py`**

```python
"""FastAPI application factory."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from predictagent.config import Settings
from predictagent.api.routers.forecast import router, init_router

logger = logging.getLogger(__name__)


def create_app(settings: Settings) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Validated application settings.

    Returns:
        Configured FastAPI instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = FastAPI(
        title="predictagent",
        description="Cell load forecast API",
        version="0.1.0",
    )
    init_router(settings)
    app.include_router(router)
    logger.info("App created; registry at %s", settings.registry.model_dir)
    return app
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_api.py -v
```

Expected: `4 passed`

**Step 6: Commit**

```bash
git add src/predictagent/api/ tests/unit/test_api.py
git commit -m "feat(api): add FastAPI app with /forecast and /health endpoints"
```

---

## Task 11: CLI entry points

**Files:**
- Create: `src/predictagent/cli.py`

**Step 1: Implement `src/predictagent/cli.py`**

```python
"""CLI entry points for predictagent."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("config/default.yaml")


def _parse_config_arg(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args, _ = parser.parse_known_args(argv)
    return args.config


def ingest() -> None:
    """Run the ingestion pipeline and exit."""
    from predictagent.config import load_settings
    from predictagent.pipeline.ingestor import run_ingestion

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _parse_config_arg()
    settings = load_settings(config_path)
    output = run_ingestion(settings)
    print(f"Ingestion complete → {output}")


def train() -> None:
    """Run the full training pipeline: ingest → features → sequence → train."""
    from predictagent.config import load_settings
    from predictagent.pipeline.ingestor import run_ingestion
    from predictagent.pipeline.features import run_feature_engineering
    from predictagent.pipeline.sequencer import run_sequencing
    from predictagent.pipeline.trainer import run_training

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _parse_config_arg()
    settings = load_settings(config_path)

    processed_csv = run_ingestion(settings)
    cell_features_dir = run_feature_engineering(settings, processed_csv)
    tensors_dir = run_sequencing(settings, cell_features_dir)
    all_metrics = run_training(settings, tensors_dir)
    print(f"Training complete: {len(all_metrics)} cells trained")
    for m in all_metrics:
        print(f"  {m.cell_name}: MAE={m.mae:.4f} RMSE={m.rmse:.4f} MAPE={m.mape:.2f}%")


def serve() -> None:
    """Start the FastAPI inference server."""
    import uvicorn
    from predictagent.config import load_settings
    from predictagent.api.app import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _parse_config_arg()
    settings = load_settings(config_path)
    app = create_app(settings)
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)
```

**Step 2: Verify CLI entry points are wired**

```bash
uv run predictagent-ingest --help
uv run predictagent-train --help
uv run predictagent-serve --help
```

Expected: Each prints usage without error.

**Step 3: Commit**

```bash
git add src/predictagent/cli.py
git commit -m "feat(cli): wire predictagent-ingest, predictagent-train, predictagent-serve"
```

---

## Task 12: Integration tests

Full pipeline round-trip using the `minimal_settings` fixture (tiny synthetic data, lookback=8, epochs=2).

**Files:**
- Create: `tests/integration/test_pipeline.py`
- Create: `tests/integration/test_api_integration.py`

**Step 1: Write integration tests**

```python
# tests/integration/test_pipeline.py
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
```

```python
# tests/integration/test_api_integration.py
import pytest
import numpy as np


@pytest.mark.integration
def test_api_forecast_after_training(minimal_settings):
    """POST /forecast returns 200 after a real training run."""
    from predictagent.pipeline.ingestor import run_ingestion
    from predictagent.pipeline.features import run_feature_engineering
    from predictagent.pipeline.sequencer import run_sequencing
    from predictagent.pipeline.trainer import run_training
    from predictagent.api.app import create_app
    from fastapi.testclient import TestClient

    # Run training pipeline
    processed_csv = run_ingestion(minimal_settings)
    cell_features_dir = run_feature_engineering(minimal_settings, processed_csv)
    tensors_dir = run_sequencing(minimal_settings, cell_features_dir)
    run_training(minimal_settings, tensors_dir)

    client = TestClient(create_app(minimal_settings))

    # Build a valid payload: lookback_steps rows
    base_ts = 1672502400
    rows = [
        {
            "timestamp": base_ts + i * 900,
            "cell_name": "S1/B2/C1",
            "prb_used_dl": 70.0,
            "prb_avail_dl": 100.0,
            "prb_used_ul": 40.0,
            "prb_avail_ul": 100.0,
            "rrc_conn_mean": 30.0,
            "drb_ue_thp_dl": 0.2,
            "pee_avg_power": 150.0,
        }
        for i in range(minimal_settings.features.lookback_steps)
    ]
    resp = client.post("/forecast", json={"cell_name": "S1/B2/C1", "rows": rows})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["predicted_prb_util_dl"] <= 1.5
```

**Step 2: Run integration tests**

```bash
uv run pytest tests/integration/ -v -m integration
```

Expected: `3 passed` (may take 30-60 seconds due to TF training)

**Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test(integration): add full pipeline and API round-trip integration tests"
```

---

## Task 13: Regression tests + baseline

Regression tests train on VIAVI sample data and assert metrics stay below a stored baseline.

**Files:**
- Create: `tests/regression/baseline.json`
- Create: `tests/regression/test_metrics_baseline.py`

**Step 1: Generate the baseline by running training once**

```bash
uv run predictagent-train --config config/default.yaml 2>&1 | tail -20
```

Note the MAE and RMSE printed for each cell. Use values that are ~10% higher as the baseline (safety margin).

**Step 2: Create `tests/regression/baseline.json`**

Populate with the actual values observed. Example structure — replace 0.999 with real observed + 10% slack:

```json
{
  "S1/B2/C1": {"mae": 0.999, "rmse": 0.999},
  "S1/B13/C1": {"mae": 0.999, "rmse": 0.999},
  "S1/N77/C1": {"mae": 0.999, "rmse": 0.999}
}
```

**Step 3: Write regression test**

```python
# tests/regression/test_metrics_baseline.py
import json
from pathlib import Path

import pytest


BASELINE_PATH = Path(__file__).parent / "baseline.json"


@pytest.mark.regression
def test_metrics_do_not_exceed_baseline(tmp_path):
    """Train on real VIAVI data and assert metrics stay below stored baseline."""
    from predictagent.config import load_settings
    from predictagent.pipeline.ingestor import run_ingestion
    from predictagent.pipeline.features import run_feature_engineering
    from predictagent.pipeline.sequencer import run_sequencing
    from predictagent.pipeline.trainer import run_training

    settings = load_settings(Path("config/default.yaml"))
    # Override paths to write to tmp
    settings = settings.model_copy(
        update={
            "data": settings.data.model_copy(
                update={"processed_dir": tmp_path / "processed"}
            ),
            "registry": settings.registry.model_copy(
                update={"model_dir": tmp_path / "models"}
            ),
        }
    )

    processed_csv = run_ingestion(settings)
    cell_dir = run_feature_engineering(settings, processed_csv)
    tensors_dir = run_sequencing(settings, cell_dir)
    all_metrics = run_training(settings, tensors_dir)

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    for m in all_metrics:
        if m.cell_name not in baseline:
            continue
        b = baseline[m.cell_name]
        assert m.mae <= b["mae"], (
            f"{m.cell_name} MAE {m.mae:.4f} exceeds baseline {b['mae']:.4f}"
        )
        assert m.rmse <= b["rmse"], (
            f"{m.cell_name} RMSE {m.rmse:.4f} exceeds baseline {b['rmse']:.4f}"
        )
```

**Step 4: Run regression tests (expect slow — full training)**

```bash
uv run pytest tests/regression/ -v -m regression
```

Expected: All cells pass their baseline assertions.

**Step 5: Commit**

```bash
git add tests/regression/
git commit -m "test(regression): add metric baseline assertions for real VIAVI data"
```

---

## Task 14: Final verification and clean-up

**Step 1: Run full default test suite**

```bash
uv run pytest -v
```

Expected: All unit and integration tests pass.

**Step 2: Verify no references to deleted files remain**

```bash
grep -r "cell_load_lstm\|cell_load_inference\|preprocessing.py\|0930-lstm" src/ tests/ --include="*.py"
```

Expected: No output.

**Step 3: Verify CLI entry points work end-to-end**

```bash
uv run predictagent-train --config config/default.yaml
uv run predictagent-serve --config config/default.yaml &
sleep 2
curl -s http://localhost:8000/health
kill %1
```

Expected: `{"status":"ok","models_cached":0}`

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```
