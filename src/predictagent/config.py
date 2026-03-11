"""Configuration loading and validation for predictagent."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

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
