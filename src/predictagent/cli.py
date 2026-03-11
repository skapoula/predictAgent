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
