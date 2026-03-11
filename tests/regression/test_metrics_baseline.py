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
