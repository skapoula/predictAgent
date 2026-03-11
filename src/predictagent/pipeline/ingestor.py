"""Raw data ingestion: load, validate, enrich, rollup CellReports CSV."""
from __future__ import annotations

import logging
from pathlib import Path

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
