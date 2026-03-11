#!/usr/bin/env python3
"""Prepare Viavi CellReports dataset with sector enrichment and temporal rollups."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SOURCE = Path("CellReports.csv")
OUTPUT_SORTED = Path("CellReports_1.csv")
ROLLUP_TARGETS = (5, 10, 15, 60)
SITE_FILTER_PREFIX = "S1/"


def derive_site_sector(cell_name: str) -> str | None:
    """Return composite Site/Sect identifier from the Viavi cell name."""
    if not isinstance(cell_name, str):
        return None
    parts = [segment.strip() for segment in cell_name.split("/") if segment]
    if len(parts) < 3:
        return None
    site = parts[0]
    cell_component = parts[-1]
    if not cell_component:
        return None
    # Sector is defined by the last character in the cell component (e.g. C1 -> 1).
    sector_char = cell_component[-1]
    if not sector_char.isdigit():
        digits = "".join(ch for ch in cell_component if ch.isdigit())
        sector_char = digits[-1] if digits else None
    if not sector_char:
        return None
    return f"{site}/Sect{sector_char}"


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "Viavi.Cell.Name" not in df.columns:
        raise ValueError("Source file must contain 'timestamp' and 'Viavi.Cell.Name' columns")
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "Viavi.Cell.Name"])
    df["timestamp"] = df["timestamp"].astype("int64")
    df["SiteSector"] = df["Viavi.Cell.Name"].apply(derive_site_sector)
    metric_columns = [c for c in df.columns if c not in {"timestamp", "Viavi.Cell.Name", "SiteSector"}]
    ordered_columns = ["timestamp", "Viavi.Cell.Name", "SiteSector", *metric_columns]
    df = df[ordered_columns]
    df = df.sort_values(["Viavi.Cell.Name", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def write_sorted_dataset(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)


def add_prb_utilization(df: pd.DataFrame) -> pd.DataFrame:
    required = {"RRU.PrbUsedDl", "RRU.PrbAvailDl", "RRU.PrbUsedUl", "RRU.PrbAvailUl"}
    missing = required.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns required for PRB utilisation: {missing_list}")

    result = df.copy()

    used_dl = pd.to_numeric(result["RRU.PrbUsedDl"], errors="coerce")
    avail_dl = pd.to_numeric(result["RRU.PrbAvailDl"], errors="coerce")
    used_ul = pd.to_numeric(result["RRU.PrbUsedUl"], errors="coerce")
    avail_ul = pd.to_numeric(result["RRU.PrbAvailUl"], errors="coerce")

    result["PRB.Util.DL"] = used_dl.divide(avail_dl.where(avail_dl != 0))
    result["PRB.Util.UL"] = used_ul.divide(avail_ul.where(avail_ul != 0))

    return result


def write_site_subset(df: pd.DataFrame, target: Path, site_prefix: str = SITE_FILTER_PREFIX) -> None:
    if "Viavi.Cell.Name" not in df.columns:
        raise ValueError("Dataset must include 'Viavi.Cell.Name' column for site filtering")

    filtered = df[df["Viavi.Cell.Name"].astype(str).str.startswith(site_prefix, na=False)]
    enriched = add_prb_utilization(filtered)

    target.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(target, index=False)


def rollup_metrics(df: pd.DataFrame, minutes: int, target: Path) -> pd.DataFrame:
    if minutes <= 0:
        raise ValueError("Aggregation interval must be positive")
    df_roll = df.copy()
    dt_index = pd.to_datetime(df_roll["timestamp"], unit="s", utc=True)
    df_roll["interval_start"] = dt_index.dt.floor(f"{minutes}min")
    metric_columns = [c for c in df_roll.columns if c not in {"timestamp", "Viavi.Cell.Name", "SiteSector", "interval_start"}]
    grouped = (
        df_roll.groupby(["Viavi.Cell.Name", "SiteSector", "interval_start"], dropna=False)[metric_columns]
        .mean()
        .reset_index()
    )
    grouped["timestamp"] = grouped["interval_start"].astype("int64") // 10**9
    ordered_columns = ["timestamp", "Viavi.Cell.Name", "SiteSector", *metric_columns]
    grouped = grouped[ordered_columns].sort_values(["Viavi.Cell.Name", "timestamp"], kind="mergesort")
    target.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(target, index=False)

    return grouped


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the input CellReports CSV (default: CellReports.csv)",
    )
    parser.add_argument(
        "--sorted-output",
        type=Path,
        default=OUTPUT_SORTED,
        help="Output path for the sorted dataset with SiteSector column",
    )
    parser.add_argument(
        "--rollups",
        type=int,
        nargs="*",
        default=list(ROLLUP_TARGETS),
        help="List of rollup intervals in minutes (default: 5 10 15 60)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df_sorted = load_source(args.source)
    write_sorted_dataset(df_sorted, args.sorted_output)
    sorted_site_target = args.sorted_output.with_name(
        f"{args.sorted_output.stem}_S1{args.sorted_output.suffix or ''}"
    )
    write_site_subset(df_sorted, sorted_site_target)
    for minutes in args.rollups:
        rollup_target = Path(f"CellReports_{minutes}.csv")
        rolled = rollup_metrics(df_sorted, minutes, rollup_target)
        site_target = Path(f"CellReports_{minutes}_S1.csv")
        write_site_subset(rolled, site_target)


if __name__ == "__main__":
    main()
