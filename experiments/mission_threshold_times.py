from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


THRESHOLD_COLUMNS: Mapping[str, float] = {
    "T90_s": 0.90,
    "T95_s": 0.95,
    "T99_s": 0.99,
}


def threshold_crossing_time_s(timeseries: pd.DataFrame, threshold: float) -> float:
    """Return interpolated time when reached_fraction first crosses threshold."""
    if "t_s" not in timeseries.columns or "reached_fraction" not in timeseries.columns:
        return math.nan
    work = timeseries[["t_s", "reached_fraction"]].copy()
    work["t_s"] = pd.to_numeric(work["t_s"], errors="coerce")
    work["reached_fraction"] = pd.to_numeric(work["reached_fraction"], errors="coerce")
    work = work.dropna().sort_values("t_s")
    if work.empty:
        return math.nan

    times = work["t_s"].to_numpy(dtype=float)
    fractions = work["reached_fraction"].to_numpy(dtype=float)
    if fractions[0] >= threshold:
        return float(times[0])

    for idx in range(1, len(times)):
        prev_fraction = fractions[idx - 1]
        curr_fraction = fractions[idx]
        if curr_fraction < threshold:
            continue
        prev_time = times[idx - 1]
        curr_time = times[idx]
        denom = curr_fraction - prev_fraction
        if denom <= 0.0:
            return float(curr_time)
        ratio = (threshold - prev_fraction) / denom
        return float(prev_time + ratio * (curr_time - prev_time))
    return math.nan


def run_dir_for_row(results_dir: Path, row: pd.Series) -> Path:
    """Resolve ordinary and component-ablation run directory layouts."""
    if "ablation_variant" in row.index and "base_run_id" in row.index:
        variant = row.get("ablation_variant")
        base_run_id = row.get("base_run_id")
        if pd.notna(variant) and pd.notna(base_run_id):
            return results_dir / str(variant) / str(base_run_id)
    return results_dir / str(row["run_id"])


def add_threshold_times(run_index: pd.DataFrame, results_dir: str | Path) -> pd.DataFrame:
    out = run_index.copy()
    root = Path(results_dir)
    for column in THRESHOLD_COLUMNS:
        out[column] = np.nan

    for idx, row in out.iterrows():
        timeseries_path = run_dir_for_row(root, row) / "timeseries.csv"
        if not timeseries_path.exists():
            continue
        timeseries = pd.read_csv(timeseries_path)
        for column, threshold in THRESHOLD_COLUMNS.items():
            out.at[idx, column] = threshold_crossing_time_s(timeseries, threshold)
    return out


def formatted_threshold_time(
    values: pd.Series,
    total_runs: int,
    horizon_s: float | None,
) -> str:
    finite = pd.to_numeric(values, errors="coerce").dropna().sort_values(ignore_index=True)
    reached = int(len(finite))
    if reached == 0:
        if horizon_s is None or math.isnan(float(horizon_s)):
            return f"not reached (0/{total_runs})"
        return f">{float(horizon_s):.1f} (0/{total_runs})"
    median_time = censored_median_time_s(finite, total_runs, horizon_s)
    if math.isnan(median_time):
        return f"not reached ({reached}/{total_runs})"
    if horizon_s is not None and not math.isnan(float(horizon_s)) and median_time >= float(horizon_s):
        return f">{float(horizon_s):.1f} ({reached}/{total_runs})"
    return f"{median_time:.1f} ({reached}/{total_runs})"


def censored_median_time_s(
    values: pd.Series,
    total_runs: int,
    horizon_s: float | None,
) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().sort_values(ignore_index=True)
    reached = int(len(finite))
    if reached == 0:
        return float(horizon_s) if horizon_s is not None and not math.isnan(float(horizon_s)) else math.nan
    median_low = (total_runs - 1) // 2
    median_high = total_runs // 2
    if median_high >= reached:
        return float(horizon_s) if horizon_s is not None and not math.isnan(float(horizon_s)) else math.nan
    return float((finite.iloc[median_low] + finite.iloc[median_high]) / 2.0)


def add_formatted_threshold_columns(
    table: pd.DataFrame,
    source: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    summaries = []
    for keys, group in source.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        total = int(len(group))
        horizon = float(group["horizon_s"].max()) if "horizon_s" in group.columns and not group.empty else math.nan
        for column in THRESHOLD_COLUMNS:
            record[column.replace("_s", "")] = formatted_threshold_time(group[column], total, horizon)
        summaries.append(record)
    time_table = pd.DataFrame(summaries)
    return table.merge(time_table, on=group_cols, how="left")
