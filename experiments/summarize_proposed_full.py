from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mission_threshold_times import add_threshold_times


TELEMETRY_TOTAL_COLUMNS = [
    "obstacle_projection_calls",
    "obstacle_projection_agent_adjustments",
    "pairwise_projection_calls",
    "pairwise_projection_agent_adjustments",
    "projection_agent_adjustments",
    "pre_projection_obstacle_contacts",
    "pre_projection_obstacle_violation_agent_steps",
    "pre_projection_pairwise_violation_pairs",
]


def _safe_sum(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _safe_max(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else 0.0


def _finite_median(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _finite_max(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def _summary_record(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        raise ValueError("Cannot summarize an empty proposed-controller dataframe")
    t99_values = pd.to_numeric(df.get("T99_s"), errors="coerce")
    record: dict[str, Any] = {
        "runs": int(len(df)),
        "safety_success_rate": float(
            (
                (pd.to_numeric(df["cumulative_collision_pairs"], errors="coerce").fillna(0) == 0)
                & (pd.to_numeric(df["cumulative_obstacle_contacts"], errors="coerce").fillna(0) == 0)
            ).mean()
        ),
        "mission_success_rate_T90_metric": float((pd.to_numeric(df["final_reached_fraction"], errors="coerce") >= 0.90).mean()),
        "min_final_reached_fraction": float(pd.to_numeric(df["final_reached_fraction"], errors="coerce").min()),
        "median_final_reached_fraction": float(pd.to_numeric(df["final_reached_fraction"], errors="coerce").median()),
        "total_collision_pairs": _safe_sum(df, "cumulative_collision_pairs"),
        "total_obstacle_contacts": _safe_sum(df, "cumulative_obstacle_contacts"),
        "median_T90_s": _finite_median(df, "T90_s"),
        "median_T95_s": _finite_median(df, "T95_s"),
        "median_T99_s": _finite_median(df, "T99_s"),
        "max_T99_s": _finite_max(df, "T99_s"),
        "T99_reached_runs": int(t99_values.notna().sum()) if t99_values is not None else 0,
        "pre_projection_obstacle_max_penetration_m": _safe_max(df, "pre_projection_obstacle_max_penetration_m"),
        "pre_projection_pairwise_max_penetration_m": _safe_max(df, "pre_projection_pairwise_max_penetration_m"),
        "median_runtime_s": _finite_median(df, "runtime_s"),
        "total_runtime_s_sum": float(pd.to_numeric(df.get("runtime_s"), errors="coerce").fillna(0).sum()),
    }
    for column in TELEMETRY_TOTAL_COLUMNS:
        record[f"{column}_total"] = _safe_sum(df, column)
    return record


def _write_group_csv(df: pd.DataFrame, group_col: str, path: Path) -> None:
    rows = []
    for key, group in df.groupby(group_col, sort=False):
        row = {group_col: key}
        row.update(_summary_record(group))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize(results_dir: Path) -> None:
    run_index = results_dir / "run_index.csv"
    if not run_index.exists():
        raise FileNotFoundError(run_index)
    df = pd.read_csv(run_index)
    df = add_threshold_times(df, results_dir)
    df.to_csv(results_dir / "run_index_with_thresholds.csv", index=False)

    primary_df = df[df["seed"] != 99].copy() if "seed" in df.columns else df.copy()
    summary = {
        "all_256": _summary_record(df),
        "primary_excluding_seed99_255": _summary_record(primary_df),
    }
    (results_dir / "summary_overall_final_command_filter.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_group_csv(primary_df, "scenario", results_dir / "summary_by_scenario_final_command_filter.csv")
    _write_group_csv(primary_df, "n_agents", results_dir / "summary_by_n_agents_final_command_filter.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proposed-controller command-filter summary files.")
    parser.add_argument("--results", type=Path, default=Path("results/proposed_full_sweep"))
    args = parser.parse_args()
    summarize(args.results)
    print(f"Wrote proposed-controller summaries to {args.results}")


if __name__ == "__main__":
    main()
