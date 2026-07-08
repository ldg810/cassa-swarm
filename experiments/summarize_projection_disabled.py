from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.mission_threshold_times import add_threshold_times


ROOT = Path(__file__).resolve().parents[1]

GROUP_ORDER = [
    ("Overall", None),
    ("Obstacle corridor", "corridor_obstacles"),
    ("Crossing traffic", "crossing_traffic"),
    ("Merge--split", "merge_split"),
    ("Open flock", "open_flock"),
]


METRIC_ORDER = [
    "runs",
    "safety_success_rate",
    "inter_agent_collision_pair_samples",
    "obstacle_contact_samples",
    "T90_median_s",
    "T90_reached_runs",
    "T95_median_s",
    "T95_reached_runs",
    "T99_median_s",
    "T99_reached_runs",
    "mean_reached_fraction",
    "obstacle_clearance_violation_agent_steps",
    "pairwise_clearance_violation_pairs",
    "clearance_envelope_violation_steps_total",
    "max_conservative_envelope_penetration_m",
    "obstacle_max_conservative_penetration_m",
    "pairwise_max_conservative_penetration_m",
    "min_obstacle_contact_boundary_margin_m",
    "min_inter_agent_distance_m",
    "obstacle_projection_agent_adjustments",
    "pairwise_projection_agent_adjustments",
]


def _num(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else math.nan


def _max_or_zero(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if len(values) else 0.0


def _min_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if len(values) else math.nan


def _derived_obstacle_contact_boundary_margin(
    group: pd.DataFrame,
    obstacle_contact_margin_offset: float | None,
) -> float:
    direct = _min_or_nan(group.get("min_obstacle_contact_boundary_margin_m", pd.Series(dtype=float)))
    if not math.isnan(direct):
        return direct
    if obstacle_contact_margin_offset is None:
        return math.nan
    conservative_margin = _min_or_nan(
        group.get("pre_projection_obstacle_min_clearance_margin_m", pd.Series(dtype=float))
    )
    if math.isnan(conservative_margin):
        return math.nan
    return float(conservative_margin + obstacle_contact_margin_offset)


def _summarize(
    group: pd.DataFrame,
    obstacle_contact_margin_offset: float | None = None,
) -> dict[str, Any]:
    t90 = pd.to_numeric(group["T90_s"], errors="coerce")
    t95 = pd.to_numeric(group["T95_s"], errors="coerce")
    t99 = pd.to_numeric(group["T99_s"], errors="coerce")
    obstacle_steps = int(_num(group, "pre_projection_obstacle_violation_agent_steps").sum())
    pairwise_pairs = int(_num(group, "pre_projection_pairwise_violation_pairs").sum())
    obstacle_max = _max_or_zero(group.get("pre_projection_obstacle_max_penetration_m", pd.Series(dtype=float)))
    pairwise_max = _max_or_zero(group.get("pre_projection_pairwise_max_penetration_m", pd.Series(dtype=float)))
    return {
        "runs": int(len(group)),
        "safety_success_rate": float(
            ((_num(group, "cumulative_collision_pairs") == 0) & (_num(group, "cumulative_obstacle_contacts") == 0)).mean()
        ),
        "inter_agent_collision_pair_samples": int(_num(group, "cumulative_collision_pairs").sum()),
        "obstacle_contact_samples": int(_num(group, "cumulative_obstacle_contacts").sum()),
        "T90_median_s": _median_or_nan(t90),
        "T90_reached_runs": int(t90.notna().sum()),
        "T95_median_s": _median_or_nan(t95),
        "T95_reached_runs": int(t95.notna().sum()),
        "T99_median_s": _median_or_nan(t99),
        "T99_reached_runs": int(t99.notna().sum()),
        "mean_reached_fraction": float(_num(group, "final_reached_fraction").mean()),
        "obstacle_clearance_violation_agent_steps": obstacle_steps,
        "pairwise_clearance_violation_pairs": pairwise_pairs,
        "clearance_envelope_violation_steps_total": obstacle_steps + pairwise_pairs,
        "max_conservative_envelope_penetration_m": max(obstacle_max, pairwise_max),
        "obstacle_max_conservative_penetration_m": obstacle_max,
        "pairwise_max_conservative_penetration_m": pairwise_max,
        "min_obstacle_contact_boundary_margin_m": _derived_obstacle_contact_boundary_margin(
            group,
            obstacle_contact_margin_offset,
        ),
        "min_inter_agent_distance_m": _min_or_nan(group.get("min_inter_agent_distance_m", pd.Series(dtype=float))),
        "obstacle_projection_agent_adjustments": int(_num(group, "obstacle_projection_agent_adjustments").sum()),
        "pairwise_projection_agent_adjustments": int(_num(group, "pairwise_projection_agent_adjustments").sum()),
    }


def _read_index(results_dir: Path) -> pd.DataFrame | None:
    threshold_index_path = results_dir / "run_index_with_thresholds.csv"
    index_path = results_dir / "run_index.csv"
    if threshold_index_path.exists():
        return pd.read_csv(threshold_index_path)
    if not index_path.exists():
        return None
    df = pd.read_csv(index_path)
    if not {"T90_s", "T95_s", "T99_s"}.issubset(df.columns):
        df = add_threshold_times(df, results_dir)
    return df


def _from_proposed_record(record: dict[str, Any]) -> dict[str, Any]:
    obstacle_steps = int(record.get("pre_projection_obstacle_violation_agent_steps_total", 0))
    pairwise_pairs = int(record.get("pre_projection_pairwise_violation_pairs_total", 0))
    obstacle_max = float(record.get("pre_projection_obstacle_max_penetration_m", 0.0))
    pairwise_max = float(record.get("pre_projection_pairwise_max_penetration_m", 0.0))
    runs = int(record.get("runs", 0))
    return {
        "runs": runs,
        "safety_success_rate": record.get("safety_success_rate", math.nan),
        "inter_agent_collision_pair_samples": int(record.get("total_collision_pairs", 0)),
        "obstacle_contact_samples": int(record.get("total_obstacle_contacts", 0)),
        "T90_median_s": record.get("median_T90_s", math.nan),
        "T90_reached_runs": record.get("T90_reached_runs", runs),
        "T95_median_s": record.get("median_T95_s", math.nan),
        "T95_reached_runs": record.get("T95_reached_runs", runs),
        "T99_median_s": record.get("median_T99_s", math.nan),
        "T99_reached_runs": record.get("T99_reached_runs", runs),
        "mean_reached_fraction": record.get("mean_reached_fraction", math.nan),
        "obstacle_clearance_violation_agent_steps": obstacle_steps,
        "pairwise_clearance_violation_pairs": pairwise_pairs,
        "clearance_envelope_violation_steps_total": obstacle_steps + pairwise_pairs,
        "max_conservative_envelope_penetration_m": max(obstacle_max, pairwise_max),
        "obstacle_max_conservative_penetration_m": obstacle_max,
        "pairwise_max_conservative_penetration_m": pairwise_max,
        "min_obstacle_contact_boundary_margin_m": record.get("min_obstacle_contact_boundary_margin_m", math.nan),
        "min_inter_agent_distance_m": record.get("min_inter_agent_distance_m", math.nan),
        "obstacle_projection_agent_adjustments": int(record.get("obstacle_projection_agent_adjustments_total", 0)),
        "pairwise_projection_agent_adjustments": int(record.get("pairwise_projection_agent_adjustments_total", 0)),
    }


def _load_on_reference(on_reference_results: Path) -> dict[str, dict[str, Any]]:
    raw_df = _read_index(on_reference_results)
    if raw_df is not None:
        # The primary conservative obstacle envelope is the actual contact
        # boundary plus the 0.34 m obstacle margin and 0.02 m projection buffer.
        obstacle_contact_margin_offset = 0.36
        if "seed" in raw_df.columns:
            raw_df = raw_df[pd.to_numeric(raw_df["seed"], errors="coerce") != 99].copy()
        return {
            group_label: _summarize(
                raw_df if scenario is None else raw_df[raw_df["scenario"] == scenario],
                obstacle_contact_margin_offset=obstacle_contact_margin_offset,
            )
            for group_label, scenario in GROUP_ORDER
        }

    overall_path = on_reference_results / "summary_overall_final_command_filter.json"
    scenario_path = on_reference_results / "summary_by_scenario_final_command_filter.csv"
    if not overall_path.exists() or not scenario_path.exists():
        raise FileNotFoundError(
            "ON reference must contain run_index_with_thresholds.csv/run_index.csv or "
            "summary_overall_final_command_filter.json plus summary_by_scenario_final_command_filter.csv"
        )

    overall = json.loads(overall_path.read_text(encoding="utf-8"))["primary_excluding_seed99_255"]
    scenarios = pd.read_csv(scenario_path).set_index("scenario")
    reference = {"Overall": _from_proposed_record(overall)}
    for group_label, scenario in GROUP_ORDER[1:]:
        reference[group_label] = _from_proposed_record(scenarios.loc[scenario].to_dict())
    return reference


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    if digits == 0:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def _write_markdown(summary: pd.DataFrame, path: Path) -> None:
    selected = [
        "safety_success_rate",
        "inter_agent_collision_pair_samples",
        "obstacle_contact_samples",
        "T90_median_s",
        "T95_median_s",
        "T99_median_s",
        "mean_reached_fraction",
        "clearance_envelope_violation_steps_total",
        "max_conservative_envelope_penetration_m",
        "min_obstacle_contact_boundary_margin_m",
        "min_inter_agent_distance_m",
    ]
    lines = [
        "# Projection-Disabled CASSA Sweep Summary",
        "",
        "Projection-disabled run: 255 runs, seeds 0-4, with command-level obstacle braking, local safety, and speed governors enabled while post-integration projection correction is disabled.",
        "",
        "Projection-ON reference columns were regenerated on 2026-07-08 from `results/projection_on_regression_2026-07-08` and now match the manuscript.",
        "",
    ]
    header = ["Group"]
    count_metrics = {
        "inter_agent_collision_pair_samples",
        "obstacle_contact_samples",
        "clearance_envelope_violation_steps_total",
    }
    for metric in selected:
        header.extend([f"OFF {metric}", f"ON ref {metric}"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for _, row in summary.iterrows():
        values = [str(row["group"])]
        for metric in selected:
            digits = 0 if metric in count_metrics else 3
            values.append(_fmt(row[f"projection_off_{metric}"], digits))
            values.append(_fmt(row[f"projection_on_reference_{metric}"], digits))
        lines.append("| " + " | ".join(values) + " |")

    lines += [
        "",
        "## Projection Counters",
        "",
        "| Group | OFF obstacle adj. | ON ref obstacle adj. | OFF pairwise adj. | ON ref pairwise adj. |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {group} | {off_obs} | {on_obs} | {off_pair} | {on_pair} |".format(
                group=row["group"],
                off_obs=_fmt(row["projection_off_obstacle_projection_agent_adjustments"], 0),
                on_obs=_fmt(row["projection_on_reference_obstacle_projection_agent_adjustments"], 0),
                off_pair=_fmt(row["projection_off_pairwise_projection_agent_adjustments"], 0),
                on_pair=_fmt(row["projection_on_reference_pairwise_projection_agent_adjustments"], 0),
            )
        )
    lines += [
        "",
        "## Files",
        "",
        "- `run_index.csv`: raw per-run metrics from the projection-disabled sweep.",
        "- `run_index_with_thresholds.csv`: per-run metrics plus interpolated T90/T95/T99 crossing times.",
        "- `projection_disabled_summary.csv`: wide aggregate/per-scenario OFF vs ON-reference comparison.",
        "- `projection_disabled_summary_long.csv`: long-form metric table for paper/table reshaping.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(results_dir: Path, on_reference_results: Path) -> None:
    df = _read_index(results_dir)
    if df is None:
        raise FileNotFoundError(results_dir / "run_index.csv")
    on_reference = _load_on_reference(on_reference_results)

    rows: list[dict[str, Any]] = []
    for group_label, scenario in GROUP_ORDER:
        group = df if scenario is None else df[df["scenario"] == scenario]
        off = _summarize(group)
        on = on_reference[group_label]
        row = {"group": group_label, "scenario_key": scenario or "overall"}
        for metric in METRIC_ORDER:
            row[f"projection_off_{metric}"] = off.get(metric)
            row[f"projection_on_reference_{metric}"] = on.get(metric)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "projection_disabled_summary.csv", index=False)

    long_rows = []
    for _, row in summary.iterrows():
        for metric in METRIC_ORDER:
            long_rows.append({
                "group": row["group"],
                "scenario_key": row["scenario_key"],
                "metric": metric,
                "projection_off": row[f"projection_off_{metric}"],
                "projection_on_reference": row[f"projection_on_reference_{metric}"],
            })
    pd.DataFrame(long_rows).to_csv(results_dir / "projection_disabled_summary_long.csv", index=False)
    _write_markdown(summary, results_dir / "projection_disabled_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a projection-disabled CASSA full sweep.")
    parser.add_argument("--results", type=Path, default=Path("results/projection_disabled_sweep"))
    parser.add_argument(
        "--on-reference-results",
        type=Path,
        default=ROOT / "expected_results" / "proposed_full_sweep",
        help="Projection-ON raw results directory or compact proposed_full_sweep reference snapshot.",
    )
    args = parser.parse_args()
    summarize(args.results, args.on_reference_results)
    print(f"Wrote projection-disabled summaries to {args.results}")


if __name__ == "__main__":
    main()
