from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.mission_threshold_times import add_threshold_times


GROUP_ORDER = [
    ("Overall", None),
    ("Obstacle corridor", "corridor_obstacles"),
    ("Crossing traffic", "crossing_traffic"),
    ("Merge--split", "merge_split"),
    ("Open flock", "open_flock"),
]


ON_REFERENCE: dict[str, dict[str, Any]] = {
    "Overall": {
        "runs": 255,
        "safety_success_rate": 1.0,
        "inter_agent_collision_pair_samples": 0,
        "obstacle_contact_samples": 0,
        "T90_median_s": 59.92857142857143,
        "T90_reached_runs": 255,
        "T95_median_s": 61.875,
        "T95_reached_runs": 255,
        "T99_median_s": 66.0,
        "T99_reached_runs": 255,
        "mean_reached_fraction": 0.9995223529411765,
        "obstacle_clearance_violation_agent_steps": 3571,
        "pairwise_clearance_violation_pairs": 9,
        "clearance_envelope_violation_steps_total": 3580,
        "max_conservative_envelope_penetration_m": 0.042703204824753,
        "obstacle_max_conservative_penetration_m": 0.042703204824753,
        "pairwise_max_conservative_penetration_m": 0.010169639279825,
        "min_obstacle_contact_boundary_margin_m": 0.317297,
        "min_inter_agent_distance_m": 0.140001,
        "obstacle_projection_agent_adjustments": 3571,
        "pairwise_projection_agent_adjustments": 18,
    },
    "Obstacle corridor": {
        "runs": 125,
        "safety_success_rate": 1.0,
        "inter_agent_collision_pair_samples": 0,
        "obstacle_contact_samples": 0,
        "T90_median_s": 68.75,
        "T90_reached_runs": 125,
        "T95_median_s": 71.5,
        "T95_reached_runs": 125,
        "T99_median_s": 76.5,
        "T99_reached_runs": 125,
        "mean_reached_fraction": 0.999056,
        "obstacle_clearance_violation_agent_steps": 3472,
        "pairwise_clearance_violation_pairs": 9,
        "clearance_envelope_violation_steps_total": 3481,
        "max_conservative_envelope_penetration_m": 0.042703204824753,
        "obstacle_max_conservative_penetration_m": 0.042703204824753,
        "pairwise_max_conservative_penetration_m": 0.010169639279825,
        "min_obstacle_contact_boundary_margin_m": 0.317297,
        "min_inter_agent_distance_m": 0.140001,
        "obstacle_projection_agent_adjustments": 3472,
        "pairwise_projection_agent_adjustments": 18,
    },
    "Crossing traffic": {
        "runs": 90,
        "safety_success_rate": 1.0,
        "inter_agent_collision_pair_samples": 0,
        "obstacle_contact_samples": 0,
        "T90_median_s": 54.6875,
        "T90_reached_runs": 90,
        "T95_median_s": 55.725,
        "T95_reached_runs": 90,
        "T99_median_s": 58.25,
        "T99_reached_runs": 90,
        "mean_reached_fraction": 0.9999777777777779,
        "obstacle_clearance_violation_agent_steps": 99,
        "pairwise_clearance_violation_pairs": 0,
        "clearance_envelope_violation_steps_total": 99,
        "max_conservative_envelope_penetration_m": 0.0358961633810535,
        "obstacle_max_conservative_penetration_m": 0.0358961633810535,
        "pairwise_max_conservative_penetration_m": 0.0,
        "min_obstacle_contact_boundary_margin_m": 0.324104,
        "min_inter_agent_distance_m": 0.220035,
        "obstacle_projection_agent_adjustments": 99,
        "pairwise_projection_agent_adjustments": 0,
    },
    "Merge--split": {
        "runs": 20,
        "safety_success_rate": 1.0,
        "inter_agent_collision_pair_samples": 0,
        "obstacle_contact_samples": 0,
        "T90_median_s": 58.28125,
        "T90_reached_runs": 20,
        "T95_median_s": 59.10416666666667,
        "T95_reached_runs": 20,
        "T99_median_s": 60.625,
        "T99_reached_runs": 20,
        "mean_reached_fraction": 1.0,
        "obstacle_clearance_violation_agent_steps": 0,
        "pairwise_clearance_violation_pairs": 0,
        "clearance_envelope_violation_steps_total": 0,
        "max_conservative_envelope_penetration_m": 0.0,
        "obstacle_max_conservative_penetration_m": 0.0,
        "pairwise_max_conservative_penetration_m": 0.0,
        "min_obstacle_contact_boundary_margin_m": 0.320632,
        "min_inter_agent_distance_m": 0.220751,
        "obstacle_projection_agent_adjustments": 0,
        "pairwise_projection_agent_adjustments": 0,
    },
    "Open flock": {
        "runs": 20,
        "safety_success_rate": 1.0,
        "inter_agent_collision_pair_samples": 0,
        "obstacle_contact_samples": 0,
        "T90_median_s": 51.725,
        "T90_reached_runs": 20,
        "T95_median_s": 52.3,
        "T95_reached_runs": 20,
        "T99_median_s": 53.70833333333333,
        "T99_reached_runs": 20,
        "mean_reached_fraction": 0.9999,
        "obstacle_clearance_violation_agent_steps": 0,
        "pairwise_clearance_violation_pairs": 0,
        "clearance_envelope_violation_steps_total": 0,
        "max_conservative_envelope_penetration_m": 0.0,
        "obstacle_max_conservative_penetration_m": 0.0,
        "pairwise_max_conservative_penetration_m": 0.0,
        "min_obstacle_contact_boundary_margin_m": math.nan,
        "min_inter_agent_distance_m": 0.253521,
        "obstacle_projection_agent_adjustments": 0,
        "pairwise_projection_agent_adjustments": 0,
    },
}


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


def _summarize(group: pd.DataFrame) -> dict[str, Any]:
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
        "min_obstacle_contact_boundary_margin_m": _min_or_nan(group.get("min_obstacle_contact_boundary_margin_m", pd.Series(dtype=float))),
        "min_inter_agent_distance_m": _min_or_nan(group.get("min_inter_agent_distance_m", pd.Series(dtype=float))),
        "obstacle_projection_agent_adjustments": int(_num(group, "obstacle_projection_agent_adjustments").sum()),
        "pairwise_projection_agent_adjustments": int(_num(group, "pairwise_projection_agent_adjustments").sum()),
    }


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
        "Projection-ON timing reference columns use the live manuscript-consistent regression values. Projection-count reference columns use the published manuscript/packaged reference values, because the live ON regression adds 205 merge-split obstacle adjustments.",
        "",
        "Note: `expected_results/proposed_full_sweep` is stale for merge-split timing (T99=52.354 s there versus 60.625 s in live ON and 60.6 s in the manuscript table). It is still consistent with the manuscript projection-audit count for merge-split (0 adjustments).",
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


def summarize(results_dir: Path) -> None:
    index_path = results_dir / "run_index.csv"
    threshold_index_path = results_dir / "run_index_with_thresholds.csv"
    if threshold_index_path.exists():
        df = pd.read_csv(threshold_index_path)
    elif index_path.exists():
        df = pd.read_csv(index_path)
    else:
        raise FileNotFoundError(index_path)
    if not {"T90_s", "T95_s", "T99_s"}.issubset(df.columns):
        df = add_threshold_times(df, results_dir)
    df.to_csv(results_dir / "run_index_with_thresholds.csv", index=False)

    rows: list[dict[str, Any]] = []
    for group_label, scenario in GROUP_ORDER:
        group = df if scenario is None else df[df["scenario"] == scenario]
        off = _summarize(group)
        on = ON_REFERENCE[group_label]
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
    args = parser.parse_args()
    summarize(args.results)
    print(f"Wrote projection-disabled summaries to {args.results}")


if __name__ == "__main__":
    main()
