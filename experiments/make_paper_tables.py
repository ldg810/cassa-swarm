from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mission_threshold_times import add_formatted_threshold_columns, add_threshold_times


SCENARIO_LABELS = {
    "corridor_obstacles": "Obstacle corridor",
    "crossing_traffic": "Crossing traffic",
    "merge_split": "Merge--split",
    "open_flock": "Open flock",
}

SHORT_SCENARIO_LABELS = {
    "corridor_obstacles": "Corridor",
    "crossing_traffic": "Crossing",
    "merge_split": "Merge--split",
    "open_flock": "Open flock",
}

CONTROLLER_LABELS = {
    "proposed": "CASSA (Proposed)",
    "decentralized_mpc": "Short-horizon predictive surrogate",
    "cbf_qp": "CBF-QP",
    "cbf_qp_osqp": "CBF-QP (OSQP pkg.)",
    "orca": "ORCA/RVO",
    "rvo2_external": "RVO2 package",
    "mader_like": "MADER-inspired surrogate",
    "apf": "APF",
    "boids": "Boids",
    "proposed_no_shield": "No shield",
}

CONTROLLER_ORDER = [
    "proposed",
    "decentralized_mpc",
    "cbf_qp_osqp",
    "rvo2_external",
    "mader_like",
    "apf",
    "boids",
    "proposed_no_shield",
]

TABLE_CONTROLLER_EXCLUDE = {"cbf_qp", "orca"}

CAPTIONS = {
    "table_1_controller_comparison": (
        "Matched 120 s controller comparison under 4 Hz updates.",
        "tab:controller-comparison",
    ),
    "table_2_proposed_by_scenario": (
        "Scenario-wise CASSA performance under 4 Hz updates.",
        "tab:proposed-scenarios",
    ),
    "table_4_seed_extension": (
        "Hard-condition seed-extension results for CASSA.",
        "tab:seed-extension",
    ),
    "table_5_parameter_summary": (
        "Compact experimental settings.",
        "tab:parameter-summary",
    ),
    "table_6_projection_audit": (
        "Command-filter and projection telemetry for the final 4 Hz CASSA sweep.",
        "tab:projection-audit",
    ),
    "table_9_crazyflow_replay": (
        "Crazyflow replay of dynamics-aware CASSA references.",
        "tab:crazyflow-replay",
    ),
}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path / "run_index.csv")
    df = add_threshold_times(df, path)
    df["safety_success"] = (
        (df["cumulative_collision_pairs"] == 0)
        & (df["cumulative_obstacle_contacts"] == 0)
    )
    return df


def _exclude_visualization_runs(df: pd.DataFrame) -> pd.DataFrame:
    # Seed 99 is reserved by the sweep configs for the saved trajectory sample.
    return df[df["seed"] != 99].copy()


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    table = (
        df.groupby(group_cols, sort=False)
        .agg(
            Runs=("run_id", "count"),
            **{
                "Safety success": ("safety_success", "mean"),
                "Min reached fraction": ("final_reached_fraction", "min"),
                "Mean reached fraction": ("final_reached_fraction", "mean"),
                "Inter-agent collisions": ("cumulative_collision_pairs", "sum"),
                "Obstacle contacts": ("cumulative_obstacle_contacts", "sum"),
            },
        )
        .reset_index()
    )
    return add_formatted_threshold_columns(table, df, group_cols)


def _controller_table(
    main_df: pd.DataFrame,
    proposed_df: pd.DataFrame,
    stronger_df: pd.DataFrame | None = None,
    external_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    baselines = main_df[main_df["controller"] != "proposed"].copy()
    proposed = proposed_df[proposed_df["controller"] == "proposed"].copy()
    frames = [proposed, baselines]
    if stronger_df is not None and not stronger_df.empty:
        frames.append(stronger_df.copy())
    if external_df is not None and not external_df.empty:
        frames.append(external_df.copy())
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[~combined["controller"].isin(TABLE_CONTROLLER_EXCLUDE)].copy()
    table = _summary(combined, ["controller"])
    table["Controller"] = table["controller"].map(CONTROLLER_LABELS).fillna(table["controller"])
    table["order"] = table["controller"].map({name: idx for idx, name in enumerate(CONTROLLER_ORDER)})
    table = table.sort_values("order").drop(columns=["controller", "order", "Min reached fraction"])
    return table[
        [
            "Controller",
            "Runs",
            "Safety success",
            "T90",
            "T95",
            "T99",
            "Mean reached fraction",
            "Inter-agent collisions",
            "Obstacle contacts",
        ]
    ]


def _scenario_table(proposed_df: pd.DataFrame) -> pd.DataFrame:
    if "controller" in proposed_df.columns:
        proposed_df = proposed_df[proposed_df["controller"] == "proposed"].copy()
    table = _summary(proposed_df, ["scenario"])
    table["Scenario"] = table["scenario"].map(SCENARIO_LABELS).fillna(table["scenario"])
    order = {name: idx for idx, name in enumerate(["corridor_obstacles", "crossing_traffic", "merge_split", "open_flock"])}
    table["order"] = table["scenario"].map(order)
    table = table.sort_values("order").drop(columns=["scenario", "order", "Inter-agent collisions", "Obstacle contacts"])
    return table[
        [
            "Scenario",
            "Runs",
            "Safety success",
            "T90",
            "T95",
            "T99",
            "Min reached fraction",
            "Mean reached fraction",
        ]
    ]


def _seed_extension_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    table = _summary(seed_df, ["scenario", "n_agents", "packet_loss", "latency_ms"])
    table["Scenario"] = table["scenario"].map(SCENARIO_LABELS).fillna(table["scenario"])
    table = table.rename(columns={"n_agents": "N", "packet_loss": "Packet loss", "latency_ms": "Latency (ms)"})
    table = table.sort_values(["Scenario", "N", "Packet loss", "Latency (ms)"])
    return table[
        [
            "Scenario",
            "N",
            "Packet loss",
            "Latency (ms)",
            "Runs",
            "Safety success",
            "T90",
            "T95",
            "T99",
            "Min reached fraction",
            "Mean reached fraction",
            "Inter-agent collisions",
            "Obstacle contacts",
        ]
    ]


def _condition(row: pd.Series) -> str:
    scenario = SHORT_SCENARIO_LABELS.get(str(row["scenario"]), str(row["scenario"]))
    return f"{scenario} N={int(row['n_agents'])}; PL={float(row['packet_loss']):g}; L={int(round(float(row['latency_ms'])))}"


def _rate_percent(df: pd.DataFrame) -> pd.Series:
    if "projection_agent_adjustment_rate_per_agent_step" not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return df["projection_agent_adjustment_rate_per_agent_step"] * 100.0


def _mean_correction_mm(df: pd.DataFrame) -> pd.Series:
    if "projection_mean_displacement_m_per_adjustment" not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return df["projection_mean_displacement_m_per_adjustment"].fillna(0.0) * 1000.0


def _max_correction_mm(df: pd.DataFrame) -> pd.Series:
    if "projection_max_displacement_m" not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return df["projection_max_displacement_m"].fillna(0.0) * 1000.0


def _projection_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["Condition"] = work.apply(_condition, axis=1)
    work["Projection rate (%)"] = _rate_percent(work)
    work["Mean correction (mm)"] = _mean_correction_mm(work)
    work["Max correction (mm)"] = _max_correction_mm(work)
    table = (
        work.groupby(["Condition"], sort=False)
        .agg(
            Runs=("run_id", "count"),
            **{
                "Safety success": ("safety_success", "mean"),
                "Min reached": ("final_reached_fraction", "min"),
                "Projection rate (%)": ("Projection rate (%)", "mean"),
                "Mean correction (mm)": ("Mean correction (mm)", "mean"),
                "Max correction (mm)": ("Max correction (mm)", "max"),
            },
        )
        .reset_index()
    )
    return table


def _command_filter_telemetry_table(results_path: Path) -> pd.DataFrame:
    overall_path = results_path / "summary_overall_final_command_filter.json"
    scenario_path = results_path / "summary_by_scenario_final_command_filter.csv"
    if not overall_path.exists() or not scenario_path.exists():
        raise FileNotFoundError(
            "Command-filter telemetry requires summary_overall_final_command_filter.json "
            "and summary_by_scenario_final_command_filter.csv"
        )
    import json

    overall = json.loads(overall_path.read_text(encoding="utf-8"))["primary_excluding_seed99_255"]
    scenario_df = pd.read_csv(scenario_path)

    rows: list[dict[str, Any]] = []

    def append_row(label: str, source: dict[str, Any] | pd.Series) -> None:
        rows.append(
            {
                "Group": label,
                "Runs": source["runs"],
                "Safety success": source["safety_success_rate"],
                "Min reached": source["min_final_reached_fraction"],
                "Obstacle proj. adj.": source["obstacle_projection_agent_adjustments_total"],
                "Pairwise proj. adj.": source["pairwise_projection_agent_adjustments_total"],
                "Pre-proj obs. contacts": source["pre_projection_obstacle_contacts_total"],
                "Clearance violation steps": source["pre_projection_obstacle_violation_agent_steps_total"],
            }
        )

    append_row("Overall", overall)
    order = ["corridor_obstacles", "crossing_traffic", "merge_split", "open_flock"]
    scenario_order = {name: idx for idx, name in enumerate(order)}
    scenario_df["order"] = scenario_df["scenario"].map(scenario_order)
    for _, row in scenario_df.sort_values("order").iterrows():
        append_row(SCENARIO_LABELS.get(str(row["scenario"]), str(row["scenario"])), row)
    return pd.DataFrame(rows)


def _stress_label(row: pd.Series, include_delay: bool) -> str:
    drop = float(row.get("local_safety_dropout", 0.0))
    pos_cm = float(row.get("local_safety_position_noise_std_m", 0.0)) * 100.0
    vel_cm = float(row.get("local_safety_velocity_noise_std_mps", 0.0)) * 100.0
    parts = [f"drop {drop:.2f}"]
    if include_delay:
        delay_ms = int(round(float(row.get("local_safety_latency_ms", 0.0))))
        parts.append(f"delay {delay_ms} ms")
    parts.append(f"pos {pos_cm:g} cm")
    parts.append(f"vel {vel_cm:g} cm/s")
    return "; ".join(parts)


def _sensing_table(df: pd.DataFrame, include_delay: bool) -> pd.DataFrame:
    work = df.copy()
    work["Condition"] = work.apply(_condition, axis=1)
    stress_col = "Local sensing stress" if include_delay else "Sensing stress"
    work[stress_col] = work.apply(lambda row: _stress_label(row, include_delay), axis=1)
    work["Projection rate (%)"] = _rate_percent(work)
    work["Max correction (mm)"] = _max_correction_mm(work)
    table = (
        work.groupby(["Condition", stress_col], sort=False)
        .agg(
            **{
                "Safety success": ("safety_success", "mean"),
                "Min reached": ("final_reached_fraction", "min"),
                "Projection rate (%)": ("Projection rate (%)", "mean"),
                "Max correction (mm)": ("Max correction (mm)", "max"),
            },
        )
        .reset_index()
    )
    return table


def _parameter_table() -> pd.DataFrame:
    rows = [
        ("Agent", "3D bounded double integrator; primary-sim mass label 0.027 kg; speed/acceleration limits 1.5 m/s and 3.0 m/s^2; collision/safe/safety-influence radii 0.12/0.39/0.85 m."),
        ("Mission", "Arena [-42,42] x [-42,42] x [0.3,2.5] m; altitude bounds 0.4-2.4 m; goal tolerance 0.75 m; completed agents remain in physical traffic."),
        ("Scenarios", "4 Hz communication and obstacle-perception updates with dt=0.05 s integration and a 120 s horizon; CASSA local-safety observations are also sampled at 4 Hz; open flock, crossing traffic, merge-split, and 21-obstacle corridor."),
        ("Comm.", "4.0 m radius, 12-neighbor cap; packet loss 0.0/0.1/0.2/0.4; latency 0/50/100/200 ms; communicated-state noise 0.015 m position and 0.025 m/s velocity."),
        ("CASSA filter", "Obstacle margin 0.34 m; command-filter buffer 0.15 m; lookahead 0.90 m; obstacle projection fallback enabled and reported as telemetry; terminal slot spacing 0.50 m; crossing-flow governor radius 1.20 m, clearance scale 0.42 m, speed floors 0.28/0.82, closing trigger 0.03 m/s."),
        ("Sensing", "Main CASSA full sweep: all-around local relative-state sensing and obstacle perception are sampled at 4 Hz; local dropout/delay/noise are zero; no FOV or occlusion model."),
        ("Weights", "Separation/alignment/cohesion 1.5/0.3/0.08; goal/obstacle/boundary/damping 5.0/2.5/1.1/0.22; safety/closing 2.5/1.0."),
    ]
    return pd.DataFrame(rows, columns=["Group", "Compact setting"])


def _crazyflow_table(results_path: Path) -> pd.DataFrame:
    index_path = results_path / "run_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    df = pd.read_csv(index_path)
    completed = df[df["replay_completed"] == True].copy()  # noqa: E712
    if completed.empty:
        raise ValueError(f"No completed Crazyflow replay rows found in {index_path}")

    def summarize(label: str, group: pd.DataFrame) -> dict[str, Any]:
        t99 = pd.to_numeric(group["T99_s"], errors="coerce")
        return {
            "Group": label,
            "Runs": int(len(group)),
            "Safety success": float(pd.to_numeric(group["crazyflow_safety_success"], errors="coerce").mean()),
            "T99 median (s)": float(t99.dropna().median()) if t99.notna().any() else math.nan,
            "T99 max (s)": float(t99.dropna().max()) if t99.notna().any() else math.nan,
            "Collision pairs": int(pd.to_numeric(group["crazyflow_collision_pairs"], errors="coerce").fillna(0).sum()),
            "Obstacle contacts": int(pd.to_numeric(group["crazyflow_obstacle_contacts"], errors="coerce").fillna(0).sum()),
            "Max tracking error (m)": float(pd.to_numeric(group["crazyflow_max_tracking_error_m"], errors="coerce").max()),
            "Max downwash force (N)": float(pd.to_numeric(group["crazyflow_max_downwash_force_N"], errors="coerce").max()),
        }

    rows = [summarize("Overall", completed)]
    for n_agents, group in completed.groupby("n_agents", sort=True):
        rows.append(summarize(f"N={int(n_agents)}", group))
    return pd.DataFrame(rows)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _format_value(header: str, value: Any) -> str:
    if isinstance(value, str):
        return _latex_escape(value)
    if pd.isna(value):
        return ""
    number = float(value)
    if header in {
        "Runs",
        "N",
        "Collision pairs",
        "Inter-agent collisions",
        "Obstacle contacts",
        "Obstacle proj. adj.",
        "Pairwise proj. adj.",
        "Pre-proj obs. contacts",
        "Clearance violation steps",
    }:
        return f"{int(round(number)):,}"
    if header == "Packet loss":
        return f"{number:.1f}"
    if header == "Latency (ms)":
        return f"{int(round(number))}"
    if header == "Safety success":
        return f"{number:.2f}"
    if header in {"T99 median (s)", "T99 max (s)"}:
        return f"{number:.1f}"
    if "fraction" in header.lower() or header == "Min reached":
        return f"{number:.3f}"
    if "Projection rate" in header:
        return f"{number:.3f}"
    if "correction" in header:
        return f"{number:.2f}"
    if "distance" in header.lower() or "tracking" in header.lower() or "downwash" in header.lower():
        return f"{number:.3f}"
    return f"{number:g}"


def _write_tex_table(table: pd.DataFrame, out_path: Path, key: str) -> None:
    caption, label = CAPTIONS[key]
    headers = list(table.columns)
    rows = [
        " & ".join(_format_value(header, row[header]) for header in headers) + r" \\"
        for _, row in table.iterrows()
    ]
    colspec = r"lp{0.78\textwidth}" if key == "table_5_parameter_summary" else "l" + "r" * (len(headers) - 1)
    resize_begin = "" if key == "table_5_parameter_summary" else "\\resizebox{\\textwidth}{!}{%\n"
    resize_end = "" if key == "table_5_parameter_summary" else "}\n"
    out_path.write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{_latex_escape(caption)}}}\n"
        f"\\label{{{label}}}\n"
        f"{resize_begin}"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        + " & ".join(_latex_escape(header) for header in headers)
        + r" \\"
        + "\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"{resize_end}"
        "\\end{table}\n",
        encoding="utf-8",
    )


def _write_pair(table: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{name}.csv", index=False)
    _write_tex_table(table, out_dir / f"{name}.tex", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate paper tables from current run_index.csv files.")
    parser.add_argument("--main-results", default="results/classical_baselines_full_sweep")
    parser.add_argument("--proposed-results", default="results/proposed_full_sweep")
    parser.add_argument("--stronger-baseline-results", default="results/secondary_baselines_full_sweep")
    parser.add_argument("--external-baseline-results", default="results/external_package_baselines_full_sweep")
    parser.add_argument("--seed-extension-results", default="results/seed_extension_120s")
    parser.add_argument("--projection-results", default="results/proposed_full_sweep")
    parser.add_argument("--crazyflow-results", default="results/crazyflow_replay")
    parser.add_argument("--out", default="paper_assets/tables")
    args = parser.parse_args()

    main_df = _exclude_visualization_runs(_load(Path(args.main_results)))
    proposed_df = _exclude_visualization_runs(_load(Path(args.proposed_results)))
    stronger_path = Path(args.stronger_baseline_results)
    stronger_df = _exclude_visualization_runs(_load(stronger_path)) if (stronger_path / "run_index.csv").exists() else None
    external_path = Path(args.external_baseline_results)
    external_df = _exclude_visualization_runs(_load(external_path)) if (external_path / "run_index.csv").exists() else None
    seed_df = _exclude_visualization_runs(_load(Path(args.seed_extension_results)))
    out_dir = Path(args.out)

    _write_pair(_controller_table(main_df, proposed_df, stronger_df, external_df), out_dir, "table_1_controller_comparison")
    _write_pair(_scenario_table(proposed_df), out_dir, "table_2_proposed_by_scenario")
    _write_pair(_parameter_table(), out_dir, "table_5_parameter_summary")
    _write_pair(_seed_extension_table(seed_df), out_dir, "table_4_seed_extension")
    _write_pair(_command_filter_telemetry_table(Path(args.proposed_results)), out_dir, "table_6_projection_audit")
    crazyflow_path = Path(args.crazyflow_results)
    if (crazyflow_path / "run_index.csv").exists():
        _write_pair(_crazyflow_table(crazyflow_path), out_dir, "table_9_crazyflow_replay")
    print(f"Wrote paper tables to {out_dir}")


if __name__ == "__main__":
    main()
