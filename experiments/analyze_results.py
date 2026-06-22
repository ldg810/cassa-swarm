from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_run_index(results_dir: Path) -> pd.DataFrame:
    index_path = results_dir / "run_index.csv"
    if index_path.exists():
        return pd.read_csv(index_path)
    rows = []
    for p in results_dir.glob("*/metrics.json"):
        with p.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(f"No run_index.csv or metrics.json files found under {results_dir}")
    return pd.DataFrame(rows)


def _ci95(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in df.columns.values]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario", "controller", "n_agents", "packet_loss", "latency_ms"]
    metrics = [
        "mission_success",
        "final_reached_fraction",
        "min_inter_agent_distance_m",
        "collision_rate_per_pair_step",
        "safety_violation_rate_per_pair_step",
        "obstacle_contact_rate_per_agent_step",
        "completion_time_s",
        "energy_proxy_accel2_s_per_agent",
        "mean_messages_per_agent_per_update",
        "runtime_ms_per_agent_step",
        "runtime_s",
    ]
    available = [m for m in metrics if m in df.columns]
    work = df.copy()
    if "mission_success" in work.columns:
        work["mission_success"] = work["mission_success"].astype(float)

    agg_spec = {}
    for m in available:
        agg_spec[m] = ["mean", "std", "min", "max", _ci95]
    summary = work.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    summary = _flatten_columns(summary)
    summary = summary.rename(columns={c: c.replace("_<lambda_0>", "_ci95") for c in summary.columns})
    summary["n_repeats"] = work.groupby(group_cols, dropna=False).size().values
    return summary


def _best_rows_for_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    # Main table: baseline communication condition, all controllers/scenarios/sizes.
    cols = [
        "scenario", "controller", "n_agents", "packet_loss", "latency_ms", "n_repeats",
        "mission_success_mean", "final_reached_fraction_mean", "collision_rate_per_pair_step_mean",
        "min_inter_agent_distance_m_mean", "safety_violation_rate_per_pair_step_mean",
        "energy_proxy_accel2_s_per_agent_mean", "runtime_ms_per_agent_step_mean",
    ]
    cols = [c for c in cols if c in summary.columns]
    base = summary[(summary["packet_loss"] == 0.0) & (summary["latency_ms"] == 0.0)]
    return base[cols].copy().sort_values(["scenario", "n_agents", "controller"])


def _robustness_table(summary: pd.DataFrame, variable: str) -> pd.DataFrame:
    cols = [
        "scenario", "controller", "n_agents", "packet_loss", "latency_ms", "n_repeats",
        "mission_success_mean", "final_reached_fraction_mean", "collision_rate_per_pair_step_mean",
        "min_inter_agent_distance_m_mean", "mean_messages_per_agent_per_update_mean",
    ]
    cols = [c for c in cols if c in summary.columns]
    if variable == "packet_loss":
        # Rows with multiple packet-loss values and fixed latency are useful for robustness plots.
        work = summary.copy()
        return work[cols].sort_values(["scenario", "n_agents", "controller", "packet_loss", "latency_ms"])
    if variable == "latency_ms":
        work = summary.copy()
        return work[cols].sort_values(["scenario", "n_agents", "controller", "latency_ms", "packet_loss"])
    raise ValueError(variable)


def _write_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.4g") -> None:
    try:
        text = df.to_latex(index=False, float_format=lambda x: float_format % x)
    except Exception:
        text = df.to_string(index=False)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MAV swarm simulation results into paper-ready tables.")
    parser.add_argument("--results", required=True, help="Directory containing run_index.csv or run folders")
    parser.add_argument("--out", default=None, help="Output directory for summaries/tables")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else results_dir / "summary"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = _load_run_index(results_dir)
    df.to_csv(out_dir / "all_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "summary_by_condition.csv", index=False)

    main_table = _best_rows_for_main_table(summary)
    main_table.to_csv(tables_dir / "table_1_main_performance.csv", index=False)
    _write_latex_table(main_table, tables_dir / "table_1_main_performance.tex")

    packet_table = _robustness_table(summary, "packet_loss")
    packet_table.to_csv(tables_dir / "table_2_packet_loss_robustness.csv", index=False)
    _write_latex_table(packet_table, tables_dir / "table_2_packet_loss_robustness.tex")

    latency_table = _robustness_table(summary, "latency_ms")
    latency_table.to_csv(tables_dir / "table_3_latency_robustness.csv", index=False)
    _write_latex_table(latency_table, tables_dir / "table_3_latency_robustness.tex")

    print(f"Wrote summary CSV and paper tables to {out_dir}")


if __name__ == "__main__":
    main()
