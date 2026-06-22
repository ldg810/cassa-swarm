from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.mission_threshold_times import threshold_crossing_time_s


RUN_ID_RE = re.compile(
    r"(?P<scenario>.+)__proposed__N(?P<n_agents>\d+)__pl(?P<packet_loss>\d+)__lat(?P<latency_ms>\d+)__seed(?P<seed>\d+)"
)
SCENARIO_LABELS = {
    "corridor_obstacles": "Obstacle corridor",
    "crossing_traffic": "Crossing traffic",
    "open_flock": "Open flock",
    "merge_split": "Merge--split",
}
SCENARIO_ORDER = {name: idx for idx, name in enumerate(SCENARIO_LABELS)}


def _source_run_id(metrics: dict[str, Any]) -> str:
    source = Path(str(metrics.get("source_trajectory", "")))
    if source.name in {"trajectory.csv", "trajectory.npz"}:
        return source.parent.name
    return source.name


def parse_source_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_ID_RE.fullmatch(run_id)
    if not match:
        raise ValueError(f"Cannot parse source run id: {run_id}")
    packet_loss_code = int(match.group("packet_loss"))
    return {
        "scenario": match.group("scenario"),
        "Scenario": SCENARIO_LABELS.get(match.group("scenario"), match.group("scenario").replace("_", " ").title()),
        "N": int(match.group("n_agents")),
        "Packet loss": packet_loss_code / 100.0,
        "Latency (ms)": int(match.group("latency_ms")),
        "Seed": int(match.group("seed")),
    }


def _format_time(value: float, horizon_s: float) -> str:
    if pd.isna(value):
        return f">{horizon_s:.0f}"
    return f"{float(value):.1f}"


def summarize_replay_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    timeseries = pd.read_csv(run_dir / "timeseries.csv")
    source = parse_source_run_id(_source_run_id(metrics))
    horizon_s = float(metrics.get("simulated_s", timeseries["t_s"].max()))

    t90 = threshold_crossing_time_s(timeseries, 0.90)
    t95 = threshold_crossing_time_s(timeseries, 0.95)
    t99 = threshold_crossing_time_s(timeseries, 0.99)
    safety_success = (
        int(metrics.get("cumulative_collision_pairs", 0)) == 0
        and int(metrics.get("cumulative_obstacle_contacts", 0)) == 0
    )

    return {
        "scenario": source["scenario"],
        "Scenario": source["Scenario"],
        "Case": f"N{source['N']} PL={source['Packet loss']:g} L={source['Latency (ms)']} seed={source['Seed']}",
        "N": source["N"],
        "Packet loss": source["Packet loss"],
        "Latency (ms)": source["Latency (ms)"],
        "Seed": source["Seed"],
        "Safety success": 1.0 if safety_success else 0.0,
        "T90": _format_time(t90, horizon_s),
        "T95": _format_time(t95, horizon_s),
        "T99": _format_time(t99, horizon_s),
        "Min distance (m)": float(metrics.get("min_inter_agent_distance_m", float("nan"))),
        "Collision pairs": int(metrics.get("cumulative_collision_pairs", 0)),
        "Obstacle contacts": int(metrics.get("cumulative_obstacle_contacts", 0)),
        "P95 tracking error (m)": float(metrics.get("final_tracking_error_p95_m", float("nan"))),
        "Max tracking error (m)": float(metrics.get("max_tracking_error_m", float("nan"))),
        "Max downwash force (N)": float(metrics.get("max_logged_downwash_force_N", 0.0)),
        "RTF": float(metrics.get("real_time_factor", 0.0)),
        "Downwash": "on" if bool(metrics.get("downwash_enabled", False)) else "off",
    }


def build_crazyflow_table(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    rows = [
        summarize_replay_run(path)
        for path in sorted(results_dir.iterdir())
        if path.is_dir() and (path / "metrics.json").exists() and (path / "timeseries.csv").exists()
    ]
    if not rows:
        raise FileNotFoundError(f"No Crazyflow replay metrics found in {results_dir}")
    table = pd.DataFrame(rows)
    table["_scenario_order"] = table["scenario"].map(lambda value: SCENARIO_ORDER.get(str(value), len(SCENARIO_ORDER)))
    table = table.sort_values(["_scenario_order", "N", "Packet loss", "Latency (ms)", "Seed"]).drop(columns=["_scenario_order"])
    return table.reset_index(drop=True)


def _latex_escape(value: Any) -> str:
    text = str(value)
    for src, dst in {"&": r"\&", "%": r"\%", "_": r"\_"}.items():
        text = text.replace(src, dst)
    return text


def _format_value(header: str, value: Any) -> str:
    if isinstance(value, str):
        return _latex_escape(value)
    if pd.isna(value):
        return ""
    number = float(value)
    if header in {"N", "Latency (ms)", "Seed", "Collision pairs", "Obstacle contacts"}:
        return f"{int(round(number))}"
    if header in {"Safety success", "Packet loss"}:
        return f"{number:.2f}".rstrip("0").rstrip(".")
    if "distance" in header.lower() or "tracking" in header.lower():
        return f"{number:.3f}"
    if "downwash" in header.lower():
        return f"{number:.3f}"
    if header == "RTF":
        return f"{number:.2f}"
    return f"{number:g}"


def write_table_pair(table: pd.DataFrame, out_dir: str | Path, name: str = "table_9_crazyflow_replay") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    tex_path = out_dir / f"{name}.tex"
    table = table.drop(columns=["scenario"], errors="ignore")
    table.to_csv(csv_path, index=False)
    headers = list(table.columns)
    col_spec = "ll" + "r" * max(len(headers) - 2, 0)
    body = "\n".join(
        " & ".join(_format_value(header, row[header]) for header in headers) + r" \\"
        for _, row in table.iterrows()
    )
    tex_path.write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Crazyflow replay of representative and targeted trajectories with rotor-drag dynamics and downwash enabled.}\n"
        "\\label{tab:crazyflow-replay}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        + " & ".join(_latex_escape(header) for header in headers)
        + r" \\"
        + "\n\\midrule\n"
        + body
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper table from Crazyflow replay results.")
    parser.add_argument("--results", default="results/crazyflow_replay")
    parser.add_argument("--out", default="paper_assets/tables")
    parser.add_argument("--name", default="table_9_crazyflow_replay")
    args = parser.parse_args()
    table = build_crazyflow_table(args.results)
    write_table_pair(table, args.out, args.name)
    print(f"Wrote {len(table)} Crazyflow replay rows to {Path(args.out) / (args.name + '.csv')}")


if __name__ == "__main__":
    main()
