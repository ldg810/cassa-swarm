from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mav_swarm_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mission_threshold_times import THRESHOLD_COLUMNS, add_threshold_times, censored_median_time_s


TRAJECTORY_SCENARIO_ORDER = ["corridor_obstacles", "crossing_traffic", "merge_split"]
TRAJECTORY_SCENARIO_LABELS = {
    "corridor_obstacles": "Obstacle corridor",
    "crossing_traffic": "Crossing traffic",
    "open_flock": "Open flock",
    "merge_split": "Merge-split",
}
CONTROLLER_LABELS = {
    "apf": "APF",
    "boids": "Boids",
    "cbf_qp_osqp": "CBF-QP (OSQP)",
    "decentralized_mpc": "Short-horizon",
    "mader_like": "MADER-like",
    "proposed": "CASSA (Proposed)",
    "proposed_no_shield": "No shield",
    "rvo2_external": "RVO2 package",
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


def _ci_column_for_mean(df: pd.DataFrame, y: str) -> str | None:
    if not y.endswith("_mean"):
        return None
    candidates = [y.replace("_mean", "__ci95"), y.replace("_mean", "_ci95")]
    return next((candidate for candidate in candidates if candidate in df.columns), None)


def _read_summary(summary_path: Path) -> pd.DataFrame:
    if summary_path.is_dir():
        summary_path = summary_path / "summary_by_condition.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return pd.read_csv(summary_path)


def _plot_grouped_lines(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    group_order: list[str] | None = None,
    group_labels: dict[str, str] | None = None,
    legend_ncol: int = 1,
    legend_fontsize: float | None = None,
    legend_loc: str = "best",
) -> None:
    if df.empty or y not in df.columns:
        return
    fig = plt.figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(111)
    grouped = df.groupby(group)
    keys = list(grouped.groups)
    if group_order is not None:
        order_index = {name: idx for idx, name in enumerate(group_order)}
        keys = sorted(keys, key=lambda key: (order_index.get(str(key), len(order_index)), str(key)))
    else:
        keys = sorted(keys)
    for g in keys:
        sub = grouped.get_group(g)
        sub = sub.sort_values(x)
        label = group_labels.get(str(g), str(g)) if group_labels else str(g)
        ax.plot(sub[x], sub[y], marker="o", label=label)
        ci_col = _ci_column_for_mean(sub, y)
        if ci_col in sub.columns:
            yv = pd.to_numeric(sub[y], errors="coerce").to_numpy()
            ci = pd.to_numeric(sub[ci_col], errors="coerce").fillna(0.0).to_numpy()
            xv = pd.to_numeric(sub[x], errors="coerce").to_numpy()
            ax.fill_between(xv, yv - ci, yv + ci, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=legend_ncol, fontsize=legend_fontsize, loc=legend_loc)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_bar(df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    if df.empty or y not in df.columns:
        return
    df = df.sort_values(x)
    fig = plt.figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(df[x].astype(str), df[y])
    ci_col = _ci_column_for_mean(df, y)
    if ci_col is not None:
        ax.errorbar(df[x].astype(str), df[y], yerr=df[ci_col].fillna(0.0), fmt="none", capsize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _read_raw_results(results_dir: Optional[Path]) -> pd.DataFrame | None:
    if results_dir is None:
        return None
    run_index_path = results_dir / "run_index.csv"
    if not run_index_path.exists():
        return None
    df = pd.read_csv(run_index_path)
    if "seed" in df.columns:
        df = df[df["seed"] != 99].copy()
    return add_threshold_times(df, results_dir)


def _summary_list(comparison_summary: pd.DataFrame | Iterable[pd.DataFrame] | None) -> list[pd.DataFrame]:
    if comparison_summary is None:
        return []
    if isinstance(comparison_summary, pd.DataFrame):
        return [comparison_summary]
    return [df for df in comparison_summary if df is not None and not df.empty]


def _runtime_plot_data(
    summary: pd.DataFrame,
    comparison_summary: pd.DataFrame | Iterable[pd.DataFrame] | None,
    scenario: str,
) -> pd.DataFrame:
    base = summary[(summary["scenario"] == scenario) & (summary["packet_loss"] == 0.0) & (summary["latency_ms"] == 0.0)]
    frames = [base]
    for comparison_df in _summary_list(comparison_summary):
        comparison_base = comparison_df[
            (comparison_df["scenario"] == scenario)
            & (comparison_df["packet_loss"] == 0.0)
            & (comparison_df["latency_ms"] == 0.0)
            & (comparison_df["controller"] != "proposed")
        ]
        frames.append(comparison_base)
    runtime = pd.concat(frames, ignore_index=True, sort=False)
    runtime = runtime[runtime["controller"].isin(CONTROLLER_ORDER)].copy()
    if runtime.empty:
        return runtime
    order_index = {name: idx for idx, name in enumerate(CONTROLLER_ORDER)}
    runtime["_controller_order"] = runtime["controller"].map(order_index)
    runtime = runtime.sort_values(["_controller_order", "n_agents"]).drop(columns=["_controller_order"])
    return runtime


def _threshold_time_summary(df: pd.DataFrame, x: str) -> pd.DataFrame:
    rows = []
    horizon_s = float(df["horizon_s"].max()) if "horizon_s" in df.columns and not df.empty else math.nan
    for x_value, group in df.groupby(x, sort=True):
        row = {x: x_value}
        total = int(len(group))
        for column in THRESHOLD_COLUMNS:
            row[column] = censored_median_time_s(group[column], total, horizon_s)
            values = pd.to_numeric(group[column], errors="coerce")
            if not math.isnan(horizon_s):
                values = values.fillna(horizon_s)
            finite = values.dropna()
            if finite.empty:
                row[f"{column}_q25"] = math.nan
                row[f"{column}_q75"] = math.nan
            else:
                row[f"{column}_q25"] = float(finite.quantile(0.25))
                row[f"{column}_q75"] = float(finite.quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_threshold_times(
    df: pd.DataFrame,
    x: str,
    xlabel: str,
    title: str,
    out_path: Path,
    horizon_s: float | None = None,
    nominal_baseline: dict[str, float] | None = None,
    legend_loc: str = "center right",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    if df.empty:
        return
    fig = plt.figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(111)
    labels = {
        "T90_s": "T90",
        "T95_s": "T95",
        "T99_s": "T99",
    }
    for column, label in labels.items():
        if column not in df.columns:
            continue
        sub = df[[x, column]].dropna().sort_values(x)
        if sub.empty:
            continue
        line, = ax.plot(sub[x], sub[column], marker="o", label=label)
        q25_col = f"{column}_q25"
        q75_col = f"{column}_q75"
        if q25_col in df.columns and q75_col in df.columns:
            band = df[[x, q25_col, q75_col]].dropna().sort_values(x)
            if not band.empty:
                ax.fill_between(
                    band[x],
                    band[q25_col],
                    band[q75_col],
                    color=line.get_color(),
                    alpha=0.12,
                    linewidth=0,
                )
        if nominal_baseline is not None:
            baseline_value = nominal_baseline.get(column)
            if baseline_value is not None and not math.isnan(float(baseline_value)):
                ax.axhline(
                    float(baseline_value),
                    linestyle="--",
                    linewidth=1.25,
                    alpha=0.58,
                    color=line.get_color(),
                )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median time to reached-fraction threshold (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    if nominal_baseline is not None:
        handles.append(Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.25, alpha=0.75))
        legend_labels.append("Nominal no-loss/no-latency")
    legend_kwargs = {"frameon": False, "loc": legend_loc}
    if legend_bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
    ax.legend(handles, legend_labels, **legend_kwargs)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _choose_representative_n(df: pd.DataFrame) -> int:
    values = sorted(int(v) for v in df["n_agents"].dropna().unique())
    if not values:
        return 0
    # Use the upper middle value; for [50, 100] this returns 100, which is usually
    # the minimum publishable swarm size for the target paper.
    return values[len(values) // 2]


def _choose_scenario(df: pd.DataFrame) -> str:
    preferred = ["corridor_obstacles", "crossing_traffic", "open_flock"]
    scenarios = set(df["scenario"].unique()) if "scenario" in df.columns else set()
    for s in preferred:
        if s in scenarios:
            return s
    return sorted(scenarios)[0]


def _scenario_label(scenario: str) -> str:
    return TRAJECTORY_SCENARIO_LABELS.get(scenario, scenario.replace("_", " "))


def make_standard_figures(
    summary: pd.DataFrame,
    out_dir: Path,
    results_dir: Optional[Path] = None,
    comparison_summary: pd.DataFrame | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario = _choose_scenario(summary)
    scenario_label = _scenario_label(scenario).lower()
    raw = _read_raw_results(results_dir)
    base = summary[(summary["scenario"] == scenario) & (summary["packet_loss"] == 0.0) & (summary["latency_ms"] == 0.0)]
    runtime_base = _runtime_plot_data(summary, comparison_summary, scenario)

    if raw is not None:
        raw_base = raw[
            (raw["scenario"] == scenario)
            & (raw["packet_loss"] == 0.0)
            & (raw["latency_ms"] == 0.0)
            & (raw["controller"] == "proposed")
        ]
        threshold_scaling = _threshold_time_summary(raw_base, "n_agents")
        horizon_s = float(raw_base["horizon_s"].max()) if "horizon_s" in raw_base.columns and not raw_base.empty else None
        _plot_threshold_times(
            threshold_scaling,
            x="n_agents",
            xlabel="Number of micro air vehicles",
            title=f"Time to reached-fraction thresholds ({scenario_label}, CASSA (Proposed))",
            out_path=out_dir / "fig_1_time_scalability",
            horizon_s=horizon_s,
            legend_loc="upper left",
        )
    _plot_grouped_lines(
        runtime_base,
        x="n_agents",
        y="runtime_ms_per_agent_step_mean",
        group="controller",
        xlabel="Number of micro air vehicles",
        ylabel="Runtime (ms per agent-step)",
        title=f"Computational scaling ({scenario_label})",
        out_path=out_dir / "fig_2_runtime_scaling",
        group_order=CONTROLLER_ORDER,
        group_labels=CONTROLLER_LABELS,
        legend_ncol=2,
        legend_fontsize=8,
        legend_loc="upper left",
    )

    # Packet-loss robustness: choose the most common nonzero/fixed latency condition.
    n_pick_robust = _choose_representative_n(summary[summary["scenario"] == scenario])
    pl_summary = summary[(summary["scenario"] == scenario) & (summary["n_agents"] == n_pick_robust)]
    if raw is not None and not pl_summary.empty:
        pl = raw[(raw["scenario"] == scenario) & (raw["n_agents"] == n_pick_robust) & (raw["controller"] == "proposed")]
        candidates = pl.groupby("latency_ms").size().sort_values(ascending=False)
        latency = float(candidates.index[0])
        pl = pl[pl["latency_ms"] == latency]
        pl_times = _threshold_time_summary(pl, "packet_loss")
        nominal = raw[
            (raw["scenario"] == scenario)
            & (raw["n_agents"] == n_pick_robust)
            & (raw["controller"] == "proposed")
            & (raw["packet_loss"] == 0.0)
            & (raw["latency_ms"] == 0.0)
        ]
        nominal_baseline = None
        if not nominal.empty:
            nominal_row = _threshold_time_summary(nominal, "packet_loss")
            if not nominal_row.empty:
                nominal_baseline = {
                    column: float(nominal_row.iloc[0][column])
                    for column in THRESHOLD_COLUMNS
                    if column in nominal_row.columns and pd.notna(nominal_row.iloc[0][column])
                }
        horizon_s = float(pl["horizon_s"].max()) if "horizon_s" in pl.columns and not pl.empty else None
        _plot_threshold_times(
            pl_times,
            x="packet_loss",
            xlabel="Packet loss probability",
            title=f"Packet-loss sweep under fixed {latency:g} ms latency",
            out_path=out_dir / "fig_3_packet_loss_time",
            horizon_s=horizon_s,
            nominal_baseline=nominal_baseline,
            legend_loc="upper left",
            legend_bbox_to_anchor=(0.04, 0.88),
        )
        _plot_grouped_lines(
            pl_summary[pl_summary["latency_ms"] == latency],
            x="packet_loss",
            y="min_inter_agent_distance_m_mean",
            group="controller",
            xlabel="Packet loss probability",
            ylabel="Minimum inter-agent distance (m)",
            title=f"Separation under packet loss at {latency:g} ms latency",
            out_path=out_dir / "fig_aux_packet_loss_distance",
        )

    # Latency robustness: choose the most common packet-loss condition.
    lat_summary = summary[(summary["scenario"] == scenario) & (summary["n_agents"] == n_pick_robust)]
    if raw is not None and not lat_summary.empty:
        lat = raw[(raw["scenario"] == scenario) & (raw["n_agents"] == n_pick_robust) & (raw["controller"] == "proposed")]
        candidates = lat.groupby("packet_loss").size().sort_values(ascending=False)
        packet_loss = float(candidates.index[0])
        lat = lat[lat["packet_loss"] == packet_loss]
        lat_times = _threshold_time_summary(lat, "latency_ms")
        horizon_s = float(lat["horizon_s"].max()) if "horizon_s" in lat.columns and not lat.empty else None
        _plot_threshold_times(
            lat_times,
            x="latency_ms",
            xlabel="Communication latency (ms)",
            title=f"Time to reached-fraction thresholds at packet loss {packet_loss:g}",
            out_path=out_dir / "fig_4_latency_time",
            horizon_s=horizon_s,
        )

    # Ablation chart at the largest available N for the chosen scenario under baseline comm.
    if not base.empty:
        n_pick = int(base["n_agents"].max())
        abl = base[(base["n_agents"] == n_pick) & (base["controller"].isin([
            "boids", "apf", "proposed_no_shield", "proposed"
        ]))]
        _plot_bar(
            abl,
            x="controller",
            y="collision_rate_per_pair_step_mean",
            xlabel="Controller",
            ylabel="Collision rate per pair-step",
            title=f"Ablation and baselines at N={n_pick} ({scenario_label})",
            out_path=out_dir / "fig_aux_ablation_collision",
        )


def _set_3d_limits(ax, traj: pd.DataFrame, obs: Optional[pd.DataFrame]) -> None:
    coords = traj[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    coords = coords[np.isfinite(coords).all(axis=1)]
    if coords.size == 0:
        return

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    if obs is not None and not obs.empty:
        for _, row in obs.iterrows():
            cx = float(row["cx"])
            cy = float(row["cy"])
            radius = float(row["radius"])
            zmin = float(row.get("height_min", row.get("cz", mins[2])))
            zmax = float(row.get("height_max", row.get("cz", maxs[2])))
            mins = np.minimum(mins, np.array([cx - radius, cy - radius, zmin]))
            maxs = np.maximum(maxs, np.array([cx + radius, cy + radius, zmax]))

    spans = np.maximum(maxs - mins, 1e-6)
    pad = np.maximum(spans * 0.06, np.array([0.2, 0.2, 0.08]))
    mins -= pad
    maxs += pad
    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))
    ax.set_box_aspect(tuple(np.maximum(maxs - mins, 1e-6)))


def _plot_obstacle_cylinder(ax, row: pd.Series) -> None:
    cx = float(row["cx"])
    cy = float(row["cy"])
    radius = float(row["radius"])
    zmin = float(row.get("height_min", row.get("cz", 0.0)))
    zmax = float(row.get("height_max", row.get("cz", zmin)))
    theta = np.linspace(0.0, 2.0 * np.pi, 64)
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)
    ax.plot(xs, ys, np.full_like(xs, zmin), color="black", linewidth=0.7, alpha=0.55)
    ax.plot(xs, ys, np.full_like(xs, zmax), color="black", linewidth=0.7, alpha=0.55)
    for idx in range(0, len(theta), 8):
        ax.plot([xs[idx], xs[idx]], [ys[idx], ys[idx]], [zmin, zmax], color="black", linewidth=0.45, alpha=0.35)


def _set_2d_limits(ax, traj: pd.DataFrame, obs: pd.DataFrame | None) -> None:
    mins = traj[["x", "y"]].min().to_numpy(dtype=float)
    maxs = traj[["x", "y"]].max().to_numpy(dtype=float)
    if obs is not None and not obs.empty:
        for _, row in obs.iterrows():
            cx = float(row["cx"])
            cy = float(row["cy"])
            radius = float(row["radius"])
            mins = np.minimum(mins, np.array([cx - radius, cy - radius]))
            maxs = np.maximum(maxs, np.array([cx + radius, cy + radius]))

    spans = np.maximum(maxs - mins, 1e-6)
    pad = np.maximum(spans * 0.06, np.array([0.2, 0.2]))
    ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    ax.set_aspect("equal", adjustable="box")


def _run_scenario(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            scenario = metrics.get("scenario")
            if scenario:
                return str(scenario)
        except (OSError, json.JSONDecodeError):
            pass
    return run_dir.name.split("__", 1)[0]


def _trajectory_sort_key(run_dir: Path) -> tuple:
    name = run_dir.name
    return (
        "proposed" not in name,
        "seed99" not in name,
        name,
    )


def _collect_trajectory_runs(results_dir: Path) -> list[tuple[str, Path]]:
    runs_by_scenario: dict[str, list[Path]] = {}
    for traj_path in sorted(results_dir.glob("*/trajectory.csv")):
        run_dir = traj_path.parent
        scenario = _run_scenario(run_dir)
        runs_by_scenario.setdefault(scenario, []).append(run_dir)

    ordered: list[tuple[str, Path]] = []
    for scenario in TRAJECTORY_SCENARIO_ORDER:
        candidates = runs_by_scenario.get(scenario, [])
        if candidates:
            ordered.append((scenario, sorted(candidates, key=_trajectory_sort_key)[0]))
    if ordered:
        return ordered

    for scenario in sorted(runs_by_scenario):
        candidates = sorted(runs_by_scenario[scenario], key=_trajectory_sort_key)
        ordered.append((scenario, candidates[0]))
    return ordered


def _plot_trajectory_pair(run_dir: Path, ax, ax_top, scenario: str, scenario_label: str) -> None:
    traj = pd.read_csv(run_dir / "trajectory.csv")
    for agent_idx, (_, sub) in enumerate(traj.groupby("agent")):
        sub = sub.sort_values("t_s")
        color = plt.cm.viridis((agent_idx % 100) / 99.0)
        ax.plot(sub["x"], sub["y"], sub["z"], linewidth=0.55, alpha=0.64, color=color)
        ax_top.plot(sub["x"], sub["y"], linewidth=0.55, alpha=0.55, color=color)

    starts = traj.sort_values("t_s").groupby("agent")[["x", "y", "z"]].first()
    ax.scatter(starts["x"], starts["y"], starts["z"], marker="o", s=8, color="black", alpha=0.35)
    ax_top.scatter(starts["x"], starts["y"], marker="o", s=8, color="black", alpha=0.35)

    if {"goal_x", "goal_y", "goal_z"}.issubset(traj.columns):
        goals = traj.groupby("agent")[["goal_x", "goal_y", "goal_z"]].last().drop_duplicates()
        ax.scatter(goals["goal_x"], goals["goal_y"], goals["goal_z"], marker="x", s=18, color="tab:red", alpha=0.65)
        ax_top.scatter(goals["goal_x"], goals["goal_y"], marker="x", s=18, color="tab:red", alpha=0.65)

    obs = None
    obs_path = run_dir / "obstacles.csv"
    if obs_path.exists():
        obs = pd.read_csv(obs_path)
        for _, row in obs.iterrows():
            _plot_obstacle_cylinder(ax, row)
            ax_top.add_patch(
                Circle(
                    (float(row["cx"]), float(row["cy"])),
                    float(row["radius"]),
                    facecolor="0.55",
                    edgecolor="0.15",
                    alpha=0.35,
                )
            )

    _set_3d_limits(ax, traj, obs)
    _set_2d_limits(ax_top, traj, obs)
    box_aspect = (2.15, 2.15, 0.55) if scenario == "crossing_traffic" else (3.2, 1.15, 0.55)
    try:
        ax.set_box_aspect(box_aspect, zoom=1.15)
    except TypeError:
        ax.set_box_aspect(box_aspect)
    ax.set_xlabel("x (m)", labelpad=-1, fontsize=8)
    ax.set_ylabel("y (m)", labelpad=-1, fontsize=8)
    ax.set_zlabel("z (m)", labelpad=-2, fontsize=8)
    ax.set_zticks([0, 1, 2, 3])
    ax.tick_params(axis="both", which="major", labelsize=8, pad=0)
    ax_top.tick_params(axis="both", which="major", labelsize=8)
    ax_top.set_xlabel("x position (m)", fontsize=9)
    ax_top.set_ylabel("y position (m)", fontsize=9)
    ax.set_title(f"{scenario_label}: 3D view", pad=8.0)
    ax_top.set_title(f"{scenario_label}: top-down view", pad=8.0)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.3)
    ax_top.grid(True, alpha=0.3)


def _make_trajectory_figure(run_dir: Path):
    fig = plt.figure(figsize=(10.5, 5.8))
    ax = fig.add_subplot(121, projection="3d")
    ax_top = fig.add_subplot(122)
    scenario = _run_scenario(run_dir)
    scenario_label = TRAJECTORY_SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title())
    _plot_trajectory_pair(run_dir, ax, ax_top, scenario, scenario_label)
    parts = run_dir.name.split("__")
    title = f"Sample trajectories: {parts[0]} ({parts[1]})" if len(parts) >= 2 else "Sample trajectories"
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, ax


def _make_multi_scenario_trajectory_figure(run_dirs: list[tuple[str, Path]]):
    n_rows = len(run_dirs)
    fig = plt.figure(figsize=(10.8, max(5.8, 2.95 * n_rows)))
    grid = fig.add_gridspec(
        n_rows,
        2,
        width_ratios=[1.15, 1.0],
        left=0.045,
        right=0.985,
        top=0.975,
        bottom=0.055,
        hspace=0.54,
        wspace=0.14,
    )
    for row_idx, (scenario, run_dir) in enumerate(run_dirs):
        ax = fig.add_subplot(grid[row_idx, 0], projection="3d")
        ax_top = fig.add_subplot(grid[row_idx, 1])
        scenario_label = TRAJECTORY_SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title())
        _plot_trajectory_pair(run_dir, ax, ax_top, scenario, scenario_label)
    return fig


def _plot_trajectory_from_run(run_dir: Path, out_dir: Path) -> bool:
    traj_path = run_dir / "trajectory.csv"
    if not traj_path.exists():
        return False
    fig, _ = _make_trajectory_figure(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_5_sample_trajectories"
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


def find_and_plot_trajectory(results_dir: Optional[Path], out_dir: Path) -> None:
    if results_dir is None or not results_dir.exists():
        return
    run_dirs = _collect_trajectory_runs(results_dir)
    if len(run_dirs) >= 2:
        fig = _make_multi_scenario_trajectory_figure(run_dirs)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "fig_5_sample_trajectories"
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return
    if len(run_dirs) == 1:
        _plot_trajectory_from_run(run_dirs[0][1], out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from summary results.")
    parser.add_argument("--summary", required=True, help="summary_by_condition.csv or directory containing it")
    parser.add_argument("--out", default="figures/ijmav", help="Output figure directory")
    parser.add_argument("--results", default=None, help="Optional raw results directory for trajectory plots")
    parser.add_argument("--trajectory-results", default=None, help="Optional trajectory-sample results directory")
    parser.add_argument(
        "--comparison-summary",
        "--extra-comparison-summary",
        dest="comparison_summaries",
        action="append",
        default=[],
        help="Optional baseline summary used for comparison-only plots; repeat to merge baseline sweeps",
    )
    args = parser.parse_args()

    summary = _read_summary(Path(args.summary))
    out_dir = Path(args.out)
    results_dir = Path(args.results) if args.results else None
    trajectory_results_dir = Path(args.trajectory_results) if args.trajectory_results else results_dir
    comparison_summaries = [_read_summary(Path(path)) for path in args.comparison_summaries]
    make_standard_figures(summary, out_dir, results_dir=results_dir, comparison_summary=comparison_summaries)
    find_and_plot_trajectory(trajectory_results_dir, out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
