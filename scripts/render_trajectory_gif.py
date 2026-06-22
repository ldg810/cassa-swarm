#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mav_swarm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    traj_path = run_dir / "trajectory.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory.csv in {run_dir}")
    traj = pd.read_csv(traj_path).sort_values(["t_s", "agent"])
    obs_path = run_dir / "obstacles.csv"
    obs = pd.read_csv(obs_path) if obs_path.exists() else None
    return traj, obs


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _agent_arrays(traj: pd.DataFrame, max_agents: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    agents = np.array(sorted(traj["agent"].unique()))
    if max_agents is not None and len(agents) > max_agents:
        keep_idx = np.linspace(0, len(agents) - 1, max_agents).round().astype(int)
        agents = agents[keep_idx]
        traj = traj[traj["agent"].isin(agents)]

    times = np.array(sorted(traj["t_s"].unique()), dtype=float)
    coords = np.empty((len(times), len(agents), 3), dtype=float)
    for a_idx, agent in enumerate(agents):
        sub = traj[traj["agent"] == agent].sort_values("t_s")
        if len(sub) != len(times):
            for dim_idx, dim in enumerate(["x", "y", "z"]):
                coords[:, a_idx, dim_idx] = np.interp(times, sub["t_s"], sub[dim])
        else:
            coords[:, a_idx, :] = sub[["x", "y", "z"]].to_numpy()
    return agents, times, coords


def _interpolate_coords(times: np.ndarray, coords: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    out = np.empty((len(frame_times), coords.shape[1], coords.shape[2]), dtype=float)
    for agent_idx in range(coords.shape[1]):
        for dim_idx in range(coords.shape[2]):
            out[:, agent_idx, dim_idx] = np.interp(frame_times, times, coords[:, agent_idx, dim_idx])
    return out


def _axis_limits(coords: np.ndarray, obs: pd.DataFrame | None) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    xyz_min = coords.reshape(-1, 3).min(axis=0)
    xyz_max = coords.reshape(-1, 3).max(axis=0)
    if obs is not None and not obs.empty:
        xmin = (obs["cx"] - obs["radius"]).min()
        xmax = (obs["cx"] + obs["radius"]).max()
        ymin = (obs["cy"] - obs["radius"]).min()
        ymax = (obs["cy"] + obs["radius"]).max()
        zmin = obs["height_min"].min()
        zmax = obs["height_max"].max()
        xyz_min = np.minimum(xyz_min, np.array([xmin, ymin, zmin], dtype=float))
        xyz_max = np.maximum(xyz_max, np.array([xmax, ymax, zmax], dtype=float))

    pad = np.maximum((xyz_max - xyz_min) * 0.08, np.array([1.0, 1.0, 0.25]))
    return tuple(zip(xyz_min - pad, xyz_max + pad))  # type: ignore[return-value]


def _draw_cylinder_wire(ax, row: pd.Series, color: str = "0.25") -> None:
    theta = np.linspace(0, 2 * np.pi, 64)
    xs = row.cx + row.radius * np.cos(theta)
    ys = row.cy + row.radius * np.sin(theta)
    z0 = row.height_min
    z1 = row.height_max
    ax.plot(xs, ys, np.full_like(xs, z0), color=color, linewidth=0.7, alpha=0.7)
    ax.plot(xs, ys, np.full_like(xs, z1), color=color, linewidth=0.7, alpha=0.7)
    for idx in range(0, len(theta), 8):
        ax.plot([xs[idx], xs[idx]], [ys[idx], ys[idx]], [z0, z1], color=color, linewidth=0.45, alpha=0.45)


def _goals(traj: pd.DataFrame, agents: np.ndarray) -> np.ndarray | None:
    if not {"goal_x", "goal_y", "goal_z"}.issubset(traj.columns):
        return None
    rows = []
    for agent in agents:
        sub = traj[traj["agent"] == agent].sort_values("t_s")
        rows.append(sub[["goal_x", "goal_y", "goal_z"]].iloc[-1].to_numpy(dtype=float))
    return np.vstack(rows)


def _scenario_label(metrics: dict) -> str:
    labels = {
        "corridor_obstacles": "Obstacle corridor",
        "crossing_traffic": "Crossing traffic",
        "open_flock": "Open flock",
        "merge_split": "Merge-split",
    }
    scenario = str(metrics.get("scenario", "")).strip()
    return labels.get(scenario, scenario.replace("_", " ").title() if scenario else "Trajectory")


def _title(metrics: dict, n_agents: int) -> str:
    if not metrics:
        return f"Trajectory animation: proposed controller, N={n_agents}"
    scenario_label = _scenario_label(metrics)
    parts = [
        f"{scenario_label}: proposed",
        f"N={int(metrics.get('n_agents', n_agents))}",
        f"seed={int(metrics.get('seed', -1))}",
        f"loss={100.0 * float(metrics.get('packet_loss', 0.0)):.0f}%",
        f"lat={float(metrics.get('latency_ms', 0.0)):.0f} ms",
    ]
    if metrics.get("final_reached_fraction") is not None:
        parts.append(f"reached={float(metrics['final_reached_fraction']):.3f}")
    if metrics.get("cumulative_collision_pairs") is not None:
        parts.append(f"collisions={int(metrics['cumulative_collision_pairs'])}")
    if metrics.get("cumulative_obstacle_contacts") is not None:
        parts.append(f"contacts={int(metrics['cumulative_obstacle_contacts'])}")
    return ", ".join(parts)


def render_gif(
    run_dir: Path,
    output: Path,
    preview: Path | None,
    fps: int,
    max_frames: int,
    trail_s: float,
    max_agents: int | None,
    trail_max_agents: int | None,
    color_by_reached: bool,
    dpi: int,
) -> None:
    traj, obs = _load_run(run_dir)
    metrics = _load_metrics(run_dir)
    agents, times, coords = _agent_arrays(traj, max_agents=max_agents)
    frame_times = np.linspace(times.min(), times.max(), max_frames)
    frames = _interpolate_coords(times, coords, frame_times)
    goals = _goals(traj, agents)
    xlim, ylim, zlim = _axis_limits(coords, obs)
    trail_frames = max(2, int(round(trail_s / max((frame_times[1] - frame_times[0]), 1e-6))))

    if color_by_reached and goals is not None:
        final_goal_dist = np.linalg.norm(coords[-1] - goals, axis=1)
        reached = final_goal_dist <= 0.75
        colors = np.empty((len(agents), 4), dtype=float)
        colors[reached] = matplotlib.colors.to_rgba("tab:blue", alpha=0.92)
        colors[~reached] = matplotlib.colors.to_rgba("tab:red", alpha=0.92)
    else:
        reached = None
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(agents)))

    fig = plt.figure(figsize=(10.5, 5.8), dpi=dpi)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    for ax in [ax3d]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=24, azim=-58)
        ax.grid(True, alpha=0.25)

    ax2d.set_xlim(*xlim)
    ax2d.set_ylim(*ylim)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("x (m)")
    ax2d.set_ylabel("y (m)")
    ax2d.grid(True, alpha=0.25)

    if obs is not None and not obs.empty:
        for _, row in obs.iterrows():
            _draw_cylinder_wire(ax3d, row)
            ax2d.add_patch(Circle((row.cx, row.cy), row.radius, facecolor="0.55", edgecolor="0.15", alpha=0.35))

    # Faint full-run path context.
    for agent_idx in range(len(agents)):
        ax3d.plot(coords[:, agent_idx, 0], coords[:, agent_idx, 1], coords[:, agent_idx, 2], color=colors[agent_idx], alpha=0.08, linewidth=0.45)
        ax2d.plot(coords[:, agent_idx, 0], coords[:, agent_idx, 1], color=colors[agent_idx], alpha=0.07, linewidth=0.45)

    if goals is not None:
        ax3d.scatter(goals[:, 0], goals[:, 1], goals[:, 2], marker="x", s=12, color="tab:red", alpha=0.45)
        ax2d.scatter(goals[:, 0], goals[:, 1], marker="x", s=10, color="tab:red", alpha=0.45)

    trail_agent_indices = np.arange(len(agents))
    if trail_max_agents is not None and len(trail_agent_indices) > trail_max_agents:
        keep_idx = np.linspace(0, len(trail_agent_indices) - 1, trail_max_agents).round().astype(int)
        trail_agent_indices = trail_agent_indices[keep_idx]
    trail3d = [ax3d.plot([], [], [], color=colors[i], linewidth=0.7, alpha=0.65)[0] for i in trail_agent_indices]
    trail2d = [ax2d.plot([], [], color=colors[i], linewidth=0.7, alpha=0.65)[0] for i in trail_agent_indices]
    initial_xyz = frames[0]
    marker_size = 14 if len(agents) <= 300 else 7
    pts3d = ax3d.scatter(initial_xyz[:, 0], initial_xyz[:, 1], initial_xyz[:, 2], s=marker_size, c=colors, depthshade=True)
    pts2d = ax2d.scatter(initial_xyz[:, 0], initial_xyz[:, 1], s=marker_size, c=colors, edgecolor="none", alpha=0.95)
    if color_by_reached and reached is not None:
        ax2d.scatter([], [], s=22, color="tab:blue", label=f"Reached ({int(reached.sum())})")
        ax2d.scatter([], [], s=22, color="tab:red", label=f"Unreached ({int((~reached).sum())})")
        ax2d.legend(loc="upper right", fontsize=8, frameon=True)
    time_text = fig.text(0.5, 0.955, "", ha="center", va="center", fontsize=12, weight="bold")
    fig.suptitle(_title(metrics, len(agents)), y=0.995, fontsize=10, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(frame_idx: int):
        xyz = frames[frame_idx]
        start = max(0, frame_idx - trail_frames)
        trail = frames[start : frame_idx + 1]
        for line_idx, agent_idx in enumerate(trail_agent_indices):
            trail3d[line_idx].set_data(trail[:, agent_idx, 0], trail[:, agent_idx, 1])
            trail3d[line_idx].set_3d_properties(trail[:, agent_idx, 2])
            trail2d[line_idx].set_data(trail[:, agent_idx, 0], trail[:, agent_idx, 1])

        pts3d._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        pts2d.set_offsets(xyz[:, :2])
        time_text.set_text(f"t = {frame_times[frame_idx]:5.1f} s / {frame_times[-1]:.1f} s")
        return [*trail3d, *trail2d, pts3d, pts2d, time_text]

    output.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frame_times), interval=1000 / fps, blit=False)
    anim.save(output, writer=PillowWriter(fps=fps))
    if preview is not None:
        update(len(frame_times) // 2)
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trajectory.csv run directory as an animated GIF.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--max-frames", type=int, default=220)
    parser.add_argument("--trail-s", type=float, default=8.0)
    parser.add_argument("--max-agents", type=int, default=None)
    parser.add_argument("--trail-max-agents", type=int, default=None)
    parser.add_argument("--color-by-reached", action="store_true")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    render_gif(
        run_dir=args.run_dir,
        output=args.output,
        preview=args.preview,
        fps=args.fps,
        max_frames=args.max_frames,
        trail_s=args.trail_s,
        max_agents=args.max_agents,
        trail_max_agents=args.trail_max_agents,
        color_by_reached=args.color_by_reached,
        dpi=args.dpi,
    )
    print(args.output)


if __name__ == "__main__":
    main()
