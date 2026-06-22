"""Replay a paper trajectory inside Crazyflow dynamics.

The input trajectory is the long-format CSV produced by the MAV swarm paper
simulator:

    t_s,agent,x,y,z,vx,vy,vz,goal_x,goal_y,goal_z

Crazyflow tracks the saved trajectory using ``Control.state`` and logs the
actual physically simulated states, pairwise clearance, obstacle contacts, and
downwash force magnitudes.

Examples:
    python examples/replay_paper_trajectory.py --trajectory /path/to/trajectory.csv --output-dir /tmp/replay
    MUJOCO_GL=egl python examples/replay_paper_trajectory.py --trajectory /path/to/trajectory.csv --device gpu
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.control import Control
from crazyflow.sim import (
    DownwashConfig,
    Physics,
    Sim,
    compute_downwash_forces,
    use_downwash,
)


class CylinderObstacle(NamedTuple):
    """Cylindrical obstacle used by the paper simulator."""

    center: np.ndarray
    radius: float
    height_min: float = 0.0
    height_max: float = 3.0


class PaperTrajectoryReference:
    """Dense array representation of a long-format paper trajectory."""

    def __init__(
        self,
        *,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        goals: np.ndarray,
    ) -> None:
        self.times = np.asarray(times, dtype=np.float32)
        self.positions = np.asarray(positions, dtype=np.float32)
        self.velocities = np.asarray(velocities, dtype=np.float32)
        self.goals = np.asarray(goals, dtype=np.float32)
        if self.positions.shape != self.velocities.shape or self.positions.shape != self.goals.shape:
            raise ValueError("positions, velocities, and goals must have matching shapes")
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError("trajectory arrays must have shape (n_times, n_agents, 3)")
        if self.times.ndim != 1 or self.times.shape[0] != self.positions.shape[0]:
            raise ValueError("times must have shape (n_times,)")

    @property
    def n_agents(self) -> int:
        return int(self.positions.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.times[-1])

    @property
    def final_goals(self) -> np.ndarray:
        return self.goals[-1]

    @classmethod
    def from_csv(cls, path: Path) -> "PaperTrajectoryReference":
        path = Path(path)
        if path.suffix.lower() == ".npz":
            return cls.from_npz(path)
        required = ["t_s", "agent", "x", "y", "z", "vx", "vy", "vz", "goal_x", "goal_y", "goal_z"]
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        missing = set(required).difference(header)
        if missing:
            raise ValueError(f"trajectory CSV missing columns: {sorted(missing)}")
        usecols = [header.index(name) for name in required]
        data = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=usecols,
            dtype=np.float32,
        )
        if data.size == 0:
            raise ValueError(f"trajectory CSV is empty: {path}")
        if data.ndim == 1:
            data = data[None, :]

        order = np.lexsort((data[:, 1], data[:, 0]))
        data = data[order]
        times = np.unique(data[:, 0]).astype(np.float32)
        agents = data[:, 1].astype(np.int64)
        unique_agents = np.unique(agents)
        if unique_agents.size == 0 or not np.array_equal(unique_agents, np.arange(unique_agents.size)):
            raise ValueError("agent ids must be contiguous and start at zero")

        n_times = int(times.shape[0])
        n_agents = int(unique_agents.size)
        expected_rows = n_times * n_agents
        if data.shape[0] != expected_rows:
            raise ValueError("trajectory CSV must contain one row for every time/agent pair")

        agent_grid = agents.reshape(n_times, n_agents)
        expected_agent_grid = np.broadcast_to(np.arange(n_agents, dtype=np.int64), agent_grid.shape)
        if np.array_equal(agent_grid, expected_agent_grid):
            positions = data[:, 2:5].reshape(n_times, n_agents, 3).astype(np.float32)
            velocities = data[:, 5:8].reshape(n_times, n_agents, 3).astype(np.float32)
            goals = data[:, 8:11].reshape(n_times, n_agents, 3).astype(np.float32)
        else:
            time_indices = np.searchsorted(times, data[:, 0])
            positions = np.full((n_times, n_agents, 3), np.nan, dtype=np.float32)
            velocities = np.full_like(positions, np.nan)
            goals = np.full_like(positions, np.nan)
            positions[time_indices, agents] = data[:, 2:5]
            velocities[time_indices, agents] = data[:, 5:8]
            goals[time_indices, agents] = data[:, 8:11]
            if np.isnan(positions).any() or np.isnan(velocities).any() or np.isnan(goals).any():
                raise ValueError("trajectory CSV must contain one row for every time/agent pair")

        return cls(times=times, positions=positions, velocities=velocities, goals=goals)

    @classmethod
    def from_npz(cls, path: Path) -> "PaperTrajectoryReference":
        with np.load(Path(path)) as data:
            required = {"times", "positions", "velocities", "goals"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"trajectory NPZ missing arrays: {sorted(missing)}")
            return cls(
                times=data["times"].astype(np.float32),
                positions=data["positions"].astype(np.float32),
                velocities=data["velocities"].astype(np.float32),
                goals=data["goals"].astype(np.float32),
            )

    def _segment_at(self, t: float) -> tuple[int, int, float]:
        if t <= float(self.times[0]):
            return 0, 0, 0.0
        if t >= float(self.times[-1]):
            last = len(self.times) - 1
            return last, last, 0.0
        right = int(np.searchsorted(self.times, t, side="right"))
        left = right - 1
        dt = float(self.times[right] - self.times[left])
        alpha = 0.0 if dt <= 0.0 else (float(t) - float(self.times[left])) / dt
        return left, right, float(alpha)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left, right, alpha = self._segment_at(t)
        if left == right:
            pos = self.positions[left]
            vel = self.velocities[left]
            goals = self.goals[left]
            if left > 0:
                dt = float(self.times[left] - self.times[left - 1])
                acc = (self.velocities[left] - self.velocities[left - 1]) / max(dt, 1e-6)
            else:
                acc = np.zeros_like(vel)
            return pos, vel, acc, goals

        pos = (1.0 - alpha) * self.positions[left] + alpha * self.positions[right]
        vel = (1.0 - alpha) * self.velocities[left] + alpha * self.velocities[right]
        goals = (1.0 - alpha) * self.goals[left] + alpha * self.goals[right]
        dt = float(self.times[right] - self.times[left])
        acc = (self.velocities[right] - self.velocities[left]) / max(dt, 1e-6)
        return pos.astype(np.float32), vel.astype(np.float32), acc.astype(np.float32), goals.astype(np.float32)

    def command_at(self, t: float) -> np.ndarray:
        pos, vel, acc, _ = self.sample(t)
        command = np.zeros((1, self.n_agents, 13), dtype=np.float32)
        command[0, :, 0:3] = pos
        command[0, :, 3:6] = vel
        command[0, :, 6:9] = acc
        return command


def load_obstacles(path: Path | None) -> list[CylinderObstacle]:
    if path is None or not Path(path).exists():
        return []
    obstacles: list[CylinderObstacle] = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"cx", "cy", "cz", "radius", "height_min", "height_max"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"obstacle CSV missing columns: {sorted(missing)}")
        for row in reader:
            obstacles.append(
                CylinderObstacle(
                    center=np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])], dtype=np.float32),
                    radius=float(row["radius"]),
                    height_min=float(row["height_min"]),
                    height_max=float(row["height_max"]),
                )
            )
    return obstacles


def pairwise_metrics(
    positions: np.ndarray,
    *,
    collision_radius: float,
    safety_radius: float,
    active_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    if active_mask is not None:
        positions = positions[np.asarray(active_mask, dtype=bool)]
    if positions.shape[0] < 2:
        return {
            "min_inter_agent_distance_m": float("inf"),
            "collision_pairs": 0,
            "safety_violation_pairs": 0,
        }
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    tri = np.triu_indices(positions.shape[0], k=1)
    pair_dist = dist[tri]
    return {
        "min_inter_agent_distance_m": float(np.min(pair_dist)),
        "collision_pairs": int(np.sum(pair_dist < collision_radius)),
        "safety_violation_pairs": int(np.sum(pair_dist < safety_radius)),
    }


def count_obstacle_contacts(
    positions: np.ndarray,
    obstacles: list[CylinderObstacle],
    *,
    clearance: float,
    active_mask: np.ndarray | None = None,
) -> int:
    if active_mask is not None:
        positions = positions[np.asarray(active_mask, dtype=bool)]
    contacts = 0
    for obstacle in obstacles:
        rel_xy = positions[:, :2] - obstacle.center[:2]
        dist_xy = np.linalg.norm(rel_xy, axis=1)
        z_ok = (positions[:, 2] >= obstacle.height_min) & (positions[:, 2] <= obstacle.height_max)
        contacts += int(np.sum((dist_xy < obstacle.radius + clearance) & z_ok))
    return contacts


def initialize_replay_state(sim: Sim, reference: PaperTrajectoryReference) -> None:
    pos, vel, _, _ = reference.sample(0.0)
    n_drones = reference.n_agents
    quat = np.zeros((1, n_drones, 4), dtype=np.float32)
    quat[0, :, 3] = 1.0
    zeros_3 = np.zeros((1, n_drones, 3), dtype=np.float32)
    rotor_vel = np.zeros((1, n_drones, 4), dtype=np.float32)
    rotor_vel[0, :, 0] = np.asarray(sim.data.params.mass)[0, :, 0] * 9.81

    states = sim.data.states.replace(
        pos=jnp.asarray(pos[None, :, :]),
        quat=jnp.asarray(quat),
        vel=jnp.asarray(vel[None, :, :]),
        ang_vel=jnp.asarray(zeros_3),
        force=jnp.asarray(zeros_3),
        torque=jnp.asarray(zeros_3),
        rotor_vel=jnp.asarray(rotor_vel),
    )
    sim.data = sim.data.replace(states=states, core=sim.data.core.replace(mjx_synced=False))


def infer_obstacle_path(trajectory_path: Path) -> Path | None:
    candidate = trajectory_path.parent / "obstacles.csv"
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trajectory", type=Path, required=True, help="paper trajectory.csv or trajectory.npz path")
    parser.add_argument("--obstacles", type=Path, default=None, help="optional obstacles.csv path")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for replay outputs")
    parser.add_argument("--duration", type=float, default=None, help="override replay duration")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu", help="JAX backend device")
    parser.add_argument("--freq", type=int, default=500, help="Crazyflow physics frequency")
    parser.add_argument("--state-freq", type=int, default=100, help="Crazyflow state control frequency")
    parser.add_argument("--log-every", type=float, default=0.5, help="timeseries log interval in seconds")
    parser.add_argument("--trajectory-stride", type=int, default=20, help="control steps per actual trajectory row")
    parser.add_argument("--goal-tolerance", type=float, default=0.75, help="goal reach tolerance in meters")
    parser.add_argument("--completion-fraction", type=float, default=0.90, help="mission completion threshold")
    parser.add_argument("--collision-radius", type=float, default=0.12, help="inter-drone collision radius")
    parser.add_argument("--safety-radius", type=float, default=0.35, help="inter-drone safety radius")
    parser.add_argument("--no-downwash", action="store_true", help="disable downwash disturbance")
    parser.add_argument("--downwash-strength", type=float, default=1.0, help="downwash strength multiplier")
    parser.add_argument("--downwash-force-cap-ratio", type=float, default=0.6, help="max downwash force / weight")
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="do not block on GPU completion after each control step",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = PaperTrajectoryReference.from_csv(args.trajectory)
    obstacle_path = args.obstacles if args.obstacles is not None else infer_obstacle_path(args.trajectory)
    obstacles = load_obstacles(obstacle_path)
    duration_s = reference.duration_s if args.duration is None else min(args.duration, reference.duration_s)

    sim = Sim(
        n_worlds=1,
        n_drones=reference.n_agents,
        physics=Physics.so_rpy_rotor_drag,
        control=Control.state,
        freq=args.freq,
        attitude_freq=args.freq,
        state_freq=args.state_freq,
        integrator="rk4",
        device=args.device,
        fused_mjx_model=True,
    )
    downwash_config = DownwashConfig(
        strength=args.downwash_strength,
        force_cap_ratio=args.downwash_force_cap_ratio,
    )
    if not args.no_downwash:
        use_downwash(sim, downwash_config)
    sim.reset()
    initialize_replay_state(sim, reference)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_dir / "timeseries.csv"
    actual_path = output_dir / "actual_trajectory.csv"
    metrics_path = output_dir / "metrics.json"

    physics_steps_per_control = sim.freq // sim.control_freq
    n_control_steps = int(round(duration_s * sim.control_freq))
    log_interval = max(int(round(args.log_every * sim.control_freq)), 1)
    cumulative_collision_pairs = 0
    cumulative_safety_violation_pairs = 0
    cumulative_obstacle_contacts = 0
    active_cumulative_collision_pairs = 0
    active_cumulative_safety_violation_pairs = 0
    active_cumulative_obstacle_contacts = 0
    min_distance_global = float("inf")
    active_min_distance_global = float("inf")
    max_tracking_error = 0.0
    max_tracking_error_time = 0.0
    max_speed = 0.0
    max_logged_downwash_force = 0.0
    min_distance_time = 0.0
    first_collision_time = None
    first_safety_violation_time = None
    first_obstacle_contact_time = None
    first_active_collision_time = None
    first_active_safety_violation_time = None
    first_active_obstacle_contact_time = None
    max_collision_pairs = 0
    max_collision_time = 0.0
    max_obstacle_contacts = 0
    max_obstacle_contact_time = 0.0
    max_active_collision_pairs = 0
    max_active_collision_time = 0.0
    max_active_obstacle_contacts = 0
    max_active_obstacle_contact_time = 0.0
    active_min_distance_time = 0.0
    logged_downwash_forces: list[float] = []
    records: list[dict[str, float | int]] = []
    trajectory_rows: list[dict[str, float | int]] = []
    completed_mask = np.zeros(reference.n_agents, dtype=bool)
    start_wall = time.perf_counter()

    try:
        for step in range(n_control_steps + 1):
            t = step / sim.control_freq
            ref_pos, _, _, goals = reference.sample(t)
            actual_pos = np.asarray(sim.data.states.pos[0])
            actual_vel = np.asarray(sim.data.states.vel[0])
            tracking_error = np.linalg.norm(actual_pos - ref_pos, axis=1)
            goal_dist = np.linalg.norm(actual_pos - goals, axis=1)
            completed_mask |= goal_dist <= args.goal_tolerance
            active_mask = ~completed_mask
            speed = np.linalg.norm(actual_vel, axis=1)
            pair = pairwise_metrics(
                actual_pos,
                collision_radius=args.collision_radius,
                safety_radius=args.safety_radius,
            )
            active_pair = pairwise_metrics(
                actual_pos,
                collision_radius=args.collision_radius,
                safety_radius=args.safety_radius,
                active_mask=active_mask,
            )
            obstacle_contacts = count_obstacle_contacts(
                actual_pos,
                obstacles,
                clearance=args.collision_radius,
            )
            active_obstacle_contacts = count_obstacle_contacts(
                actual_pos,
                obstacles,
                clearance=args.collision_radius,
                active_mask=active_mask,
            )

            cumulative_collision_pairs += int(pair["collision_pairs"])
            cumulative_safety_violation_pairs += int(pair["safety_violation_pairs"])
            cumulative_obstacle_contacts += obstacle_contacts
            active_cumulative_collision_pairs += int(active_pair["collision_pairs"])
            active_cumulative_safety_violation_pairs += int(active_pair["safety_violation_pairs"])
            active_cumulative_obstacle_contacts += active_obstacle_contacts
            if float(pair["min_inter_agent_distance_m"]) < min_distance_global:
                min_distance_global = float(pair["min_inter_agent_distance_m"])
                min_distance_time = t
            if float(active_pair["min_inter_agent_distance_m"]) < active_min_distance_global:
                active_min_distance_global = float(active_pair["min_inter_agent_distance_m"])
                active_min_distance_time = t
            if int(pair["collision_pairs"]) > 0 and first_collision_time is None:
                first_collision_time = t
            if int(pair["safety_violation_pairs"]) > 0 and first_safety_violation_time is None:
                first_safety_violation_time = t
            if obstacle_contacts > 0 and first_obstacle_contact_time is None:
                first_obstacle_contact_time = t
            if int(active_pair["collision_pairs"]) > 0 and first_active_collision_time is None:
                first_active_collision_time = t
            if int(active_pair["safety_violation_pairs"]) > 0 and first_active_safety_violation_time is None:
                first_active_safety_violation_time = t
            if active_obstacle_contacts > 0 and first_active_obstacle_contact_time is None:
                first_active_obstacle_contact_time = t
            if int(pair["collision_pairs"]) > max_collision_pairs:
                max_collision_pairs = int(pair["collision_pairs"])
                max_collision_time = t
            if obstacle_contacts > max_obstacle_contacts:
                max_obstacle_contacts = obstacle_contacts
                max_obstacle_contact_time = t
            if int(active_pair["collision_pairs"]) > max_active_collision_pairs:
                max_active_collision_pairs = int(active_pair["collision_pairs"])
                max_active_collision_time = t
            if active_obstacle_contacts > max_active_obstacle_contacts:
                max_active_obstacle_contacts = active_obstacle_contacts
                max_active_obstacle_contact_time = t
            if float(np.max(tracking_error)) > max_tracking_error:
                max_tracking_error = float(np.max(tracking_error))
                max_tracking_error_time = t
            max_speed = max(max_speed, float(np.max(speed)))

            if step % log_interval == 0 or step == n_control_steps:
                if args.no_downwash:
                    downwash_norms = np.zeros(reference.n_agents, dtype=np.float32)
                else:
                    downwash_norms = np.linalg.norm(
                        np.asarray(compute_downwash_forces(sim.data, downwash_config)[0]),
                        axis=1,
                    )
                max_logged_downwash_force = max(max_logged_downwash_force, float(np.max(downwash_norms)))
                logged_downwash_forces.append(float(np.mean(downwash_norms)))
                record = {
                    "t_s": t,
                    "tracking_error_mean_m": float(np.mean(tracking_error)),
                    "tracking_error_p95_m": float(np.quantile(tracking_error, 0.95)),
                    "tracking_error_max_m": float(np.max(tracking_error)),
                    "mean_goal_distance_m": float(np.mean(goal_dist)),
                    "reached_fraction": float(np.mean(completed_mask)),
                    "min_inter_agent_distance_m": float(pair["min_inter_agent_distance_m"]),
                    "collision_pairs": int(pair["collision_pairs"]),
                    "safety_violation_pairs": int(pair["safety_violation_pairs"]),
                    "obstacle_contacts": obstacle_contacts,
                    "active_min_inter_agent_distance_m": float(active_pair["min_inter_agent_distance_m"]),
                    "active_collision_pairs": int(active_pair["collision_pairs"]),
                    "active_safety_violation_pairs": int(active_pair["safety_violation_pairs"]),
                    "active_obstacle_contacts": active_obstacle_contacts,
                    "speed_max_mps": float(np.max(speed)),
                    "downwash_force_mean_N": float(np.mean(downwash_norms)),
                    "downwash_force_max_N": float(np.max(downwash_norms)),
                }
                records.append(record)
                print(
                    f"t={t:5.1f}s | err_p95={record['tracking_error_p95_m']:.3f} m | "
                    f"min_d={record['min_inter_agent_distance_m']:.3f} m | "
                    f"coll={record['collision_pairs']}/{record['active_collision_pairs']} | "
                    f"obs={record['obstacle_contacts']}/{record['active_obstacle_contacts']} | "
                    f"reached={record['reached_fraction']:.2f}"
                )

            if step % args.trajectory_stride == 0 or step == n_control_steps:
                for agent_id in range(reference.n_agents):
                    trajectory_rows.append(
                        {
                            "t_s": t,
                            "agent": agent_id,
                            "x": float(actual_pos[agent_id, 0]),
                            "y": float(actual_pos[agent_id, 1]),
                            "z": float(actual_pos[agent_id, 2]),
                            "vx": float(actual_vel[agent_id, 0]),
                            "vy": float(actual_vel[agent_id, 1]),
                            "vz": float(actual_vel[agent_id, 2]),
                            "ref_x": float(ref_pos[agent_id, 0]),
                            "ref_y": float(ref_pos[agent_id, 1]),
                            "ref_z": float(ref_pos[agent_id, 2]),
                            "goal_x": float(goals[agent_id, 0]),
                            "goal_y": float(goals[agent_id, 1]),
                            "goal_z": float(goals[agent_id, 2]),
                        }
                    )

            if step == n_control_steps:
                break
            sim.state_control(reference.command_at(t))
            sim.step(physics_steps_per_control)
            if not args.no_sync:
                jax.block_until_ready(sim.data.states.pos)
    finally:
        sim.close()

    wall_s = time.perf_counter() - start_wall
    final = records[-1]
    possible_pairs = max(1, reference.n_agents * (reference.n_agents - 1) // 2)
    metrics = {
        "source_trajectory": str(args.trajectory),
        "source_obstacles": str(obstacle_path) if obstacle_path is not None else None,
        "n_agents": reference.n_agents,
        "simulated_s": float(duration_s),
        "wall_s": float(wall_s),
        "real_time_factor": float(duration_s / wall_s) if wall_s > 0 else 0.0,
        "device": args.device,
        "jax_backend": jax.default_backend(),
        "downwash_enabled": not args.no_downwash,
        "downwash_strength": float(args.downwash_strength) if not args.no_downwash else 0.0,
        "collision_radius_m": float(args.collision_radius),
        "safety_radius_m": float(args.safety_radius),
        "goal_tolerance_m": float(args.goal_tolerance),
        "completion_fraction": float(args.completion_fraction),
        "min_inter_agent_distance_m": float(min_distance_global),
        "min_inter_agent_distance_time_s": float(min_distance_time),
        "cumulative_collision_pairs": int(cumulative_collision_pairs),
        "first_collision_time_s": first_collision_time,
        "max_collision_pairs_single_step": int(max_collision_pairs),
        "max_collision_pairs_single_step_time_s": float(max_collision_time),
        "collision_rate_per_pair_step": float(
            cumulative_collision_pairs / max(1, (n_control_steps + 1) * possible_pairs)
        ),
        "cumulative_safety_violation_pairs": int(cumulative_safety_violation_pairs),
        "first_safety_violation_time_s": first_safety_violation_time,
        "cumulative_obstacle_contacts": int(cumulative_obstacle_contacts),
        "first_obstacle_contact_time_s": first_obstacle_contact_time,
        "max_obstacle_contacts_single_step": int(max_obstacle_contacts),
        "max_obstacle_contacts_single_step_time_s": float(max_obstacle_contact_time),
        "active_min_inter_agent_distance_m": float(active_min_distance_global),
        "active_min_inter_agent_distance_time_s": float(active_min_distance_time),
        "active_cumulative_collision_pairs": int(active_cumulative_collision_pairs),
        "first_active_collision_time_s": first_active_collision_time,
        "active_max_collision_pairs_single_step": int(max_active_collision_pairs),
        "active_max_collision_pairs_single_step_time_s": float(max_active_collision_time),
        "active_cumulative_safety_violation_pairs": int(active_cumulative_safety_violation_pairs),
        "first_active_safety_violation_time_s": first_active_safety_violation_time,
        "active_cumulative_obstacle_contacts": int(active_cumulative_obstacle_contacts),
        "first_active_obstacle_contact_time_s": first_active_obstacle_contact_time,
        "active_max_obstacle_contacts_single_step": int(max_active_obstacle_contacts),
        "active_max_obstacle_contacts_single_step_time_s": float(max_active_obstacle_contact_time),
        "final_reached_fraction": float(final["reached_fraction"]),
        "final_mean_goal_distance_m": float(final["mean_goal_distance_m"]),
        "final_tracking_error_mean_m": float(final["tracking_error_mean_m"]),
        "final_tracking_error_p95_m": float(final["tracking_error_p95_m"]),
        "final_tracking_error_max_m": float(final["tracking_error_max_m"]),
        "max_tracking_error_m": float(max_tracking_error),
        "max_tracking_error_time_s": float(max_tracking_error_time),
        "max_speed_mps": float(max_speed),
        "mean_logged_downwash_force_N": float(np.mean(logged_downwash_forces)) if logged_downwash_forces else 0.0,
        "max_logged_downwash_force_N": float(max_logged_downwash_force),
        "mission_success": bool(
            float(final["reached_fraction"]) >= args.completion_fraction
            and cumulative_collision_pairs == 0
            and cumulative_obstacle_contacts == 0
        ),
        "active_traffic_success": bool(
            float(final["reached_fraction"]) >= args.completion_fraction
            and active_cumulative_collision_pairs == 0
            and active_cumulative_obstacle_contacts == 0
        ),
    }

    with timeseries_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with actual_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        "summary | "
        f"success={metrics['mission_success']} | "
        f"active_success={metrics['active_traffic_success']} | "
        f"reached={metrics['final_reached_fraction']:.3f} | "
        f"min_d={metrics['min_inter_agent_distance_m']:.3f} m | "
        f"collisions={metrics['cumulative_collision_pairs']}/"
        f"{metrics['active_cumulative_collision_pairs']} | "
        f"obstacle_contacts={metrics['cumulative_obstacle_contacts']}/"
        f"{metrics['active_cumulative_obstacle_contacts']} | "
        f"rtf={metrics['real_time_factor']:.2f}"
    )
    print(f"saved replay metrics to {metrics_path}")


if __name__ == "__main__":
    main()
