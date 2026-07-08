from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import RunSpec, save_yaml
from .controllers import compute_acceleration, get_controller_traits, limit_norm
from .metrics import (
    connectivity_metrics,
    count_pairs_within,
    mission_progress,
    order_parameter,
    pairwise_min_distance,
)
from .scenarios import Obstacle, make_scenario

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

EPS = 1e-9


@dataclass
class ProjectionTelemetry:
    pairwise_projection_calls: int = 0
    pairwise_projection_pair_adjustments: int = 0
    pairwise_projection_agent_adjustments: int = 0
    pairwise_projection_total_displacement_m: float = 0.0
    pairwise_projection_max_displacement_m: float = 0.0
    pairwise_projection_velocity_corrections: int = 0
    obstacle_projection_calls: int = 0
    obstacle_projection_agent_adjustments: int = 0
    obstacle_projection_total_displacement_m: float = 0.0
    obstacle_projection_max_displacement_m: float = 0.0
    obstacle_projection_velocity_corrections: int = 0
    pre_projection_state_checks: int = 0
    pre_projection_collision_pairs: int = 0
    pre_projection_obstacle_contacts: int = 0
    pre_projection_pairwise_violation_pairs: int = 0
    pre_projection_pairwise_violation_agent_steps: int = 0
    pre_projection_pairwise_total_penetration_m: float = 0.0
    pre_projection_pairwise_max_penetration_m: float = 0.0
    pre_projection_pairwise_penetrations_m: List[float] = field(default_factory=list)
    pre_projection_obstacle_violation_agent_steps: int = 0
    pre_projection_obstacle_total_penetration_m: float = 0.0
    pre_projection_obstacle_max_penetration_m: float = 0.0
    pre_projection_obstacle_min_clearance_margin_m: float | None = None
    pre_projection_obstacle_penetrations_m: List[float] = field(default_factory=list)

    def record_pairwise_iteration(
        self,
        position_delta: np.ndarray,
        pair_adjustments: int,
        velocity_corrections: int,
    ) -> None:
        norms = np.linalg.norm(position_delta, axis=1)
        adjusted = norms > EPS
        if pair_adjustments <= 0 or not np.any(adjusted):
            return
        self.pairwise_projection_pair_adjustments += int(pair_adjustments)
        self.pairwise_projection_agent_adjustments += int(np.sum(adjusted))
        self.pairwise_projection_total_displacement_m += float(np.sum(norms[adjusted]))
        self.pairwise_projection_max_displacement_m = max(
            self.pairwise_projection_max_displacement_m,
            float(np.max(norms[adjusted])),
        )
        self.pairwise_projection_velocity_corrections += int(velocity_corrections)

    def record_obstacle_adjustment(self, displacement_m: float, velocity_corrected: bool) -> None:
        if displacement_m <= EPS:
            return
        self.obstacle_projection_agent_adjustments += 1
        self.obstacle_projection_total_displacement_m += float(displacement_m)
        self.obstacle_projection_max_displacement_m = max(
            self.obstacle_projection_max_displacement_m,
            float(displacement_m),
        )
        if velocity_corrected:
            self.obstacle_projection_velocity_corrections += 1

    def record_pre_projection_pairwise(
        self,
        pairs: List[Tuple[int, int]],
        penetrations_m: np.ndarray,
    ) -> None:
        if not pairs:
            return
        positive = penetrations_m[penetrations_m > EPS]
        if positive.size == 0:
            return
        affected_agents = {int(i) for pair in pairs for i in pair}
        self.pre_projection_pairwise_violation_pairs += int(positive.size)
        self.pre_projection_pairwise_violation_agent_steps += len(affected_agents)
        self.pre_projection_pairwise_total_penetration_m += float(np.sum(positive))
        self.pre_projection_pairwise_max_penetration_m = max(
            self.pre_projection_pairwise_max_penetration_m,
            float(np.max(positive)),
        )
        self.pre_projection_pairwise_penetrations_m.extend(float(v) for v in positive)

    def record_pre_projection_obstacle(
        self,
        penetrations_m: np.ndarray,
        clearance_margins_m: np.ndarray,
    ) -> None:
        if clearance_margins_m.size:
            min_margin = float(np.min(clearance_margins_m))
            if self.pre_projection_obstacle_min_clearance_margin_m is None:
                self.pre_projection_obstacle_min_clearance_margin_m = min_margin
            else:
                self.pre_projection_obstacle_min_clearance_margin_m = min(
                    self.pre_projection_obstacle_min_clearance_margin_m,
                    min_margin,
                )

        positive = penetrations_m[penetrations_m > EPS]
        if positive.size == 0:
            return
        self.pre_projection_obstacle_violation_agent_steps += int(positive.size)
        self.pre_projection_obstacle_total_penetration_m += float(np.sum(positive))
        self.pre_projection_obstacle_max_penetration_m = max(
            self.pre_projection_obstacle_max_penetration_m,
            float(np.max(positive)),
        )
        self.pre_projection_obstacle_penetrations_m.extend(float(v) for v in positive)


def make_run_id(spec: RunSpec) -> str:
    pl = int(round(spec.comm.packet_loss * 100))
    lat = int(round(spec.comm.latency_ms))
    run_id = f"{spec.scenario}__{spec.controller}__N{spec.n_agents}__pl{pl:02d}__lat{lat:03d}__seed{spec.seed}"
    suffix = spec.metadata.get("run_id_suffix")
    if suffix:
        run_id = f"{run_id}__{suffix}"
    return run_id


def _find_neighbors(
    own_positions: np.ndarray,
    est_positions: np.ndarray,
    est_velocities: np.ndarray,
    comm_radius: float,
    k_neighbors: int,
    packet_loss: float,
    rng: np.random.Generator,
    active_mask: np.ndarray | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], int, List[Tuple[int, int]]]:
    n = own_positions.shape[0]
    neighbor_pos: List[np.ndarray] = []
    neighbor_vel: List[np.ndarray] = []
    message_count = 0
    undirected_edges = set()
    if active_mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)

    if cKDTree is not None:
        tree = cKDTree(est_positions)
        candidate_lists = tree.query_ball_point(own_positions, comm_radius)
    else:
        candidate_lists = []
        for i in range(n):
            d = np.linalg.norm(est_positions - own_positions[i], axis=1)
            candidate_lists.append(list(np.where(d <= comm_radius)[0]))

    for i in range(n):
        if not active[i]:
            neighbor_pos.append(np.empty((0, 3), dtype=float))
            neighbor_vel.append(np.empty((0, 3), dtype=float))
            continue

        candidates = [j for j in candidate_lists[i] if j != i and active[j]]
        if candidates:
            d = np.linalg.norm(est_positions[candidates] - own_positions[i], axis=1)
            order = np.argsort(d)[:k_neighbors]
            selected = np.asarray(candidates, dtype=int)[order]
            if packet_loss > 0.0 and selected.size > 0:
                keep = rng.random(selected.size) >= packet_loss
                selected = selected[keep]
        else:
            selected = np.empty(0, dtype=int)

        if selected.size > 0:
            neighbor_pos.append(est_positions[selected].copy())
            neighbor_vel.append(est_velocities[selected].copy())
            message_count += int(selected.size)
            for j in selected:
                undirected_edges.add((min(i, int(j)), max(i, int(j))))
        else:
            neighbor_pos.append(np.empty((0, 3), dtype=float))
            neighbor_vel.append(np.empty((0, 3), dtype=float))

    return neighbor_pos, neighbor_vel, message_count, list(undirected_edges)


def _make_observations(
    positions: np.ndarray,
    velocities: np.ndarray,
    history_pos: List[np.ndarray],
    history_vel: List[np.ndarray],
    spec: RunSpec,
    use_prediction: bool,
    rng: np.random.Generator,
    active_mask: np.ndarray | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], int, List[Tuple[int, int]]]:
    lag_steps = max(0, int(round((spec.comm.latency_ms / 1000.0) / spec.sim.dt_s)))
    source_idx = max(0, len(history_pos) - 1 - lag_steps)
    pos_delayed = history_pos[source_idx]
    vel_delayed = history_vel[source_idx]

    if use_prediction:
        est_positions = pos_delayed + vel_delayed * lag_steps * spec.sim.dt_s
    else:
        est_positions = pos_delayed.copy()
    est_velocities = vel_delayed.copy()

    if spec.comm.position_noise_std_m > 0:
        est_positions = est_positions + rng.normal(0.0, spec.comm.position_noise_std_m, est_positions.shape)
    if spec.comm.velocity_noise_std_mps > 0:
        est_velocities = est_velocities + rng.normal(0.0, spec.comm.velocity_noise_std_mps, est_velocities.shape)

    return _find_neighbors(
        own_positions=positions,
        est_positions=est_positions,
        est_velocities=est_velocities,
        comm_radius=spec.comm.comm_radius_m,
        k_neighbors=spec.comm.k_neighbors,
        packet_loss=spec.comm.packet_loss,
        rng=rng,
        active_mask=active_mask,
    )


def _metadata_flag(spec: RunSpec, key: str) -> bool:
    return bool(spec.metadata.get(key, False))


def _projection_globally_enabled(spec: RunSpec) -> bool:
    return bool(spec.metadata.get("projection_enabled", True))


def _metadata_float(spec: RunSpec, key: str, default: float) -> float:
    value = spec.metadata.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _uses_cassa_reference_stack(spec: RunSpec, traits) -> bool:
    if _metadata_flag(spec, "communication_only_safety"):
        return False
    return traits.family == "proposed" and traits.use_safety_shield


def _uses_local_safety_layer(spec: RunSpec, traits) -> bool:
    if _metadata_flag(spec, "communication_only_safety"):
        return False
    if _metadata_flag(spec, "disable_local_safety_shield"):
        return False
    return (
        _uses_cassa_reference_stack(spec, traits)
        or _metadata_flag(spec, "enable_shared_local_safety")
        or _metadata_flag(spec, "enable_shared_cassa_safety_layer")
    )


def _uses_cassa_speed_governors(spec: RunSpec, traits) -> bool:
    if _metadata_flag(spec, "communication_only_safety"):
        return False
    if _metadata_flag(spec, "disable_density_controls"):
        return False
    return (
        _uses_cassa_reference_stack(spec, traits)
        or _metadata_flag(spec, "enable_shared_speed_governors")
        or _metadata_flag(spec, "enable_shared_cassa_safety_layer")
    )


def _uses_proposed_local_safety(spec: RunSpec, traits) -> bool:
    return _uses_local_safety_layer(spec, traits)


def _uses_density_controls(spec: RunSpec, traits) -> bool:
    return _uses_cassa_speed_governors(spec, traits) and spec.n_agents >= 300


def _uses_speed_controls(spec: RunSpec, traits) -> bool:
    degraded_comm = spec.comm.packet_loss > 0.0 or spec.comm.latency_ms > 0.0
    return _uses_cassa_speed_governors(spec, traits) and (spec.n_agents >= 300 or degraded_comm)


def _uses_pairwise_brake_v3(spec: RunSpec, traits) -> bool:
    if _metadata_flag(spec, "disable_crossing_governor"):
        return False
    return _uses_cassa_speed_governors(spec, traits) and _metadata_flag(spec, "enable_pairwise_brake_v3")


def _uses_pairwise_projection_stack(spec: RunSpec, traits) -> bool:
    return _uses_cassa_reference_stack(spec, traits) or _metadata_flag(spec, "enable_shared_pairwise_projection")


def _uses_pairwise_projection(spec: RunSpec, traits) -> bool:
    return (
        _uses_pairwise_projection_stack(spec, traits)
        and _projection_globally_enabled(spec)
        and not _metadata_flag(spec, "disable_pairwise_projection")
    )


def _uses_obstacle_command_filter(spec: RunSpec, traits) -> bool:
    if _metadata_flag(spec, "disable_obstacle_command_filter"):
        return False
    return (
        _uses_cassa_reference_stack(spec, traits)
        or _metadata_flag(spec, "enable_shared_obstacle_filter")
        or _metadata_flag(spec, "enable_shared_cassa_safety_layer")
    )


def _uses_obstacle_projection(spec: RunSpec, traits) -> bool:
    projection_stack = _uses_cassa_reference_stack(spec, traits) or _metadata_flag(spec, "enable_shared_obstacle_projection")
    return (
        projection_stack
        and _projection_globally_enabled(spec)
        and _metadata_flag(spec, "enable_obstacle_projection")
        and not _metadata_flag(spec, "disable_obstacle_projection")
    )


def _local_safety_neighbor_limit(spec: RunSpec) -> int:
    if spec.n_agents <= 1:
        return 0
    if spec.n_agents >= 700:
        return min(spec.n_agents - 1, 64)
    if spec.n_agents >= 300:
        return min(spec.n_agents - 1, 48)
    return min(spec.n_agents - 1, max(spec.comm.k_neighbors, 24))


def _local_safety_update_interval_steps(spec: RunSpec) -> int:
    return max(
        1,
        int(round(1.0 / max(spec.comm.local_safety_update_rate_hz, EPS) / spec.sim.dt_s)),
    )


def _obstacle_update_interval_steps(spec: RunSpec) -> int:
    return max(
        1,
        int(round(1.0 / max(spec.comm.obstacle_update_rate_hz, EPS) / spec.sim.dt_s)),
    )


def _make_local_safety_observations(
    positions: np.ndarray,
    velocities: np.ndarray,
    spec: RunSpec,
    rng: np.random.Generator,
    history_pos: List[np.ndarray] | None = None,
    history_vel: List[np.ndarray] | None = None,
    active_mask: np.ndarray | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    neighbor_radius = spec.mav.safety_influence_m + 2.0 * spec.mav.max_speed_mps * spec.sim.dt_s
    lag_steps = max(0, int(round((spec.comm.local_safety_latency_ms / 1000.0) / spec.sim.dt_s)))
    if lag_steps > 0 and history_pos is not None and history_vel is not None:
        source_idx = max(0, len(history_pos) - 1 - lag_steps)
        est_positions = history_pos[source_idx].copy()
        est_velocities = history_vel[source_idx].copy()
    else:
        est_positions = positions
        est_velocities = velocities
    if spec.comm.local_safety_position_noise_std_m > 0.0:
        est_positions = est_positions + rng.normal(
            0.0,
            spec.comm.local_safety_position_noise_std_m,
            est_positions.shape,
        )
    if spec.comm.local_safety_velocity_noise_std_mps > 0.0:
        est_velocities = est_velocities + rng.normal(
            0.0,
            spec.comm.local_safety_velocity_noise_std_mps,
            est_velocities.shape,
        )
    neighbor_pos, neighbor_vel, _, _ = _find_neighbors(
        own_positions=positions,
        est_positions=est_positions,
        est_velocities=est_velocities,
        comm_radius=neighbor_radius,
        k_neighbors=_local_safety_neighbor_limit(spec),
        packet_loss=spec.comm.local_safety_dropout,
        rng=rng,
        active_mask=active_mask,
    )
    return neighbor_pos, neighbor_vel


def _active_positions(positions: np.ndarray, active_mask: np.ndarray | None) -> np.ndarray:
    if active_mask is None:
        return positions
    return positions[np.asarray(active_mask, dtype=bool)]


def _update_completed_mask(
    completed_mask: np.ndarray,
    positions: np.ndarray,
    goals: np.ndarray,
    goal_tolerance: float,
) -> np.ndarray:
    completed = np.asarray(completed_mask, dtype=bool).copy()
    reached_now = np.linalg.norm(positions - goals, axis=1) <= float(goal_tolerance)
    completed |= reached_now
    return completed


def _active_pairwise_min_distance(positions: np.ndarray, active_mask: np.ndarray | None) -> float:
    active = _active_positions(positions, active_mask)
    if active.shape[0] < 2:
        return float("inf")
    return pairwise_min_distance(active)


def _count_active_pairs_within(positions: np.ndarray, radius: float, active_mask: np.ndarray | None) -> int:
    active = _active_positions(positions, active_mask)
    if active.shape[0] < 2:
        return 0
    return count_pairs_within(active, radius)


def _count_active_obstacle_contacts(
    positions: np.ndarray,
    obstacles: List[Obstacle],
    radius: float,
    active_mask: np.ndarray | None,
) -> int:
    return _count_obstacle_contacts(_active_positions(positions, active_mask), obstacles, radius)


def _reassign_terminal_goal_slots(
    spec: RunSpec,
    traits,
    positions: np.ndarray,
    base_goals: np.ndarray,
    completed_mask: np.ndarray,
) -> np.ndarray:
    if not _uses_density_controls(spec, traits):
        return base_goals.copy()

    reassigned = base_goals.copy()
    active_idx = np.where(~np.asarray(completed_mask, dtype=bool))[0]
    if active_idx.size <= 1:
        return reassigned

    # Avoid adding a large O(N^2) assignment pass before the terminal queue forms.
    if active_idx.size > 700:
        return reassigned

    active_positions = positions[active_idx]
    free_slots = base_goals[active_idx]
    distances = np.linalg.norm(active_positions[:, None, :] - free_slots[None, :, :], axis=2)
    current_distances = np.diag(distances)
    agent_order = np.argsort(-current_distances)
    remaining_slots = set(range(active_idx.size))

    for local_agent in agent_order:
        if not remaining_slots:
            break
        remaining = np.fromiter(remaining_slots, dtype=int)
        slot = int(remaining[np.argmin(distances[local_agent, remaining])])
        reassigned[active_idx[local_agent]] = free_slots[slot]
        remaining_slots.remove(slot)

    return reassigned


def _local_safety_gain(spec: RunSpec) -> float:
    if spec.n_agents >= 700:
        return 1.45
    if spec.n_agents >= 300:
        return 1.20
    if spec.comm.packet_loss > 0.0 or spec.comm.latency_ms > 0.0:
        return 1.15
    return 1.0


def _staged_control_goals(
    spec: RunSpec,
    traits,
    t: float,
    positions: np.ndarray,
    start_positions: np.ndarray,
    final_goals: np.ndarray,
    arena_min: np.ndarray,
    arena_max: np.ndarray,
) -> np.ndarray:
    if not _uses_density_controls(spec, traits):
        return final_goals

    travel = final_goals - start_positions
    control_goals = final_goals.copy()

    if spec.scenario == "crossing_traffic":
        horizontal_travel = travel[:, :2]
        y_flow = np.abs(horizontal_travel[:, 1]) > np.abs(horizontal_travel[:, 0])
        x_flow = ~y_flow
        if np.any(y_flow) and np.any(x_flow):
            x_flow_clear = float(np.mean(positions[x_flow, 0] > 0.9)) >= 0.65 or t >= 7.5
            if not x_flow_clear:
                pre_conflict_stage = start_positions + 0.40 * travel
                pre_conflict_stage = np.clip(
                    pre_conflict_stage,
                    arena_min + [0.05, 0.05, 0.05],
                    arena_max - [0.05, 0.05, 0.05],
                )
                control_goals[y_flow] = pre_conflict_stage[y_flow]

    return control_goals


def _density_aware_target_speeds(
    spec: RunSpec,
    traits,
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_pos: List[np.ndarray],
    neighbor_vel: List[np.ndarray],
    obstacles: List[Obstacle],
    goals: np.ndarray | None = None,
    completed_mask: np.ndarray | None = None,
    time_s: float = 0.0,
) -> np.ndarray | None:
    if not _uses_speed_controls(spec, traits):
        return None

    speeds = np.full(spec.n_agents, spec.mav.max_speed_mps, dtype=float)
    completed = (
        np.asarray(completed_mask, dtype=bool)
        if completed_mask is not None
        else np.zeros(spec.n_agents, dtype=bool)
    )
    safe = spec.mav.safe_radius_m
    influence = spec.mav.safety_influence_m
    for i in range(spec.n_agents):
        if completed[i]:
            speeds[i] = 0.0
            continue

        factor = 1.0
        min_factor = 0.50
        nearest = float("inf")
        dense_count = 0
        max_closing = 0.0
        obstacle_risky = False
        pjs = neighbor_pos[i]
        if pjs.size > 0:
            rel = positions[i][None, :] - pjs
            dist = np.linalg.norm(rel, axis=1) + EPS
            nearest = float(np.min(dist))
            dense_count = int(np.sum(dist < influence))
            dirs = rel / dist[:, None]
            relv = velocities[i][None, :] - neighbor_vel[i]
            closing_speed = -np.sum(relv * dirs, axis=1)
            max_closing = float(np.max(np.maximum(closing_speed, 0.0)))

            if nearest < safe + 0.03:
                factor = min(factor, 0.35)
            elif nearest < safe + 0.12:
                factor = min(factor, 0.60)
            elif nearest < influence and dense_count >= 8:
                factor = min(factor, 0.82)
            if dense_count >= 16:
                factor = min(factor, 0.72)
            if max_closing > 0.55 and nearest < safe + 0.20:
                factor = min(factor, 0.65)

        for obs in obstacles:
            center = np.asarray(obs.center, dtype=float)
            rel_xy = positions[i, :2] - center[:2]
            dist_xy = float(np.linalg.norm(rel_xy))
            signed_xy = dist_xy - obs.radius
            z_ok = obs.height_min <= positions[i, 2] <= obs.height_max
            if not z_ok:
                continue
            if dist_xy > EPS:
                axis_xy = rel_xy / dist_xy
                radial_speed = float(np.dot(velocities[i, :2], axis_xy))
            else:
                radial_speed = -spec.mav.max_speed_mps
            clearance_margin = spec.mav.obstacle_clearance_margin_m
            predicted_signed_xy = signed_xy + min(radial_speed, 0.0) * spec.sim.dt_s
            if predicted_signed_xy <= spec.mav.collision_radius_m + 0.08 + clearance_margin:
                factor = min(factor, 0.40)
                min_factor = min(min_factor, 0.40)
                obstacle_risky = True
            elif signed_xy < 0.40 + clearance_margin and radial_speed < -0.05:
                factor = min(factor, 0.50)
                min_factor = min(min_factor, 0.45)
                obstacle_risky = True
            elif signed_xy < 0.55 + clearance_margin:
                factor = min(factor, 0.76)

        if goals is not None:
            dist_to_goal = float(np.linalg.norm(goals[i] - positions[i]))
            safe_progress_agent = (
                nearest >= safe + 0.22
                and max_closing <= 0.35
                and not obstacle_risky
            )
            if safe_progress_agent:
                remaining_time = max(float(spec.sim.horizon_s) - float(time_s), spec.sim.dt_s)
                required_speed = dist_to_goal / remaining_time
                progress_factor = min(1.0, required_speed / max(spec.mav.max_speed_mps, EPS) + 0.05)
                factor = max(factor, progress_factor)
            safe_tail_agent = (
                dist_to_goal <= 1.8
                and safe_progress_agent
            )
            if safe_tail_agent:
                factor = max(factor, 0.88)
                min_factor = max(min_factor, 0.80)

        speeds[i] = max(min_factor * spec.mav.max_speed_mps, factor * spec.mav.max_speed_mps)
    return speeds


def _apply_pairwise_brake_v3(
    spec: RunSpec,
    traits,
    positions: np.ndarray,
    velocities: np.ndarray,
    start_positions: np.ndarray,
    final_goals: np.ndarray,
    target_speeds: np.ndarray | None,
    active_mask: np.ndarray | None,
) -> np.ndarray | None:
    if not _uses_pairwise_brake_v3(spec, traits):
        return target_speeds

    speeds = (
        np.full(spec.n_agents, spec.mav.max_speed_mps, dtype=float)
        if target_speeds is None
        else np.asarray(target_speeds, dtype=float).copy()
    )
    active = (
        np.ones(spec.n_agents, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    active_idx = np.where(active)[0]
    if active_idx.size < 2:
        return speeds
    if spec.scenario != "crossing_traffic":
        return speeds

    radius = _metadata_float(spec, "pairwise_brake_v3_radius_m", 1.20)
    clearance = _metadata_float(spec, "pairwise_brake_v3_clearance_m", 0.42)
    min_factor = np.clip(_metadata_float(spec, "pairwise_brake_v3_min_factor", 0.28), 0.05, 1.0)
    priority_min_factor = np.clip(
        _metadata_float(spec, "pairwise_brake_v3_priority_min_factor", 0.82),
        min_factor,
        1.0,
    )
    closing_trigger = _metadata_float(spec, "pairwise_brake_v3_closing_trigger_mps", 0.03)
    if radius <= clearance + EPS:
        radius = clearance + 0.05

    active_positions = positions[active_idx]
    if cKDTree is not None:
        local_pairs = cKDTree(active_positions).query_pairs(radius)
        pairs = [(int(active_idx[i]), int(active_idx[j])) for i, j in local_pairs]
    else:
        pairs = []
        for local_i in range(active_idx.size):
            d = np.linalg.norm(active_positions[local_i + 1 :] - active_positions[local_i], axis=1)
            for offset in np.where(d < radius)[0]:
                pairs.append((int(active_idx[local_i]), int(active_idx[local_i + 1 + offset])))
    if not pairs:
        return speeds

    travel = final_goals - start_positions
    y_flow = np.abs(travel[:, 1]) > np.abs(travel[:, 0])
    crossing_priority = spec.scenario == "crossing_traffic"

    pair_array = np.asarray(pairs, dtype=int)
    idx_i = pair_array[:, 0]
    idx_j = pair_array[:, 1]
    rel = positions[idx_i] - positions[idx_j]
    dist = np.linalg.norm(rel, axis=1)
    axis = np.zeros_like(rel)
    nonzero = dist >= EPS
    axis[nonzero] = rel[nonzero] / dist[nonzero, None]
    if np.any(~nonzero):
        zero_axes = (idx_i[~nonzero] + idx_j[~nonzero]) % 3
        axis[np.where(~nonzero)[0], zero_axes] = 1.0
        dist = dist.copy()
        dist[~nonzero] = EPS

    radial_rate = np.einsum("ij,ij->i", velocities[idx_i] - velocities[idx_j], axis)
    active_pair = (dist < radius) & ((radial_rate < -closing_trigger) | (dist < clearance))
    if not np.any(active_pair):
        return speeds

    idx_i = idx_i[active_pair]
    idx_j = idx_j[active_pair]
    dist = dist[active_pair]
    radial_rate = radial_rate[active_pair]

    severity = np.clip((radius - dist) / max(radius - clearance, 0.05), 0.0, 1.0)
    closing = radial_rate < 0.0
    if np.any(closing):
        closing_boost = 0.15 * np.minimum(
            -radial_rate[closing] / max(spec.mav.max_speed_mps, EPS),
            1.0,
        )
        severity[closing] = np.minimum(1.0, severity[closing] + closing_boost)

    low_factor = 1.0 - severity * (1.0 - min_factor)

    if not crossing_priority:
        return speeds

    mixed_flow = y_flow[idx_i] != y_flow[idx_j]
    if not np.any(mixed_flow):
        return speeds

    mixed_i = idx_i[mixed_flow]
    mixed_j = idx_j[mixed_flow]
    mixed_low_factor = low_factor[mixed_flow]
    high_factor = 1.0 - severity[mixed_flow] * (1.0 - priority_min_factor)

    low_idx = np.where(y_flow[mixed_i], mixed_i, mixed_j)
    high_idx = np.where(y_flow[mixed_i], mixed_j, mixed_i)
    np.minimum.at(speeds, low_idx, spec.mav.max_speed_mps * mixed_low_factor)
    np.minimum.at(speeds, high_idx, spec.mav.max_speed_mps * high_factor)

    return speeds


def _count_obstacle_contacts(positions: np.ndarray, obstacles: List[Obstacle], radius: float) -> int:
    if not obstacles:
        return 0
    count = 0
    for obs in obstacles:
        center = np.asarray(obs.center, dtype=float)
        dxy = np.linalg.norm(positions[:, :2] - center[:2], axis=1)
        z_ok = (positions[:, 2] >= obs.height_min) & (positions[:, 2] <= obs.height_max)
        count += int(np.sum((dxy <= (obs.radius + radius)) & z_ok))
    return count


def _min_obstacle_contact_boundary_margin(
    positions: np.ndarray,
    obstacles: List[Obstacle],
    radius: float,
    active_mask: np.ndarray | None = None,
) -> float | None:
    if not obstacles:
        return None
    active = (
        np.ones(positions.shape[0], dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    best: float | None = None
    for obs in obstacles:
        center = np.asarray(obs.center, dtype=float)
        dxy = np.linalg.norm(positions[:, :2] - center[:2], axis=1)
        z_ok = (positions[:, 2] >= obs.height_min) & (positions[:, 2] <= obs.height_max)
        considered = z_ok & active
        if not np.any(considered):
            continue
        margin = dxy[considered] - (float(obs.radius) + float(radius))
        candidate = float(np.min(margin))
        best = candidate if best is None else min(best, candidate)
    return best


def _obstacle_projection_clearance(spec: RunSpec, obstacle: Obstacle) -> float:
    return (
        float(obstacle.radius)
        + spec.mav.collision_radius_m
        + 0.02
        + spec.mav.obstacle_clearance_margin_m
    )


def _apply_command_obstacle_filter(
    spec: RunSpec,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    obstacles: List[Obstacle],
    active_mask: np.ndarray | None,
    obstacle_positions: np.ndarray | None = None,
) -> np.ndarray:
    if not obstacles:
        return accelerations

    filtered = accelerations.copy()
    dt = max(float(spec.sim.dt_s), EPS)
    max_accel = float(spec.mav.max_accel_mps2)
    lookahead_margin = _metadata_float(spec, "obstacle_command_filter_lookahead_margin_m", 0.35)
    buffer_m = _metadata_float(spec, "obstacle_command_filter_buffer_m", 0.05)
    active = (
        np.ones(positions.shape[0], dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )

    for i in np.where(active)[0]:
        obstacle_pos_i = positions[i] if obstacle_positions is None else obstacle_positions[i]
        for obs in obstacles:
            if not (obs.height_min <= obstacle_pos_i[2] <= obs.height_max):
                continue

            center = np.asarray(obs.center, dtype=float)
            rel_xy = obstacle_pos_i[:2] - center[:2]
            dist_xy = float(np.linalg.norm(rel_xy))
            if dist_xy < EPS:
                angle = 2.399963229728653 * (i + 1)
                axis_xy = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                dist_xy = EPS
            else:
                axis_xy = rel_xy / dist_xy

            clearance = _obstacle_projection_clearance(spec, obs)
            margin = dist_xy - clearance
            if margin > lookahead_margin:
                continue

            radial_speed = float(np.dot(velocities[i, :2], axis_xy))
            radial_accel = float(np.dot(filtered[i, :2], axis_xy))
            radial_speed_next = radial_speed + radial_accel * dt
            usable_margin = max(margin - buffer_m, 0.0)
            allowed_inward_speed = float(np.sqrt(max(0.0, 2.0 * max_accel * usable_margin)))
            required_next_radial_speed = -allowed_inward_speed
            if margin <= buffer_m:
                required_next_radial_speed = max(required_next_radial_speed, 0.0)

            if radial_speed_next >= required_next_radial_speed:
                continue

            required_radial_accel = (required_next_radial_speed - radial_speed) / dt
            if radial_accel < required_radial_accel:
                filtered[i, :2] += (required_radial_accel - radial_accel) * axis_xy

    return limit_norm(filtered, spec.mav.max_accel_mps2)


def _apply_bounds(positions: np.ndarray, velocities: np.ndarray, arena_min: np.ndarray, arena_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    below = positions < arena_min
    above = positions > arena_max
    hit = below | above
    positions = np.clip(positions, arena_min, arena_max)
    velocities = velocities.copy()
    velocities[hit] = 0.0
    return positions, velocities


def _overlap_pairs(
    positions: np.ndarray,
    min_distance: float,
    active_mask: np.ndarray | None = None,
) -> List[Tuple[int, int]]:
    if active_mask is not None:
        active_idx = np.where(np.asarray(active_mask, dtype=bool))[0]
        if active_idx.size < 2:
            return []
        active_positions = positions[active_idx]
    else:
        active_idx = None
        active_positions = positions

    if cKDTree is not None:
        pairs = cKDTree(active_positions).query_pairs(min_distance)
        if active_idx is None:
            return [(int(i), int(j)) for i, j in pairs]
        return [(int(active_idx[i]), int(active_idx[j])) for i, j in pairs]

    pairs: List[Tuple[int, int]] = []
    n = active_positions.shape[0]
    for i in range(n):
        d = np.linalg.norm(active_positions[i + 1 :] - active_positions[i], axis=1)
        for offset in np.where(d < min_distance)[0]:
            j = int(i + 1 + offset)
            if active_idx is None:
                pairs.append((i, j))
            else:
                pairs.append((int(active_idx[i]), int(active_idx[j])))
    return pairs


def _percentile_or_zero(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _record_pre_projection_state(
    positions: np.ndarray,
    obstacles: List[Obstacle],
    spec: RunSpec,
    telemetry: ProjectionTelemetry,
    active_mask: np.ndarray | None,
    pairwise_projection_enabled: bool,
    obstacle_projection_enabled: bool,
    pairwise_telemetry_enabled: bool | None = None,
    obstacle_telemetry_enabled: bool | None = None,
) -> None:
    pairwise_envelope_enabled = (
        pairwise_projection_enabled
        if pairwise_telemetry_enabled is None
        else bool(pairwise_telemetry_enabled)
    )
    obstacle_envelope_enabled = (
        obstacle_projection_enabled
        if obstacle_telemetry_enabled is None
        else bool(obstacle_telemetry_enabled)
    )

    telemetry.pre_projection_state_checks += 1
    telemetry.pre_projection_collision_pairs += _count_active_pairs_within(
        positions,
        spec.mav.collision_radius_m,
        active_mask,
    )
    telemetry.pre_projection_obstacle_contacts += _count_active_obstacle_contacts(
        positions,
        obstacles,
        spec.mav.collision_radius_m,
        active_mask,
    )

    if pairwise_envelope_enabled:
        pairwise_clearance = spec.mav.collision_radius_m + 0.02
        pairs = _overlap_pairs(positions, pairwise_clearance, active_mask=active_mask)
        if pairs:
            penetrations = np.asarray([
                pairwise_clearance - float(np.linalg.norm(positions[i] - positions[j]))
                for i, j in pairs
            ], dtype=float)
            telemetry.record_pre_projection_pairwise(pairs, penetrations)

    if obstacle_envelope_enabled and obstacles:
        active = (
            np.ones(positions.shape[0], dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        for obs in obstacles:
            center = np.asarray(obs.center, dtype=float)
            clearance = _obstacle_projection_clearance(spec, obs)
            rel_xy = positions[:, :2] - center[:2]
            dist_xy = np.linalg.norm(rel_xy, axis=1)
            z_ok = (positions[:, 2] >= obs.height_min) & (positions[:, 2] <= obs.height_max)
            considered = z_ok & active
            if not np.any(considered):
                continue
            clearance_margins = dist_xy[considered] - clearance
            penetrations = clearance - dist_xy[considered]
            telemetry.record_pre_projection_obstacle(penetrations, clearance_margins)


def _resolve_pairwise_overlaps(
    positions: np.ndarray,
    velocities: np.ndarray,
    min_distance: float,
    arena_min: np.ndarray,
    arena_max: np.ndarray,
    max_iters: int = 4,
    telemetry: ProjectionTelemetry | None = None,
    active_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    repaired_positions = positions.copy()
    repaired_velocities = velocities.copy()
    min_distance = float(min_distance)
    call_had_adjustment = False

    for _ in range(max_iters):
        pairs = _overlap_pairs(repaired_positions, min_distance, active_mask=active_mask)
        if not pairs:
            break

        position_delta = np.zeros_like(repaired_positions)
        velocity_delta = np.zeros_like(repaired_velocities)
        pair_adjustments = 0
        velocity_corrections = 0
        for i, j in pairs:
            rel = repaired_positions[i] - repaired_positions[j]
            dist = float(np.linalg.norm(rel))
            if dist < EPS:
                axis = np.zeros(3, dtype=float)
                axis[(i + j) % 3] = 1.0
                dist = EPS
            else:
                axis = rel / dist

            overlap = min_distance - dist
            if overlap <= 0.0:
                continue

            pair_adjustments += 1
            push = 0.5 * (overlap + 1e-6) * axis
            position_delta[i] += push
            position_delta[j] -= push

            rel_v = repaired_velocities[i] - repaired_velocities[j]
            separation_rate = float(np.dot(rel_v, axis))
            if separation_rate < 0.0:
                velocity_corrections += 1
                impulse = -0.5 * separation_rate * axis
                velocity_delta[i] += impulse
                velocity_delta[j] -= impulse

        if telemetry is not None:
            telemetry.record_pairwise_iteration(position_delta, pair_adjustments, velocity_corrections)
        call_had_adjustment = call_had_adjustment or pair_adjustments > 0
        repaired_positions = repaired_positions + position_delta
        repaired_positions = np.clip(repaired_positions, arena_min, arena_max)
        repaired_velocities = repaired_velocities + velocity_delta

    if telemetry is not None and call_had_adjustment:
        telemetry.pairwise_projection_calls += 1

    return repaired_positions, repaired_velocities


def _resolve_obstacle_overlaps(
    positions: np.ndarray,
    velocities: np.ndarray,
    obstacles: List[Obstacle],
    clearance: float,
    arena_min: np.ndarray,
    arena_max: np.ndarray,
    telemetry: ProjectionTelemetry | None = None,
    active_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not obstacles:
        return positions, velocities

    repaired_positions = positions.copy()
    repaired_velocities = velocities.copy()
    clearance = float(clearance)
    call_had_adjustment = False
    active = (
        np.ones(repaired_positions.shape[0], dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )

    for obs in obstacles:
        center = np.asarray(obs.center, dtype=float)
        rel_xy = repaired_positions[:, :2] - center[:2]
        dist_xy = np.linalg.norm(rel_xy, axis=1)
        z_ok = (repaired_positions[:, 2] >= obs.height_min) & (repaired_positions[:, 2] <= obs.height_max)
        mask = (dist_xy < clearance) & z_ok & active
        if not np.any(mask):
            continue

        idxs = np.where(mask)[0]
        for idx in idxs:
            dist = float(dist_xy[idx])
            if dist < EPS:
                angle = 2.399963229728653 * (idx + 1)
                axis_xy = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                dist = EPS
            else:
                axis_xy = rel_xy[idx] / dist

            displacement = clearance - dist + 1e-6
            repaired_positions[idx, :2] += displacement * axis_xy
            radial_speed = float(np.dot(repaired_velocities[idx, :2], axis_xy))
            velocity_corrected = radial_speed < 0.0
            if radial_speed < 0.0:
                repaired_velocities[idx, :2] -= radial_speed * axis_xy
            if telemetry is not None:
                telemetry.record_obstacle_adjustment(displacement, velocity_corrected)
            call_had_adjustment = True

    repaired_positions = np.clip(repaired_positions, arena_min, arena_max)
    if telemetry is not None and call_had_adjustment:
        telemetry.obstacle_projection_calls += 1
    return repaired_positions, repaired_velocities


def run_simulation(spec: RunSpec, out_dir: str | Path | None = None) -> Dict[str, object]:
    rng = np.random.default_rng(spec.seed)
    scenario = make_scenario(
        spec.scenario,
        spec.n_agents,
        spec.seed,
        spec.sim.arena_min,
        spec.sim.arena_max,
        initial_spacing_scale=spec.mav.initial_spacing_scale,
        terminal_slot_min_spacing_m=spec.mav.terminal_slot_min_spacing_m,
    )
    positions = scenario.positions.copy()
    start_positions = positions.copy()
    velocities = scenario.velocities.copy()
    goals = scenario.goals.copy()
    obstacles = scenario.obstacles
    arena_min = scenario.arena_min
    arena_max = scenario.arena_max

    traits = get_controller_traits(spec.controller)
    terminal_traffic_enabled = (
        _uses_proposed_local_safety(spec, traits)
        and spec.sim.completed_agents_leave_traffic
    )
    steps = int(round(spec.sim.horizon_s / spec.sim.dt_s))
    comm_interval_steps = max(1, int(round(1.0 / max(spec.comm.update_rate_hz, EPS) / spec.sim.dt_s)))
    local_safety_interval_steps = _local_safety_update_interval_steps(spec)
    obstacle_interval_steps = _obstacle_update_interval_steps(spec)
    n = spec.n_agents
    possible_pairs = max(1, n * (n - 1) // 2)
    completed_mask = np.zeros(n, dtype=bool)
    assigned_goals = goals.copy()
    active_mask = ~completed_mask if terminal_traffic_enabled else None

    history_pos = [positions.copy()]
    history_vel = [velocities.copy()]
    neighbor_pos, neighbor_vel, messages, obs_edges = _make_observations(
        positions,
        velocities,
        history_pos,
        history_vel,
        spec,
        traits.use_prediction,
        rng,
        active_mask=active_mask,
    )
    cached_safety_neighbor_pos = None
    cached_safety_neighbor_vel = None
    cached_obstacle_positions = positions.copy()

    timeseries_records = []
    trajectory_records = []
    cumulative_collision_pairs = 0
    cumulative_obstacle_contacts = 0
    cumulative_safety_pairs = 0
    cumulative_energy = 0.0
    cumulative_messages = 0
    projection_telemetry = ProjectionTelemetry()
    min_distance_global = float("inf")
    min_obstacle_contact_boundary_margin: float | None = None
    completion_time = None
    start_wall = time.perf_counter()

    for step in range(steps + 1):
        t = step * spec.sim.dt_s
        active_mask = ~completed_mask if terminal_traffic_enabled else None
        if step % comm_interval_steps == 0:
            neighbor_pos, neighbor_vel, messages, obs_edges = _make_observations(
                positions,
                velocities,
                history_pos,
                history_vel,
                spec,
                traits.use_prediction,
                rng,
                active_mask=active_mask,
            )
            cumulative_messages += messages
        if step % obstacle_interval_steps == 0:
            cached_obstacle_positions = positions.copy()

        dmin = _active_pairwise_min_distance(positions, active_mask)
        min_distance_global = min(min_distance_global, dmin)
        collision_pairs = _count_active_pairs_within(positions, spec.mav.collision_radius_m, active_mask)
        safety_pairs = _count_active_pairs_within(positions, spec.mav.safe_radius_m, active_mask)
        obstacle_contacts = _count_active_obstacle_contacts(
            positions,
            obstacles,
            spec.mav.collision_radius_m,
            active_mask,
        )
        obstacle_contact_boundary_margin = _min_obstacle_contact_boundary_margin(
            positions,
            obstacles,
            spec.mav.collision_radius_m,
            active_mask,
        )
        if obstacle_contact_boundary_margin is not None:
            min_obstacle_contact_boundary_margin = (
                obstacle_contact_boundary_margin
                if min_obstacle_contact_boundary_margin is None
                else min(min_obstacle_contact_boundary_margin, obstacle_contact_boundary_margin)
            )
        cumulative_collision_pairs += collision_pairs
        cumulative_safety_pairs += safety_pairs
        cumulative_obstacle_contacts += obstacle_contacts
        if terminal_traffic_enabled:
            reported_completed = _update_completed_mask(
                completed_mask,
                positions,
                assigned_goals,
                spec.sim.goal_tolerance_m,
            )
            goal_distances = np.linalg.norm(positions - assigned_goals, axis=1)
            goal_distances[reported_completed] = 0.0
            reached_fraction = float(np.mean(reported_completed))
            mean_goal_distance = float(np.mean(goal_distances))
        else:
            reported_completed = completed_mask
            reached_fraction, mean_goal_distance = mission_progress(positions, goals, spec.sim.goal_tolerance_m)
        if completion_time is None and reached_fraction >= spec.sim.completion_fraction:
            completion_time = t

        if step % spec.sim.log_stride == 0 or step == steps:
            conn = connectivity_metrics(positions, spec.comm.comm_radius_m)
            timeseries_records.append({
                "t_s": t,
                "step": step,
                "min_inter_agent_distance_m": dmin,
                "collision_pairs": collision_pairs,
                "safety_violation_pairs": safety_pairs,
                "obstacle_contacts": obstacle_contacts,
                "obstacle_contact_boundary_margin_m": obstacle_contact_boundary_margin,
                "reached_fraction": reached_fraction,
                "mean_goal_distance_m": mean_goal_distance,
                "order_parameter": order_parameter(velocities),
                "largest_component_ratio": conn["largest_component_ratio"],
                "n_components": conn["n_components"],
                "messages_last_update": messages,
            })

        if spec.sim.save_trajectory and (step % spec.sim.trajectory_stride == 0 or step == steps):
            # Long format keeps plotting simple and compresses well as parquet/csv.gz if desired.
            for i in range(n):
                trajectory_records.append({
                    "t_s": t,
                    "agent": i,
                    "x": positions[i, 0],
                    "y": positions[i, 1],
                    "z": positions[i, 2],
                    "vx": velocities[i, 0],
                    "vy": velocities[i, 1],
                    "vz": velocities[i, 2],
                    "goal_x": assigned_goals[i, 0],
                    "goal_y": assigned_goals[i, 1],
                    "goal_z": assigned_goals[i, 2],
                })

        if step == steps:
            completed_mask = reported_completed
            break

        if terminal_traffic_enabled:
            completed_mask = reported_completed
            velocities[completed_mask] = 0.0
            if np.any(completed_mask):
                assigned_goals = _reassign_terminal_goal_slots(
                    spec,
                    traits,
                    positions,
                    goals,
                    completed_mask,
                )
            active_mask = ~completed_mask

        control_goals = _staged_control_goals(
            spec,
            traits,
            t,
            positions,
            start_positions,
            assigned_goals,
            arena_min,
            arena_max,
        )
        if terminal_traffic_enabled and np.any(completed_mask):
            control_goals[completed_mask] = positions[completed_mask]
        safety_neighbor_pos = None
        safety_neighbor_vel = None
        if _uses_proposed_local_safety(spec, traits):
            if (
                cached_safety_neighbor_pos is None
                or cached_safety_neighbor_vel is None
                or step % local_safety_interval_steps == 0
            ):
                cached_safety_neighbor_pos, cached_safety_neighbor_vel = _make_local_safety_observations(
                    positions,
                    velocities,
                    spec,
                    rng,
                    history_pos=history_pos,
                    history_vel=history_vel,
                    active_mask=active_mask,
                )
            safety_neighbor_pos = cached_safety_neighbor_pos
            safety_neighbor_vel = cached_safety_neighbor_vel
        target_speeds = _density_aware_target_speeds(
            spec,
            traits,
            positions,
            velocities,
            safety_neighbor_pos if safety_neighbor_pos is not None else neighbor_pos,
            safety_neighbor_vel if safety_neighbor_vel is not None else neighbor_vel,
            obstacles,
            goals=control_goals,
            completed_mask=completed_mask if terminal_traffic_enabled else None,
            time_s=t,
        )
        target_speeds = _apply_pairwise_brake_v3(
            spec,
            traits,
            positions,
            velocities,
            start_positions,
            goals,
            target_speeds,
            active_mask,
        )
        acc = compute_acceleration(
            spec.controller,
            positions,
            velocities,
            control_goals,
            neighbor_pos,
            neighbor_vel,
            obstacles,
            arena_min,
            arena_max,
            spec.weights,
            spec.mav,
            target_speeds,
            shield_neighbor_pos=safety_neighbor_pos,
            shield_neighbor_vel=safety_neighbor_vel,
            safety_gain=_local_safety_gain(spec) if safety_neighbor_pos is not None else 1.0,
            dt_s=spec.sim.dt_s,
            apply_safety_shield=_uses_local_safety_layer(spec, traits),
            obstacle_positions=cached_obstacle_positions,
        )
        if _uses_obstacle_command_filter(spec, traits):
            acc = _apply_command_obstacle_filter(
                spec,
                positions,
                velocities,
                acc,
                obstacles,
                active_mask,
                obstacle_positions=cached_obstacle_positions,
            )
        if terminal_traffic_enabled and np.any(completed_mask):
            acc[completed_mask] = 0.0
        cumulative_energy += float(np.sum(np.linalg.norm(acc, axis=1) ** 2) * spec.sim.dt_s)
        velocities = velocities + acc * spec.sim.dt_s
        if terminal_traffic_enabled and np.any(completed_mask):
            velocities[completed_mask] = 0.0
        velocities = limit_norm(velocities, spec.mav.max_speed_mps)
        positions = positions + velocities * spec.sim.dt_s
        positions, velocities = _apply_bounds(positions, velocities, arena_min, arena_max)
        if terminal_traffic_enabled and np.any(completed_mask):
            velocities[completed_mask] = 0.0
        pairwise_projection_stack_enabled = _uses_pairwise_projection_stack(spec, traits)
        pairwise_projection_enabled = _uses_pairwise_projection(spec, traits)
        obstacle_projection_enabled = _uses_obstacle_projection(spec, traits)
        obstacle_filter_enabled = _uses_obstacle_command_filter(spec, traits)
        runtime_safety_active = (
            _uses_local_safety_layer(spec, traits)
            or pairwise_projection_stack_enabled
            or obstacle_projection_enabled
            or obstacle_filter_enabled
        )
        if runtime_safety_active:
            _record_pre_projection_state(
                positions,
                obstacles,
                spec,
                projection_telemetry,
                active_mask,
                pairwise_projection_enabled,
                obstacle_projection_enabled or obstacle_filter_enabled,
                pairwise_telemetry_enabled=pairwise_projection_stack_enabled,
                obstacle_telemetry_enabled=obstacle_projection_enabled or obstacle_filter_enabled,
            )
            for _ in range(2):
                if obstacle_projection_enabled:
                    for obs in obstacles:
                        positions, velocities = _resolve_obstacle_overlaps(
                            positions,
                            velocities,
                            [obs],
                            _obstacle_projection_clearance(spec, obs),
                            arena_min,
                            arena_max,
                            telemetry=projection_telemetry,
                            active_mask=active_mask,
                        )
                if pairwise_projection_enabled:
                    positions, velocities = _resolve_pairwise_overlaps(
                        positions,
                        velocities,
                        spec.mav.collision_radius_m + 0.02,
                        arena_min,
                        arena_max,
                        telemetry=projection_telemetry,
                        active_mask=active_mask,
                    )
            velocities = limit_norm(velocities, spec.mav.max_speed_mps)
            if terminal_traffic_enabled and np.any(completed_mask):
                velocities[completed_mask] = 0.0
        history_pos.append(positions.copy())
        history_vel.append(velocities.copy())

    runtime_s = time.perf_counter() - start_wall
    if terminal_traffic_enabled:
        completed_mask = _update_completed_mask(
            completed_mask,
            positions,
            assigned_goals,
            spec.sim.goal_tolerance_m,
        )
        final_distances = np.linalg.norm(positions - assigned_goals, axis=1)
        final_distances[completed_mask] = 0.0
        final_reached_fraction = float(np.mean(completed_mask))
        final_mean_goal_distance = float(np.mean(final_distances))
    else:
        final_reached_fraction, final_mean_goal_distance = mission_progress(positions, goals, spec.sim.goal_tolerance_m)
    ts = pd.DataFrame(timeseries_records)
    collision_rate_per_pair_step = cumulative_collision_pairs / max(1, (steps + 1) * possible_pairs)
    safety_violation_rate_per_pair_step = cumulative_safety_pairs / max(1, (steps + 1) * possible_pairs)
    obstacle_contact_rate_per_agent_step = cumulative_obstacle_contacts / max(1, (steps + 1) * n)
    projection_agent_steps = max(1, steps * n)
    projection_adjustments = (
        projection_telemetry.pairwise_projection_agent_adjustments
        + projection_telemetry.obstacle_projection_agent_adjustments
    )
    projection_total_displacement = (
        projection_telemetry.pairwise_projection_total_displacement_m
        + projection_telemetry.obstacle_projection_total_displacement_m
    )
    projection_max_displacement = max(
        projection_telemetry.pairwise_projection_max_displacement_m,
        projection_telemetry.obstacle_projection_max_displacement_m,
    )
    projection_mean_displacement = (
        projection_total_displacement / projection_adjustments
        if projection_adjustments > 0 else 0.0
    )
    projection_pair_steps = max(1, steps * possible_pairs)
    pre_pairwise_violation_pairs = projection_telemetry.pre_projection_pairwise_violation_pairs
    pre_pairwise_violation_agent_steps = projection_telemetry.pre_projection_pairwise_violation_agent_steps
    pre_obstacle_violation_agent_steps = projection_telemetry.pre_projection_obstacle_violation_agent_steps
    pre_pairwise_mean_penetration = (
        projection_telemetry.pre_projection_pairwise_total_penetration_m / pre_pairwise_violation_pairs
        if pre_pairwise_violation_pairs > 0 else 0.0
    )
    pre_obstacle_mean_penetration = (
        projection_telemetry.pre_projection_obstacle_total_penetration_m / pre_obstacle_violation_agent_steps
        if pre_obstacle_violation_agent_steps > 0 else 0.0
    )

    metrics = {
        "run_id": make_run_id(spec),
        "scenario": spec.scenario,
        "controller": spec.controller,
        "n_agents": n,
        "seed": spec.seed,
        "packet_loss": spec.comm.packet_loss,
        "latency_ms": spec.comm.latency_ms,
        "comm_radius_m": spec.comm.comm_radius_m,
        "k_neighbors": spec.comm.k_neighbors,
        "update_rate_hz": spec.comm.update_rate_hz,
        "obstacle_update_rate_hz": spec.comm.obstacle_update_rate_hz,
        "local_safety_dropout": spec.comm.local_safety_dropout,
        "local_safety_latency_ms": spec.comm.local_safety_latency_ms,
        "local_safety_update_rate_hz": spec.comm.local_safety_update_rate_hz,
        "local_safety_position_noise_std_m": spec.comm.local_safety_position_noise_std_m,
        "local_safety_velocity_noise_std_mps": spec.comm.local_safety_velocity_noise_std_mps,
        "horizon_s": spec.sim.horizon_s,
        "dt_s": spec.sim.dt_s,
        "min_inter_agent_distance_m": float(min_distance_global),
        "min_obstacle_contact_boundary_margin_m": (
            None
            if min_obstacle_contact_boundary_margin is None
            else float(min_obstacle_contact_boundary_margin)
        ),
        "cumulative_collision_pairs": int(cumulative_collision_pairs),
        "collision_rate_per_pair_step": float(collision_rate_per_pair_step),
        "cumulative_safety_violation_pairs": int(cumulative_safety_pairs),
        "safety_violation_rate_per_pair_step": float(safety_violation_rate_per_pair_step),
        "cumulative_obstacle_contacts": int(cumulative_obstacle_contacts),
        "obstacle_contact_rate_per_agent_step": float(obstacle_contact_rate_per_agent_step),
        "final_reached_fraction": float(final_reached_fraction),
        "mission_success": bool(final_reached_fraction >= spec.sim.completion_fraction and cumulative_collision_pairs == 0),
        "completion_time_s": float(completion_time) if completion_time is not None else None,
        "final_mean_goal_distance_m": float(final_mean_goal_distance),
        "energy_proxy_accel2_s_per_agent": float(cumulative_energy / n),
        "mean_messages_per_agent_per_update": float(cumulative_messages / max(1, n * ((steps // comm_interval_steps) + 1))),
        "runtime_s": float(runtime_s),
        "runtime_ms_per_agent_step": float(1000.0 * runtime_s / max(1, n * (steps + 1))),
        "scenario_description": scenario.description,
        "pairwise_projection_calls": int(projection_telemetry.pairwise_projection_calls),
        "pairwise_projection_pair_adjustments": int(projection_telemetry.pairwise_projection_pair_adjustments),
        "pairwise_projection_agent_adjustments": int(projection_telemetry.pairwise_projection_agent_adjustments),
        "pairwise_projection_total_displacement_m": float(projection_telemetry.pairwise_projection_total_displacement_m),
        "pairwise_projection_max_displacement_m": float(projection_telemetry.pairwise_projection_max_displacement_m),
        "pairwise_projection_velocity_corrections": int(projection_telemetry.pairwise_projection_velocity_corrections),
        "pairwise_projection_agent_adjustment_rate_per_agent_step": float(
            projection_telemetry.pairwise_projection_agent_adjustments / projection_agent_steps
        ),
        "obstacle_projection_calls": int(projection_telemetry.obstacle_projection_calls),
        "obstacle_projection_agent_adjustments": int(projection_telemetry.obstacle_projection_agent_adjustments),
        "obstacle_projection_total_displacement_m": float(projection_telemetry.obstacle_projection_total_displacement_m),
        "obstacle_projection_max_displacement_m": float(projection_telemetry.obstacle_projection_max_displacement_m),
        "obstacle_projection_velocity_corrections": int(projection_telemetry.obstacle_projection_velocity_corrections),
        "obstacle_projection_agent_adjustment_rate_per_agent_step": float(
            projection_telemetry.obstacle_projection_agent_adjustments / projection_agent_steps
        ),
        "projection_agent_adjustments": int(projection_adjustments),
        "projection_agent_adjustment_rate_per_agent_step": float(projection_adjustments / projection_agent_steps),
        "projection_total_displacement_m": float(projection_total_displacement),
        "projection_mean_displacement_m_per_adjustment": float(projection_mean_displacement),
        "projection_max_displacement_m": float(projection_max_displacement),
        "pre_projection_state_checks": int(projection_telemetry.pre_projection_state_checks),
        "pre_projection_collision_pairs": int(projection_telemetry.pre_projection_collision_pairs),
        "pre_projection_collision_rate_per_pair_step": float(
            projection_telemetry.pre_projection_collision_pairs / projection_pair_steps
        ),
        "pre_projection_obstacle_contacts": int(projection_telemetry.pre_projection_obstacle_contacts),
        "pre_projection_obstacle_contact_rate_per_agent_step": float(
            projection_telemetry.pre_projection_obstacle_contacts / projection_agent_steps
        ),
        "pre_projection_pairwise_violation_pairs": int(pre_pairwise_violation_pairs),
        "pre_projection_pairwise_violation_rate_per_pair_step": float(
            pre_pairwise_violation_pairs / projection_pair_steps
        ),
        "pre_projection_pairwise_violation_agent_steps": int(pre_pairwise_violation_agent_steps),
        "pre_projection_pairwise_violation_agent_rate_per_agent_step": float(
            pre_pairwise_violation_agent_steps / projection_agent_steps
        ),
        "pre_projection_pairwise_mean_penetration_m": float(pre_pairwise_mean_penetration),
        "pre_projection_pairwise_p95_penetration_m": float(
            _percentile_or_zero(projection_telemetry.pre_projection_pairwise_penetrations_m, 95.0)
        ),
        "pre_projection_pairwise_p99_penetration_m": float(
            _percentile_or_zero(projection_telemetry.pre_projection_pairwise_penetrations_m, 99.0)
        ),
        "pre_projection_pairwise_max_penetration_m": float(
            projection_telemetry.pre_projection_pairwise_max_penetration_m
        ),
        "pre_projection_obstacle_violation_agent_steps": int(pre_obstacle_violation_agent_steps),
        "pre_projection_obstacle_violation_rate_per_agent_step": float(
            pre_obstacle_violation_agent_steps / projection_agent_steps
        ),
        "pre_projection_obstacle_mean_penetration_m": float(pre_obstacle_mean_penetration),
        "pre_projection_obstacle_p95_penetration_m": float(
            _percentile_or_zero(projection_telemetry.pre_projection_obstacle_penetrations_m, 95.0)
        ),
        "pre_projection_obstacle_p99_penetration_m": float(
            _percentile_or_zero(projection_telemetry.pre_projection_obstacle_penetrations_m, 99.0)
        ),
        "pre_projection_obstacle_max_penetration_m": float(
            projection_telemetry.pre_projection_obstacle_max_penetration_m
        ),
        "pre_projection_obstacle_min_clearance_margin_m": (
            None
            if projection_telemetry.pre_projection_obstacle_min_clearance_margin_m is None
            else float(projection_telemetry.pre_projection_obstacle_min_clearance_margin_m)
        ),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        run_dir = out_dir / metrics["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        ts.to_csv(run_dir / "timeseries.csv", index=False)
        save_yaml(spec.to_dict(), run_dir / "config.yaml")
        if spec.sim.save_trajectory:
            pd.DataFrame(trajectory_records).to_csv(run_dir / "trajectory.csv", index=False)
        if obstacles:
            pd.DataFrame([
                {"cx": o.center[0], "cy": o.center[1], "cz": o.center[2], "radius": o.radius, "height_min": o.height_min, "height_max": o.height_max}
                for o in obstacles
            ]).to_csv(run_dir / "obstacles.csv", index=False)

    return {"metrics": metrics, "timeseries": ts, "trajectory": pd.DataFrame(trajectory_records) if trajectory_records else None}
