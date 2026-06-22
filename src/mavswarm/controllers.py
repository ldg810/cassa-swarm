from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .config import ControllerWeights, MAVParams
from .scenarios import Obstacle

EPS = 1e-9


@dataclass(frozen=True)
class ControllerTraits:
    use_prediction: bool
    use_safety_shield: bool
    family: str


def get_controller_traits(name: str) -> ControllerTraits:
    if name == "boids":
        return ControllerTraits(use_prediction=False, use_safety_shield=False, family="boids")
    if name == "apf":
        return ControllerTraits(use_prediction=False, use_safety_shield=False, family="apf")
    if name == "orca":
        return ControllerTraits(use_prediction=False, use_safety_shield=False, family="orca")
    if name in {"cbf_qp", "cbf_qp_osqp", "decentralized_mpc", "mader_like", "rvo2_external"}:
        return ControllerTraits(use_prediction=False, use_safety_shield=False, family=name)
    if name == "proposed_no_shield":
        return ControllerTraits(use_prediction=True, use_safety_shield=False, family="proposed")
    if name == "proposed":
        return ControllerTraits(use_prediction=True, use_safety_shield=True, family="proposed")
    raise ValueError(f"Unknown controller: {name}")


def limit_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + EPS))
    return vectors * scale


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + EPS)


def _goal_accel(pos_i: np.ndarray, vel_i: np.ndarray, goal_i: np.ndarray, target_speed: float) -> np.ndarray:
    to_goal = goal_i - pos_i
    dist = np.linalg.norm(to_goal)
    if dist < EPS:
        desired_v = np.zeros(3)
    else:
        # Slow down near goal to reduce oscillation.
        speed = min(target_speed, 0.7 * dist)
        desired_v = speed * to_goal / dist
    return desired_v - vel_i


def _obstacle_accel(pos_i: np.ndarray, obstacles: Sequence[Obstacle], influence: float = 1.2) -> np.ndarray:
    acc = np.zeros(3, dtype=float)
    for obs in obstacles:
        center = np.asarray(obs.center, dtype=float)
        # Cylinder-like horizontal avoidance with mild vertical component.
        dxy = pos_i[:2] - center[:2]
        dist_xy = np.linalg.norm(dxy) + EPS
        signed = dist_xy - obs.radius
        if signed < influence:
            strength = (influence - signed) / max(influence, EPS)
            dir_xy = dxy / dist_xy
            acc[:2] += strength**2 * dir_xy / max(signed + 0.08, 0.08)
            if obs.height_min <= pos_i[2] <= obs.height_max:
                z_dir = 1.0 if pos_i[2] >= center[2] else -1.0
                acc[2] += 0.2 * strength * z_dir
    return acc


def _boundary_accel(pos_i: np.ndarray, vel_i: np.ndarray, arena_min: np.ndarray, arena_max: np.ndarray, margin: float = 0.55) -> np.ndarray:
    acc = np.zeros(3, dtype=float)
    for dim in range(3):
        lower_d = pos_i[dim] - arena_min[dim]
        upper_d = arena_max[dim] - pos_i[dim]
        if lower_d < margin:
            acc[dim] += (margin - lower_d) / margin
        if upper_d < margin:
            acc[dim] -= (margin - upper_d) / margin
    return acc - 0.10 * vel_i


def _neighbor_terms(
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    nbr_pos: np.ndarray,
    nbr_vel: np.ndarray,
    mav: MAVParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nbr_pos.size == 0:
        return np.zeros(3), np.zeros(3), np.zeros(3)

    rel = pos_i[None, :] - nbr_pos
    dist = np.linalg.norm(rel, axis=1) + EPS
    dirs = rel / dist[:, None]

    # Separation increases steeply below safety influence.
    sep_mask = dist < mav.safety_influence_m
    if np.any(sep_mask):
        depth = np.maximum(mav.safety_influence_m - dist[sep_mask], 0.0)
        sep = np.sum((depth[:, None] / (dist[sep_mask, None] + 0.05)) * dirs[sep_mask], axis=0)
    else:
        sep = np.zeros(3)

    align = np.mean(nbr_vel, axis=0) - vel_i
    cohesion = np.mean(nbr_pos, axis=0) - pos_i
    return sep, align, cohesion


def _time_to_collision_s(rel_pos: np.ndarray, rel_vel: np.ndarray, radius: float) -> float:
    a = float(np.dot(rel_vel, rel_vel))
    b = float(2.0 * np.dot(rel_pos, rel_vel))
    c = float(np.dot(rel_pos, rel_pos) - radius**2)
    if c <= 0.0:
        return 0.0
    if a < EPS or b >= 0.0:
        return float("inf")
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return float("inf")
    sqrt_disc = float(np.sqrt(disc))
    candidates = [
        (-b - sqrt_disc) / (2.0 * a),
        (-b + sqrt_disc) / (2.0 * a),
    ]
    positive = [t for t in candidates if t >= 0.0]
    if not positive:
        return float("inf")
    return float(min(positive))


def _orca_pairwise_accel(
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    goal_i: np.ndarray,
    nbr_pos: np.ndarray,
    nbr_vel: np.ndarray,
    mav: MAVParams,
    tau_s: float = 3.0,
) -> np.ndarray:
    if nbr_pos.size == 0:
        return np.zeros(3, dtype=float)

    acc = np.zeros(3, dtype=float)
    safe = mav.safe_radius_m
    influence = max(mav.safety_influence_m + 0.7, 1.8 * safe)
    for pj, vj in zip(nbr_pos, nbr_vel):
        rel = pos_i - pj
        dist = float(np.linalg.norm(rel)) + EPS
        axis = rel / dist
        rel_vel = vel_i - vj
        ttc = _time_to_collision_s(rel, rel_vel, safe)
        if dist > influence and ttc > tau_s:
            continue

        closing_speed = max(0.0, -float(np.dot(rel_vel, axis)))
        distance_weight = max(0.0, (influence - dist) / max(influence - safe, 0.05))
        time_weight = max(0.0, (tau_s - min(ttc, tau_s)) / tau_s)
        overlap_weight = max(0.0, (safe + 0.05 - dist) / max(safe + 0.05, EPS))

        correction = (
            1.4 * distance_weight
            + 3.0 * time_weight
            + 5.0 * overlap_weight
            + 0.8 * closing_speed
        ) * axis

        lateral = np.array([-axis[1], axis[0], 0.0], dtype=float)
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm > EPS and time_weight > 0.0:
            lateral /= lateral_norm
            sign = 1.0 if float(np.dot(lateral, goal_i - pos_i)) >= 0.0 else -1.0
            correction += 0.25 * time_weight * sign * lateral

        acc += correction
    return acc


def _project_accel_to_halfspaces(
    nominal_acc: np.ndarray,
    a_rows: Sequence[np.ndarray],
    b_values: Sequence[float],
    *,
    active_pool_size: int = 6,
) -> np.ndarray:
    if not a_rows:
        return nominal_acc
    a_mat = np.asarray(a_rows, dtype=float)
    b_vec = np.asarray(b_values, dtype=float)
    valid = np.linalg.norm(a_mat, axis=1) > EPS
    if not np.any(valid):
        return nominal_acc
    a_mat = a_mat[valid]
    b_vec = b_vec[valid]

    violations = a_mat @ nominal_acc - b_vec
    if float(np.max(violations)) <= 1e-8:
        return nominal_acc

    adjusted = nominal_acc.copy()
    for _ in range(4):
        violations = a_mat @ adjusted - b_vec
        active = np.where(violations > 1e-8)[0]
        if active.size == 0:
            break
        order = active[np.argsort(violations[active])[::-1]][:active_pool_size]
        for idx in order:
            row = a_mat[int(idx)]
            current_violation = float(np.dot(row, adjusted) - b_vec[int(idx)])
            if current_violation <= 1e-8:
                continue
            adjusted = adjusted - (current_violation / (float(np.dot(row, row)) + EPS)) * row
    return adjusted


def _project_accel_to_halfspaces_osqp(
    nominal_acc: np.ndarray,
    a_rows: Sequence[np.ndarray],
    b_values: Sequence[float],
) -> np.ndarray:
    if not a_rows:
        return nominal_acc
    try:
        import osqp
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError("cbf_qp_osqp requires the optional osqp and scipy packages") from exc

    a_mat = np.asarray(a_rows, dtype=float)
    b_vec = np.asarray(b_values, dtype=float)
    valid = np.linalg.norm(a_mat, axis=1) > EPS
    if not np.any(valid):
        return nominal_acc
    a_mat = a_mat[valid]
    b_vec = b_vec[valid]
    if float(np.max(a_mat @ nominal_acc - b_vec)) <= 1e-8:
        return nominal_acc

    solver = osqp.OSQP()
    solver.setup(
        P=sp.eye(3, format="csc"),
        q=-nominal_acc,
        A=sp.csc_matrix(a_mat),
        l=np.full(a_mat.shape[0], -np.inf),
        u=b_vec,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=4000,
        polishing=False,
        verbose=False,
    )
    result = solver.solve()
    if result.x is None or result.info.status_val not in {1, 2}:
        return _project_accel_to_halfspaces(nominal_acc, a_rows, b_values)
    return np.asarray(result.x, dtype=float)


def _cbf_halfspaces(
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    nbr_pos: np.ndarray,
    nbr_vel: np.ndarray,
    obstacles: Sequence[Obstacle],
    mav: MAVParams,
    obstacle_pos_i: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    a_rows: list[np.ndarray] = []
    b_values: list[float] = []
    pair_clearance = mav.safe_radius_m
    pair_influence = max(2.0 * pair_clearance, mav.safety_influence_m + 0.8)
    k0 = 2.0
    k1 = 2.8

    for pj, vj in zip(nbr_pos, nbr_vel):
        rel = pos_i - pj
        dist = float(np.linalg.norm(rel))
        if dist > pair_influence:
            continue
        rel_vel = vel_i - vj
        h = dist**2 - pair_clearance**2
        h_dot = 2.0 * float(np.dot(rel, rel_vel))
        lower_bound = -2.0 * float(np.dot(rel_vel, rel_vel)) - k1 * h_dot - k0 * h
        a_rows.append(-2.0 * rel)
        b_values.append(-lower_bound)

    obstacle_clearance_margin = mav.collision_radius_m + mav.obstacle_clearance_margin_m
    obs_ref = pos_i if obstacle_pos_i is None else obstacle_pos_i
    for obs in obstacles:
        if not (obs.height_min - 0.15 <= obs_ref[2] <= obs.height_max + 0.15):
            continue
        center = np.asarray(obs.center, dtype=float)
        rel_xy = obs_ref[:2] - center[:2]
        dist_xy = float(np.linalg.norm(rel_xy))
        clearance = obs.radius + obstacle_clearance_margin
        if dist_xy > clearance + 1.1:
            continue
        rel_vel_xy = vel_i[:2]
        h = dist_xy**2 - clearance**2
        h_dot = 2.0 * float(np.dot(rel_xy, rel_vel_xy))
        lower_bound = -2.0 * float(np.dot(rel_vel_xy, rel_vel_xy)) - 2.6 * h_dot - 2.4 * h
        row = np.array([-2.0 * rel_xy[0], -2.0 * rel_xy[1], 0.0], dtype=float)
        a_rows.append(row)
        b_values.append(-lower_bound)

    return a_rows, b_values


def _cbf_qp_accel(
    nominal_acc: np.ndarray,
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    nbr_pos: np.ndarray,
    nbr_vel: np.ndarray,
    obstacles: Sequence[Obstacle],
    mav: MAVParams,
    obstacle_pos_i: np.ndarray | None = None,
) -> np.ndarray:
    a_rows, b_values = _cbf_halfspaces(pos_i, vel_i, nbr_pos, nbr_vel, obstacles, mav, obstacle_pos_i)
    return _project_accel_to_halfspaces(nominal_acc, a_rows, b_values)


def _cbf_qp_osqp_accel(
    nominal_acc: np.ndarray,
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    nbr_pos: np.ndarray,
    nbr_vel: np.ndarray,
    obstacles: Sequence[Obstacle],
    mav: MAVParams,
    obstacle_pos_i: np.ndarray | None = None,
) -> np.ndarray:
    a_rows, b_values = _cbf_halfspaces(pos_i, vel_i, nbr_pos, nbr_vel, obstacles, mav, obstacle_pos_i)
    return _project_accel_to_halfspaces_osqp(nominal_acc, a_rows, b_values)


@lru_cache(maxsize=1)
def _load_rvo2_library():
    lib_path = Path(__file__).resolve().parents[2] / "external" / "librvo2_c_api.so"
    if not lib_path.exists():
        raise RuntimeError(f"RVO2 wrapper library not found at {lib_path}")
    lib = ctypes.CDLL(str(lib_path))
    lib.rvo2_compute_velocities.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rvo2_compute_velocities.restype = ctypes.c_int
    return lib


def _rvo2_external_accel(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    obstacles: Sequence[Obstacle],
    arena_min: np.ndarray,
    arena_max: np.ndarray,
    mav: MAVParams,
    target_speeds: np.ndarray | None,
    dt_s: float,
    obstacle_positions: np.ndarray | None = None,
) -> np.ndarray:
    n = positions.shape[0]
    base_acc = np.zeros_like(positions)
    preferred = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        target_speed = mav.max_speed_mps if target_speeds is None else float(target_speeds[i])
        goal = _goal_accel(positions[i], velocities[i], goals[i], target_speed)
        obs_pos_i = positions[i] if obstacle_positions is None else obstacle_positions[i]
        obs = _obstacle_accel(obs_pos_i, obstacles, influence=1.2 + mav.obstacle_clearance_margin_m)
        bound = _boundary_accel(positions[i], velocities[i], arena_min, arena_max)
        damping = -velocities[i]
        base_acc[i] = (
            1.25 * goal
            + 4.0 * obs
            + 1.15 * bound
            + 0.25 * damping
        )
        pref_xy = velocities[i, :2] + base_acc[i, :2] * dt_s
        speed = float(np.linalg.norm(pref_xy))
        max_speed = min(mav.max_speed_mps, target_speed)
        if speed > max_speed > EPS:
            pref_xy = pref_xy * (max_speed / speed)
        preferred[i] = pref_xy

    out = np.zeros((n, 2), dtype=np.float64)
    pos_xy = np.ascontiguousarray(positions[:, :2], dtype=np.float64)
    vel_xy = np.ascontiguousarray(velocities[:, :2], dtype=np.float64)
    pref_xy = np.ascontiguousarray(preferred, dtype=np.float64)
    code = _load_rvo2_library().rvo2_compute_velocities(
        int(n),
        pos_xy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        vel_xy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pref_xy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(dt_s),
        float(max(mav.safety_influence_m + 1.0, 2.0 * mav.safe_radius_m)),
        int(12),
        2.5,
        2.0,
        float(mav.collision_radius_m),
        float(mav.max_speed_mps),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"RVO2 wrapper failed with status {code}")

    acc = np.zeros_like(positions)
    acc[:, :2] = (out - velocities[:, :2]) / max(dt_s, EPS)
    acc[:, 2] = base_acc[:, 2]
    return acc


def _candidate_directions(pos_i: np.ndarray, goal_i: np.ndarray) -> list[np.ndarray]:
    goal_dir = _unit(goal_i - pos_i)
    if float(np.linalg.norm(goal_dir[:2])) < EPS:
        lateral = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        lateral = np.array([-goal_dir[1], goal_dir[0], 0.0], dtype=float)
        lateral = _unit(lateral)
    vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    raw = [
        goal_dir,
        _unit(goal_dir + 0.55 * lateral),
        _unit(goal_dir - 0.55 * lateral),
        _unit(goal_dir + 0.30 * vertical),
        _unit(goal_dir - 0.30 * vertical),
        lateral,
        -lateral,
    ]
    return [direction for direction in raw if float(np.linalg.norm(direction)) > 0.5]


def _trajectory_score(
    accel: np.ndarray,
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    goal_i: np.ndarray,
    nbr_pos: np.ndarray,
    nbr_vel: np.ndarray,
    obstacles: Sequence[Obstacle],
    arena_min: np.ndarray,
    arena_max: np.ndarray,
    mav: MAVParams,
    *,
    horizon_s: float,
    samples: int,
    pair_clearance: float,
    obstacle_margin: float,
    pair_weight: float,
    obstacle_weight: float,
    progress_weight: float,
) -> float:
    times = np.linspace(horizon_s / samples, horizon_s, samples)
    pred = pos_i[None, :] + times[:, None] * vel_i[None, :] + 0.5 * (times[:, None] ** 2) * accel[None, :]
    pred = np.minimum(np.maximum(pred, arena_min[None, :]), arena_max[None, :])

    goal_dist = np.linalg.norm(pred - goal_i[None, :], axis=1)
    score = progress_weight * float(goal_dist[-1]) + 0.15 * float(np.mean(goal_dist))
    score += 0.08 * float(np.dot(accel, accel))

    if nbr_pos.size > 0:
        nbr_pred = nbr_pos[None, :, :] + times[:, None, None] * nbr_vel[None, :, :]
        dist = np.linalg.norm(pred[:, None, :] - nbr_pred, axis=2)
        violation = np.maximum(pair_clearance - dist, 0.0)
        if violation.size:
            score += pair_weight * float(np.sum(violation**2))
            score += 0.05 / (float(np.min(dist)) + 0.05)

    obstacle_clearance = mav.collision_radius_m + obstacle_margin
    for obs in obstacles:
        center = np.asarray(obs.center, dtype=float)
        in_height = (obs.height_min - 0.10 <= pred[:, 2]) & (pred[:, 2] <= obs.height_max + 0.10)
        if not np.any(in_height):
            continue
        dist_xy = np.linalg.norm(pred[:, :2] - center[None, :2], axis=1)
        clearance = obs.radius + obstacle_clearance
        violation = np.maximum(clearance - dist_xy[in_height], 0.0)
        if violation.size:
            score += obstacle_weight * float(np.sum(violation**2))
            score += 0.03 / (float(np.min(dist_xy[in_height])) + 0.05)

    lower_violation = np.maximum(arena_min[None, :] + 0.20 - pred, 0.0)
    upper_violation = np.maximum(pred - (arena_max[None, :] - 0.20), 0.0)
    score += 600.0 * float(np.sum(lower_violation**2 + upper_violation**2))
    return score


def _predictive_pairwise_accel(
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    nbr_pos: np.ndarray,
    nbr_vel: np.ndarray,
    *,
    clearance: float,
    horizon_s: float,
    gain: float,
) -> np.ndarray:
    if nbr_pos.size == 0:
        return np.zeros(3, dtype=float)
    acc = np.zeros(3, dtype=float)
    influence = clearance + 0.85
    for pj, vj in zip(nbr_pos, nbr_vel):
        rel = pos_i - pj
        rel_vel = vel_i - vj
        speed2 = float(np.dot(rel_vel, rel_vel))
        if speed2 > EPS:
            closest_t = float(np.clip(-float(np.dot(rel, rel_vel)) / speed2, 0.0, horizon_s))
        else:
            closest_t = 0.0
        closest = rel + closest_t * rel_vel
        dist = float(np.linalg.norm(closest)) + EPS
        if dist > influence:
            continue
        direction = closest / dist
        if dist < clearance + 0.05:
            current_dist = float(np.linalg.norm(rel)) + EPS
            direction = rel / current_dist
        closing = max(0.0, -float(np.dot(rel_vel, direction)))
        depth = max(0.0, (influence - dist) / max(influence - clearance, 0.05))
        acc += gain * (depth**2 + 0.35 * closing) * direction
    return acc


def _predictive_obstacle_accel(
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    obstacles: Sequence[Obstacle],
    mav: MAVParams,
    *,
    margin: float,
    horizon_s: float,
    gain: float,
) -> np.ndarray:
    acc = np.zeros(3, dtype=float)
    for obs in obstacles:
        if not (obs.height_min - 0.20 <= pos_i[2] <= obs.height_max + 0.20):
            continue
        center = np.asarray(obs.center, dtype=float)
        rel = pos_i[:2] - center[:2]
        vel = vel_i[:2]
        speed2 = float(np.dot(vel, vel))
        if speed2 > EPS:
            closest_t = float(np.clip(-float(np.dot(rel, vel)) / speed2, 0.0, horizon_s))
        else:
            closest_t = 0.0
        closest = rel + closest_t * vel
        dist = float(np.linalg.norm(closest)) + EPS
        clearance = obs.radius + mav.collision_radius_m + margin
        influence = clearance + 1.0
        if dist > influence:
            continue
        direction_xy = closest / dist
        if dist < clearance + 0.05:
            current_dist = float(np.linalg.norm(rel)) + EPS
            direction_xy = rel / current_dist
        depth = max(0.0, (influence - dist) / max(influence - clearance, 0.05))
        acc[:2] += gain * depth**2 * direction_xy
    return acc


def _trajectory_library_accel(
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    goal_i: np.ndarray,
    nbr_pos: np.ndarray,
    nbr_vel: np.ndarray,
    obstacles: Sequence[Obstacle],
    arena_min: np.ndarray,
    arena_max: np.ndarray,
    mav: MAVParams,
    target_speed: float,
    base_goal: np.ndarray,
    base_obs: np.ndarray,
    base_bound: np.ndarray,
    base_damping: np.ndarray,
    *,
    mode: str,
    obstacle_pos_i: np.ndarray | None = None,
) -> np.ndarray:
    obs_ref = pos_i if obstacle_pos_i is None else obstacle_pos_i
    if mode == "mader_like":
        pair_clearance = mav.safe_radius_m + 0.08
        pair_pred = _predictive_pairwise_accel(
            pos_i,
            vel_i,
            nbr_pos,
            nbr_vel,
            clearance=pair_clearance,
            horizon_s=2.4,
            gain=4.2,
        )
        obstacle_pred = _predictive_obstacle_accel(
            obs_ref,
            vel_i,
            obstacles,
            mav,
            margin=mav.obstacle_clearance_margin_m + 0.12,
            horizon_s=2.0,
            gain=3.5,
        )
        return (
            0.75 * base_goal
            + pair_pred
            + 3.6 * base_obs
            + obstacle_pred
            + 1.25 * base_bound
            + 0.35 * base_damping
        )
    else:
        pair_clearance = mav.safe_radius_m
        pair_pred = _predictive_pairwise_accel(
            pos_i,
            vel_i,
            nbr_pos,
            nbr_vel,
            clearance=pair_clearance,
            horizon_s=1.35,
            gain=3.0,
        )
        obstacle_pred = _predictive_obstacle_accel(
            obs_ref,
            vel_i,
            obstacles,
            mav,
            margin=mav.obstacle_clearance_margin_m + 0.05,
            horizon_s=1.35,
            gain=2.8,
        )
        return (
            1.15 * base_goal
            + pair_pred
            + 3.3 * base_obs
            + obstacle_pred
            + 1.10 * base_bound
            + 0.25 * base_damping
        )


def _safety_shield(
    nominal_acc: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_pos: List[np.ndarray],
    neighbor_vel: List[np.ndarray],
    weights: ControllerWeights,
    mav: MAVParams,
    gain: float = 1.0,
) -> np.ndarray:
    corrected = nominal_acc.copy()
    n = positions.shape[0]
    safe = mav.safe_radius_m
    influence = mav.safety_influence_m
    for i in range(n):
        p_i = positions[i]
        v_i = velocities[i]
        pjs = neighbor_pos[i]
        vjs = neighbor_vel[i]
        if pjs.size == 0:
            continue
        rel = p_i[None, :] - pjs
        dist = np.linalg.norm(rel, axis=1) + EPS
        dirs = rel / dist[:, None]
        relv = v_i[None, :] - vjs
        closing_speed = -np.sum(relv * dirs, axis=1)  # positive when approaching
        mask = dist < influence
        if not np.any(mask):
            continue
        h = np.maximum(influence - dist[mask], 0.0)
        # Blend distance-based and time-to-collision-like penalties.
        dist_term = weights.safety * h / max(influence - safe, 0.05)
        closing_term = weights.closing * np.maximum(closing_speed[mask], 0.0)
        correction = gain * np.sum((dist_term + closing_term)[:, None] * dirs[mask], axis=0)
        corrected[i] += correction
    return corrected


def compute_acceleration(
    controller: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    neighbor_pos: List[np.ndarray],
    neighbor_vel: List[np.ndarray],
    obstacles: Sequence[Obstacle],
    arena_min: np.ndarray,
    arena_max: np.ndarray,
    weights: ControllerWeights,
    mav: MAVParams,
    target_speeds: np.ndarray | None = None,
    shield_neighbor_pos: List[np.ndarray] | None = None,
    shield_neighbor_vel: List[np.ndarray] | None = None,
    safety_gain: float = 1.0,
    dt_s: float = 0.05,
    apply_safety_shield: bool | None = None,
    obstacle_positions: np.ndarray | None = None,
) -> np.ndarray:
    traits = get_controller_traits(controller)
    n = positions.shape[0]
    acc = np.zeros_like(positions)

    if traits.family == "rvo2_external":
        return limit_norm(
            _rvo2_external_accel(
                positions,
                velocities,
                goals,
                obstacles,
                arena_min,
                arena_max,
                mav,
                target_speeds,
                dt_s,
                obstacle_positions,
            ),
            mav.max_accel_mps2,
        )

    for i in range(n):
        pos_i = positions[i]
        vel_i = velocities[i]
        obs_pos_i = pos_i if obstacle_positions is None else obstacle_positions[i]
        sep, align, coh = _neighbor_terms(pos_i, vel_i, neighbor_pos[i], neighbor_vel[i], mav)
        target_speed = mav.max_speed_mps if target_speeds is None else float(target_speeds[i])
        goal = _goal_accel(pos_i, vel_i, goals[i], target_speed)
        obs = _obstacle_accel(obs_pos_i, obstacles, influence=1.2 + mav.obstacle_clearance_margin_m)
        bound = _boundary_accel(pos_i, vel_i, arena_min, arena_max)
        damping = -vel_i

        if traits.family == "apf":
            acc[i] = (
                2.0 * sep
                + 1.45 * goal
                + 2.8 * obs
                + 1.2 * bound
                + 0.35 * damping
            )
        elif traits.family == "boids":
            acc[i] = (
                1.7 * sep
                + 0.40 * align
                + 0.25 * coh
                + 1.0 * goal
                + 2.2 * obs
                + 1.0 * bound
                + 0.25 * damping
            )
        elif traits.family == "orca":
            pairwise_orca = _orca_pairwise_accel(
                pos_i,
                vel_i,
                goals[i],
                neighbor_pos[i],
                neighbor_vel[i],
                mav,
            )
            acc[i] = (
                1.25 * goal
                + pairwise_orca
                + 4.0 * obs
                + 1.15 * bound
                + 0.25 * damping
            )
        elif traits.family == "cbf_qp":
            nominal = (
                1.45 * goal
                + 3.0 * obs
                + 1.15 * bound
                + 0.30 * damping
            )
            acc[i] = _cbf_qp_accel(
                nominal,
                pos_i,
                vel_i,
                neighbor_pos[i],
                neighbor_vel[i],
                obstacles,
                mav,
                obs_pos_i,
            )
        elif traits.family == "cbf_qp_osqp":
            nominal = (
                1.45 * goal
                + 3.0 * obs
                + 1.15 * bound
                + 0.30 * damping
            )
            acc[i] = _cbf_qp_osqp_accel(
                nominal,
                pos_i,
                vel_i,
                neighbor_pos[i],
                neighbor_vel[i],
                obstacles,
                mav,
                obs_pos_i,
            )
        elif traits.family in {"decentralized_mpc", "mader_like"}:
            acc[i] = _trajectory_library_accel(
                pos_i,
                vel_i,
                goals[i],
                neighbor_pos[i],
                neighbor_vel[i],
                obstacles,
                arena_min,
                arena_max,
                mav,
                target_speed,
                goal,
                obs,
                bound,
                damping,
                mode=traits.family,
                obstacle_pos_i=obs_pos_i,
            )
        elif traits.family == "proposed":
            # Tuned but still transparent bio-inspired nominal controller.
            acc[i] = (
                weights.separation * sep
                + weights.alignment * align
                + weights.cohesion * coh
                + weights.goal * goal
                + weights.obstacle * obs
                + weights.boundary * bound
                + weights.damping * damping
            )
        else:
            raise RuntimeError("Invalid controller family")

    shield_enabled = traits.use_safety_shield if apply_safety_shield is None else apply_safety_shield
    if shield_enabled:
        safety_pos = neighbor_pos if shield_neighbor_pos is None else shield_neighbor_pos
        safety_vel = neighbor_vel if shield_neighbor_vel is None else shield_neighbor_vel
        acc = _safety_shield(acc, positions, velocities, safety_pos, safety_vel, weights, mav, safety_gain)

    return limit_norm(acc, mav.max_accel_mps2)
