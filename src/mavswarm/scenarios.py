from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

TERMINAL_SLOT_MIN_SPACING_M = 0.42


@dataclass(frozen=True)
class Obstacle:
    center: Tuple[float, float, float]
    radius: float
    height_min: float = 0.0
    height_max: float = 3.0


@dataclass
class ScenarioState:
    positions: np.ndarray
    velocities: np.ndarray
    goals: np.ndarray
    obstacles: List[Obstacle]
    arena_min: np.ndarray
    arena_max: np.ndarray
    description: str


def _axis_points(bounds: Tuple[float, float], count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([(bounds[0] + bounds[1]) * 0.5], dtype=float)
    return np.linspace(bounds[0], bounds[1], count)


def _lattice_counts(n: int, spans: np.ndarray) -> Tuple[int, int, int]:
    """Choose a compact 3D lattice that maximizes nearest-neighbor spacing."""
    if n <= 1:
        return (1, 1, 1)

    best_counts = (n, 1, 1)
    best_key = (-1.0, -float(n), 0.0)
    max_z = min(n, int(np.ceil(np.cbrt(n) * 8.0)) + 8)
    for nz in range(1, max_z + 1):
        nxy = int(np.ceil(n / nz))
        for nx in range(1, nxy + 1):
            ny = int(np.ceil(nxy / nx))
            counts = np.asarray([nx, ny, nz], dtype=int)
            capacity = int(np.prod(counts))
            if capacity < n:
                continue
            spacings = np.full(3, np.inf, dtype=float)
            active = counts > 1
            spacings[active] = spans[active] / (counts[active] - 1)
            min_spacing = float(np.min(spacings))
            max_spacing = float(np.max(spacings[active])) if np.any(active) else float("inf")
            unused = capacity - n
            key = (min_spacing, -float(unused), -max_spacing)
            if key > best_key:
                best_key = key
                best_counts = (int(nx), int(ny), int(nz))
    return best_counts


def _lattice_counts_with_spacing(n: int, spans: np.ndarray, min_spacing: float) -> Tuple[int, int, int]:
    """Choose lattice counts that preserve a requested minimum spacing when possible."""
    if n <= 1:
        return (1, 1, 1)

    max_counts = np.floor(spans / max(min_spacing, 1e-6)).astype(int) + 1
    max_counts = np.maximum(max_counts, 1)
    if int(np.prod(max_counts)) < n:
        return _lattice_counts(n, spans)

    best_counts = tuple(int(v) for v in max_counts)
    best_key = (-1.0, -float(np.prod(max_counts)), 0.0)
    for nx in range(1, int(max_counts[0]) + 1):
        for ny in range(1, int(max_counts[1]) + 1):
            min_nz = int(np.ceil(n / (nx * ny)))
            if min_nz > int(max_counts[2]):
                continue
            for nz in range(min_nz, int(max_counts[2]) + 1):
                counts = np.asarray([nx, ny, nz], dtype=int)
                capacity = int(np.prod(counts))
                spacings = np.full(3, np.inf, dtype=float)
                active = counts > 1
                spacings[active] = spans[active] / (counts[active] - 1)
                min_actual = float(np.min(spacings))
                if min_actual + 1e-9 < min_spacing:
                    continue
                max_actual = float(np.max(spacings[active])) if np.any(active) else float("inf")
                key = (min_actual, -float(capacity - n), -max_actual)
                if key > best_key:
                    best_key = key
                    best_counts = (int(nx), int(ny), int(nz))
    return best_counts


def _grid_positions(n: int, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    spans = np.asarray([x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]], dtype=float)
    nx, ny, nz = _lattice_counts(n, spans)
    xs = _axis_points(x_range, nx)
    ys = _axis_points(y_range, ny)
    zs = _axis_points(z_range, nz)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    pts = pts[rng.permutation(len(pts))[:n]]
    return pts


def _bounded_interval(center: float, lower: float, upper: float, desired_span: float) -> Tuple[float, float]:
    span = min(float(desired_span), max(float(upper - lower), 0.0))
    lo = center - 0.5 * span
    hi = center + 0.5 * span
    if lo < lower:
        hi += lower - lo
        lo = lower
    if hi > upper:
        lo -= hi - upper
        hi = upper
    lo = max(lower, lo)
    hi = min(upper, hi)
    return (float(lo), float(hi))


def _inner_axis_bounds(arena_min: np.ndarray, arena_max: np.ndarray, axis: int) -> Tuple[float, float]:
    span = float(arena_max[axis] - arena_min[axis])
    margin = min(max(0.8, 0.08 * span), 0.30 * span)
    return (float(arena_min[axis] + margin), float(arena_max[axis] - margin))


def _axis_value(arena_min: np.ndarray, arena_max: np.ndarray, axis: int, fraction: float) -> float:
    low, high = _inner_axis_bounds(arena_min, arena_max, axis)
    return float(low + fraction * (high - low))


def _axis_window(
    arena_min: np.ndarray,
    arena_max: np.ndarray,
    axis: int,
    center_fraction: float,
    span_fraction: float,
    min_span: float,
) -> Tuple[float, float]:
    low, high = _inner_axis_bounds(arena_min, arena_max, axis)
    span = high - low
    center = low + center_fraction * span
    desired_span = max(min_span, span_fraction * span)
    return _bounded_interval(center, low, high, desired_span)


def _altitude_window(arena_min: np.ndarray, arena_max: np.ndarray, low_fraction: float, high_fraction: float) -> Tuple[float, float]:
    span = float(arena_max[2] - arena_min[2])
    low = float(arena_min[2] + low_fraction * span)
    high = float(arena_min[2] + high_fraction * span)
    if high <= low:
        high = min(float(arena_max[2] - 0.05), low + 0.1)
    return (low, high)


def _altitude_value(arena_min: np.ndarray, arena_max: np.ndarray, fraction: float) -> float:
    return float(arena_min[2] + fraction * float(arena_max[2] - arena_min[2]))


def _terminal_slots(
    n: int,
    base_goal: np.ndarray,
    arena_min: np.ndarray,
    arena_max: np.ndarray,
    neighbor_goals: List[np.ndarray],
    min_spacing: float = TERMINAL_SLOT_MIN_SPACING_M,
) -> np.ndarray:
    low = arena_min + np.array([0.35, 0.45, 0.40])
    high = arena_max - np.array([0.50, 0.45, 0.40])
    root_n = float(np.sqrt(max(n, 1)))

    x_span = min(high[0] - low[0], max(2.2, 0.12 * root_n + 2.3))
    y_span = min(high[1] - low[1], max(4.8, 0.35 * root_n + 2.5))
    z_span = min(high[2] - low[2], max(1.2, 0.04 * root_n + 0.8))

    same_exit_neighbors = [
        other for other in neighbor_goals
        if not np.allclose(other, base_goal) and abs(float(other[0] - base_goal[0])) < 1.0
    ]
    if same_exit_neighbors:
        nearest_y_gap = min(abs(float(other[1] - base_goal[1])) for other in same_exit_neighbors)
        y_span = min(y_span, 0.85 * nearest_y_gap)

    bounds = [
        _bounded_interval(float(base_goal[0]), float(low[0]), float(high[0]), x_span),
        _bounded_interval(float(base_goal[1]), float(low[1]), float(high[1]), y_span),
        _bounded_interval(float(base_goal[2]), float(low[2]), float(high[2]), z_span),
    ]
    spans = np.asarray([hi - lo for lo, hi in bounds], dtype=float)
    nx, ny, nz = _lattice_counts_with_spacing(n, spans, min_spacing)
    xs = _axis_points(bounds[0], nx)
    ys = _axis_points(bounds[1], ny)
    zs = _axis_points(bounds[2], nz)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")
    slots = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    order = np.lexsort((slots[:, 0], slots[:, 2], slots[:, 1]))
    if len(order) == n:
        return slots[order]
    selected = np.linspace(0, len(order) - 1, n, dtype=int)
    return slots[order[selected]]


def _spread_terminal_goals(
    positions: np.ndarray,
    goals: np.ndarray,
    arena_min: np.ndarray,
    arena_max: np.ndarray,
    terminal_slot_min_spacing_m: float = TERMINAL_SLOT_MIN_SPACING_M,
) -> np.ndarray:
    spread = goals.copy()
    unique_goals: List[np.ndarray] = []
    labels = []
    for goal in goals:
        match = None
        for idx, existing in enumerate(unique_goals):
            if np.allclose(goal, existing):
                match = idx
                break
        if match is None:
            unique_goals.append(goal.copy())
            match = len(unique_goals) - 1
        labels.append(match)

    labels_arr = np.asarray(labels, dtype=int)
    for label, base_goal in enumerate(unique_goals):
        agent_idx = np.where(labels_arr == label)[0]
        slots = _terminal_slots(
            len(agent_idx),
            base_goal,
            arena_min,
            arena_max,
            unique_goals,
            min_spacing=terminal_slot_min_spacing_m,
        )
        group_pos = positions[agent_idx]
        travel = base_goal - np.mean(group_pos, axis=0)
        lateral_axis = 1 if abs(float(travel[0])) >= abs(float(travel[1])) else 0
        forward_axis = 0 if lateral_axis == 1 else 1
        agent_order = np.lexsort((group_pos[:, forward_axis], group_pos[:, 2], group_pos[:, lateral_axis]))
        slot_order = np.lexsort((slots[:, forward_axis], slots[:, 2], slots[:, lateral_axis]))
        spread[agent_idx[agent_order]] = slots[slot_order]
    return spread


def make_scenario(
    name: str,
    n_agents: int,
    seed: int,
    arena_min,
    arena_max,
    *,
    initial_spacing_scale: float = 1.0,
    terminal_slot_min_spacing_m: float = TERMINAL_SLOT_MIN_SPACING_M,
) -> ScenarioState:
    rng = np.random.default_rng(seed)
    arena_min = np.asarray(arena_min, dtype=float)
    arena_max = np.asarray(arena_max, dtype=float)
    center_y = _axis_value(arena_min, arena_max, 1, 0.5)
    center_x = _axis_value(arena_min, arena_max, 0, 0.5)
    mid_z = _altitude_value(arena_min, arena_max, 0.45)
    low_z_range = _altitude_window(arena_min, arena_max, 0.23, 0.68)
    high_z_range = _altitude_window(arena_min, arena_max, 0.27, 0.73)
    spacing_scale = max(float(initial_spacing_scale), 1.0)

    if name == "open_flock":
        positions = _grid_positions(
            n_agents,
            _axis_window(arena_min, arena_max, 0, 0.05, 0.07 * spacing_scale, 2.5 * spacing_scale),
            _axis_window(arena_min, arena_max, 1, 0.5, 0.28 * spacing_scale, 6.8 * spacing_scale),
            low_z_range,
            rng,
        )
        velocities = np.zeros((n_agents, 3), dtype=float)
        velocities[:, 0] = rng.normal(0.15, 0.03, n_agents)
        goals = np.tile(np.array([_axis_value(arena_min, arena_max, 0, 1.0), center_y, mid_z]), (n_agents, 1))
        obstacles: List[Obstacle] = []
        desc = "Open-space flocking from left to right."

    elif name == "corridor_obstacles":
        positions = _grid_positions(
            n_agents,
            _axis_window(arena_min, arena_max, 0, 0.04, 0.06 * spacing_scale, 2.0 * spacing_scale),
            _axis_window(arena_min, arena_max, 1, 0.5, 0.25 * spacing_scale, 4.4 * spacing_scale),
            low_z_range,
            rng,
        )
        velocities = np.zeros((n_agents, 3), dtype=float)
        goals = np.tile(np.array([_axis_value(arena_min, arena_max, 0, 1.0), center_y, mid_z]), (n_agents, 1))
        inner_x_low, inner_x_high = _inner_axis_bounds(arena_min, arena_max, 0)
        obstacle_count = max(4, int(round((inner_x_high - inner_x_low) / 9.5)))
        obstacle_xs = np.linspace(
            _axis_value(arena_min, arena_max, 0, 0.15),
            _axis_value(arena_min, arena_max, 0, 0.85),
            obstacle_count,
        )
        y_offsets = np.array([-0.95, 1.00, -0.85, 0.82], dtype=float)
        row_offsets = np.array([-4.8, 0.0, 4.8], dtype=float)
        radii = np.array([0.62, 0.72, 0.78, 0.68], dtype=float)
        obstacles = [
            Obstacle(
                center=(
                    float(x),
                    float(center_y + row_offset + y_offsets[(idx + row_idx) % len(y_offsets)]),
                    mid_z,
                ),
                radius=float(radii[(idx + row_idx) % len(radii)]),
            )
            for row_idx, row_offset in enumerate(row_offsets)
            for idx, x in enumerate(obstacle_xs)
        ]
        desc = "Obstacle-rich corridor with three lateral rows of alternating cylindrical obstacles."

    elif name == "crossing_traffic":
        n1 = n_agents // 2
        n2 = n_agents - n1
        p1 = _grid_positions(
            n1,
            _axis_window(arena_min, arena_max, 0, 0.04, 0.06 * spacing_scale, 1.8 * spacing_scale),
            _axis_window(arena_min, arena_max, 1, 0.5, 0.22 * spacing_scale, 5.6 * spacing_scale),
            low_z_range,
            rng,
        )
        p2 = _grid_positions(
            n2,
            _axis_window(arena_min, arena_max, 0, 0.5, 0.22 * spacing_scale, 5.6 * spacing_scale),
            _axis_window(arena_min, arena_max, 1, 0.04, 0.06 * spacing_scale, 1.8 * spacing_scale),
            high_z_range,
            rng,
        )
        positions = np.vstack([p1, p2])
        velocities = np.zeros((n_agents, 3), dtype=float)
        velocities[:n1, 0] = rng.normal(0.12, 0.03, n1)
        velocities[n1:, 1] = rng.normal(0.12, 0.03, n2)
        goals = np.zeros((n_agents, 3), dtype=float)
        goals[:n1, :] = np.array([_axis_value(arena_min, arena_max, 0, 1.0), center_y, mid_z])
        goals[n1:, :] = np.array([center_x, _axis_value(arena_min, arena_max, 1, 1.0), _altitude_value(arena_min, arena_max, 0.50)])
        obstacles = [Obstacle(center=(center_x, center_y, mid_z), radius=0.55)]
        desc = "Two sub-swarms crossing at a congested intersection."

    elif name == "merge_split":
        n1 = n_agents // 3
        n2 = n_agents // 3
        n3 = n_agents - n1 - n2
        p1 = _grid_positions(
            n1,
            _axis_window(arena_min, arena_max, 0, 0.04, 0.05 * spacing_scale, 1.4 * spacing_scale),
            _axis_window(arena_min, arena_max, 1, 0.24, 0.07 * spacing_scale, 2.0 * spacing_scale),
            low_z_range,
            rng,
        )
        p2 = _grid_positions(
            n2,
            _axis_window(arena_min, arena_max, 0, 0.04, 0.05 * spacing_scale, 1.4 * spacing_scale),
            _axis_window(arena_min, arena_max, 1, 0.50, 0.07 * spacing_scale, 2.0 * spacing_scale),
            low_z_range,
            rng,
        )
        p3 = _grid_positions(
            n3,
            _axis_window(arena_min, arena_max, 0, 0.04, 0.05 * spacing_scale, 1.4 * spacing_scale),
            _axis_window(arena_min, arena_max, 1, 0.76, 0.07 * spacing_scale, 2.0 * spacing_scale),
            low_z_range,
            rng,
        )
        positions = np.vstack([p1, p2, p3])
        velocities = np.zeros((n_agents, 3), dtype=float)
        goals = np.zeros((n_agents, 3), dtype=float)
        goals[:n1] = np.array([_axis_value(arena_min, arena_max, 0, 1.0), _axis_value(arena_min, arena_max, 1, 0.76), _altitude_value(arena_min, arena_max, 0.41)])
        goals[n1:n1+n2] = np.array([_axis_value(arena_min, arena_max, 0, 1.0), center_y, mid_z])
        goals[n1+n2:] = np.array([_axis_value(arena_min, arena_max, 0, 1.0), _axis_value(arena_min, arena_max, 1, 0.24), _altitude_value(arena_min, arena_max, 0.50)])
        obstacles = [
            Obstacle(center=(_axis_value(arena_min, arena_max, 0, 0.44), _axis_value(arena_min, arena_max, 1, 0.43), mid_z), radius=0.70),
            Obstacle(center=(_axis_value(arena_min, arena_max, 0, 0.44), _axis_value(arena_min, arena_max, 1, 0.57), mid_z), radius=0.70),
            Obstacle(center=(_axis_value(arena_min, arena_max, 0, 0.56), _axis_value(arena_min, arena_max, 1, 0.43), mid_z), radius=0.70),
            Obstacle(center=(_axis_value(arena_min, arena_max, 0, 0.56), _axis_value(arena_min, arena_max, 1, 0.57), mid_z), radius=0.70),
        ]
        desc = "Three groups cross through a shared central gate before splitting toward three exits."

    elif name == "failure_robustness":
        positions = _grid_positions(
            n_agents,
            _axis_window(arena_min, arena_max, 0, 0.05, 0.07 * spacing_scale, 2.5 * spacing_scale),
            _axis_window(arena_min, arena_max, 1, 0.5, 0.28 * spacing_scale, 6.8 * spacing_scale),
            low_z_range,
            rng,
        )
        velocities = np.zeros((n_agents, 3), dtype=float)
        goals = np.tile(np.array([_axis_value(arena_min, arena_max, 0, 1.0), center_y, mid_z]), (n_agents, 1))
        obstacles = [Obstacle(center=(_axis_value(arena_min, arena_max, 0, 0.49), center_y, mid_z), radius=0.65)]
        desc = "Open flocking with externally configured communication/sensing degradation."

    else:
        raise ValueError(f"Unknown scenario: {name}")

    positions = np.clip(positions, arena_min + [0.05, 0.05, 0.05], arena_max - [0.05, 0.05, 0.05])
    goals = _spread_terminal_goals(
        positions,
        goals,
        arena_min,
        arena_max,
        terminal_slot_min_spacing_m=terminal_slot_min_spacing_m,
    )
    return ScenarioState(
        positions=positions,
        velocities=velocities,
        goals=goals,
        obstacles=obstacles,
        arena_min=arena_min,
        arena_max=arena_max,
        description=desc,
    )
