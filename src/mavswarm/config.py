from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class MAVParams:
    """Crazyflie-class simplified micro air vehicle parameters.

    The simulator uses a double-integrator translational model with bounded
    speed and acceleration. These limits should be edited if a different MAV
    platform is assumed in the paper.
    """

    mass_kg: float = 0.027
    max_speed_mps: float = 1.5
    max_accel_mps2: float = 3.0
    collision_radius_m: float = 0.12
    safe_radius_m: float = 0.35
    safety_influence_m: float = 0.85
    altitude_min_m: float = 0.4
    altitude_max_m: float = 2.4
    initial_spacing_scale: float = 1.05
    terminal_slot_min_spacing_m: float = 0.50
    obstacle_clearance_margin_m: float = 0.10


@dataclass
class CommParams:
    comm_radius_m: float = 4.0
    k_neighbors: int = 12
    packet_loss: float = 0.0
    latency_ms: float = 0.0
    update_rate_hz: float = 20.0
    obstacle_update_rate_hz: float = 20.0
    position_noise_std_m: float = 0.015
    velocity_noise_std_mps: float = 0.025
    local_safety_dropout: float = 0.0
    local_safety_latency_ms: float = 0.0
    local_safety_update_rate_hz: float = 20.0
    local_safety_position_noise_std_m: float = 0.0
    local_safety_velocity_noise_std_mps: float = 0.0


@dataclass
class ControllerWeights:
    separation: float = 2.2
    alignment: float = 0.55
    cohesion: float = 0.42
    goal: float = 1.25
    obstacle: float = 2.7
    boundary: float = 1.5
    damping: float = 0.42
    safety: float = 5.0
    closing: float = 1.4


@dataclass
class SimParams:
    dt_s: float = 0.05
    horizon_s: float = 65.0
    log_stride: int = 10
    arena_min: List[float] = field(default_factory=lambda: [-42.0, -42.0, 0.3])
    arena_max: List[float] = field(default_factory=lambda: [42.0, 42.0, 2.5])
    goal_tolerance_m: float = 0.75
    completion_fraction: float = 0.90
    save_trajectory: bool = False
    trajectory_stride: int = 5
    completed_agents_leave_traffic: bool = False


@dataclass
class RunSpec:
    scenario: str = "corridor_obstacles"
    controller: str = "proposed"
    n_agents: int = 100
    seed: int = 0
    sim: SimParams = field(default_factory=SimParams)
    mav: MAVParams = field(default_factory=MAVParams)
    comm: CommParams = field(default_factory=CommParams)
    weights: ControllerWeights = field(default_factory=ControllerWeights)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SweepConfig:
    name: str = "ijmav_quick"
    repeats: int = 3
    seeds: Optional[List[int]] = None
    scenarios: List[str] = field(default_factory=lambda: ["open_flock", "corridor_obstacles", "crossing_traffic"])
    controllers: List[str] = field(default_factory=lambda: [
        "boids", "apf", "proposed_no_shield", "proposed"
    ])
    n_agents: List[int] = field(default_factory=lambda: [50, 100, 200])
    packet_loss: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    latency_ms: List[float] = field(default_factory=lambda: [0.0, 50.0, 100.0])
    scenario_subset_for_full_factorial: List[str] = field(default_factory=lambda: ["corridor_obstacles"])
    controller_subset_for_full_factorial: List[str] = field(default_factory=lambda: ["proposed"])
    sim: SimParams = field(default_factory=SimParams)
    mav: MAVParams = field(default_factory=MAVParams)
    comm: CommParams = field(default_factory=CommParams)
    weights: ControllerWeights = field(default_factory=ControllerWeights)
    max_runs: Optional[int] = None
    blocks: Optional[List[Dict[str, Any]]] = None


def _dataclass_from_dict(cls, data: Optional[Dict[str, Any]]):
    if data is None:
        return cls()
    base = cls()
    for key, value in data.items():
        if hasattr(base, key):
            setattr(base, key, value)
    return base


def load_sweep_config(path: str | Path) -> SweepConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = SweepConfig()
    for key, value in raw.items():
        if key == "sim":
            cfg.sim = _dataclass_from_dict(SimParams, value)
        elif key == "mav":
            cfg.mav = _dataclass_from_dict(MAVParams, value)
        elif key == "comm":
            cfg.comm = _dataclass_from_dict(CommParams, value)
        elif key == "weights":
            cfg.weights = _dataclass_from_dict(ControllerWeights, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
