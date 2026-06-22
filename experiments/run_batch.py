from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mavswarm.config import RunSpec, load_sweep_config  # noqa: E402
from mavswarm.sim import make_run_id, run_simulation  # noqa: E402


def _as_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, list):
        return value
    return [value]


def _set_if_present(obj, key: str, value):
    if value is not None and hasattr(obj, key):
        setattr(obj, key, value)


def expand_sweep(config_path: str | Path) -> List[RunSpec]:
    cfg = load_sweep_config(config_path)
    default_seeds = cfg.seeds if cfg.seeds is not None else list(range(cfg.repeats))

    if cfg.blocks:
        blocks = cfg.blocks
    else:
        # Sensible default: baseline scaling + packet-loss and latency robustness.
        blocks = [
            {
                "name": "baseline_scaling",
                "scenarios": cfg.scenarios,
                "controllers": cfg.controllers,
                "n_agents": cfg.n_agents,
                "packet_loss": [0.0],
                "latency_ms": [0.0],
            },
            {
                "name": "packet_loss_robustness",
                "scenarios": cfg.scenario_subset_for_full_factorial,
                "controllers": cfg.controller_subset_for_full_factorial,
                "n_agents": [100],
                "packet_loss": cfg.packet_loss,
                "latency_ms": [50.0],
            },
            {
                "name": "latency_robustness",
                "scenarios": cfg.scenario_subset_for_full_factorial,
                "controllers": cfg.controller_subset_for_full_factorial,
                "n_agents": [100],
                "packet_loss": [0.1],
                "latency_ms": cfg.latency_ms,
            },
        ]

    specs: List[RunSpec] = []
    for block in blocks:
        seeds = _as_list(block.get("seeds"), default_seeds)
        scenarios = _as_list(block.get("scenarios"), cfg.scenarios)
        controllers = _as_list(block.get("controllers"), cfg.controllers)
        n_agents_values = _as_list(block.get("n_agents"), cfg.n_agents)
        packet_loss_values = _as_list(block.get("packet_loss"), cfg.packet_loss)
        latency_values = _as_list(block.get("latency_ms"), cfg.latency_ms)

        for scenario in scenarios:
            for controller in controllers:
                for n_agents in n_agents_values:
                    for packet_loss in packet_loss_values:
                        for latency_ms in latency_values:
                            for seed in seeds:
                                sim = copy.deepcopy(cfg.sim)
                                mav = copy.deepcopy(cfg.mav)
                                comm = copy.deepcopy(cfg.comm)
                                weights = copy.deepcopy(cfg.weights)
                                metadata = {"block": block.get("name", "unnamed")}

                                # Common override names accepted inside each block.
                                for key, value in block.items():
                                    if key.startswith("sim."):
                                        _set_if_present(sim, key.split(".", 1)[1], value)
                                    elif key.startswith("mav."):
                                        _set_if_present(mav, key.split(".", 1)[1], value)
                                    elif key.startswith("comm."):
                                        _set_if_present(comm, key.split(".", 1)[1], value)
                                    elif key.startswith("weights."):
                                        _set_if_present(weights, key.split(".", 1)[1], value)
                                    elif key.startswith("metadata."):
                                        metadata[key.split(".", 1)[1]] = value

                                # Convenience aliases.
                                if "horizon_s" in block:
                                    sim.horizon_s = float(block["horizon_s"])
                                if "save_trajectory" in block:
                                    sim.save_trajectory = bool(block["save_trajectory"])
                                if "trajectory_stride" in block:
                                    sim.trajectory_stride = int(block["trajectory_stride"])
                                if "comm_radius_m" in block:
                                    comm.comm_radius_m = float(block["comm_radius_m"])
                                if "k_neighbors" in block:
                                    comm.k_neighbors = int(block["k_neighbors"])
                                if "update_rate_hz" in block:
                                    comm.update_rate_hz = float(block["update_rate_hz"])
                                if "position_noise_std_m" in block:
                                    comm.position_noise_std_m = float(block["position_noise_std_m"])
                                if "velocity_noise_std_mps" in block:
                                    comm.velocity_noise_std_mps = float(block["velocity_noise_std_mps"])

                                comm.packet_loss = float(packet_loss)
                                comm.latency_ms = float(latency_ms)
                                specs.append(RunSpec(
                                    scenario=str(scenario),
                                    controller=str(controller),
                                    n_agents=int(n_agents),
                                    seed=int(seed),
                                    sim=sim,
                                    mav=mav,
                                    comm=comm,
                                    weights=weights,
                                    metadata=metadata,
                                ))
    if cfg.max_runs is not None:
        specs = specs[: int(cfg.max_runs)]
    deduped: List[RunSpec] = []
    seen_run_ids: set[str] = set()
    for spec in specs:
        run_id = make_run_id(spec)
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        deduped.append(spec)
    specs = deduped
    return specs


def _run_one(args):
    spec, out_dir, overwrite = args
    run_dir = Path(out_dir) / make_run_id(spec)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    result = run_simulation(spec, out_dir=out_dir)
    return result["metrics"]


def _apply_rate_overrides(
    specs: List[RunSpec],
    *,
    comm_update_rate_hz: float | None,
    local_safety_update_rate_hz: float | None,
    obstacle_update_rate_hz: float | None,
) -> None:
    for spec in specs:
        if comm_update_rate_hz is not None:
            spec.comm.update_rate_hz = float(comm_update_rate_hz)
        if local_safety_update_rate_hz is not None:
            spec.comm.local_safety_update_rate_hz = float(local_safety_update_rate_hz)
        if obstacle_update_rate_hz is not None:
            spec.comm.obstacle_update_rate_hz = float(obstacle_update_rate_hz)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAV swarm simulation sweeps.")
    parser.add_argument("--config", required=True, help="YAML sweep config")
    parser.add_argument("--out", default="results/ijmav", help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of runs for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Re-run existing runs")
    parser.add_argument("--comm-update-rate-hz", type=float, default=None, help="Override communication update rate for all runs")
    parser.add_argument("--local-safety-update-rate-hz", type=float, default=None, help="Override local safety observation update rate for all runs")
    parser.add_argument("--obstacle-update-rate-hz", type=float, default=None, help="Override obstacle perception update rate for all runs")
    args = parser.parse_args()

    specs = expand_sweep(args.config)
    _apply_rate_overrides(
        specs,
        comm_update_rate_hz=args.comm_update_rate_hz,
        local_safety_update_rate_hz=args.local_safety_update_rate_hz,
        obstacle_update_rate_hz=args.obstacle_update_rate_hz,
    )
    if args.max_runs is not None:
        specs = specs[: args.max_runs]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(spec, out_dir, args.overwrite) for spec in specs]
    metrics: List[Dict] = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc="simulations"):
            metrics.append(_run_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_run_one, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="simulations"):
                metrics.append(fut.result())

    df = pd.DataFrame(metrics).sort_values(["scenario", "controller", "n_agents", "packet_loss", "latency_ms", "seed"])
    df.to_csv(out_dir / "run_index.csv", index=False)
    print(f"Wrote {len(df)} run metrics to {out_dir / 'run_index.csv'}")


if __name__ == "__main__":
    main()
