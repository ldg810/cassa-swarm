from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.mission_threshold_times import threshold_crossing_time_s  # noqa: E402
from experiments.run_batch import expand_sweep  # noqa: E402
from mavswarm.config import RunSpec  # noqa: E402
from mavswarm.sim import make_run_id, run_simulation  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _source_ready(run_dir: Path) -> bool:
    return (
        (run_dir / "metrics.json").exists()
        and (run_dir / "timeseries.csv").exists()
        and (run_dir / "trajectory.csv").exists()
        and (run_dir / "trajectory.csv").stat().st_size > 0
    )


def _trajectory_npz_ready(run_dir: Path) -> bool:
    return (run_dir / "trajectory.npz").exists() and (run_dir / "trajectory.npz").stat().st_size > 0


def _replay_ready(run_dir: Path) -> bool:
    return (
        (run_dir / "metrics.json").exists()
        and (run_dir / "timeseries.csv").exists()
        and (run_dir / "metrics.json").stat().st_size > 0
    )


def _sort_key(spec: RunSpec) -> tuple[Any, ...]:
    block_order = {
        "matched_scaling": 0,
        "matched_packet_loss_robustness": 1,
        "matched_latency_robustness": 2,
        "matched_high_density_stress": 3,
    }
    scenario_order = {
        "open_flock": 0,
        "corridor_obstacles": 1,
        "crossing_traffic": 2,
        "merge_split": 3,
    }
    return (
        spec.n_agents,
        block_order.get(str(spec.metadata.get("block")), 99),
        scenario_order.get(spec.scenario, 99),
        spec.comm.packet_loss,
        spec.comm.latency_ms,
        spec.seed,
    )


def _select_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs = [spec for spec in expand_sweep(args.config) if spec.controller == "proposed"]
    for spec in specs:
        if args.comm_update_rate_hz is not None:
            spec.comm.update_rate_hz = float(args.comm_update_rate_hz)
        if args.local_safety_update_rate_hz is not None:
            spec.comm.local_safety_update_rate_hz = float(args.local_safety_update_rate_hz)
        if args.obstacle_update_rate_hz is not None:
            spec.comm.obstacle_update_rate_hz = float(args.obstacle_update_rate_hz)
    if args.scenario:
        allowed = set(args.scenario)
        specs = [spec for spec in specs if spec.scenario in allowed]
    if args.n_agents:
        allowed_n = set(args.n_agents)
        specs = [spec for spec in specs if spec.n_agents in allowed_n]
    if args.block:
        allowed_blocks = set(args.block)
        specs = [spec for spec in specs if str(spec.metadata.get("block")) in allowed_blocks]
    specs = sorted(specs, key=_sort_key)
    if args.start_index:
        specs = specs[args.start_index :]
    if args.max_runs is not None:
        specs = specs[: args.max_runs]
    return specs


def _ensure_trajectory_npz(run_dir: Path, overwrite: bool = False) -> Path:
    npz_path = run_dir / "trajectory.npz"
    if _trajectory_npz_ready(run_dir) and not overwrite:
        return npz_path

    csv_path = run_dir / "trajectory.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing source trajectory CSV: {csv_path}")
    required = ["t_s", "agent", "x", "y", "z", "vx", "vy", "vz", "goal_x", "goal_y", "goal_z"]
    data = pd.read_csv(csv_path, usecols=required)
    data = data.sort_values(["t_s", "agent"], kind="mergesort")
    times = np.asarray(sorted(data["t_s"].unique()), dtype=np.float32)
    agents = data["agent"].to_numpy(dtype=np.int64)
    unique_agents = np.unique(agents)
    if unique_agents.size == 0 or not np.array_equal(unique_agents, np.arange(unique_agents.size)):
        raise ValueError(f"Agent ids must be contiguous in {csv_path}")
    n_times = int(times.shape[0])
    n_agents = int(unique_agents.size)
    if len(data) != n_times * n_agents:
        raise ValueError(f"Trajectory must contain one row for every time/agent pair: {csv_path}")
    agent_grid = agents.reshape(n_times, n_agents)
    expected_grid = np.broadcast_to(np.arange(n_agents, dtype=np.int64), agent_grid.shape)
    if not np.array_equal(agent_grid, expected_grid):
        raise ValueError(f"Trajectory rows are not complete after sorting: {csv_path}")
    positions = data[["x", "y", "z"]].to_numpy(dtype=np.float32).reshape(n_times, n_agents, 3)
    velocities = data[["vx", "vy", "vz"]].to_numpy(dtype=np.float32).reshape(n_times, n_agents, 3)
    goals = data[["goal_x", "goal_y", "goal_z"]].to_numpy(dtype=np.float32).reshape(n_times, n_agents, 3)
    np.savez(
        npz_path,
        times=times,
        positions=positions,
        velocities=velocities,
        goals=goals,
    )
    return npz_path


def _generate_source(spec: RunSpec, source_root: Path, source_stride: int, overwrite: bool) -> dict[str, Any]:
    run_id = make_run_id(spec)
    run_dir = source_root / run_id
    if _source_ready(run_dir) and not overwrite:
        _ensure_trajectory_npz(run_dir)
        return _read_json(run_dir / "metrics.json")

    source_spec = copy.deepcopy(spec)
    source_spec.sim.save_trajectory = True
    source_spec.sim.trajectory_stride = int(source_stride)
    result = run_simulation(source_spec, out_dir=source_root)
    _ensure_trajectory_npz(run_dir, overwrite=True)
    return dict(result["metrics"])


def _generate_source_task(task: tuple[RunSpec, str, int, bool]) -> tuple[str, dict[str, Any]]:
    spec, source_root, source_stride, overwrite = task
    metrics = _generate_source(spec, Path(source_root), source_stride, overwrite)
    return make_run_id(spec), metrics


def _replay_command(args: argparse.Namespace, spec: RunSpec, source_dir: Path, replay_dir: Path) -> list[str]:
    script = ROOT / "external" / "crazyflow" / "examples" / "replay_paper_trajectory.py"
    return [
        str(args.crazyflow_python),
        str(script),
        "--trajectory",
        str(_ensure_trajectory_npz(source_dir)),
        "--output-dir",
        str(replay_dir),
        "--device",
        args.device,
        "--freq",
        str(args.freq),
        "--state-freq",
        str(args.state_freq),
        "--log-every",
        str(args.log_every),
        "--trajectory-stride",
        str(args.replay_trajectory_stride),
        "--goal-tolerance",
        str(spec.sim.goal_tolerance_m),
        "--completion-fraction",
        str(spec.sim.completion_fraction),
        "--collision-radius",
        str(spec.mav.collision_radius_m),
        "--safety-radius",
        str(spec.mav.safe_radius_m),
        "--downwash-strength",
        str(args.downwash_strength),
        "--downwash-force-cap-ratio",
        str(args.downwash_force_cap_ratio),
    ]


def _replay_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MUJOCO_GL", "egl")
    if args.device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    else:
        env.pop("JAX_PLATFORMS", None)
    return env


def _run_replay(args: argparse.Namespace, spec: RunSpec, source_root: Path, replay_root: Path) -> tuple[bool, str | None]:
    run_id = make_run_id(spec)
    source_dir = source_root / run_id
    replay_dir = replay_root / run_id
    replay_dir.mkdir(parents=True, exist_ok=True)
    if _replay_ready(replay_dir) and not args.overwrite_replay:
        return True, None

    cmd = _replay_command(args, spec, source_dir, replay_dir)
    log_path = replay_dir / "replay_stdout.log"
    start_wall = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=_replay_env(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start_wall
    if proc.returncode != 0:
        failure = {
            "run_id": run_id,
            "returncode": proc.returncode,
            "elapsed_wall_s": elapsed,
            "command": cmd,
            "log_path": str(log_path),
        }
        with (replay_dir / "replay_failed.json").open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2)
        return False, str(log_path)
    return True, None


def _run_replay_task(task: tuple[RunSpec, str, str, argparse.Namespace]) -> tuple[bool, str | None, float]:
    spec, source_root, replay_root, args = task
    start_wall = time.perf_counter()
    replay_ok, failure_log = _run_replay(args, spec, Path(source_root), Path(replay_root))
    return replay_ok, failure_log, time.perf_counter() - start_wall


def _thresholds(timeseries_path: Path) -> dict[str, float]:
    if not timeseries_path.exists():
        return {"T90_s": math.nan, "T95_s": math.nan, "T99_s": math.nan}
    timeseries = pd.read_csv(timeseries_path)
    return {
        "T90_s": threshold_crossing_time_s(timeseries, 0.90),
        "T95_s": threshold_crossing_time_s(timeseries, 0.95),
        "T99_s": threshold_crossing_time_s(timeseries, 0.99),
    }


def _summarize_run(spec: RunSpec, source_root: Path, replay_root: Path, replay_ok: bool, failure_log: str | None) -> dict[str, Any]:
    run_id = make_run_id(spec)
    source_dir = source_root / run_id
    replay_dir = replay_root / run_id
    row: dict[str, Any] = {
        "run_id": run_id,
        "block": spec.metadata.get("block"),
        "scenario": spec.scenario,
        "n_agents": spec.n_agents,
        "seed": spec.seed,
        "packet_loss": spec.comm.packet_loss,
        "latency_ms": spec.comm.latency_ms,
        "replay_completed": replay_ok,
        "failure_log": failure_log,
    }

    if (source_dir / "metrics.json").exists():
        src = _read_json(source_dir / "metrics.json")
        row.update(
            {
                "source_final_reached_fraction": src.get("final_reached_fraction"),
                "source_completion_time_s": src.get("completion_time_s"),
                "source_collision_pairs": src.get("cumulative_collision_pairs"),
                "source_obstacle_contacts": src.get("cumulative_obstacle_contacts"),
                "source_projection_adjustments": src.get("projection_agent_adjustments"),
                "source_projection_max_displacement_m": src.get("projection_max_displacement_m"),
            }
        )

    if replay_ok and (replay_dir / "metrics.json").exists():
        metrics = _read_json(replay_dir / "metrics.json")
        thresholds = _thresholds(replay_dir / "timeseries.csv")
        safety_success = (
            int(metrics.get("cumulative_collision_pairs", 0)) == 0
            and int(metrics.get("cumulative_obstacle_contacts", 0)) == 0
        )
        active_safety_success = (
            int(metrics.get("active_cumulative_collision_pairs", 0)) == 0
            and int(metrics.get("active_cumulative_obstacle_contacts", 0)) == 0
        )
        row.update(
            {
                "crazyflow_safety_success": safety_success,
                "crazyflow_active_safety_success": active_safety_success,
                "crazyflow_mission_success": metrics.get("mission_success"),
                "crazyflow_active_traffic_success": metrics.get("active_traffic_success"),
                "crazyflow_final_reached_fraction": metrics.get("final_reached_fraction"),
                "crazyflow_collision_pairs": metrics.get("cumulative_collision_pairs"),
                "crazyflow_obstacle_contacts": metrics.get("cumulative_obstacle_contacts"),
                "crazyflow_active_collision_pairs": metrics.get("active_cumulative_collision_pairs"),
                "crazyflow_active_obstacle_contacts": metrics.get("active_cumulative_obstacle_contacts"),
                "crazyflow_min_inter_agent_distance_m": metrics.get("min_inter_agent_distance_m"),
                "crazyflow_active_min_inter_agent_distance_m": metrics.get("active_min_inter_agent_distance_m"),
                "crazyflow_final_tracking_error_p95_m": metrics.get("final_tracking_error_p95_m"),
                "crazyflow_max_tracking_error_m": metrics.get("max_tracking_error_m"),
                "crazyflow_max_speed_mps": metrics.get("max_speed_mps"),
                "crazyflow_max_downwash_force_N": metrics.get("max_logged_downwash_force_N"),
                "crazyflow_wall_s": metrics.get("wall_s"),
                "crazyflow_real_time_factor": metrics.get("real_time_factor"),
                "crazyflow_jax_backend": metrics.get("jax_backend"),
                "crazyflow_downwash_enabled": metrics.get("downwash_enabled"),
                **thresholds,
            }
        )
    return row


def _write_summaries(rows: list[dict[str, Any]], replay_root: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(
        ["n_agents", "block", "scenario", "packet_loss", "latency_ms", "seed"],
        kind="stable",
    )
    df.to_csv(replay_root / "run_index.csv", index=False)

    completed = df[df["replay_completed"] == True].copy()  # noqa: E712
    if completed.empty:
        return

    def _finite_median(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.median()) if not values.empty else math.nan

    group_cols = ["block", "scenario", "n_agents", "packet_loss", "latency_ms"]
    aggregate_rows = []
    for keys, group in completed.groupby(group_cols, dropna=False, sort=False):
        record = dict(zip(group_cols, keys))
        record.update(
            {
                "runs": int(len(group)),
                "safety_success_rate": float(pd.to_numeric(group["crazyflow_safety_success"]).mean()),
                "mission_success_rate": float(pd.to_numeric(group["crazyflow_mission_success"]).mean()),
                "min_reached_fraction": float(pd.to_numeric(group["crazyflow_final_reached_fraction"], errors="coerce").min()),
                "collision_pairs_total": int(pd.to_numeric(group["crazyflow_collision_pairs"], errors="coerce").fillna(0).sum()),
                "obstacle_contacts_total": int(pd.to_numeric(group["crazyflow_obstacle_contacts"], errors="coerce").fillna(0).sum()),
                "max_tracking_error_m": float(pd.to_numeric(group["crazyflow_max_tracking_error_m"], errors="coerce").max()),
                "median_T90_s": _finite_median(group["T90_s"]),
                "median_T95_s": _finite_median(group["T95_s"]),
                "median_T99_s": _finite_median(group["T99_s"]),
                "T99_reached_runs": int(pd.to_numeric(group["T99_s"], errors="coerce").notna().sum()),
            }
        )
        aggregate_rows.append(record)
    pd.DataFrame(aggregate_rows).to_csv(replay_root / "summary_by_condition.csv", index=False)

    overall = {
        "runs_total": int(len(df)),
        "runs_completed": int(len(completed)),
        "runs_failed": int(len(df) - len(completed)),
        "safety_success_rate": float(pd.to_numeric(completed["crazyflow_safety_success"]).mean()),
        "mission_success_rate": float(pd.to_numeric(completed["crazyflow_mission_success"]).mean()),
        "collision_pairs_total": int(pd.to_numeric(completed["crazyflow_collision_pairs"], errors="coerce").fillna(0).sum()),
        "obstacle_contacts_total": int(pd.to_numeric(completed["crazyflow_obstacle_contacts"], errors="coerce").fillna(0).sum()),
        "min_reached_fraction": float(pd.to_numeric(completed["crazyflow_final_reached_fraction"], errors="coerce").min()),
        "min_inter_agent_distance_m": float(pd.to_numeric(completed["crazyflow_min_inter_agent_distance_m"], errors="coerce").min()),
        "max_tracking_error_m": float(pd.to_numeric(completed["crazyflow_max_tracking_error_m"], errors="coerce").max()),
        "median_T90_s": _finite_median(completed["T90_s"]),
        "median_T95_s": _finite_median(completed["T95_s"]),
        "median_T99_s": _finite_median(completed["T99_s"]),
        "T99_reached_runs": int(pd.to_numeric(completed["T99_s"], errors="coerce").notna().sum()),
    }
    pd.DataFrame([overall]).to_csv(replay_root / "summary_overall.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the proposed-controller full sweep through Crazyflow replay.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ijars_matched_120s_all_controllers.yaml")
    parser.add_argument("--sources", type=Path, default=ROOT / "results" / "crazyflow_proposed_full_sources")
    parser.add_argument("--replays", type=Path, default=ROOT / "results" / "crazyflow_replay")
    parser.add_argument("--crazyflow-python", type=Path, default=Path("/home/dongoo/tmp/conda_envs/crazyflow/bin/python"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--freq", type=int, default=500)
    parser.add_argument("--state-freq", type=int, default=100)
    parser.add_argument("--log-every", type=float, default=0.5)
    parser.add_argument("--source-trajectory-stride", type=int, default=1)
    parser.add_argument("--source-workers", type=int, default=1, help="Parallel workers for source trajectory generation.")
    parser.add_argument("--replay-workers", type=int, default=1, help="Parallel workers for Crazyflow replay subprocesses.")
    parser.add_argument("--replay-trajectory-stride", type=int, default=100)
    parser.add_argument("--comm-update-rate-hz", type=float, default=None, help="Override communication update rate for all source runs.")
    parser.add_argument("--local-safety-update-rate-hz", type=float, default=None, help="Override local-safety sensing update rate for all source runs.")
    parser.add_argument("--obstacle-update-rate-hz", type=float, default=None, help="Override obstacle-perception update rate for all source runs.")
    parser.add_argument("--downwash-strength", type=float, default=1.0)
    parser.add_argument("--downwash-force-cap-ratio", type=float, default=0.6)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--scenario", action="append", help="Optional scenario filter; can be repeated.")
    parser.add_argument("--n-agents", action="append", type=int, help="Optional agent-count filter; can be repeated.")
    parser.add_argument("--block", action="append", help="Optional block filter; can be repeated.")
    parser.add_argument("--overwrite-source", action="store_true")
    parser.add_argument("--overwrite-replay", action="store_true")
    parser.add_argument("--source-only", action="store_true", help="Generate missing source trajectories and exit before Crazyflow replay.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    source_root = args.sources.resolve()
    replay_root = args.replays.resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    replay_root.mkdir(parents=True, exist_ok=True)

    specs = _select_specs(args)
    print(f"Selected {len(specs)} proposed runs from {args.config}")
    print(f"Source trajectories: {source_root}")
    print(f"Crazyflow replays:  {replay_root}")
    print(f"Device: {args.device}; downwash: on; source stride: {args.source_trajectory_stride}")

    if args.source_workers > 1:
        pending_specs = [
            spec
            for spec in specs
            if args.overwrite_source or not _source_ready(source_root / make_run_id(spec)) or not _trajectory_npz_ready(source_root / make_run_id(spec))
        ]
        print(f"Pre-generating {len(pending_specs)} missing source trajectories with {args.source_workers} workers")
        if pending_specs:
            tasks = [
                (spec, str(source_root), args.source_trajectory_stride, args.overwrite_source)
                for spec in pending_specs
            ]
            with ProcessPoolExecutor(max_workers=args.source_workers) as executor:
                futures = [executor.submit(_generate_source_task, task) for task in tasks]
                for count, future in enumerate(as_completed(futures), start=1):
                    run_id, metrics = future.result()
                    print(
                        f"  source [{count:03d}/{len(futures):03d}] {run_id} | "
                        f"reached={float(metrics.get('final_reached_fraction', 0.0)):.3f} "
                        f"coll={metrics.get('cumulative_collision_pairs')} "
                        f"obs={metrics.get('cumulative_obstacle_contacts')}"
                    )

    if args.source_only:
        print("Source-only mode complete; skipping Crazyflow replay.")
        return

    if args.replay_workers > 1:
        missing_sources = [
            spec
            for spec in specs
            if args.overwrite_source
            or not _source_ready(source_root / make_run_id(spec))
            or not _trajectory_npz_ready(source_root / make_run_id(spec))
        ]
        if missing_sources:
            print(f"Generating {len(missing_sources)} missing source trajectories before parallel replay")
            for count, spec in enumerate(missing_sources, start=1):
                run_id = make_run_id(spec)
                metrics = _generate_source(spec, source_root, args.source_trajectory_stride, args.overwrite_source)
                print(
                    f"  source [{count:03d}/{len(missing_sources):03d}] {run_id} | "
                    f"reached={float(metrics.get('final_reached_fraction', 0.0)):.3f} "
                    f"coll={metrics.get('cumulative_collision_pairs')} "
                    f"obs={metrics.get('cumulative_obstacle_contacts')}"
                )

        rows: list[dict[str, Any]] = []
        pending_specs = []
        for spec in specs:
            replay_dir = replay_root / make_run_id(spec)
            if _replay_ready(replay_dir) and not args.overwrite_replay:
                rows.append(_summarize_run(spec, source_root, replay_root, True, None))
            else:
                pending_specs.append(spec)
        _write_summaries(rows, replay_root)
        print(
            f"Replay ready: {len(rows)}; pending: {len(pending_specs)}; "
            f"workers: {args.replay_workers}"
        )

        if pending_specs:
            tasks = [
                (spec, str(source_root), str(replay_root), args)
                for spec in pending_specs
            ]
            with ProcessPoolExecutor(max_workers=args.replay_workers) as executor:
                futures = {
                    executor.submit(_run_replay_task, task): spec
                    for task, spec in zip(tasks, pending_specs, strict=True)
                }
                for count, future in enumerate(as_completed(futures), start=1):
                    spec = futures[future]
                    run_id = make_run_id(spec)
                    try:
                        replay_ok, failure_log, elapsed = future.result()
                    except Exception as exc:  # noqa: BLE001 - keep long sweeps resumable.
                        replay_ok = False
                        failure_log = None
                        elapsed = math.nan
                        print(f"  replay [{count:03d}/{len(pending_specs):03d}] {run_id} failed: {exc}")
                    row = _summarize_run(spec, source_root, replay_root, replay_ok, failure_log)
                    rows.append(row)
                    _write_summaries(rows, replay_root)
                    if replay_ok:
                        print(
                            f"  replay [{count:03d}/{len(pending_specs):03d}] {run_id} | "
                            f"safe={row.get('crazyflow_safety_success')} "
                            f"reached={float(row.get('crazyflow_final_reached_fraction', 0.0)):.3f} "
                            f"coll={row.get('crazyflow_collision_pairs')} "
                            f"obs={row.get('crazyflow_obstacle_contacts')} "
                            f"T99={row.get('T99_s')} "
                            f"max_err={float(row.get('crazyflow_max_tracking_error_m', 0.0)):.3f} m "
                            f"rtf={float(row.get('crazyflow_real_time_factor', 0.0)):.2f} "
                            f"elapsed={elapsed:.1f}s"
                        )
                    elif failure_log:
                        print(f"  replay [{count:03d}/{len(pending_specs):03d}] {run_id} failed; log={failure_log}")
                    if not replay_ok and args.stop_on_error:
                        break

        _write_summaries(rows, replay_root)
        print(f"Wrote run index to {replay_root / 'run_index.csv'}")
        if (replay_root / "summary_overall.csv").exists():
            print(f"Wrote overall summary to {replay_root / 'summary_overall.csv'}")
        return

    rows: list[dict[str, Any]] = []
    completed_before = 0
    for index, spec in enumerate(specs, start=1):
        run_id = make_run_id(spec)
        run_start = time.perf_counter()
        print(f"[{index:03d}/{len(specs):03d}] {run_id}")
        try:
            source_metrics = _generate_source(spec, source_root, args.source_trajectory_stride, args.overwrite_source)
            print(
                "  source | "
                f"reached={float(source_metrics.get('final_reached_fraction', 0.0)):.3f} "
                f"coll={source_metrics.get('cumulative_collision_pairs')} "
                f"obs={source_metrics.get('cumulative_obstacle_contacts')}"
            )
            replay_ok, failure_log = _run_replay(args, spec, source_root, replay_root)
            row = _summarize_run(spec, source_root, replay_root, replay_ok, failure_log)
            rows.append(row)
            _write_summaries(rows, replay_root)
            if replay_ok:
                completed_before += 1
                print(
                    "  replay | "
                    f"safe={row.get('crazyflow_safety_success')} "
                    f"reached={float(row.get('crazyflow_final_reached_fraction', 0.0)):.3f} "
                    f"coll={row.get('crazyflow_collision_pairs')} "
                    f"obs={row.get('crazyflow_obstacle_contacts')} "
                    f"T99={row.get('T99_s')} "
                    f"max_err={float(row.get('crazyflow_max_tracking_error_m', 0.0)):.3f} m "
                    f"rtf={float(row.get('crazyflow_real_time_factor', 0.0)):.2f}"
                )
            else:
                print(f"  replay failed; log={failure_log}")
                if args.stop_on_error:
                    break
        except Exception as exc:  # noqa: BLE001 - keep long sweeps resumable.
            failure_log = None
            rows.append(_summarize_run(spec, source_root, replay_root, False, failure_log))
            _write_summaries(rows, replay_root)
            print(f"  run failed: {exc}")
            if args.stop_on_error:
                raise
        elapsed = time.perf_counter() - run_start
        print(f"  elapsed={elapsed:.1f}s")

    _write_summaries(rows, replay_root)
    print(f"Wrote run index to {replay_root / 'run_index.csv'}")
    if (replay_root / "summary_overall.csv").exists():
        print(f"Wrote overall summary to {replay_root / 'summary_overall.csv'}")


if __name__ == "__main__":
    main()
