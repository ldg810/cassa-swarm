from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str | Path]) -> None:
    text = " ".join(str(part) for part in cmd)
    print(f"\n$ {text}", flush=True)
    subprocess.check_call([str(part) for part in cmd], cwd=ROOT)


def run_batch(config: str, out: str, workers: int, overwrite: bool, max_runs: int | None = None) -> None:
    cmd: list[str | Path] = [
        sys.executable,
        "-m",
        "experiments.run_batch",
        "--config",
        ROOT / "configs" / config,
        "--out",
        ROOT / "results" / out,
        "--workers",
        str(workers),
    ]
    if overwrite:
        cmd.append("--overwrite")
    if max_runs is not None:
        cmd += ["--max-runs", str(max_runs)]
    run(cmd)


def run_smoke(workers: int) -> None:
    run_batch("proposed_full_sweep.yaml", "smoke_proposed", workers, overwrite=True, max_runs=2)
    run_batch("external_package_baselines_full_sweep.yaml", "smoke_external_baselines", workers, overwrite=True, max_runs=2)
    run([sys.executable, "-m", "experiments.summarize_proposed_full", "--results", ROOT / "results" / "smoke_proposed"])


def run_double_integrator(workers: int, overwrite: bool) -> None:
    run_batch("proposed_full_sweep.yaml", "proposed_full_sweep", workers, overwrite)
    run([sys.executable, "-m", "experiments.summarize_proposed_full", "--results", ROOT / "results" / "proposed_full_sweep"])
    run_batch("classical_baselines_full_sweep.yaml", "classical_baselines_full_sweep", workers, overwrite)
    run_batch("secondary_baselines_full_sweep.yaml", "secondary_baselines_full_sweep", workers, overwrite)
    run_batch("external_package_baselines_full_sweep.yaml", "external_package_baselines_full_sweep", workers, overwrite)
    run_batch("seed_extension_120s.yaml", "seed_extension_120s", workers, overwrite)


def run_projection_disabled(workers: int, overwrite: bool) -> None:
    run_batch("projection_disabled_full_sweep.yaml", "projection_disabled_sweep", workers, overwrite)
    run([
        sys.executable,
        "-m",
        "experiments.summarize_projection_disabled",
        "--results",
        ROOT / "results" / "projection_disabled_sweep",
    ])


def run_tables_and_figures() -> None:
    proposed_summary = ROOT / "results" / "proposed_full_sweep" / "summary"
    classical_summary = ROOT / "results" / "classical_baselines_full_sweep" / "summary"
    secondary_summary = ROOT / "results" / "secondary_baselines_full_sweep" / "summary"
    external_summary = ROOT / "results" / "external_package_baselines_full_sweep" / "summary"
    run([sys.executable, "-m", "experiments.analyze_results", "--results", ROOT / "results" / "proposed_full_sweep", "--out", proposed_summary])
    run([sys.executable, "-m", "experiments.analyze_results", "--results", ROOT / "results" / "classical_baselines_full_sweep", "--out", classical_summary])
    run([sys.executable, "-m", "experiments.analyze_results", "--results", ROOT / "results" / "secondary_baselines_full_sweep", "--out", secondary_summary])
    run([sys.executable, "-m", "experiments.analyze_results", "--results", ROOT / "results" / "external_package_baselines_full_sweep", "--out", external_summary])
    run([
        sys.executable,
        "-m",
        "experiments.plot_results",
        "--summary",
        proposed_summary / "summary_by_condition.csv",
        "--comparison-summary",
        classical_summary / "summary_by_condition.csv",
        "--comparison-summary",
        secondary_summary / "summary_by_condition.csv",
        "--comparison-summary",
        external_summary / "summary_by_condition.csv",
        "--results",
        ROOT / "results" / "proposed_full_sweep",
        "--trajectory-results",
        ROOT / "results" / "demo_n1000_trajectories",
        "--out",
        ROOT / "paper_assets" / "figures",
    ])
    run([
        sys.executable,
        "-m",
        "experiments.make_paper_tables",
        "--main-results",
        ROOT / "results" / "classical_baselines_full_sweep",
        "--proposed-results",
        ROOT / "results" / "proposed_full_sweep",
        "--stronger-baseline-results",
        ROOT / "results" / "secondary_baselines_full_sweep",
        "--external-baseline-results",
        ROOT / "results" / "external_package_baselines_full_sweep",
        "--seed-extension-results",
        ROOT / "results" / "seed_extension_120s",
        "--crazyflow-results",
        ROOT / "results" / "crazyflow_replay",
        "--out",
        ROOT / "paper_assets" / "tables",
    ])


def run_demo_gifs(workers: int, overwrite: bool) -> None:
    run_batch("demo_n1000_trajectories.yaml", "demo_n1000_trajectories", workers, overwrite)
    demo_root = ROOT / "results" / "demo_n1000_trajectories"
    commands = [
        (
            demo_root / "crossing_traffic__proposed__N1000__pl00__lat000__seed0",
            ROOT / "demo" / "n1000_crossing.gif",
            ROOT / "demo" / "n1000_crossing_midframe.png",
        ),
        (
            demo_root / "corridor_obstacles__proposed__N1000__pl00__lat000__seed0",
            ROOT / "demo" / "n1000_obstacle.gif",
            ROOT / "demo" / "n1000_obstacle_midframe.png",
        ),
    ]
    for run_dir, gif_path, preview_path in commands:
        run([
            sys.executable,
            ROOT / "scripts" / "render_trajectory_gif.py",
            run_dir,
            "--output",
            gif_path,
            "--preview",
            preview_path,
            "--fps",
            "18",
            "--max-frames",
            "180",
            "--trail-s",
            "8.0",
            "--trail-max-agents",
            "180",
            "--dpi",
            "105",
        ])


def run_crazyflow(args: argparse.Namespace) -> None:
    crazyflow_python = Path(args.crazyflow_python or os.environ.get("CRAZYFLOW_PYTHON", sys.executable))
    cmd: list[str | Path] = [
        sys.executable,
        "-m",
        "experiments.run_crazyflow_proposed_full",
        "--config",
        ROOT / "configs" / "crazyflow_full_sweep.yaml",
        "--sources",
        ROOT / "results" / "crazyflow_sources",
        "--replays",
        ROOT / "results" / "crazyflow_replay",
        "--crazyflow-python",
        crazyflow_python,
        "--device",
        args.crazyflow_device,
        "--source-workers",
        str(args.workers),
        "--replay-workers",
        str(args.crazyflow_workers),
    ]
    if args.overwrite:
        cmd += ["--overwrite-source", "--overwrite-replay"]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce IJARS paper simulations, tables, figures, and demos.")
    parser.add_argument(
        "target",
        choices=["smoke", "double-integrator", "projection-disabled", "tables", "demo-gifs", "crazyflow", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-crazyflow", action="store_true", help="Also run Crazyflow when target=all.")
    parser.add_argument("--crazyflow-python", default=None, help="Python executable inside a Crazyflow environment.")
    parser.add_argument("--crazyflow-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--crazyflow-workers", type=int, default=1)
    args = parser.parse_args()

    if args.target == "smoke":
        run_smoke(args.workers)
    elif args.target == "double-integrator":
        run_double_integrator(args.workers, args.overwrite)
    elif args.target == "projection-disabled":
        run_projection_disabled(args.workers, args.overwrite)
    elif args.target == "tables":
        run_tables_and_figures()
    elif args.target == "demo-gifs":
        run_demo_gifs(args.workers, args.overwrite)
    elif args.target == "crazyflow":
        run_crazyflow(args)
    elif args.target == "all":
        run_double_integrator(args.workers, args.overwrite)
        run_demo_gifs(args.workers, args.overwrite)
        if args.include_crazyflow:
            run_crazyflow(args)
        run_tables_and_figures()


if __name__ == "__main__":
    main()
