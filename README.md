# cassa-swarm

This repository contains the simulation code, experiment configurations, compact reference outputs, and manuscript assets for:

**CASSA: A Communication-Aware Safety Stack for Large-Scale Aerial Swarm Simulation under Degraded Nominal Communication**

The repository is intended to support reproduction of the simulation results reported in the IJARS manuscript. It is a standalone release derived from the working research tree and includes only the reported experiment configurations, post-processing scripts, current paper tables, Crazyflow replay interface, and demonstration animations.

## Scope

CASSA is evaluated here as a simulator-level swarm safety stack. The reported results use 4 Hz nominal communication, 4 Hz local relative-state safety sampling, and 4 Hz obstacle-information updates. The final CASSA settings are:

- `safe_radius_m=0.39`
- `obstacle_clearance_margin_m=0.34`
- `obstacle_command_filter_buffer_m=0.15`
- `obstacle_command_filter_lookahead_margin_m=0.90`
- `enable_obstacle_projection=true`

This release also includes a projection-disabled audit sweep. In that audit,
the command-level obstacle braking filter, local safety shield, and speed
governors remain enabled, while only the post-integration state-space
projection fallback is disabled with `metadata.projection_enabled=false`.

Only experiments reported in the current manuscript are retained in this release.

## Repository Contents

| Path | Description |
| --- | --- |
| `src/mavswarm/` | Double-integrator swarm simulator, controllers, scenarios, and metrics |
| `configs/` | Sweep definitions used to reproduce the reported results |
| `experiments/` | Batch execution, analysis, table generation, plotting, and Crazyflow replay utilities |
| `paper_assets/` | Current manuscript tables, LaTeX source snapshot, references, and method/metric summary |
| `expected_results/` | Compact reference summaries from the current manuscript run |
| `demo/` | Pre-rendered 1000-agent trajectory animations |
| `external/` | RVO2 C API wrapper and license files used by the external baseline |

Full raw result directories are not included because they are large and can be regenerated from the provided configurations.

## Demonstrations

1000-MAV crossing traffic:

![1000-MAV crossing traffic](demo/n1000_crossing.gif)

1000-MAV obstacle corridor:

![1000-MAV obstacle corridor](demo/n1000_obstacle.gif)

To regenerate the GIFs:

```bash
python scripts/reproduce_paper_results.py demo-gifs --workers 4 --overwrite
```

## Installation

```bash
cd cassa-swarm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The RVO2 package baseline uses `external/librvo2_c_api.so`, which is included for Linux x86_64. To rebuild the wrapper, place the official RVO2 source tree at `external/RVO2` or set `RVO2_ROOT`, then run:

```bash
bash scripts/build_rvo2_wrapper.sh
```

## Quick Validation

```bash
python scripts/reproduce_paper_results.py smoke --workers 2
```

The smoke target runs two CASSA cases and two external-baseline cases, then verifies the CASSA summary post-processing path. This is intended as an installation and execution check, not as a replacement for the full sweeps.

## Reproducing Double-Integrator Results

```bash
python scripts/reproduce_paper_results.py double-integrator --workers 16
python scripts/reproduce_paper_results.py projection-disabled --workers 16
python scripts/reproduce_paper_results.py tables
```

The non-Crazyflow manuscript results are generated from the following configurations:

| Config | Output directory | Purpose |
| --- | --- | --- |
| `configs/proposed_full_sweep.yaml` | `results/proposed_full_sweep` | CASSA full sweep, threshold times, and projection telemetry |
| `configs/projection_disabled_full_sweep.yaml` | `results/projection_disabled_sweep` | 255-run CASSA audit with only post-integration projection fallback disabled |
| `configs/classical_baselines_full_sweep.yaml` | `results/classical_baselines_full_sweep` | Boids, APF, no-shield, and matched CASSA controller rows |
| `configs/secondary_baselines_full_sweep.yaml` | `results/secondary_baselines_full_sweep` | Short-horizon predictive, MADER-inspired, ORCA/RVO, and internal CBF-QP surrogate rows |
| `configs/external_package_baselines_full_sweep.yaml` | `results/external_package_baselines_full_sweep` | CBF-QP via OSQP and RVO2 package baseline rows |
| `configs/seed_extension_120s.yaml` | `results/seed_extension_120s` | Additional hard-condition seeds |
| `configs/trajectory_samples.yaml` | `results/trajectory_samples` | Saved trajectories for manuscript trajectory figures |

Generated tables are written to `paper_assets/tables/`. Generated figures are written to `paper_assets/figures/`.

## Reproducing Crazyflow Replay

Crazyflow replay evaluates CASSA reference trajectories using Crazyflie-class rotor-drag dynamics and downwash. Install Crazyflow in a separate environment:

```bash
git clone https://github.com/ldg810/crazyflow.git external/crazyflow
python -m venv .venv-crazyflow
source .venv-crazyflow/bin/activate
pip install -e external/crazyflow
```

Then run:

```bash
CRAZYFLOW_PYTHON=.venv-crazyflow/bin/python \
python scripts/reproduce_paper_results.py crazyflow --workers 8 --crazyflow-workers 1 --crazyflow-device gpu
python scripts/reproduce_paper_results.py tables
```

The Crazyflow source/replay config is `configs/crazyflow_full_sweep.yaml`. Replay summaries are written to `results/crazyflow_replay/`.

## Reference Outputs

Compact reference summaries from the manuscript version are stored in `expected_results/`:

| Path | Contents |
| --- | --- |
| `expected_results/proposed_full_sweep/` | CASSA 4 Hz full-sweep summary JSON/CSV used for projection telemetry |
| `expected_results/projection_disabled_sweep/` | Projection-disabled 255-run audit run index and OFF-vs-ON-reference summaries |
| `expected_results/local_sensitivity_sweep/` | Local-safety stream sensitivity mini-sweep for the N=200 obstacle-corridor condition |
| `expected_results/crazyflow_replay/` | Crazyflow 255-run summary and replay table reference |
| `paper_assets/tables/` | Current manuscript table CSV/LaTeX files |
| `paper_assets/methodology_metrics_summary.md` | Objective definitions of the main methods and reported metrics |

Floating-point values may vary slightly across CPU, BLAS, OSQP, and JAX/Crazyflow versions. For the same configs and seeds, safety counts, reached-fraction thresholds, and controller ordering are expected to remain stable.

## Direct Commands

The wrapper script is provided for convenience. Individual stages can also be run directly:

```bash
python -m experiments.run_batch --config configs/proposed_full_sweep.yaml --out results/proposed_full_sweep --workers 16
python -m experiments.summarize_proposed_full --results results/proposed_full_sweep
python -m experiments.run_batch --config configs/projection_disabled_full_sweep.yaml --out results/projection_disabled_sweep --workers 16
python -m experiments.summarize_projection_disabled --results results/projection_disabled_sweep --on-reference-results results/proposed_full_sweep
python -m experiments.run_batch --config configs/classical_baselines_full_sweep.yaml --out results/classical_baselines_full_sweep --workers 16
python -m experiments.run_batch --config configs/secondary_baselines_full_sweep.yaml --out results/secondary_baselines_full_sweep --workers 16
python -m experiments.run_batch --config configs/external_package_baselines_full_sweep.yaml --out results/external_package_baselines_full_sweep --workers 16
python -m experiments.run_batch --config configs/seed_extension_120s.yaml --out results/seed_extension_120s --workers 16
python -m experiments.make_paper_tables
```

## Citation

If this repository is used before publication, cite the accompanying manuscript and include the commit hash of this repository in the citation note.
