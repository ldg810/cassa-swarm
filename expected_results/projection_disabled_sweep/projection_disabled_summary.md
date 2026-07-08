# Projection-Disabled CASSA Sweep Summary

Projection-disabled run: 255 runs, seeds 0-4, with command-level obstacle braking, local safety, and speed governors enabled while post-integration projection correction is disabled.

Projection-ON timing reference columns use the live manuscript-consistent regression values. Projection-count reference columns use the published manuscript/packaged reference values, because the live ON regression adds 205 merge-split obstacle adjustments.

Note: `expected_results/proposed_full_sweep` is stale for merge-split timing (T99=52.354 s there versus 60.625 s in live ON and 60.6 s in the manuscript table). It is still consistent with the manuscript projection-audit count for merge-split (0 adjustments).

| Group | OFF safety_success_rate | ON ref safety_success_rate | OFF inter_agent_collision_pair_samples | ON ref inter_agent_collision_pair_samples | OFF obstacle_contact_samples | ON ref obstacle_contact_samples | OFF T90_median_s | ON ref T90_median_s | OFF T95_median_s | ON ref T95_median_s | OFF T99_median_s | ON ref T99_median_s | OFF mean_reached_fraction | ON ref mean_reached_fraction | OFF clearance_envelope_violation_steps_total | ON ref clearance_envelope_violation_steps_total | OFF max_conservative_envelope_penetration_m | ON ref max_conservative_envelope_penetration_m | OFF min_obstacle_contact_boundary_margin_m | ON ref min_obstacle_contact_boundary_margin_m | OFF min_inter_agent_distance_m | ON ref min_inter_agent_distance_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 0.996 | 1.000 | 2 | 0 | 0 | 0 | 59.929 | 59.929 | 61.875 | 61.875 | 66.000 | 66.000 | 1.000 | 1.000 | 137,235 | 3,580 | 0.145 | 0.043 | 0.215 | 0.317 | 0.108 | 0.140 |
| Obstacle corridor | 0.992 | 1.000 | 2 | 0 | 0 | 0 | 68.700 | 68.750 | 71.500 | 71.500 | 77.000 | 76.500 | 0.999 | 0.999 | 127,183 | 3,481 | 0.145 | 0.043 | 0.215 | 0.317 | 0.108 | 0.140 |
| Crossing traffic | 1.000 | 1.000 | 0 | 0 | 0 | 0 | 54.676 | 54.688 | 55.650 | 55.725 | 58.250 | 58.250 | 1.000 | 1.000 | 3,224 | 99 | 0.137 | 0.036 | 0.223 | 0.324 | 0.215 | 0.220 |
| Merge--split | 1.000 | 1.000 | 0 | 0 | 0 | 0 | 58.208 | 58.281 | 58.938 | 59.104 | 60.292 | 60.625 | 1.000 | 1.000 | 6,828 | 0 | 0.138 | 0.000 | 0.222 | 0.321 | 0.240 | 0.221 |
| Open flock | 1.000 | 1.000 | 0 | 0 | 0 | 0 | 51.725 | 51.725 | 52.300 | 52.300 | 53.708 | 53.708 | 1.000 | 1.000 | 0 | 0 | 0.000 | 0.000 | NA | NA | 0.254 | 0.254 |

## Projection Counters

| Group | OFF obstacle adj. | ON ref obstacle adj. | OFF pairwise adj. | ON ref pairwise adj. |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0 | 3,571 | 0 | 18 |
| Obstacle corridor | 0 | 3,472 | 0 | 18 |
| Crossing traffic | 0 | 99 | 0 | 0 |
| Merge--split | 0 | 0 | 0 | 0 |
| Open flock | 0 | 0 | 0 | 0 |

## Files

- `run_index.csv`: raw per-run metrics from the projection-disabled sweep.
- `run_index_with_thresholds.csv`: per-run metrics plus interpolated T90/T95/T99 crossing times.
- `projection_disabled_summary.csv`: wide aggregate/per-scenario OFF vs ON-reference comparison.
- `projection_disabled_summary_long.csv`: long-form metric table for paper/table reshaping.
