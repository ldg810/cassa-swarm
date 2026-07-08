# Local-Safety Stream Sensitivity Mini-Sweep

Representative condition: obstacle corridor, N=200, proposed CASSA, seeds 0-4.

Nominal communication backdrop is fixed at packet loss=0.2, latency=50 ms, 12-neighbor cap, 4 Hz communication, and the standard communicated-state noise.

Only the inter-agent local relative-state stream feeding the local pairwise shield and active speed governor path is degraded. Obstacle perception remains idealized at the primary-sweep setting, and post-integration projection fallback remains enabled.

| Condition | Safety success | Collision-pair samples | Obstacle contacts | Min d_ij (m) | Min obstacle boundary margin (m) | T90 | T95 | T99 | Mean reached | Obs proj adj | Pair proj adj | Clearance violations | Empirical dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| idealized_baseline | 1.000 | 0 | 0 | 0.273 | 0.360 | 69.000 (5/5) | 71.750 (5/5) | 77.000 (5/5) | 1.000 | 24 | 0 | 24 | 0.000 |
| dropout_0p05 | 1.000 | 0 | 0 | 0.246 | 0.360 | 68.500 (5/5) | 72.000 (5/5) | 76.500 (5/5) | 1.000 | 30 | 0 | 30 | 0.051 |
| dropout_0p10 | 1.000 | 0 | 0 | 0.164 | 0.360 | 69.500 (5/5) | 72.333 (5/5) | 76.000 (5/5) | 1.000 | 30 | 0 | 30 | 0.100 |
| dropout_0p20 | 1.000 | 0 | 0 | 0.162 | 0.360 | 69.333 (5/5) | 71.750 (5/5) | 78.000 (5/5) | 1.000 | 30 | 0 | 30 | 0.199 |
| dropout_0p40 | 1.000 | 0 | 0 | 0.140 | 0.360 | 70.250 (5/5) | 73.000 (5/5) | 80.500 (5/5) | 1.000 | 43 | 10 | 48 | 0.400 |
| latency_050ms | 1.000 | 0 | 0 | 0.244 | 0.360 | 69.625 (5/5) | 72.500 (5/5) | 78.500 (5/5) | 1.000 | 33 | 0 | 33 | 0.000 |
| latency_100ms | 1.000 | 0 | 0 | 0.264 | 0.360 | 70.250 (5/5) | 73.000 (5/5) | 78.500 (5/5) | 0.999 | 25 | 0 | 25 | 0.000 |
| latency_250ms | 1.000 | 0 | 0 | 0.209 | 0.360 | 72.333 (5/5) | 75.250 (5/5) | 80.000 (5/5) | 1.000 | 21 | 0 | 21 | 0.000 |
| noise_0p005_0p010 | 1.000 | 0 | 0 | 0.253 | 0.360 | 69.000 (5/5) | 71.500 (5/5) | 77.750 (5/5) | 1.000 | 28 | 0 | 28 | 0.000 |
| noise_0p010_0p020 | 1.000 | 0 | 0 | 0.234 | 0.360 | 69.000 (5/5) | 72.000 (5/5) | 79.500 (5/5) | 1.000 | 23 | 0 | 23 | 0.000 |
| noise_0p020_0p040 | 1.000 | 0 | 0 | 0.278 | 0.360 | 69.000 (5/5) | 71.500 (5/5) | 76.000 (5/5) | 1.000 | 25 | 0 | 25 | 0.000 |
| noise_0p050_0p100 | 1.000 | 0 | 0 | 0.234 | 0.360 | 68.667 (5/5) | 71.375 (5/5) | 77.500 (5/5) | 1.000 | 30 | 0 | 30 | 0.000 |
| combined_moderate | 1.000 | 0 | 0 | 0.172 | 0.360 | 70.875 (5/5) | 74.250 (5/5) | 79.000 (5/5) | 1.000 | 23 | 0 | 23 | 0.099 |
| combined_severe | 1.000 | 0 | 0 | 0.179 | 0.360 | 72.500 (5/5) | 75.700 (5/5) | 81.500 (5/5) | 1.000 | 41 | 0 | 41 | 0.200 |

## Notes

- Baseline row keeps all local-stream degradation flags at zero and is the regression-gate replay of the matching primary-sweep condition.
- Local dropout is per-agent per 4 Hz local-safety sample and uses zero-order hold of the previous local sample; dropped neighbors are not treated as absent.
- Latency is rounded to integer 0.05 s integration steps; 50/100/250 ms correspond to 1/2/5 steps.
- Empirical dropout is the mean realized zero-order-hold rate across the five seeds for each condition.
- Obstacle perception is not degraded in this experiment; it remains the idealized obstacle-information stream from the primary sweep.
- Projection fallback is enabled, and obstacle/pairwise projection adjustment totals are reported to expose any increased reliance.
