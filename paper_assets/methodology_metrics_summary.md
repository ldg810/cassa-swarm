# CASSA 방법론 및 결과 지표 정리

이 문서는 현재 원고에서 사용된 방법론, 실험 설정, 결과 지표, 결과 표 수치를 정리한다. 포함된 내용은 원고에 명시된 모델, 제어 구조, 실험 조건, 지표 정의, 수치 결과에 한정한다.

## 1. 연구 범위와 정보 스트림

### 1.1 연구 범위

- CASSA는 `Communication-Aware Swarm Safety Architecture`의 약어이다.
- 원고는 CASSA를 단일 충돌 회피 primitive 또는 formal safety certificate가 아니라 architecture-level simulation study로 정의한다.
- 주 실험의 대상은 nominal communication degradation 조건에서 simulator-level collision/contact safety와 mission progress가 유지되는지이다.
- 주 실험은 local perception stack 전체를 평가하지 않는다.
- 주 실험은 local relative-state safety stream을 이상화된 interface로 유지한다.
- 주 실험에서 nominal communication stream에는 packet loss, latency, finite communication radius, neighbor cap, communicated-state noise가 적용된다.
- 주 실험에서 local relative-state safety stream에는 local dropout, estimator delay, relative-state noise, field-of-view limits, occlusion, missed detection, sensor saturation이 적용되지 않는다.
- 주 실험에서 obstacle information stream은 4 Hz로 갱신되는 known cylindrical obstacle geometry와 clearance envelope를 사용한다.
- Safety result는 implemented simulator, stated sensing assumptions, retained projection fallback, bounded double-integrator model, Crazyflow replay model 조건에 한정된다.

### 1.2 정보 스트림

| Information stream | 사용 모듈 | 주 실험에서 모델링된 요소 | 결과 해석 범위 |
| --- | --- | --- | --- |
| Nominal communication | Flocking, alignment, cohesion, nominal neighbor prediction | 4 Hz updates, 4.0 m radius, 12-neighbor cap, packet loss, latency, communicated-state noise | Nominal-communication degradation evaluation |
| Local relative-state safety | Pairwise shield, density governor, crossing-flow governor | 4 Hz sampling, finite 0.85 m safety influence radius | Idealized close-range safety interface 조건의 controller stack evaluation |
| Obstacle information | Command-level obstacle filter | 4 Hz update, known cylindrical obstacle geometry, clearance envelope | Modeled static-obstacle clearance |

### 1.3 Assumption 1

Assumption 1은 `idealized local relative-state safety interface for factor-isolated evaluation`이다.

- 각 agent는 safety influence radius 내부의 nearby agents에 대한 short-range all-around relative-state stream에 접근한다고 가정한다.
- 이 stream은 4 Hz로 sampled된다.
- 이 stream은 local safety shield와 speed governors에서만 사용된다.
- 주 실험에서 다음 항목은 모델링하지 않는다.
  - local-safety dropout
  - estimator delay
  - relative-state noise
  - field-of-view limits
  - occlusion
  - missed detections
  - sensor saturation
- 이 stream은 특정 lidar, vision, UWB, multi-sensor perception stack의 sensor model이 아니라 controller에 제공되는 state-estimation interface로 취급된다.

### 1.4 Nominal-communication robustness 정의

원고에서 nominal-communication robustness는 다음 조건에서 reported simulator-level collision/contact metrics와 completion metrics가 유지되는지를 뜻한다.

- nominal neighbor-coordination stream degraded by packet loss
- latency
- finite communication radius
- neighbor cap
- communicated-state noise

이 정의는 separate short-range local relative-state safety stream degradation에 대한 robustness를 포함하지 않는다.

## 2. Agent 모델

### 2.1 상태 및 동역학

각 aerial robot은 3D bounded double integrator로 모델링된다.

$$
\dot{\mathbf{p}}_i = \mathbf{v}_i
$$

$$
\dot{\mathbf{v}}_i = \mathbf{u}_i
$$

변수 정의:

| 기호 | 의미 |
| --- | --- |
| $\mathbf{p}_i \in \mathbb{R}^3$ | agent $i$의 position |
| $\mathbf{v}_i \in \mathbb{R}^3$ | agent $i$의 velocity |
| $\mathbf{u}_i \in \mathbb{R}^3$ | agent $i$의 commanded acceleration |

### 2.2 기본 제한값

| 항목 | 값 |
| --- | --- |
| Speed limit | 1.5 m/s |
| Acceleration limit | 3.0 m/s^2 |
| Integration step | 0.05 s |
| Primary simulator mass label | 0.027 kg |
| Collision radius | 0.12 m |
| Safe radius | 0.39 m |
| Safety influence radius | 0.85 m |

### 2.3 Safety와 mission completion 분리

- Safety success는 run 전체에서 inter-agent collision pairs와 obstacle contacts가 모두 0일 때 성립한다.
- Mission completion은 time-to-threshold로 보고된다.
- 단일 operational pass/fail 기준이 필요한 경우, 원고는 safety success와 선택된 completion threshold 도달 여부의 conjunction으로 정의한다.

## 3. CASSA Architecture

CASSA는 5-stage decentralized pipeline으로 구성된다.

### 3.1 Stage 1: Communication-aware nominal coordination

각 agent는 delayed, lossy, neighbor-limited communication observations로부터 다음 acceleration terms를 계산한다.

- goal acceleration
- separation
- alignment
- cohesion
- obstacle repulsion
- boundary repulsion
- damping

Delayed neighbor states는 prediction이 enabled된 경우 delayed velocity로 forward propagation된다.

### 3.2 Stage 2: Local safety shielding

- Short-range relative-state observations를 사용한다.
- 이 observations는 communication packet-loss model과 분리되어 있다.
- Distance penalty와 closing-speed penalty를 nominal acceleration에 추가한다.

### 3.3 Stage 3: Crossing-flow pairwise speed governor

- Different crossing-flow groups에 속한 nearby closing pairs에 대해 target-speed reduction을 적용한다.
- Pre-integration 단계에서 residual pairwise closing velocity를 줄이는 데 사용된다.
- Crossing-traffic scenario에서 asymmetric priority rule이 활성화된다.

### 3.4 Stage 4: Density-aware speed regulation

- Dense neighborhood에서 target speed를 감소시킨다.
- Local-safety neighbor set 기반으로 density, nearest-neighbor distance, closing-speed factors를 계산한다.

### 3.5 Stage 5: Command-level obstacle filter

- State integration 전에 obstacle-directed acceleration을 braking-distance constraint로 필터링한다.
- Cylindrical clearance envelope를 사용한다.
- Retained projection fallback은 telemetry로 기록된다.

## 4. Nominal Controller

### 4.1 Communicated-neighbor set

$\mathcal{N}^{\mathrm{comm}}_i$는 다음 조건 적용 후 agent $i$가 사용하는 communicated-neighbor set이다.

- communication radius
- maximum neighbor count
- latency
- packet loss
- communicated-state noise

Communicated neighbor $j$에 대해:

$$
d_{ij}=\|\mathbf{p}_i-\hat{\mathbf{p}}_j\|
$$

$$
\mathbf{n}_{ij}=\frac{\mathbf{p}_i-\hat{\mathbf{p}}_j}{d_{ij}+\epsilon}
$$

### 4.2 Flocking terms

Separation:

$$
\mathbf{s}_i=
\sum_{j\in\mathcal{N}^{\mathrm{comm}}_i}
\frac{\max(0,r_{\mathrm{inf}}-d_{ij})}{d_{ij}+0.05}\mathbf{n}_{ij}
$$

Alignment:

$$
\mathbf{a}_i =
\frac{1}{n_i^{\mathrm{comm}}}
\sum_{j\in\mathcal{N}^{\mathrm{comm}}_i}\hat{\mathbf{v}}_j-\mathbf{v}_i
$$

Cohesion:

$$
\mathbf{c}_i =
\frac{1}{n_i^{\mathrm{comm}}}
\sum_{j\in\mathcal{N}^{\mathrm{comm}}_i}\hat{\mathbf{p}}_j-\mathbf{p}_i
$$

No communicated neighbor인 경우 alignment와 cohesion은 0으로 설정된다.

### 4.3 Goal term

Goal term은 velocity-tracking acceleration이다.

$$
\mathbf{g}_i=\mathbf{v}^{g}_i-\mathbf{v}_i
$$

$\mathbf{v}^{g}_i$는 현재 control goal 방향이며 속도는 다음으로 제한된다.

$$
\min(v^{\mathrm{tar}}_i,0.7\|\mathbf{g}^{\mathrm{pos}}_i-\mathbf{p}_i\|)
$$

### 4.4 Obstacle repulsion term

Cylindrical obstacle에 대해 signed horizontal distance를 다음과 같이 정의한다.

$$
q_{io}=d_{io}^{xy}-r_o
$$

Obstacle influence distance:

$$
\rho_o=1.2+m_{\mathrm{obs}}
$$

Horizontal obstacle term은 $q_{io}<\rho_o$일 때 활성화된다.

$$
\mathbf{o}^{xy}_i =
\mathbf{o}^{xy}_i+\eta_{io}\mathbf{n}^{o}_{i}
$$

$$
\eta_{io}=k_{io}/h_{io}
$$

$$
h_{io}=\max(q_{io}+0.08,0.08)
$$

$k_{io}$는 implementation에서 사용하는 squared normalized clearance deficit이다.

### 4.5 Boundary term

- Arena face로부터 0.55 m 이내에서 linear repulsion이 적용된다.
- Small velocity damping component가 포함된다.

### 4.6 Nominal command

Nominal acceleration command:

$$
\mathbf{u}^{\mathrm{nom}}_i =
w_s \mathbf{s}_i + w_a \mathbf{a}_i + w_c \mathbf{c}_i +
w_g \mathbf{g}_i + w_o \mathbf{o}_i + w_b \mathbf{b}_i -
w_d \mathbf{v}_i
$$

Weights:

| Term | Weight |
| --- | ---: |
| Separation | 1.5 |
| Alignment | 0.3 |
| Cohesion | 0.08 |
| Goal | 5.0 |
| Obstacle | 2.5 |
| Boundary | 1.1 |
| Damping | 0.22 |

### 4.7 Delayed neighbor prediction

Prediction이 enabled된 경우 delayed neighbor position은 delayed velocity로 forward propagation된다.

$$
\hat{\mathbf{p}}_j(t) =
\mathbf{p}_j(t-\tau) + \mathbf{v}_j(t-\tau)\tau
$$

이 prediction은 nominal coordination에 사용된다. Safety-critical close-range avoidance는 separate local observation model을 사용한다.

## 5. Local Safety Shield 및 Speed Governors

### 5.1 Local safety shield

Local relative-state safety abstraction은 $\mathcal{N}^{\mathrm{safe}}_i$를 사용한다.

$$
\Delta \mathbf{u}_i =
\gamma \sum_{j \in \mathcal{N}^{\mathrm{safe}}_i}
\left[
w_{\mathrm{safe}}
\frac{\max(0,r_{\mathrm{inf}} - d_{ij})}{r_{\mathrm{inf}} - r_{\mathrm{safe}}}
+
w_{\mathrm{close}} \max(0,-\mathbf{v}_{ij}^{T}\mathbf{n}_{ij})
\right]\mathbf{n}_{ij}
$$

변수 정의:

| 기호 | 의미 |
| --- | --- |
| $d_{ij}$ | $\|\mathbf{p}_i-\mathbf{p}_j\|$ |
| $\mathbf{n}_{ij}$ | $(\mathbf{p}_i-\mathbf{p}_j)/d_{ij}$ |
| $\mathbf{v}_{ij}$ | $\mathbf{v}_i-\mathbf{v}_j$ |
| $\gamma$ | density-dependent safety gain |
| $w_{\mathrm{safe}}$ | 2.5 |
| $w_{\mathrm{close}}$ | 1.0 |

Commanded acceleration:

$$
\mathbf{u}_i = \mathrm{sat}_{u_{\max}}\left(\mathbf{u}^{\mathrm{nom}}_i + \Delta \mathbf{u}_i\right)
$$

### 5.2 Density-aware target-speed factor

Target speed:

$$
v^{\mathrm{tar}}_i = v_{\max} f_i
$$

Target-speed factor:

$$
f_i =
\max(f_{\min},\min(1,f_{\mathrm{near}},f_{\mathrm{dense}},f_{\mathrm{close}},f_{\mathrm{obs}},f_{\mathrm{time}}))
$$

Density-aware governor parameterization:

| Factor | Condition | Value |
| --- | --- | ---: |
| $f_{\mathrm{near}}$ | nearest neighbor within $r_{\mathrm{safe}}+0.03$ m | 0.35 |
| $f_{\mathrm{near}}$ | nearest neighbor within $r_{\mathrm{safe}}+0.12$ m | 0.60 |
| $f_{\mathrm{dense}}$ | at least 8 neighbors inside safety influence radius | 0.82 |
| $f_{\mathrm{dense}}$ | at least 16 neighbors inside safety influence radius | 0.72 |
| $f_{\mathrm{close}}$ | maximum closing speed exceeds 0.55 m/s within $r_{\mathrm{safe}}+0.20$ m | 0.65 |

Additional factors:

- $f_{\mathrm{obs}}$는 next-step radial obstacle clearance가 conservative obstacle envelope에 접근할 때 감소한다.
- $f_{\mathrm{time}}$는 terminal progress relaxation으로 사용된다.

### 5.3 Crossing-flow pairwise speed governor

| 항목 | 값 또는 정의 |
| --- | --- |
| Influence radius | 1.20 m |
| Clearance scale | 0.42 m |
| Closing-speed trigger | 0.03 m/s |
| Through-flow speed floor | 0.82 |
| Transverse-flow speed floor | 0.28 |

Flow group assignment:

- Start-to-goal displacement의 absolute y-axis component가 x-axis component보다 큰 agents는 transverse flow로 분류된다.
- 나머지 agents는 through-flow로 분류된다.
- Asymmetric priority rule은 crossing-traffic scenario에서만 활성화된다.
- Mixed dominant axes가 없거나 crossing-traffic scenario가 아닌 경우 asymmetric priority rule은 비활성화된다.

## 6. Command-level Obstacle Filter

### 6.1 Clearance envelope

Cylindrical obstacle에 대해 conservative radial clearance는 다음과 같다.

$$
r^{\mathrm{obs}}_{\mathrm{clr}} =
r_{\mathrm{obs}} + r_{\mathrm{coll}} + \epsilon_{\mathrm{obs}} + m_{\mathrm{obs}}
$$

| 항목 | 값 |
| --- | ---: |
| $r_{\mathrm{obs}}$ | obstacle radius |
| $r_{\mathrm{coll}}$ | 0.12 m |
| $\epsilon_{\mathrm{obs}}$ | 0.02 m |
| $m_{\mathrm{obs}}$ | 0.34 m |
| Command-filter buffer $b_o$ | 0.15 m |
| Lookahead margin | 0.90 m |

### 6.2 Radial quantities

| 기호 | 정의 |
| --- | --- |
| $d_o$ | horizontal distance from obstacle center |
| $v_r$ | $\mathbf{n}_o^T \mathbf{v}_{xy}$ |
| $a_r$ | $\mathbf{n}_o^T \mathbf{u}_{xy}$ |
| $\mathbf{n}_o$ | outward obstacle normal |

### 6.3 Allowed inward speed

$$
v^{\mathrm{in}}_{\max} =
\sqrt{2 u_{\max}\max(d_o-r^{\mathrm{obs}}_{\mathrm{clr}}-b_o,0)}
$$

### 6.4 Next-step radial-speed condition

Command는 다음 조건을 만족하도록 조정된다.

$$
v_r + a_r \Delta t \geq -v^{\mathrm{in}}_{\max}
$$

### 6.5 Obstacle braking filter procedure

Inputs:

- current position $\mathbf{p}_i$
- current velocity $\mathbf{v}_i$
- nominal acceleration $\mathbf{u}^{\mathrm{nom}}_i$
- cylindrical obstacles
- $\Delta t=0.05$ s
- $u_{\max}=3.0$ m/s^2
- lookahead 0.90 m
- buffer 0.15 m
- margin 0.34 m
- allowance 0.02 m

Steps:

1. Nominal acceleration을 global acceleration bound로 saturate한다.
2. Agent의 height interval에 포함되는 obstacle별로 horizontal outward normal, radial distance, radial speed, radial acceleration을 계산한다.
3. Agent가 $r^{\mathrm{obs}}_{\mathrm{clr}}$ 주변 0.90 m lookahead band 밖에 있으면 해당 obstacle check를 skip한다.
4. Remaining buffered braking distance로 $v^{\mathrm{in}}_{\max}$를 계산한다.
5. $v_r+a_r\Delta t\geq -v^{\mathrm{in}}_{\max}$를 요구한다.
6. Next-step radial speed가 허용 범위보다 inward 방향이면 missing outward radial acceleration component만 추가한다.
7. Final acceleration을 $u_{\max}$로 다시 saturate하고 bounded double-integrator update에 전달한다.

### 6.6 Projection fallback

- Obstacle or pairwise constraints가 final acceleration saturation 이후 서로 경쟁하는 경우 retained state-space projection fallback이 사용될 수 있다.
- Projection fallback adjustment는 simulator telemetry로 기록된다.
- Final 4 Hz CASSA full sweep에서 기록된 projection telemetry:
  - obstacle-projection agent adjustments: 3,571
  - pairwise-projection agent adjustments: 18
  - post-integration/pre-projection obstacle contacts: 0
  - conservative pre-projection clearance-violation steps: 3,571

## 7. Simulation Setup

### 7.1 Scenarios

| Scenario | Description |
| --- | --- |
| Open flock | Basic scalable goal-directed motion |
| Obstacle corridor | 21 cylindrical obstacles arranged in three lateral rows |
| Crossing traffic | Two sub-swarms pass through a congested intersection |
| Merge-split | Three groups cross through a shared central gate and split toward separate exits |

### 7.2 Run set

| 항목 | 값 |
| --- | ---: |
| Primary CASSA aggregate runs | 255 |
| Seeds used in aggregate rows | 0-4 |
| Reserved seed | 99 |
| Reserved seed use | Figure 5 trajectory samples only |
| Reserved seed included in aggregate statistics | No |
| Horizon | 120 s |
| Integration step | 0.05 s |
| Nominal communication update rate | 4 Hz |
| Local relative-state safety stream sampling | 4 Hz |
| Obstacle information update rate | 4 Hz |

Scenario coverage:

| Scenario | Runs |
| --- | ---: |
| Obstacle corridor | 125 |
| Crossing traffic | 90 |
| Merge-split | 20 |
| Open flock | 20 |

Scale coverage:

| Agents | Runs |
| ---: | ---: |
| 50 | 20 |
| 100 | 90 |
| 200 | 90 |
| 500 | 35 |
| 1000 | 20 |

### 7.3 Arena and mission settings

| 항목 | 값 |
| --- | --- |
| Arena x range | [-42, 42] m |
| Arena y range | [-42, 42] m |
| Arena z range | [0.3, 2.5] m |
| Altitude bounds | 0.4-2.4 m |
| Goal tolerance | 0.75 m |
| Completed agents | Remain in physical traffic |
| Terminal slot spacing | 0.50 m |

### 7.4 Communication settings

| 항목 | 값 |
| --- | --- |
| Communication radius | 4.0 m |
| Neighbor cap | 12 |
| Packet loss probabilities | 0.0, 0.1, 0.2, 0.4 |
| Latency values | 0, 50, 100, 200 ms |
| Communicated position noise | 0.015 m |
| Communicated velocity noise | 0.025 m/s |

### 7.5 Sensing settings

| 항목 | 값 |
| --- | --- |
| Local relative-state safety stream sampling | 4 Hz |
| Obstacle information update | 4 Hz |
| Local dropout | 0 |
| Local estimator delay | 0 |
| Local relative-state noise | 0 |
| Field-of-view model | Not modeled |
| Occlusion model | Not modeled |
| Missed-detection model | Not modeled |
| Sensor-saturation model | Not modeled |

## 8. Controller Rows

### 8.1 Controller set

The matched controller set includes:

| Controller row | Description in manuscript |
| --- | --- |
| CASSA (Proposed) | Complete CASSA controller |
| No shield | CASSA variant without local safety shield |
| Boids | Reactive flocking baseline |
| APF | Artificial potential field baseline |
| Short-horizon predictive surrogate | Short-horizon predictive surrogate using goal tracking, closest-approach pairwise penalties, and obstacle penalties |
| MADER-inspired surrogate | Predictive surrogate with longer pairwise and obstacle lookahead, larger clearance margins, and lower goal aggressiveness |
| CBF-QP (OSQP package) | Three-dimensional acceleration-projection CBF-QP solved through OSQP |
| RVO2 package | RVO2 C++ reciprocal-velocity library through a wrapper |

### 8.2 Matched simulator-interface conditions

Every controller row receives:

- same goal set
- same initial conditions
- same obstacle map
- same arena bounds
- 4 Hz communication updates
- 4 Hz obstacle-information updates
- 0.05 s integration step
- 1.5 m/s speed limit
- 3.0 m/s^2 acceleration limit
- 4.0 m communication radius
- 12-neighbor cap
- packet-loss/latency/noise model
- same seeds
- 120 s horizon
- completed agents remain in physical traffic

Controller-specific conditions:

- CASSA uses the nominal communication model plus modeled local relative-state safety stream, target-speed governors, command-level obstacle filter, and reported projection fallback.
- Baseline rows retain their native sensing and safety structure.
- Baseline rows are not given CASSA's local relative-state safety stream, crossing-flow speed governor, density-aware target-speed regulator, or command-level obstacle filter.
- OSQP row starts from common goal/obstacle/boundary nominal acceleration and solves acceleration-projection CBF-QP with active pairwise, cylindrical-obstacle, and boundary half-space constraints.
- If OSQP does not return solved status, the implementation falls back to the same half-space projection routine.
- RVO2 row sends current horizontal positions, horizontal velocities, and preferred velocities to the RVO2 C++ simulator at each control step.
- RVO2 altitude, cylindrical-obstacle, and boundary terms are supplied by the common simulator interface.

## 9. Result Metrics

### 9.1 Primary metrics

| Metric | Definition |
| --- | --- |
| Safety success | Run-level binary success. Requires zero inter-agent collision pairs and zero obstacle contacts over the run. |
| T90 | First time when at least 90% of agents reached terminal goal tolerance. |
| T95 | First time when at least 95% of agents reached terminal goal tolerance. |
| T99 | First time when at least 99% of agents reached terminal goal tolerance. |
| Horizon-censored threshold | If threshold is not reached before configured horizon, threshold time is reported as horizon-censored. |

### 9.2 Supporting metrics

| Metric | Definition |
| --- | --- |
| Final reached fraction | Fraction of agents that reached terminal goal tolerance by the end of the run. |
| Mean reached fraction | Mean final reached fraction across runs in a row. |
| Min reached fraction | Minimum final reached fraction across runs in a row. |
| Inter-agent collisions | Cumulative inter-agent collision-pair samples under the 0.12 m collision radius. |
| Obstacle contacts | Cumulative agent-time-step samples inside obstacle contact/clearance region. |
| Minimum inter-agent distance | Minimum pairwise distance recorded in a run or aggregate. |
| Safety-violation rate | Reported support metric associated with safety-radius violations. |
| Acceleration-squared energy proxy | Sum or aggregate based on squared acceleration commands. |
| Mean messages per agent per update | Communication load metric. |
| Runtime per agent-step | Computational cost metric in ms per agent-step. |
| Completion time | Time to reach completion thresholds or terminal completion state. |

### 9.3 Command-filter telemetry metrics

| Metric | Definition |
| --- | --- |
| Obstacle projection adjustments | Retained post-integration state-space correction agent adjustments associated with obstacles. |
| Pairwise projection adjustments | Retained post-integration state-space correction agent adjustments associated with pairwise constraints. |
| Pre-projection obstacle contacts | Post-integration/pre-projection obstacle-contact samples. |
| Clearance violation steps | Conservative pre-projection clearance-violation agent steps. |

### 9.4 Crazyflow replay metrics

| Metric | Definition |
| --- | --- |
| Safety success | Replay-level collision/contact safety success. |
| T99 median | Median time to 99% reached fraction in replay. |
| T99 max | Maximum T99 in replay group. |
| Collision pairs | Inter-agent collision-pair samples in Crazyflow replay. |
| Obstacle contacts | Obstacle-contact samples in Crazyflow replay. |
| Max tracking error | Maximum tracking error in Crazyflow replay. |
| Max downwash force | Maximum downwash force in Crazyflow replay. |

## 10. Matched Controller Comparison Results

Table values correspond to the matched 120 s controller comparison under 4 Hz updates.

| Controller | Runs | Safety success | T90 | T95 | T99 | Mean reached fraction | Inter-agent collisions | Obstacle contacts |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| CASSA (Proposed) | 255 | 1.00 | 59.9 (255/255) | 61.9 (255/255) | 66.0 (255/255) | 1.000 | 0 | 0 |
| Short-horizon predictive surrogate | 255 | 0.89 | 88.0 (165/255) | 91.5 (155/255) | 98.5 (151/255) | 0.877 | 2,053 | 0 |
| CBF-QP (OSQP pkg.) | 255 | 0.45 | 71.5 (255/255) | 73.5 (255/255) | 77.0 (255/255) | 1.000 | 30,052 | 5,479 |
| RVO2 package | 255 | 0.24 | 75.5 (229/255) | 80.5 (210/255) | 90.0 (206/255) | 0.961 | 0 | 897,116 |
| MADER-inspired surrogate | 255 | 0.92 | >120 (59/255) | >120 (45/255) | >120 (19/255) | 0.502 | 772 | 0 |
| APF | 255 | 0.54 | 73.2 (255/255) | 75.3 (255/255) | 79.2 (242/255) | 0.998 | 4,973 | 997 |
| Boids | 255 | 0.65 | 83.8 (255/255) | 85.8 (247/255) | 90.5 (196/255) | 0.992 | 2,833 | 886 |
| No shield | 255 | 0.07 | 55.0 (255/255) | 55.8 (255/255) | 57.0 (255/255) | 1.000 | 11,616 | 293,456 |

Additional reported confidence interval:

- Observed CASSA safety-success rate: 255/255
- Exact two-sided 95% Clopper-Pearson interval: approximately 0.986-1.000

## 11. Scenario-wise CASSA Results

Table values correspond to scenario-wise CASSA performance under 4 Hz updates.

| Scenario | Runs | Safety success | T90 | T95 | T99 | Min reached fraction | Mean reached fraction |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| Obstacle corridor | 125 | 1.00 | 68.8 (125/125) | 71.5 (125/125) | 76.5 (125/125) | 0.992 | 0.999 |
| Crossing traffic | 90 | 1.00 | 54.7 (90/90) | 55.7 (90/90) | 58.2 (90/90) | 0.998 | 1.000 |
| Merge-split | 20 | 1.00 | 58.3 (20/20) | 59.1 (20/20) | 60.6 (20/20) | 1.000 | 1.000 |
| Open flock | 20 | 1.00 | 51.7 (20/20) | 52.3 (20/20) | 53.7 (20/20) | 0.998 | 1.000 |

## 12. Nominal Communication Packet Loss and Latency Results

### 12.1 Packet-loss response

Figure 3 setting:

- Scenario: representative N=200 obstacle-corridor scenario
- Fixed latency: 50 ms
- Packet loss is applied to nominal communication stream only.
- Local relative-state safety stream remains available under Assumption 1.
- Solid-line zero-loss point uses fixed 50 ms latency and is not the nominal no-loss/no-latency baseline.
- Dashed horizontal lines show nominal medians for the same representative N=200 corridor condition.

Reported values in text:

| Condition | Reported T99 range or value |
| --- | ---: |
| Nominal no-loss/no-latency N=200 corridor | approximately 66.3 s |
| Fixed-50 ms latency packet-loss sweep | approximately 75.5-77.0 s |

### 12.2 Latency response

Figure 4 setting:

- Scenario: representative obstacle-corridor scenario
- Metric: T90/T95/T99 time-to-threshold
- Latency is applied to nominal communication stream only.
- Local relative-state safety observations follow Assumption 1.
- Lines show medians.
- Shaded bands show interquartile ranges across runs at each latency value.

## 13. Runtime Scaling Result

Figure 2 setting:

- Metric: milliseconds per agent-step
- Controller families: matched reactive rows, CASSA-ablation rows, predictive-surrogate rows, external-package baseline rows
- Condition: nominal no-loss/no-latency obstacle-corridor condition
- All plotted controller families include 1000-agent points.
- Lines show means.
- Shaded bands show 95% confidence intervals where replicate data are available.

## 14. Command-filter and Projection Telemetry Results

Table values correspond to final 4 Hz CASSA full sweep.

| Group | Runs | Safety success | Obstacle projection adjustments | Pairwise projection adjustments | Pre-projection obstacle contacts | Clearance violation steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 255 | 1.00 | 3,571 | 18 | 0 | 3,571 |
| Obstacle corridor | 125 | 1.00 | 3,472 | 18 | 0 | 3,472 |
| Crossing traffic | 90 | 1.00 | 99 | 0 | 0 | 99 |
| Merge-split | 20 | 1.00 | 0 | 0 | 0 | 0 |
| Open flock | 20 | 1.00 | 0 | 0 | 0 | 0 |

## 15. Crazyflow Replay Method and Results

### 15.1 Crazyflow replay setup

- Reference set: 255 CASSA reference trajectories
- Replay simulator: Crazyflow
- Rotor-drag dynamics: enabled
- Downwash: enabled
- Additional trajectory correction during replay: none
- Reference generator information model: same separated 4 Hz information model as primary sweep
- Replay reference generation obstacle margin: 0.34 m
- Command-filter buffer: 0.15 m
- Lookahead margin: 0.90 m
- Obstacle projection during reference generation: enabled
- Crazyflow model: `cf2x_L250`
- Physics model: `so_rpy_rotor_drag`
- Crazyflow mass parameterization: 31.9 g
- Primary double-integrator mass label: 0.027 kg

### 15.2 Crazyflow replay results

| Group | Runs | Safety success | T99 median (s) | T99 max (s) | Collision pairs | Obstacle contacts | Max tracking error (m) | Max downwash force (N) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 255 | 1.00 | 66.0 | 104.0 | 0 | 0 | 0.217 | 0.083 |
| N=50 | 20 | 1.00 | 56.3 | 63.4 | 0 | 0 | 0.184 | 0.029 |
| N=100 | 90 | 1.00 | 59.3 | 79.0 | 0 | 0 | 0.177 | 0.047 |
| N=200 | 90 | 1.00 | 61.9 | 82.5 | 0 | 0 | 0.187 | 0.058 |
| N=500 | 35 | 1.00 | 84.0 | 89.3 | 0 | 0 | 0.198 | 0.070 |
| N=1000 | 20 | 1.00 | 101.6 | 104.0 | 0 | 0 | 0.217 | 0.083 |

## 16. Reported Figure Contents

| Figure | Content | Metrics |
| --- | --- | --- |
| Figure 1 | Obstacle-corridor scalability of CASSA by swarm size | T90, T95, T99 medians and interquartile ranges |
| Figure 2 | Runtime scaling comparison | ms per agent-step means and 95% confidence intervals where replicate data are available |
| Figure 3 | Packet-loss response at fixed 50 ms latency in representative N=200 obstacle-corridor scenario | T90, T95, T99 vs packet-loss probability |
| Figure 4 | Nominal-communication latency response in representative obstacle-corridor scenario | T90, T95, T99 vs latency |
| Figure 5 | Representative trajectories for obstacle-corridor, crossing-traffic, and merge-split scenarios | 3D view and top-down view of 100-agent sample |
| Figure 6 | Crazyflie 2.1+ platform and Crazyflow `cf2x_L250` replay parameters | Platform image and replay parameter summary |

## 17. Result Interpretation Boundaries Stated in the Manuscript

The following boundaries are stated as part of the manuscript's methodology and metrics definition.

- Safety success is defined by zero collision/contact events under the implemented simulator metric.
- Collision/contact safety is distinct from strict safe-radius clearance preservation.
- Inter-agent collision radius is 0.12 m.
- Safe radius is 0.39 m and is used as a local-safety influence scale, not as the binary success threshold.
- Obstacle contacts are cumulative agent-time-step samples, not unique contact episodes.
- Projection fallback is enabled in the final 4 Hz CASSA sweep.
- Projection adjustments are reported as telemetry.
- Local relative-state safety stream is not degraded by dropout, estimator delay, noise, field-of-view limits, occlusion, missed detections, or sensor saturation in the main sweep.
- Packet-loss and latency sweeps degrade nominal communication stream only.
- Crazyflow replay is performed for CASSA reference set, not for all baseline controller rows.
- Crazyflow replay uses Crazyflow model and stated information abstraction.
