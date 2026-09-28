# Training environment

`hgbp_sim.env` provides the stand as a control task: `HGBPVecEnv` runs a batch of stands
(recommended for training), `HGBPEnv` wraps one stand as a `gymnasium.Env`. Settings are
fields of `EnvConfig`. The learned controller built on this interface is specified in
[controller-spec.md](controller-spec.md).

```python
from hgbp_sim import HGBPVecEnv, EnvConfig

env = HGBPVecEnv(256, EnvConfig(), seed=0)
obs = env.reset()                                  # (256, 51)
obs, reward, terminated, truncated, info = env.step(env.expert_action())
```

## Observation

51 values, scaled to order one (names in `hgbp_sim.OBS_NAMES`, groups in `OBS_GROUPS`):

* **measurements**, refrigerant-agnostic: pressures as saturation temperatures,
  superheat, subcooling, mass flow normalized by swept volume × speed × suction vapor
  density, power normalized by swept volume × speed × suction pressure, temperatures,
  speed, valve positions;
* **setpoints**, **tracking errors** and clipped **integrated errors** (kelvin);
* **context**: swept volume, nominal speed and eight physical refrigerant descriptors;
* **status** of the compressor start/stop interlock.

One environment instance is one refrigerant; run one instance per fluid to train across
refrigerants.

## Action

Four values in `[-1, 1]` for valves 1-4, plus an optional fifth run request
(`start_stop_action=True`).
* `action_mode="incremental"` (default): each valve command moves by `a * max_rate`.
* `action_mode="absolute"`: the action maps to the command directly.

An interlock state machine (OFF / STARTING / RUNNING / STOPPING, with start permissives
and anti-short-cycle timers) has authority over the compressor. Without the run action the
compressor is started automatically and stopped after the last point.

## Episodes

* **Test points:** 1-3 per episode, sampled from `Envelope`: evaporating -30 to 12 °C,
  condensing 30 to 57 °C, superheat 3 to 25 K, intermediate temperature between the
  cooling water and the condensing temperature, 50-140 % speed. Points that fail the
  feasibility filters (estimated discharge temperature, equilibrium at nominal charge) are
  resampled.
* **Completion:** a point is done once suction and discharge saturation temperature and
  superheat stay inside the tolerance band (0.3 K, 0.2 K, 0.5 K) for `dwell_required`
  seconds, or when its maximum hold time runs out. After the last point the compressor
  must be stopped, and the episode ends successfully.
* **Start modes:** `warm` (equilibrium at the first point or another one, from the
  steady-state solver), `cold` (equalized stand at ambient, compressor off, optionally
  some liquid migrated to the suction side) or `random`.
* **Randomized per episode:** plant parameters, ambient and water temperature, the charge
  (`charge_range`, plus a share `p_charge_extreme` of episodes in
  `charge_extreme_range`) and the cold-start liquid distribution.

## Reward

Per step (weights in `EnvConfig`):
* minus the normalized tracking errors in kelvin (intermediate pressure at half weight);
* a bonus inside the tolerance band and when a point completes;
* penalties for valve movement, liquid at the compressor inlet, approaching the discharge
  temperature limit, idling when a start is possible, blocked start requests and running
  during shutdown;
* a bonus for a completed shutdown, and a large penalty plus termination on a trip.

## Labels and the expert

`info` carries noise-free process values, the true charge factor and a `steady` flag
(labels for a charge estimator), privileged states for an asymmetric critic, and every
reward component. `env.expert_action()` returns the baseline controller's action in the
environment's action space, for behaviour cloning, DAgger or reward shaping.

## Baseline controller

`BaselineController` (`hgbp_sim/control.py`) is four PI loops:

| controlled variable | valve |
|---|---|
| discharge pressure | 1 |
| suction pressure | 2 |
| superheat | 3 |
| intermediate pressure | 4 |

The superheat loop is slow on purpose (integral time 60 s): the suction temperature
responds to valve 3 over about a minute, because the mixing exchanger's plates have to
change temperature, and a faster loop limit-cycles into floodback. Every valve moves
suction pressure, discharge pressure and superheat together
(`examples/open_loop_step.py`), which is the coupling a coordinated controller can
exploit.

## Throughput

A batch runs in parallel on numba's thread pool. On an 8-core laptop
(`examples/benchmark.py`):

| stands | environment steps per second |
|---|---|
| 16 | about 4 700 |
| 256 | about 16 000 (4 000 simulated seconds per second) |
| 1024 | about 21 000 |

* Run one `HGBPVecEnv` with many stands per process, not many processes with one stand
  each.
* The thread pool uses all cores by default. With several training processes on one
  machine, give each a share with `NUMBA_NUM_THREADS` (for example 4).
* `hgbp_sim` selects numba's `workqueue` threading layer unless `NUMBA_THREADING_LAYER` is
  set: the OpenMP layer keeps idle threads spinning, which slows the Python code between
  steps. The workqueue layer must not be used from several Python threads at once.
* Resets solve the steady state for warm starts and check the test points' feasibility;
  that costs tens of milliseconds per stand.
