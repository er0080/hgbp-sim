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
* **setpoints** (saturation temperatures, the RGT, the speed), **tracking errors** and
  clipped **integrated errors** (kelvin);
* **context**: swept volume, nominal speed and eight physical refrigerant descriptors;
* **status** of the compressor start/stop interlock.

One environment instance is one refrigerant; run one instance per fluid to train across
refrigerants. The suction-line volume flow the stand's meter reads is in `env.meas["Vdot"]`
(not part of the observation).

## Action

Four values in `[-1, 1]` for valves 1-4, plus an optional fifth run request
(`start_stop_action=True`).
* `action_mode="incremental"` (default): each valve command moves by `a * max_rate`.
* `action_mode="absolute"`: the action maps to the command directly.

An interlock state machine (OFF / STARTING / RUNNING / STOPPING, with start permissives
and anti-short-cycle timers) has authority over the compressor. Without the run action the
compressor is started automatically and stopped after the last point.

## Episodes

* **Test points:** 1-3 per episode, as the test procedure gives them: saturated suction
  and discharge temperatures, a return gas (suction) temperature (RGT) and a VFD
  frequency. They are sampled from `Envelope`:
  * evaporating and condensing temperature inside the compressor's operating envelope
    (`Envelope.polygon`, the manufacturer's R410A map);
  * RGT of 65 °F for 30 % of the points, otherwise -10 to 35 °C, with the superheat it
    implies between 5 and 50 K;
  * 35-75 Hz (the nominal speed is taken as 60 Hz).
* **Liquid pressure setpoint:** the geometric mean of the suction and discharge pressures,
  but at least the saturation pressure 6 K above the cooling water inlet (40 °F, down to
  38 °F: `T_wi_range`).
* **Feasibility:** points that fail the filters are resampled. The filters are an estimated
  discharge temperature, and an equilibrium at nominal charge with every valve below 95 %
  and a margin to the trips (10 K on discharge temperature, 1 bar on the pressures). Warm
  starts need the same margins at their starting state.
* **Completion:** the stand is stable once suction and discharge saturation temperature and
  RGT stay inside the tolerance band (0.3 K, 0.2 K, 0.5 K) for `dwell_required` seconds.
  * With `n_collections = 0` (default) the point is then done.
  * With `n_collections > 0` (the stand's procedure is 3 x 15 min: `n_collections=3`,
    `collection_time=900`) the stand must stay stable for that many collections. Leaving
    the band loses the collection in progress; it restarts once the stand is stable again.
    `info` reports `stable`, `collecting`, `collected`, `collection_lost` and
    `collections_done`.
  * A point also ends when its maximum hold time runs out. The collections come on top of
    `hold_time`; raise `episode_time` to fit them.
  * After the last point the compressor must be stopped, and the episode ends successfully.
* **Given schedules:** `env.reset(idx, schedule=dict(points=..., k=..., hold=...))` replaces
  the sampled points of those stands with a procedure's own list (for example a rating
  schedule in lift order), up to `max_points` (default 4) per episode.
  `Envelope.point(props, T_evap, T_cond, RGT, N, T_wi)` builds a point from saturated
  suction and discharge temperatures, RGT (°C) and speed (rpm), with the clamped liquid
  pressure. Given points skip the feasibility filter. The rest of the episode
  (parameters, charge, ambient, water, start) is drawn as usual.
* **Start modes:** `warm` (equilibrium at the first point or another one, from the
  steady-state solver), `cold` (equalized stand at ambient, compressor off, optionally
  some liquid migrated to the suction side) or `random`.
* **Stand definition:** `EnvConfig(stand="webui/config/stand_defaults.json")` takes the
  plant parameters and the baseline loops' UT35A settings (tuning, output limits, PV
  filters) from a defaults file, the same document the web UI reads. Without it, the
  built-in parameters and gains apply.
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
| return gas (suction) temperature | 3 |
| intermediate (liquid) pressure | 4 |

With `EnvConfig(stand=...)` the loops use the stand's UT35A settings from the defaults
file; otherwise the built-in gains below.

The built-in suction temperature loop is slow on purpose (integral time 60 s): the suction temperature
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
| 16 | about 4 000 |
| 256 | about 14 500 (3 600 simulated seconds per second) |
| 1024 | about 19 500 |

* Run one `HGBPVecEnv` with many stands per process, not many processes with one stand
  each.
* The thread pool uses all cores by default. With several training processes on one
  machine, give each a share with `NUMBA_NUM_THREADS` (for example 4).
* `hgbp_sim` selects numba's `workqueue` threading layer unless `NUMBA_THREADING_LAYER` is
  set: the OpenMP layer keeps idle threads spinning, which slows the Python code between
  steps. The workqueue layer must not be used from several Python threads at once.
* Resets solve the steady state for warm starts and check the test points' feasibility;
  that costs tens of milliseconds per stand.
