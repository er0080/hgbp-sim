# Development

## Setup and tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m hgbp_sim.kernel.warmup      # compile the model once (1-2 min)
pytest                                # about 1 min
```

## The compiled model

The model runs as numba-compiled code in `hgbp_sim/kernel/`. Everything else is Python.

* **Compile cache.** Numba compiles on first use and caches the machine code next to the
  sources (`__pycache__`). Only the first run after installing or after changing the kernel
  pays the compile time; `python -m hgbp_sim.kernel.warmup` does it ahead (the Docker
  image does it at build time). Numba only notices changes to a function's own source
  file: after editing `kernel/model.py`, the cached steady-state solver (`kernel/steady.py`)
  still runs the old model. Delete the `*.nbi` / `*.nbc` files in `hgbp_sim/kernel/__pycache__`
  after a kernel change.
* **Scalar code per stand.** Kernel functions work on one stand; the `*_batch` functions
  loop over a batch, in a serial and a parallel build. A single stand runs serially.
* **Adding physics.** Model code goes into the kernel (`kernel/model.py`), as scalar code
  that numba can compile. Parameters reach it as one record per stand
  (`kernel/layout.py`), so a new `PlantParams` field is available there by name.
* **Units.** The model and the library are in SI throughout. The web UI converts for
  display only.

## Reference data

`tests/test_reference.py` checks the model against `tests/reference/reference.npz`:
* the right-hand side and outputs at recorded states (to rounding);
* trajectories of six scenarios: cold start, test points, speed step, charge recovery,
  floodback, trip, and a randomized batch;
* steady-state solves.

After an intentional change of the physics, regenerate it:

```bash
python tests/reference/make_reference.py
```

## Performance

`examples/benchmark.py` measures the live stand, the training environment and the
steady-state solver. On an 8-core laptop (Xeon W-10885M):

| | |
|---|---|
| live stand (one stand with its PID loops) | about 350 × real time |
| training environment, 256 / 1024 stands | about 14 500 / 19 500 environment steps / s |
| creating or resetting 256 stands | about 2-2.5 s each |
| steady-state solve, one stand | 25-30 ms |

Where the time goes:
* **Live stand.** About 0.55 ms per 0.2 s control step: a third in the model, the rest
  in the Python around it (loops, interlock, measurement, history). The web backend builds
  a snapshot only when it sends one (`LiveStand.step(snapshot=False)`).
* **Batches.** The model itself. A right-hand side evaluation costs about 9 µs on one
  core; RK4 takes four per sub-step, usually one sub-step per 0.05 s.
* **Steady-state solves.** The Levenberg-Marquardt Jacobian (finite differences, two
  model evaluations per column for the receiver pool) and the mixing exchanger's
  pseudo-time march. A point without an equilibrium is given up once its residual stops
  moving.

## Property tables

R410A, R454B, R454C and R134a tables ship in `hgbp_sim/data/`. Any other CoolProp fluid is tabulated on
first use (needs CoolProp, a few seconds) and cached in `~/.cache/hgbp_sim/`.

## Layout

```
hgbp_sim/
  params.py        PlantParams, randomization, nominal charge
  geometry.py      volumes and heat capacities from the component and piping specification
  properties.py    refrigerant property tables (built from CoolProp, cached)
  components.py    valves, actuators, compressor map, flow resistances
  plant.py         HGBPPlant: a batch of stands (states, parameters, inputs), cold start, sensors, trips
  steady_state.py  equilibrium solver
  kernel/          the compiled model: property lookups, right-hand side, integrator, solver
  control.py       PID and the four-loop BaselineController
  scenarios.py     operating envelope, named test points, schedules
  env.py           HGBPVecEnv (batched) and HGBPEnv (gymnasium)
  interlock.py     compressor start/stop interlock (shared by the environment and the live stand)
  live.py          LiveStand: the web UI's engine
  ut35a.py         Yokogawa UT35A settings <-> PID gains
  defaults.py      stand defaults document (built-in values, JSON file, validation)
  data/            prebuilt property tables
webui/             operator interface: React frontend, FastAPI backend, Docker, config/stand_defaults.json
docs/              documentation
examples/          example scripts
figures/           figures used in the docs
tests/             tests; reference/ holds the model's reference data
```
