# hgbp-sim

Dynamic simulation of a **hot-gas-bypass (HGBP) refrigeration compressor test stand**, for
training and evaluating controllers that automate compressor performance testing, and for
running the stand interactively in a browser.

## The stand

![the stand](figures/stand.svg)

| valve | controls |
|---|---|
| 1 discharge line | discharge pressure |
| 2 hot gas bypass | suction pressure |
| 3 liquid quench | suction temperature (superheat) |
| 4 condenser cooling water | intermediate (condensing) pressure |

A test point is suction pressure, discharge pressure, superheat, intermediate pressure and
compressor speed.

## Features

* **Physics-based model** of the stand as built: brazed-plate condenser and mixing
  exchanger (five cells each, vapor quality through both), liquid receiver, specified
  piping, pressure drops, real-fluid refrigerant properties (any CoolProp fluid; R410A by
  default).
* **Two-phase liquid section:** the receiver's liquid keeps its own temperature. It
  subcools after a pressure rise and flashes after a drop, sending flash gas to the quench
  valve. The condenser holds its condensing film, and a full receiver floods it.
* **Refrigerant charge as a parameter:** under- and overcharged stands behave as they
  should (receiver level, loss of the liquid seal, condenser flooding, floodback).
* **Fast:** compiled with numba. On an 8-core laptop the live stand runs about 350 times
  faster than real time, and a batch of stands trains at 15 000-19 000 environment steps
  per second.
* **Training environment:** batched and `gymnasium` versions, refrigerant-agnostic
  observations, a baseline PID expert, domain randomization and charge labels.
* **Web UI:** operator panel with the stand's UT35A controller settings, trends, metric or
  US units.
* **Steady-state solver** for warm starts, feasibility checks and performance maps.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m hgbp_sim.kernel.warmup      # compile the model once (1-2 min, then cached)
```

Runtime dependencies are `numpy` and `numba`; `[dev]` adds CoolProp, gymnasium,
matplotlib and pytest.

## Quick start

A training environment:

```python
from hgbp_sim import HGBPEnv, EnvConfig

env = HGBPEnv(EnvConfig())
obs, info = env.reset(seed=0)
for _ in range(2400):                      # 600 s at the 0.25 s control interval
    action = env.expert_action()           # the baseline PID; replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

The web UI:

```bash
docker compose -f webui/docker-compose.yml up --build     # then open http://localhost:8000
```

## Documentation

| document | contents |
|---|---|
| [docs/model.md](docs/model.md) | the stand model: hardware, structure, numerics, parameters, limitations |
| [docs/pressure-drop.md](docs/pressure-drop.md) | exchanger and line pressure drops and their coefficients |
| [docs/charge.md](docs/charge.md) | refrigerant charge and the nominal charge |
| [docs/training.md](docs/training.md) | training environment, reward, baseline controller, throughput |
| [docs/controller-spec.md](docs/controller-spec.md) | specification for a neural-network controller |
| [docs/web-ui.md](docs/web-ui.md) | operating the live stand, controller settings, defaults file, API |
| [docs/examples.md](docs/examples.md) | example scripts and results |
| [docs/development.md](docs/development.md) | tests, the compiled kernel, reference data, code layout |

## License

MIT, see `LICENSE`.
