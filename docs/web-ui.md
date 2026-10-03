# Web UI: the live stand

A browser interface to run the simulated stand like the real one (`webui/`).

## Running it

```bash
# Docker: builds the frontend and the backend, compiles the model
docker compose -f webui/docker-compose.yml up --build      # open http://localhost:8000

# local development
pip install -e ".[dev]" fastapi "uvicorn[standard]"
uvicorn webui.backend.app:app --reload --port 8000
cd webui/frontend && npm install && npm run dev            # http://localhost:5173
```

The image builds on x86-64 and arm64. For an x86-64 image on Apple Silicon, add
`--platform linux/amd64`.

## Tabs

* **Operator:** live schematic, four PID faceplates (auto / manual, setpoint, manual
  output), the compressor panel (start / stop, permissives, trip reset), process values,
  charging and recovery, a saturation pressure calculator (dew point, x = 1) and the event
  log.
* **Trends:** pressures with setpoints, superheat and subcooling (with the receiver
  liquid's own subcooling), valves, temperatures, mixing exchanger, flow / power / speed and
  inventory (receiver level, condenser flooding and liquid); CSV export.
* **Settings:** simulation speed, noise, control interval, ambient and water temperature,
  charging rate, PID tuning and every plant parameter (with units, descriptions and
  defaults). Refrigerant, volumes and charge apply at the next cold or warm start.

The top bar has the speed factor (up to 100×), pause, cold / warm start and the display
units.

The schematic shows both brazed plate exchangers cell by cell. A cell shows its quality while
it is two-phase and its temperature otherwise, and its shading follows the liquid. The
receiver shows:
* its level and liquid temperature;
* the liquid's subcooling, or "flashing" while it boils off after a pressure drop;
* whether the liquid line has its seal.

The condenser shows its liquid (film and draining condensate) and how much of it is
flooded.

## Starting the compressor

The loops run whenever they are in AUTO, compressor on or off, as the stand's controllers
do. At standstill the pressures are far from their setpoints, so the loops drive their
valves to a limit and the start permissives fail (valve 1 needs at least 10 %, valve 2 at
least 5 %). To start:

1. put the four loops in MAN at start positions (for example 50 / 60 / 0 / 30 %);
2. request the run;
3. switch the loops to AUTO once the compressor turns (about 600 rpm).

With the discharge loop in AUTO before that, it closes valve 1 against the starting
compressor and the stand trips on high discharge pressure within seconds. A PV filter on
the discharge loop also slows its answer to the start. With the stand's defaults file
(`FL` = 4 s) the discharge pressure peaks at about 41.6 bar, just over the 41.4 bar trip;
with `FL` = 2 s it peaks at about 39.6 bar. The
anti-short-cycle timers (60 s minimum off, 120 s minimum run) can be switched off on the
Settings tab.

## PID tuning (Yokogawa UT35A)

Each loop is set up like the stand's UT35A controllers (manual IM 05P01D31-01EN, in
`docs/`):

| setting | meaning | range |
|---|---|---|
| `P` | proportional band, % of the input range `RL..RH` | 0.1-999.9 % |
| `I` | integral time | 1-6000 s or OFF |
| `D` | derivative time (on PV) | 1-6000 s or OFF |
| `DR` | `DIR` (output rises with PV) or `RVS` | |
| `OL..OH` | output limits | 0-100 % |
| `FL` | PV input filter: a first-order lag on the PV, ahead of the display and the PID | 1-120 s or OFF |

A setting read off a real controller carries over directly, provided `RL..RH` matches its
input range. Setpoints stay inside the input range. The control interval defaults to
0.2 s, the UT35A's control period. Faceplates show the filtered PV; the schematic, trends
and permissives show the sensor values.

As on the real stand, the third loop controls the suction *temperature* (a warm start
derives its setpoint from the test point's superheat); the training environment's baseline
controller works on superheat.

## Defaults file

All defaults (simulation settings, the four loops and every plant parameter) come from
`webui/config/stand_defaults.json`, which docker compose mounts at `/config`
(`HGBP_DEFAULTS=/config/stand_defaults.json`).

* Edit it on the host and restart: `docker compose -f webui/docker-compose.yml restart`.
* Keys left out keep their built-in value. An unknown key or an out-of-range value stops
  the backend with a message naming it (`docker compose -f webui/docker-compose.yml logs`).
* A missing file is created from the built-in defaults; `python -m hgbp_sim.defaults <path>`
  writes them too.
* Units are listed in the file's `_about` entry: loop setpoints and ranges in bar absolute
  or °C, plant parameters in SI.
* The stand's file sets `FL = 4` with the stand's own tuning. The built-in defaults leave
  `FL` off: their gains were tuned without a filter, and with it their discharge pressure
  loop overshoots to the trip at a cold start.

## Display units

The units selector switches each browser between metric and US units (remembered in the
browser). It applies to operating values, the start dialog, conditions and the CSV export;
the plant parameter table stays in SI.

| quantity | metric | US |
|---|---|---|
| pressure | bar (absolute) | psia |
| pressure drop | kPa | psi |
| temperature | °C | °F |
| superheat, subcooling | K | °F |
| refrigerant flow | g/s | lb/h |
| water flow | kg/min | gpm |
| heat | kW | Btu/h |
| electrical power | kW | kW |
| charge | kg | lb |

## Backend API

The engine is `hgbp_sim.live.LiveStand` (one stand, four PID loops, the start/stop
interlock, trips, charging, a rolling history), which can also be scripted directly. The
backend (`webui/backend/app.py`) serves it over HTTP in metric units:

| endpoint | purpose |
|---|---|
| `GET /api/state` | current snapshot |
| `POST /api/loop/{name}` | mode, `sp`, `out`, `P`, `I`, `D`, `FL` |
| `POST /api/compressor` | run / stop, speed, trip reset |
| `POST /api/sim` | pause, speed factor, noise, control interval, conditions, charging rate |
| `POST /api/init` | cold or warm start |
| `GET/POST /api/params` | plant parameters |
| `POST /api/charge` | add or recover refrigerant |
| `GET /api/history` | trend history |
| `GET /api/export.csv?units=metric\|english` | history as CSV, columns labelled with units |
| `GET /api/saturation?T=` | dew point pressure of the refrigerant at `T` °C |
| `WS /ws` | the latest snapshot and new history rows, at most ten messages per second |
