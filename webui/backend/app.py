"""FastAPI backend for the live test-stand UI.

Runs one :class:`hgbp_sim.live.LiveStand` in a background thread at
``dt_ctrl / speed_factor`` wall-clock intervals, exposes a small REST API for
operator actions and settings, streams snapshots and history rows over a
WebSocket (at most 10 messages per second), and serves the built React frontend.

    uvicorn webui.backend.app:app --host 0.0.0.0 --port 8000

Defaults (simulation settings, loop tuning in UT35A units, plant parameters)
come from the JSON file named by ``HGBP_DEFAULTS``; it is created from the
built-in defaults if it does not exist.  Without the variable the built-in
defaults apply.
"""
from __future__ import annotations

import asyncio
import csv
import io
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hgbp_sim.defaults import load_defaults
from hgbp_sim.live import HISTORY_CHANNELS, LOOPS, LiveStand

STATIC_DIR = os.environ.get(
    "HGBP_STATIC_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist"))


class Runner(threading.Thread):
    """Steps the stand in real time (scaled by ``speed_factor``)."""
    HOLD_MAX = 0.02          # longest the runner holds the lock at a time [s]
    YIELD = 0.002            # shortest pause between two holds, for requests and the stream [s]

    def __init__(self, stand: LiveStand):
        super().__init__(daemon=True)
        self.stand = stand
        self.lock = threading.Lock()
        self.stop_flag = False
        self.snapshot = stand.snapshot()

    def run(self) -> None:
        next_t = time.monotonic()
        t_prev = next_t
        while not self.stop_flag:
            st = self.stand
            if st.paused:
                time.sleep(0.05)
                next_t = t_prev = time.monotonic()
                continue
            # steps due by now (at least one), several per hold when the schedule runs
            # ahead of the stand; the lock is released every HOLD_MAX so that operator
            # requests and the stream are served at any speed
            steps = 0
            with self.lock:
                t_hold = time.monotonic()
                while True:
                    st.step(snapshot=False)
                    steps += 1
                    next_t += st.dt_ctrl / st.speed_factor
                    now = time.monotonic()
                    if next_t > now or now - t_hold > self.HOLD_MAX or st.paused:
                        break
                self.snapshot = st.last_snapshot()        # one per hold: the stream sends at most 10/s
            if next_t < now - 0.5:
                next_t = now                   # far behind: the stand runs flat out, drop the backlog
            time.sleep(max(next_t - time.monotonic(), self.YIELD))
            # achieved simulation speed (simulated seconds per wall second), smoothed
            now = time.monotonic()
            rate = steps * st.dt_ctrl / max(now - t_prev, 1e-6)
            t_prev = now
            st.achieved_speed = 0.9 * st.achieved_speed + 0.1 * rate if st.achieved_speed else rate


DEFAULTS_PATH = os.environ.get("HGBP_DEFAULTS") or None
stand = LiveStand(defaults=load_defaults(DEFAULTS_PATH))
runner = Runner(stand)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runner.start()
    yield
    runner.stop_flag = True


app = FastAPI(title="HGBP test stand simulator", lifespan=lifespan)


def _snapshot() -> dict:
    with runner.lock:
        return stand.snapshot()


# ------------------------------------------------------------------ models
class LoopCmd(BaseModel):
    mode: str | None = None
    sp: float | None = None
    out: float | None = None
    P: float | None = None                 # proportional band [%]
    I: float | str | None = None           # integral time [s] or "OFF"
    D: float | str | None = None           # derivative time [s] or "OFF"
    FL: float | str | None = None          # PV input filter [s] or "OFF"


class CompressorCmd(BaseModel):
    run: bool | None = None
    speed: float | None = None
    reset: bool = False


class SimCmd(BaseModel):
    paused: bool | None = None
    speed_factor: float | None = None
    noise: bool | None = None
    dt_ctrl: float | None = None
    charge_rate_g_s: float | None = None
    short_cycle_timers: bool | None = None
    T_amb: float | None = None
    T_wi: float | None = None


class InitCmd(BaseModel):
    mode: str = "cold"                 # "cold" | "warm"
    T_amb: float | None = None
    T_wi: float | None = None
    liquid_in_suction: float | None = None
    point: str | dict | None = None    # warm: named point or {T_evap, T_cond, T_int, SH, N}


class ParamsCmd(BaseModel):
    values: dict


class ChargeCmd(BaseModel):
    delta_kg: float


# ------------------------------------------------------------------- routes
@app.get("/api/state")
def get_state():
    return _snapshot()


@app.post("/api/loop/{name}")
def post_loop(name: str, cmd: LoopCmd):
    if name not in LOOPS:
        raise HTTPException(404, f"unknown loop {name}")
    with runner.lock:
        try:
            stand.set_loop(name, **cmd.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return stand.snapshot()["loops"][name]


@app.post("/api/compressor")
def post_compressor(cmd: CompressorCmd):
    with runner.lock:
        stand.set_compressor(run=cmd.run, speed=cmd.speed, reset=cmd.reset)
        return stand.snapshot()["compressor"]


@app.post("/api/sim")
def post_sim(cmd: SimCmd):
    with runner.lock:
        stand.set_sim(paused=cmd.paused, speed_factor=cmd.speed_factor, noise=cmd.noise, dt_ctrl=cmd.dt_ctrl,
                      charge_rate=None if cmd.charge_rate_g_s is None else cmd.charge_rate_g_s / 1000.0,
                      short_cycle_timers=cmd.short_cycle_timers)
        if cmd.T_amb is not None or cmd.T_wi is not None:
            stand.set_conditions(T_amb=cmd.T_amb, T_wi=cmd.T_wi)
    return _snapshot()


@app.post("/api/init")
def post_init(cmd: InitCmd):
    with runner.lock:
        if cmd.mode == "warm":
            ok = stand.warm_start(cmd.point or "MT_standard", T_amb=cmd.T_amb, T_wi=cmd.T_wi)
            if not ok:
                raise HTTPException(409, "no equilibrium at this point with the current charge / parameters")
        else:
            stand.cold_start(T_amb=cmd.T_amb, T_wi=cmd.T_wi, liquid_in_suction=cmd.liquid_in_suction)
        return stand.snapshot()


@app.get("/api/params")
def get_params():
    with runner.lock:
        return {**stand.params_view(), "defaults_path": DEFAULTS_PATH}


@app.post("/api/params")
def post_params(cmd: ParamsCmd):
    with runner.lock:
        try:
            res = stand.set_params(cmd.values)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        res["view"] = {**stand.params_view(), "defaults_path": DEFAULTS_PATH}
        return res


@app.post("/api/charge")
def post_charge(cmd: ChargeCmd):
    with runner.lock:
        stand.add_charge(cmd.delta_kg)
        return stand.snapshot()["charge"]


@app.get("/api/saturation")
def get_saturation(T: float):
    """Dew point (x = 1) pressure of the stand's refrigerant at saturation temperature ``T`` [°C]."""
    with runner.lock:
        return stand.saturation(T)


@app.get("/api/history")
def get_history(since: float = -1.0, stride: int = 1, max_points: int = 15000):
    with runner.lock:
        h = stand.history_since(since, max(1, stride), max_points=max(100, max_points))
    return JSONResponse(h)          # plain lists of floats: no need for FastAPI's generic encoder


# Units of the history channels as the API gives them, and their US (English) display units
# (the same conversions as the UI's units.ts): name -> (factor, offset)
_CSV_UNITS = {
    "bar": ("psia", 14.5037738, 0.0), "°C": ("°F", 1.8, 32.0), "K": ("°F", 1.8, 0.0),
    "g/s": ("lb/h", 7.93664144, 0.0), "kg": ("lb", 2.20462262, 0.0), "kg/min": ("gpm", 0.264172052, 0.0),
    "W_heat": ("Btu/h", 3.41214163, 0.0), "kW_heat": ("kBtu/h", 3.41214163, 0.0),
}
_CHANNEL_UNITS = {
    "t": "s", "N": "rpm", "sp_N": "rpm", "W": "W", "Q_w": "W_heat", "Q_mx": "kW_heat", "state": "-",
    **{k: "bar" for k in ("P_s", "P_d", "P_i", "sp_P_d", "sp_P_s", "sp_P_i")},
    **{k: "°C" for k in ("Tsat_s", "Tsat_d", "Tsat_i", "T_s", "T_d", "T_co", "T_wi", "T_wo", "sp_T_s", "T_qo", "T_go",
                         "T_sh", "T_cw")},
    "SH": "K", "SC": "K", "mdot": "g/s", "mdot_w": "kg/min", "M_q_liq": "kg", "charge": "kg",
}


@app.get("/api/export.csv")
def export_csv(units: str = "metric"):
    """The history as CSV, every column labelled with its unit; ``units``: metric (the
    API's units) or english."""
    if units not in ("metric", "english"):
        raise HTTPException(400, "units must be metric or english")
    with runner.lock:
        h = stand.history_since(-1.0)
    cols = []
    for k in HISTORY_CHANNELS:
        base = _CHANNEL_UNITS.get(k, "-")
        if units == "english" and base in _CSV_UNITS:
            unit, f, o = _CSV_UNITS[base]
            cols.append((k, unit, f, o))
        else:
            cols.append((k, base.replace("_heat", ""), 1.0, 0.0))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f"{k} [{unit}]" for k, unit, _, _ in cols])
    for i in range(len(h["t"])):
        w.writerow([h[k][i] * f + o if (f, o) != (1.0, 0.0) else h[k][i] for k, _, f, o in cols])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=hgbp_stand_{units}.csv"})


WS_INTERVAL = 0.1        # shortest interval between two messages to a client [s]


@app.websocket("/ws")
async def ws(websocket: WebSocket, since: float | None = None):
    """The latest snapshot after every control step, at most every
    ``WS_INTERVAL`` (at high speed factors one message covers several steps),
    with every history row recorded since the previous message.  ``since``: the
    time of the last history row the client already holds (from /api/history);
    without it the stream starts with the rows recorded from now on."""
    await websocket.accept()
    with runner.lock:
        count = stand.rows_recorded if since is None else stand.rows_through(since)
    last_step = -1
    try:
        while True:
            snap = runner.snapshot
            if snap["step"] == last_step:
                await asyncio.sleep(0.02)
                continue
            with runner.lock:                  # costs only the new rows
                rows, count = stand.rows_after(count)
            last_step = snap["step"]
            await websocket.send_json({"type": "step", "data": snap, "rows": rows})
            await asyncio.sleep(WS_INTERVAL)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------- frontend
if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/{path:path}", response_class=HTMLResponse)
    def spa(path: str):
        full = os.path.join(STATIC_DIR, path)
        if path and os.path.isfile(full):
            return FileResponse(full)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
else:  # pragma: no cover
    @app.get("/")
    def no_frontend():
        return HTMLResponse("<p>Frontend not built. Run <code>npm run build</code> in webui/frontend.</p>")
