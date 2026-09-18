"""FastAPI backend for the live test-stand UI.

Runs one :class:`hgbp_sim.live.LiveStand` in a background thread at
``dt_ctrl / speed_factor`` wall-clock intervals, exposes a small REST API for
operator actions and settings, streams a snapshot per control step over a
WebSocket, and serves the built React frontend.

    uvicorn webui.backend.app:app --host 0.0.0.0 --port 8000
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
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hgbp_sim.live import HISTORY_CHANNELS, LOOPS, LiveStand

STATIC_DIR = os.environ.get(
    "HGBP_STATIC_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist"))


class Runner(threading.Thread):
    """Steps the stand in real time (scaled by ``speed_factor``)."""

    def __init__(self, stand: LiveStand):
        super().__init__(daemon=True)
        self.stand = stand
        self.lock = threading.Lock()
        self.stop_flag = False
        self.snapshot = stand.snapshot()
        self.row = stand.last_row() if hasattr(stand, "_last_row") else None

    def run(self) -> None:
        next_t = time.monotonic()
        while not self.stop_flag:
            st = self.stand
            if st.paused:
                time.sleep(0.05)
                next_t = time.monotonic()
                continue
            with self.lock:
                self.snapshot = st.step()
                self.row = st.last_row()
            next_t += st.dt_ctrl / st.speed_factor
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()      # cannot keep up: run flat out


stand = LiveStand()
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
    Kp: float | None = None
    Ki: float | None = None
    Kd: float | None = None


class CompressorCmd(BaseModel):
    run: bool | None = None
    speed: float | None = None
    reset: bool = False


class SimCmd(BaseModel):
    paused: bool | None = None
    speed_factor: float | None = None
    noise: bool | None = None
    dt_ctrl: float | None = None
    T_amb: float | None = None
    T_wi: float | None = None


class InitCmd(BaseModel):
    mode: str = "cold"                 # "cold" | "warm"
    T_amb: float | None = None
    T_wi: float | None = None
    liquid_in_accumulator: float | None = None
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
        stand.set_loop(name, **cmd.model_dump())
        return stand.snapshot()["loops"][name]


@app.post("/api/compressor")
def post_compressor(cmd: CompressorCmd):
    with runner.lock:
        stand.set_compressor(run=cmd.run, speed=cmd.speed, reset=cmd.reset)
        return stand.snapshot()["compressor"]


@app.post("/api/sim")
def post_sim(cmd: SimCmd):
    with runner.lock:
        stand.set_sim(paused=cmd.paused, speed_factor=cmd.speed_factor, noise=cmd.noise, dt_ctrl=cmd.dt_ctrl)
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
            stand.cold_start(T_amb=cmd.T_amb, T_wi=cmd.T_wi, liquid_in_accumulator=cmd.liquid_in_accumulator)
        return stand.snapshot()


@app.get("/api/params")
def get_params():
    with runner.lock:
        return stand.params_view()


@app.post("/api/params")
def post_params(cmd: ParamsCmd):
    with runner.lock:
        try:
            res = stand.set_params(cmd.values)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        res["view"] = stand.params_view()
        return res


@app.post("/api/charge")
def post_charge(cmd: ChargeCmd):
    with runner.lock:
        stand.add_charge(cmd.delta_kg)
        return stand.snapshot()["charge"]


@app.get("/api/history")
def get_history(since: float = -1.0, stride: int = 1):
    with runner.lock:
        return stand.history_since(since, max(1, stride))


@app.get("/api/export.csv")
def export_csv():
    with runner.lock:
        h = stand.history_since(-1.0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HISTORY_CHANNELS)
    for i in range(len(h["t"])):
        w.writerow([h[k][i] for k in HISTORY_CHANNELS])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=hgbp_stand.csv"})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    last = -1
    try:
        while True:
            snap, row = runner.snapshot, runner.row
            if snap["step"] != last:
                last = snap["step"]
                await websocket.send_json({"type": "step", "data": snap, "row": row})
            else:
                await asyncio.sleep(0.02)
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
