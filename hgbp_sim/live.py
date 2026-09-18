"""Interactive "live stand": one simulated test stand driven like the real
thing, for operator training, controller tuning and UI work.

The engine is framework-agnostic (no web code here).  It owns

* one :class:`HGBPPlant` (n = 1) advanced in control steps of ``dt_ctrl``,
* four PID loops with auto / manual mode, setpoints and live-editable gains
  (bumpless transfer on mode changes),
* the compressor start/stop interlock (:mod:`hgbp_sim.interlock`) with
  trip latching and operator reset,
* refrigerant charging / recovery while running,
* a rolling history of every channel for trend displays,
* parameter editing (live, or deferred to the next re-initialization for
  structural parameters such as volumes and refrigerant).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .components import cv_balance
from .control import DEFAULT_GAINS, BaselineController
from .interlock import ST_OFF, STATE_NAMES, Interlock, permissives
from .params import PlantParams, param_metadata
from .plant import HGBPPlant
from .properties import get_tables
from .scenarios import NAMED_POINTS, named_point
from .steady_state import solve_steady_state

C2K = 273.15
FLUIDS = ("R134a", "R1234yf", "R1234ze(E)", "R404A", "R407C", "R410A", "R32", "R22", "R290", "R600a")

# loop name -> (label, process value key, unit, valve index, gain scale to internal SI)
LOOPS = {
    "dpv": dict(label="Discharge pressure", pv="P_d", unit="bar", valve=0, scale=1e5, valve_label="1 discharge pressure valve"),
    "spv": dict(label="Suction pressure", pv="P_s", unit="bar", valve=1, scale=1e5, valve_label="2 suction pressure (HGBP) valve"),
    "stv": dict(label="Suction superheat", pv="SH", unit="K", valve=2, scale=1.0, valve_label="3 suction temperature (liquid) valve"),
    "water": dict(label="Intermediate pressure", pv="P_i", unit="bar", valve=3, scale=1e5, valve_label="4 cooling water valve"),
}
HISTORY_CHANNELS = (
    "t", "P_s", "P_d", "P_i", "Tsat_s", "Tsat_d", "Tsat_i", "T_s", "T_d", "T_co", "T_wi", "T_wo",
    "SH", "SC", "mdot", "W", "N", "u1", "u2", "u3", "u4",
    "sp_P_d", "sp_P_s", "sp_SH", "sp_P_i", "sp_N",
    "x_out", "fill_s", "fill_i", "charge", "T_sh", "T_cw", "mdot_w", "state", "Q_w",
)


class LiveStand:
    def __init__(self, params: PlantParams | None = None, dt_ctrl: float = 1.0, dt_sim: float = 0.05,
                 seed: int | None = 0, history_len: int = 4 * 3600, T_amb: float = 25.0,
                 T_wi: float = 20.0):
        self.params = params if params is not None else PlantParams()
        self.dt_ctrl = float(dt_ctrl)
        self.dt_sim = float(dt_sim)
        self.rng = np.random.default_rng(seed)
        self.history_len = int(history_len)
        self.noise = True
        self.speed_factor = 1.0
        self.paused = False
        self.step_count = 0
        self.T_amb = T_amb + C2K
        self.T_wi = T_wi + C2K
        self.pending_params: dict = {}
        self.charge_pending = 0.0           # kg still to add (+) / recover (-)
        self.charge_rate = 0.005            # kg/s
        self.trip_reasons: list[str] = []
        self.events: deque = deque(maxlen=200)
        self._build()
        self.cold_start()

    # ------------------------------------------------------------- building
    def _build(self) -> None:
        self.plant = HGBPPlant(self.params, n=1, dt=self.dt_sim, rng=self.rng)
        self.props = self.plant.props
        self.ctrl = BaselineController(1)
        self.mode = {k: "auto" for k in LOOPS}
        self.manual_out = {k: 0.0 for k in LOOPS}
        self.gains = {k: dict(DEFAULT_GAINS[k]) for k in LOOPS}
        self._apply_gains()
        self.interlock = Interlock(1, 60.0, 120.0)
        self.run_request = False
        self.speed_sp = float(self.params.N_nom)
        pt = named_point("MT_standard", self.props, self.params.N_nom)
        self.sp = dict(P_d=pt["P_d"], P_s=pt["P_s"], SH=pt["SH"], P_i=pt["P_i"])
        self.history = {k: deque(maxlen=self.history_len) for k in HISTORY_CHANNELS}
        self.t = 0.0

    def _apply_gains(self) -> None:
        for k, pid in self._pids().items():
            g = self.gains[k]
            pid.Kp, pid.Ki, pid.Kd = (np.asarray(g[n], float) for n in ("Kp", "Ki", "Kd"))

    def _pids(self) -> dict:
        return dict(dpv=self.ctrl.pid_1, spv=self.ctrl.pid_2, stv=self.ctrl.pid_3, water=self.ctrl.pid_4)

    def log(self, msg: str) -> None:
        self.events.appendleft(dict(t=round(self.t, 1), msg=msg))

    # ------------------------------------------------------- initialization
    def apply_pending(self) -> None:
        if self.pending_params:
            self.params = self.params.replace(**self.pending_params)
            self.pending_params = {}
        old = (self.mode, self.manual_out, self.gains, self.sp, self.speed_sp, self.history, self.t)
        self._build()
        self.mode, self.manual_out, self.gains, self.sp, self.speed_sp, self.history, self.t = old
        self._apply_gains()

    def cold_start(self, T_amb: float | None = None, T_wi: float | None = None,
                   liquid_in_accumulator: float | None = None) -> None:
        """Equalized stand at ambient, compressor off, valves at rest positions."""
        self.apply_pending()
        if T_amb is not None:
            self.T_amb = T_amb + C2K
        if T_wi is not None:
            self.T_wi = T_wi + C2K
        self.plant.set_inputs(T_amb=self.T_amb, T_wi=self.T_wi)
        u0 = self.ctrl.u_off.copy()
        self.plant.cold_start(T_amb=self.T_amb, u_pos=u0[None, :], liquid_in_accumulator=liquid_in_accumulator)
        for k, pid in self._pids().items():
            pid.reset(u0[LOOPS[k]["valve"]])
            self.manual_out[k] = float(u0[LOOPS[k]["valve"]])
        self.interlock.reset(running=False)
        self.run_request = False
        self.trip_reasons = []
        self.charge_pending = 0.0
        self.plant.set_inputs(u_cmd=u0[None, :], N_cmd=0.0)
        self.log("cold start")

    def warm_start(self, point: dict | str, T_amb: float | None = None, T_wi: float | None = None) -> bool:
        """Equilibrium at a test point (``named_point`` name or dict with
        T_evap, T_cond, T_int [degC], SH [K], N [rpm]); compressor running,
        loops in auto at the point's setpoints.  Returns False if no
        equilibrium exists (e.g. wrong charge); the stand is then left as is."""
        self.apply_pending()
        if T_amb is not None:
            self.T_amb = T_amb + C2K
        if T_wi is not None:
            self.T_wi = T_wi + C2K
        self.plant.set_inputs(T_amb=self.T_amb, T_wi=self.T_wi)
        if isinstance(point, str):
            pt = named_point(point, self.props, self.speed_sp)
        else:
            pr = self.props
            pt = dict(P_s=float(pr.P_sat(point["T_evap"] + C2K)), P_d=float(pr.P_sat(point["T_cond"] + C2K)),
                      SH=float(point["SH"]), P_i=float(pr.P_sat(point["T_int"] + C2K)),
                      N=float(point.get("N", self.speed_sp)))
        res = solve_steady_state(self.plant, pt["P_s"], pt["P_d"], pt["SH"], pt["N"], P_i=pt["P_i"])
        if not bool(res["converged"][0]):
            self.log("warm start failed: no equilibrium at this point with the current charge")
            return False
        self.plant.set_state(0, res["x"])
        self.plant.set_inputs(u_cmd=res["u"], N_cmd=pt["N"])
        self.sp = dict(P_d=pt["P_d"], P_s=pt["P_s"], SH=pt["SH"], P_i=pt["P_i"])
        self.speed_sp = pt["N"]
        for k, pid in self._pids().items():
            pid.reset(res["u"][0, LOOPS[k]["valve"]])
            self.manual_out[k] = float(res["u"][0, LOOPS[k]["valve"]])
            self.mode[k] = "auto"
        self.interlock.reset(running=True)
        self.run_request = True
        self.trip_reasons = []
        self.charge_pending = 0.0
        self.log("warm start at equilibrium")
        return True

    # ------------------------------------------------------------ operator
    def set_loop(self, name: str, mode: str | None = None, sp: float | None = None,
                 out: float | None = None, Kp=None, Ki=None, Kd=None) -> None:
        info = LOOPS[name]
        pid = self._pids()[name]
        if sp is not None:
            self.sp[info["pv"]] = float(sp) * (1e5 if info["unit"] == "bar" else 1.0)
        if mode is not None and mode != self.mode[name]:
            self.mode[name] = mode
            if mode == "auto":
                pid.reset(self.manual_out[name])            # bumpless: start from the manual output
            else:
                self.manual_out[name] = float(pid.u[0])      # take over the current output
            self.log(f"{info['label']} loop -> {mode}")
        if out is not None:
            self.manual_out[name] = float(np.clip(out, 0.0, 1.0))
        g = self.gains[name]
        for key, val in (("Kp", Kp), ("Ki", Ki), ("Kd", Kd)):
            if val is not None:
                g[key] = float(val) / info["scale"]
        self._apply_gains()

    def set_compressor(self, run: bool | None = None, speed: float | None = None, reset: bool = False) -> None:
        if speed is not None:
            self.speed_sp = float(np.clip(speed, self.params.N_min, self.params.N_max))
        if reset:
            self.interlock.acknowledge()
            if not self.interlock.tripped[0]:
                self.trip_reasons = []
                self.log("trip reset")
        if run is not None:
            self.run_request = bool(run)
            self.log("start requested" if run else "stop requested")

    def add_charge(self, delta_kg: float) -> None:
        """Queue refrigerant to be added (+) or recovered (-) at ``charge_rate``."""
        self.charge_pending += float(delta_kg)
        self.log(f"charge change queued: {delta_kg:+.3f} kg")

    def set_sim(self, paused: bool | None = None, speed_factor: float | None = None,
                noise: bool | None = None, dt_ctrl: float | None = None) -> None:
        if paused is not None:
            self.paused = bool(paused)
        if speed_factor is not None:
            self.speed_factor = float(np.clip(speed_factor, 0.1, 100.0))
        if noise is not None:
            self.noise = bool(noise)
        if dt_ctrl is not None:
            self.dt_ctrl = float(np.clip(dt_ctrl, 0.2, 10.0))

    def set_conditions(self, T_amb: float | None = None, T_wi: float | None = None) -> None:
        if T_amb is not None:
            self.T_amb = float(T_amb) + C2K
        if T_wi is not None:
            self.T_wi = float(T_wi) + C2K
        self.plant.set_inputs(T_amb=self.T_amb, T_wi=self.T_wi)

    # ----------------------------------------------------------- parameters
    def set_params(self, values: dict) -> dict:
        """Apply parameter changes.  Structural ones (refrigerant, volumes,
        charge) are deferred until the next cold/warm start."""
        meta = {m["name"]: m for m in param_metadata()}
        applied, deferred = [], []
        for name, val in values.items():
            if name not in meta:
                continue
            m = meta[name]
            if m["kind"] == "float":
                val = float(val)
            elif m["kind"] == "bool":
                val = bool(val)
            if m["requires_init"]:
                self.pending_params[name] = val
                deferred.append(name)
            else:
                self.params = self.params.replace(**{name: val})
                if m["kind"] == "float":
                    getattr(self.plant.p, name)[:] = val
                else:
                    setattr(self.plant.p, name, val)
                applied.append(name)
        return dict(applied=applied, deferred=deferred)

    def params_view(self) -> dict:
        vals = {}
        for m in param_metadata():
            v = self.pending_params.get(m["name"], getattr(self.params, m["name"]))
            if m["name"] == "charge":
                v = float(self.plant.p.charge[0])
            vals[m["name"]] = v
        return dict(values=vals, meta=param_metadata(), pending=sorted(self.pending_params),
                    fluids=list(FLUIDS), named_points=NAMED_POINTS,
                    nominal_charge=float(self.plant.nominal_charge()[0]))

    # -------------------------------------------------------------- physics
    def _inject_charge(self, dm: float) -> None:
        """Add (dm > 0) liquid from a cylinder at ambient temperature to the
        accumulator, or recover (dm < 0) fluid at the tank's mean enthalpy."""
        pl, pr, p = self.plant, self.props, self.plant.p
        x = pl.x[0]
        P_s, h_s = x[HGBPPlant.P_S], x[HGBPPlant.H_S]
        S = pr.state(np.array([P_s]), np.array([h_s]))
        if dm > 0:
            h_in = float(pr.sat(pr.P_sat(np.array([self.T_amb])))["h_l"][0])
            E = dm * (h_in - h_s)
        else:
            E = 0.0
        dP, dh = cv_balance(p.V_s[0], S.rho[0], S.drho_dP[0], S.drho_dh[0], dm, E)
        x[HGBPPlant.P_S] += dP
        x[HGBPPlant.H_S] += dh
        pl.aux = None

    def step(self) -> dict:
        """One control interval.  Returns the snapshot."""
        pl, p, dt = self.plant, self.plant.p, self.dt_ctrl
        meas = pl.measure(noise=self.noise)
        running = bool(pl.x[0, HGBPPlant.N_] > 0.5 * p.N_min[0])
        u = np.zeros(4)
        for k, info in LOOPS.items():
            pid = self._pids()[k]
            pv = meas[info["pv"]][0]
            if self.mode[k] == "auto":
                u[info["valve"]] = pid.update(self.sp[info["pv"]], pv, dt, active=np.array([running]))[0]
            else:
                pid.reset(self.manual_out[k])
                u[info["valve"]] = self.manual_out[k]
        perm = permissives(meas["P_s"], meas["P_d"], pl.x[:, HGBPPlant.U1], pl.x[:, HGBPPlant.U2], p,
                           self.interlock.tripped)
        il = self.interlock.step(np.array([self.run_request]), perm, pl.x[:, HGBPPlant.N_],
                                 np.array([self.speed_sp]), p.N_min, dt)
        if il["switched"][0]:
            self.log(f"compressor {STATE_NAMES[self.interlock.state[0]]}")
        if il["blocked"][0] and self.step_count % 10 == 0:
            self.log("start request blocked by permissives")
        if self.charge_pending != 0.0:
            dm = float(np.sign(self.charge_pending) * min(abs(self.charge_pending), self.charge_rate * dt))
            self._inject_charge(dm)
            self.charge_pending -= dm
            if self.charge_pending == 0.0:
                self.log("charge change complete")
        aux = pl.step(dt, u_cmd=u[None, :], N_cmd=il["N_cmd"], T_amb=self.T_amb, T_wi=self.T_wi)
        p.charge[0] = aux["M_tot"][0]
        trips = pl.trips()
        hard = [k for k in ("high_P_d", "low_P_s", "high_P_s", "high_T_d") if bool(trips[k][0])]
        if hard and not self.interlock.tripped[0]:
            self.interlock.tripped[:] = True
            self.run_request = False
            self.trip_reasons = hard
            self.log("TRIP: " + ", ".join(hard))
        self.t += dt
        self.step_count += 1
        self._record(aux, meas)
        return self.snapshot(aux, meas, trips)

    # --------------------------------------------------------------- output
    def _record(self, aux, meas) -> None:
        pr = self.props
        row = dict(
            t=self.t, P_s=aux["P_s"][0] / 1e5, P_d=aux["P_d"][0] / 1e5, P_i=aux["P_i"][0] / 1e5,
            Tsat_s=aux["T_sat_s"][0] - C2K, Tsat_d=aux["T_sat_d"][0] - C2K, Tsat_i=aux["T_sat_i"][0] - C2K,
            T_s=meas["T_s"][0] - C2K, T_d=meas["T_d"][0] - C2K, T_co=meas["T_co"][0] - C2K,
            T_wi=self.T_wi - C2K, T_wo=aux["T_wo"][0] - C2K, SH=meas["SH"][0], SC=meas["SC"][0],
            mdot=meas["mdot"][0] * 1e3, W=meas["W"][0], N=aux["N"][0],
            u1=aux["u1"][0], u2=aux["u2"][0], u3=aux["u3"][0], u4=aux["u4"][0],
            sp_P_d=self.sp["P_d"] / 1e5, sp_P_s=self.sp["P_s"] / 1e5, sp_SH=self.sp["SH"], sp_P_i=self.sp["P_i"] / 1e5,
            sp_N=self.speed_sp, x_out=aux["x_out"][0], fill_s=aux["fill_s"][0], fill_i=aux["fill_i"][0],
            charge=aux["M_tot"][0], T_sh=aux["T_sh"][0] - C2K, T_cw=aux["T_cw"][0] - C2K,
            mdot_w=aux["mdot_w"][0] * 60.0, state=int(self.interlock.state[0]), Q_w=aux["Q_w"][0],
        )
        for k, v in row.items():
            self.history[k].append(float(v))
        self._last_row = row

    def last_row(self) -> dict:
        return dict(self._last_row)

    def history_since(self, t0: float = -1.0, stride: int = 1) -> dict:
        t = np.fromiter(self.history["t"], float)
        i0 = int(np.searchsorted(t, t0, side="right")) if t0 >= 0 else 0
        out = {}
        for k, dq in self.history.items():
            arr = np.fromiter(dq, float)[i0::stride]
            out[k] = arr.tolist()
        return out

    def snapshot(self, aux=None, meas=None, trips=None) -> dict:
        pl, p = self.plant, self.plant.p
        aux = aux if aux is not None else pl.outputs()
        meas = meas if meas is not None else pl.measure(noise=False)
        trips = trips if trips is not None else pl.trips()
        f = lambda v: float(np.asarray(v).ravel()[0])
        running = f(aux["N"]) > 0.5 * f(p.N_min)
        loops = {}
        for k, info in LOOPS.items():
            unit_scale = 1e5 if info["unit"] == "bar" else 1.0
            g = self.gains[k]
            loops[k] = dict(label=info["label"], valve_label=info["valve_label"], unit=info["unit"],
                            pv=f(meas[info["pv"]]) / unit_scale, sp=self.sp[info["pv"]] / unit_scale,
                            out=f(aux["u%d" % (info["valve"] + 1)]), mode=self.mode[k],
                            manual_out=self.manual_out[k],
                            Kp=g["Kp"] * info["scale"], Ki=g["Ki"] * info["scale"], Kd=g["Kd"] * info["scale"])
        perm_detail = dict(
            P_s_above_min=f(meas["P_s"]) > 1.5 * f(p.P_s_min), P_s_below_max=f(meas["P_s"]) < 0.9 * f(p.P_s_max),
            P_d_below_max=f(meas["P_d"]) < 0.8 * f(p.P_d_max), valve1_open=f(aux["u1"]) >= 0.1,
            valve2_open=f(aux["u2"]) >= 0.05, no_trip=not bool(self.interlock.tripped[0]),
            off_time_elapsed=bool(self.interlock.state[0] != ST_OFF or self.interlock.t_state[0] >= self.interlock.min_off_time),
        )
        return dict(
            t=self.t, step=self.step_count, paused=self.paused, speed_factor=self.speed_factor, noise=self.noise,
            dt_ctrl=self.dt_ctrl, fluid=self.params.fluid,
            compressor=dict(state=STATE_NAMES[int(self.interlock.state[0])], tripped=bool(self.interlock.tripped[0]),
                            trip_reasons=list(self.trip_reasons), run_request=self.run_request,
                            speed_sp=self.speed_sp, speed=f(aux["N"]), running=running,
                            permissive_ok=all(perm_detail.values()), permissives=perm_detail,
                            t_state=f(self.interlock.t_state), min_off_time=self.interlock.min_off_time,
                            min_run_time=self.interlock.min_run_time, N_min=f(p.N_min), N_max=f(p.N_max)),
            loops=loops,
            meas=dict(P_s=f(meas["P_s"]) / 1e5, P_d=f(meas["P_d"]) / 1e5, P_i=f(meas["P_i"]) / 1e5,
                      T_s=f(meas["T_s"]) - C2K, T_d=f(meas["T_d"]) - C2K, T_co=f(meas["T_co"]) - C2K,
                      T_wi=self.T_wi - C2K, T_wo=f(aux["T_wo"]) - C2K, T_amb=self.T_amb - C2K,
                      Tsat_s=f(meas["T_sat_s"]) - C2K, Tsat_d=f(aux["T_sat_d"]) - C2K, Tsat_i=f(meas["T_sat_i"]) - C2K,
                      SH=f(meas["SH"]), SC=f(meas["SC"]), mdot=f(meas["mdot"]) * 1e3, W=f(meas["W"]),
                      N=f(aux["N"]), u1=f(aux["u1"]), u2=f(aux["u2"]), u3=f(aux["u3"]), u4=f(aux["u4"])),
            true=dict(x_out=f(aux["x_out"]), x_s=f(aux["x_s"]), x_i=f(aux["x_i"]), fill_s=f(aux["fill_s"]),
                      fill_i=f(aux["fill_i"]), T_sh=f(aux["T_sh"]) - C2K, T_cw=f(aux["T_cw"]) - C2K,
                      mdot_1=f(aux["mdot_1"]) * 1e3, mdot_2=f(aux["mdot_2"]) * 1e3, mdot_3=f(aux["mdot_3"]) * 1e3,
                      mdot_w=f(aux["mdot_w"]) * 60.0, Q_w=f(aux["Q_w"]), Q_r=f(aux["Q_r"]), W_el=f(aux["W_el"]),
                      eta_v=f(aux["eta_v"]), eta_s=f(aux["eta_s"]), Pr=f(aux["Pr"]),
                      M_s=f(aux["M_s"]), M_d=f(aux["M_d"]), M_i=f(aux["M_i"])),
            alarms=dict(floodback=bool(trips["floodback"][0]) and running,
                        accumulator_liquid=bool(trips["accumulator_liquid"][0]),
                        condenser_dry=bool(trips["condenser_dry"][0]),
                        condenser_flooded=bool(trips["condenser_flooded"][0]),
                        high_T_d_warning=f(aux["T_d"]) > f(p.T_d_max) - 15.0,
                        high_P_d_warning=f(aux["P_d"]) > 0.9 * f(p.P_d_max)),
            limits=dict(P_d_max=f(p.P_d_max) / 1e5, P_s_min=f(p.P_s_min) / 1e5, P_s_max=f(p.P_s_max) / 1e5,
                        T_d_max=f(p.T_d_max) - C2K),
            charge=dict(kg=f(aux["M_tot"]), nominal_kg=f(pl.nominal_charge()), pending_kg=self.charge_pending,
                        rate_kg_s=self.charge_rate),
            pending_params=sorted(self.pending_params),
            events=list(self.events)[:30],
        )
