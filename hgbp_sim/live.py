"""Interactive "live stand": one simulated test stand driven like the real
thing, for operator training, controller tuning and UI work.

The engine is framework-agnostic (no web code here).  It owns

* one :class:`HGBPPlant` (n = 1) advanced in control steps of ``dt_ctrl``,
* four PID loops with auto / manual mode, setpoints and live-editable tuning
  in the units of the stand's Yokogawa UT35A controllers (:mod:`hgbp_sim.ut35a`;
  bumpless transfer on mode changes),
* the compressor start/stop interlock (:mod:`hgbp_sim.interlock`) with
  trip latching, operator reset and switchable anti-short-cycle timers,
* refrigerant charging / recovery while running,
* a rolling history of every channel for trend displays,
* parameter editing (live, or deferred to the next re-initialization for
  structural parameters such as volumes and refrigerant),
* defaults for all of the above from one document (:mod:`hgbp_sim.defaults`).
"""
from __future__ import annotations

from collections import deque
from itertools import islice

import numpy as np

from . import ut35a
from .control import BaselineController
from .defaults import merge_defaults
from .geometry import volume_table
from .interlock import ST_OFF, STATE_NAMES, Interlock, permissives
from .params import FLUIDS, PlantParams, clamp_to_range, param_metadata
from .plant import HGBPPlant
from .properties import get_tables
from .scenarios import NAMED_POINTS, named_point
from .steady_state import solve_steady_state

C2K = 273.15

# loop name -> label, process value key (SI in the plant), display unit, valve index,
# gain scale (SI -> per display unit)
LOOPS = {
    "dpv": dict(label="Discharge pressure", pv="P_d", unit="bar", valve=0, scale=1e5,
                valve_label="1 discharge pressure valve"),
    "spv": dict(label="Suction pressure", pv="P_s", unit="bar", valve=1, scale=1e5,
                valve_label="2 suction pressure (HGBP) valve"),
    "stv": dict(label="Suction temperature", pv="T_s", unit="°C", valve=2, scale=1.0,
                valve_label="3 suction temperature (liquid) valve"),
    "water": dict(label="Intermediate (liquid) pressure", pv="P_i", unit="bar", valve=3, scale=1e5,
                  valve_label="4 cooling water valve"),
}
_TO_SI = {"bar": lambda v: v * 1e5, "°C": lambda v: v + C2K, "K": lambda v: v}
_FROM_SI = {"bar": lambda v: v / 1e5, "°C": lambda v: v - C2K, "K": lambda v: v}
# anti-short-cycle timers of the interlock [s] (both 0 when the timers are switched off)
MIN_OFF_TIME, MIN_RUN_TIME = 60.0, 120.0
HISTORY_CHANNELS = (
    "t", "P_s", "P_d", "P_i", "Tsat_s", "Tsat_d", "Tsat_i", "T_s", "T_d", "T_co", "T_wi", "T_wo",
    "SH", "SC", "mdot", "W", "N", "u1", "u2", "u3", "u4",
    "sp_P_d", "sp_P_s", "sp_T_s", "sp_P_i", "sp_N",
    "x_out", "y_liq", "x_qo", "rec_level", "cond_flood", "M_q_liq", "T_qo", "T_go", "charge", "T_sh", "T_cw",
    "mdot_w", "state", "Q_w", "Q_mx",
)


class LiveStand:
    """``defaults`` is a validated defaults document (see
    :func:`hgbp_sim.defaults.load_defaults`); without one the built-in
    defaults apply.  ``params``, ``dt_ctrl``, ``T_amb`` and ``T_wi`` override it."""

    def __init__(self, params: PlantParams | None = None, dt_ctrl: float | None = None, dt_sim: float = 0.05,
                 seed: int | None = 0, history_len: int = 60000, T_amb: float | None = None,
                 T_wi: float | None = None, defaults: dict | None = None):
        self.defaults = defaults if defaults is not None else merge_defaults({})
        sim = self.defaults["simulation"]
        self.params = params if params is not None else PlantParams(**self.defaults["plant"])
        self.dt_ctrl = float(sim["dt_ctrl"] if dt_ctrl is None else dt_ctrl)
        self.dt_sim = float(dt_sim)
        self.rng = np.random.default_rng(seed)
        self.history_len = int(history_len)
        self.rows_recorded = 0              # history rows recorded so far (keeps counting past history_len)
        self.noise = bool(sim["noise"])
        self.short_cycle_timers = bool(sim["short_cycle_timers"])
        self.speed_factor = float(sim["speed_factor"])
        self.paused = False
        self.achieved_speed = 0.0          # set by a real-time runner (simulated s per wall s)
        self.step_count = 0
        self.T_amb = float(sim["T_amb"] if T_amb is None else T_amb) + C2K
        self.T_wi = float(sim["T_wi"] if T_wi is None else T_wi) + C2K
        self.pending_params: dict = {}
        self.charge_pending = 0.0           # kg still to add (+) / recover (-)
        self.charge_rate = float(sim["charge_rate_g_s"]) / 1000.0     # kg/s
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
        loops = self.defaults["loops"]
        # UT35A settings per loop: P, I, D, DR, input range RL..RH, output limits OL..OH
        self.tuning = {k: {n: v for n, v in loops[k].items() if n != "SP"} for k in LOOPS}
        self._apply_tuning()
        self.interlock = Interlock(1, MIN_OFF_TIME, MIN_RUN_TIME)
        self._apply_timers()
        self.run_request = False
        self.speed_sp = float(self.params.N_nom)
        self.sp = {info["pv"]: float(_TO_SI[info["unit"]](loops[k]["SP"])) for k, info in LOOPS.items()}
        self.history = {k: deque(maxlen=self.history_len) for k in HISTORY_CHANNELS}
        self.t = 0.0
        self.pv_filt = {k: None for k in LOOPS}   # each controller's filtered PV (SI), see _filter_pv

    def _T_s_for(self, pt: dict) -> float:
        """Suction temperature setpoint [K] giving the point's superheat."""
        return float(self.props.T_sat(np.array([pt["P_s"]]))[0]) + float(pt["SH"])

    def _apply_tuning(self) -> None:
        for k, pid in self._pids().items():
            t = self.tuning[k]
            gains = ut35a.to_gains(t["P"], t["I"], t["D"], t["DR"], t["RH"] - t["RL"])
            pid.Kp, pid.Ki, pid.Kd = (np.asarray(g / LOOPS[k]["scale"], float) for g in gains)
            pid.u_min, pid.u_max = t["OL"] / 100.0, t["OH"] / 100.0

    def _apply_timers(self) -> None:
        on = self.short_cycle_timers
        self.interlock.min_off_time = MIN_OFF_TIME if on else 0.0
        self.interlock.min_run_time = MIN_RUN_TIME if on else 0.0

    def _filter_pv(self, k: str, pv: float, dt: float) -> float:
        """The UT35A's PV input filter FL: a first-order lag with time constant FL on the
        PV input, ahead of both the PV display and the control computation (OFF: the PV
        passes unchanged).  Controller side only; the plant is not involved."""
        FL, prev = self.tuning[k].get("FL", ut35a.OFF), self.pv_filt[k]
        if FL == ut35a.OFF or prev is None:
            self.pv_filt[k] = float(pv)
        else:
            self.pv_filt[k] = prev + (1.0 - np.exp(-dt / FL)) * (float(pv) - prev)
        return self.pv_filt[k]

    def _pids(self) -> dict:
        return dict(dpv=self.ctrl.pid_1, spv=self.ctrl.pid_2, stv=self.ctrl.pid_3, water=self.ctrl.pid_4)

    def log(self, msg: str) -> None:
        self.events.appendleft(dict(t=round(self.t, 1), msg=msg))

    # ------------------------------------------------------- initialization
    def apply_pending(self) -> None:
        if self.pending_params:
            self.params = self.params.replace(**self.pending_params)
            self.pending_params = {}
        old = (self.mode, self.manual_out, self.tuning, self.sp, self.speed_sp, self.history, self.t)
        self._build()
        self.mode, self.manual_out, self.tuning, self.sp, self.speed_sp, self.history, self.t = old
        self._apply_tuning()

    def cold_start(self, T_amb: float | None = None, T_wi: float | None = None,
                   liquid_in_suction: float | None = None) -> None:
        """Equalized stand at ambient, compressor off, valves at rest positions."""
        self.apply_pending()
        if T_amb is not None:
            self.T_amb = T_amb + C2K
        if T_wi is not None:
            self.T_wi = T_wi + C2K
        self.plant.set_inputs(T_amb=self.T_amb, T_wi=self.T_wi)
        u0 = self.ctrl.u_off.copy()
        self.plant.cold_start(T_amb=self.T_amb, u_pos=u0[None, :], liquid_in_suction=liquid_in_suction)
        self.pv_filt = {k: None for k in LOOPS}   # the filters start from the new PVs
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
        self.pv_filt = {k: None for k in LOOPS}
        self.sp = dict(P_d=pt["P_d"], P_s=pt["P_s"], T_s=self._T_s_for(pt), P_i=pt["P_i"])
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
                 out: float | None = None, P=None, I=None, D=None, FL=None) -> None:
        """Operator actions on one loop.  ``sp`` in the loop's display unit and
        inside its input range; ``P`` [%], ``I`` and ``D`` [s or "OFF"] and the PV
        input filter ``FL`` [s or "OFF"] as on the UT35A (raises ValueError, before
        changing anything, if one is invalid)."""
        info = LOOPS[name]
        pid = self._pids()[name]
        tune = dict(self.tuning[name])
        if P is not None:
            tune["P"] = ut35a.normalize_band(P)
        if I is not None:
            tune["I"] = ut35a.normalize_time(I, "I")
        if D is not None:
            tune["D"] = ut35a.normalize_time(D, "D")
        if FL is not None:
            tune["FL"] = ut35a.normalize_filter(FL)
        if sp is not None:
            sp = float(sp)
            if not tune["RL"] <= sp <= tune["RH"]:
                raise ValueError(f"SP {sp} {info['unit']} is outside the input range {tune['RL']}..{tune['RH']}")
            self.sp[info["pv"]] = float(_TO_SI[info["unit"]](sp))
        if mode is not None and mode != self.mode[name]:
            self.mode[name] = mode
            if mode == "auto":
                pid.reset(self.manual_out[name])            # bumpless: start from the manual output
            else:
                self.manual_out[name] = float(pid.u[0])      # take over the current output
            self.log(f"{info['label']} loop -> {mode}")
        if out is not None:
            self.manual_out[name] = float(np.clip(out, 0.0, 1.0))
        if tune != self.tuning[name]:
            self.tuning[name] = tune
            self._apply_tuning()
            sec = lambda v: v if v == ut35a.OFF else f"{v} s"
            self.log(f"{info['label']} tuning: P {tune['P']} %, I {sec(tune['I'])}, D {sec(tune['D'])}, "
                     f"FL {sec(tune.get('FL', ut35a.OFF))}")

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
                noise: bool | None = None, dt_ctrl: float | None = None,
                charge_rate: float | None = None, short_cycle_timers: bool | None = None) -> None:
        """``charge_rate`` in kg/s (1 .. 250 g/s) for adding / recovering refrigerant;
        ``short_cycle_timers`` switches the minimum off / run times on or off."""
        if short_cycle_timers is not None and bool(short_cycle_timers) != self.short_cycle_timers:
            self.short_cycle_timers = bool(short_cycle_timers)
            self._apply_timers()
            self.log("anti-short-cycle timers " + ("on" if self.short_cycle_timers else "off"))
        if charge_rate is not None:
            self.charge_rate = float(np.clip(charge_rate, 0.001, 0.25))
        if paused is not None:
            self.paused = bool(paused)
        if speed_factor is not None:
            self.speed_factor = float(np.clip(speed_factor, 0.1, 100.0))
        if noise is not None:
            self.noise = bool(noise)
        if dt_ctrl is not None:
            self.dt_ctrl = float(np.clip(dt_ctrl, 0.1, 10.0))

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
            if name == "charge" and val is None:
                pass                                      # nominal charge
            elif m["kind"] == "float":
                val = clamp_to_range(name, float(val))
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
        """Current values, metadata (defaults from the defaults document),
        and the defaults document itself for the Settings tab."""
        vals, meta = {}, param_metadata()
        for m in meta:
            v = self.pending_params.get(m["name"], getattr(self.params, m["name"]))
            if m["name"] == "charge":
                v = float(self.plant.p.charge[0])
            vals[m["name"]] = v
            m["default"] = self.defaults["plant"][m["name"]]
        p = self.plant.p
        f = lambda v: float(np.atleast_1d(v)[0])
        derived = dict(volumes=volume_table(p), V_s=f(p.V_s), V_d=f(p.V_d), V_i=f(p.V_i),
                       C_mw=f(p.C_mw_cell) * self.plant.MX, C_cw=f(p.C_cw), C_sw=f(p.C_sw), C_dw=f(p.C_dw),
                       C_rw=f(p.C_rw))
        return dict(values=vals, meta=meta, pending=sorted(self.pending_params),
                    fluids=list(FLUIDS), named_points=NAMED_POINTS, derived=derived,
                    nominal_charge=float(self.plant.nominal_charge()[0]), defaults=self.defaults)

    def saturation(self, T_sat: float) -> dict:
        """Dew point (saturated vapor, x = 1) pressure [bar, absolute] of the stand's
        refrigerant at saturation temperature ``T_sat`` [degC], for compressor state points
        (the operator panel's calculator), with the tables' valid temperature range [degC]
        and whether ``T_sat`` lies inside it.  Inverts the tables' dew line on their own
        pressure grid; the model is not involved."""
        pr = self.props
        P = np.exp(np.linspace(np.log(pr.p_min), np.log(pr.p_max), pr.n_p))    # the tables' grid
        T_dew = pr.sat(P)["T_v"]
        T = float(T_sat) + C2K
        return dict(fluid=pr.fluid, T_sat=float(T_sat), P=float(np.interp(T, T_dew, P)) / 1e5,
                    T_min=float(T_dew[0]) - C2K, T_max=float(T_dew[-1]) - C2K,
                    in_range=bool(T_dew[0] <= T <= T_dew[-1]))

    # -------------------------------------------------------------- physics
    def _inject_charge(self, dm: float) -> None:
        """Add (dm > 0) liquid from a cylinder at ambient temperature at the
        receiver's charging port, or recover (dm < 0) what the liquid line
        carries there (liquid while the receiver keeps its seal).  The mass and
        energy states are updated exactly; the plant's projection makes (P, h)
        consistent at the next sub-step."""
        pl, pr = self.plant, self.props
        x = pl.x[0]
        if dm > 0:
            h_in = float(pr.sat(pr.P_sat(np.array([self.T_amb])))["h_l"][0])
        else:
            a = pl.outputs()
            seal = float(a["ll_fill"][0]) >= 1.0
            h_i = float(x[HGBPPlant.H_I])
            h_in = float(pr.sat(np.array([x[HGBPPlant.P_I]]))["h_l"][0]) if seal else h_i
        x[HGBPPlant.M_I] += dm
        x[HGBPPlant.U_I] += dm * h_in       # (P, h) follow through the conservation projection
        pl.aux = None

    def step(self) -> dict:
        """One control interval.  Returns the snapshot."""
        pl, p, dt = self.plant, self.plant.p, self.dt_ctrl
        meas = pl.measure(noise=self.noise)
        u = np.zeros(4)
        for k, info in LOOPS.items():
            pid = self._pids()[k]
            pv = self._filter_pv(k, meas[info["pv"]][0], dt)       # the controller sees the filtered PV
            if self.mode[k] == "auto":          # controls whether or not the compressor runs, like a UT35A
                u[info["valve"]] = pid.update(self.sp[info["pv"]], pv, dt)[0]
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
        p.charge[0] = pl.conserved_mass()[0]
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
            sp_P_d=self.sp["P_d"] / 1e5, sp_P_s=self.sp["P_s"] / 1e5, sp_T_s=self.sp["T_s"] - C2K, sp_P_i=self.sp["P_i"] / 1e5,
            sp_N=self.speed_sp, x_out=aux["x_out"][0], y_liq=aux["y_liq"][0], x_qo=aux["x_qo"][0],
            rec_level=aux["rec_level"][0], cond_flood=aux["cond_flood"][0], M_q_liq=aux["M_q_liq"][0],
            T_qo=aux["T_qo"][0] - C2K, T_go=aux["T_go"][0] - C2K,
            charge=self.plant.conserved_mass()[0], T_sh=aux["T_sh"][0] - C2K, T_cw=aux["T_cw"][0] - C2K,
            mdot_w=aux["mdot_w"][0] * 60.0, state=int(self.interlock.state[0]), Q_w=aux["Q_w"][0],
            Q_mx=aux["Q_mx"][0] / 1000.0,
        )
        for k, v in row.items():
            self.history[k].append(float(v))
        self.rows_recorded += 1
        self._last_row = row

    def last_row(self) -> dict:
        return dict(self._last_row)

    def rows_after(self, count: int) -> tuple[dict, int]:
        """History rows recorded after the first ``count`` rows (those still held),
        and the new count.  Costs only the new rows, however long the history."""
        k = min(self.rows_recorded - int(count), len(self.history["t"]))
        if k <= 0:
            return {}, self.rows_recorded
        return {name: list(islice(reversed(dq), k))[::-1] for name, dq in self.history.items()}, self.rows_recorded

    def rows_through(self, t0: float) -> int:
        """Row count up to and including the last row with time <= ``t0`` (a
        client that holds the history up to ``t0`` continues with :meth:`rows_after`)."""
        newer = 0
        for t in reversed(self.history["t"]):
            if t <= t0:
                break
            newer += 1
        return self.rows_recorded - newer

    def history_since(self, t0: float = -1.0, stride: int = 1, max_points: int | None = None) -> dict:
        """Rows with t > t0 (all rows if t0 < 0).  ``max_points`` picks a
        stride automatically so that at most that many rows are returned."""
        t = np.fromiter(self.history["t"], float)
        i0 = int(np.searchsorted(t, t0, side="right")) if t0 >= 0 else 0
        n = len(t) - i0
        if max_points is not None and n > max_points:
            stride = max(stride, int(np.ceil(n / max_points)))
        out = {}
        for k, dq in self.history.items():
            arr = np.fromiter(dq, float)[i0::stride]
            out[k] = arr.tolist()
        return out

    def _nominal_charge(self) -> float:
        """Nominal charge of the stand as built [kg] (follows the volumes: recomputed
        when the plant's parameters are repacked, i.e. rebuilt or reassigned)."""
        rec = self.plant.p.packed()
        if getattr(self, "_nc_rec", None) is not rec:
            self._nc_rec, self._nc = rec, float(self.plant.nominal_charge()[0])
        return self._nc

    def snapshot(self, aux=None, meas=None, trips=None) -> dict:
        pl, p = self.plant, self.plant.p
        aux = aux if aux is not None else pl.outputs()
        meas = meas if meas is not None else pl.measure(noise=False)
        trips = trips if trips is not None else pl.trips()
        f = lambda v: float(np.asarray(v).ravel()[0])
        running = f(aux["N"]) > 0.5 * f(p.N_min)
        loops = {}
        for k, info in LOOPS.items():
            conv = _FROM_SI[info["unit"]]
            loops[k] = dict(label=info["label"], valve_label=info["valve_label"], unit=info["unit"],
                            pv=float(conv(self.pv_filt[k] if self.pv_filt[k] is not None else f(meas[info["pv"]]))),
                            sp=float(conv(self.sp[info["pv"]])),
                            out=f(aux["u%d" % (info["valve"] + 1)]), mode=self.mode[k],
                            manual_out=self.manual_out[k], **self.tuning[k])
        perm_detail = dict(
            P_s_above_min=f(meas["P_s"]) > 1.5 * f(p.P_s_min), P_s_below_max=f(meas["P_s"]) < 0.9 * f(p.P_s_max),
            P_d_below_max=f(meas["P_d"]) < 0.8 * f(p.P_d_max), valve1_open=f(aux["u1"]) >= 0.1,
            valve2_open=f(aux["u2"]) >= 0.05, no_trip=not bool(self.interlock.tripped[0]),
            off_time_elapsed=bool(self.interlock.state[0] != ST_OFF or self.interlock.t_state[0] >= self.interlock.min_off_time),
        )
        return dict(
            t=self.t, step=self.step_count, paused=self.paused, speed_factor=self.speed_factor,
            achieved_speed=self.achieved_speed, noise=self.noise,
            dt_ctrl=self.dt_ctrl, fluid=self.params.fluid,
            compressor=dict(state=STATE_NAMES[int(self.interlock.state[0])], tripped=bool(self.interlock.tripped[0]),
                            trip_reasons=list(self.trip_reasons), run_request=self.run_request,
                            speed_sp=self.speed_sp, speed=f(aux["N"]), running=running,
                            permissive_ok=all(perm_detail.values()), permissives=perm_detail,
                            t_state=f(self.interlock.t_state), min_off_time=self.interlock.min_off_time,
                            min_run_time=self.interlock.min_run_time, short_cycle_timers=self.short_cycle_timers,
                            N_min=f(p.N_min), N_max=f(p.N_max)),
            loops=loops,
            meas=dict(P_s=f(meas["P_s"]) / 1e5, P_d=f(meas["P_d"]) / 1e5, P_i=f(meas["P_i"]) / 1e5,
                      T_s=f(meas["T_s"]) - C2K, T_d=f(meas["T_d"]) - C2K, T_co=f(meas["T_co"]) - C2K,
                      T_wi=self.T_wi - C2K, T_wo=f(aux["T_wo"]) - C2K, T_amb=self.T_amb - C2K,
                      Tsat_s=f(meas["T_sat_s"]) - C2K, Tsat_d=f(aux["T_sat_d"]) - C2K, Tsat_i=f(meas["T_sat_i"]) - C2K,
                      SH=f(meas["SH"]), SC=f(meas["SC"]), mdot=f(meas["mdot"]) * 1e3, W=f(meas["W"]),
                      N=f(aux["N"]), u1=f(aux["u1"]), u2=f(aux["u2"]), u3=f(aux["u3"]), u4=f(aux["u4"])),
            true=dict(x_out=f(aux["x_out"]), y_liq=f(aux["y_liq"]), x_i=f(aux["x_i"]), fill_i=f(aux["fill_i"]),
                      rec_level=f(aux["rec_level"]), ll_fill=f(aux["ll_fill"]), cond_flood=f(aux["cond_flood"]),
                      x_qo=f(aux["x_qo"]), T_qo=f(aux["T_qo"]) - C2K, T_go=f(aux["T_go"]) - C2K,
                      M_q_liq=f(aux["M_q_liq"]), Q_mx=f(aux["Q_mx"]),
                      x_q=[float(v) for v in aux["x_q"][0]], T_q=[float(v) - C2K for v in aux["T_q"][0]],
                      T_g=[float(v) - C2K for v in aux["T_g"][0]], T_mw=[float(v) - C2K for v in aux["T_mw"][0]],
                      x_c=[float(v) for v in aux["x_c"][0]], T_c=[float(v) - C2K for v in aux["T_c"][0]],
                      T_wc=[float(v) - C2K for v in aux["T_wc"][0]], T_cwc=[float(v) - C2K for v in aux["T_cwc"][0]],
                      T_sh=f(aux["T_sh"]) - C2K, T_cw=f(aux["T_cw"]) - C2K, T_rw=f(aux["T_rw"]) - C2K,
                      mdot_1=f(aux["mdot_1"]) * 1e3, mdot_2=f(aux["mdot_2"]) * 1e3, mdot_3=f(aux["mdot_3"]) * 1e3,
                      mdot_w=f(aux["mdot_w"]) * 60.0, Q_w=f(aux["Q_w"]), Q_r=f(aux["Q_r"]), W_el=f(aux["W_el"]),
                      eta_v=f(aux["eta_v"]), eta_s=f(aux["eta_s"]), Pr=f(aux["Pr"]),
                      M_s=f(aux["M_s"]), M_d=f(aux["M_d"]), M_i=f(aux["M_i"]),
                      P_h=f(aux["P_h"]) / 1e5,
                      Tsat_h=float(self.plant.props.T_sat(np.maximum(aux["P_h"][:1], self.plant.props.p_min))[0]) - C2K,
                      dP_cr=f(aux["dP_cr"]) / 1e3, dP_cw=f(aux["dP_cw"]) / 1e3,
                      dP_mg=f(aux["dP_mg"]) / 1e3, dP_mq=f(aux["dP_mq"]) / 1e3,
                      dP_suc=f(aux["dP_suc"]) / 1e3, dP_dis=f(aux["dP_dis"]) / 1e3, dP_hdr=f(aux["dP_hdr"]) / 1e3,
                      dP_bp=f(aux["dP_bp"]) / 1e3),
            alarms=dict(floodback=bool(trips["floodback"][0]) and running,
                        mixer_wet=bool(trips["mixer_wet"][0]) and running,
                        no_liquid_seal=bool(trips["no_liquid_seal"][0]),
                        receiver_full=bool(trips["receiver_full"][0]),
                        condenser_flooded=bool(trips["condenser_flooded"][0]),
                        high_T_d_warning=f(aux["T_d"]) > f(p.T_d_max) - 15.0,
                        high_P_d_warning=f(aux["P_d"]) > 0.9 * f(p.P_d_max)),
            limits=dict(P_d_max=f(p.P_d_max) / 1e5, P_s_min=f(p.P_s_min) / 1e5, P_s_max=f(p.P_s_max) / 1e5,
                        T_d_max=f(p.T_d_max) - C2K),
            charge=dict(kg=float(pl.conserved_mass()[0]), nominal_kg=self._nominal_charge(),
                        pending_kg=self.charge_pending, rate_kg_s=self.charge_rate),
            pending_params=sorted(self.pending_params),
            events=list(self.events)[:30],
        )
