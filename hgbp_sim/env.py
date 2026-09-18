"""Gymnasium-style environments for learning closed-loop control of the stand.

Two flavours:

* :class:`HGBPVecEnv` - batched core.  ``n`` stands integrate in lock-step
  with numpy; use it for custom training loops (RL or imitation) at high
  throughput.
* :class:`HGBPEnv` - single-environment ``gymnasium.Env`` wrapper (n = 1).

Observation (``OBS_NAMES``, ``OBS_GROUPS``)
    Measured process values in refrigerant-agnostic form (pressures as
    saturation temperatures, mass flow and power normalized by swept volume
    and speed), valve positions, setpoints, tracking errors and clipped
    integrated errors in kelvin, compressor and refrigerant context
    (swept volume, nominal speed, physical descriptors of the fluid) and the
    status of the start/stop interlock.  All entries are scaled to O(1).

Action (``[-1, 1]``)
    a[0:4]  valves 1..4 (discharge pressure, suction pressure / hot gas
            bypass, suction temperature / liquid, cooling water);
            ``action_mode="incremental"`` (default) moves each command by
            ``a * max_rate``, ``"absolute"`` maps u = (a + 1) / 2
    a[4]    run request (only with ``start_stop_action=True``): > 0 asks the
            interlock state machine to run the compressor, < 0 to stop it.
            Without this action the compressor is started automatically
            after a random delay and stopped after the last test point.

Interlock state machine (always active, mirrors what a PLC would enforce)
    OFF -> STARTING when a run request is present, the permissives hold
    (suction/discharge pressure in range, valves 1 and 2 open, no trip) and
    the anti-short-cycle off time has elapsed; STARTING -> RUNNING once the
    speed is up; RUNNING -> STOPPING on a stop request after the minimum run
    time; STOPPING -> OFF once the compressor has stopped.

Episode
    A schedule of test points (P_s, P_d, SH, P_i, N).  A point is *completed*
    when suction pressure, discharge pressure and superheat have stayed inside
    the tolerance band for ``dwell_required`` seconds (the stand is "stable"
    in the test-standard sense) or when its maximum hold time elapses.  After
    the last point the compressor must be stopped (shutdown phase); the
    episode terminates successfully once it is off.

Reward (per control step)
    minus the weighted normalized tracking errors while running, a bonus while
    inside the tolerance band, a bonus when a test point completes by dwell,
    an actuator-movement penalty, penalties for liquid at the compressor inlet
    and for approaching the discharge-temperature limit, penalties for idling
    when a start is possible / running during shutdown / blocked start
    requests, a shutdown bonus, and a large penalty (with termination) on a
    safety trip.

``info`` carries the true (noise-free) process values, the true charge
factor and a ``steady`` flag (in tolerance for ``steady_time``) as training
labels for a charge estimator, and the privileged states for an asymmetric
critic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .control import BaselineController
from .params import PlantParams
from .plant import HGBPPlant
from .scenarios import Envelope, sample_schedule
from .steady_state import solve_steady_state

C2K = 273.15

OBS_GROUPS = {
    "measurements": ("Tsat_s", "Tsat_d", "Tsat_i", "T_s", "T_d", "T_co", "SH", "SC",
                     "mdot_norm", "W_norm", "T_wi", "T_wo", "T_amb", "N_rel",
                     "u1", "u2", "u3", "u4"),
    "setpoints": ("Tsat_s_sp", "Tsat_d_sp", "SH_sp", "Tsat_i_sp", "N_sp_rel"),
    "errors": ("e_Tsat_s", "e_Tsat_d", "e_SH", "e_Tsat_i",
               "ie_Tsat_s", "ie_Tsat_d", "ie_SH", "ie_Tsat_i"),
    "context": ("V_disp_rel", "N_nom_rel",
                "rf_T_crit", "rf_P_crit", "rf_M_molar", "rf_P_sat_ref", "rf_h_fg_ref",
                "rf_rho_v_ref", "rf_rho_l_ref", "rf_dPsat_dT_ref"),
    "status": ("running", "run_required", "permissive_ok",
               "st_off", "st_starting", "st_running", "st_stopping",
               "t_point", "t_in_tol", "t_since_switch"),
}
OBS_NAMES = tuple(n for g in OBS_GROUPS.values() for n in g)
_OBS_SCALE = np.array(
    [50.0, 50.0, 50.0, 50.0, 150.0, 80.0, 30.0, 20.0, 1.0, 3.0, 50.0, 50.0, 50.0, 1.0,
     1.0, 1.0, 1.0, 1.0]
    + [50.0, 50.0, 30.0, 50.0, 1.0]
    + [10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0]
    + [1.0] * 10
    + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 600.0, 600.0, 600.0])
assert len(_OBS_SCALE) == len(OBS_NAMES)

# interlock states
ST_OFF, ST_STARTING, ST_RUNNING, ST_STOPPING = 0, 1, 2, 3
STATE_NAMES = ("OFF", "STARTING", "RUNNING", "STOPPING")


@dataclass
class EnvConfig:
    dt_ctrl: float = 1.0                 # control interval [s]
    dt_sim: float = 0.05                 # integration step [s]
    integrator: str = "rk4"
    episode_time: float = 2400.0         # hard truncation [s]
    k_points: tuple = (1, 3)             # number of test points per episode
    hold_time: tuple = (300.0, 900.0)    # maximum time per test point [s]
    advance_on_dwell: bool = True        # complete a point once stable for dwell_required
    dwell_required: float = 180.0        # [s] contiguous time inside the tolerance band
    steady_time: float = 120.0           # [s] in tolerance -> info["steady"] (charge-estimation label)
    shutdown_phase: bool = True          # require compressor stop after the last point
    start_mode: str = "random"           # "cold" | "warm" | "random"
    p_cold: float = 0.3                  # probability of a cold start in "random"
    start_delay: tuple = (5.0, 30.0)     # automatic start: delay before the run request [s]
    p_warm_at_setpoint: float = 0.5      # warm start exactly at first point (else at a random point)
    cold_liquid_in_accumulator: tuple = (0.0, 1.0)   # sampled per cold start
    action_mode: str = "incremental"     # "absolute" | "incremental"
    max_rate: float = 0.05               # incremental: max command change per step
    start_stop_action: bool = False      # 5th action = run request
    min_off_time: float = 60.0           # anti-short-cycle [s]
    min_run_time: float = 120.0          # minimum run time before a stop is honoured [s]
    randomize_params: bool = True
    randomize_spec: dict | None = None
    noise: bool = True
    T_amb_range: tuple = (15.0, 35.0)    # degC
    T_wi_range: tuple = (12.0, 30.0)     # degC
    charge_range: tuple = (0.85, 1.15)   # factor on nominal charge (normal episodes)
    charge_extreme_range: tuple = (0.3, 1.8)   # under-/over-charged episodes
    p_charge_extreme: float = 0.15
    envelope: Envelope = field(default_factory=Envelope)
    filter_feasible: bool = True         # resample points the stand cannot reach at nominal charge
    feasible_u_max: float = 0.95         # a point needs every valve below this at equilibrium
    include_integrated_error: bool = True
    ie_clip: float = 5.0
    # reward shaping (errors in kelvin of saturation temperature / superheat)
    err_scale: tuple = (1.0, 0.7, 1.0, 1.0)     # Tsat_s, Tsat_d, SH, Tsat_i
    err_weight: tuple = (1.0, 1.0, 1.0, 0.5)
    tol: tuple = (0.3, 0.2, 0.5)                # tolerance band on Tsat_s, Tsat_d, SH [K]
    err_cap: float = 10.0
    w_track: float = 1.0
    w_tol_bonus: float = 0.5
    w_point_done: float = 5.0
    w_action: float = 0.05
    w_floodback: float = 2.0
    w_T_d: float = 1.0
    w_idle: float = 0.5
    w_blocked_start: float = 0.2
    w_run_in_shutdown: float = 0.5
    w_shutdown_done: float = 5.0
    trip_penalty: float = 50.0
    terminate_on_trip: bool = True
    auto_reset: bool = True


class HGBPVecEnv:
    """Batched environment (``n`` independent stands, one refrigerant)."""

    def __init__(self, n: int = 1, config: EnvConfig | None = None,
                 params: PlantParams | None = None, seed: int | None = None):
        self.cfg = config if config is not None else EnvConfig()
        self.n = int(n)
        self.rng = np.random.default_rng(seed)
        self.params = params if params is not None else PlantParams()
        cfg = self.cfg
        self.plant = HGBPPlant(self.params, n=self.n, dt=cfg.dt_sim, integrator=cfg.integrator,
                               rng=self.rng, randomize=cfg.randomize_params,
                               randomize_spec=cfg.randomize_spec)
        self.props = self.plant.props
        self.expert = BaselineController(self.n)
        self.obs_dim = len(OBS_NAMES)
        self.act_dim = 5 if cfg.start_stop_action else 4
        self._rf_vec = self.props.descriptor_vector()
        n_ = self.n
        self.t_ep = np.zeros(n_)
        self.t_point = np.zeros(n_)
        self.t_in_tol = np.zeros(n_)
        self.t_state = np.zeros(n_)
        self.state = np.zeros(n_, int)
        self.run_required = np.ones(n_, bool)
        self.schedule_complete = np.zeros(n_, bool)
        self.k = np.zeros(n_, int)
        self.sched_points = np.zeros((n_, 4, 5))
        self.sched_hold = np.zeros((n_, 4))
        self.sched_k = np.ones(n_, int)
        self.sp = np.zeros((n_, 5))          # P_s, P_d, SH, P_i, N
        self.start_at = np.zeros(n_)
        self.u_cmd = np.zeros((n_, 4))
        self.ie = np.zeros((n_, 4))
        self.charge_factor = np.ones(n_)
        self.meas: dict | None = None
        self.episode_return = np.zeros(n_)
        self.episode_len = np.zeros(n_, int)
        self.points_completed = np.zeros(n_, int)
        self.reset()

    # ------------------------------------------------------------- helpers
    def _tsat(self, P):
        return self.props.T_sat(np.maximum(P, self.props.p_min)) - C2K

    def _running(self):
        return self.plant.x[:, HGBPPlant.N_] > 0.5 * self.params.N_min

    def _permissives(self):
        m, p, x = self.meas, self.plant.p, self.plant.x
        return ((m["P_s"] > 1.5 * p.P_s_min) & (m["P_s"] < 0.9 * p.P_s_max)
                & (m["P_d"] < 0.8 * p.P_d_max)
                & (x[:, HGBPPlant.U1] >= 0.1) & (x[:, HGBPPlant.U2] >= 0.05))

    def _errors_K(self, P_s, P_d, SH, P_i):
        return np.stack([self._tsat(self.sp[:, 0]) - self._tsat(P_s),
                         self._tsat(self.sp[:, 1]) - self._tsat(P_d),
                         self.sp[:, 2] - SH,
                         self._tsat(self.sp[:, 3]) - self._tsat(P_i)], 1)

    # ------------------------------------------------------------------ reset
    def reset(self, idx=None) -> np.ndarray:
        cfg, pl, rng = self.cfg, self.plant, self.rng
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx))
        m = len(idx)
        if m == 0:
            return self._observe()
        if cfg.randomize_params:
            pl.resample_params(idx)
        T_amb = rng.uniform(*cfg.T_amb_range, m) + C2K
        T_wi = rng.uniform(*cfg.T_wi_range, m) + C2K
        pl.T_amb[idx], pl.T_wi[idx] = T_amb, T_wi
        extreme = rng.uniform(size=m) < cfg.p_charge_extreme
        fac = np.where(extreme, rng.uniform(*cfg.charge_extreme_range, m), rng.uniform(*cfg.charge_range, m))
        self.charge_factor[idx] = fac
        pl.p.charge[idx] = pl.nominal_charge(idx) * fac

        sch = sample_schedule(rng, m, 4, cfg.envelope, pl.props, self.params, T_wi,
                              hold_range=cfg.hold_time, k_range=cfg.k_points)
        if cfg.filter_feasible:
            self._filter_feasible(sch, idx)
        self.sched_points[idx], self.sched_hold[idx], self.sched_k[idx] = sch["points"], sch["hold"], sch["k"]
        self.k[idx] = 0
        self.t_point[idx] = 0.0
        self.t_in_tol[idx] = 0.0
        self.t_ep[idx] = 0.0
        self.sp[idx] = sch["points"][:, 0, :]
        self.ie[idx] = 0.0
        self.run_required[idx] = True
        self.schedule_complete[idx] = False
        self.episode_return[idx] = 0.0
        self.episode_len[idx] = 0
        self.points_completed[idx] = 0

        if cfg.start_mode == "cold":
            cold = np.ones(m, bool)
        elif cfg.start_mode == "warm":
            cold = np.zeros(m, bool)
        else:
            cold = rng.uniform(size=m) < cfg.p_cold

        warm_idx = idx[~cold]
        if len(warm_idx):
            mw = len(warm_idx)
            at_sp = rng.uniform(size=mw) < cfg.p_warm_at_setpoint
            alt = cfg.envelope.sample(rng, mw, pl.props, self.params, pl.T_wi[warm_idx])
            first = self.sp[warm_idx]
            tgt = {f: np.where(at_sp, first[:, c], alt[f]) for c, f in enumerate(("P_s", "P_d", "SH", "P_i", "N"))}
            res = solve_steady_state(pl, tgt["P_s"], tgt["P_d"], tgt["SH"], tgt["N"], P_i=tgt["P_i"],
                                     idx=warm_idx)
            ok = res["converged"]
            good = warm_idx[ok]
            if len(good):
                pl.set_state(good, res["x"][ok])
                self.u_cmd[good] = res["u"][ok]
                pl.u_cmd[good] = res["u"][ok]
                pl.N_cmd[good] = tgt["N"][ok]
                self.start_at[good] = -1.0
                self.state[good] = ST_RUNNING
                self.t_state[good] = cfg.min_run_time
                self.expert.reset(res["u"][ok], good)
            cold = cold | np.isin(idx, warm_idx[~ok])

        cold_idx = idx[cold]
        if len(cold_idx):
            mc = len(cold_idx)
            u0 = np.broadcast_to(self.expert.u_off, (mc, 4))
            liq = rng.uniform(*cfg.cold_liquid_in_accumulator, mc)
            pl.cold_start(cold_idx, T_amb=pl.T_amb[cold_idx], u_pos=u0, liquid_in_accumulator=liq)
            self.u_cmd[cold_idx] = u0
            self.start_at[cold_idx] = rng.uniform(*cfg.start_delay, mc)
            self.state[cold_idx] = ST_OFF
            self.t_state[cold_idx] = cfg.min_off_time
            self.expert.reset(u0, cold_idx)
        pl.aux = None
        self.meas = pl.measure(noise=cfg.noise)
        return self._observe()

    def _filter_feasible(self, sch, idx, rounds: int = 3) -> None:
        """Replace scheduled points that have no equilibrium (with all valves
        below ``feasible_u_max``) for the environment's parameters at nominal
        charge.  Points still infeasible after ``rounds`` resamples are kept."""
        cfg, pl, rng = self.cfg, self.plant, self.rng
        pts, k_act = sch["points"], sch["k"]
        for j in range(pts.shape[1]):
            todo = np.flatnonzero(k_act > j)
            for _ in range(rounds):
                if len(todo) == 0:
                    break
                q = pts[todo, j, :]
                res = solve_steady_state(pl, q[:, 0], q[:, 1], q[:, 2], q[:, 4], P_i=q[:, 3],
                                         fill=0.4, idx=idx[todo])
                bad = ~(res["converged"] & (res["u"] < cfg.feasible_u_max).all(1))
                if not bad.any():
                    break
                new = cfg.envelope.sample(rng, int(bad.sum()), pl.props, self.params, pl.T_wi[idx[todo[bad]]])
                for c, f in enumerate(("P_s", "P_d", "SH", "P_i", "N")):
                    pts[todo[bad], j, c] = new[f]
                todo = todo[bad]

    # ------------------------------------------------------------------- step
    def step(self, action):
        cfg, pl = self.cfg, self.plant
        a = np.clip(np.asarray(action, float).reshape(self.n, self.act_dim), -1.0, 1.0)
        if cfg.action_mode == "absolute":
            u_new = 0.5 * (a[:, :4] + 1.0)
        else:
            u_new = np.clip(self.u_cmd + a[:, :4] * cfg.max_rate, 0.0, 1.0)
        du = u_new - self.u_cmd
        self.u_cmd = u_new
        if cfg.start_stop_action:
            run_req = a[:, 4] > 0.0
        else:
            run_req = (self.t_ep >= self.start_at) & self.run_required

        # ---- interlock state machine (evaluated on the last measurements)
        perm = self._permissives()
        st = self.state.copy()
        off_ok = (st == ST_OFF) & (self.t_state >= cfg.min_off_time)
        blocked = (st == ST_OFF) & run_req & ~(perm & off_ok)
        start = (st == ST_OFF) & run_req & perm & off_ok
        stop = np.isin(st, (ST_STARTING, ST_RUNNING)) & ~run_req & (self.t_state >= cfg.min_run_time)
        N = pl.x[:, HGBPPlant.N_]
        up = (st == ST_STARTING) & (N >= 0.9 * self.sp[:, 4])
        down = (st == ST_STOPPING) & (N < 0.5 * self.params.N_min)
        new = st.copy()
        new[start] = ST_STARTING
        new[up] = ST_RUNNING
        new[stop] = ST_STOPPING
        new[down] = ST_OFF
        switched = new != st
        self.state = new
        self.t_state = np.where(switched, 0.0, self.t_state + cfg.dt_ctrl)
        N_cmd = np.where(np.isin(self.state, (ST_STARTING, ST_RUNNING)), self.sp[:, 4], 0.0)

        aux = pl.step(cfg.dt_ctrl, u_cmd=self.u_cmd, N_cmd=N_cmd)
        self.meas = pl.measure(noise=cfg.noise)
        self.t_ep += cfg.dt_ctrl
        self.t_point += cfg.dt_ctrl
        running = self._running()

        # ---- tracking (true values, kelvin)
        e = self._errors_K(aux["P_s"], aux["P_d"], aux["SH"], aux["P_i"])
        en = np.minimum(np.abs(e) / np.asarray(cfg.err_scale), cfg.err_cap)
        wts = np.asarray(cfg.err_weight)
        in_tol = (np.abs(e[:, :3]) < np.asarray(cfg.tol)).all(1) & running & (aux["x_out"] >= 1.0)
        self.t_in_tol = np.where(in_tol, self.t_in_tol + cfg.dt_ctrl, 0.0)
        steady = self.t_in_tol >= cfg.steady_time
        active = running & self.run_required
        r_track = -cfg.w_track * (en * wts).sum(1) / wts.sum() * active
        r_bonus = cfg.w_tol_bonus * in_tol
        r_action = -cfg.w_action * np.abs(du).sum(1)
        r_flood = -cfg.w_floodback * ((aux["x_out"] < 1.0) & running)
        r_Td = -cfg.w_T_d * np.maximum(0.0, (aux["T_d"] - (pl.p.T_d_max - 15.0)) / 15.0)
        idle = self.run_required & (self.state == ST_OFF) & perm & off_ok & ~run_req
        r_idle = -cfg.w_idle * idle * cfg.start_stop_action
        r_blocked = -cfg.w_blocked_start * blocked
        r_shut = -cfg.w_run_in_shutdown * (~self.run_required & running)
        trips = pl.trips()
        tripped = trips["high_P_d"] | trips["low_P_s"] | trips["high_P_s"] | trips["high_T_d"]
        r_trip = -cfg.trip_penalty * tripped

        # ---- measured errors for the integral features
        em = self._errors_K(self.meas["P_s"], self.meas["P_d"], self.meas["SH"], self.meas["P_i"]) \
            / np.asarray(cfg.err_scale)
        self.ie = np.clip(self.ie + em * cfg.dt_ctrl / 60.0, -cfg.ie_clip, cfg.ie_clip) * running[:, None]

        # ---- schedule progression: completion by dwell or by timeout
        by_dwell = cfg.advance_on_dwell & (self.t_in_tol >= cfg.dwell_required) & self.run_required
        by_timeout = (self.t_point >= self.sched_hold[np.arange(self.n), self.k]) & self.run_required
        done_point = by_dwell | by_timeout
        r_point = cfg.w_point_done * by_dwell
        self.points_completed += by_dwell
        last = self.k >= self.sched_k - 1
        advance = done_point & ~last
        if advance.any():
            self.k[advance] += 1
            self.t_point[advance] = 0.0
            self.t_in_tol[advance] = 0.0
            self.sp[advance] = self.sched_points[advance, self.k[advance], :]
            self.ie[advance] = 0.0
        finished = done_point & last
        self.schedule_complete |= finished
        if cfg.shutdown_phase:
            self.run_required = self.run_required & ~finished
            complete = self.schedule_complete & (self.state == ST_OFF)
        else:
            complete = self.schedule_complete
        r_done = cfg.w_shutdown_done * complete

        terminated = (tripped & cfg.terminate_on_trip) | complete
        truncated = (self.t_ep >= cfg.episode_time) & ~terminated
        reward = (r_track + r_bonus + r_action + r_flood + r_Td + r_idle + r_blocked
                  + r_shut + r_point + r_done + r_trip)
        self.episode_return += reward
        self.episode_len += 1
        obs = self._observe()
        info = dict(
            true=dict(P_s=aux["P_s"], P_d=aux["P_d"], P_i=aux["P_i"], SH=aux["SH"], SC=aux["SC"],
                      T_s=aux["T_s"], T_d=aux["T_d"], x_out=aux["x_out"], x_s=aux["x_s"], x_i=aux["x_i"],
                      fill_s=aux["fill_s"], fill_i=aux["fill_i"], mdot=aux["mdot_c"], W=aux["W_el"],
                      mdot_1=aux["mdot_1"], mdot_2=aux["mdot_2"], mdot_3=aux["mdot_3"], mdot_w=aux["mdot_w"],
                      N=aux["N"], charge=aux["charge"], T_sh=aux["T_sh"], T_cw=aux["T_cw"]),
            charge_factor=self.charge_factor.copy(), steady=steady, t_in_tol=self.t_in_tol.copy(),
            error=e, in_tol=in_tol, running=running, state=self.state.copy(), permissive_ok=perm,
            run_required=self.run_required.copy(), point_done=by_dwell, schedule_complete=self.schedule_complete.copy(),
            trips=trips, tripped=tripped,
            r_track=r_track, r_bonus=r_bonus, r_action=r_action, r_flood=r_flood, r_Td=r_Td,
            r_idle=r_idle, r_blocked=r_blocked, r_shut=r_shut, r_point=r_point, r_done=r_done,
            u_cmd=self.u_cmd.copy(), setpoint=self.sp.copy(), t=self.t_ep.copy(),
        )
        done = terminated | truncated
        if done.any():
            info["episode_return"] = np.where(done, self.episode_return, np.nan)
            info["episode_len"] = np.where(done, self.episode_len, -1)
            info["episode_points_completed"] = np.where(done, self.points_completed, -1)
            if cfg.auto_reset:
                info["final_obs"] = obs.copy()
                self.reset(np.flatnonzero(done))
                obs = self._observe()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ observation
    def _observe(self) -> np.ndarray:
        m, cfg, p = self.meas, self.cfg, self.plant.p
        running = self._running()
        n = self.n
        Tsat_s, Tsat_d, Tsat_i = self._tsat(m["P_s"]), self._tsat(m["P_d"]), self._tsat(m["P_i"])
        e = self._errors_K(m["P_s"], m["P_d"], m["SH"], m["P_i"])
        ie = self.ie if cfg.include_integrated_error else np.zeros_like(self.ie)
        rho_ref = self.props.sat(np.maximum(m["P_s"], self.props.p_min))["rho_v"]
        swept = p.V_disp * np.maximum(m["N"], 1.0) / 60.0
        on = m["N"] > 0.5 * p.N_min
        mdot_norm = np.where(on, m["mdot"] / (rho_ref * swept), 0.0)
        W_norm = np.where(on, m["W"] / (swept * m["P_s"]), 0.0)
        st = self.state
        raw = np.concatenate([
            np.stack([Tsat_s, Tsat_d, Tsat_i, m["T_s"] - C2K, m["T_d"] - C2K, m["T_co"] - C2K,
                      m["SH"], m["SC"], mdot_norm, W_norm, m["T_wi"] - C2K, m["T_wo"] - C2K,
                      m["T_amb"] - C2K, m["N"] / p.N_nom, m["u1"], m["u2"], m["u3"], m["u4"]], 1),
            np.stack([self._tsat(self.sp[:, 0]), self._tsat(self.sp[:, 1]), self.sp[:, 2],
                      self._tsat(self.sp[:, 3]), self.sp[:, 4] / p.N_nom], 1),
            e, ie,
            np.stack([p.V_disp / 250e-6, p.N_nom / 1500.0], 1),
            np.broadcast_to(self._rf_vec, (n, len(self._rf_vec))),
            np.stack([running, self.run_required, self._permissives(),
                      st == ST_OFF, st == ST_STARTING, st == ST_RUNNING, st == ST_STOPPING,
                      np.minimum(self.t_point, 600.0), np.minimum(self.t_in_tol, 600.0),
                      np.minimum(self.t_state, 600.0)], 1).astype(float),
        ], axis=1)
        return raw / _OBS_SCALE

    # ----------------------------------------------------------------- expert
    def expert_action(self) -> np.ndarray:
        """Action the baseline PID controller would take now (for imitation)."""
        sp = dict(P_s=self.sp[:, 0], P_d=self.sp[:, 1], SH=self.sp[:, 2], P_i=self.sp[:, 3])
        u = self.expert(self.meas, sp, self.cfg.dt_ctrl, running=self._running())
        if self.cfg.action_mode == "absolute":
            a = 2.0 * u - 1.0
        else:
            a = np.clip((u - self.u_cmd) / self.cfg.max_rate, -1.0, 1.0)
        if self.cfg.start_stop_action:
            a = np.concatenate([a, np.where(self.run_required, 1.0, -1.0)[:, None]], 1)
        return a


try:  # optional gymnasium wrapper
    import gymnasium as gym

    class HGBPEnv(gym.Env):
        """Single-stand gymnasium environment."""
        metadata = {"render_modes": []}

        def __init__(self, config: EnvConfig | None = None, params: PlantParams | None = None,
                     seed: int | None = None):
            cfg = config if config is not None else EnvConfig()
            cfg.auto_reset = False
            self.vec = HGBPVecEnv(1, cfg, params, seed)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.vec.obs_dim,), np.float64)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (self.vec.act_dim,), np.float64)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.vec.rng = np.random.default_rng(seed)
                self.vec.plant.rng = self.vec.rng
            obs = self.vec.reset()
            return obs[0], {}

        def step(self, action):
            obs, r, term, trunc, info = self.vec.step(np.asarray(action)[None, :])
            info1 = {k: (v[0] if isinstance(v, np.ndarray) and v.shape[:1] == (1,) else v)
                     for k, v in info.items() if k not in ("true", "trips")}
            info1["true"] = {k: float(v[0]) for k, v in info["true"].items()}
            info1["trips"] = {k: bool(v[0]) for k, v in info["trips"].items()}
            return obs[0], float(r[0]), bool(term[0]), bool(trunc[0]), info1

        def expert_action(self):
            return self.vec.expert_action()[0]

        @property
        def plant(self):
            return self.vec.plant

except ImportError:  # pragma: no cover
    HGBPEnv = None  # type: ignore
