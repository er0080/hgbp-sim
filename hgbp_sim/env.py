"""Gymnasium-style environments for learning closed-loop control of the stand.

Two flavours:

* :class:`HGBPVecEnv` - batched core.  ``n`` stands integrate in lock-step
  with numpy; use it for custom training loops (RL or imitation) at high
  throughput.
* :class:`HGBPEnv` - single-environment ``gymnasium.Env`` wrapper (n = 1).

Observation (see ``OBS_NAMES``): measured process values (with sensor lag and
noise), actual valve positions, the current test-point setpoints, tracking
errors and (optionally) clipped integrated errors, all normalized to O(1).

Action: 4 values in [-1, 1] for valves 1..4 (discharge pressure, suction
pressure / hot gas bypass, suction temperature / liquid, cooling water)

* ``action_mode="incremental"`` -> u_cmd += a * max_rate  (default)
* ``action_mode="absolute"``    -> valve commands u = (a + 1) / 2

Reward (per control step): normalized tracking error of suction pressure,
discharge pressure, superheat and (lower weight) intermediate pressure, an
in-tolerance bonus, an actuator movement penalty, penalties for liquid
reaching the compressor and for approaching the discharge-temperature limit,
and a large penalty (with termination) on a safety trip.

Refrigerant charge is sampled per episode (nominal times a random factor,
with a configurable share of clearly under-/over-charged episodes).
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

OBS_NAMES = (
    "P_s", "P_d", "P_i", "T_s", "T_d", "T_co", "SH", "SC", "mdot", "W",
    "T_wi", "T_wo", "T_amb", "N",
    "u1", "u2", "u3", "u4",
    "P_s_sp", "P_d_sp", "SH_sp", "P_i_sp", "N_sp",
    "e_P_s", "e_P_d", "e_SH", "e_P_i",
    "ie_P_s", "ie_P_d", "ie_SH", "ie_P_i",
    "running", "t_point",
)
_OBS_SCALE = np.array([
    10e5, 30e5, 30e5, 50.0, 150.0, 80.0, 30.0, 20.0, 0.1, 5e3,
    50.0, 50.0, 50.0, 2000.0,
    1.0, 1.0, 1.0, 1.0,
    10e5, 30e5, 30.0, 30e5, 2000.0,
    1e5, 2e5, 10.0, 2e5,
    1.0, 1.0, 1.0, 1.0,
    1.0, 600.0])


@dataclass
class EnvConfig:
    dt_ctrl: float = 1.0                 # control interval [s]
    dt_sim: float = 0.05                 # integration step [s]
    integrator: str = "rk4"
    episode_time: float = 1800.0         # truncation [s]
    k_points: tuple = (1, 3)             # number of test points per episode
    hold_time: tuple = (300.0, 900.0)    # dwell per test point [s]
    start_mode: str = "random"           # "cold" | "warm" | "random"
    p_cold: float = 0.3                  # probability of a cold start in "random"
    start_delay: tuple = (5.0, 30.0)     # cold start: compressor start command time [s]
    p_warm_at_setpoint: float = 0.5      # warm start exactly at first point (else at a random point)
    cold_liquid_in_accumulator: tuple = (0.0, 1.0)   # sampled per cold start
    action_mode: str = "incremental"     # "absolute" | "incremental"
    max_rate: float = 0.05               # incremental: max command change per step
    randomize_params: bool = True
    randomize_spec: dict | None = None
    noise: bool = True
    T_amb_range: tuple = (15.0, 35.0)    # degC
    T_wi_range: tuple = (12.0, 30.0)     # degC
    charge_range: tuple = (0.85, 1.15)   # factor on nominal charge (normal episodes)
    charge_extreme_range: tuple = (0.3, 1.8)   # under-/over-charged episodes
    p_charge_extreme: float = 0.15
    envelope: Envelope = field(default_factory=Envelope)
    include_integrated_error: bool = True
    ie_clip: float = 5.0
    # reward shaping
    err_scale: tuple = (0.10e5, 0.20e5, 1.0, 0.30e5)  # Pa, Pa, K, Pa
    err_weight: tuple = (1.0, 1.0, 1.0, 0.5)
    tol: tuple = (0.02e5, 0.05e5, 0.5)         # test-standard style band on P_s, P_d, SH
    err_cap: float = 10.0
    w_track: float = 1.0
    w_tol_bonus: float = 0.5
    w_action: float = 0.05
    w_floodback: float = 2.0
    w_T_d: float = 1.0
    trip_penalty: float = 50.0
    terminate_on_trip: bool = True
    auto_reset: bool = True


class HGBPVecEnv:
    """Batched environment (``n`` independent stands)."""

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
        self.expert = BaselineController(self.n)
        self.obs_dim = len(OBS_NAMES)
        self.act_dim = 4
        n_ = self.n
        self.t_ep = np.zeros(n_)
        self.t_point = np.zeros(n_)
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
        self.reset()

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
        # refrigerant charge
        extreme = rng.uniform(size=m) < cfg.p_charge_extreme
        fac = np.where(extreme, rng.uniform(*cfg.charge_extreme_range, m), rng.uniform(*cfg.charge_range, m))
        self.charge_factor[idx] = fac
        pl.p.charge[idx] = pl.nominal_charge(idx) * fac

        sch = sample_schedule(rng, m, 4, cfg.envelope, pl.props, self.params, T_wi,
                              hold_range=cfg.hold_time, k_range=cfg.k_points)
        self.sched_points[idx], self.sched_hold[idx], self.sched_k[idx] = sch["points"], sch["hold"], sch["k"]
        self.k[idx] = 0
        self.t_point[idx] = 0.0
        self.t_ep[idx] = 0.0
        self.sp[idx] = sch["points"][:, 0, :]
        self.ie[idx] = 0.0
        self.episode_return[idx] = 0.0
        self.episode_len[idx] = 0

        if cfg.start_mode == "cold":
            cold = np.ones(m, bool)
        elif cfg.start_mode == "warm":
            cold = np.zeros(m, bool)
        else:
            cold = rng.uniform(size=m) < cfg.p_cold

        # --- warm starts: equilibrium at the first point or at a random point
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
                self.expert.reset(res["u"][ok], good)
            cold_fallback = np.isin(idx, warm_idx[~ok])
            cold = cold | cold_fallback

        cold_idx = idx[cold]
        if len(cold_idx):
            mc = len(cold_idx)
            u0 = np.broadcast_to(self.expert.u_off, (mc, 4))
            liq = rng.uniform(*cfg.cold_liquid_in_accumulator, mc)
            pl.cold_start(cold_idx, T_amb=pl.T_amb[cold_idx], u_pos=u0, liquid_in_accumulator=liq)
            self.u_cmd[cold_idx] = u0
            self.start_at[cold_idx] = rng.uniform(*cfg.start_delay, mc)
            self.expert.reset(u0, cold_idx)
        pl.aux = None
        self.meas = pl.measure(noise=cfg.noise)
        return self._observe()

    # ------------------------------------------------------------------- step
    def _running(self):
        return self.plant.x[:, HGBPPlant.N_] > 0.5 * self.params.N_min

    def step(self, action):
        cfg, pl = self.cfg, self.plant
        a = np.clip(np.asarray(action, float).reshape(self.n, 4), -1.0, 1.0)
        if cfg.action_mode == "absolute":
            u_new = 0.5 * (a + 1.0)
        else:
            u_new = np.clip(self.u_cmd + a * cfg.max_rate, 0.0, 1.0)
        du = u_new - self.u_cmd
        self.u_cmd = u_new
        N_cmd = np.where(self.t_ep >= self.start_at, self.sp[:, 4], 0.0)
        aux = pl.step(cfg.dt_ctrl, u_cmd=self.u_cmd, N_cmd=N_cmd)
        self.meas = pl.measure(noise=cfg.noise)
        self.t_ep += cfg.dt_ctrl
        self.t_point += cfg.dt_ctrl
        running = self._running()

        # --- reward from true (noise-free) process values
        e = np.stack([self.sp[:, 0] - aux["P_s"], self.sp[:, 1] - aux["P_d"],
                      self.sp[:, 2] - aux["SH"], self.sp[:, 3] - aux["P_i"]], 1)
        en = np.minimum(np.abs(e) / np.asarray(cfg.err_scale), cfg.err_cap)
        wts = np.asarray(cfg.err_weight)
        in_tol = (np.abs(e[:, :3]) < np.asarray(cfg.tol)).all(1) & running & (aux["x_out"] >= 1.0)
        r_track = -cfg.w_track * (en * wts).sum(1) / wts.sum() * running
        r_bonus = cfg.w_tol_bonus * in_tol
        r_action = -cfg.w_action * np.abs(du).sum(1)
        r_flood = -cfg.w_floodback * ((aux["x_out"] < 1.0) & running)
        Td_margin = np.maximum(0.0, (aux["T_d"] - (pl.p.T_d_max - 15.0)) / 15.0)
        r_Td = -cfg.w_T_d * Td_margin
        trips = pl.trips()
        tripped = trips["high_P_d"] | trips["low_P_s"] | trips["high_P_s"] | trips["high_T_d"]
        r_trip = -cfg.trip_penalty * tripped
        reward = r_track + r_bonus + r_action + r_flood + r_Td + r_trip
        # --- integrated (measured) error features
        em = np.stack([self.sp[:, 0] - self.meas["P_s"], self.sp[:, 1] - self.meas["P_d"],
                       self.sp[:, 2] - self.meas["SH"], self.sp[:, 3] - self.meas["P_i"]], 1) / np.asarray(cfg.err_scale)
        self.ie = np.clip(self.ie + em * cfg.dt_ctrl / 60.0, -cfg.ie_clip, cfg.ie_clip) * running[:, None]

        # --- schedule progression
        advance = (self.t_point >= self.sched_hold[np.arange(self.n), self.k]) & (self.k < self.sched_k - 1)
        if advance.any():
            self.k[advance] += 1
            self.t_point[advance] = 0.0
            self.sp[advance] = self.sched_points[advance, self.k[advance], :]
            self.ie[advance] = 0.0
        terminated = tripped & cfg.terminate_on_trip
        truncated = (self.t_ep >= cfg.episode_time) & ~terminated
        self.episode_return += reward
        self.episode_len += 1
        obs = self._observe()
        info = dict(
            true=dict(P_s=aux["P_s"], P_d=aux["P_d"], P_i=aux["P_i"], SH=aux["SH"], SC=aux["SC"],
                      T_s=aux["T_s"], T_d=aux["T_d"], x_out=aux["x_out"], x_s=aux["x_s"], x_i=aux["x_i"],
                      fill_s=aux["fill_s"], fill_i=aux["fill_i"], mdot=aux["mdot_c"], W=aux["W_el"],
                      mdot_1=aux["mdot_1"], mdot_2=aux["mdot_2"], mdot_3=aux["mdot_3"], mdot_w=aux["mdot_w"],
                      N=aux["N"], charge=aux["charge"]),
            charge_factor=self.charge_factor.copy(),
            error=e, in_tol=in_tol, running=running, trips=trips, tripped=tripped,
            r_track=r_track, r_bonus=r_bonus, r_action=r_action, r_flood=r_flood, r_Td=r_Td,
            u_cmd=self.u_cmd.copy(), setpoint=self.sp.copy(), t=self.t_ep.copy(),
        )
        done = terminated | truncated
        if done.any():
            info["episode_return"] = np.where(done, self.episode_return, np.nan)
            info["episode_len"] = np.where(done, self.episode_len, -1)
            if cfg.auto_reset:
                info["final_obs"] = obs.copy()
                self.reset(np.flatnonzero(done))
                obs = self._observe()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ observation
    def _observe(self) -> np.ndarray:
        m, cfg = self.meas, self.cfg
        running = self._running()
        e = np.stack([self.sp[:, 0] - m["P_s"], self.sp[:, 1] - m["P_d"],
                      self.sp[:, 2] - m["SH"], self.sp[:, 3] - m["P_i"]], 1)
        ie = self.ie if cfg.include_integrated_error else np.zeros_like(self.ie)
        raw = np.stack([
            m["P_s"], m["P_d"], m["P_i"], m["T_s"] - C2K, m["T_d"] - C2K, m["T_co"] - C2K,
            m["SH"], m["SC"], m["mdot"], m["W"],
            m["T_wi"] - C2K, m["T_wo"] - C2K, m["T_amb"] - C2K, m["N"],
            m["u1"], m["u2"], m["u3"], m["u4"],
            self.sp[:, 0], self.sp[:, 1], self.sp[:, 2], self.sp[:, 3], self.sp[:, 4],
            e[:, 0], e[:, 1], e[:, 2], e[:, 3],
            ie[:, 0], ie[:, 1], ie[:, 2], ie[:, 3],
            running.astype(float), np.minimum(self.t_point, 600.0),
        ], 1)
        return raw / _OBS_SCALE

    # ----------------------------------------------------------------- expert
    def expert_action(self) -> np.ndarray:
        """Action the baseline PID controller would take now (for imitation)."""
        sp = dict(P_s=self.sp[:, 0], P_d=self.sp[:, 1], SH=self.sp[:, 2], P_i=self.sp[:, 3])
        u = self.expert(self.meas, sp, self.cfg.dt_ctrl, running=self._running())
        if self.cfg.action_mode == "absolute":
            return 2.0 * u - 1.0
        return np.clip((u - self.u_cmd) / self.cfg.max_rate, -1.0, 1.0)


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
