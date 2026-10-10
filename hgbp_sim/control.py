"""Baseline controllers for the HGBP stand (batched PID loops).

The baseline is the conventional way such stands are automated: four
single-input single-output PI loops

    discharge pressure    -> valve 1, discharge pressure valve   (reverse acting)
    suction pressure      -> valve 2, hot gas bypass valve       (direct acting)
    suction temperature   -> valve 3, quench valve               (reverse acting)
    intermediate pressure -> valve 4, cooling water valve        (reverse acting)

It serves as a sanity check of the plant, as a comparison for learned
controllers and as an "expert" for imitation learning / warm-starting RL.
"""
from __future__ import annotations

import numpy as np


class PID:
    """Discrete PI(D) controller vectorized over a batch of loops.

    * error e = sp - y; a negative ``Kp``/``Ki`` gives reverse action
    * derivative on measurement with first-order filter ``tau_d``
    * output clamped to [u_min, u_max] with conditional integration
      (integrator freezes when pushing further into saturation)
    * gains may be scalars or per-loop arrays of shape (n,)
    """

    def __init__(self, Kp, Ki, Kd=0.0, u_min: float = 0.0, u_max: float = 1.0,
                 tau_d: float = 2.0, n: int = 1):
        self.Kp, self.Ki, self.Kd = (np.asarray(g, float) for g in (Kp, Ki, Kd))
        self.u_min, self.u_max, self.tau_d = float(u_min), float(u_max), float(tau_d)
        self.n = n
        self.I = np.zeros(n)
        self.y_f = np.full(n, np.nan)
        self.d = np.zeros(n)
        self.u = np.zeros(n)

    def reset(self, u0=0.0, idx=None) -> None:
        """Bumpless (re)initialization: output equals ``u0`` at zero error."""
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx))
        u0 = np.broadcast_to(np.asarray(u0, float), (len(idx),))
        self.I[idx] = np.clip(u0, self.u_min, self.u_max)
        self.y_f[idx] = np.nan
        self.d[idx] = 0.0
        self.u[idx] = self.I[idx]

    def update(self, sp, y, dt: float, active=None):
        if self.n == 1 and active is None and np.ndim(sp) == 0 and np.ndim(y) == 0:
            return self._update_one(float(sp), float(y), dt)
        sp = np.broadcast_to(np.asarray(sp, float), (self.n,))
        y = np.broadcast_to(np.asarray(y, float), (self.n,))
        active = np.ones(self.n, bool) if active is None else np.asarray(active, bool)
        e = sp - y
        first = np.isnan(self.y_f)
        self.y_f = np.where(first, y, self.y_f)
        if np.any(self.Kd != 0.0):
            a = dt / (self.tau_d + dt)
            y_new = self.y_f + a * (y - self.y_f)
            self.d = -self.Kd * (y_new - self.y_f) / dt
            self.y_f = y_new
        u_unsat = self.Kp * e + self.I + self.d
        pushing = ((u_unsat > self.u_max) & (self.Ki * e > 0)) | ((u_unsat < self.u_min) & (self.Ki * e < 0))
        integrate = active & ~pushing
        self.I = np.where(integrate, self.I + self.Ki * e * dt, self.I)
        self.u = np.where(active, np.clip(self.Kp * e + self.I + self.d, self.u_min, self.u_max), self.u)
        return self.u.copy()

    def _update_one(self, sp: float, y: float, dt: float):
        """``update`` for a single active loop in plain floats (the live stand's
        loops; the same arithmetic without the array overhead)."""
        Kp, Ki, Kd = self.Kp.item(), self.Ki.item(), self.Kd.item()
        I, d, y_f = self.I.item(), self.d.item(), self.y_f.item()
        e = sp - y
        if y_f != y_f:                  # first call
            y_f = y
        if Kd != 0.0:
            a = dt / (self.tau_d + dt)
            y_new = y_f + a * (y - y_f)
            d = -Kd * (y_new - y_f) / dt
            y_f = y_new
        u_unsat = Kp * e + I + d
        pushing = (u_unsat > self.u_max and Ki * e > 0) or (u_unsat < self.u_min and Ki * e < 0)
        if not pushing:
            I = I + Ki * e * dt
        u = min(max(Kp * e + I + d, self.u_min), self.u_max)
        self.I[0], self.d[0], self.y_f[0], self.u[0] = I, d, y_f, u
        return np.array([u])


# Tuned by batched gain sweeps over a 4-point schedule (see README); the superheat loop
# retuned for the mixing exchanger, whose suction temperature responds over about a minute.
DEFAULT_GAINS = dict(
    dpv=dict(Kp=-0.05e-5, Ki=-0.01e-5, Kd=0.0),      # discharge pressure -> valve 1 [per Pa], Ti = 5 s
    spv=dict(Kp=0.15e-5, Ki=0.03e-5, Kd=0.0),        # suction pressure -> valve 2 [per Pa], Ti = 5 s
    stv=dict(Kp=-0.003, Ki=-0.00005, Kd=0.0),        # suction temperature -> valve 3 [per K], Ti = 60 s
    water=dict(Kp=-0.40e-5, Ki=-0.0133e-5, Kd=0.0),  # intermediate pressure -> valve 4 [per Pa], Ti = 30 s
)


LOOP_KEYS = ("dpv", "spv", "stv", "water")
_LOOP_PV = dict(dpv="P_d", spv="P_s", stv="T_s", water="P_i")
_U_LIMITS = dict(dpv=(0.02, 1.0), spv=(0.0, 1.0), stv=(0.0, 1.0), water=(0.02, 1.0))


class BaselineController:
    """Four-loop PI controller producing absolute valve commands in [0, 1],
    ordered [valve 1, valve 2, valve 3, valve 4].

    ``gains`` and ``limits`` (output range per loop) override the built-in tuning;
    ``pv_filter`` gives a PV input filter time constant [s] per loop (the UT35A's FL,
    a first-order lag ahead of the PID; missing or None: no filter).  A stand's own
    settings come from its defaults file, see :func:`hgbp_sim.defaults.controller_settings`."""

    def __init__(self, n: int = 1, gains: dict | None = None, limits: dict | None = None,
                 pv_filter: dict | None = None):
        g = DEFAULT_GAINS if gains is None else {**DEFAULT_GAINS, **gains}
        lim = _U_LIMITS if limits is None else {**_U_LIMITS, **limits}
        self.n = n
        self.pid_1, self.pid_2, self.pid_3, self.pid_4 = (
            PID(n=n, u_min=lim[k][0], u_max=lim[k][1], **g[k]) for k in LOOP_KEYS)
        self.pv_filter = {k: (pv_filter or {}).get(k) for k in LOOP_KEYS}
        self._pv_f = {k: np.full(n, np.nan) for k in LOOP_KEYS}
        # valve positions while the compressor is off
        self.u_off = np.array([0.5, 0.6, 0.0, 0.3])

    def reset(self, u0=None, idx=None) -> None:
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx))
        u0 = np.broadcast_to(np.asarray(self.u_off if u0 is None else u0, float), (len(idx), 4))
        self.pid_1.reset(u0[:, 0], idx)
        self.pid_2.reset(u0[:, 1], idx)
        self.pid_3.reset(u0[:, 2], idx)
        self.pid_4.reset(u0[:, 3], idx)
        for k in LOOP_KEYS:
            self._pv_f[k][idx] = np.nan             # the filters restart from the next PV

    def _pv(self, k: str, y, dt: float):
        """The PV the loop sees: the measurement through the loop's input filter."""
        FL = self.pv_filter[k]
        if not FL:
            return y
        prev = self._pv_f[k]
        self._pv_f[k] = np.where(np.isnan(prev), y, prev + (1.0 - np.exp(-dt / FL)) * (y - prev))
        return self._pv_f[k]

    def __call__(self, meas: dict, sp: dict, dt: float, running=None) -> np.ndarray:
        """``meas`` from ``HGBPPlant.measure``; ``sp`` has P_d, P_s, T_s (return gas
        temperature) and P_i targets.  Returns (n, 4) commands."""
        running = np.ones(self.n, bool) if running is None else np.asarray(running, bool)
        pids = (self.pid_1, self.pid_2, self.pid_3, self.pid_4)
        u = np.stack([pid.update(sp[_LOOP_PV[k]], self._pv(k, meas[_LOOP_PV[k]], dt), dt, running)
                      for k, pid in zip(LOOP_KEYS, pids)], axis=1)
        return np.where(running[:, None], u, self.u_off[None, :] * np.ones((self.n, 1)))
