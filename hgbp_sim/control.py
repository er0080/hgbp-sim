"""Baseline controllers for the HGBP stand (batched PID loops).

The baseline is the conventional way such stands are automated: four
single-input single-output PI loops

    discharge pressure    -> valve 1, discharge pressure valve   (reverse acting)
    suction pressure      -> valve 2, hot gas bypass valve       (direct acting)
    suction superheat     -> valve 3, liquid injection valve     (reverse acting)
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


# Tuned by batched gain sweeps over a 4-point schedule (see README).
DEFAULT_GAINS = dict(
    dpv=dict(Kp=-0.05e-5, Ki=-0.01e-5, Kd=0.0),      # discharge pressure -> valve 1 [per Pa], Ti = 5 s
    spv=dict(Kp=0.15e-5, Ki=0.03e-5, Kd=0.0),        # suction pressure -> valve 2 [per Pa], Ti = 5 s
    stv=dict(Kp=-0.008, Ki=-0.0016, Kd=0.0),         # superheat -> valve 3 [per K], Ti = 5 s
    water=dict(Kp=-0.40e-5, Ki=-0.0133e-5, Kd=0.0),  # intermediate pressure -> valve 4 [per Pa], Ti = 30 s
)


class BaselineController:
    """Four-loop PI controller producing absolute valve commands in [0, 1],
    ordered [valve 1, valve 2, valve 3, valve 4]."""

    def __init__(self, n: int = 1, gains: dict | None = None):
        g = DEFAULT_GAINS if gains is None else {**DEFAULT_GAINS, **gains}
        self.n = n
        self.pid_1 = PID(n=n, u_min=0.02, u_max=1.0, **g["dpv"])
        self.pid_2 = PID(n=n, u_min=0.0, u_max=1.0, **g["spv"])
        self.pid_3 = PID(n=n, u_min=0.0, u_max=1.0, **g["stv"])
        self.pid_4 = PID(n=n, u_min=0.02, u_max=1.0, **g["water"])
        # valve positions while the compressor is off
        self.u_off = np.array([0.5, 0.6, 0.0, 0.3])

    def reset(self, u0=None, idx=None) -> None:
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx))
        u0 = np.broadcast_to(np.asarray(self.u_off if u0 is None else u0, float), (len(idx), 4))
        self.pid_1.reset(u0[:, 0], idx)
        self.pid_2.reset(u0[:, 1], idx)
        self.pid_3.reset(u0[:, 2], idx)
        self.pid_4.reset(u0[:, 3], idx)

    def __call__(self, meas: dict, sp: dict, dt: float, running=None) -> np.ndarray:
        """``meas`` from ``HGBPPlant.measure``; ``sp`` has P_s, P_d, SH, P_i targets.
        Returns (n, 4) commands."""
        running = np.ones(self.n, bool) if running is None else np.asarray(running, bool)
        u1 = self.pid_1.update(sp["P_d"], meas["P_d"], dt, running)
        u2 = self.pid_2.update(sp["P_s"], meas["P_s"], dt, running)
        u3 = self.pid_3.update(sp["SH"], meas["SH"], dt, running)
        u4 = self.pid_4.update(sp["P_i"], meas["P_i"], dt, running)
        u = np.stack([u1, u2, u3, u4], axis=1)
        return np.where(running[:, None], u, self.u_off[None, :] * np.ones((self.n, 1)))
