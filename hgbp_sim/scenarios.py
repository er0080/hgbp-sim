"""Operating envelope, named rating points and test schedules.

A test point is (P_s, P_d, SH, P_i, N): suction pressure, discharge pressure,
suction superheat, intermediate (condensing) pressure and compressor speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C2K = 273.15
POINT_FIELDS = ("P_s", "P_d", "SH", "P_i", "N")


@dataclass
class Envelope:
    """Ranges from which test points are sampled (temperatures in degC)."""
    T_evap: tuple = (-30.0, 12.0)
    T_cond: tuple = (30.0, 65.0)          # saturation temperature at discharge pressure
    SH: tuple = (3.0, 25.0)
    N_frac: tuple = (0.5, 1.4)            # fraction of nominal speed
    Pr_max: float = 14.0                  # max pressure ratio
    dT_lift_min: float = 20.0             # min T_cond - T_evap
    dT_int_above_water: tuple = (6.0, 15.0)   # T_int - T_water_in range
    dT_int_below_cond: float = 4.0        # T_int at least this far below T_cond
    dT_int_above_evap: float = 12.0       # T_int at least this far above T_evap
    T_int_max: float = 58.0
    T_d_margin: float = 8.0               # estimated discharge temperature must stay this far below the trip

    def estimate_T_d(self, props, params, P_s, P_d, SH):
        """Rough discharge temperature from the compressor map (adiabatic,
        nominal isentropic efficiency, no shell losses)."""
        h1 = props.h_PT(P_s, props.T_sat(P_s) + SH)
        s1 = props.state(P_s, h1, need_s=True).s
        h2s = props.h_Ps_vapor(P_d, s1)
        h2 = h1 + (h2s - h1) / params.eta_s0
        return props.T_vapor(P_d, h2)

    def sample(self, rng: np.random.Generator, n: int, props, params, T_wi) -> dict:
        """Rejection-sample ``n`` feasible test points.  Returns SI arrays."""
        T_wi = np.broadcast_to(np.asarray(T_wi, float), (n,))
        out = {k: np.zeros(n) for k in POINT_FIELDS + ("T_evap", "T_cond", "T_int")}
        todo = np.ones(n, bool)
        N_lo = max(params.N_min, self.N_frac[0] * params.N_nom)
        N_hi = min(params.N_max, self.N_frac[1] * params.N_nom)
        for _ in range(500):
            m = int(todo.sum())
            if m == 0:
                break
            Te = rng.uniform(*self.T_evap, m)
            Tc = rng.uniform(*self.T_cond, m)
            sh = rng.uniform(*self.SH, m)
            N = rng.uniform(N_lo, N_hi, m)
            Twi = T_wi[todo] - C2K
            Ti_lo = np.maximum(Twi + rng.uniform(*self.dT_int_above_water, m), Te + self.dT_int_above_evap)
            Ti_hi = np.minimum(Tc - self.dT_int_below_cond, self.T_int_max)
            Ti = Ti_lo + rng.uniform(0.0, 1.0, m) * np.maximum(Ti_hi - Ti_lo, 0.0)
            P_s = props.P_sat(Te + C2K)
            P_d = props.P_sat(Tc + C2K)
            P_i = props.P_sat(Ti + C2K)
            T_d_est = self.estimate_T_d(props, params, P_s, P_d, sh)
            ok = (P_d / P_s <= self.Pr_max) & (Tc - Te >= self.dT_lift_min) & (Ti_hi > Ti_lo) \
                & (T_d_est < params.T_d_max - self.T_d_margin)
            idx = np.flatnonzero(todo)[ok]
            out["P_s"][idx], out["P_d"][idx], out["SH"][idx] = P_s[ok], P_d[ok], sh[ok]
            out["P_i"][idx], out["N"][idx] = P_i[ok], N[ok]
            out["T_evap"][idx], out["T_cond"][idx], out["T_int"][idx] = Te[ok], Tc[ok], Ti[ok]
            todo[idx] = False
        if todo.any():
            raise RuntimeError("could not sample feasible test points; check envelope / water temperature")
        return out


# Named rating-type conditions (degC evaporating / condensing / intermediate, K superheat)
NAMED_POINTS = {
    "MT_standard": dict(T_evap=-10.0, T_cond=45.0, T_int=38.0, SH=10.0),
    "LT_standard": dict(T_evap=-30.0, T_cond=40.0, T_int=34.0, SH=10.0),
    "HT_standard": dict(T_evap=5.0, T_cond=50.0, T_int=42.0, SH=10.0),
    "AC_standard": dict(T_evap=7.2, T_cond=54.4, T_int=45.0, SH=11.1),
    "high_lift": dict(T_evap=-25.0, T_cond=60.0, T_int=45.0, SH=10.0),
    "low_lift": dict(T_evap=0.0, T_cond=35.0, T_int=30.0, SH=10.0),
}


def named_point(name: str, props, N: float) -> dict:
    d = NAMED_POINTS[name]
    return dict(P_s=float(props.P_sat(d["T_evap"] + C2K)), P_d=float(props.P_sat(d["T_cond"] + C2K)),
                SH=float(d["SH"]), P_i=float(props.P_sat(d["T_int"] + C2K)), N=float(N),
                T_evap=d["T_evap"], T_cond=d["T_cond"], T_int=d["T_int"])


def sample_schedule(rng: np.random.Generator, n: int, k_max: int, envelope: Envelope,
                    props, params, T_wi, hold_range=(300.0, 900.0), k_range=(1, 4)) -> dict:
    """Random multi-point schedules for ``n`` environments.

    Returns dict of arrays: ``points`` (n, k_max, 5) = [P_s, P_d, SH, P_i, N],
    ``hold`` (n, k_max) seconds, ``k`` (n,) number of active points.
    """
    pts = np.zeros((n, k_max, len(POINT_FIELDS)))
    hold = np.zeros((n, k_max))
    k = np.minimum(rng.integers(k_range[0], k_range[1] + 1, size=n), k_max)
    for j in range(k_max):
        s = envelope.sample(rng, n, props, params, T_wi)
        for c, f in enumerate(POINT_FIELDS):
            pts[:, j, c] = s[f]
        hold[:, j] = rng.uniform(*hold_range, size=n)
    return dict(points=pts, hold=hold, k=k)
