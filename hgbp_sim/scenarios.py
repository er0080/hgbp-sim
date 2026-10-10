"""Operating envelope, named rating points and test schedules.

A test point is (P_s, P_d, RGT, P_i, N): suction pressure, discharge pressure,
return gas (suction) temperature, intermediate (liquid) pressure and compressor speed,
as a test procedure specifies it: saturated suction and discharge temperatures, a return
gas temperature and a VFD frequency.  The liquid pressure setpoint is the geometric mean
of the suction and discharge pressures, clamped from below by what the cooling water can
hold.

The named points below are the web UI's warm-start conditions (superheat and an
intermediate temperature); the environment's schedules come from :class:`Envelope`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C2K = 273.15
POINT_FIELDS = ("P_s", "P_d", "RGT", "P_i", "N")


# The compressor's operating envelope (manufacturer's R410A map at 10 K superheat):
# vertices (evaporating, condensing) [degC], counter-clockwise
R410A_ENVELOPE = ((-24.0, 20.0), (5.0, 20.0), (15.0, 30.0), (20.0, 40.0), (20.0, 55.0),
                  (15.0, 65.0), (0.0, 65.0), (-10.0, 60.0), (-20.0, 50.0), (-24.0, 37.0))


def inside_polygon(x, y, poly) -> np.ndarray:
    """Points (x, y) inside the polygon ``poly`` (vertex list), by ray casting."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = np.zeros(np.broadcast(x, y).shape, bool)
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if y1 == y2:
            continue
        cross = ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        inside ^= cross
    return inside


@dataclass
class Envelope:
    """Where test points are sampled (temperatures in degC): inside the compressor's
    operating envelope (``polygon``, evaporating vs condensing temperature), with the
    stand's own feasibility limits on top."""
    polygon: tuple = R410A_ENVELOPE
    RGT: tuple = (-10.0, 35.0)            # return gas (suction) temperature
    p_RGT_standard: float = 0.3           # share of points at the standard return gas temperature
    RGT_standard: float = 18.33           # 65 degF
    SH: tuple = (5.0, 50.0)               # superheat RGT - T_evap a point may imply [K]
    N_frac: tuple = (35.0 / 60.0, 75.0 / 60.0)   # fraction of nominal speed: 35-75 Hz, nominal at 60 Hz
    Pr_max: float = 14.0                  # max pressure ratio
    dT_lift_min: float = 20.0             # min T_cond - T_evap
    dT_int_above_water: float = 6.0       # liquid saturation temperature at least this far above the water inlet
    dT_int_below_cond: float = 4.0        # ... and at least this far below T_cond
    T_d_margin: float = 25.0              # estimated discharge temperature must stay this far below the trip
                                          # (the startup transient overshoots the steady estimate)

    @property
    def T_evap(self):
        return (min(v[0] for v in self.polygon), max(v[0] for v in self.polygon))

    @property
    def T_cond(self):
        return (min(v[1] for v in self.polygon), max(v[1] for v in self.polygon))

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
        out = {k: np.zeros(n) for k in POINT_FIELDS + ("SH", "T_evap", "T_cond", "T_int")}
        todo = np.ones(n, bool)
        N_lo = max(params.N_min, self.N_frac[0] * params.N_nom)
        N_hi = min(params.N_max, self.N_frac[1] * params.N_nom)
        for _ in range(500):
            m = int(todo.sum())
            if m == 0:
                break
            Te = rng.uniform(*self.T_evap, m)
            Tc = rng.uniform(*self.T_cond, m)
            in_env = inside_polygon(Te, Tc, self.polygon)
            rgt = np.where(rng.uniform(size=m) < self.p_RGT_standard, self.RGT_standard,
                           rng.uniform(*self.RGT, m))
            sh = rgt - Te
            N = rng.uniform(N_lo, N_hi, m)
            P_s = props.P_sat(Te + C2K)
            P_d = props.P_sat(Tc + C2K)
            P_i = np.maximum(np.sqrt(P_s * P_d), props.P_sat(T_wi[todo] + self.dT_int_above_water))
            Ti = props.T_sat(P_i) - C2K
            T_d_est = self.estimate_T_d(props, params, P_s, P_d, sh)
            ok = in_env & (P_d / P_s <= self.Pr_max) & (Tc - Te >= self.dT_lift_min) \
                & (sh >= self.SH[0]) & (sh <= self.SH[1]) & (Ti <= Tc - self.dT_int_below_cond) \
                & (T_d_est < params.T_d_max - self.T_d_margin)
            idx = np.flatnonzero(todo)[ok]
            out["P_s"][idx], out["P_d"][idx], out["RGT"][idx] = P_s[ok], P_d[ok], rgt[ok] + C2K
            out["P_i"][idx], out["N"][idx], out["SH"][idx] = P_i[ok], N[ok], sh[ok]
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

    Returns dict of arrays: ``points`` (n, k_max, 5) = [P_s, P_d, RGT, P_i, N] (SI),
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
