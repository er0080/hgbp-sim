"""Component sub-models (valves, actuators, compressor, flow resistances, helpers).

Written once for scalars and compiled with numba: the ``@vectorize`` functions
are ufuncs (they take numpy arrays and broadcast) and are called with scalars
inside the model kernel; ``@njit`` functions are kernel-only.  Thin Python
wrappers give the ones with optional arguments their defaults.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, vectorize

from .kernel import props as kp

_vec = vectorize(cache=True)


# ------------------------------------------------------------------ helpers
@_vec
def regroot(z, eps):
    """Regularized signed square root: ~ sign(z)*sqrt(|z|) for |z| >> eps and
    ~ z/sqrt(eps) for |z| << eps.  Smooth at zero (avoids infinite gain of the
    orifice equation when the pressure difference vanishes)."""
    return z / (z * z + eps * eps) ** 0.25


def clip(x, lo, hi):
    """Faster equivalent of np.clip for arrays with scalar/array bounds."""
    return np.minimum(np.maximum(x, lo), hi)


@_vec
def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


CHAR_LINEAR, CHAR_EQPCT, CHAR_QUICK = 0, 1, 2
_CHAR = {"linear": CHAR_LINEAR, "eqpct": CHAR_EQPCT, "quick": CHAR_QUICK}


@_vec
def valve_fraction(u, code, R):
    """Installed flow fraction f(u) in [0, 1] for stem position u in [0, 1];
    ``code`` 0 linear, 1 modified equal-percentage (f(0) = 0, f(1) = 1), 2 quick opening."""
    u = min(max(u, 0.0), 1.0)
    if code == CHAR_LINEAR:
        return u
    if code == CHAR_EQPCT:
        return (R ** u - 1.0) / (R - 1.0)
    return math.sqrt(u)


def valve_characteristic(u, kind: str, R):
    """Installed flow fraction f(u) for characteristic ``kind`` (linear, eqpct, quick)."""
    if kind not in _CHAR:
        raise ValueError(f"unknown valve characteristic {kind!r}")
    return valve_fraction(u, _CHAR[kind], R)


# ------------------------------------------------------------------- valves
# Valves are sized by their Kv: the water flow [m^3/h] the fully open valve
# passes at a 1 bar drop.  With Q = Kv sqrt(dP[bar] / SG) that is
#     mdot [kg/s] = (Kv / 36000) sqrt(rho [kg/m^3] dP [Pa])
# so the flow functions below take C = Kv / 36000 in kg/s per sqrt(Pa kg/m^3).
KV_TO_C = 1.0 / 36000.0


@_vec
def kv_to_C(Kv):
    """Flow coefficient C [kg/s per sqrt(Pa kg/m^3)] of a valve rated ``Kv``
    [m^3/h].  Scale by the installed flow fraction f(u) before use."""
    return Kv * KV_TO_C


@njit(cache=True, inline="always")
def _series_share(C, rho_up, C_hx, rho_hx):
    """Share of a series pair's pressure drop taken by the valve (coefficient
    ``C`` at upstream density ``rho_up``), the rest by a resistance ``C_hx``
    at density ``rho_hx`` (none if ``C_hx`` <= 0); both obey m = C sqrt(rho dP)."""
    if C_hx <= 0.0:
        return 1.0
    r = C / max(C_hx, 1e-12)
    return 1.0 / (1.0 + r * r * rho_up / max(rho_hx, 1e-9))


@_vec
def gas_valve_flow_s(C, P1, P2, rho1, rho2, kappa, xT, eps, C_hx, rho_hx):
    """:func:`gas_valve_flow` with every argument given (``C_hx`` <= 0: no series resistance)."""
    dP = P1 - P2
    fwd = dP >= 0.0
    P_up = P1 if fwd else P2
    rho_up = rho1 if fwd else rho2
    Fk = kappa / 1.4
    x = abs(dP) / P_up
    x_ch = Fk * xT
    Y = 1.0 - min(x, x_ch) / (3.0 * x_ch)
    if C_hx > 0.0:
        for _ in range(2):          # the valve's share and its expansion factor
            share = _series_share(C * Y, rho_up, C_hx, rho_hx)
            Y = 1.0 - min(x * share, x_ch) / (3.0 * x_ch)
    share = _series_share(C * Y, rho_up, C_hx, rho_hx)
    x_eff = min(x, x_ch / share)
    dPe = math.copysign(x_eff * P_up, dP) if dP != 0.0 else 0.0
    return C * Y * math.sqrt(share * rho_up) * regroot(dPe, eps)


def gas_valve_flow(C, P1, P2, rho1, rho2, kappa, xT, eps, C_hx=None, rho_hx=None):
    """Compressible flow through a valve/orifice (ISA-style with choking).

    Positive flow is from side 1 to side 2.  ``rho1``/``rho2`` are the
    densities on each side; the upstream one is used.  ``C_hx`` optionally
    adds a resistance in series downstream (a heat exchanger side, density
    ``rho_hx``): the pair shares ``P1 - P2``, and the valve chokes on its own
    share of it.
    """
    return gas_valve_flow_s(C, P1, P2, rho1, rho2, kappa, xT, eps,
                            0.0 if C_hx is None else C_hx, 1.0 if rho_hx is None else rho_hx)


@_vec
def liquid_valve_flow_s(C, P1, P2, rho1, rho2, f_choke, eps, C_hx, rho_hx):
    """:func:`liquid_valve_flow` with every argument given (``C_hx`` <= 0: no series resistance)."""
    dP = P1 - P2
    fwd = dP >= 0.0
    P_up = P1 if fwd else P2
    rho_up = rho1 if fwd else rho2
    share = _series_share(C, rho_up, C_hx, rho_hx)
    dPe = math.copysign(min(abs(dP), f_choke * P_up / share), dP) if dP != 0.0 else 0.0
    return C * math.sqrt(share * rho_up) * regroot(dPe, eps)


def liquid_valve_flow(C, P1, P2, rho1, rho2, f_choke, eps, C_hx=None, rho_hx=None):
    """Incompressible/flashing flow through an expansion valve, signed;
    optionally with a series resistance ``C_hx`` as in :func:`gas_valve_flow`."""
    return liquid_valve_flow_s(C, P1, P2, rho1, rho2, f_choke, eps,
                               0.0 if C_hx is None else C_hx, 1.0 if rho_hx is None else rho_hx)


@_vec
def martin_xi(Re, beta_deg):
    """Single-phase Darcy friction factor of a chevron plate channel (Martin 1996,
    VDI Heat Atlas): dP = xi L / d_h rho u^2 / 2; ``beta_deg`` is the corrugation
    angle to the flow, d_h = 2 b / phi."""
    Re = max(Re, 1e-6)
    if Re < 2000.0:
        xi0 = 64.0 / Re
        xi1 = 597.0 / Re + 3.85
    else:
        xi0 = (1.8 * math.log10(Re) - 1.5) ** -2
        xi1 = 39.0 / Re ** 0.289
    be = math.radians(beta_deg)
    c = math.cos(be)
    inv = c / math.sqrt(0.18 * math.tan(be) + 0.36 * math.sin(be) + xi0 / c) + (1.0 - c) / math.sqrt(3.8 * xi1)
    return inv ** -2


@_vec
def amalfi_ftp(G, d_h, rho_m, rho_l, rho_v, sigma, beta_deg):
    """Two-phase Fanning friction factor of flow boiling in plate heat exchangers
    (Amalfi, Vakili-Farahani & Thome 2016, 1513 points, 13 studies):
    dP = 2 f L G^2 / (d_h rho_m), rho_m the homogeneous density."""
    We = max(G * G * d_h / (rho_m * sigma), 1e-12)
    Bd = (rho_l - rho_v) * 9.81 * d_h * d_h / sigma
    C = 2.125 * (beta_deg / 70.0) ** 9.993 + 0.955
    return C * 15.698 * We ** -0.475 * Bd ** 0.255 * (rho_l / rho_v) ** -0.571


def series_C(*Cs):
    """Coefficient of flow resistances in series (same fluid density):
    1 / C^2 = sum 1 / C_k^2.  A zero coefficient (closed valve) gives zero."""
    inv = sum(1.0 / np.maximum(np.asarray(C, float), 1e-30) ** 2 for C in Cs)
    return 1.0 / np.sqrt(inv)


@njit(cache=True, inline="always")
def series_C3(C1, C2, C3):
    """:func:`series_C` of three coefficients (kernel)."""
    return 1.0 / math.sqrt(1.0 / max(C1, 1e-30) ** 2 + 1.0 / max(C2, 1e-30) ** 2 + 1.0 / max(C3, 1e-30) ** 2)


@_vec
def hx_drop(mdot, C, rho):
    """Signed friction pressure drop m |m| / (C^2 rho) of a heat exchanger side [Pa]."""
    return mdot * abs(mdot) / (C * C * max(rho, 1e-9))


@_vec
def water_valve_flow(C, dP, rho):
    """Cooling water: incompressible, never flashing, driven by the fixed
    supply-to-return pressure difference ``dP`` (``C`` may be the series
    coefficient of the valve, the piping and the condenser, see :func:`series_C`)."""
    return C * math.sqrt(rho * max(dP, 0.0))


@_vec
def counterflow_effectiveness(NTU, Cr):
    """Effectiveness of a counterflow heat exchanger, Cr = C_min / C_max in 0..1."""
    Cr = min(max(Cr, 0.0), 1.0)
    d = 1.0 - Cr
    if d > 1e-6:
        ex = math.exp(-NTU * d)
        return (1.0 - ex) / max(1.0 - Cr * ex, 1e-12)
    return NTU / (1.0 + NTU)       # balanced limit Cr -> 1


@_vec
def churchill_f(Re, rel_rough):
    """Darcy friction factor of a round pipe, all regimes (Churchill 1977)."""
    Re = max(Re, 1.0)
    A = (-2.457 * math.log((7.0 / Re) ** 0.9 + 0.27 * rel_rough)) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + (A + B) ** -1.5) ** (1.0 / 12.0)


PIPE_ROUGHNESS = 1.5e-6            # drawn copper tube [m]


@_vec
def pipe_drop_s(mdot, d, L, K, rho, mu, rough):
    """:func:`pipe_drop` with the roughness given."""
    A = 0.25 * math.pi * d * d
    Re = abs(mdot) * d / (A * max(mu, 1e-7))
    f = churchill_f(Re, rough / max(d, 1e-4))
    return mdot * abs(mdot) / (2.0 * max(rho, 1e-6) * A * A) * (f * L / d + K)


def pipe_drop(mdot, d, L, K, rho, mu, rough=PIPE_ROUGHNESS):
    """Signed friction and fittings pressure drop of a round pipe of bore ``d``,
    length ``L`` and fittings loss coefficient ``K`` (velocity heads) [Pa].
    Two-phase flow is taken as homogeneous (``rho``, ``mu`` of the mixture)."""
    return pipe_drop_s(mdot, d, L, K, rho, mu, rough)


@_vec
def mixture_viscosity(x, mu_l, mu_v):
    """Homogeneous two-phase viscosity (McAdams); single phase outside 0 <= x <= 1."""
    x = min(max(x, 0.0), 1.0)
    return 1.0 / (x / mu_v + (1.0 - x) / mu_l)


# ---------------------------------------------------------------- actuator
@_vec
def actuator_rate(u_pos, u_cmd, tau, rate_max):
    """First-order lag with slew-rate limit: returns du/dt."""
    u_cmd = min(max(u_cmd, 0.0), 1.0)
    return min(max((u_cmd - u_pos) / tau, -rate_max), rate_max)


# -------------------------------------------------------------- compressor
@njit(cache=True)
def compressor_s(p, tab, P_s, h_s, s_s, rho_s, P_d, N, T_shell):
    """Quasi-steady compressor map with shell heat exchange (parameter record
    ``p``, property tables ``tab``).  Returns mass flow, discharge enthalpy,
    adiabatic discharge enthalpy and temperature, volumetric and isentropic
    efficiency, shaft and electrical power, motor loss, gas -> shell heat and
    pressure ratio (the order of ``COMPRESSOR_KEYS``)."""
    Nr = max(N, 0.0)
    Pr = max(P_d / P_s, 1.0)
    eta_v = min(max(p.eta_v0 - p.c_cl * (Pr ** (1.0 / p.kappa) - 1.0), 0.0), 1.0)
    mdot = eta_v * rho_s * p.V_disp * Nr / 60.0

    h2s = max(kp.h_Ps_vapor(tab, P_d, s_s), h_s)
    eta_s = p.eta_s0 * (1.0 - p.a_s * (Pr - p.Pr_opt) ** 2) * (1.0 - p.b_N * (Nr / p.N_nom - 1.0) ** 2)
    eta_s = min(max(eta_s, 0.25), 0.95)
    a = p.f_motor_gas * (1.0 / p.eta_motor - 1.0) / eta_s
    h_s1 = (h_s + a * h2s) / (1.0 + a)        # suction gas after motor heating
    dh_ad = max(h2s - h_s1, 0.0) / eta_s
    h2_ad = h_s1 + dh_ad
    W_shaft = mdot * dh_ad
    W_el = W_shaft / p.eta_motor
    Q_motor = W_el - W_shaft

    T2_ad = kp.T_vapor(tab, P_d, h2_ad)
    C_g = mdot * p.cp_gas
    eps = 1.0 - math.exp(-p.UA_gs / max(C_g, 1e-9))
    Q_gs = eps * C_g * (T2_ad - T_shell)                # gas -> shell [W]
    h2 = h2_ad - (Q_gs / max(mdot, 1e-9) if mdot > 1e-9 else 0.0)
    return mdot, h2, h2_ad, T2_ad, eta_v, eta_s, W_shaft, W_el, Q_motor, Q_gs, Pr


COMPRESSOR_KEYS = ("mdot", "h2", "h2_ad", "T2_ad", "eta_v", "eta_s", "W_shaft", "W_el", "Q_motor", "Q_gs", "Pr")


@njit(cache=True)
def _compressor_array(rec, tab, P_s, h_s, s_s, rho_s, P_d, N, T_shell, out):
    for k in range(P_s.shape[0]):
        r = compressor_s(rec[k], tab, P_s[k], h_s[k], s_s[k], rho_s[k], P_d[k], N[k], T_shell[k])
        for f in range(11):
            out[f, k] = r[f]


def compressor(p, props, P_s, h_s, s_s, rho_s, P_d, N, T_shell):
    """:func:`compressor_s` for a batch (parameter namespace ``p``, one
    environment per element).  Returns a dict of arrays."""
    rec = p.packed()
    a = [np.ascontiguousarray(np.broadcast_to(np.asarray(v, float), (len(rec),)))
         for v in (P_s, h_s, s_s, rho_s, P_d, N, T_shell)]
    out = np.empty((11, len(rec)))
    _compressor_array(rec, props.tab, *a, out)
    return {k: out[i] for i, k in enumerate(COMPRESSOR_KEYS)}


# ------------------------------------------------------------ control volume
@njit(cache=True, inline="always")
def cv_balance(V, rho, drho_dP, drho_dh, dm, E):
    """Pressure and enthalpy rates of a lumped control volume.

    Mass balance   : V (rho_P dP/dt + rho_h dh/dt) = dm
    Energy balance : M dh/dt = E + V dP/dt          (M = rho V)
    where dm is the net mass inflow and E = sum(m_in (h_in - h)) - sum(m_out (h_out - h)) + Q.
    """
    denom = V * (drho_dP + drho_dh / rho)
    dP = (dm - drho_dh * E / rho) / denom
    dh = (E + V * dP) / (rho * V)
    return dP, dh
