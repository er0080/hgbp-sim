"""Vectorized component sub-models (valves, actuators, compressor, helpers).

All functions accept and return numpy arrays (batch dimension first) and are
free of Python-level branching so they work on a batch of environments.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ helpers
def regroot(z, eps):
    """Regularized signed square root: ~ sign(z)*sqrt(|z|) for |z| >> eps and
    ~ z/sqrt(eps) for |z| << eps.  Smooth at zero (avoids infinite gain of the
    orifice equation when the pressure difference vanishes)."""
    return z / np.power(z * z + eps * eps, 0.25)


def clip(x, lo, hi):
    """Faster equivalent of np.clip for arrays with scalar/array bounds."""
    return np.minimum(np.maximum(x, lo), hi)


def smoothstep(t):
    t = clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def valve_characteristic(u, kind: str, R):
    """Installed flow fraction f(u) in [0, 1] for stem position u in [0, 1]."""
    u = clip(u, 0.0, 1.0)
    if kind == "linear":
        return u
    if kind == "eqpct":
        # modified equal-percentage: f(0) = 0, f(1) = 1
        return (np.power(R, u) - 1.0) / (R - 1.0)
    if kind == "quick":
        return np.sqrt(u)
    raise ValueError(f"unknown valve characteristic {kind!r}")


# ------------------------------------------------------------------- valves
# Valves are sized by their Kv: the water flow [m^3/h] the fully open valve
# passes at a 1 bar drop.  With Q = Kv sqrt(dP[bar] / SG) that is
#     mdot [kg/s] = (Kv / 36000) sqrt(rho [kg/m^3] dP [Pa])
# so the flow functions below take C = Kv / 36000 in kg/s per sqrt(Pa kg/m^3).
KV_TO_C = 1.0 / 36000.0


def kv_to_C(Kv):
    """Flow coefficient C [kg/s per sqrt(Pa kg/m^3)] of a valve rated ``Kv``
    [m^3/h].  Scale by the installed flow fraction f(u) before use."""
    return Kv * KV_TO_C


def _series_share(C, rho_up, C_hx, rho_hx):
    """Share of a series pair's pressure drop taken by the valve (coefficient
    ``C`` at upstream density ``rho_up``), the rest by a resistance ``C_hx``
    at density ``rho_hx``; both obey m = C sqrt(rho dP)."""
    if C_hx is None:
        return 1.0
    r = C / np.maximum(C_hx, 1e-12)
    return 1.0 / (1.0 + r * r * rho_up / np.maximum(rho_hx, 1e-9))


def gas_valve_flow(C, P1, P2, rho1, rho2, kappa, xT, eps, C_hx=None, rho_hx=None):
    """Compressible flow through a valve/orifice (ISA-style with choking).

    Positive flow is from side 1 to side 2.  ``rho1``/``rho2`` are the
    densities on each side; the upstream one is used.  ``C_hx`` optionally
    adds a resistance in series downstream (a heat exchanger side, density
    ``rho_hx``): the pair shares ``P1 - P2``, and the valve chokes on its own
    share of it.
    """
    dP = P1 - P2
    fwd = dP >= 0.0
    P_up = np.where(fwd, P1, P2)
    rho_up = np.where(fwd, rho1, rho2)
    Fk = kappa / 1.4
    x = np.abs(dP) / P_up
    x_ch = Fk * xT
    Y = 1.0 - np.minimum(x, x_ch) / (3.0 * x_ch)
    share = 1.0
    for _ in range(2 if C_hx is not None else 0):   # the valve's share and its expansion factor
        share = _series_share(C * Y, rho_up, C_hx, rho_hx)
        Y = 1.0 - np.minimum(x * share, x_ch) / (3.0 * x_ch)
    share = _series_share(C * Y, rho_up, C_hx, rho_hx)
    x_eff = np.minimum(x, x_ch / share)
    dPe = np.sign(dP) * x_eff * P_up
    return C * Y * np.sqrt(share * rho_up) * regroot(dPe, eps)


def liquid_valve_flow(C, P1, P2, rho1, rho2, f_choke, eps, C_hx=None, rho_hx=None):
    """Incompressible/flashing flow through an expansion valve, signed;
    optionally with a series resistance ``C_hx`` as in :func:`gas_valve_flow`."""
    dP = P1 - P2
    fwd = dP >= 0.0
    P_up = np.where(fwd, P1, P2)
    rho_up = np.where(fwd, rho1, rho2)
    share = _series_share(C, rho_up, C_hx, rho_hx)
    dPe = np.sign(dP) * np.minimum(np.abs(dP), f_choke * P_up / share)
    return C * np.sqrt(share * rho_up) * regroot(dPe, eps)


def martin_xi(Re, beta_deg):
    """Single-phase Darcy friction factor of a chevron plate channel (Martin 1996,
    VDI Heat Atlas): dP = xi L / d_h rho u^2 / 2; ``beta_deg`` is the corrugation
    angle to the flow, d_h = 2 b / phi."""
    Re = np.maximum(Re, 1e-6)
    lam = Re < 2000.0
    xi0 = np.where(lam, 64.0 / Re, (1.8 * np.log10(np.maximum(Re, 2000.0)) - 1.5) ** -2)
    xi1 = np.where(lam, 597.0 / Re + 3.85, 39.0 / np.maximum(Re, 2000.0) ** 0.289)
    be = np.radians(beta_deg)
    c = np.cos(be)
    inv = c / np.sqrt(0.18 * np.tan(be) + 0.36 * np.sin(be) + xi0 / c) + (1.0 - c) / np.sqrt(3.8 * xi1)
    return inv ** -2


def amalfi_ftp(G, d_h, rho_m, rho_l, rho_v, sigma, beta_deg):
    """Two-phase Fanning friction factor of flow boiling in plate heat exchangers
    (Amalfi, Vakili-Farahani & Thome 2016, 1513 points, 13 studies):
    dP = 2 f L G^2 / (d_h rho_m), rho_m the homogeneous density."""
    We = np.maximum(G * G * d_h / (rho_m * sigma), 1e-12)
    Bd = (rho_l - rho_v) * 9.81 * d_h * d_h / sigma
    C = 2.125 * (beta_deg / 70.0) ** 9.993 + 0.955
    return C * 15.698 * We ** -0.475 * Bd ** 0.255 * (rho_l / rho_v) ** -0.571


def series_C(*Cs):
    """Coefficient of flow resistances in series (same fluid density):
    1 / C^2 = sum 1 / C_k^2.  A zero coefficient (closed valve) gives zero."""
    inv = sum(1.0 / np.maximum(np.asarray(C, float), 1e-30) ** 2 for C in Cs)
    return 1.0 / np.sqrt(inv)


def hx_drop(mdot, C, rho):
    """Signed friction pressure drop m |m| / (C^2 rho) of a heat exchanger side [Pa]."""
    return mdot * np.abs(mdot) / (C * C * np.maximum(rho, 1e-9))


def water_valve_flow(C, dP, rho):
    """Cooling water: incompressible, never flashing, driven by the fixed
    supply-to-return pressure difference ``dP`` (``C`` may be the series
    coefficient of the valve, the piping and the condenser, see :func:`series_C`)."""
    return C * np.sqrt(rho * np.maximum(dP, 0.0))


def counterflow_effectiveness(NTU, Cr):
    """Effectiveness of a counterflow heat exchanger, Cr = C_min / C_max in 0..1."""
    Cr = clip(Cr, 0.0, 1.0)
    d = 1.0 - Cr
    ex = np.exp(-NTU * d)
    eps = (1.0 - ex) / np.maximum(1.0 - Cr * ex, 1e-12)
    return np.where(d > 1e-6, eps, NTU / (1.0 + NTU))       # balanced limit Cr -> 1


def churchill_f(Re, rel_rough):
    """Darcy friction factor of a round pipe, all regimes (Churchill 1977)."""
    Re = np.maximum(Re, 1.0)
    A = (-2.457 * np.log((7.0 / Re) ** 0.9 + 0.27 * rel_rough)) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + (A + B) ** -1.5) ** (1.0 / 12.0)


def pipe_drop(mdot, d, L, K, rho, mu, rough=1.5e-6):
    """Signed friction and fittings pressure drop of a round pipe of bore ``d``,
    length ``L`` and fittings loss coefficient ``K`` (velocity heads) [Pa].
    Two-phase flow is taken as homogeneous (``rho``, ``mu`` of the mixture)."""
    A = 0.25 * np.pi * d * d
    Re = np.abs(mdot) * d / (A * np.maximum(mu, 1e-7))
    f = churchill_f(Re, rough / np.maximum(d, 1e-4))
    return mdot * np.abs(mdot) / (2.0 * np.maximum(rho, 1e-6) * A * A) * (f * L / d + K)


def mixture_viscosity(x, mu_l, mu_v):
    """Homogeneous two-phase viscosity (McAdams); single phase outside 0 <= x <= 1."""
    x = clip(x, 0.0, 1.0)
    return 1.0 / (x / mu_v + (1.0 - x) / mu_l)


# ---------------------------------------------------------------- actuator
def actuator_rate(u_pos, u_cmd, tau, rate_max):
    """First-order lag with slew-rate limit: returns du/dt."""
    u_cmd = clip(u_cmd, 0.0, 1.0)
    return clip((u_cmd - u_pos) / tau, -rate_max, rate_max)


# -------------------------------------------------------------- compressor
def compressor(p, props, P_s, h_s, s_s, rho_s, P_d, N, T_shell):
    """Quasi-steady compressor map with shell heat exchange.

    Returns dict with mass flow, discharge enthalpy, adiabatic discharge
    temperature, shaft & electrical power, motor loss and gas->shell heat.
    """
    Nr = np.maximum(N, 0.0)
    Pr = np.maximum(P_d / P_s, 1.0)
    eta_v = clip(p.eta_v0 - p.c_cl * (np.power(Pr, 1.0 / p.kappa) - 1.0), 0.0, 1.0)
    mdot = eta_v * rho_s * p.V_disp * Nr / 60.0

    h2s = np.maximum(props.h_Ps_vapor(P_d, s_s), h_s)
    eta_s = p.eta_s0 * (1.0 - p.a_s * (Pr - p.Pr_opt) ** 2) \
        * (1.0 - p.b_N * (Nr / p.N_nom - 1.0) ** 2)
    eta_s = clip(eta_s, 0.25, 0.95)
    a = p.f_motor_gas * (1.0 / p.eta_motor - 1.0) / eta_s
    h_s1 = (h_s + a * h2s) / (1.0 + a)        # suction gas after motor heating
    dh_ad = np.maximum(h2s - h_s1, 0.0) / eta_s
    h2_ad = h_s1 + dh_ad
    W_shaft = mdot * dh_ad
    W_el = W_shaft / p.eta_motor
    Q_motor = W_el - W_shaft

    T2_ad = props.T_vapor(P_d, h2_ad)
    C_g = mdot * p.cp_gas
    eps = 1.0 - np.exp(-p.UA_gs / np.maximum(C_g, 1e-9))
    Q_gs = eps * C_g * (T2_ad - T_shell)                # gas -> shell [W]
    h2 = h2_ad - np.where(mdot > 1e-9, Q_gs / np.maximum(mdot, 1e-9), 0.0)
    return dict(mdot=mdot, h2=h2, h2_ad=h2_ad, T2_ad=T2_ad, eta_v=eta_v,
                eta_s=eta_s, W_shaft=W_shaft, W_el=W_el, Q_motor=Q_motor,
                Q_gs=Q_gs, Pr=Pr)


# ------------------------------------------------------------ control volume
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
