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
def gas_valve_flow(C, P1, P2, rho1, rho2, kappa, xT, eps):
    """Compressible flow through a valve/orifice (ISA-style with choking).

    Positive flow is from side 1 to side 2.  ``rho1``/``rho2`` are the
    densities on each side; the upstream one is used.
    """
    dP = P1 - P2
    fwd = dP >= 0.0
    P_up = np.where(fwd, P1, P2)
    rho_up = np.where(fwd, rho1, rho2)
    Fk = kappa / 1.4
    x = np.abs(dP) / P_up
    x_ch = Fk * xT
    x_eff = np.minimum(x, x_ch)
    Y = 1.0 - x_eff / (3.0 * x_ch)
    dPe = np.sign(dP) * x_eff * P_up
    return C * Y * np.sqrt(rho_up) * regroot(dPe, eps)


def liquid_valve_flow(C, P1, P2, rho1, rho2, f_choke, eps):
    """Incompressible/flashing flow through an expansion valve, signed."""
    dP = P1 - P2
    fwd = dP >= 0.0
    P_up = np.where(fwd, P1, P2)
    rho_up = np.where(fwd, rho1, rho2)
    dPe = np.sign(dP) * np.minimum(np.abs(dP), f_choke * P_up)
    return C * np.sqrt(rho_up) * regroot(dPe, eps)


def line_flow(K, P1, P2, rho1, rho2, eps):
    """Pipe/fitting resistance: m = K sqrt(rho dP), signed and regularized."""
    dP = P1 - P2
    rho_up = np.where(dP >= 0.0, rho1, rho2)
    return K * np.sqrt(rho_up) * regroot(dP, eps)


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
