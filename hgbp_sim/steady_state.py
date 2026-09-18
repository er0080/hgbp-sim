"""Steady-state (equilibrium) solver for the HGBP plant.

Given target suction pressure, discharge pressure, suction superheat,
intermediate (condensing) pressure and compressor speed, find the four valve
positions and the remaining states for which all time derivatives vanish.
Used for warm-start initialization, for checking whether a test point is
reachable with the installed charge, and for steady-state performance data.

In steady state the liquid inventory of the condenser is fixed by the total
refrigerant charge (mass conservation), so the charge (or, alternatively, a
condenser liquid fill fraction) is an input of the solve.

The solver is a batched Levenberg-Marquardt iteration with finite-difference
Jacobians; all environments in the batch are solved simultaneously.
"""
from __future__ import annotations

import numpy as np

from .components import compressor, gas_valve_flow, liquid_valve_flow

# unknowns: u1, u2, u3, u4, h_d, T_sw, T_dw, T_cw, T_sh
_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 2e4, 10.0, 10.0, 10.0, 10.0])
# residuals: dP_s, dh_s, dT_sw, dP_d, dh_d, dT_dw, dP_i, dh_i, dT_cw, dT_sh
_RSCALE = np.array([1e3, 1e2, 0.1, 1e3, 1e2, 0.1, 1e3, 1e2, 0.1, 0.1])


def _invert_characteristic(f, kind, R):
    f = np.clip(f, 1e-4, 1.0)
    if kind == "linear":
        return f
    if kind == "eqpct":
        return np.log1p(f * (R - 1.0)) / np.log(R)
    if kind == "quick":
        return f * f
    raise ValueError(kind)


def fill_enthalpy(props, P, V, fill):
    """Mean enthalpy of a volume V at pressure P holding liquid fill fraction ``fill``."""
    sat = props.sat(P)
    M_l = sat["rho_l"] * V * fill
    M_v = sat["rho_v"] * V * (1.0 - fill)
    xq = M_v / (M_l + M_v)
    return sat["h_l"] + xq * (sat["h_v"] - sat["h_l"])


def initial_guess(plant, p, P_s, h_s, P_d, P_i, N, T_amb, T_wi):
    """Physically motivated starting point for the equilibrium iteration."""
    pr = plant.props
    S = pr.state(P_s, h_s, need_s=True)
    T_sh0 = T_amb + 30.0
    cp = compressor(p, pr, P_s, h_s, S.s, S.rho, P_d, N, T_sh0)
    mdot_c, h2 = cp["mdot"], cp["h2"]
    h_d = h2 - 300.0
    D = pr.state(P_d, h_d)
    sati = pr.sat(P_i)
    h_l = sati["h_l"]
    ones = np.ones_like(P_s)
    T_g, rho_g = pr.vapor_props(P_i, h_d)
    # valve 1 passes the full compressor flow
    g1 = gas_valve_flow(ones, P_d, P_i, D.rho, rho_g, p.kappa, p.xT, p.eps_valve)
    u1 = _invert_characteristic(mdot_c / np.maximum(g1 * p.C_dpv, 1e-12), p.dpv_char, p.dpv_R)
    # suction mixer energy balance -> split between bypass and liquid
    frac_l = np.clip((h_d - h_s) / np.maximum(h_d - h_l, 1.0), 0.02, 0.9)
    mdot_3 = frac_l * mdot_c
    mdot_2 = mdot_c - mdot_3
    g2 = gas_valve_flow(ones, P_i, P_s, rho_g, S.rho, p.kappa, p.xT, p.eps_valve)
    u2 = _invert_characteristic(mdot_2 / np.maximum(g2 * p.C_spv, 1e-12), p.spv_char, p.spv_R)
    l3 = liquid_valve_flow(ones, P_i, P_s, sati["rho_l"], S.rho, p.f_choke_liq, p.eps_valve)
    u3 = _invert_characteristic(mdot_3 / np.maximum(l3 * p.C_stv, 1e-12), p.stv_char, p.stv_R)
    # condenser duty and water flow (bisection on the effectiveness relation)
    Q = mdot_3 * (h_d - h_l)
    T_cw = sati["T_l"] - Q / (0.6 * p.UA_r_2ph)
    lo, hi = np.full_like(P_s, 1e-4), np.ones_like(P_s)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        Cw = p.mdot_w_max * mid * p.cp_w
        UA_w = p.UA_w0 * np.power(mid, 0.8)
        eps = 1.0 - np.exp(-UA_w / Cw)
        Qw = eps * Cw * (T_cw - T_wi)
        too_much = Qw > Q
        hi = np.where(too_much, mid, hi)
        lo = np.where(too_much, lo, mid)
    u4 = _invert_characteristic(0.5 * (lo + hi), p.w_char, p.w_R)
    T_sw = (p.UA_sa * T_amb + p.UA_sg * S.T) / (p.UA_sa + p.UA_sg)
    T_dw = (p.UA_da * T_amb + p.UA_dg * D.T) / (p.UA_da + p.UA_dg)
    C_g = mdot_c * p.cp_gas
    eps_g = 1.0 - np.exp(-p.UA_gs / np.maximum(C_g, 1e-9))
    T_sh = T_amb + (eps_g * C_g * (cp["T2_ad"] - T_amb) + (1 - p.f_motor_gas) * cp["Q_motor"]) \
        / (p.UA_sha + eps_g * C_g)
    return np.stack([u1, u2, u3, u4, h_d, T_sw, T_dw, T_cw, T_sh], axis=1)


def solve_steady_state(plant, P_s, P_d, SH, N, P_i=None, charge=None, fill=None,
                       T_amb=None, T_wi=None, idx=None, max_iter=40, tol=2e-3,
                       verbose=False):
    """Solve for the equilibrium of environments ``idx`` at the given targets.

    Parameters
    ----------
    plant : HGBPPlant
    P_s, P_d : target suction / discharge pressure [Pa]
    SH : suction superheat [K]          N : compressor speed [rpm]
    P_i : intermediate (condensing) pressure [Pa]; default P_d - dP_i_margin
    charge : total refrigerant mass [kg] (default: the plant's ``p.charge``);
        ignored if ``fill`` is given
    fill : condenser liquid fill fraction instead of a charge (the resulting
        total mass is returned as ``charge``)
    T_amb, T_wi : ambient / cooling-water inlet temperature [K] (default: the
        plant's current inputs)

    Returns
    -------
    dict with ``x`` (m, NX) full state matrix (sensor states set to their
    steady values), ``u`` (m, 4) valve positions, ``converged`` (m,) bool,
    ``resid`` (m,) final scaled residual norm, ``aux`` outputs at the
    solution, ``charge`` (m,) total mass at the solution.
    """
    pr = plant.props
    idx = np.arange(plant.n) if idx is None else np.atleast_1d(np.asarray(idx))
    m = len(idx)
    bc = lambda v, default: np.broadcast_to(np.asarray(default if v is None else v, float), (m,)).copy()
    P_s, P_d, SH, N = bc(P_s, 0), bc(P_d, 0), bc(SH, 0), bc(N, 0)
    T_amb = bc(T_amb, plant.T_amb[idx])
    T_wi = bc(T_wi, plant.T_wi[idx])
    p = plant.params_subset(idx)
    P_i = bc(P_i, P_d - p.dP_i_margin)
    h_s = pr.h_PT(P_s, pr.T_sat(P_s) + SH)
    if fill is not None:
        fill = bc(fill, 0.4)
        charge = None
    else:
        charge = bc(charge, p.charge)
    rho_s = pr.state(P_s, h_s).rho

    z = initial_guess(plant, p, P_s, h_s, P_d, P_i, N, T_amb, T_wi)
    zlo = np.array([1e-3, 1e-3, 1e-3, 1e-3, -np.inf, 150.0, 150.0, 150.0, 150.0])
    zhi = np.array([1.0, 1.0, 1.0, 1.0, np.inf, 600.0, 600.0, 600.0, 600.0])
    z = np.clip(z, zlo, zhi)

    nres, nz = 10, 9
    p_rep = plant.params_subset(idx, repeat=nz + 1)

    def h_i_of(Zf, pp, K):
        Pi = np.repeat(P_i, K)
        if fill is not None:
            return fill_enthalpy(pr, Pi, pp.V_i, np.repeat(fill, K))
        rho_d = pr.state(np.repeat(P_d, K), Zf[:, 4]).rho
        M_i = np.repeat(charge, K) - np.repeat(rho_s, K) * pp.V_s - rho_d * pp.V_d
        rho_i = np.maximum(M_i, 1e-6) / pp.V_i
        return pr.h_from_P_rho(Pi, rho_i)

    def residual_batch(Z):  # Z: (m, K, 9) -> (m, K, 10)
        K = Z.shape[1]
        Zf = Z.reshape(m * K, nz)
        rep = lambda a: np.repeat(a, K)
        pp = p_rep if K == nz + 1 else p
        x = np.zeros((m * K, plant.NX))
        x[:, 0], x[:, 1], x[:, 2] = rep(P_s), rep(h_s), Zf[:, 5]
        x[:, 3], x[:, 4], x[:, 5] = rep(P_d), Zf[:, 4], Zf[:, 6]
        x[:, 6], x[:, 7], x[:, 8] = rep(P_i), h_i_of(Zf, pp, K), Zf[:, 7]
        x[:, 9], x[:, 10] = Zf[:, 8], rep(N)
        x[:, 11:15] = Zf[:, 0:4]
        dx = plant.rhs(x, Zf[:, 0:4], rep(N), rep(T_amb), rep(T_wi), p=pp)
        r = dx[:, :10] / _RSCALE
        return r.reshape(m, K, nres)

    lam = np.full(m, 1e-2)
    r = residual_batch(z[:, None, :])[:, 0, :]
    rn = np.linalg.norm(r, axis=1)
    h = 1e-4
    it = 0
    for it in range(max_iter):
        active = rn > tol
        if not active.any():
            break
        Zp = np.repeat(z[:, None, :], nz + 1, axis=1)
        for k in range(nz):
            Zp[:, k + 1, k] += h * _SCALE[k]
        R = residual_batch(Zp)
        r0 = R[:, 0, :]
        J = np.transpose((R[:, 1:, :] - r0[:, None, :]) / h, (0, 2, 1))   # (m, 10, 9)
        JtJ = J.transpose(0, 2, 1) @ J
        Jtr = np.einsum("mij,mi->mj", J, r0)
        diag = np.diagonal(JtJ, axis1=1, axis2=2)[:, :, None] + 1e-9
        for _ in range(8):
            A = JtJ + lam[:, None, None] * (np.eye(nz)[None] * diag)
            try:
                step = -np.linalg.solve(A, Jtr[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                step = np.zeros_like(z)
            z_new = np.clip(z + step * _SCALE, zlo, zhi)
            r_new = residual_batch(z_new[:, None, :])[:, 0, :]
            rn_new = np.linalg.norm(r_new, axis=1)
            better = (rn_new < rn) & active
            z = np.where(better[:, None], z_new, z)
            r = np.where(better[:, None], r_new, r)
            rn = np.where(better, rn_new, rn)
            lam = np.where(better, lam * 0.3, np.where(active, lam * 5.0, lam))
            if better.all() or not active.any():
                break
        if verbose:
            print(f"iter {it}: max resid {rn.max():.3e}, mean {rn.mean():.3e}")
        lam = np.clip(lam, 1e-6, 1e6)

    converged = rn <= tol
    x = np.zeros((m, plant.NX))
    x[:, 0], x[:, 1], x[:, 2] = P_s, h_s, z[:, 5]
    x[:, 3], x[:, 4], x[:, 5] = P_d, z[:, 4], z[:, 6]
    x[:, 6], x[:, 7], x[:, 8] = P_i, h_i_of(z, p, 1), z[:, 7]
    x[:, 9], x[:, 10] = z[:, 8], N
    x[:, 11:15] = z[:, 0:4]
    u = z[:, 0:4].copy()
    _, aux = plant.rhs(x, u, N, T_amb, T_wi, want_aux=True, p=p)
    x[:, 15], x[:, 16], x[:, 19] = aux["T_s"], aux["T_d"], aux["T_co"]
    x[:, 17], x[:, 18] = aux["mdot_c"], aux["W_el"]
    # physical plausibility: superheated suction stream, valves inside their range
    converged = converged & (aux["x_out"] > 1.0) & (u < 0.999).all(axis=1) & (u > 1.5e-3).all(axis=1)
    return dict(x=x, u=u, converged=converged, resid=rn, aux=aux, charge=aux["M_tot"],
                iterations=it + 1)
