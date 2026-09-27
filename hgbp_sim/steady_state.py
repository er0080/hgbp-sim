"""Steady-state (equilibrium) solver for the HGBP plant.

Given target suction pressure, discharge pressure, suction superheat,
intermediate (condensing) pressure and compressor speed, find the four valve
positions and the remaining states for which all time derivatives vanish.
Used for warm-start initialization, for checking whether a test point is
reachable with the installed charge, and for steady-state performance data.

In steady state the liquid inventory of the intermediate section (receiver
level, condenser flooding) is fixed by the total refrigerant charge (mass
conservation), so the charge (or, alternatively, a receiver level) is an
input of the solve.

The solve is split in two, because the mixing exchanger's internal profile
barely changes its outlet state (a generously sized counterflow exchanger is
insensitive to where the quench dries out), which makes one coupled Newton
problem ill-conditioned:

1. the stand as a whole, with the suction side's overall steady mass and
   energy balance as its equations (a batched Levenberg-Marquardt iteration
   with finite-difference Jacobians over valve positions, discharge enthalpy
   and wall temperatures), holding the exchanger profile fixed;
2. the exchanger profile for the resulting flows, by marching its own
   (stable) quench-cell and plate-wall dynamics to rest.

The two are repeated until the liquid held up in the exchanger (which enters
the charge balance) stops changing.  All environments are solved at once.
"""
from __future__ import annotations

import numpy as np

from .components import compressor, gas_valve_flow, kv_to_C, liquid_valve_flow, series_C, water_valve_flow

from .params import liquid_volume_for_level

_N_BASE = 10          # u1..u4, h_d, T_sw, T_dw, T_cw, T_sh, T_rw
_ZSCALE = np.array([1.0, 1.0, 1.0, 1.0, 2e4, 10.0, 10.0, 10.0, 10.0, 10.0])
# residuals: suction side mass [kg/s] and energy [W] balance, dT_sw, dP_d, dh_d, dT_dw,
# dP_i, dh_i, dT_cw, dT_rw, dT_sh
_RSCALE = np.array([1e-3, 100.0, 0.1, 1e3, 1e2, 0.1, 1e3, 1e2, 0.1, 0.1, 0.1])


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
    n = plant.MX
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
    u1 = _invert_characteristic(mdot_c / np.maximum(g1 * kv_to_C(p.Kv_dpv), 1e-12), p.dpv_char, p.dpv_R)
    # suction side energy balance -> split between bypass and quench
    frac_l = np.clip((h_d - h_s) / np.maximum(h_d - h_l, 1.0), 0.02, 0.9)
    mdot_3 = frac_l * mdot_c
    mdot_2 = mdot_c - mdot_3
    g2 = gas_valve_flow(ones, P_i, P_s, rho_g, S.rho, p.kappa, p.xT, p.eps_valve)
    u2 = _invert_characteristic(mdot_2 / np.maximum(g2 * kv_to_C(p.Kv_spv), 1e-12), p.spv_char, p.spv_R)
    l3 = liquid_valve_flow(ones, P_i, P_s, sati["rho_l"], S.rho, p.f_choke_liq, p.eps_valve)
    u3 = _invert_characteristic(mdot_3 / np.maximum(l3 * kv_to_C(p.Kv_stv), 1e-12), p.stv_char, p.stv_R)
    # condenser duty and water flow (bisection on the effectiveness relation)
    Q = mdot_3 * (h_d - h_l)
    T_cw = sati["T_l"] - Q / (0.6 * p.alpha_r_2ph * p.cond_A)
    C_fix = [kv_to_C(p.cond_Kv_w), kv_to_C(p.Kv_wpipe)]
    lo, hi = np.full_like(P_s, 1e-4), np.ones_like(P_s)
    for _ in range(40):
        mid = 0.5 * (lo + hi)                                 # installed flow fraction f(u4)
        mdot_w = water_valve_flow(series_C(kv_to_C(p.Kv_w) * mid, *C_fix), p.P_w_sup - p.P_w_ret, p.rho_w)
        Cw = mdot_w * p.cp_w
        UA_w = p.alpha_w0 * p.cond_A * np.power(mdot_w / p.mdot_w_ref, 0.8)
        eps = 1.0 - np.exp(-UA_w / Cw)
        Qw = eps * Cw * (T_cw - T_wi)
        too_much = Qw > Q
        hi = np.where(too_much, mid, hi)
        lo = np.where(too_much, lo, mid)
    u4 = _invert_characteristic(0.5 * (lo + hi), p.w_char, p.w_R)
    T_sw = (p.UA_sa * T_amb + p.UA_sg * S.T) / (p.UA_sa + p.UA_sg)
    T_dw = (p.UA_da * T_amb + p.UA_dg * D.T) / (p.UA_da + p.UA_dg)
    T_rw = (p.rec_UA_a * T_amb + p.rec_UA_r * sati["T_l"]) / (p.rec_UA_a + p.rec_UA_r)
    C_g = mdot_c * p.cp_gas
    eps_g = 1.0 - np.exp(-p.UA_gs / np.maximum(C_g, 1e-9))
    T_sh = T_amb + (eps_g * C_g * (cp["T2_ad"] - T_amb) + (1 - p.f_motor_gas) * cp["Q_motor"]) \
        / (p.UA_sha + eps_g * C_g)
    # mixing exchanger: quench side from the valve 3 outlet to the suction state, gas side
    # from the bypass inlet (bottom) to the suction state (top); walls in between
    frac = (np.arange(n) + 1.0) / n                           # top -> bottom
    h_q = h_l[:, None] + (h_s - h_l)[:, None] * np.minimum(frac[None, :] * 1.5, 1.0)
    T_q = pr.state(np.repeat(P_s, n), h_q.reshape(-1)).T.reshape(-1, n)
    T_2 = pr.T_vapor(P_s, h_d)
    T_gp = S.T[:, None] + (T_2 - S.T)[:, None] * (np.arange(n)[None, :] + 0.5) / n
    T_mw = 0.5 * (T_q + T_gp)
    return np.concatenate([np.stack([u1, u2, u3, u4, h_d, T_sw, T_dw, T_cw, T_sh, T_rw], axis=1), h_q, T_mw],
                          axis=1)


def _repeat_params(p, K):
    """Parameter namespace with every environment repeated ``K`` times."""
    out = type(p)()
    for key, v in vars(p).items():
        if isinstance(v, np.ndarray):
            setattr(out, key, np.repeat(v, K, axis=0))
        elif isinstance(v, dict):
            setattr(out, key, {kk: np.repeat(vv, K, axis=0) for kk, vv in v.items()})
        else:
            setattr(out, key, v)
    return out


def _march_mixer(plant, x, u, N, T_amb, T_wi, p, zlo, zhi, max_steps=600, tol_h=1.0, tol_T=1e-3, cfl=0.5):
    """March the mixing exchanger's quench cells and walls (everything else
    frozen, suction pressure held) until they are at rest.  Only the steady
    state matters, so every cell and wall takes its own pseudo-time step,
    ``cfl`` times its own time constant (local time stepping).  Returns the
    number of steps taken."""
    n = plant.MX
    hs = slice(plant.HS.start, plant.HS.start + n)
    k = 0
    for k in range(max_steps):
        dx, a = plant.rhs(x, u, N, T_amb, T_wi, p=p, hold_P_s=True, want_aux=True)
        rh, rT = dx[:, hs], dx[:, plant.TMW]
        if (np.abs(rh) < tol_h).all() and (np.abs(rT) < tol_T).all():
            break
        C = plant._suction_cells(x[:, plant.P_S], x[:, plant.HS])
        M_q = C["rho"][:, :n] * p.V_sc[:, :n]
        fq = np.maximum(np.abs(a["mdot_3"]) / p.mx_mdot_q_ref, 0.04)
        UA_q = p.mx_alpha_e * p.A_mx_cell * np.sqrt(fq)                  # upper bound of the cell's UA
        UA_g = p.mx_alpha_g0 * p.A_mx_cell * np.power(
            np.maximum(np.maximum(a["mdot_2"], 0.0), 1e-3 * p.mx_mdot_g_ref) / p.mx_mdot_g_ref, 0.8)
        dt_h = cfl * M_q / (np.abs(a["mdot_3"])[:, None] + UA_q[:, None] / 1000.0)
        dt_T = cfl * p.C_mw_cell / (UA_g + UA_q)
        x[:, hs] = np.clip(x[:, hs] + dt_h * rh, zlo[:, :n], zhi[:, :n])
        x[:, plant.TMW] = np.clip(x[:, plant.TMW] + dt_T[:, None] * rT, zlo[:, n:], zhi[:, n:])
    return k


def _newton_mixer(plant, x, u, N, T_amb, T_wi, p, zlo, zhi, iters=6):
    """Damped Newton polish of the exchanger profile (quench cell enthalpies and
    wall temperatures) after the march brought it close to rest."""
    n = plant.MX
    mix = np.r_[plant.HS.start:plant.HS.start + n, plant.TMW.start:plant.TMW.stop]
    sc = np.array([2e3] * n + [0.1] * n)            # unknown scales [J/kg], [K]
    rsc = np.array([1.0] * n + [1e-3] * n)          # residual scales [J/kg/s], [K/s]
    m, k = x.shape[0], len(mix)
    eps = 1e-3

    p_rep = _repeat_params(p, k + 1)

    def res(xx, pp):
        K = len(xx) // m
        return plant.rhs(xx, np.repeat(u, K, 0), np.repeat(N, K), np.repeat(T_amb, K), np.repeat(T_wi, K),
                         p=pp, hold_P_s=True)[:, mix] / rsc

    for _ in range(iters):
        r0 = res(x, p)
        if (np.abs(r0) < 1.0).all():
            break
        X = np.repeat(x[:, None, :], k + 1, axis=1)
        for j in range(k):
            X[:, j + 1, mix[j]] += eps * sc[j]
        R = res(X.reshape(m * (k + 1), -1), p_rep).reshape(m, k + 1, k)
        J = np.transpose((R[:, 1:, :] - R[:, :1, :]) / eps, (0, 2, 1))       # (m, k, k) in scaled units
        try:
            step = -np.linalg.solve(J + 1e-9 * np.eye(k)[None], r0[:, :, None])[:, :, 0] * sc
        except np.linalg.LinAlgError:
            break
        base = np.abs(r0).max(1)
        for a in (1.0, 0.5, 0.25, 0.1):
            xn = x.copy()
            xn[:, mix] = np.clip(x[:, mix] + a * step, zlo, zhi)
            better = np.abs(res(xn, p)).max(1) < base
            if better.all() or a == 0.1:
                x[:, mix] = np.where(better[:, None], xn[:, mix], x[:, mix])
                break
    return x


def solve_steady_state(plant, P_s, P_d, SH, N, P_i=None, charge=None, fill=None,
                       T_amb=None, T_wi=None, idx=None, max_iter=40, tol=2e-3,
                       verbose=False, rounds=10):
    """Solve for the equilibrium of environments ``idx`` at the given targets.

    Parameters
    ----------
    plant : HGBPPlant
    P_s, P_d : target suction / discharge pressure [Pa]
    SH : suction superheat at the probe [K]      N : compressor speed [rpm]
    P_i : intermediate (condensing) pressure [Pa]; default P_d - dP_i_margin
    charge : total refrigerant mass [kg] (default: the plant's ``p.charge``);
        ignored if ``fill`` is given
    fill : receiver level (share of its volume, condenser drained) instead of
        a charge (the resulting total mass is returned as ``charge``)
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
    n = plant.MX
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

    nz, nres = _N_BASE, len(_RSCALE)
    z0 = initial_guess(plant, p, P_s, h_s, P_d, P_i, N, T_amb, T_wi)
    z, prof = z0[:, :nz].copy(), z0[:, nz:].copy()        # outer unknowns, exchanger profile
    zlo = np.array([1e-3] * 4 + [-np.inf] + [150.0] * 5)
    zhi = np.array([1.0] * 4 + [np.inf] + [600.0] * 5)
    z = np.clip(z, zlo, zhi)
    # the exchanger holds no subcooled liquid and its walls stay between the suction
    # saturation temperature and the hot gas temperature
    sat_s = pr.sat(P_s)
    T_hot = pr.T_vapor(P_s, z[:, 4]) + 20.0
    h_hot = pr.h_PT(P_s, T_hot)
    plo = np.concatenate([sat_s["h_l"][:, None] * np.ones((1, n)), (sat_s["T_l"] - 0.5)[:, None] * np.ones((1, n))], 1)
    phi = np.concatenate([h_hot[:, None] * np.ones((1, n)), T_hot[:, None] * np.ones((1, n))], 1)
    prof = np.clip(prof, plo, phi)
    p_rep = _repeat_params(p, nz + 1)
    res_idx = [plant.T_SW, plant.P_D, plant.H_D, plant.T_DW, plant.P_I, plant.H_I, plant.T_CW, plant.T_RW,
               plant.T_SH]
    # bypass gas side inventory (quasi-steady): estimated, then taken from the solution
    M_g = p.V_gc.sum(1) * pr.vapor_props(P_s, 0.5 * (h_s + z[:, 4]))[1]

    def build_x(Zf, Pf, K, pp, M_gK):
        rep = lambda a: np.repeat(a, K)
        x = np.zeros((m * K, plant.NX))
        x[:, plant.P_S] = rep(P_s)
        x[:, plant.HS] = np.concatenate([Pf[:, :n], rep(h_s)[:, None], rep(h_s)[:, None]], 1)
        x[:, plant.TMW] = Pf[:, n:]
        x[:, plant.T_SW], x[:, plant.T_DW], x[:, plant.T_CW] = Zf[:, 5], Zf[:, 6], Zf[:, 7]
        x[:, plant.T_SH], x[:, plant.T_RW] = Zf[:, 8], Zf[:, 9]
        x[:, plant.P_D], x[:, plant.H_D] = rep(P_d), Zf[:, 4]
        x[:, plant.P_I] = rep(P_i)
        x[:, plant.N_] = rep(N)
        x[:, plant.UV] = Zf[:, 0:4]
        Pi = rep(P_i)
        if fill is not None:
            V_L = liquid_volume_for_level(pp, rep(fill))
            x[:, plant.H_I] = fill_enthalpy(pr, Pi, pp.V_i, V_L / pp.V_i)
        else:
            rho_d = pr.state(rep(P_d), Zf[:, 4]).rho
            C = plant._suction_cells(rep(P_s), x[:, plant.HS])
            M_i = rep(charge) - (C["rho"] * pp.V_sc).sum(1) - M_gK - rho_d * pp.V_d
            x[:, plant.H_I] = pr.h_from_P_rho(Pi, np.maximum(M_i, 1e-6) / pp.V_i)
        return x

    def residual_batch(Z, prof, M_gm):
        K = Z.shape[1]
        Zf = Z.reshape(m * K, nz)
        Pf = np.repeat(prof, K, axis=0)
        pp = p_rep if K == nz + 1 else p
        x = build_x(Zf, Pf, K, pp, np.repeat(M_gm, K))
        dx, a = plant.rhs(x, Zf[:, 0:4], np.repeat(N, K), np.repeat(T_amb, K), np.repeat(T_wi, K), p=pp,
                          want_aux=True)
        Tw = x[:, plant.TMW]
        Q_amb = ((pp.mx_UA_a / n)[:, None] * (np.repeat(T_amb, K)[:, None] - Tw)).sum(1)
        mass = a["mdot_2"] + a["mdot_3"] - a["mdot_c"]
        energy = a["mdot_2"] * a["h_2f"] + a["mdot_3"] * a["h_3f"] + a["Q_sg"] + Q_amb - a["mdot_c"] * np.repeat(h_s, K)
        r = np.concatenate([np.stack([mass, energy], 1), dx[:, res_idx]], 1) / _RSCALE
        return r.reshape(m, K, nres)

    def lm(z, prof, M_g):
        lam = np.full(m, 1e-2)
        r = residual_batch(z[:, None, :], prof, M_g)[:, 0, :]
        rn = np.linalg.norm(r, axis=1)
        h = 1e-4
        it = 0
        for it in range(max_iter):
            active = rn > tol
            if not active.any():
                break
            Zp = np.repeat(z[:, None, :], nz + 1, axis=1)
            for k in range(nz):
                Zp[:, k + 1, k] += h * _ZSCALE[k]
            R = residual_batch(Zp, prof, M_g)
            r0 = R[:, 0, :]
            J = np.transpose((R[:, 1:, :] - r0[:, None, :]) / h, (0, 2, 1))   # (m, nres, nz)
            JtJ = J.transpose(0, 2, 1) @ J
            Jtr = np.einsum("mij,mi->mj", J, r0)
            diag = np.diagonal(JtJ, axis1=1, axis2=2)[:, :, None] + 1e-9
            for _ in range(8):
                A = JtJ + lam[:, None, None] * (np.eye(nz)[None] * diag)
                try:
                    step = -np.linalg.solve(A, Jtr[:, :, None])[:, :, 0]
                except np.linalg.LinAlgError:
                    step = np.zeros_like(z)
                z_new = np.clip(z + step * _ZSCALE, zlo, zhi)
                r_new = residual_batch(z_new[:, None, :], prof, M_g)[:, 0, :]
                rn_new = np.linalg.norm(r_new, axis=1)
                better = (rn_new < rn) & active
                z = np.where(better[:, None], z_new, z)
                r = np.where(better[:, None], r_new, r)
                rn = np.where(better, rn_new, rn)
                lam = np.where(better, lam * 0.3, np.where(active, lam * 5.0, lam))
                if better.all() or not active.any():
                    break
            lam = np.clip(lam, 1e-6, 1e6)
        return z, rn, it + 1

    it_total = 0
    M_suc = np.full(m, np.inf)
    rn = np.full(m, np.inf)
    for rnd in range(rounds):
        z, rn, its = lm(z, prof, M_g)
        it_total += its
        # exchanger profile for these flows, then its hold-up in the charge balance
        x = build_x(z, prof, 1, p, M_g)
        steps = _march_mixer(plant, x, z[:, 0:4], N, T_amb, T_wi, p, plo, phi,
                             max_steps=400 if rnd == 0 else 250)
        x = _newton_mixer(plant, x, z[:, 0:4], N, T_amb, T_wi, p, plo, phi)
        prof = np.concatenate([x[:, plant.HS][:, :n], x[:, plant.TMW]], 1)
        _, aux = plant.rhs(build_x(z, prof, 1, p, M_g), z[:, 0:4], N, T_amb, T_wi, want_aux=True, p=p)
        M_g = aux["M_g"]
        dM = np.abs(aux["M_s"] - M_suc)
        M_suc = aux["M_s"]
        rn = np.linalg.norm(residual_batch(z[:, None, :], prof, M_g)[:, 0, :], axis=1)
        if verbose:
            print(f"round {rnd}: outer resid max {rn.max():.2e} ({its} it), mixer march {steps} steps, "
                  f"suction inventory change {dM.max():.1e} kg")
        if (rn <= tol).all() and (fill is not None or (dM < 1e-4).all()):
            break

    x = build_x(z, prof, 1, p, M_g)
    u = z[:, 0:4].copy()
    dx, aux = plant.rhs(x, u, N, T_amb, T_wi, want_aux=True, p=p, hold_P_s=True)
    x[:, plant.HG] = aux["h_g"]
    x[:, plant.TM_S], x[:, plant.TM_D], x[:, plant.TM_CO] = aux["T_s"], aux["T_d"], aux["T_co"]
    x[:, plant.MM], x[:, plant.WM] = aux["mdot_c"], aux["W_el"]
    mix_ok = (np.abs(dx[:, plant.HS][:, :n]) < 20.0).all(1) & (np.abs(dx[:, plant.TMW]) < 0.02).all(1)
    # physical plausibility: superheated suction stream without liquid at the compressor,
    # valves inside their range
    converged = (rn <= tol) & mix_ok & (aux["x_l1"] > 1.0) & (aux["y_liq"] <= p.y_flood) \
        & (u < 0.999).all(axis=1) & (u > 1.5e-3).all(axis=1)
    return dict(x=x, u=u, converged=converged, resid=rn, aux=aux, charge=aux["M_tot"],
                iterations=it_total)
