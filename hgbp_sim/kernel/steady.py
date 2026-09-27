"""Compiled (numba) steady-state solver, one environment at a time; the method
is described in :mod:`hgbp_sim.steady_state`, which wraps it."""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..components import (compressor_s, gas_valve_flow_s, kv_to_C, liquid_valve_flow_s, series_C3,
                          water_valve_flow)
from . import batch_variants
from . import props as kp
from .layout import (A_M_g, A_M_s, A_Q_sg, A_T_co, A_T_d, A_T_s, A_W_el, A_h_2f, A_h_3f, A_h_g, A_mdot_2,
                     A_mdot_3, A_mdot_c, A_x_l1, A_y_liq, MX, NX, X_H_D, X_H_I, X_HG, X_HQ, X_MM, X_N,
                     X_P_D, X_P_I, X_P_S, X_T_CW, X_T_DW, X_T_RW, X_T_SH, X_T_SW, X_TM_CO, X_TM_D, X_TM_S,
                     X_TMW, X_U1, X_WM)
from .model import NS, rhs

NZ = 10               # outer unknowns: u1..u4, h_d, T_sw, T_dw, T_cw, T_sh, T_rw
ZSCALE = np.array([1.0, 1.0, 1.0, 1.0, 2e4, 10.0, 10.0, 10.0, 10.0, 10.0])
# residuals: suction side mass [kg/s] and energy [W] balance, dT_sw, dP_d, dh_d, dT_dw,
# dP_i, dh_i, dT_cw, dT_rw, dT_sh
RSCALE = np.array([1e-3, 100.0, 0.1, 1e3, 1e2, 0.1, 1e3, 1e2, 0.1, 0.1, 0.1])
NR = 11
RES_IDX = (X_T_SW, X_P_D, X_H_D, X_T_DW, X_P_I, X_H_I, X_T_CW, X_T_RW, X_T_SH)
ZLO = np.array([1e-3] * 4 + [-np.inf] + [150.0] * 5)
ZHI = np.array([1.0] * 4 + [np.inf] + [600.0] * 5)
NP = 2 * MX           # exchanger profile: quench cell enthalpies, wall temperatures


@njit(cache=True)
def _solve_linear(A, b, out):
    """``out`` = A^-1 b by Gaussian elimination with partial pivoting (A, b are
    overwritten); False if A is singular."""
    n = b.shape[0]
    for c in range(n):
        piv = c
        for r in range(c + 1, n):
            if abs(A[r, c]) > abs(A[piv, c]):
                piv = r
        if A[piv, c] == 0.0 or not np.isfinite(A[piv, c]):
            return False
        if piv != c:
            for k in range(n):
                A[c, k], A[piv, k] = A[piv, k], A[c, k]
            b[c], b[piv] = b[piv], b[c]
        for r in range(c + 1, n):
            f = A[r, c] / A[c, c]
            if f != 0.0:
                for k in range(c, n):
                    A[r, k] -= f * A[c, k]
                b[r] -= f * b[c]
    for r in range(n - 1, -1, -1):
        s = b[r]
        for k in range(r + 1, n):
            s -= A[r, k] * out[k]
        out[r] = s / A[r, r]
    return True


@njit(cache=True)
def invert_characteristic(f, code, R):
    """Stem position for installed flow fraction ``f`` (inverse of components.valve_fraction)."""
    f = min(max(f, 1e-4), 1.0)
    if code == 0:
        return f
    if code == 1:
        return math.log1p(f * (R - 1.0)) / math.log(R)
    return f * f


@njit(cache=True)
def fill_enthalpy(tab, P, V, fill):
    """Mean enthalpy of a volume V at pressure P holding liquid fill fraction ``fill``."""
    s = kp.sat(tab, P)
    M_l = s.rho_l * V * fill
    M_v = s.rho_v * V * (1.0 - fill)
    xq = M_v / (M_l + M_v)
    return s.h_l + xq * (s.h_v - s.h_l)


@njit(cache=True)
def liquid_volume_for_level(p, level):
    """Liquid volume of the intermediate section with the receiver filled to ``level``
    (see params.liquid_volume_for_level)."""
    ramp = min(max((level - p.rec_dip) / 0.01, 0.0), 1.0)
    return level * p.rec_V + ramp * p.V_line_liq


@njit(cache=True)
def initial_guess(p, tab, P_s, h_s, P_d, P_i, N, T_amb, T_wi, z, prof):
    """Physically motivated starting point for the equilibrium iteration."""
    n = MX
    S = kp.state(tab, P_s, h_s)
    T_sh0 = T_amb + 30.0
    mdot_c, h2, _, T2_ad, _, _, _, _, Q_motor, _, _ = compressor_s(p, tab, P_s, h_s, S.s, S.rho, P_d, N, T_sh0)
    h_d = h2 - 300.0
    D = kp.state(tab, P_d, h_d)
    sati = kp.sat(tab, P_i)
    h_l = sati.h_l
    rho_g = kp.vapor_props(tab, P_i, h_d)[1]
    # valve 1 passes the full compressor flow
    g1 = gas_valve_flow_s(1.0, P_d, P_i, D.rho, rho_g, p.kappa, p.xT, p.eps_valve, 0.0, 1.0)
    z[0] = invert_characteristic(mdot_c / max(g1 * kv_to_C(p.Kv_dpv), 1e-12), p.dpv_code, p.dpv_R)
    # suction side energy balance -> split between bypass and quench
    frac_l = min(max((h_d - h_s) / max(h_d - h_l, 1.0), 0.02), 0.9)
    mdot_3 = frac_l * mdot_c
    mdot_2 = mdot_c - mdot_3
    g2 = gas_valve_flow_s(1.0, P_i, P_s, rho_g, S.rho, p.kappa, p.xT, p.eps_valve, 0.0, 1.0)
    z[1] = invert_characteristic(mdot_2 / max(g2 * kv_to_C(p.Kv_spv), 1e-12), p.spv_code, p.spv_R)
    l3 = liquid_valve_flow_s(1.0, P_i, P_s, sati.rho_l, S.rho, p.f_choke_liq, p.eps_valve, 0.0, 1.0)
    z[2] = invert_characteristic(mdot_3 / max(l3 * kv_to_C(p.Kv_stv), 1e-12), p.stv_code, p.stv_R)
    # condenser duty and water flow (bisection on the effectiveness relation)
    Q = mdot_3 * (h_d - h_l)
    T_cw = sati.T_l - Q / (0.6 * p.alpha_r_2ph * p.cond_A)
    lo, hi = 1e-4, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)                                 # installed flow fraction f(u4)
        mdot_w = water_valve_flow(series_C3(kv_to_C(p.Kv_w) * mid, kv_to_C(p.cond_Kv_w), kv_to_C(p.Kv_wpipe)),
                                  p.P_w_sup - p.P_w_ret, p.rho_w)
        Cw = mdot_w * p.cp_w
        UA_w = p.alpha_w0 * p.cond_A * (mdot_w / p.mdot_w_ref) ** 0.8
        Qw = (1.0 - math.exp(-UA_w / Cw)) * Cw * (T_cw - T_wi)
        if Qw > Q:
            hi = mid
        else:
            lo = mid
    z[3] = invert_characteristic(0.5 * (lo + hi), p.w_code, p.w_R)
    z[4] = h_d
    z[5] = (p.UA_sa * T_amb + p.UA_sg * S.T) / (p.UA_sa + p.UA_sg)
    z[6] = (p.UA_da * T_amb + p.UA_dg * D.T) / (p.UA_da + p.UA_dg)
    z[7] = T_cw
    C_g = mdot_c * p.cp_gas
    eps_g = 1.0 - math.exp(-p.UA_gs / max(C_g, 1e-9))
    z[8] = T_amb + (eps_g * C_g * (T2_ad - T_amb) + (1.0 - p.f_motor_gas) * Q_motor) / (p.UA_sha + eps_g * C_g)
    z[9] = (p.rec_UA_a * T_amb + p.rec_UA_r * sati.T_l) / (p.rec_UA_a + p.rec_UA_r)
    # mixing exchanger: quench side from the valve 3 outlet to the suction state, gas side
    # from the bypass inlet (bottom) to the suction state (top); walls in between
    T_2 = kp.T_vapor(tab, P_s, h_d)
    for j in range(n):
        prof[j] = h_l + (h_s - h_l) * min((j + 1.0) / n * 1.5, 1.0)
        T_q = kp.state(tab, P_s, prof[j]).T
        T_gp = S.T + (T_2 - S.T) * (j + 0.5) / n
        prof[n + j] = 0.5 * (T_q + T_gp)


@njit(cache=True)
def build_x(p, tab, P_s, h_s, P_d, P_i, N, z, prof, M_g, charge, fill, use_fill, x):
    """Full state from the outer unknowns ``z`` and the exchanger profile; the
    intermediate section holds what the charge leaves (or the receiver level ``fill``)."""
    n = MX
    x[:] = 0.0
    x[X_P_S] = P_s
    for j in range(n):
        x[X_HQ + j] = prof[j]
        x[X_TMW + j] = prof[n + j]
    x[X_HQ + n] = h_s
    x[X_HQ + n + 1] = h_s
    x[X_T_SW], x[X_T_DW], x[X_T_CW] = z[5], z[6], z[7]
    x[X_T_SH], x[X_T_RW] = z[8], z[9]
    x[X_P_D], x[X_H_D] = P_d, z[4]
    x[X_P_I] = P_i
    x[X_N] = N
    for k in range(4):
        x[X_U1 + k] = z[k]
    if use_fill:
        V_L = liquid_volume_for_level(p, fill)
        x[X_H_I] = fill_enthalpy(tab, P_i, p.V_i, V_L / p.V_i)
    else:
        rho_d = kp.state(tab, P_d, z[4]).rho
        M_c = 0.0
        for j in range(NS):
            M_c += kp.state(tab, P_s, x[X_HQ + j]).rho * p.V_sc[j]
        M_i = charge - M_c - M_g - rho_d * p.V_d
        x[X_H_I] = kp.h_from_P_rho(tab, P_i, max(M_i, 1e-6) / p.V_i)


@njit(cache=True)
def residual(p, tab, P_s, h_s, P_d, P_i, N, T_amb, T_wi, z, prof, M_g, charge, fill, use_fill, r, x, dx, a, hg):
    """Scaled steady-state residuals of the stand at ``z`` (exchanger profile fixed):
    the suction side's overall mass and energy balance and the rates of the other states."""
    n = MX
    build_x(p, tab, P_s, h_s, P_d, P_i, N, z, prof, M_g, charge, fill, use_fill, x)
    rhs(x, z[:4], N, T_amb, T_wi, p, tab, False, dx, True, a, hg)
    Q_amb = 0.0
    for j in range(n):
        Q_amb += (p.mx_UA_a / n) * (T_amb - x[X_TMW + j])
    r[0] = (a[A_mdot_2] + a[A_mdot_3] - a[A_mdot_c]) / RSCALE[0]
    r[1] = (a[A_mdot_2] * a[A_h_2f] + a[A_mdot_3] * a[A_h_3f] + a[A_Q_sg] + Q_amb - a[A_mdot_c] * h_s) / RSCALE[1]
    for k in range(9):
        r[2 + k] = dx[RES_IDX[k]] / RSCALE[2 + k]


@njit(cache=True)
def _norm(r):
    s = 0.0
    for v in r:
        s += v * v
    return math.sqrt(s)


@njit(cache=True)
def lm(p, tab, P_s, h_s, P_d, P_i, N, T_amb, T_wi, z, prof, M_g, charge, fill, use_fill, max_iter, tol,
       x, dx, a, hg):
    """Levenberg-Marquardt iteration on the outer unknowns (finite-difference
    Jacobian); returns the final residual norm and the iterations taken."""
    r = np.empty(NR)
    r_new = np.empty(NR)
    Rk = np.empty(NR)
    J = np.empty((NR, NZ))
    JtJ = np.empty((NZ, NZ))
    Jtr = np.empty(NZ)
    A = np.empty((NZ, NZ))
    b = np.empty(NZ)
    step = np.empty(NZ)
    zp = np.empty(NZ)
    z_new = np.empty(NZ)
    lam = 1e-2
    residual(p, tab, P_s, h_s, P_d, P_i, N, T_amb, T_wi, z, prof, M_g, charge, fill, use_fill, r, x, dx, a, hg)
    rn = _norm(r)
    h = 1e-4
    it = 0
    for it in range(max_iter):
        if not rn > tol:
            break
        for k in range(NZ):
            zp[:] = z
            zp[k] += h * ZSCALE[k]
            residual(p, tab, P_s, h_s, P_d, P_i, N, T_amb, T_wi, zp, prof, M_g, charge, fill, use_fill, Rk,
                     x, dx, a, hg)
            for i in range(NR):
                J[i, k] = (Rk[i] - r[i]) / h
        for i in range(NZ):
            s = 0.0
            for m in range(NR):
                s += J[m, i] * r[m]
            Jtr[i] = s
            for j in range(NZ):
                s = 0.0
                for m in range(NR):
                    s += J[m, i] * J[m, j]
                JtJ[i, j] = s
        for _ in range(8):
            for i in range(NZ):
                for j in range(NZ):
                    A[i, j] = JtJ[i, j]
                A[i, i] += lam * (JtJ[i, i] + 1e-9)
                b[i] = -Jtr[i]
            if not _solve_linear(A, b, step):
                step[:] = 0.0
            for k in range(NZ):
                z_new[k] = min(max(z[k] + step[k] * ZSCALE[k], ZLO[k]), ZHI[k])
            residual(p, tab, P_s, h_s, P_d, P_i, N, T_amb, T_wi, z_new, prof, M_g, charge, fill, use_fill, r_new,
                     x, dx, a, hg)
            rn_new = _norm(r_new)
            if rn_new < rn:
                z[:] = z_new
                r[:] = r_new
                rn = rn_new
                lam *= 0.3
                break
            lam *= 5.0
        lam = min(max(lam, 1e-6), 1e6)
    return rn, it + 1


@njit(cache=True)
def march_mixer(p, tab, x, u, N, T_amb, T_wi, plo, phi, max_steps, dx, a, hg, tol_h=1.0, tol_T=1e-3, cfl=0.5):
    """March the mixing exchanger's quench cells and walls (everything else
    frozen, suction pressure held) until they are at rest.  Only the steady
    state matters, so every cell and wall takes its own pseudo-time step,
    ``cfl`` times its own time constant (local time stepping).  Returns the
    number of steps taken."""
    n = MX
    k = 0
    for k in range(max_steps):
        rhs(x, u, N, T_amb, T_wi, p, tab, True, dx, True, a, hg)
        rest = True
        for j in range(n):
            if not (abs(dx[X_HQ + j]) < tol_h and abs(dx[X_TMW + j]) < tol_T):
                rest = False
        if rest:
            break
        m3 = abs(a[A_mdot_3])
        fq = max(m3 / p.mx_mdot_q_ref, 0.04)
        UA_q = p.mx_alpha_e * p.A_mx_cell * math.sqrt(fq)                  # upper bound of the cell's UA
        UA_g = p.mx_alpha_g0 * p.A_mx_cell * (
            max(max(a[A_mdot_2], 0.0), 1e-3 * p.mx_mdot_g_ref) / p.mx_mdot_g_ref) ** 0.8
        dt_T = cfl * p.C_mw_cell / (UA_g + UA_q)
        P_s = x[X_P_S]
        for j in range(n):
            M_q = kp.state(tab, P_s, x[X_HQ + j]).rho * p.V_sc[j]
            dt_h = cfl * M_q / (m3 + UA_q / 1000.0)
            x[X_HQ + j] = min(max(x[X_HQ + j] + dt_h * dx[X_HQ + j], plo[j]), phi[j])
            x[X_TMW + j] = min(max(x[X_TMW + j] + dt_T * dx[X_TMW + j], plo[n + j]), phi[n + j])
    return k


@njit(cache=True)
def _mixer_res(p, tab, x, u, N, T_amb, T_wi, out, dx, a, hg):
    n = MX
    rhs(x, u, N, T_amb, T_wi, p, tab, True, dx, False, a, hg)
    for j in range(n):
        out[j] = dx[X_HQ + j] / 1.0              # [J/kg/s]
        out[n + j] = dx[X_TMW + j] / 1e-3        # [K/s] in mK/s


@njit(cache=True)
def newton_mixer(p, tab, x, u, N, T_amb, T_wi, plo, phi, dx, a, hg, iters=6):
    """Damped Newton polish of the exchanger profile (quench cell enthalpies and
    wall temperatures) after the march brought it close to rest."""
    n = MX
    k = NP
    eps = 1e-3
    r0, rk = np.empty(k), np.empty(k)
    J = np.empty((k, k))
    b, step = np.empty(k), np.empty(k)
    xp = np.empty(NX)
    xn = np.empty(NX)
    for _ in range(iters):
        _mixer_res(p, tab, x, u, N, T_amb, T_wi, r0, dx, a, hg)
        small = True
        for i in range(k):
            if not abs(r0[i]) < 1.0:
                small = False
        if small:
            break
        for j in range(k):
            xp[:] = x
            ix = X_HQ + j if j < n else X_TMW + j - n
            xp[ix] += eps * (2e3 if j < n else 0.1)
            _mixer_res(p, tab, xp, u, N, T_amb, T_wi, rk, dx, a, hg)
            for i in range(k):
                J[i, j] = (rk[i] - r0[i]) / eps
        for i in range(k):
            J[i, i] += 1e-9
            b[i] = -r0[i]
        if not _solve_linear(J, b, step):
            break
        base = 0.0
        for i in range(k):
            base = max(base, abs(r0[i]))
        for al in (1.0, 0.5, 0.25, 0.1):
            xn[:] = x
            for j in range(k):
                ix = X_HQ + j if j < n else X_TMW + j - n
                xn[ix] = min(max(x[ix] + al * step[j] * (2e3 if j < n else 0.1), plo[j]), phi[j])
            _mixer_res(p, tab, xn, u, N, T_amb, T_wi, rk, dx, a, hg)
            worst = 0.0
            for i in range(k):
                worst = max(worst, abs(rk[i]))
            if worst < base:
                x[:] = xn
                break
            if al == 0.1:
                break


@njit(cache=True)
def solve_one(p, tab, P_s, P_d, SH, N, P_i, charge, fill, use_fill, T_amb, T_wi, max_iter, tol, rounds,
              x, a, info):
    """Equilibrium of one environment; fills the state ``x`` (sensor states at their
    steady values), the auxiliary outputs ``a`` and ``info`` = (converged, residual
    norm, iterations)."""
    n = MX
    h_s = kp.h_PT(tab, P_s, kp.sat(tab, P_s).T_l + SH)
    z = np.empty(NZ)
    prof = np.empty(NP)
    initial_guess(p, tab, P_s, h_s, P_d, P_i, N, T_amb, T_wi, z, prof)
    for k in range(NZ):
        z[k] = min(max(z[k], ZLO[k]), ZHI[k])
    # the exchanger holds no subcooled liquid and its walls stay between the suction
    # saturation temperature and the hot gas temperature
    s = kp.sat(tab, P_s)
    T_hot = kp.T_vapor(tab, P_s, z[4]) + 20.0
    h_hot = kp.h_PT(tab, P_s, T_hot)
    plo, phi = np.empty(NP), np.empty(NP)
    for j in range(n):
        plo[j], plo[n + j] = s.h_l, s.T_l - 0.5
        phi[j], phi[n + j] = h_hot, T_hot
    for j in range(NP):
        prof[j] = min(max(prof[j], plo[j]), phi[j])
    # bypass gas side inventory (quasi-steady): estimated, then taken from the solution
    V_g = 0.0
    for j in range(n):
        V_g += p.V_gc[j]
    M_g = V_g * kp.vapor_props(tab, P_s, 0.5 * (h_s + z[4]))[1]

    dx = np.empty(NX)
    hg = np.empty(MX)
    r = np.empty(NR)
    it_total = 0
    M_suc = np.inf
    rn = np.inf
    for rnd in range(rounds):
        rn, its = lm(p, tab, P_s, h_s, P_d, P_i, N, T_amb, T_wi, z, prof, M_g, charge, fill, use_fill,
                     max_iter, tol, x, dx, a, hg)
        it_total += its
        # exchanger profile for these flows, then its hold-up in the charge balance
        build_x(p, tab, P_s, h_s, P_d, P_i, N, z, prof, M_g, charge, fill, use_fill, x)
        march_mixer(p, tab, x, z[:4], N, T_amb, T_wi, plo, phi, 400 if rnd == 0 else 250, dx, a, hg)
        newton_mixer(p, tab, x, z[:4], N, T_amb, T_wi, plo, phi, dx, a, hg)
        for j in range(n):
            prof[j], prof[n + j] = x[X_HQ + j], x[X_TMW + j]
        build_x(p, tab, P_s, h_s, P_d, P_i, N, z, prof, M_g, charge, fill, use_fill, x)
        rhs(x, z[:4], N, T_amb, T_wi, p, tab, False, dx, True, a, hg)
        M_g = a[A_M_g]
        dM = abs(a[A_M_s] - M_suc)
        M_suc = a[A_M_s]
        residual(p, tab, P_s, h_s, P_d, P_i, N, T_amb, T_wi, z, prof, M_g, charge, fill, use_fill, r, x, dx, a, hg)
        rn = _norm(r)
        if rn <= tol and (use_fill or dM < 1e-4):
            break

    build_x(p, tab, P_s, h_s, P_d, P_i, N, z, prof, M_g, charge, fill, use_fill, x)
    rhs(x, z[:4], N, T_amb, T_wi, p, tab, True, dx, True, a, hg)
    for j in range(n):
        x[X_HG + j] = a[A_h_g + j]
    x[X_TM_S], x[X_TM_D], x[X_TM_CO] = a[A_T_s], a[A_T_d], a[A_T_co]
    x[X_MM], x[X_WM] = a[A_mdot_c], a[A_W_el]
    mix_ok = True
    for j in range(n):
        if not (abs(dx[X_HQ + j]) < 20.0 and abs(dx[X_TMW + j]) < 0.02):
            mix_ok = False
    # physical plausibility: superheated suction stream without liquid at the compressor,
    # valves inside their range
    ok = rn <= tol and mix_ok and a[A_x_l1] > 1.0 and a[A_y_liq] <= p.y_flood
    for k in range(4):
        ok = ok and 1.5e-3 < z[k] < 0.999
    info[0] = 1.0 if ok else 0.0
    info[1] = rn
    info[2] = it_total


def _solve_batch(P, tab, P_s, P_d, SH, N, P_i, charge, fill, use_fill, T_amb, T_wi, max_iter, tol, rounds,
                 X, AUX, INFO):
    for e in prange(X.shape[0]):
        solve_one(P[e], tab, P_s[e], P_d[e], SH[e], N[e], P_i[e], charge[e], fill[e], use_fill, T_amb[e], T_wi[e],
                  max_iter, tol, rounds, X[e], AUX[e], INFO[e])


solve_batch = batch_variants(_solve_batch)
