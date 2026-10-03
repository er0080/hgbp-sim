"""Compiled (numba) model kernel of the HGBP stand: right-hand side, the
projection onto the conserved states, the step size control and the
integrator.

Every function works on one environment (a state vector ``x`` of length
``NX``, a parameter record ``p`` of ``layout.PARAM_DTYPE``, the property tables
``tab``); the ``*_batch`` functions run a batch in parallel, one environment
per loop iteration.  The model itself is described in :mod:`hgbp_sim.plant`.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..components import (PIPE_ROUGHNESS, actuator_rate, amalfi_ftp, compressor_s, counterflow_effectiveness,
                          cv_balance, gas_valve_flow_s, hx_drop, kv_to_C, liquid_valve_flow_s, martin_xi,
                          mixture_viscosity, pipe_drop_s, series_C3, smoothstep, valve_fraction,
                          water_valve_flow)
from . import batch_variants
from . import props as kp
from .layout import (A_M_d, A_M_g, A_M_i, A_M_q_liq, A_M_s, A_M_tot, A_N, A_P_d, A_P_h, A_P_i, A_P_s, A_P_tee, A_Pr,
                     A_Q_dg, A_Q_mx, A_Q_q, A_Q_r, A_Q_rw, A_Q_sc, A_Q_sg, A_Q_w, A_SC, A_SH, A_T2_ad, A_T_c, A_T_co,
                     A_T_cw, A_T_cwc, A_T_d, A_T_dw, A_T_g, A_T_go, A_T_i, A_T_l1, A_T_mw, A_T_q, A_T_qo, A_T_rw, A_T_s,
                     A_T_sat_d, A_T_sat_i, A_T_sat_s, A_T_sh, A_T_sw, A_T_wc, A_T_wo, A_Tm_co, A_Tm_d, A_Tm_s, A_W_el,
                     A_W_shaft, A_Wm, A_cond_flood, A_dP_bp, A_dP_cr, A_dP_cw, A_dP_dis, A_dP_drn, A_dP_hdr, A_dP_liq,
                     A_dP_mg, A_dP_mog, A_dP_moq, A_dP_mq, A_dP_mq_ch, A_dP_q, A_dP_q_dist, A_dP_q_in, A_dP_q_out,
                     A_dP_qc, A_dP_suc, A_dx, A_eta_s, A_eta_v, A_fill_i, A_h2, A_h_2f, A_h_3f, A_h_c, A_h_cin, A_h_co,
                     A_h_d, A_h_g, A_h_go, A_h_i, A_h_l1, A_h_l2, A_h_q, A_ll_fill, A_mdot_1, A_mdot_2, A_mdot_3,
                     A_mdot_c, A_mdot_cr, A_mdot_w, A_mm, A_rec_level, A_rho_s, A_u1, A_x_c, A_x_i, A_x_l1, A_x_l2,
                     A_x_out, A_x_q, A_x_qo, A_y_liq, MX, NX, X_HG, X_HQ, X_H_D, X_H_I, X_MM, X_M_D, X_M_I, X_M_S, X_N,
                     X_P_D, X_P_I, X_P_S, X_TCW, X_TMW, X_TM_CO, X_TM_D, X_TM_S, X_T_DW, X_T_RW, X_T_SH, X_T_SW, X_U1,
                     X_U_D, X_U_I, X_U_S, X_WM)
# (layout's X_* state and A_* output indices are compile-time constants in the kernel)

G = 9.81                    # gravity [m/s^2]
DRYOUT_BAND = 15e3          # quench enthalpy band above the dew point over which a cell dries out [J/kg]
STIFF_FILL = 0.97           # liquid fill above which a volume counts as liquid-full (stiff)
STIFF_SUBDIV = 4            # sub-step refinement for stiff volumes
LAM_DT_MAX = 1.8            # largest (estimated fastest local rate x sub-step); RK4 is stable to 2.78,
                            # the margin covers the estimate running up to ~35 % low
MAX_SUBDIV = 32             # cap on the sub-step refinement
MAX_DH_STEP = 40e3          # suction cell enthalpy change per sub-step above which it is redone finer [J/kg]
MAX_DP_STEP = 0.2           # relative pressure change per sub-step above which it is redone finer
INTEGRATORS = {"rk4": 0, "heun": 1, "euler": 2}
NS = MX + 2                 # dynamic suction side cells: quench cells, L1 (tee + suction line), L2 (compressor)
CF_STAGNANT = 2e-3          # condenser flow below which its vapor counts as stagnant (the section's own) [kg/s]


# ------------------------------------------------------------------ helpers
@njit(cache=True, inline="always")
def _clip(v, lo, hi):
    return min(max(v, lo), hi)


@njit(cache=True, inline="always")
def _nanmax(a, b):
    """max that propagates NaN (like np.maximum)."""
    if b > a or b != b:
        return b
    return a


@njit(cache=True, inline="always")
def _pipe(D, t, L, K, mdot, rho, mu, share):
    """Friction and fittings drop of a refrigerant line (or the ``share`` of it)."""
    return pipe_drop_s(mdot, max(D - 2.0 * t, 1e-4), share * L, share * K, rho, mu, PIPE_ROUGHNESS)


@njit(cache=True, inline="always")
def _equiv_C(m0, dP0, rho_ref):
    """Equivalent series coefficient (m = C sqrt(rho_ref dP)) of flow-dependent
    drops ``dP0`` evaluated at the flow ``m0`` a valve passes on its own."""
    dP0 = abs(dP0)
    return abs(m0) / math.sqrt(rho_ref * max(dP0, 1.0)) if dP0 > 1.0 else 1e3


@njit(cache=True, inline="always")
def _valve_slope(Cf, rho_up, dP, eps):
    """|d mdot / d dP| of the regularized orifice equation (choking ignored: an
    upper bound used only to size the integration step)."""
    z2 = dP * dP
    e2 = eps * eps
    return Cf * math.sqrt(max(rho_up, 1e-9)) * (0.5 * z2 + e2) / (z2 + e2) ** 1.25


@njit(cache=True)
def intermediate_split(p, fill_i):
    """Where the liquid of the intermediate section sits, from its liquid
    volume share ``fill_i``.  Returns receiver level, liquid line fill
    (1 = liquid seal at valve 3) and flooded share of the condenser."""
    seg = p.seg_i
    V_L = fill_i * p.V_i
    part = np.empty(6)
    lower = np.empty(6)
    cum = 0.0
    for k in range(6):
        cum += seg[k]
        lower[k] = cum - seg[k]
        part[k] = _clip(V_L - lower[k], 0.0, seg[k])
    rec_level = (part[0] + part[2]) / p.rec_V
    w_ll = max(seg[1], 2e-4)                    # the seal forms over at least 0.2 L
    ll_fill = _clip((V_L - lower[1]) / w_ll, 0.0, 1.0)
    cond_flood = part[4] / max(seg[4], 1e-9)
    return rec_level, ll_fill, cond_flood


@njit(cache=True)
def _quench_drops(p, mdot, rho_in, rho_c, x_c, rho_l, rho_v, mu_l, mu_v, sigma, seg):
    """Flow-dependent pressure drops of the mixing exchanger's quench side (S3 -> S4)
    at quench flow ``mdot``, static heads excluded.  ``rho_in``: the valve 3 outlet
    mixture; ``rho_c``, ``x_c``: density and quality of the quench cells (top ->
    bottom).  Fills ``seg`` with each cell's friction plus acceleration and returns
    the drop ahead of the distributor, ahead of the channels (distributor, S3
    connection and port) and after them (S4 port and connection, into the outlet pipe)."""
    n = MX
    m = abs(mdot)
    sgn = 0.0 if mdot == 0.0 else math.copysign(1.0, mdot)
    dh = 2.0 * p.mx_b / p.mx_phi
    Gc = m / ((p.V_mx_q / p.mx_V_ch) * p.mx_b * p.mx_W)          # channel mass flux
    dz = p.mx_H / n
    beta = p.mx_beta
    for j in range(n):
        # friction: two-phase (Amalfi et al.) in the evaporating cells, Martin's single-phase
        # correlation for vapor (after dry-out) or liquid, blended across the dome
        f_tp = amalfi_ftp(Gc, dh, rho_c[j], rho_l, rho_v, sigma, beta)
        dp_tp = 2.0 * f_tp * Gc * Gc * dz / (dh * rho_c[j])
        mu = mu_v if x_c[j] >= 0.5 else mu_l
        dp_sp = martin_xi(Gc * dh / mu, beta) * dz / dh * Gc * Gc / (2.0 * rho_c[j])
        w_sp = smoothstep((x_c[j] - 0.975) / 0.05) + smoothstep((0.025 - x_c[j]) / 0.05)
        fric = sgn * ((1.0 - w_sp) * dp_tp + w_sp * dp_sp)
        # acceleration of the evaporating flow (homogeneous), forward flow only
        v_prev = 1.0 / rho_in if j == 0 else 1.0 / rho_c[j - 1]
        acc = Gc * Gc * (1.0 / rho_c[j] - v_prev) if mdot > 0.0 else 0.0
        seg[j] = fric + acc
    # ports (Shah & Focke: 0.75 velocity heads each end) and the 7/8 in connections:
    # expansion from the connection into the port at S3, contraction from the port into
    # the connection and expansion into the outlet pipe at S4
    A_p, A_c = 0.25 * np.pi * p.mx_d_port ** 2, 0.25 * np.pi * p.mx_d_S34 ** 2
    d_mo = max(p.D_mo - 2.0 * p.t_mo, p.mx_d_S34)
    a_cp = (p.mx_d_S34 / p.mx_d_port) ** 2
    K_in = (1.0 - a_cp) ** 2
    K_out = 0.5 * (1.0 - a_cp) + (1.0 - (p.mx_d_S34 / d_mo) ** 2) ** 2
    rho_out = rho_c[n - 1]
    dist = mdot * m / (kv_to_C(p.mx_Kv_dist) ** 2 * rho_in)
    inlet = dist + 0.75 * (mdot * m / (A_p * A_p * 2.0 * rho_in)) + K_in * (mdot * m / (A_c * A_c * 2.0 * rho_in))
    outlet = 0.75 * (mdot * m / (A_p * A_p * 2.0 * rho_out)) + K_out * (mdot * m / (A_c * A_c * 2.0 * rho_out))
    return dist, inlet, outlet


@njit(cache=True, inline="always")
def _quench_heat(Hq, Tq, T_mw, H_up, dTs, phi_f, Q_2ph, UA_e, UA_v, h_vs):
    """Heat from the plate wall into a quench cell.  A cell still holding liquid
    boils over its whole area; in the cell where the quench dries out the
    evaporating share of the area is what the incoming liquid needs, the rest
    heats vapor (blended over DRYOUT_BAND above the dew point)."""
    w_wet = 1.0 - smoothstep((Hq - h_vs) / DRYOUT_BAND)
    Tq = Tq + w_wet * dTs                        # liquid boils at the cell's own pressure
    # ... never more than the share of the cell's enthalpy rise below the dew point
    phi_h = _clip((h_vs - H_up) / max(Hq - H_up, 1.0), 0.0, 1.0)
    phi = min(phi_f, phi_h if Hq > H_up else 1.0)
    Q_dry = phi * Q_2ph + (1.0 - phi) * UA_v * (T_mw - Tq)
    return w_wet * UA_e * (T_mw - Tq) + (1.0 - w_wet) * Q_dry


@njit(cache=True)
def _cond_cell_heat(tab, P, s, h_up, Tc, Tw, m, A, a2, a1, w_flow, T_iv):
    """Heat from the refrigerant into the plate wall of one condenser cell (area
    ``A`` above the liquid) for flow ``m`` entering with enthalpy ``h_up``.  On a
    wall below the condensing temperature ``Tc`` vapor condenses (``a2``), and
    superheated vapor gives up its superheat to the condensate surface
    (``a1``, effectiveness over the cell: wet-wall desuperheating); a wall above
    it stays dry and only exchanges sensible heat with the vapor.  Without flow
    (``w_flow`` -> 0) a dry wall sees the section's own vapor at ``T_iv``."""
    if A <= 0.0:
        return 0.0
    wet = Tw < Tc
    Q = a2 * A * (Tc - Tw) if wet else 0.0
    T_t = s.T_v if wet else Tw                     # the vapor cools (or warms) toward this
    if h_up > s.h_v and m > 0.0:
        dT = kp.T_vapor(tab, P, h_up) - s.T_v
        cp = _clip((h_up - s.h_v) / dT if dT > 0.05 else s.cp_v, 300.0, 1e5)
        Qs = (1.0 - math.exp(-a1 * A / (m * cp))) * m * (h_up - s.h_v - cp * (T_t - s.T_v))
    else:
        Qs = 0.0 if wet else a1 * A * (Tc - Tw)    # saturated vapor against a dry wall
    if not wet:
        Qs = w_flow * Qs + (1.0 - w_flow) * a1 * A * (T_iv - Tw)
    return Q + Qs


@njit(cache=True)
def _condenser(p, tab, P, s, T_w, f_fl, h_hot, h_vap, T_iv, m_hot, Q, h_mid):
    """Refrigerant side of the condenser, quasi-steady, marched over its ``MX``
    cells from S3 (top) to S4 (bottom) against the plate walls ``T_w``; ``f_fl``
    is each cell's flooded share (handled with the subcooled zone).

    The vapor flowing in is what the plates condense, so the flow ``m`` is the
    one that leaves the last cell above the liquid at the bubble point:
    m (h_in - h_l) = sum Q.  It enters as hot gas from the header (``h_hot``, up
    to the header surplus ``m_hot``) and otherwise as the section's vapor
    (``h_vap``).  The flow, the condensing temperatures along the glide and the
    sensible heat of the superheated vapor depend on each other; three passes
    of a fixed-point iteration settle them (the sensible heat changes the flow
    by at most the superheat's share of the enthalpy drop).

    Fills ``Q`` (refrigerant -> wall per cell) and ``h_mid`` (the cells' mean
    enthalpy above the liquid); returns the flow."""
    n = MX
    A = p.cond_A / n
    a2, a1 = p.alpha_r_2ph, p.alpha_r_1ph
    dh_lv = max(s.h_v - s.h_l, 1e3)
    Tc = np.empty(n)
    m = 0.0
    for j in range(n):
        Tc[j] = 0.5 * (s.T_l + s.T_v)
        if T_w[j] < Tc[j]:
            m += a2 * A * (1.0 - f_fl[j]) * (Tc[j] - T_w[j])
    m /= dh_lv                                      # the condensing flow, if the vapor desuperheats fully
    w = 0.0
    for _ in range(3):
        s_hot = min(m_hot / m, 1.0) if m > 1e-12 else (1.0 if m_hot > 0.0 else 0.0)
        h_in = s_hot * h_hot + (1.0 - s_hot) * h_vap
        w = smoothstep(m / CF_STAGNANT)
        h = h_in
        Q_sum = 0.0
        for j in range(n):
            Q[j] = _cond_cell_heat(tab, P, s, h, Tc[j], T_w[j], m, A * (1.0 - f_fl[j]), a2, a1, w, T_iv)
            Q_sum += Q[j]
            h = max(h - Q[j] / max(m, 1e-9), s.h_l)
        m = max(Q_sum, 0.0) / max(h_in - s.h_l, 1e4)
        # profile at this flow -> condensing temperature of every cell (linear along the glide)
        h = h_in
        for j in range(n):
            h_o = h - Q[j] / max(m, 1e-9)
            h_mid[j] = _clip(0.5 * (h + h_o), s.h_l, max(h_in, h_vap))
            Tc[j] = s.T_l + _clip((h_mid[j] - s.h_l) / dh_lv, 0.0, 1.0) * (s.T_v - s.T_l)
            h = h_o
    for j in range(n):                              # without flow the cells hold the section's vapor
        h_mid[j] = w * h_mid[j] + (1.0 - w) * h_vap
    return m


# ----------------------------------------------------------------- RHS
@njit(cache=True)
def rhs(x, u_cmd, N_cmd, T_amb, T_wi, p, tab, hold_P_s, dx, want_aux, a, hg_out):
    """Time derivative ``dx`` of state ``x``; returns the fastest local rate [1/s]
    (for the step size control).  Writes the bypass gas side enthalpies of this
    evaluation to ``hg_out`` and, with ``want_aux``, the auxiliary outputs to
    ``a`` (layout.AUX_FIELDS).  ``hold_P_s``: the suction cell enthalpy rates at
    constant suction pressure (each cell's own energy balance; the steady-state
    solver uses these as residuals next to dP_s/dt)."""
    n = MX
    P_s = x[X_P_S]
    H = x[X_HQ:X_HQ + NS]                           # quench cells, L1, L2
    T_mw = x[X_TMW:X_TMW + n]
    T_sw = x[X_T_SW]
    P_d, h_d, T_dw = x[X_P_D], x[X_H_D], x[X_T_DW]
    P_i, h_i, T_rw = x[X_P_I], x[X_H_I], x[X_T_RW]
    T_cw = x[X_TCW:X_TCW + n]                       # condenser plate walls (top -> bottom)
    T_sh, N = x[X_T_SH], x[X_N]
    u1, u2, u3, u4 = x[X_U1], x[X_U1 + 1], x[X_U1 + 2], x[X_U1 + 3]
    Tm_s, Tm_d, mm, Wm, Tm_co = x[X_TM_S], x[X_TM_D], x[X_MM], x[X_WM], x[X_TM_CO]
    h_l1, h_l2 = H[n], H[n + 1]
    Vc = p.V_sc

    # ---- suction side cells (common pressure)
    c_rho, c_T, c_x = np.empty(NS), np.empty(NS), np.empty(NS)
    rho_P, rho_h, M_c = np.empty(NS), np.empty(NS), np.empty(NS)
    for j in range(NS):
        S = kp.state(tab, P_s, H[j])
        c_rho[j], c_T[j], c_x[j], rho_P[j], rho_h[j] = S.rho, S.T, S.x, S.drho_dP, S.drho_dh
        M_c[j] = S.rho * Vc[j]
    T_sat_s, h_ls, h_vs = S.T_sat, S.h_l, S.h_v
    sat_s = kp.sat(tab, P_s)
    D = kp.state(tab, P_d, h_d)                     # discharge volume
    I = kp.state(tab, P_i, h_i)                     # intermediate section

    # ---- compressor, drawing from its internal suction volume (L2); it takes in at most
    # 1 - comp_x_min liquid by mass, more liquid collects in the shell
    h_cin = max(h_l2, h_ls + p.comp_x_min * (h_vs - h_ls))
    Cin = kp.state(tab, P_s, h_cin)
    mdot_c, h2, h2_ad, T2_ad, eta_v, eta_s, W_shaft, W_el, Q_motor, Q_gs, Pr_c = compressor_s(
        p, tab, P_s, h_cin, Cin.s, Cin.rho, P_d, N, T_sh)

    # ---- intermediate section inventory: receiver, liquid seal, condenser flooding
    M_i = I.rho * p.V_i
    fill_i = min((1.0 - _clip(I.x, 0.0, 1.0)) * M_i / (I.rho_l * p.V_i), 1.0)
    rec_level, ll_fill, cond_flood = intermediate_split(p, fill_i)
    dry = 1.0 - smoothstep(ll_fill)                            # 1: no liquid seal at valve 3
    h_cv_out = (h_i if I.x < 0.0 else I.h_l) * (1.0 - dry) + h_i * dry
    rho_co = I.rho_l * (1.0 - dry) + I.rho * dry
    h_iv = I.h_v * (1.0 - dry) + h_i * dry                     # vapor phase of the section

    # ---- condenser cells (top -> bottom): liquid backing up from a full receiver floods
    # them from the bottom; the condensing flow (condensing duty over the latent heat,
    # exact when the vapor desuperheats completely) sets the refrigerant side's drop
    sat_i = kp.sat(tab, P_i)
    f_fl = np.empty(n)
    m_cond = 0.0
    T_c2 = 0.5 * (sat_i.T_l + sat_i.T_v)
    for j in range(n):
        f_fl[j] = _clip(cond_flood * n - (n - 1 - j), 0.0, 1.0)
        m_cond += p.alpha_r_2ph * (p.cond_A / n) * (1.0 - f_fl[j]) * max(T_c2 - T_cw[j], 0.0)
    w2 = (1.0 - smoothstep((I.x - 0.85) / 0.15)) * (1.0 - smoothstep(-I.x / 0.05))
    a_c = w2 * (1.0 - cond_flood)                   # condensing (two-phase) share of the column

    # ---- condenser pressure drop.  P_i is measured at the outlet (after the receiver,
    # ahead of valve 3); the header at the inlet sits above it by the friction drop of
    # the condensing flow and below it by the static head of the refrigerant column
    # (homogeneous two-phase over the condensing area, liquid where flooded).
    rho_v = I.rho if I.x >= 1.0 else sat_i.rho_v
    rho_l = sat_i.rho_l
    mdot_cr = m_cond / max(sat_i.h_v - sat_i.h_l, 1e4)
    rho_fr = w2 * 2.0 / (1.0 / rho_v + 1.0 / rho_l) + (1.0 - w2) * rho_v
    r_lv = max(rho_l / rho_v, 1.001)
    rho_hm = rho_v * math.log(r_lv) / (1.0 - 1.0 / r_lv)          # column mean, quality 1 -> 0
    rho_col = cond_flood * rho_l + a_c * rho_hm + (1.0 - cond_flood - a_c) * rho_v
    dP_cr = (1.0 - cond_flood) * hx_drop(mdot_cr, kv_to_C(p.cond_Kv_r), rho_fr) - G * p.cond_H * rho_col

    # ---- pressure chain of the intermediate side, from the P_i sensor (receiver outlet)
    # upstream: condensate drain, condenser, the header from the condenser back to the
    # valve 2 branch (taken halfway along the header), which is the header pressure P_h
    mu_l_i, mu_v_i, _ = kp.transport(tab, P_i)
    mu_v_d = kp.transport(tab, P_d)[1]
    mu_l_s, mu_v_s, sigma_s = kp.transport(tab, P_s)
    mu_co = mu_l_i * (1.0 - dry) + mu_v_i * dry
    dP_drn = _pipe(p.D_drn, p.t_drn, p.L_drn, p.K_drn, mdot_cr, rho_co, mu_co, 1.0)
    rho_hg = kp.vapor_props(tab, P_i + dP_drn + dP_cr, h_d)[1]      # header gas
    dP_hdr2 = _pipe(p.D_hdr, p.t_hdr, p.L_hdr, p.K_hdr, mdot_cr, rho_hg, mu_v_i, 0.5)
    P_h = max(P_i + dP_drn + dP_cr + dP_hdr2, tab.p_min * 1.001)   # hot gas header at the branch

    # ---- suction side: P_s is the compressor suction port; the tee sits above it by the
    # suction line's drop (compressor flow, the line's own mixture)
    mu_l1 = mixture_viscosity(c_x[n], mu_l_s, mu_v_s)
    dP_suc = _pipe(p.D_suc, p.t_suc, p.L_suc, p.K_suc, mdot_c, c_rho[n], mu_l1, 1.0)
    P_tee = P_s + dP_suc

    # ---- valve 1: discharge -> header branch, in series with the discharge line and the
    # first half of the header (equivalent resistance at the valve-alone flow)
    T_g, rho_g = kp.vapor_props(tab, P_h, h_d)                    # header gas after throttling
    f1 = valve_fraction(u1, p.dpv_code, p.dpv_R)
    C1 = kv_to_C(p.Kv_dpv) * f1
    m0 = gas_valve_flow_s(C1, P_d, P_h, D.rho, rho_g, p.kappa, p.xT, p.eps_valve, 0.0, 1.0)
    dP0 = (_pipe(p.D_dis, p.t_dis, p.L_dis, p.K_dis, m0, D.rho, mu_v_d, 1.0)
           + _pipe(p.D_hdr, p.t_hdr, p.L_hdr, p.K_hdr, m0, rho_g, mu_v_i, 0.5))
    mdot_1 = gas_valve_flow_s(C1, P_d, P_h, D.rho, rho_g, p.kappa, p.xT, p.eps_valve,
                              _equiv_C(m0, dP0, rho_g), rho_g)
    dP_dis = _pipe(p.D_dis, p.t_dis, p.L_dis, p.K_dis, mdot_1, D.rho, mu_v_d, 1.0)
    dP_hdr1 = _pipe(p.D_hdr, p.t_hdr, p.L_hdr, p.K_hdr, mdot_1, rho_g, mu_v_i, 0.5)
    h_1f = h_d if mdot_1 >= 0.0 else h_iv

    # ---- valve 2: header branch -> bypass line -> mixing exchanger gas side (S1 at the
    # bottom, rising to S2) -> outlet leg -> tee, all in series
    f2 = valve_fraction(u2, p.spv_code, p.spv_R)
    C2 = kv_to_C(p.Kv_spv) * f2
    rho_gin = kp.vapor_props(tab, P_s, h_d)[1]                    # bypass gas after throttling
    rho_gm = 2.0 / (1.0 / rho_gin + 1.0 / sat_s.rho_v)
    head_g = G * p.mx_H * rho_gm
    C_mg = kv_to_C(p.mx_Kv_g)
    m0 = gas_valve_flow_s(C2, P_h, P_tee + head_g, rho_g, c_rho[n], p.kappa, p.xT, p.eps_valve, 0.0, 1.0)
    dP0 = (_pipe(p.D_bp, p.t_bp, p.L_bp, p.K_bp, m0, rho_gin, mu_v_s, 1.0) + hx_drop(m0, C_mg, rho_gm)
           + _pipe(p.D_mo, p.t_mo, p.L_mo, p.K_mo, m0, sat_s.rho_v, mu_v_s, 0.5))
    mdot_2 = gas_valve_flow_s(C2, P_h, P_tee + head_g, rho_g, c_rho[n], p.kappa, p.xT, p.eps_valve,
                              _equiv_C(m0, dP0, rho_gm), rho_gm)
    dP_bp = _pipe(p.D_bp, p.t_bp, p.L_bp, p.K_bp, mdot_2, rho_gin, mu_v_s, 1.0)
    dP_mog = _pipe(p.D_mo, p.t_mo, p.L_mo, p.K_mo, mdot_2, sat_s.rho_v, mu_v_s, 0.5)
    dP_mg = hx_drop(mdot_2, C_mg, rho_gm) + head_g              # S1 above S2
    m_from_inlet = max(min(mdot_2, max(mdot_1, 0.0)), 0.0)
    fwd2 = mdot_2 > 0.0
    if fwd2:
        h_2f = (m_from_inlet * h_d + (max(mdot_2, 0.0) - m_from_inlet) * h_iv) / max(mdot_2, 1e-12)
    else:
        h_2f = h_l1

    # ---- valve 3: receiver / liquid line -> mixing exchanger quench side (S3 on top), in
    # series with it (distributor, ports, channels; see _quench_drops).  The drop is not
    # quadratic in the flow, so the series pair is solved with the side's equivalent
    # resistance at the valve-alone flow (the valve takes almost all of the difference).
    # Each cell adds its static head and boils at its own pressure; the distributor and
    # the S3 port are ahead of the channels and do not raise the boiling pressure.
    f3 = valve_fraction(u3, p.stv_code, p.stv_R)
    rho_c, x_c = c_rho[:n], c_x[:n]
    head_q = np.empty(n)
    head_sum = 0.0
    for j in range(n):
        head_q[j] = G * (p.mx_H / n) * rho_c[j]
        head_sum += head_q[j]
    S3 = kp.state(tab, P_s, h_cv_out)                             # valve 3 outlet, flashed
    rho_3 = S3.rho
    mu_3 = mixture_viscosity(S3.x, mu_l_s, mu_v_s)
    mu_qo = mixture_viscosity(x_c[n - 1], mu_l_s, mu_v_s)
    C3 = kv_to_C(p.Kv_stv) * f3
    P_3dn = P_tee - head_sum
    seg = np.empty(n)
    # liquid line, quench line, quench side, outlet leg at the valve-alone flow
    m0 = liquid_valve_flow_s(C3, P_i, P_3dn, rho_co, rho_c[0], p.f_choke_liq, p.eps_valve, 0.0, 1.0)
    _, q_in, q_out = _quench_drops(p, m0, rho_3, rho_c, x_c, sat_s.rho_l, sat_s.rho_v, mu_l_s, mu_v_s,
                                   sigma_s, seg)
    seg_sum = 0.0
    for j in range(n):
        seg_sum += seg[j]
    dP0 = (_pipe(p.D_liq, p.t_liq, p.L_liq, p.K_liq, m0, rho_co, mu_co, 1.0)
           + _pipe(p.D_q, p.t_q, p.L_q, p.K_q, m0, rho_3, mu_3, 1.0) + q_in + seg_sum + q_out
           + _pipe(p.D_mo, p.t_mo, p.L_mo, p.K_mo, m0, rho_c[n - 1], mu_qo, 0.5))
    mdot_3 = liquid_valve_flow_s(C3, P_i, P_3dn, rho_co, rho_c[0], p.f_choke_liq, p.eps_valve,
                                 _equiv_C(m0, dP0, rho_3), rho_3)
    fwd3 = mdot_3 >= 0.0
    q_dist, q_in, q_out = _quench_drops(p, mdot_3, rho_3, rho_c, x_c, sat_s.rho_l, sat_s.rho_v, mu_l_s,
                                        mu_v_s, sigma_s, seg)
    dP_liq = _pipe(p.D_liq, p.t_liq, p.L_liq, p.K_liq, mdot_3, rho_co, mu_co, 1.0)
    dP_q = _pipe(p.D_q, p.t_q, p.L_q, p.K_q, mdot_3, rho_3, mu_3, 1.0)
    dP_moq = _pipe(p.D_mo, p.t_mo, p.L_mo, p.K_mo, mdot_3, rho_c[n - 1], mu_qo, 0.5)
    # cell centres above the compressor port: suction line, outlet leg, S4 port, cells below;
    # boiling temperature shift of each cell (Clausius-Clapeyron at P_s)
    seg_q = np.empty(n)
    seg_q_sum = 0.0
    for j in range(n):
        seg_q[j] = seg[j] - head_q[j]                          # P(top of cell) - P(bottom of cell)
        seg_q_sum += seg_q[j]
    dTs_dP = sat_s.T_v * (1.0 / sat_s.rho_v - 1.0 / sat_s.rho_l) / max(sat_s.h_v - sat_s.h_l, 1e3)
    dP_qc = np.empty(n)
    dTs = np.empty(n)
    below = 0.0
    base = dP_suc + dP_moq + q_out
    for j in range(n - 1, -1, -1):
        below += seg_q[j]
        dP_qc[j] = base + below - 0.5 * seg_q[j]
        dTs[j] = dTs_dP * dP_qc[j]
    dP_mq = q_in + seg_q_sum + q_out                           # S3 (ahead of the distributor) above S4
    dP_mq_ch = seg_q_sum                                       # across the channels

    # ---- walls of suction and discharge lines, receiver shell
    Q_sg = p.UA_sg * (T_sw - c_T[n])                  # suction line wall -> refrigerant
    dT_sw = (p.UA_sa * (T_amb - T_sw) - Q_sg) / p.C_sw
    Q_dg = p.UA_dg * (T_dw - D.T)                     # wall -> discharge gas
    dT_dw = (p.UA_da * (T_amb - T_dw) - Q_dg) / p.C_dw
    Q_rw = p.rec_UA_r * (T_rw - I.T)                  # receiver shell -> refrigerant
    dT_rw = (p.rec_UA_a * (T_amb - T_rw) - Q_rw) / p.C_rw

    # ---- condenser, water side.  Water enters at the liquid end: it first subcools the
    # leaving liquid (flooded zone, or the draining condensate film on cond_sc_film of
    # the area), then cools the wall.  The plant loop's supply-to-return difference
    # drives it through valve 4, the piping and the condenser in series.
    f_w = valve_fraction(u4, p.w_code, p.w_R)
    C_cw = kv_to_C(p.cond_Kv_w)
    mdot_w = water_valve_flow(series_C3(kv_to_C(p.Kv_w) * f_w, C_cw, kv_to_C(p.Kv_wpipe)),
                              p.P_w_sup - p.P_w_ret, p.rho_w)
    dP_cw = hx_drop(mdot_w, C_cw, p.rho_w)
    Cw = mdot_w * p.cp_w
    UA_w = p.alpha_w0 * p.cond_A * max(mdot_w / p.mdot_w_ref, 1e-6) ** 0.8
    a_sc = max(cond_flood, p.cond_sc_film) * (1.0 - dry)
    T_l = I.T if I.x < 0.0 else I.T_sat                       # liquid entering the zone
    C_l = max(mdot_3, 0.0) * I.cp_l
    C_min, C_max = min(C_l, Cw), max(C_l, Cw)
    UA_sc0 = p.alpha_sc * p.cond_A
    UA_sc = a_sc * UA_sc0 * UA_w / (UA_sc0 + UA_w)             # liquid side in series with water side
    eps_sc = counterflow_effectiveness(UA_sc / max(C_min, 1e-9), C_min / max(C_max, 1e-9))
    Q_sc = eps_sc * C_min * max(T_l - T_wi, 0.0)              # liquid -> water
    h_co = h_cv_out - Q_sc / max(mdot_3, 1e-9)
    T_co = (T_l - Q_sc / max(C_l, 1e-9)) * (1.0 - dry) + I.T * dry
    SC = I.T_sat - T_co
    h_3i = h_cv_out if fwd3 else H[0]                         # leaving the intermediate section
    h_3f = h_co if fwd3 else H[0]                             # entering the quench side
    T_w1 = T_wi + Q_sc / max(Cw, 1e-9)               # water leaving the subcooled zone

    # ---- condenser, refrigerant -> walls: the hot gas surplus of the header (or the
    # section's vapor) marched from S3 down to the liquid
    h_vap_i = max(h_i, I.h_v)                         # the section's vapor
    T_iv = I.T if I.x >= 1.0 else sat_i.T_v
    Q_c, h_cm = np.empty(n), np.empty(n)
    m_cr = _condenser(p, tab, P_i, sat_i, T_cw, f_fl, h_d, h_vap_i, T_iv, max(mdot_1 - max(mdot_2, 0.0), 0.0),
                      Q_c, h_cm)
    Q_r = 0.0
    for j in range(n):
        Q_r += Q_c[j]

    # ---- condenser, walls -> water: counterflow, the water rising from S1 (bottom, after
    # the subcooled zone at its inlet end) to S2 through the wall cells.  Every wall cell
    # meets the water over its whole area (a flooded cell's wall follows the water)
    T_wat = T_w1
    Q_ww = 0.0
    T_wc, dT_cw = np.empty(n), np.empty(n)
    eps_w = 1.0 - math.exp(-UA_w / n / max(Cw, 1e-9))
    for j in range(n - 1, -1, -1):
        dTw = eps_w * (T_cw[j] - T_wat)
        Q_wj = Cw * dTw                                          # wall -> water
        T_wc[j] = T_wat + 0.5 * dTw
        T_wat += dTw
        Q_ww += Q_wj
        dT_cw[j] = (Q_c[j] - Q_wj + (p.UA_ca / n) * (T_amb - T_cw[j])) / (p.C_cw / n)
    Q_w = Q_sc + Q_ww                                # water duty
    T_wo = T_wat if Cw > 1e-9 else T_wi

    # ---- mixing exchanger, gas side: quasi-steady march from S1 (bottom cell n-1)
    # up to S2 against the plate walls; the gas does not condense (walls below
    # saturation are clamped)
    m2p = max(mdot_2, 0.0)
    UA_g = p.mx_alpha_g0 * p.A_mx_cell * (max(m2p, 1e-3 * p.mx_mdot_g_ref) / p.mx_mdot_g_ref) ** 0.8
    hg = h_2f
    Q_g = np.empty(n)
    Q_g_sum = 0.0
    for j in range(n - 1, -1, -1):
        Tw_eff = max(T_mw[j], T_sat_s)
        # vapor at the wall temperature (linearized from the dew point; an effectiveness target)
        h_eq = h_vs + sat_s.cp_v * max(Tw_eff - sat_s.T_v, 0.0)
        dT = kp.T_vapor(tab, P_s, hg) - Tw_eff
        dh = hg - h_eq
        cp_s = _clip(dh / dT if abs(dT) > 0.05 else 1100.0, 300.0, 1e5)
        Q = (1.0 - math.exp(-UA_g / max(m2p * cp_s, 1e-9))) * m2p * dh
        hg = hg - Q / max(m2p, 1e-12)
        Q_g[j] = Q
        hg_out[j] = hg
    for j in range(n):
        Q_g_sum += Q_g[j]
    h_go = hg                                        # leaving at S2
    rho_gc = np.empty(n)
    Fg_b = 0.0
    for j in range(n):
        rho_gc[j] = kp.vapor_props(tab, P_s, hg_out[j])[1]
        Fg_b += p.V_gc[j] * (1.2 * rho_gc[j] / P_s)   # drho/dP|h of the vapor (ideal-gas-like)
    Fg_b = -Fg_b

    # ---- mixing exchanger, quench side: plate -> quench, by flow regime
    m3p = max(mdot_3, 0.0)
    fq = max(m3p / p.mx_mdot_q_ref, 0.04)
    UA_e = p.mx_alpha_e * p.A_mx_cell * math.sqrt(fq)
    UA_v = p.mx_alpha_v0 * p.A_mx_cell * fq ** 0.8
    # stream entering each cell (upwind): valve 3 / the cell above, or from below on reverse flow
    H_up, phi_f, Q_2ph = np.empty(n), np.empty(n), np.empty(n)
    Q_q, dT_mw = np.empty(n), np.empty(n)
    Q_q_sum = 0.0
    for j in range(n):
        if fwd3:
            H_up[j] = h_3f if j == 0 else H[j - 1]
        else:
            H_up[j] = H[j + 1] if j < n - 1 else h_l1
        dT_sat = T_mw[j] - (T_sat_s + dTs[j])
        Q_2ph[j] = UA_e * dT_sat
        need = abs(mdot_3) * max(h_vs - H_up[j], 0.0)
        phi_f[j] = _clip(need / max(UA_e * max(dT_sat, 0.0), 1e-9), 0.0, 1.0)
        Q_q[j] = _quench_heat(H[j], c_T[j], T_mw[j], H_up[j], dTs[j], phi_f[j], Q_2ph[j], UA_e, UA_v, h_vs)
        Q_q_sum += Q_q[j]
        dT_mw[j] = (Q_g[j] - Q_q[j] + (p.mx_UA_a / n) * (T_amb - T_mw[j])) / p.C_mw_cell

    # ---- lumped-pressure balances of the suction side.  In every cell
    #   M dh/dt = F_in (h_in - h) + Q + V dP/dt,   dM/dt = V (rho_P dP/dt + rho_h dh/dt)
    # and the flow leaving a cell is F_in - dM/dt; all are affine in dP/dt, which
    # follows from the compressor drawing mdot_c out of the last cell.  For dP/dt the
    # enthalpy rates respond to the pressure only through the compression term V/M
    # (the inflows are taken at their dP/dt = 0 values): every cell then contributes
    # its isentropic capacitance V (rho_P + rho_h / rho) > 0, so the pressure solve
    # stays well posed even where cold liquid enters a vapor-filled cell.
    A_face, B_face = np.empty(n + 1), np.empty(n + 1)
    a_f, b_f = mdot_3, 0.0                           # flow entering the current quench cell
    A_face[0], B_face[0] = mdot_3, 0.0
    for j in range(n):
        if fwd3:
            h_up = h_3f if j == 0 else H[j - 1]
            adv_a = a_f
        else:
            h_up = H[j + 1] if j < n - 1 else h_l1
            adv_a = -mdot_3
        c = (adv_a * (h_up - H[j]) + Q_q[j]) / M_c[j]
        d = Vc[j] / M_c[j]
        a_f = a_f - Vc[j] * rho_h[j] * c
        b_f = b_f - Vc[j] * (rho_P[j] + rho_h[j] * d)
        A_face[j + 1], B_face[j + 1] = a_f, b_f
    Fq_a, Fq_b = a_f, b_f                            # quench outlet (S4) -> tee
    Fg_a = mdot_2                                    # gas outlet (S2) -> tee
    wq = 1.0 if fwd3 else 0.0
    wg = 1.0 if fwd2 else 0.0
    dhq, dhg = H[n - 1] - h_l1, h_go - h_l1
    c = (wg * Fg_a * dhg + wq * Fq_a * dhq + Q_sg) / M_c[n]
    d = Vc[n] / M_c[n]
    F12_a = Fg_a + Fq_a - Vc[n] * rho_h[n] * c
    F12_b = Fg_b + Fq_b - Vc[n] * (rho_P[n] + rho_h[n] * d)
    w12 = 1.0 if F12_a > 0.0 else 0.0
    c = (w12 * F12_a * (h_l1 - h_l2) - mdot_c * (h_cin - h_l2)) / M_c[n + 1]
    d = Vc[n + 1] / M_c[n + 1]
    Fo_a = F12_a - Vc[n + 1] * rho_h[n + 1] * c
    Fo_b = F12_b - Vc[n + 1] * (rho_P[n + 1] + rho_h[n + 1] * d)
    dP_s = (mdot_c - Fo_a) / min(Fo_b, -1e-15)
    # With dP/dt known, every face flow follows; the enthalpy rates are evaluated
    # again with the upwind side chosen by each face's actual direction (a fast
    # pressure rise can briefly reverse the flow between quench cells).
    Pr = 0.0 if hold_P_s else dP_s
    F = np.empty(n + 1)
    for j in range(n + 1):
        F[j] = A_face[j] + B_face[j] * Pr             # into quench cell j through its top face j
    Fg_now = Fg_a + Fg_b * Pr
    F12 = F12_a + F12_b * Pr
    for j in range(n):
        h_above = h_3f if j == 0 else H[j - 1]
        h_below = H[j + 1] if j < n - 1 else h_l1
        dx[X_HQ + j] = (max(F[j], 0.0) * (h_above - H[j]) + max(-F[j + 1], 0.0) * (h_below - H[j])
                        + Q_q[j] + Vc[j] * Pr) / M_c[j]
    dx[X_HQ + n] = (max(Fg_now, 0.0) * dhg + max(F[n], 0.0) * dhq + max(-F12, 0.0) * (h_l2 - h_l1)
                    + Q_sg + Vc[n] * Pr) / M_c[n]
    dx[X_HQ + n + 1] = (max(F12, 0.0) * (h_l1 - h_l2) - mdot_c * (h_cin - h_l2) + Vc[n + 1] * Pr) / M_c[n + 1]

    # ---- fastest local rates (1/s) for the integrator's step subdivision: valve
    # conductance over the capacitance of each pressure node, and advection plus
    # heat transfer of the quench cells
    k1 = _valve_slope(C1, max(D.rho, rho_g), P_d - P_h, p.eps_valve)
    k2 = _valve_slope(C2, rho_g, P_h - P_s, p.eps_valve)
    k3 = _valve_slope(kv_to_C(p.Kv_stv) * f3, rho_co, P_i - P_s, p.eps_valve)
    C_d = p.V_d * max(D.drho_dP + D.drho_dh / D.rho, 1e-12)
    C_i = p.V_i * max(I.drho_dP + I.drho_dh / I.rho, 1e-12)
    C_s = max(-Fo_b, 1e-12)
    # (heat flow sensitivity to the cell's own enthalpy, by a finite difference of the
    # heat transfer formula; the temperature moves only where the cell holds vapor)
    dh_fd = 200.0
    lam = _nanmax(_nanmax(k1 / C_d, (k1 + k2 + k3) / C_i), (k2 + k3 + mdot_c / P_s) / C_s)
    lam_q = 0.0
    for j in range(n):
        dT_fd = dh_fd / 1100.0 if c_x[j] >= 1.0 else 0.0
        dQ = _quench_heat(H[j] + dh_fd, c_T[j] + dT_fd, T_mw[j], H_up[j], dTs[j], phi_f[j], Q_2ph[j],
                          UA_e, UA_v, h_vs) - Q_q[j]
        F_cell = max(abs(F[j]), abs(F[j + 1]))
        r = (F_cell + abs(dQ) / dh_fd) / M_c[j]
        lam_q = r if j == 0 else _nanmax(lam_q, r)
    lam = _nanmax(lam, lam_q)

    # ---- tee and suction line: droplets from the quench outlet evaporate on the way to
    # the probe; the probe reads the vapor temperature
    Fq = max(Fq_a + Fq_b * dP_s, 0.0) * wq
    Fg = max(Fg_a + Fg_b * dP_s, 0.0) * wg
    F_in = Fq + Fg
    x_qo = x_c[n - 1]
    y_in = _clip(1.0 - x_qo, 0.0, 1.0) * Fq / max(F_in, 1e-9)
    t_res = M_c[n] / max(F_in, 1e-6)
    y_eq = _clip(1.0 - c_x[n], 0.0, 1.0)
    y_liq = max(y_in * math.exp(-t_res / p.tee_tau_evap), y_eq)
    h_vap = (h_l1 - y_liq * h_ls) / max(1.0 - y_liq, 1e-3) if y_liq > 1e-12 else h_l1
    T_port = kp.T_vapor(tab, P_s, h_vap) if y_liq < 0.999 else c_T[n]
    x_out = 1.0 - y_liq if y_liq > 0.0 else c_x[n]

    # ---- compressor shell
    dT_sh = (Q_gs + (1.0 - p.f_motor_gas) * Q_motor - p.UA_sha * (T_sh - T_amb)) / p.C_shell

    # ---- lumped volumes with their own pressure
    dm_d = mdot_c - mdot_1
    E_d = mdot_c * (h2 - h_d) - mdot_1 * (h_1f - h_d) + Q_dg
    dP_d, dh_d = cv_balance(p.V_d, D.rho, D.drho_dP, D.drho_dh, dm_d, E_d)
    dm_i = mdot_1 - mdot_2 - mdot_3
    E_i = mdot_1 * (h_1f - h_i) - mdot_2 * (h_2f - h_i) - mdot_3 * (h_3i - h_i) - Q_r + Q_rw
    dP_i, dh_i = cv_balance(p.V_i, I.rho, I.drho_dP, I.drho_dh, dm_i, E_i)

    # ---- speed, actuators, sensors
    Nc = _clip(N_cmd, p.N_min, p.N_max) if N_cmd > 0.0 else 0.0
    dN = _clip((Nc - N) / p.tau_N, -p.ramp_N, p.ramp_N)

    dx[X_P_S] = dP_s
    for j in range(n):
        dx[X_TMW + j] = dT_mw[j]
        dx[X_HG + j] = 0.0
    dx[X_T_SW] = dT_sw
    dx[X_P_D], dx[X_H_D], dx[X_T_DW] = dP_d, dh_d, dT_dw
    dx[X_P_I], dx[X_H_I], dx[X_T_RW] = dP_i, dh_i, dT_rw
    for j in range(n):
        dx[X_TCW + j] = dT_cw[j]
    dx[X_T_SH], dx[X_N] = dT_sh, dN
    dx[X_U1] = actuator_rate(u1, u_cmd[0], p.tau_dpv, p.rate_dpv)
    dx[X_U1 + 1] = actuator_rate(u2, u_cmd[1], p.tau_spv, p.rate_spv)
    dx[X_U1 + 2] = actuator_rate(u3, u_cmd[2], p.tau_stv, p.rate_stv)
    dx[X_U1 + 3] = actuator_rate(u4, u_cmd[3], p.tau_w, p.rate_w)
    dx[X_TM_S] = (T_port - Tm_s) / p.tau_T
    dx[X_TM_D] = (D.T - Tm_d) / p.tau_T
    dx[X_TM_CO] = (T_co - Tm_co) / p.tau_T
    dx[X_MM] = (mdot_c - mm) / p.tau_m
    dx[X_WM] = (W_el - Wm) / p.tau_W
    # conserved states: net mass flow and sum(m h) + Q over each boundary
    dx[X_M_S] = mdot_2 + mdot_3 - mdot_c
    dx[X_U_S] = mdot_2 * h_2f + mdot_3 * h_3f - mdot_c * h_cin + Q_sg + Q_q_sum - Q_g_sum
    dx[X_M_D] = dm_d
    dx[X_U_D] = E_d + h_d * dm_d
    dx[X_M_I] = dm_i
    dx[X_U_I] = E_i + h_i * dm_i
    if not want_aux:
        return lam

    M_g = 0.0
    for j in range(n):
        M_g += rho_gc[j] * p.V_gc[j]
    M_s = 0.0
    M_q_liq = 0.0
    for j in range(NS):
        M_s += M_c[j]
    for j in range(n):
        M_q_liq += (1.0 - _clip(c_x[j], 0.0, 1.0)) * M_c[j]
    M_s += M_g
    M_d = D.rho * p.V_d
    a[A_P_s], a[A_P_d], a[A_P_i] = P_s, P_d, P_i
    a[A_T_s], a[A_T_d], a[A_T_i], a[A_T_co] = T_port, D.T, I.T, T_co
    a[A_T_sat_s], a[A_T_sat_d], a[A_T_sat_i] = T_sat_s, D.T_sat, I.T_sat
    a[A_SH], a[A_SC], a[A_x_out], a[A_y_liq] = T_port - T_sat_s, SC, x_out, y_liq
    a[A_x_l1], a[A_x_i], a[A_x_qo], a[A_T_qo] = c_x[n], I.x, x_qo, c_T[n - 1]
    a[A_T_go], a[A_h_go], a[A_T_l1] = kp.T_vapor(tab, P_s, h_go), h_go, c_T[n]
    a[A_fill_i], a[A_rec_level], a[A_ll_fill], a[A_cond_flood] = fill_i, rec_level, ll_fill, cond_flood
    a[A_M_q_liq] = M_q_liq
    for j in range(n):
        a[A_h_q + j], a[A_x_q + j] = H[j], c_x[j]
        a[A_T_q + j] = c_T[j] + (1.0 - smoothstep((H[j] - h_vs) / DRYOUT_BAND)) * dTs[j]
        a[A_T_g + j] = kp.vapor_props(tab, P_s, hg_out[j])[0]
        a[A_h_g + j], a[A_T_mw + j], a[A_dP_qc + j] = hg_out[j], T_mw[j], dP_qc[j]
    a[A_Q_mx], a[A_Q_q] = Q_g_sum, Q_q_sum
    a[A_rho_s], a[A_h_l1], a[A_h_l2], a[A_h_cin], a[A_x_l2] = Cin.rho, h_l1, h_l2, h_cin, c_x[n + 1]
    a[A_h_d], a[A_h_i], a[A_h_co], a[A_h2], a[A_h_2f], a[A_h_3f] = h_d, h_i, h_co, h2, h_2f, h_3f
    a[A_T2_ad] = T2_ad
    a[A_P_tee], a[A_dP_suc], a[A_dP_dis], a[A_dP_hdr] = P_tee, dP_suc, dP_dis, dP_hdr1 + dP_hdr2
    a[A_dP_bp], a[A_dP_mog], a[A_dP_moq], a[A_dP_q], a[A_dP_liq] = dP_bp, dP_mog, dP_moq, dP_q, dP_liq
    a[A_dP_drn], a[A_P_h], a[A_dP_cr], a[A_dP_cw], a[A_dP_mg] = dP_drn, P_h, dP_cr, dP_cw, dP_mg
    a[A_dP_mq], a[A_dP_mq_ch] = dP_mq, dP_mq_ch
    a[A_dP_q_dist], a[A_dP_q_in], a[A_dP_q_out], a[A_mdot_cr] = q_dist, q_in, q_out, m_cr
    a[A_mdot_c], a[A_mdot_1], a[A_mdot_2], a[A_mdot_3], a[A_mdot_w] = mdot_c, mdot_1, mdot_2, mdot_3, mdot_w
    a[A_W_el], a[A_W_shaft], a[A_Pr], a[A_eta_v], a[A_eta_s] = W_el, W_shaft, Pr_c, eta_v, eta_s
    a[A_Q_r], a[A_Q_w], a[A_Q_sc], a[A_Q_sg], a[A_Q_dg], a[A_Q_rw] = Q_r, Q_w, Q_sc, Q_sg, Q_dg, Q_rw
    T_cw_mean = 0.0
    for j in range(n):
        T_cw_mean += T_cw[j] / n
    a[A_T_wo], a[A_T_sw], a[A_T_dw], a[A_T_cw], a[A_T_rw], a[A_T_sh] = T_wo, T_sw, T_dw, T_cw_mean, T_rw, T_sh
    # condenser cells: above the liquid the marched profile, flooded shares hold liquid
    # subcooled halfway between the bubble point and the valve 3 inlet
    h_fl = sat_i.h_l - I.cp_l * max(sat_i.T_l - 0.5 * (T_l + T_co), 0.0)
    for j in range(n):
        hc = (1.0 - f_fl[j]) * h_cm[j] + f_fl[j] * h_fl
        a[A_h_c + j], a[A_x_c + j] = hc, (hc - sat_i.h_l) / max(sat_i.h_v - sat_i.h_l, 1e3)
        a[A_T_c + j] = kp.state(tab, P_i, hc).T
        a[A_T_wc + j], a[A_T_cwc + j] = T_wc[j], T_cw[j]
    a[A_N] = N
    for k in range(4):
        a[A_u1 + k] = x[X_U1 + k]
    a[A_M_s], a[A_M_g], a[A_M_d], a[A_M_i], a[A_M_tot] = M_s, M_g, M_d, M_i, M_s + M_d + M_i
    a[A_Tm_s], a[A_Tm_d], a[A_Tm_co], a[A_mm], a[A_Wm] = Tm_s, Tm_d, Tm_co, mm, Wm
    for k in range(NX):
        a[A_dx + k] = dx[k]
    return lam


# ------------------------------------------------------- conserved states
@njit(cache=True)
def suction_inventory(x, p, tab, delta):
    """Mass, internal energy and their derivatives with respect to the common
    pressure and a uniform enthalpy shift ``delta`` of the dynamic suction cells."""
    P = x[X_P_S]
    Mc = Uc = MPc = Md = UPc = Ud = 0.0
    for j in range(NS):
        H = x[X_HQ + j] + delta
        S = kp.state(tab, P, H)
        V = p.V_sc[j]
        Mc += S.rho * V
        Uc += S.rho * V * H
        MPc += S.drho_dP * V
        Md += S.drho_dh * V
        UPc += S.drho_dP * V * H
        Ud += (S.rho + S.drho_dh * H) * V
    Mg = Ug = MPg = UPg = 0.0
    for j in range(MX):
        Hg = x[X_HG + j]
        rho_g = kp.vapor_props(tab, P, Hg)[1]
        rP_g = 1.2 * rho_g / P
        Vg = p.V_gc[j]
        Mg += rho_g * Vg
        Ug += rho_g * Vg * Hg
        MPg += rP_g * Vg
        UPg += rP_g * Vg * Hg
    return Mc + Mg, Uc + Ug - P * p.V_s, MPc + MPg, Md, UPc + UPg - p.V_s, Ud


@njit(cache=True)
def project_suction(x, p, tab):
    """Correct the common suction pressure and shift the dynamic cell
    enthalpies uniformly so that the suction side holds exactly its
    conserved mass and energy (Newton; mass-only step where the joint
    step is ill-conditioned)."""
    Mt, Ut = x[X_M_S], x[X_U_S]
    delta = 0.0
    for _ in range(4):
        M, U, MP, Md, UP, Ud = suction_inventory(x, p, tab, delta)
        rM, rU = M - Mt, U - Ut
        if abs(rM) <= 1e-9 * Mt and abs(rU) <= 1e-9 * abs(Ut) + 1e-3:
            break
        det = MP * Ud - Md * UP
        den = det if det != 0.0 else 1.0
        dP = (rM * Ud - rU * Md) / den
        dd = (MP * rU - UP * rM) / den
        P = x[X_P_S]
        if not (det > 0.0 and abs(dP) < 0.05 * P and abs(dd) < 2e4):
            dP = rM / MP
            dd = 0.0
        x[X_P_S] = _clip(P - dP, tab.p_min * 1.001, tab.p_max * 0.999)
        delta = delta - dd
    for j in range(NS):
        x[X_HQ + j] += delta


@njit(cache=True)
def flash_rho_u(tab, rho_t, u_t, P0):
    """(P, h) of a volume with density ``rho_t`` and internal energy ``u_t``.

    The energy constraint fixes h = u + P / rho, and along that path the
    density is a monotonic function of P in every phase region, so a
    bracketed Newton iteration on P converges to the unique solution
    (the previous pressure ``P0`` is the starting point)."""
    lo = tab.p_min * 1.001
    hi = tab.p_max * 0.999
    P = _clip(P0, lo, hi)
    for _ in range(8):
        S = kp.state(tab, P, u_t + P / rho_t)
        r = S.rho - rho_t
        if abs(r) <= 1e-5 * rho_t:
            break
        if r > 0.0:
            hi = min(hi, P)
        elif r < 0.0:
            lo = max(lo, P)
        slope = max(S.drho_dP + S.drho_dh / rho_t, 1e-12)
        Pn = P - r / slope
        if not np.isfinite(Pn) or Pn <= lo or Pn >= hi:
            Pn = 0.5 * (lo + hi)
        P = Pn
    return P, u_t + P / rho_t


@njit(cache=True)
def post(x, p, tab):
    """Limits of speed and valve positions, then (P, h) of every volume from its
    conserved mass and energy."""
    x[X_N] = max(x[X_N], 0.0)
    for k in range(4):
        x[X_U1 + k] = _clip(x[X_U1 + k], 0.0, 1.0)
    M = x[X_M_D]
    x[X_P_D], x[X_H_D] = flash_rho_u(tab, M / p.V_d, x[X_U_D] / M, x[X_P_D])
    M = x[X_M_I]
    x[X_P_I], x[X_H_I] = flash_rho_u(tab, M / p.V_i, x[X_U_I] / M, x[X_P_I])
    project_suction(x, p, tab)


@njit(cache=True)
def sync_mass(x, p, tab):
    """Conserved mass and energy states from (P, h) (after direct state assignments)."""
    S = kp.state(tab, x[X_P_D], x[X_H_D])
    x[X_M_D] = S.rho * p.V_d
    x[X_U_D] = x[X_M_D] * (x[X_H_D] - x[X_P_D] / S.rho)
    S = kp.state(tab, x[X_P_I], x[X_H_I])
    x[X_M_I] = S.rho * p.V_i
    x[X_U_I] = x[X_M_I] * (x[X_H_I] - x[X_P_I] / S.rho)
    M, U, _, _, _, _ = suction_inventory(x, p, tab, 0.0)
    x[X_M_S], x[X_U_S] = M, U


# ------------------------------------------------------------ integration
@njit(cache=True)
def stiff(x, p, tab):
    """True if a volume is (nearly) liquid-full: subcooled liquid, or a two-phase
    mixture whose liquid volume fraction exceeds ``STIFF_FILL`` (lever rule on
    the saturation properties); the suction side counts as one volume."""
    for iP, ih in ((X_P_D, X_H_D), (X_P_I, X_H_I)):
        s = kp.sat(tab, x[iP])
        xq = (x[ih] - s.h_l) / (s.h_v - s.h_l)
        fill = (1.0 - xq) / (1.0 + xq * (s.rho_l / s.rho_v - 1.0))
        if xq < 0.0 or fill > STIFF_FILL:
            return True
    s = kp.sat(tab, x[X_P_S])
    liq = 0.0
    for j in range(NS):
        xq = (x[X_HQ + j] - s.h_l) / (s.h_v - s.h_l)
        fill = (1.0 - xq) / (1.0 + xq * (s.rho_l / s.rho_v - 1.0))
        liq += _clip(fill, 0.0, 1.0) * p.V_sc[j]
    return liq / p.V_s > STIFF_FILL


@njit(cache=True)
def subdivisions(x, lam, p, tab, dt):
    """Sub-steps needed for the next step (a power of two): the fastest local
    rate (from the last right-hand side evaluation) times the sub-step must stay
    inside RK4's stable range; liquid-full volumes take at least ``STIFF_SUBDIV``."""
    lam = lam if np.isfinite(lam) else np.inf
    k = math.ceil(min(lam * dt / LAM_DT_MAX, MAX_SUBDIV))
    if stiff(x, p, tab):
        k = max(k, STIFF_SUBDIV)
    k = min(max(k, 1), MAX_SUBDIV)
    return min(2 ** math.ceil(math.log2(k)), MAX_SUBDIV)


@njit(cache=True)
def plausible(x0, x1):
    for k in range(NX):
        if not np.isfinite(x1[k]):
            return False
    for j in range(NS):
        if abs(x1[X_HQ + j] - x0[X_HQ + j]) >= MAX_DH_STEP:
            return False
    for i in (X_P_S, X_P_D, X_P_I):
        if abs(x1[i] / x0[i] - 1.0) >= MAX_DP_STEP:
            return False
    return True


@njit(cache=True)
def step_once(x, u, N_cmd, T_amb, T_wi, p, tab, h, integrator, work):
    """One sub-step of length ``h``; returns (new state, rate estimate of its last stage)."""
    k1, k2, k3, k4, xs, hg = work[0], work[1], work[2], work[3], work[4], work[5, :MX]
    aux = work[5, :0]
    lam = rhs(x, u, N_cmd, T_amb, T_wi, p, tab, False, k1, False, aux, hg)
    if integrator == 0:                 # RK4
        for i in range(NX):
            xs[i] = x[i] + 0.5 * h * k1[i]
        rhs(xs, u, N_cmd, T_amb, T_wi, p, tab, False, k2, False, aux, hg)
        for i in range(NX):
            xs[i] = x[i] + 0.5 * h * k2[i]
        rhs(xs, u, N_cmd, T_amb, T_wi, p, tab, False, k3, False, aux, hg)
        for i in range(NX):
            xs[i] = x[i] + h * k3[i]
        lam = rhs(xs, u, N_cmd, T_amb, T_wi, p, tab, False, k4, False, aux, hg)
        xn = np.empty(NX)
        for i in range(NX):
            xn[i] = x[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    elif integrator == 1:               # Heun
        for i in range(NX):
            xs[i] = x[i] + h * k1[i]
        lam = rhs(xs, u, N_cmd, T_amb, T_wi, p, tab, False, k2, False, aux, hg)
        xn = np.empty(NX)
        for i in range(NX):
            xn[i] = x[i] + 0.5 * h * (k1[i] + k2[i])
    else:                               # explicit Euler
        xn = np.empty(NX)
        for i in range(NX):
            xn[i] = x[i] + h * k1[i]
    for j in range(MX):
        xn[X_HG + j] = hg[j]            # gas side profile of the last stage
    post(xn, p, tab)
    return xn, lam


@njit(cache=True)
def _level2(x, u, N_cmd, T_amb, T_wi, p, tab, h, integrator, work):
    return step_once(x, u, N_cmd, T_amb, T_wi, p, tab, h, integrator, work)


@njit(cache=True)
def _level1(x, u, N_cmd, T_amb, T_wi, p, tab, h, integrator, work):
    xn, lam = step_once(x, u, N_cmd, T_amb, T_wi, p, tab, h, integrator, work)
    if not plausible(x, xn):
        xn = x
        for _ in range(4):
            xn, lam = _level2(xn, u, N_cmd, T_amb, T_wi, p, tab, 0.25 * h, integrator, work)
    return xn, lam


@njit(cache=True)
def advance(x, u, N_cmd, T_amb, T_wi, p, tab, dt, k, integrator, work):
    """One step of ``dt`` in ``k`` sub-steps.  A sub-step that produces a
    non-finite state or an implausible jump (the local-rate estimate can miss a
    nonlinear transient) is redone with four times finer steps (twice at most)."""
    h = dt / k
    lam = np.nan
    for _ in range(k):
        xn, lam = step_once(x, u, N_cmd, T_amb, T_wi, p, tab, h, integrator, work)
        if not plausible(x, xn):
            xn = x
            for _ in range(4):
                xn, lam = _level1(xn, u, N_cmd, T_amb, T_wi, p, tab, 0.25 * h, integrator, work)
        x = xn
    return x, lam


# ----------------------------------------------------------------- batches
# Each batch function is compiled twice from the same source: parallel over the
# environments (numba's thread pool, NUMBA_NUM_THREADS) and serial (a single
# environment, or several processes each running its own batch).
def _step_batch(X, U, N_cmd, T_amb, T_wi, P, tab, lam, dt, n_sub, integrator, subs):
    """Advance every environment (row of ``X``) by ``n_sub`` steps of ``dt``,
    each in the sub-steps its own rate estimate ``lam`` calls for (updated in
    place); ``subs`` receives the number of sub-steps taken."""
    for e in prange(X.shape[0]):
        work = np.empty((6, NX))
        x = X[e].copy()
        l = lam[e]
        cnt = 0
        for _ in range(n_sub):
            k = subdivisions(x, l, P[e], tab, dt)
            x, l = advance(x, U[e], N_cmd[e], T_amb[e], T_wi[e], P[e], tab, dt, k, integrator, work)
            cnt += k
        X[e] = x
        lam[e] = l
        subs[e] = cnt


def _rhs_batch(X, U, N_cmd, T_amb, T_wi, P, tab, hold_P_s, DX, want_aux, AUX, HG, LAM):
    for e in prange(X.shape[0]):
        LAM[e] = rhs(X[e], U[e], N_cmd[e], T_amb[e], T_wi[e], P[e], tab, hold_P_s, DX[e], want_aux,
                     AUX[e], HG[e])


step_batch = batch_variants(_step_batch)
rhs_batch = batch_variants(_rhs_batch)


@njit(cache=True)
def sync_mass_batch(X, P, tab):
    for e in range(X.shape[0]):
        sync_mass(X[e], P[e], tab)


@njit(cache=True)
def stiff_batch(X, P, tab, out):
    for e in range(X.shape[0]):
        out[e] = stiff(X[e], P[e], tab)
