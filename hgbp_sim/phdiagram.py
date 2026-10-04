"""Pressure-enthalpy (P-h) diagram data for the live stand.

* :func:`state_points`: the stand's state points along every flow path, from the
  model outputs of one step (pressures along each path from the model's pressure
  chain, enthalpies of the cells and volumes the model carries).  Exchanger cells are
  the state leaving each cell, at the pressure where it leaves, as the schematic
  shows them: the last cell of each exchanger side is its outlet.
* :func:`compression_paths`: the compression drawn as a polytropic path between the
  compressor model's states, and the isentrope from the suction port.
* :func:`chart`: the refrigerant's saturation lines, isotherms and isentropes from the property
  tables, as the diagram's background, for the whole dome or for one view (a zoomed
  view gets its own pressure grid and isotherms at a spacing to suit it,
  :func:`view_temperatures`).

Pressures in bar (absolute), enthalpies in kJ/kg, temperatures in degC.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .kernel import props as kp

C2K = 273.15

# path order of the points (the UI draws its legs from these names)
#   suc, dis                          compressor suction port and discharge (probe)
#   v1i, hdr                          valve 1 inlet, hot gas header at the valve 2 branch
#   cin, c0..c4                       condenser inlet S3, refrigerant cells (top -> bottom)
#   rec                               receiver liquid pool
#   v3i, v3o, qin, q0..q4, qout       valve 3 in / out, quench S3, cells (top -> bottom), S4
#   v2o, gin, g4..g0, gout            valve 2 out, gas S1, cells (bottom -> top), S2
#   tee                               mixed suction gas at the tee


def _f(aux, k):
    return float(np.asarray(aux[k]).ravel()[0])


def state_points(aux, props) -> dict:
    """Named state points ``name -> [P bar, h kJ/kg, T degC, x, dT K]`` of one stand
    (row 0 of the outputs).  ``x`` is the vapor quality (below 0 subcooled, above 1
    superheated, by enthalpy); ``dT`` the subcooling (liquid) or superheat (vapor),
    0 inside the dome."""
    g = lambda k: _f(aux, k)
    vec = lambda k: np.asarray(aux[k])[0]
    n = len(vec("h_c"))
    P_s, P_d, P_i, P_h = g("P_s"), g("P_d"), g("P_i"), g("P_h")
    P_tee = P_s + g("dP_suc")
    h_d, h_L, h_co, h_3f, h_2f = g("h_d"), g("h_L"), g("h_co"), g("h_3f"), g("h_2f")
    h_q, h_g, h_c = vec("h_q"), vec("h_g"), vec("h_c")
    P_qf, P_gf, P_cf = vec("P_qf"), vec("P_gf"), vec("P_cf")      # where each cell's flow leaves it
    # condenser: S3 above the outlet by the condensing flow's drop (static head included)
    P_c4 = P_i + g("dP_drn")
    P_c3 = P_c4 + g("dP_cr")
    # mixing exchanger: S2 / S4 above the tee by the outlet legs, S1 / S3 above them
    P_g2 = P_tee + g("dP_mog")
    P_g1 = P_g2 + g("dP_mg")
    P_q4 = P_tee + g("dP_moq")
    P_q3 = P_q4 + g("dP_mq")
    # the tee mixes the two outlets (flows from the valves: steady-state weights)
    m2, m3 = max(g("mdot_2"), 0.0), max(g("mdot_3"), 0.0)
    h_tee = (m2 * float(h_g[0]) + m3 * float(h_q[-1])) / (m2 + m3) if m2 + m3 > 1e-6 else g("h_l1")
    pts = dict(suc=(P_s, g("h_cin")), dis=(P_d, h_d), v1i=(P_d - g("dP_dis"), h_d), hdr=(P_h, h_d),
               cin=(P_c3, h_d))
    for j in range(n):
        pts[f"c{j}"] = (float(P_cf[j]), float(h_c[j]))
    pts["rec"] = (P_i, h_L)
    pts["v3i"] = (P_i - g("dP_liq"), h_co)
    pts["v3o"] = (P_q3 + g("dP_q"), h_co)
    pts["qin"] = (P_q3, h_3f)
    for j in range(n):
        pts[f"q{j}"] = (float(P_qf[j]), float(h_q[j]))
    pts["qout"] = (P_q4, float(h_q[-1]))
    pts["v2o"] = (P_g1 + g("dP_bp"), h_2f)
    pts["gin"] = (P_g1, h_2f)
    for j in range(n - 1, -1, -1):
        pts[f"g{j}"] = (float(P_gf[j]), float(h_g[j]))
    pts["gout"] = (P_g2, float(h_g[0]))
    pts["tee"] = (P_tee, h_tee)
    # inside the compressor (components.compressor_s): the suction gas heated by the motor,
    # the end of the (adiabatic) compression, the outlet after the shell's cooling; and the
    # isentropic discharge state from the suction port (2s)
    m_c = g("mdot_c")
    if m_c > 1e-6:
        h2_ad = g("h2_ad")
        pts["cmh"] = (P_s, h2_ad - g("W_shaft") / m_c)
        pts["c2a"] = (P_d, h2_ad)
        pts["cout"] = (P_d, g("h2"))
    s_1 = float(props.state(np.array([P_s]), np.array([g("h_cin")])).s[0])
    pts["c2s"] = (P_d, float(h_Ps(props, np.array([P_d]), s_1)[0]))

    names = list(pts)
    P = np.maximum(np.array([pts[k][0] for k in names]), props.p_min)
    h = np.array([pts[k][1] for k in names])
    st = props.state(P, h)
    sat = props.sat(P)
    dT = np.where(st.x > 1.0, st.T - sat["T_v"], np.where(st.x < 0.0, sat["T_l"] - st.T, 0.0))
    rows = np.round(np.stack([P / 1e5, h / 1e3, st.T - C2K, st.x, dT], axis=1), 5).tolist()
    return dict(zip(names, rows))


def h_Ps(props, P, s):
    """Enthalpy [J/kg] at pressures ``P`` [Pa] and entropy ``s`` [J/(kg K)]: vapor from the
    tables, two-phase by the lever rule, the bubble point below it (no liquid tables in s)."""
    P = np.asarray(P, float)
    s = np.broadcast_to(np.asarray(s, float), P.shape)
    sat = props.sat(P)
    x = (s - sat["s_l"]) / np.maximum(sat["s_v"] - sat["s_l"], 1e-9)
    h = sat["h_l"] + np.clip(x, 0.0, 1.0) * (sat["h_v"] - sat["h_l"])
    vap = x > 1.0
    if vap.any():
        h = np.where(vap, props.h_Ps_vapor(P, s), h)
    return h


@njit(cache=True)
def _h_Ps1(tab, P, s):
    """:func:`h_Ps` at one point (compiled)."""
    i, w = kp.pidx(tab, P)
    s_l, s_v = kp.sat_row(tab, kp.S_L, i, w), kp.sat_row(tab, kp.S_V, i, w)
    x = (s - s_l) / max(s_v - s_l, 1e-9)
    if x > 1.0:
        return kp.h_Ps_vapor(tab, P, s)
    h_l = kp.sat_row(tab, kp.H_L, i, w)
    return h_l + min(max(x, 0.0), 1.0) * (kp.sat_row(tab, kp.H_V, i, w) - h_l)


@njit(cache=True)
def _march(tab, Pk, h0, eta, out):
    """Compression over the pressures ``Pk`` from ``h0``: each step the isentropic rise
    over ``eta`` (eta = 1: the isentrope).  Fills ``out``, returns the end enthalpy."""
    out[0] = h0
    for k in range(Pk.shape[0] - 1):
        s = kp.state(tab, Pk[k], out[k]).s
        out[k + 1] = out[k] + (_h_Ps1(tab, Pk[k + 1], s) - out[k]) / eta
    return out[-1]


@njit(cache=True)
def _polytropic(tab, Pk, h0, h_end, out):
    """The constant small-step efficiency whose march ends at ``h_end``, with its path in
    ``out``.  The rise is close to linear in 1 / eta: a secant in 1 / eta from the
    isentrope's rise converges in a few marches."""
    u0, f0 = 1.0, _march(tab, Pk, h0, 1.0, out) - h_end
    u1 = (h_end - h0) / max(f0 + h_end - h0, 1.0)
    for _ in range(12):
        f1 = _march(tab, Pk, h0, 1.0 / u1, out) - h_end
        if abs(f1) < 0.1 or f1 == f0:
            break
        u0, f0, u1 = u1, f1, min(max(u1 - f1 * (u1 - u0) / (f1 - f0), 0.05), 10.0)
    return 1.0 / u1


def compression_paths(props, ph, n=24) -> dict:
    """The compression on the P-h diagram, from the state points of :func:`state_points`
    (bar, kJ/kg): ``comp``, the suction gas heated by the motor (cmh) compressed to the
    model's adiabatic end state (c2a) along a polytropic path (constant small-step
    efficiency ``eta_p``, fitted to end there); ``isen``, the isentrope from the suction
    port (suc) to the discharge pressure (c2s).  The model computes only the end states;
    the path in between is the standard representation of a real compression."""
    out = {}
    P1, h1 = ph["suc"][0] * 1e5, ph["suc"][1] * 1e3
    P2 = ph["c2s"][0] * 1e5
    if P2 <= P1 * 1.001:
        return out
    hk = np.empty(n + 1)
    Pk = np.geomspace(P1, P2, n + 1)
    _march(props.tab, Pk, h1, 1.0, hk)
    out["isen"] = np.round(np.stack([Pk / 1e5, hk / 1e3], axis=1), 4).tolist()
    if "cmh" in ph:
        Pk = np.geomspace(ph["cmh"][0] * 1e5, P2, n + 1)
        eta = _polytropic(props.tab, Pk, ph["cmh"][1] * 1e3, ph["c2a"][1] * 1e3, hk)
        out["comp"] = np.round(np.stack([Pk / 1e5, hk / 1e3], axis=1), 4).tolist()
        out["eta_p"] = round(float(eta), 4)
    return out


def _closing(props, P, h_l, h_v, n=24):
    """Saturation lines from the tables' top pressure to the critical point: the
    mean of the two lines extrapolated linearly, their width closing as
    (P_c - P)^0.35 (the critical exponent).  Estimate, drawn dashed."""
    Pc = props.P_crit
    if not Pc > P[-1]:
        return None
    k = max(len(P) // 20, 2)
    mid = 0.5 * (h_l + h_v)
    slope = (mid[-1] - mid[-k]) / (P[-1] - P[-k])
    Pz = np.linspace(P[-1], Pc, n)
    w = (h_v[-1] - h_l[-1]) * ((Pc - Pz) / (Pc - P[-1])) ** 0.35
    m = mid[-1] + slope * (Pz - P[-1])
    return Pz, m - 0.5 * w, m + 0.5 * w


def view_temperatures(props, h0, h1, P0, P1, unit="C", n=12) -> tuple[list, float]:
    """Isotherm temperatures [degC] for the view h0..h1 [J/kg] x P0..P1 [Pa]: round values
    of the display unit ``unit`` ("C" or "F"), at the finest of 0.5, 1, 2, 5, 10, 20, 50
    degrees giving at most about ``n`` lines over the view's temperature range (its
    corners and the saturation temperatures at its pressures).  Also returns the step."""
    Pc = np.clip([P0, P0, P1, P1], props.p_min, props.p_max)
    st = props.state(Pc, np.array([h0, h1, h0, h1], float))
    s = props.sat(Pc[::2])
    T = np.concatenate([st.T, s["T_l"], s["T_v"]]) - C2K
    k, o = (1.8, 32.0) if unit == "F" else (1.0, 0.0)
    lo, hi = float(T.min()) * k + o, float(T.max()) * k + o
    for step in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        if (hi - lo) / step <= n:
            break
    vals = np.arange(np.floor(lo / step) * step, hi + step, step)
    return [round((v - o) / k, 6) for v in vals], step


BTU_LB_F = 0.238845897      # 1 kJ/(kg K) in Btu/(lb degF)


def view_entropies(props, h0, h1, P0, P1, unit="C", n=12) -> tuple[list, float]:
    """Isentrope entropies [J/(kg K)] for the view h0..h1 [J/kg] x P0..P1 [Pa]: round values
    in kJ/(kg K) (``unit`` "C") or Btu/(lb degF) ("F"), at the finest step of 1-2-5 giving
    at most about ``n`` lines over the view's entropy range (its corners; no lines lie
    entirely in the liquid).  Also returns the step, in the display unit."""
    Pc = np.clip([P0, P0, P1, P1], props.p_min, props.p_max)
    sv = props.state(Pc, np.array([h0, h1, h0, h1], float)).s
    s_lo = max(float(sv.min()), float(props.sat(Pc[:1])["s_l"][0]))
    k = 1e-3 * (BTU_LB_F if unit == "F" else 1.0)          # J/(kg K) -> display
    lo, hi = s_lo * k, float(sv.max()) * k
    for step in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        if (hi - lo) / step <= n:
            break
    vals = np.arange(np.floor(lo / step) * step, hi + step, step)
    return [round(v / k, 4) for v in vals], step


def chart(props, temps_C=None, P_range=None, n_p=120, entropies=None) -> dict:
    """Background of the diagram: saturated liquid (x = 0) and vapor (x = 1) lines,
    isotherms at ``temps_C`` [degC] (two-phase: linear in temperature across a
    zeotrope's glide), and the critical point, on ``n_p`` pressures spanning the tables
    or the view ``P_range`` (P0, P1) [Pa]; isentropes at ``entropies`` [J/(kg K)], over the
    two-phase and vapor regions.  The tables end below the critical pressure;
    the dome is closed from there by an estimate (``dome_ext``)."""
    lo, hi = props.p_min, props.p_max
    if P_range is not None:
        lo = min(max(lo, P_range[0] / 1.05), hi / 1.2)
        hi = max(min(hi, P_range[1] * 1.05), lo * 1.2)
    P = np.exp(np.linspace(np.log(lo), np.log(hi), n_p))
    s = props.sat(P)
    h_l, h_v, T_l, T_v = s["h_l"], s["h_v"], s["T_l"], s["T_v"]
    bar = lambda a: np.round(np.asarray(a) / 1e5, 5).tolist()
    kj = lambda a: np.round(np.asarray(a) / 1e3, 3).tolist()
    out = dict(fluid=props.fluid, P=bar(P), h_l=kj(h_l), h_v=kj(h_v), T_l=np.round(T_l - C2K, 3).tolist(),
               T_v=np.round(T_v - C2K, 3).tolist(),
               P_max=float(props.p_max) / 1e5, P_min=float(props.p_min) / 1e5,
               T_crit=float(props.T_crit) - C2K, P_crit=float(props.P_crit) / 1e5)
    ext = _closing(props, P, h_l, h_v) if hi >= props.p_max else None
    if ext is not None:
        Pz, hl_z, hv_z = ext
        out["dome_ext"] = dict(P=bar(Pz), h_l=kj(hl_z), h_v=kj(hv_z))
        out["h_crit"] = float(hl_z[-1]) / 1e3
    iso = []
    T_hi = props.t_max - 5.0
    for Tc in (temps_C or []):
        T = float(Tc) + C2K
        if not props.t_min + 2.0 < T < T_hi:
            continue
        # bubble and dew pressures at T (beyond the tables: all liquid / all vapor)
        P_b = float(np.interp(T, T_l, P, left=-np.inf, right=np.inf))
        P_dw = float(np.interp(T, T_v, P, left=-np.inf, right=np.inf))
        pp, hh = [], []
        liq, vap = P > P_b, P < P_dw
        if liq.any():
            Pl = P[liq][::-1]
            pp += list(Pl)
            hh += list(props.h_PT(Pl, np.full(len(Pl), T)))
        if np.isfinite(P_b) and np.isfinite(P_dw):            # across the dome (glide: linear in T)
            Pm = np.concatenate([[P_b], P[(P < P_b) & (P > P_dw)][::-1], [P_dw]])
            sm = props.sat(Pm)
            f = np.clip((T - sm["T_l"]) / np.maximum(sm["T_v"] - sm["T_l"], 1e-9), 0.0, 1.0)
            f[0], f[-1] = 0.0, 1.0
            pp += list(Pm)
            hh += list(sm["h_l"] + f * (sm["h_v"] - sm["h_l"]))
        if vap.any():
            Pv = P[vap][::-1]
            pp += list(Pv)
            hh += list(props.h_PT(Pv, np.full(len(Pv), T)))
        if pp:
            iso.append(dict(T=float(Tc), P=bar(pp), h=kj(hh)))
    out["isotherms"] = iso
    isen = []
    for sv in (entropies or []):
        hh = h_Ps(props, P, sv)
        ok = (sv >= s["s_l"]) & (hh <= s["h_v"] + 0.999 * s["dhmax_v"])
        if ok.sum() >= 2:
            isen.append(dict(s=float(sv), P=bar(P[ok][::-1]), h=kj(hh[ok][::-1])))
    out["isentropes"] = isen
    return out
