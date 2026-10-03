"""Pressure-enthalpy (P-h) diagram data for the live stand.

* :func:`state_points`: the stand's state points along every flow path, from the
  model outputs of one step (pressures along each path from the model's pressure
  chain, enthalpies of the cells and volumes the model carries).
* :func:`chart`: the refrigerant's saturation lines and isotherms from the property
  tables, as the diagram's background, for the whole dome or for one view (a zoomed
  view gets its own pressure grid and isotherms at a spacing to suit it,
  :func:`view_temperatures`).

Pressures in bar (absolute), enthalpies in kJ/kg, temperatures in degC.
"""
from __future__ import annotations

import numpy as np

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
    h_q, h_g, h_c, dP_qc = vec("h_q"), vec("h_g"), vec("h_c"), vec("dP_qc")
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
        pts[f"c{j}"] = (P_c3 + (P_c4 - P_c3) * (j + 0.5) / n, float(h_c[j]))
    pts["rec"] = (P_i, h_L)
    pts["v3i"] = (P_i - g("dP_liq"), h_co)
    pts["v3o"] = (P_q3 + g("dP_q"), h_co)
    pts["qin"] = (P_q3, h_3f)
    for j in range(n):
        pts[f"q{j}"] = (P_s + float(dP_qc[j]), float(h_q[j]))
    pts["qout"] = (P_q4, float(h_q[-1]))
    pts["v2o"] = (P_g1 + g("dP_bp"), h_2f)
    pts["gin"] = (P_g1, h_2f)
    for j in range(n - 1, -1, -1):
        pts[f"g{j}"] = (P_g2 + (P_g1 - P_g2) * (j + 0.5) / n, float(h_g[j]))
    pts["gout"] = (P_g2, float(h_g[0]))
    pts["tee"] = (P_tee, h_tee)

    names = list(pts)
    P = np.maximum(np.array([pts[k][0] for k in names]), props.p_min)
    h = np.array([pts[k][1] for k in names])
    st = props.state(P, h)
    sat = props.sat(P)
    dT = np.where(st.x > 1.0, st.T - sat["T_v"], np.where(st.x < 0.0, sat["T_l"] - st.T, 0.0))
    rows = np.round(np.stack([P / 1e5, h / 1e3, st.T - C2K, st.x, dT], axis=1), 5).tolist()
    return dict(zip(names, rows))


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


def chart(props, temps_C=None, P_range=None, n_p=120) -> dict:
    """Background of the diagram: saturated liquid (x = 0) and vapor (x = 1) lines,
    isotherms at ``temps_C`` [degC] (two-phase: linear in temperature across a
    zeotrope's glide), and the critical point, on ``n_p`` pressures spanning the tables
    or the view ``P_range`` (P0, P1) [Pa].  The tables end below the critical pressure;
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
    return out
