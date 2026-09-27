"""Compiled (numba) refrigerant property lookups on the tables of
:class:`hgbp_sim.properties.RefrigerantTables`.

Scalar functions of one (P, h) point: the model kernel calls them directly, and
the table class wraps them in loops over arrays.  The tables travel as a
:class:`Tab` named tuple of arrays (see ``RefrigerantTables.tab``).  Table
layout and interpolation are described in :mod:`hgbp_sim.properties`.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
from numba import njit

# saturation rows (order of properties.SAT_FIELDS)
T_L, T_V, H_L, H_V, RHO_L, RHO_V, S_L, S_V, CP_L, CP_V = range(10)
DT_L_DP, DH_L_DP, DH_V_DP, DRHO_L_DP, DRHO_V_DP, DHMAX_V, DHMAX_L = range(10, 17)
# single-phase region rows (order of properties.REG_FIELDS)
R_T, R_RHO, R_S, R_DRHO_DP, R_DRHO_DH, R_CP = range(6)
# transport rows (order of properties.TR_FIELDS)
MU_L, MU_V, SIGMA = range(3)

Tab = namedtuple("Tab", ["lp0", "inv_dlp", "p_min", "p_max", "n_p", "n_z", "sat", "tr", "vap", "liq", "P"])
"""sat (17, n_p), tr (3, n_p): saturation rows on the log-P grid ``P``; vap, liq
(6, n_p * n_z): region tables flattened as i * n_z + j."""

# state at (P, h): phase-dependent T, rho, density derivatives, s, cp; saturation values
PhS = namedtuple("PhS", ["T", "rho", "x", "drho_dP", "drho_dh", "s", "cp",
                         "T_sat", "h_l", "h_v", "rho_l", "rho_v", "cp_l"])
Sat = namedtuple("Sat", ["T_l", "T_v", "h_l", "h_v", "rho_l", "rho_v", "cp_v", "dhmax_v"])


@njit(cache=True, inline="always")
def pidx(tab, P):
    """Grid interval and weight of pressure ``P`` (clamped to the table)."""
    lp = np.log(min(max(P, tab.p_min), tab.p_max))
    f = (lp - tab.lp0) * tab.inv_dlp
    i = min(max(int(f), 0), tab.n_p - 2)
    w = min(max(f - i, 0.0), 1.0)
    return i, w


@njit(cache=True, inline="always")
def sat_row(tab, k, i, w):
    return tab.sat[k, i] * (1.0 - w) + tab.sat[k, i + 1] * w


@njit(cache=True)
def sat(tab, P):
    i, w = pidx(tab, P)
    return Sat(sat_row(tab, T_L, i, w), sat_row(tab, T_V, i, w), sat_row(tab, H_L, i, w),
               sat_row(tab, H_V, i, w), sat_row(tab, RHO_L, i, w), sat_row(tab, RHO_V, i, w),
               sat_row(tab, CP_V, i, w), sat_row(tab, DHMAX_V, i, w))


@njit(cache=True)
def transport(tab, P):
    """Saturated liquid / vapor viscosity and surface tension at ``P``."""
    i, w = pidx(tab, P)
    tr = tab.tr
    return (tr[MU_L, i] * (1.0 - w) + tr[MU_L, i + 1] * w, tr[MU_V, i] * (1.0 - w) + tr[MU_V, i + 1] * w,
            tr[SIGMA, i] * (1.0 - w) + tr[SIGMA, i + 1] * w)


@njit(cache=True, inline="always")
def _bilinear(flat, k, n_z, i, w, j, wz):
    k0 = i * n_z + j
    k1 = k0 + n_z
    return ((flat[k, k0] * (1.0 - wz) + flat[k, k0 + 1] * wz) * (1.0 - w)
            + (flat[k, k1] * (1.0 - wz) + flat[k, k1 + 1] * wz) * w)


@njit(cache=True, inline="always")
def _zidx(tab, zeta):
    f = zeta * (tab.n_z - 1)
    j = min(max(int(f), 0), tab.n_z - 2)
    return j, min(max(f - j, 0.0), 1.0)


@njit(cache=True)
def state(tab, P, h):
    """Full state at pressure [Pa] and mass enthalpy [J/kg]: single-phase from
    the region tables, two-phase analytically from the saturation line."""
    i, w = pidx(tab, P)
    h_l, h_v = sat_row(tab, H_L, i, w), sat_row(tab, H_V, i, w)
    rho_l, rho_v = sat_row(tab, RHO_L, i, w), sat_row(tab, RHO_V, i, w)
    T_l, T_v = sat_row(tab, T_L, i, w), sat_row(tab, T_V, i, w)
    cp_l = sat_row(tab, CP_L, i, w)
    hfg = h_v - h_l
    x = (h - h_l) / hfg
    if x > 1.0:
        j, wz = _zidx(tab, min(max((h - h_v) / sat_row(tab, DHMAX_V, i, w), 0.0), 1.0))
        f, nz = tab.vap, tab.n_z
        return PhS(_bilinear(f, R_T, nz, i, w, j, wz), _bilinear(f, R_RHO, nz, i, w, j, wz), x,
                   _bilinear(f, R_DRHO_DP, nz, i, w, j, wz), _bilinear(f, R_DRHO_DH, nz, i, w, j, wz),
                   _bilinear(f, R_S, nz, i, w, j, wz), _bilinear(f, R_CP, nz, i, w, j, wz),
                   T_l, h_l, h_v, rho_l, rho_v, cp_l)
    if x < 0.0:
        j, wz = _zidx(tab, min(max((h_l - h) / sat_row(tab, DHMAX_L, i, w), 0.0), 1.0))
        f, nz = tab.liq, tab.n_z
        return PhS(_bilinear(f, R_T, nz, i, w, j, wz), _bilinear(f, R_RHO, nz, i, w, j, wz), x,
                   _bilinear(f, R_DRHO_DP, nz, i, w, j, wz), _bilinear(f, R_DRHO_DH, nz, i, w, j, wz),
                   _bilinear(f, R_S, nz, i, w, j, wz), _bilinear(f, R_CP, nz, i, w, j, wz),
                   T_l, h_l, h_v, rho_l, rho_v, cp_l)
    # two-phase: homogeneous mixture on the saturation line
    v_l, v_v = 1.0 / rho_l, 1.0 / rho_v
    rho2 = 1.0 / (v_l + x * (v_v - v_l))
    dv_dh = (v_v - v_l) / hfg
    dvl_dP = -v_l * v_l * sat_row(tab, DRHO_L_DP, i, w)
    dvv_dP = -v_v * v_v * sat_row(tab, DRHO_V_DP, i, w)
    dhl_dP = sat_row(tab, DH_L_DP, i, w)
    dx_dP = -(dhl_dP + x * (sat_row(tab, DH_V_DP, i, w) - dhl_dP)) / hfg
    dv_dP = dvl_dP + x * (dvv_dP - dvl_dP) + (v_v - v_l) * dx_dP
    s_l = sat_row(tab, S_L, i, w)
    return PhS(T_l + x * (T_v - T_l), rho2, x, -rho2 * rho2 * dv_dP, -rho2 * rho2 * dv_dh,
               s_l + x * (sat_row(tab, S_V, i, w) - s_l), (cp_l + sat_row(tab, CP_V, i, w)) * 0.5,
               T_l, h_l, h_v, rho_l, rho_v, cp_l)


@njit(cache=True)
def vapor_props(tab, P, h):
    """(T, rho) of superheated vapor at (P, h), clamped to the dew line."""
    i, w = pidx(tab, P)
    zv = min(max((h - sat_row(tab, H_V, i, w)) / sat_row(tab, DHMAX_V, i, w), 0.0), 1.0)
    f = zv * (tab.n_z - 1)
    j = min(max(int(f), 0), tab.n_z - 2)
    wz = f - j
    return (_bilinear(tab.vap, R_T, tab.n_z, i, w, j, wz), _bilinear(tab.vap, R_RHO, tab.n_z, i, w, j, wz))


@njit(cache=True)
def T_vapor(tab, P, h):
    """Temperature assuming superheated vapor (clamped to the dome)."""
    return vapor_props(tab, P, h)[0]


@njit(cache=True)
def _invert_row(flat, k, n_z, i, w, target, decreasing):
    """zeta at which region row ``k`` (interpolated to the pressure) reaches ``target``."""
    cnt = 0
    base0, base1 = i * n_z, (i + 1) * n_z
    for j in range(n_z):
        r = flat[k, base0 + j] * (1.0 - w) + flat[k, base1 + j] * w
        if (r > target) if decreasing else (r < target):
            cnt += 1
    j = min(max(cnt - 1, 0), n_z - 2)
    r0 = flat[k, base0 + j] * (1.0 - w) + flat[k, base1 + j] * w
    r1 = flat[k, base0 + j + 1] * (1.0 - w) + flat[k, base1 + j + 1] * w
    d = r1 - r0
    denom = d if abs(d) > 1e-300 else 1e-300
    frac = min(max((target - r0) / denom, 0.0), 1.0)
    return min(max((j + frac) / (n_z - 1), 0.0), 1.0)


@njit(cache=True)
def h_from_P_rho(tab, P, rho):
    """Enthalpy at (P, rho): lever rule inside the dome, table inversion outside."""
    i, w = pidx(tab, P)
    rho_l, rho_v = sat_row(tab, RHO_L, i, w), sat_row(tab, RHO_V, i, w)
    h_l, h_v = sat_row(tab, H_L, i, w), sat_row(tab, H_V, i, w)
    if rho < rho_v:
        return h_v + _invert_row(tab.vap, R_RHO, tab.n_z, i, w, rho, True) * sat_row(tab, DHMAX_V, i, w)
    if rho > rho_l:
        return h_l - _invert_row(tab.liq, R_RHO, tab.n_z, i, w, rho, False) * sat_row(tab, DHMAX_L, i, w)
    x = (1.0 / max(rho, 1e-6) - 1.0 / rho_l) / (1.0 / rho_v - 1.0 / rho_l)
    return h_l + min(max(x, 0.0), 1.0) * (h_v - h_l)


@njit(cache=True)
def h_Ps_vapor(tab, P, s):
    """Enthalpy of superheated vapor at (P, s); saturated vapor below the dew line."""
    i, w = pidx(tab, P)
    return sat_row(tab, H_V, i, w) + _invert_row(tab.vap, R_S, tab.n_z, i, w, s, False) \
        * sat_row(tab, DHMAX_V, i, w)


@njit(cache=True)
def h_PT(tab, P, T):
    """Enthalpy at (P, T) of single-phase states (saturated vapor on the dome)."""
    i, w = pidx(tab, P)
    if T <= sat_row(tab, T_L, i, w) and T < sat_row(tab, T_V, i, w):
        return sat_row(tab, H_L, i, w) - _invert_row(tab.liq, R_T, tab.n_z, i, w, T, True) \
            * sat_row(tab, DHMAX_L, i, w)
    return sat_row(tab, H_V, i, w) + _invert_row(tab.vap, R_T, tab.n_z, i, w, T, False) \
        * sat_row(tab, DHMAX_V, i, w)


# ------------------------------------------------------------ array versions
@njit(cache=True)
def state_array(tab, P, h, out):
    """``out[:, k]`` = the PhS fields of point k (P, h flat arrays)."""
    for k in range(P.shape[0]):
        s = state(tab, P[k], h[k])
        for f in range(13):
            out[f, k] = s[f]


@njit(cache=True)
def sat_array(tab, P, out):
    """``out[:, k]`` = all saturation rows at P[k]."""
    for k in range(P.shape[0]):
        i, w = pidx(tab, P[k])
        for f in range(tab.sat.shape[0]):
            out[f, k] = sat_row(tab, f, i, w)


@njit(cache=True)
def sat_row_array(tab, k, P, out):
    """``out[j]`` = saturation row ``k`` at P[j]."""
    for j in range(P.shape[0]):
        i, w = pidx(tab, P[j])
        out[j] = sat_row(tab, k, i, w)


@njit(cache=True)
def transport_array(tab, P, out):
    for k in range(P.shape[0]):
        out[0, k], out[1, k], out[2, k] = transport(tab, P[k])


@njit(cache=True)
def vapor_array(tab, P, h, out):
    for k in range(P.shape[0]):
        out[0, k], out[1, k] = vapor_props(tab, P[k], h[k])


@njit(cache=True)
def h_from_P_rho_array(tab, P, rho, out):
    for k in range(P.shape[0]):
        out[k] = h_from_P_rho(tab, P[k], rho[k])


@njit(cache=True)
def h_Ps_vapor_array(tab, P, s, out):
    for k in range(P.shape[0]):
        out[k] = h_Ps_vapor(tab, P[k], s[k])


@njit(cache=True)
def h_PT_array(tab, P, T, out):
    for k in range(P.shape[0]):
        out[k] = h_PT(tab, P[k], T[k])
