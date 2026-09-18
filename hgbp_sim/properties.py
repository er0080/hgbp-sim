"""Fast, vectorized refrigerant property tables for the HGBP simulator.

The dynamic model needs refrigerant properties thousands of times per second and
for many parallel environments.  Calling CoolProp inside the ODE right-hand side
is far too slow, so this module builds interpolation tables once (from CoolProp)
and then answers every query with pure numpy array operations.

Table coordinates
-----------------
* Saturation line: uniform grid in log(P).
* Superheated vapor: (log P, zeta) with
      zeta = (h - h_v(P)) / (h(P, T_max) - h_v(P))      in [0, 1]
* Subcooled liquid: (log P, zeta) with
      zeta = (h_l(P) - h) / (h_l(P) - h(P, T_min))      in [0, 1]
* Two-phase: computed analytically from the saturation tables.

Aligning the grids with the saturation dome means bilinear interpolation never
straddles a phase boundary, which keeps density and its partial derivatives
smooth (the ODE formulation uses drho/dP|h and drho/dh|P).

Tables are cached as compressed ``.npz`` files.  The package ships a prebuilt
R134a table; other CoolProp fluids are built on first use (needs CoolProp).
"""
from __future__ import annotations

import os
import warnings

import numpy as np

# Saturation table rows (all as functions of P on the log-P grid)
SAT_FIELDS = (
    "T_l", "T_v", "h_l", "h_v", "rho_l", "rho_v", "s_l", "s_v", "cp_l", "cp_v",
    "dT_l_dP", "dh_l_dP", "dh_v_dP", "drho_l_dP", "drho_v_dP",
    "dhmax_v", "dhmax_l",
)
# Single-phase region table fields
REG_FIELDS = ("T", "rho", "s", "drho_dP", "drho_dh", "cp")

_SI = {k: i for i, k in enumerate(SAT_FIELDS)}
_RI = {k: i for i, k in enumerate(REG_FIELDS)}

PHASE_LIQUID, PHASE_TWOPHASE, PHASE_VAPOR = 0, 1, 2

_PKG_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_USER_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "hgbp_sim")


class PhState:
    """Container for the thermodynamic state of a batch of (P, h) points.

    All attributes are numpy arrays of the same shape as the inputs.
    """

    __slots__ = (
        "P", "h", "T", "rho", "x", "drho_dP", "drho_dh", "s", "cp",
        "T_sat", "h_l", "h_v", "rho_l", "rho_v", "phase",
    )

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"PhState(P={self.P}, h={self.h}, T={self.T}, rho={self.rho}, "
                f"x={self.x}, phase={self.phase})")


class RefrigerantTables:
    """Tabulated refrigerant properties with vectorized numpy lookups.

    Parameters
    ----------
    fluid : CoolProp fluid name (e.g. "R134a", "R404A", "R410A", "R32", "R290").
    n_p, n_z : grid resolution in log(P) and in the region coordinate zeta.
    p_min, p_max : pressure range of the tables [Pa].  Default p_max is
        0.92 * P_critical.
    t_min, t_max : temperature bounds for the liquid / vapor tables [K].
    cache_dir : where to look for / store the .npz cache (package data dir is
        always checked first).
    rebuild : force a rebuild from CoolProp.
    """

    def __init__(self, fluid: str = "R134a", n_p: int = 200, n_z: int = 160,
                 p_min: float = 0.2e5, p_max: float | None = None,
                 t_min: float | None = None, t_max: float | None = None,
                 cache_dir: str | None = None, rebuild: bool = False,
                 backend: str = "HEOS"):
        self.fluid = fluid
        self.backend = backend
        fname = f"{fluid}_{backend}_{n_p}x{n_z}.npz"
        candidates = [os.path.join(_PKG_DATA_DIR, fname),
                      os.path.join(cache_dir or _USER_CACHE_DIR, fname)]
        loaded = False
        if not rebuild:
            for path in candidates:
                if os.path.exists(path):
                    self._load(path)
                    loaded = True
                    break
        if not loaded:
            self._build(n_p, n_z, p_min, p_max, t_min, t_max)
            out_dir = cache_dir or _USER_CACHE_DIR
            try:
                os.makedirs(out_dir, exist_ok=True)
                self.save(os.path.join(out_dir, fname))
            except OSError as exc:  # pragma: no cover
                warnings.warn(f"could not cache property table: {exc}")
        self._finalize()

    # ------------------------------------------------------------------ build
    def _build(self, n_p, n_z, p_min, p_max, t_min, t_max) -> None:
        try:
            import CoolProp.CoolProp as CP
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"No cached property table for {self.fluid!r}; building one "
                "requires CoolProp (pip install CoolProp)") from exc

        AS = CP.AbstractState(self.backend, self.fluid)
        Tc, Pc = AS.T_critical(), AS.p_critical()
        Tmax_eos, Ttr = AS.Tmax(), AS.Ttriple()
        p_max = p_max if p_max is not None else 0.92 * Pc
        t_max = t_max if t_max is not None else min(Tmax_eos, Tc + 160.0)
        t_min = t_min if t_min is not None else max(Ttr + 5.0, 200.0)
        AS.update(CP.QT_INPUTS, 0.0, t_min + 3.0)
        p_min = max(p_min, AS.p())
        if p_min >= p_max:
            raise ValueError("invalid pressure range for property table")

        self.n_p, self.n_z = int(n_p), int(n_z)
        self.p_min, self.p_max = float(p_min), float(p_max)
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.T_crit, self.P_crit = float(Tc), float(Pc)
        self.M_molar = float(AS.molar_mass())

        lp = np.linspace(np.log(p_min), np.log(p_max), n_p)
        P = np.exp(lp)
        zeta = np.linspace(0.0, 1.0, n_z)
        sat = np.zeros((len(SAT_FIELDS), n_p))
        vap = np.zeros((len(REG_FIELDS), n_p, n_z))
        liq = np.zeros((len(REG_FIELDS), n_p, n_z))

        def region_point(h, p):
            """Return the REG_FIELDS values at (h, p); NaN on failure."""
            try:
                AS.update(CP.HmassP_INPUTS, h, p)
                return (AS.T(), AS.rhomass(), AS.smass(),
                        AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass),
                        AS.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP),
                        AS.cpmass())
            except Exception:  # noqa: BLE001 - CoolProp raises generic errors
                return (np.nan,) * len(REG_FIELDS)

        for i, p in enumerate(P):
            AS.update(CP.PQ_INPUTS, p, 0.0)
            sat[_SI["T_l"], i] = AS.T()
            sat[_SI["h_l"], i] = AS.hmass()
            sat[_SI["rho_l"], i] = AS.rhomass()
            sat[_SI["s_l"], i] = AS.smass()
            sat[_SI["cp_l"], i] = AS.cpmass() if np.isfinite(AS.cpmass()) else np.nan
            AS.update(CP.PQ_INPUTS, p, 1.0)
            sat[_SI["T_v"], i] = AS.T()
            sat[_SI["h_v"], i] = AS.hmass()
            sat[_SI["rho_v"], i] = AS.rhomass()
            sat[_SI["s_v"], i] = AS.smass()
            sat[_SI["cp_v"], i] = AS.cpmass() if np.isfinite(AS.cpmass()) else np.nan
            AS.update(CP.PT_INPUTS, p, t_max)
            h_hi = AS.hmass()
            AS.update(CP.PT_INPUTS, p, t_min)
            h_lo = AS.hmass()
            dh_v = h_hi - sat[_SI["h_v"], i]
            dh_l = sat[_SI["h_l"], i] - h_lo
            sat[_SI["dhmax_v"], i] = dh_v
            sat[_SI["dhmax_l"], i] = dh_l
            for j, z in enumerate(zeta):
                zz = max(z, 2e-4)  # stay strictly single-phase at the dome
                vap[:, i, j] = region_point(sat[_SI["h_v"], i] + zz * dh_v, p)
                liq[:, i, j] = region_point(sat[_SI["h_l"], i] - zz * dh_l, p)

        # Fill failed points by nearest valid neighbour along zeta, then along P
        for tab in (vap, liq):
            for k in range(tab.shape[0]):
                a = tab[k]
                bad = ~np.isfinite(a)
                if bad.any():
                    for i in range(n_p):
                        row = a[i]
                        m = np.isfinite(row)
                        if m.any() and not m.all():
                            row[~m] = np.interp(zeta[~m], zeta[m], row[m])
                        elif not m.any():
                            row[:] = np.nan
                    for j in range(n_z):
                        col = a[:, j]
                        m = np.isfinite(col)
                        if m.any() and not m.all():
                            col[~m] = np.interp(lp[~m], lp[m], col[m])
        # Saturation derivatives d/dP = (d/dlogP)/P
        for src, dst in (("T_l", "dT_l_dP"), ("h_l", "dh_l_dP"), ("h_v", "dh_v_dP"),
                         ("rho_l", "drho_l_dP"), ("rho_v", "drho_v_dP")):
            sat[_SI[dst]] = np.gradient(sat[_SI[src]], lp) / P
        for k in ("cp_l", "cp_v"):
            row = sat[_SI[k]]
            m = np.isfinite(row)
            if not m.all():
                row[~m] = np.interp(lp[~m], lp[m], row[m])

        self._lp = lp
        self._P = P
        self._zeta = zeta
        self._sat = sat
        self._vap = vap
        self._liq = liq

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        np.savez_compressed(
            path, lp=self._lp, zeta=self._zeta, sat=self._sat, vap=self._vap,
            liq=self._liq,
            meta=np.array([self.p_min, self.p_max, self.t_min, self.t_max,
                           self.T_crit, self.P_crit, self.M_molar]),
            fluid=np.array(self.fluid), backend=np.array(self.backend))

    def _load(self, path: str) -> None:
        d = np.load(path)
        self._lp = d["lp"]
        self._P = np.exp(self._lp)
        self._zeta = d["zeta"]
        self._sat = d["sat"]
        self._vap = d["vap"]
        self._liq = d["liq"]
        (self.p_min, self.p_max, self.t_min, self.t_max,
         self.T_crit, self.P_crit, self.M_molar) = (float(v) for v in d["meta"])
        self.n_p = len(self._lp)
        self.n_z = len(self._zeta)
        self.path = path

    def _finalize(self) -> None:
        self._lp0 = float(self._lp[0])
        self._inv_dlp = 1.0 / float(self._lp[1] - self._lp[0])
        self._nz1 = self.n_z - 1
        self.T_sat_min = float(self._sat[_SI["T_l"], 0])
        self.T_sat_max = float(self._sat[_SI["T_l"], -1])
        # flattened copies for fast gathers: index = i * n_z + j
        self._vap_flat = np.ascontiguousarray(self._vap.reshape(self._vap.shape[0], -1))
        self._liq_flat = np.ascontiguousarray(self._liq.reshape(self._liq.shape[0], -1))
        self._vap_s_rows = np.ascontiguousarray(self._vap[_RI["s"]])
        self._vap_T_rows = np.ascontiguousarray(self._vap[_RI["T"]])
        self._liq_T_rows = np.ascontiguousarray(self._liq[_RI["T"]])
        self._vap_rho_rows = np.ascontiguousarray(self._vap[_RI["rho"]])
        self._liq_rho_rows = np.ascontiguousarray(self._liq[_RI["rho"]])
        self._vap_T_flat = self._vap_T_rows.reshape(-1)
        self._vap_rho_flat = self._vap_rho_rows.reshape(-1)
        self._sat_T = np.ascontiguousarray(self._sat[_SI["T_l"]])

    # ------------------------------------------------------------- primitives
    def _pidx(self, P):
        lp = np.log(np.minimum(np.maximum(P, self.p_min), self.p_max))
        f = (lp - self._lp0) * self._inv_dlp
        i = np.minimum(np.maximum(f.astype(np.intp), 0), self.n_p - 2)
        w = np.minimum(np.maximum(f - i, 0.0), 1.0)
        return i, w

    def _sat_at(self, i, w):
        tab = self._sat
        return tab.take(i, axis=1) * (1.0 - w) + tab.take(i + 1, axis=1) * w

    def _interp2(self, flat, i, w, zeta):
        """Bilinear interpolation of all fields of a flattened region table."""
        f = zeta * self._nz1
        j = np.minimum(np.maximum(f.astype(np.intp), 0), self.n_z - 2)
        wz = np.minimum(np.maximum(f - j, 0.0), 1.0)
        k0 = i * self.n_z + j
        k1 = k0 + self.n_z
        v00, v01 = flat.take(k0, axis=1), flat.take(k0 + 1, axis=1)
        v10, v11 = flat.take(k1, axis=1), flat.take(k1 + 1, axis=1)
        return (v00 * (1.0 - wz) + v01 * wz) * (1.0 - w) + (v10 * (1.0 - wz) + v11 * wz) * w

    def _invert_row(self, rows_tab, i, w, target, decreasing=False):
        """Find zeta such that rows_tab(P, zeta) == target (per element)."""
        rows = rows_tab.take(i, axis=0) * (1.0 - w)[:, None] + rows_tab.take(i + 1, axis=0) * w[:, None]
        t = target[:, None]
        if decreasing:
            cnt = (rows > t).sum(axis=1)
        else:
            cnt = (rows < t).sum(axis=1)
        j = np.minimum(np.maximum(cnt - 1, 0), self.n_z - 2)
        ar = np.arange(len(j))
        r0, r1 = rows[ar, j], rows[ar, j + 1]
        denom = np.where(np.abs(r1 - r0) > 1e-300, r1 - r0, 1e-300)
        frac = np.minimum(np.maximum((target - r0) / denom, 0.0), 1.0)
        return np.minimum(np.maximum((j + frac) / self._nz1, 0.0), 1.0)

    # ---------------------------------------------------------------- queries
    def sat(self, P):
        """Saturation properties at pressure P (arrays).  Returns a dict."""
        P = np.asarray(P, dtype=float)
        i, w = self._pidx(P)
        v = self._sat_at(i, w)
        return {k: v[_SI[k]] for k in SAT_FIELDS}

    def T_sat(self, P):
        P = np.asarray(P, dtype=float)
        i, w = self._pidx(P)
        r = self._sat_T
        return r.take(i) * (1.0 - w) + r.take(i + 1) * w

    def P_sat(self, T):
        """Saturation pressure for temperature T [K] (bubble line)."""
        T = np.asarray(T, dtype=float)
        return np.interp(T, self._sat[_SI["T_l"]], self._P)

    def state(self, P, h, need_s: bool = False) -> PhState:
        """Full state from pressure [Pa] and mass enthalpy [J/kg]."""
        P = np.asarray(P, dtype=float)
        h = np.asarray(h, dtype=float)
        i, w = self._pidx(P)
        sat = self._sat_at(i, w)
        h_l, h_v = sat[_SI["h_l"]], sat[_SI["h_v"]]
        rho_l, rho_v = sat[_SI["rho_l"]], sat[_SI["rho_v"]]
        T_l, T_v = sat[_SI["T_l"]], sat[_SI["T_v"]]
        hfg = h_v - h_l
        x = (h - h_l) / hfg

        is_v = x > 1.0
        is_l = x < 0.0
        any_v, any_l = bool(is_v.any()), bool(is_l.any())
        # --- single-phase branches (evaluated only if some element needs them)
        if any_v:
            zv = np.minimum(np.maximum((h - h_v) / sat[_SI["dhmax_v"]], 0.0), 1.0)
            V = self._interp2(self._vap_flat, i, w, zv)
        if any_l:
            zl = np.minimum(np.maximum((h_l - h) / sat[_SI["dhmax_l"]], 0.0), 1.0)
            L = self._interp2(self._liq_flat, i, w, zl)

        # --- two-phase branch (analytic from saturation line)
        xc = np.minimum(np.maximum(x, 0.0), 1.0)
        v_l, v_v = 1.0 / rho_l, 1.0 / rho_v
        v2 = v_l + xc * (v_v - v_l)
        rho2 = 1.0 / v2
        T2 = T_l + xc * (T_v - T_l)
        dv_dh = (v_v - v_l) / hfg
        dvl_dP = -v_l * v_l * sat[_SI["drho_l_dP"]]
        dvv_dP = -v_v * v_v * sat[_SI["drho_v_dP"]]
        dx_dP = -(sat[_SI["dh_l_dP"]] + xc * (sat[_SI["dh_v_dP"]] - sat[_SI["dh_l_dP"]])) / hfg
        dv_dP = dvl_dP + xc * (dvv_dP - dvl_dP) + (v_v - v_l) * dx_dP
        drdP2 = -rho2 * rho2 * dv_dP
        drdh2 = -rho2 * rho2 * dv_dh
        cp2 = (sat[_SI["cp_l"]] + sat[_SI["cp_v"]]) * 0.5

        st = PhState()
        st.P, st.h, st.x = P, h, x
        T, rho, drdP, drdh, cpv = T2, rho2, drdP2, drdh2, cp2
        s2 = None
        if need_s:
            s2 = sat[_SI["s_l"]] + xc * (sat[_SI["s_v"]] - sat[_SI["s_l"]])
        if any_l:
            T = np.where(is_l, L[_RI["T"]], T)
            rho = np.where(is_l, L[_RI["rho"]], rho)
            drdP = np.where(is_l, L[_RI["drho_dP"]], drdP)
            drdh = np.where(is_l, L[_RI["drho_dh"]], drdh)
            cpv = np.where(is_l, L[_RI["cp"]], cpv)
            if need_s:
                s2 = np.where(is_l, L[_RI["s"]], s2)
        if any_v:
            T = np.where(is_v, V[_RI["T"]], T)
            rho = np.where(is_v, V[_RI["rho"]], rho)
            drdP = np.where(is_v, V[_RI["drho_dP"]], drdP)
            drdh = np.where(is_v, V[_RI["drho_dh"]], drdh)
            cpv = np.where(is_v, V[_RI["cp"]], cpv)
            if need_s:
                s2 = np.where(is_v, V[_RI["s"]], s2)
        st.T, st.rho, st.drho_dP, st.drho_dh, st.cp, st.s = T, rho, drdP, drdh, cpv, s2
        st.T_sat, st.h_l, st.h_v, st.rho_l, st.rho_v = T_l, h_l, h_v, rho_l, rho_v
        st.phase = np.where(is_v, PHASE_VAPOR, np.where(is_l, PHASE_LIQUID, PHASE_TWOPHASE))
        return st

    def T_Ph(self, P, h):
        return self.state(P, h).T

    def rho_Ph(self, P, h):
        return self.state(P, h).rho

    def T_vapor(self, P, h):
        """Temperature assuming superheated vapor (clamped to the dome)."""
        P = np.asarray(P, dtype=float)
        h = np.asarray(h, dtype=float)
        i, w = self._pidx(P)
        sat = self._sat_at(i, w)
        zv = np.minimum(np.maximum((h - sat[_SI["h_v"]]) / sat[_SI["dhmax_v"]], 0.0), 1.0)
        f = zv * self._nz1
        j = np.minimum(np.maximum(f.astype(np.intp), 0), self.n_z - 2)
        wz = f - j
        rows = self._vap_T_rows
        k0 = i * self.n_z + j
        k1 = k0 + self.n_z
        flat = rows.reshape(-1)
        return ((flat.take(k0) * (1.0 - wz) + flat.take(k0 + 1) * wz) * (1.0 - w)
                + (flat.take(k1) * (1.0 - wz) + flat.take(k1 + 1) * wz) * w)

    def vapor_props(self, P, h):
        """(T, rho) of superheated vapor at (P, h), clamped to the dew line."""
        P = np.asarray(P, dtype=float)
        h = np.asarray(h, dtype=float)
        i, w = self._pidx(P)
        sat = self._sat_at(i, w)
        zv = np.minimum(np.maximum((h - sat[_SI["h_v"]]) / sat[_SI["dhmax_v"]], 0.0), 1.0)
        f = zv * self._nz1
        j = np.minimum(np.maximum(f.astype(np.intp), 0), self.n_z - 2)
        wz = f - j
        k0 = i * self.n_z + j
        k1 = k0 + self.n_z
        out = []
        for flat in (self._vap_T_flat, self._vap_rho_flat):
            out.append((flat.take(k0) * (1.0 - wz) + flat.take(k0 + 1) * wz) * (1.0 - w)
                       + (flat.take(k1) * (1.0 - wz) + flat.take(k1 + 1) * wz) * w)
        return out[0], out[1]

    def h_from_P_rho(self, P, rho):
        """Enthalpy at (P, rho): two-phase lever rule inside the dome, table
        inversion in the vapor / liquid regions."""
        P = np.asarray(P, dtype=float)
        rho = np.asarray(rho, dtype=float)
        i, w = self._pidx(P)
        sat = self._sat_at(i, w)
        rho_l, rho_v = sat[_SI["rho_l"]], sat[_SI["rho_v"]]
        h_l, h_v = sat[_SI["h_l"]], sat[_SI["h_v"]]
        v = 1.0 / np.maximum(rho, 1e-6)
        x = (v - 1.0 / rho_l) / (1.0 / rho_v - 1.0 / rho_l)
        h2 = h_l + np.minimum(np.maximum(x, 0.0), 1.0) * (h_v - h_l)
        zv = self._invert_row(self._vap_rho_rows, i, w, rho, decreasing=True)
        hv = h_v + zv * sat[_SI["dhmax_v"]]
        zl = self._invert_row(self._liq_rho_rows, i, w, rho)
        hl = h_l - zl * sat[_SI["dhmax_l"]]
        return np.where(rho < rho_v, hv, np.where(rho > rho_l, hl, h2))

    def h_Ps_vapor(self, P, s):
        """Enthalpy of superheated vapor at (P, s); clamps to saturated vapor
        if s is below the dew line (wet isentropic compression)."""
        P = np.asarray(P, dtype=float)
        s = np.asarray(s, dtype=float)
        i, w = self._pidx(P)
        sat = self._sat_at(i, w)
        z = self._invert_row(self._vap_s_rows, i, w, s)
        return sat[_SI["h_v"]] + z * sat[_SI["dhmax_v"]]

    def h_PT(self, P, T):
        """Enthalpy at (P, T) for single-phase states.  Inside the dome
        (T == T_sat) the saturated-vapor value is returned."""
        P = np.asarray(P, dtype=float)
        T = np.asarray(T, dtype=float)
        i, w = self._pidx(P)
        sat = self._sat_at(i, w)
        zv = self._invert_row(self._vap_T_rows, i, w, T)
        zl = self._invert_row(self._liq_T_rows, i, w, T, decreasing=True)
        hv = sat[_SI["h_v"]] + zv * sat[_SI["dhmax_v"]]
        hl = sat[_SI["h_l"]] - zl * sat[_SI["dhmax_l"]]
        return np.where(T >= sat[_SI["T_v"]], hv, np.where(T <= sat[_SI["T_l"]], hl, hv))

    def h_sat_vapor(self, P):
        return self.sat(P)["h_v"]

    def h_sat_liquid(self, P):
        return self.sat(P)["h_l"]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"RefrigerantTables({self.fluid!r}, P=[{self.p_min/1e5:.2f}, "
                f"{self.p_max/1e5:.1f}] bar, T=[{self.t_min:.0f}, {self.t_max:.0f}] K, "
                f"grid {self.n_p}x{self.n_z})")


_TABLE_CACHE: dict[tuple, RefrigerantTables] = {}


def get_tables(fluid: str = "R134a", **kw) -> RefrigerantTables:
    """Return a process-wide shared table instance for ``fluid``."""
    key = (fluid, tuple(sorted(kw.items())))
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = RefrigerantTables(fluid, **kw)
    return _TABLE_CACHE[key]
