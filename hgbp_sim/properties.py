"""Fast refrigerant property tables for the HGBP simulator.

The dynamic model needs refrigerant properties thousands of times per second and
for many parallel environments.  Calling CoolProp inside the ODE right-hand side
is far too slow, so this module builds interpolation tables once (from CoolProp)
and then answers every query by interpolation in compiled code
(:mod:`hgbp_sim.kernel.props`; the model kernel calls the same functions).

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

from .kernel import props as kp

# Saturation table rows (all as functions of P on the log-P grid)
SAT_FIELDS = (
    "T_l", "T_v", "h_l", "h_v", "rho_l", "rho_v", "s_l", "s_v", "cp_l", "cp_v",
    "dT_l_dP", "dh_l_dP", "dh_v_dP", "drho_l_dP", "drho_v_dP",
    "dhmax_v", "dhmax_l",
)
# Saturation transport properties (functions of P on the same grid): liquid and
# vapor viscosity [Pa s], surface tension [N/m]; used by the pressure drop correlations
TR_FIELDS = ("mu_l", "mu_v", "sigma")
# Single-phase region table fields
REG_FIELDS = ("T", "rho", "s", "drho_dP", "drho_dh", "cp")

_SI = {k: i for i, k in enumerate(SAT_FIELDS)}
_TI = {k: i for i, k in enumerate(TR_FIELDS)}
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
        "T_sat", "h_l", "h_v", "rho_l", "rho_v", "cp_l", "phase",
    )

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"PhState(P={self.P}, h={self.h}, T={self.T}, rho={self.rho}, "
                f"x={self.x}, phase={self.phase})")


class RefrigerantTables:
    """Tabulated refrigerant properties with array lookups.

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
        self._tr = self._transport_rows(P)

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        np.savez_compressed(
            path, lp=self._lp, zeta=self._zeta, sat=self._sat, vap=self._vap,
            liq=self._liq, tr=self._tr,
            meta=np.array([self.p_min, self.p_max, self.t_min, self.t_max,
                           self.T_crit, self.P_crit, self.M_molar]),
            fluid=np.array(self.fluid), backend=np.array(self.backend))

    def _transport_rows(self, P, fluid: str | None = None):
        """Viscosities and surface tension at saturation (CoolProp)."""
        try:
            import CoolProp.CoolProp as CP
        except ImportError as exc:  # pragma: no cover
            raise ImportError("this property table has no transport properties; adding them "
                              "requires CoolProp (pip install CoolProp)") from exc
        AS = CP.AbstractState(self.backend, fluid or self.fluid)
        tr = np.zeros((len(TR_FIELDS), len(P)))
        for i, p in enumerate(P):
            AS.update(CP.PQ_INPUTS, float(p), 0.0)
            tr[_TI["mu_l"], i], tr[_TI["sigma"], i] = AS.viscosity(), AS.surface_tension()
            AS.update(CP.PQ_INPUTS, float(p), 1.0)
            tr[_TI["mu_v"], i] = AS.viscosity()
        return tr

    def _load(self, path: str) -> None:
        d = np.load(path)
        self._lp = d["lp"]
        self._P = np.exp(self._lp)
        self._zeta = d["zeta"]
        self._sat = d["sat"]
        self._vap = d["vap"]
        self._liq = d["liq"]
        # tables saved before the transport rows were added: compute them now
        self._tr = d["tr"] if "tr" in d.files else self._transport_rows(np.exp(d["lp"]), fluid=str(d["fluid"]))
        (self.p_min, self.p_max, self.t_min, self.t_max,
         self.T_crit, self.P_crit, self.M_molar) = (float(v) for v in d["meta"])
        self.n_p = len(self._lp)
        self.n_z = len(self._zeta)
        self.path = path

    def _finalize(self) -> None:
        self._lp0 = float(self._lp[0])
        self._inv_dlp = 1.0 / float(self._lp[1] - self._lp[0])
        self.T_sat_min = float(self._sat[_SI["T_l"], 0])
        self.T_sat_max = float(self._sat[_SI["T_l"], -1])
        self._sat_T = np.ascontiguousarray(self._sat[_SI["T_l"]])
        # the tables as the compiled lookups take them (region tables flattened: i * n_z + j)
        self.tab = kp.Tab(self._lp0, self._inv_dlp, float(self.p_min), float(self.p_max), self.n_p, self.n_z,
                          np.ascontiguousarray(self._sat), np.ascontiguousarray(self._tr),
                          np.ascontiguousarray(self._vap.reshape(self._vap.shape[0], -1)),
                          np.ascontiguousarray(self._liq.reshape(self._liq.shape[0], -1)),
                          np.ascontiguousarray(self._P))

    # ---------------------------------------------------------------- queries
    # Array wrappers of the compiled scalar lookups (hgbp_sim.kernel.props); inputs
    # broadcast against each other, results take the broadcast shape.
    @staticmethod
    def _flat(*a):
        b = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in a))
        return b[0].shape, [np.ascontiguousarray(v).reshape(-1) for v in b]

    def sat(self, P):
        """Saturation properties at pressure P (arrays).  Returns a dict."""
        shape, (Pf,) = self._flat(P)
        out = np.empty((len(SAT_FIELDS), Pf.size))
        kp.sat_array(self.tab, Pf, out)
        return {k: out[_SI[k]].reshape(shape) for k in SAT_FIELDS}

    def transport(self, P):
        """Saturated liquid / vapor viscosity and surface tension at P (dict of arrays)."""
        shape, (Pf,) = self._flat(P)
        out = np.empty((len(TR_FIELDS), Pf.size))
        kp.transport_array(self.tab, Pf, out)
        return {k: out[_TI[k]].reshape(shape) for k in TR_FIELDS}

    def T_sat(self, P):
        """Saturation (bubble point) temperature at P."""
        shape, (Pf,) = self._flat(P)
        out = np.empty(Pf.size)
        kp.sat_row_array(self.tab, _SI["T_l"], Pf, out)
        return out.reshape(shape)

    def P_sat(self, T):
        """Saturation pressure for temperature T [K] (bubble line)."""
        T = np.asarray(T, dtype=float)
        return np.interp(T, self._sat_T, self._P)

    def state(self, P, h, need_s: bool = False) -> PhState:
        """Full state from pressure [Pa] and mass enthalpy [J/kg] (``need_s`` is
        accepted for compatibility; the entropy is always computed)."""
        shape, (Pf, hf) = self._flat(P, h)
        out = np.empty((13, Pf.size))
        kp.state_array(self.tab, Pf, hf, out)
        st = PhState()
        st.P, st.h = Pf.reshape(shape), hf.reshape(shape)
        (st.T, st.rho, st.x, st.drho_dP, st.drho_dh, st.s, st.cp,
         st.T_sat, st.h_l, st.h_v, st.rho_l, st.rho_v, st.cp_l) = (r.reshape(shape) for r in out)
        st.phase = np.where(st.x > 1.0, PHASE_VAPOR, np.where(st.x < 0.0, PHASE_LIQUID, PHASE_TWOPHASE))
        return st

    def T_Ph(self, P, h):
        return self.state(P, h).T

    def rho_Ph(self, P, h):
        return self.state(P, h).rho

    def T_vapor(self, P, h):
        """Temperature assuming superheated vapor (clamped to the dome)."""
        return self.vapor_props(P, h)[0]

    def vapor_props(self, P, h):
        """(T, rho) of superheated vapor at (P, h), clamped to the dew line."""
        shape, (Pf, hf) = self._flat(P, h)
        out = np.empty((2, Pf.size))
        kp.vapor_array(self.tab, Pf, hf, out)
        return out[0].reshape(shape), out[1].reshape(shape)

    def _scalar_map(self, fn, a, b):
        shape, (af, bf) = self._flat(a, b)
        out = np.empty(af.size)
        fn(self.tab, af, bf, out)
        return out.reshape(shape)

    def h_from_P_rho(self, P, rho):
        """Enthalpy at (P, rho): two-phase lever rule inside the dome, table
        inversion in the vapor / liquid regions."""
        return self._scalar_map(kp.h_from_P_rho_array, P, rho)

    def h_Ps_vapor(self, P, s):
        """Enthalpy of superheated vapor at (P, s); clamps to saturated vapor
        if s is below the dew line (wet isentropic compression)."""
        return self._scalar_map(kp.h_Ps_vapor_array, P, s)

    def h_PT(self, P, T):
        """Enthalpy at (P, T) for single-phase states.  Inside the dome
        (T == T_sat) the saturated-vapor value is returned."""
        return self._scalar_map(kp.h_PT_array, P, T)

    def h_sat_vapor(self, P):
        return self.sat(P)["h_v"]

    def h_sat_liquid(self, P):
        return self.sat(P)["h_l"]

    # ------------------------------------------------------------ descriptors
    DESCRIPTOR_NAMES = ("T_crit", "P_crit", "M_molar", "P_sat_ref", "h_fg_ref",
                        "rho_v_ref", "rho_l_ref", "dPsat_dT_ref")
    DESCRIPTOR_SCALE = np.array([400.0, 5e6, 0.1, 5e5, 2e5, 20.0, 1200.0, 1e4])

    def descriptors(self, T_ref: float = 273.15) -> dict:
        """Physical descriptors of the refrigerant used as controller context
        (critical point, molar mass, saturation properties at ``T_ref``)."""
        T_ref = float(np.clip(T_ref, self.T_sat_min + 5.0, self.T_sat_max - 5.0))
        P = self.P_sat(np.array([T_ref]))
        s = self.sat(P)
        dPdT = 1.0 / float(s["dT_l_dP"][0])
        return dict(T_crit=self.T_crit, P_crit=self.P_crit, M_molar=self.M_molar,
                    P_sat_ref=float(P[0]), h_fg_ref=float(s["h_v"][0] - s["h_l"][0]),
                    rho_v_ref=float(s["rho_v"][0]), rho_l_ref=float(s["rho_l"][0]),
                    dPsat_dT_ref=dPdT, T_ref=T_ref)

    def descriptor_vector(self, T_ref: float = 273.15) -> np.ndarray:
        """Normalized descriptor vector (order ``DESCRIPTOR_NAMES``)."""
        d = self.descriptors(T_ref)
        return np.array([d[k] for k in self.DESCRIPTOR_NAMES]) / self.DESCRIPTOR_SCALE

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
