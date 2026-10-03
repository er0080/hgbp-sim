"""Dynamic model of a hot-gas-bypass (HGBP) compressor test stand.

Topology (the stand being simulated)
------------------------------------
::

    compressor discharge --> [discharge volume, P_d] --> valve 1 (discharge pressure)
                                                              |
                                        hot gas header (P_i + condenser pressure drop)
                                            |                        |
                          path 2: valve 2 (suction pressure,   path 1: brazed-plate condenser
                                  hot gas bypass)                      <-- cooling water (valve 4)
                                            |                        |
                                            |                   liquid receiver (dip tube) -> P_i sensor
                                            |                        |
                                            |               valve 3 (suction temperature, quench)
                                            v                        v
                        mixing exchanger S1 (bottom) -> S2 (top)   S3 (top) -> S4 (bottom)
                                 bypass gas side       counterflow      quench side
                                            \\______________ tee ______________/
                                                             |
                                    suction line (probe at its end) -> compressor suction

Hot gas leaving the compressor is throttled by valve 1 to the intermediate
pressure.  Part of it bypasses through valve 2 into the gas side of a brazed
plate mixing exchanger; the rest condenses in a water-cooled brazed plate
condenser, drains into a liquid receiver and is injected ("quench") through
valve 3 into the other side of the mixing exchanger, where it evaporates
against the bypass gas in counterflow.  The two outlets join at a tee and feed
the compressor through the suction line.  Valve 4 meters the cooling water
(driven by the fixed supply-to-return pressure difference of the plant water
loop through valve 4, the piping and the condenser) and thereby sets the
intermediate (condensing) pressure, measured after the receiver.  There is no accumulator: quench liquid
the exchanger does not evaporate reaches the compressor.

Manipulated variables (0..1 stem commands, in this order):
    u[0]  valve 1  discharge pressure valve
    u[1]  valve 2  suction pressure (HGBP) valve
    u[2]  valve 3  suction temperature (quench) valve
    u[3]  valve 4  cooling water valve
Exogenous inputs: compressor speed command, ambient temperature, cooling
water inlet temperature.  Total refrigerant charge is a parameter.

Model structure
---------------
* Discharge volume (compressor -> valve 1) and intermediate section (header,
  condenser, drain, receiver, liquid line) are lumped (P, h) volumes.  In the
  intermediate section liquid collects in the order receiver (below the dip
  tube), liquid line, receiver, drain, condenser, header: the condenser keeps
  its full condensing area until the receiver is full, and the liquid seal at
  valve 3 is lost once the receiver level drops below the dip tube.
* The condenser has ``MX`` plate wall cells (top -> bottom), like the mixing
  exchanger.  Its refrigerant side is quasi-steady: the hot gas is marched from
  S3 down the cells to the liquid (wet-wall desuperheating, condensation along
  the glide; the flow is what the plates condense), giving the quality in every
  cell; liquid backing up from a full receiver floods the cells from the bottom.
  The water rises from S1 through the cells in counterflow.
* The suction side (mixing exchanger, tee, suction line and the compressor's
  internal suction volume) shares one pressure P_s.  The quench side of the
  mixing exchanger is split into ``MX`` finite-volume cells (enthalpy states,
  liquid hold-up), each against a plate wall cell with its own temperature;
  the bypass gas side is quasi-steady (its residence time is ~0.2 s) and
  marched cell by cell against the same walls.  Flows between the cells follow
  from the common pressure (lumped-pressure finite volumes).  The suction line
  and the compressor's internal volume are two more mixed cells.
* At the tee, liquid leaving the quench side is carried as droplets that
  evaporate with the time constant ``tee_tau_evap``: the suction probe at the
  end of the suction line reads the vapor temperature, which can be
  superheated while liquid still reaches the compressor.

Every side of the two plate heat exchangers is a quasi-steady flow resistance
(friction and ports, dP ~ mdot^2 / rho_m) plus, on the refrigerant sides, the
static head of its column.  The drops add no pressure states: the header sits
above the measured liquid pressure P_i by the condenser drop, valve 2 and
valve 3 each discharge through their exchanger side in series, and the quench
cells boil at their own pressure along the S3 -> S4 column.

Mass and internal energy of every volume (the suction side as one group) are
integrated as conserved states; after every sub-step (P, h) are projected
back onto them (a bracketed density-energy flash for the discharge and
intermediate volumes, a Newton correction of the common pressure and a
uniform enthalpy shift for the suction side).  Volumes that are (nearly)
liquid-full are integrated with a finer sub-step because their pressure
dynamics are stiff.

The model is compiled (numba, :mod:`hgbp_sim.kernel.model`): every
environment of a batch integrates in its own parallel loop iteration with its
own step size control.  This class holds the batch (states, parameters,
inputs) and is the Python interface to it.
"""
from __future__ import annotations

import numpy as np

from .geometry import derive
from .kernel import layout as L
from .kernel import model as km
from .params import PlantParams, nominal_charge, sample_params
from .properties import RefrigerantTables, get_tables

MX = L.MX       # finite-volume cells per side of the mixing exchanger and of the condenser
G = km.G        # gravity [m/s^2]


class HGBPPlant:
    """Batched stand model.  State vector (per environment, ``STATE_NAMES``):

    P_s                 common suction side pressure
    h_q0..h_q{MX-1}     quench side cells of the mixing exchanger (top -> bottom)
    h_l1, h_l2          tee + suction line, compressor internal suction volume
    T_mw0..             mixing exchanger plate walls (top -> bottom)
    T_sw                suction line wall
    P_d, h_d, T_dw      discharge volume and discharge line wall
    P_i, h_i            intermediate section
    T_cw0..             condenser plate walls + water content (top -> bottom)
    T_rw                receiver shell
    T_sh, N             compressor shell temperature, speed [rpm]
    u1..u4              actual valve positions
    Tm_s, Tm_d, Tm_co   lagged temperature sensors (suction probe, discharge, liquid)
    mm, Wm              lagged mass-flow and power sensors
    M_s, M_d, M_i       refrigerant mass of the suction side, discharge, intermediate section
    U_s, U_d, U_i       their internal energy
    h_g0..h_g{MX-1}     bypass gas side enthalpies (algebraic, kept for the mass accounting)
    """
    MX = MX
    STATE_NAMES = L.STATE_NAMES
    IX = L.IX
    NX = L.NX
    NU = 4
    P_S = L.X_P_S
    HS = slice(L.X_HQ, L.X_H_L2 + 1)                # dynamic suction side cells (quench, L1, L2)
    H_L1, H_L2 = L.X_H_L1, L.X_H_L2
    TMW = slice(L.X_TMW, L.X_TMW + MX)
    T_SW = L.X_T_SW
    P_D, H_D, T_DW = L.X_P_D, L.X_H_D, L.X_T_DW
    P_I, H_I, T_RW = L.X_P_I, L.X_H_I, L.X_T_RW
    TCW = slice(L.X_TCW, L.X_TCW + MX)
    T_SH, N_ = L.X_T_SH, L.X_N
    U1, U2, U3, U4 = L.X_U1, L.X_U1 + 1, L.X_U1 + 2, L.X_U1 + 3
    UV = slice(U1, U4 + 1)
    TM_S, TM_D, MM, WM, TM_CO = L.X_TM_S, L.X_TM_D, L.X_MM, L.X_WM, L.X_TM_CO
    M_S, M_D, M_I, U_S, U_D, U_I = L.X_M_S, L.X_M_D, L.X_M_I, L.X_U_S, L.X_U_D, L.X_U_I
    HG = slice(L.X_HG, L.X_HG + MX)
    STIFF_FILL = km.STIFF_FILL
    STIFF_SUBDIV = km.STIFF_SUBDIV
    LAM_DT_MAX = km.LAM_DT_MAX
    MAX_SUBDIV = km.MAX_SUBDIV
    MAX_DH_STEP = km.MAX_DH_STEP
    MAX_DP_STEP = km.MAX_DP_STEP
    DRYOUT_BAND = km.DRYOUT_BAND
    VALVE_NAMES = ("discharge_pressure", "suction_pressure", "suction_temperature", "water")

    def __init__(self, params: PlantParams | None = None, n: int = 1,
                 dt: float = 0.05, integrator: str = "rk4",
                 rng: np.random.Generator | None = None,
                 randomize: bool = False, randomize_spec: dict | None = None,
                 props: RefrigerantTables | None = None, parallel: bool | None = None):
        if integrator not in km.INTEGRATORS:
            raise ValueError(f"unknown integrator {integrator!r}")
        self.params = params if params is not None else PlantParams()
        self.n = int(n)
        self.dt = float(dt)
        self.integrator = integrator
        self.rng = rng if rng is not None else np.random.default_rng()
        self.props = props if props is not None else get_tables(self.params.fluid)
        self.randomize = randomize
        self.randomize_spec = randomize_spec
        # environments in parallel on numba's thread pool (NUMBA_NUM_THREADS); default: when n > 1
        self.parallel = parallel
        self.p = sample_params(self.params, self.n, self.rng, randomize_spec, randomize)
        self._prepare(self.p)
        self.x = np.zeros((self.n, self.NX))
        self.t = 0.0
        self.u_cmd = np.zeros((self.n, self.NU))
        self.N_cmd = np.zeros(self.n)
        self.T_amb = np.full(self.n, 298.15)
        self.T_wi = np.full(self.n, 298.15)
        self.aux: dict | None = None
        self.n_rhs_calls = 0             # right-hand side evaluations (all environments together)
        self.substeps = np.zeros(self.n, np.int64)    # sub-steps each environment took in the last step
        self._lam = None                 # fastest local rate of each environment [1/s]
        self._lam_last = None            # ... from the last right-hand side evaluation

    # ------------------------------------------------------------ parameters
    def _prepare(self, p) -> None:
        derive(p, self.MX)
        if p.charge is None:
            p.charge = nominal_charge(p, self.props)

    def params_subset(self, idx, repeat: int = 1):
        """Parameter namespace restricted to environments ``idx`` (optionally
        each repeated ``repeat`` times consecutively)."""
        idx = np.atleast_1d(np.asarray(idx))
        ns = type(self.p)()
        for k, v in vars(self.p).items():
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray):
                vv = v[idx]
                setattr(ns, k, np.repeat(vv, repeat, axis=0) if repeat > 1 else vv)
            elif isinstance(v, dict):
                setattr(ns, k, {kk: (np.repeat(vv[idx], repeat, axis=0) if repeat > 1 else vv[idx])
                                for kk, vv in v.items()})
            else:
                setattr(ns, k, v)
        return ns

    def resample_params(self, idx=None, rng=None) -> None:
        """Re-randomize physical parameters for environments ``idx``.  The
        charge of those environments is reset to the nominal value for the
        new volumes (override afterwards via ``plant.p.charge[idx]``)."""
        rng = rng or self.rng
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        if len(idx) == 0:
            return
        new = sample_params(self.params, len(idx), rng, self.randomize_spec, self.randomize)
        new.charge = None
        self._prepare(new)
        for k, v in vars(new).items():
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray):
                getattr(self.p, k)[idx] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    getattr(self.p, k)[kk][idx] = vv

    def nominal_charge(self, idx=None):
        p = self.p if idx is None else self.params_subset(idx)
        return nominal_charge(p, self.props)

    def _suction_cells(self, P_s, H):
        """Properties of the dynamic suction side cells (P_s shared)."""
        nb, k = H.shape
        S = self.props.state(np.repeat(P_s, k), H.reshape(-1))
        return {a: getattr(S, a).reshape(nb, k) for a in
                ("rho", "T", "x", "drho_dP", "drho_dh", "s", "T_sat", "h_l", "h_v", "rho_l")}

    # ----------------------------------------------------------------- RHS
    @staticmethod
    def _rows(a, n, width=None):
        """``a`` broadcast to one C-contiguous row per environment."""
        shape = (n,) if width is None else (n, width)
        return np.ascontiguousarray(np.broadcast_to(np.asarray(a, float), shape))

    def rhs(self, x, u_cmd, N_cmd, T_amb, T_wi, want_aux: bool = False, p=None, hold_P_s: bool = False):
        """Time derivative of the state matrix ``x`` (shape (n, NX)); with
        ``want_aux`` also the auxiliary outputs (dict of arrays).

        ``p`` may override the parameter namespace (used by the steady-state
        solver to evaluate stacked copies of a subset of environments);
        ``hold_P_s`` returns the suction cell enthalpy rates at constant
        suction pressure (each cell's own energy balance; the steady-state
        solver uses these as residuals next to dP_s/dt)."""
        p = self.p if p is None else p
        x = np.ascontiguousarray(x, dtype=float)
        nb = x.shape[0]
        dx = np.zeros_like(x)
        aux = np.empty((nb, L.NA if want_aux else 0))
        hg = np.empty((nb, self.MX))
        lam = np.empty(nb)
        km.rhs_batch[self._parallel(nb)](
            x, self._rows(u_cmd, nb, self.NU), self._rows(N_cmd, nb), self._rows(T_amb, nb), self._rows(T_wi, nb),
            p.packed(), self.props.tab, bool(hold_P_s), dx, bool(want_aux), aux, hg, lam)
        self.n_rhs_calls += nb
        self._lam_last = lam
        if not want_aux:
            return dx
        return dx, self._aux_dict(aux, p.charge)

    def _parallel(self, nb: int) -> bool:
        """Run a batch of ``nb`` environments on numba's thread pool?"""
        return self.parallel if self.parallel is not None else nb > 1

    def _aux_dict(self, aux, charge) -> dict:
        """Auxiliary outputs by name from their array (layout.AUX_FIELDS)."""
        out = {k: (aux[:, o] if w == 1 else aux[:, o:o + w]) for k, (o, w) in L.AUX_SLICES.items()}
        out["t"] = self.t
        out["charge"] = charge
        return out

    # ------------------------------------------------------------ integrate
    def _sync_mass(self, x, idx=None) -> None:
        """Set the conserved mass and energy states from (P, h) (after direct
        state assignments); ``idx`` selects the environments of the rows of ``x``."""
        p = self.p if idx is None else self.params_subset(idx)
        km.sync_mass_batch(x, p.packed(), self.props.tab)

    def _stiff(self, x, p=None):
        """Per environment: True if a volume is (nearly) liquid-full (see
        :func:`hgbp_sim.kernel.model.stiff`)."""
        p = self.p if p is None else p
        out = np.zeros(x.shape[0], np.bool_)
        km.stiff_batch(np.ascontiguousarray(x, dtype=float), p.packed(), self.props.tab, out)
        return out

    def set_inputs(self, u_cmd=None, N_cmd=None, T_amb=None, T_wi=None) -> None:
        if u_cmd is not None:
            self.u_cmd = np.broadcast_to(np.asarray(u_cmd, float), (self.n, self.NU)).copy()
        if N_cmd is not None:
            self.N_cmd = np.broadcast_to(np.asarray(N_cmd, float), (self.n,)).copy()
        if T_amb is not None:
            self.T_amb = np.broadcast_to(np.asarray(T_amb, float), (self.n,)).copy()
        if T_wi is not None:
            self.T_wi = np.broadcast_to(np.asarray(T_wi, float), (self.n,)).copy()

    def step(self, duration: float | None = None, n_sub: int | None = None,
             u_cmd=None, N_cmd=None, T_amb=None, T_wi=None) -> dict:
        """Advance the plant by ``duration`` seconds (or ``n_sub`` steps of
        ``dt``) holding the inputs constant.  Every environment splits each step
        into the sub-steps its fastest local rate calls for.  Returns the
        auxiliary output dict evaluated at the new state."""
        self.set_inputs(u_cmd, N_cmd, T_amb, T_wi)
        if n_sub is None:
            n_sub = max(1, int(round((duration if duration is not None else self.dt) / self.dt)))
        args = (self.u_cmd, self.N_cmd, self.T_amb, self.T_wi)
        if self.aux is None or self._lam is None or len(self._lam) != self.n:
            self.rhs(self.x, *args)          # state was set from outside: fresh rate estimate
            self._lam = self._lam_last
        x = np.array(self.x, dtype=float, order="C")      # the previous state stays as it was
        lam = self._lam.copy()
        km.step_batch[self._parallel(self.n)](x, *args, self.p.packed(), self.props.tab, lam, self.dt,
                                              int(n_sub), km.INTEGRATORS[self.integrator], self.substeps)
        self.n_rhs_calls += int(self.substeps.sum()) * {"rk4": 4, "heun": 2, "euler": 1}[self.integrator]
        self.x = x
        self.t += n_sub * self.dt
        _, self.aux = self.rhs(self.x, *args, want_aux=True)
        self._lam = self._lam_last
        return self.aux

    def outputs(self) -> dict:
        if self.aux is None:
            _, self.aux = self.rhs(self.x, self.u_cmd, self.N_cmd, self.T_amb,
                                   self.T_wi, want_aux=True)
            self._lam = self._lam_last       # rate estimate of this state (not a previous episode's)
        return self.aux

    def conserved_mass(self):
        """Refrigerant mass from the conserved states [kg] (exact charge)."""
        return self.x[:, self.M_S] + self.x[:, self.M_D] + self.x[:, self.M_I]

    # -------------------------------------------------------- initialization
    def cold_start(self, idx=None, T_amb=None, u_pos=None, charge=None,
                   liquid_in_suction=None) -> None:
        """Equalized, compressor-off initial condition at ambient temperature.

        The total charge is distributed as saturated vapor in all volumes plus
        liquid in the intermediate section (receiver first); a share
        ``liquid_in_suction`` (default ``params.cold_liquid_in_suction``) of the
        liquid has migrated to the suction side (compressor internal volume
        first, then suction line and mixing exchanger), as after a long off
        cycle.  If the charge is too small to reach saturation at ambient the
        whole stand holds superheated vapor at a lower pressure.  Raises
        ``ValueError`` if the charge does not fit into the stand.
        """
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx))
        if len(idx) == 0:
            return
        m = len(idx)
        pr = self.props
        p = self.params_subset(idx)
        T_amb = self.T_amb[idx] if T_amb is None else np.broadcast_to(np.asarray(T_amb, float), (m,))
        self.T_amb[idx] = T_amb
        if charge is not None:
            self.p.charge[idx] = np.broadcast_to(np.asarray(charge, float), (m,))
        charge = self.p.charge[idx]
        f_suc = p.cold_liquid_in_suction if liquid_in_suction is None \
            else np.broadcast_to(np.asarray(liquid_in_suction, float), (m,))
        V_tot = p.V_s + p.V_d + p.V_i
        n = self.MX

        P_eq = pr.P_sat(T_amb)
        sat = pr.sat(P_eq)
        rho_v, rho_l = sat["rho_v"], sat["rho_l"]
        # liquid mass accounting for the vapor volume it displaces
        M_liq = np.maximum((charge - rho_v * V_tot) / (1.0 - rho_v / rho_l), 0.0)
        wet = charge > rho_v * V_tot

        # --- suction side cells fill in the order L2, L1, quench cells bottom -> top
        order = [n + 1, n] + list(range(n - 1, -1, -1))
        cap = 0.95 * rho_l[:, None] * p.V_sc
        M_ls = np.zeros((m, n + 2))
        want = f_suc * M_liq
        for k in order:
            M_ls[:, k] = np.minimum(want, cap[:, k])
            want = want - M_ls[:, k]
        cap_i = 0.95 * rho_l * p.V_i
        M_li = np.minimum(M_liq - M_ls.sum(1), cap_i)
        rem = M_liq - M_ls.sum(1) - M_li
        for k in order:                                   # overflow of the intermediate section
            add = np.minimum(rem, cap[:, k] - M_ls[:, k])
            M_ls[:, k] += add
            rem = rem - add
        if np.any(rem > 1e-6):
            raise ValueError("refrigerant charge exceeds the stand's liquid capacity")

        def h_mix(M_l, V):
            M_v = rho_v[:, None] * (V - M_l / rho_l[:, None]) if np.ndim(M_l) == 2 \
                else rho_v * (V - M_l / rho_l)
            xq = M_v / (M_v + M_l)
            hl, hv = (sat["h_l"][:, None], sat["h_v"][:, None]) if np.ndim(M_l) == 2 else (sat["h_l"], sat["h_v"])
            return hl + xq * (hv - hl)

        h_vap = sat["h_v"] + 300.0
        # suction side vapor at saturation (in equilibrium with walls at ambient)
        H = np.where(M_ls > 0.0, h_mix(M_ls, p.V_sc), sat["h_v"][:, None])
        h_i = np.where(M_li > 0.0, h_mix(M_li, p.V_i), h_vap)
        h_d = h_vap
        Hg = np.repeat(sat["h_v"][:, None], n, axis=1)
        P = P_eq.copy()

        # --- dry stand: superheated vapor at the pressure holding the charge
        if np.any(~wet):
            lo = np.full(m, pr.p_min * 1.01)
            hi = P_eq.copy()
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                rho = pr.state(mid, pr.h_PT(mid, T_amb)).rho
                too_dense = rho * V_tot > charge
                hi = np.where(too_dense, mid, hi)
                lo = np.where(too_dense, lo, mid)
            P_dry = 0.5 * (lo + hi)
            h_dry = pr.h_PT(P_dry, T_amb)
            P = np.where(wet, P, P_dry)
            H = np.where(wet[:, None], H, h_dry[:, None])
            Hg = np.where(wet[:, None], Hg, h_dry[:, None])
            h_i = np.where(wet, h_i, h_dry)
            h_d = np.where(wet, h_d, h_dry)

        x = np.zeros((m, self.NX))
        x[:, self.P_S], x[:, self.HS], x[:, self.HG] = P, H, Hg
        x[:, self.TMW] = T_amb[:, None]
        x[:, self.TCW] = T_amb[:, None]
        x[:, self.P_D], x[:, self.H_D] = P, h_d
        x[:, self.P_I], x[:, self.H_I] = P, h_i
        for i in (self.T_SW, self.T_DW, self.T_RW, self.T_SH, self.TM_S, self.TM_D, self.TM_CO):
            x[:, i] = T_amb
        u_pos = np.zeros((m, self.NU)) if u_pos is None else np.broadcast_to(np.asarray(u_pos, float), (m, self.NU))
        x[:, self.UV] = u_pos
        self._sync_mass(x, idx)
        # the vapor above is taken slightly superheated: give the intermediate section
        # exactly the remaining mass so that the stand holds the requested charge
        rho_i = (charge - x[:, self.M_S] - x[:, self.M_D]) / p.V_i
        x[:, self.H_I] = pr.h_from_P_rho(x[:, self.P_I], rho_i)
        self._sync_mass(x, idx)
        self.x[idx] = x
        self.u_cmd[idx] = u_pos
        self.N_cmd[idx] = 0.0
        self.aux = None

    def set_state(self, idx, x, sync_mass: bool = True) -> None:
        idx = np.atleast_1d(np.asarray(idx))
        x = np.asarray(x, float).reshape(len(idx), self.NX).copy()
        if sync_mass:
            self._sync_mass(x, idx)
        self.x[idx] = x
        self.aux = None

    # ------------------------------------------------------------- sensing
    def measure(self, rng: np.random.Generator | None = None, noise: bool = True) -> dict:
        """Sensor readings (with lag from the sensor states and optional noise)."""
        rng = rng or self.rng
        a = self.outputs()
        p = self.p
        n = self.n
        z = (lambda s: rng.standard_normal(n) * s) if noise else (lambda s: 0.0)
        P_s = a["P_s"] + z(p.sig_P_s)
        P_d = a["P_d"] + z(p.sig_P_d)
        P_i = a["P_i"] + z(p.sig_P_d)
        T_s = a["Tm_s"] + z(p.sig_T)
        T_d = a["Tm_d"] + z(p.sig_T)
        T_co = a["Tm_co"] + z(p.sig_T)
        mdot = a["mm"] * (1.0 + z(p.sig_m_rel))
        W = a["Wm"] * (1.0 + z(p.sig_W_rel))
        T_sat_s = self.props.T_sat(np.maximum(P_s, self.props.p_min))
        T_sat_i = self.props.T_sat(np.maximum(P_i, self.props.p_min))
        return dict(P_s=P_s, P_d=P_d, P_i=P_i, T_s=T_s, T_d=T_d, T_co=T_co,
                    SH=T_s - T_sat_s, SC=T_sat_i - T_co, T_sat_s=T_sat_s, T_sat_i=T_sat_i,
                    mdot=mdot, W=W, N=a["N"], u1=a["u1"], u2=a["u2"], u3=a["u3"], u4=a["u4"],
                    T_wi=self.T_wi.copy(), T_wo=a["T_wo"] + z(p.sig_T), T_amb=self.T_amb.copy(),
                    rec_level=a["rec_level"].copy())          # sight glass (read by eye, no noise)

    # --------------------------------------------------------------- checks
    def trips(self) -> dict:
        """Hard trips and soft condition flags (boolean arrays)."""
        a = self.outputs()
        p = self.p
        return dict(
            high_P_d=a["P_d"] > p.P_d_max,
            low_P_s=a["P_s"] < p.P_s_min,
            high_P_s=a["P_s"] > p.P_s_max,
            high_T_d=a["T_d"] > p.T_d_max,
            floodback=a["y_liq"] > p.y_flood,           # liquid reaching the compressor
            mixer_wet=a["x_qo"] < 1.0,                  # quench liquid leaving the mixing exchanger
            no_liquid_seal=a["ll_fill"] < 0.5,          # undercharged: receiver below the dip tube
            receiver_full=a["rec_level"] > 0.9,         # overcharged: receiver above its rated fill
            condenser_flooded=a["cond_flood"] > 0.05,   # liquid backing up into the condenser
        )
