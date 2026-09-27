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
dynamics are stiff.  Everything is vectorized over a batch dimension so many
environments integrate at once.
"""
from __future__ import annotations

import numpy as np

from .components import (actuator_rate, amalfi_ftp, clip, compressor, counterflow_effectiveness,
                         cv_balance, gas_valve_flow, hx_drop, kv_to_C, liquid_valve_flow, martin_xi,
                         mixture_viscosity, pipe_drop, series_C, smoothstep, valve_characteristic,
                         water_valve_flow)
from .geometry import derive
from .params import PlantParams, nominal_charge, sample_params
from .properties import RefrigerantTables, get_tables

MX = 5          # finite-volume cells per side of the mixing exchanger
G = 9.81        # gravity [m/s^2]


def _valve_slope(Cf, rho_up, dP, eps):
    """|d mdot / d dP| of the regularized orifice equation (choking ignored: an
    upper bound used only to size the integration step)."""
    z2 = dP * dP
    e2 = eps * eps
    return Cf * np.sqrt(np.maximum(rho_up, 1e-9)) * (0.5 * z2 + e2) / np.power(z2 + e2, 1.25)


def _state_names():
    return tuple(["P_s"] + [f"h_q{j}" for j in range(MX)] + ["h_l1", "h_l2"]
                 + [f"T_mw{j}" for j in range(MX)]
                 + ["T_sw", "P_d", "h_d", "T_dw", "P_i", "h_i", "T_cw", "T_rw", "T_sh", "N",
                    "u1", "u2", "u3", "u4", "Tm_s", "Tm_d", "mm", "Wm", "Tm_co",
                    "M_s", "M_d", "M_i", "U_s", "U_d", "U_i"]
                 + [f"h_g{j}" for j in range(MX)])


class HGBPPlant:
    """Batched stand model.  State vector (per environment, ``STATE_NAMES``):

    P_s                 common suction side pressure
    h_q0..h_q{MX-1}     quench side cells of the mixing exchanger (top -> bottom)
    h_l1, h_l2          tee + suction line, compressor internal suction volume
    T_mw0..             mixing exchanger plate walls (top -> bottom)
    T_sw                suction line wall
    P_d, h_d, T_dw      discharge volume and discharge line wall
    P_i, h_i, T_cw      intermediate section, condenser plates + water content
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
    STATE_NAMES = _state_names()
    IX = {k: i for i, k in enumerate(STATE_NAMES)}
    NX = len(STATE_NAMES)
    NU = 4
    P_S = IX["P_s"]
    HS = slice(IX["h_q0"], IX["h_l2"] + 1)          # dynamic suction side cells (quench, L1, L2)
    H_L1, H_L2 = IX["h_l1"], IX["h_l2"]
    TMW = slice(IX["T_mw0"], IX["T_mw0"] + MX)
    T_SW = IX["T_sw"]
    P_D, H_D, T_DW = IX["P_d"], IX["h_d"], IX["T_dw"]
    P_I, H_I, T_CW, T_RW = IX["P_i"], IX["h_i"], IX["T_cw"], IX["T_rw"]
    T_SH, N_ = IX["T_sh"], IX["N"]
    U1, U2, U3, U4 = IX["u1"], IX["u2"], IX["u3"], IX["u4"]
    UV = slice(U1, U4 + 1)
    TM_S, TM_D, MM, WM, TM_CO = IX["Tm_s"], IX["Tm_d"], IX["mm"], IX["Wm"], IX["Tm_co"]
    M_S, M_D, M_I, U_S, U_D, U_I = IX["M_s"], IX["M_d"], IX["M_i"], IX["U_s"], IX["U_d"], IX["U_i"]
    HG = slice(IX["h_g0"], IX["h_g0"] + MX)
    # lumped volumes with their own pressure: (P index, h index, M index, U index, volume name)
    CV = ((P_D, H_D, M_D, U_D, "V_d"), (P_I, H_I, M_I, U_I, "V_i"))
    STIFF_FILL = 0.97           # liquid fill above which a volume counts as liquid-full (stiff)
    STIFF_SUBDIV = 4            # sub-step refinement for stiff volumes
    LAM_DT_MAX = 1.8            # largest (estimated fastest local rate x sub-step); RK4 is stable to 2.78,
                                # the margin covers the estimate running up to ~35 % low
    MAX_SUBDIV = 32             # cap on the sub-step refinement
    MAX_DH_STEP = 40e3          # suction cell enthalpy change per sub-step above which it is redone finer [J/kg]
    MAX_DP_STEP = 0.2           # relative pressure change per sub-step above which it is redone finer
    DRYOUT_BAND = 15e3          # quench enthalpy band above the dew point over which a cell dries out [J/kg]
    VALVE_NAMES = ("discharge_pressure", "suction_pressure", "suction_temperature", "water")

    def __init__(self, params: PlantParams | None = None, n: int = 1,
                 dt: float = 0.05, integrator: str = "rk4",
                 rng: np.random.Generator | None = None,
                 randomize: bool = False, randomize_spec: dict | None = None,
                 props: RefrigerantTables | None = None):
        self.params = params if params is not None else PlantParams()
        self.n = int(n)
        self.dt = float(dt)
        self.integrator = integrator
        self.rng = rng if rng is not None else np.random.default_rng()
        self.props = props if props is not None else get_tables(self.params.fluid)
        self.randomize = randomize
        self.randomize_spec = randomize_spec
        self.p = sample_params(self.params, self.n, self.rng, randomize_spec, randomize)
        self._prepare(self.p)
        self.x = np.zeros((self.n, self.NX))
        self.t = 0.0
        self.u_cmd = np.zeros((self.n, self.NU))
        self.N_cmd = np.zeros(self.n)
        self.T_amb = np.full(self.n, 298.15)
        self.T_wi = np.full(self.n, 298.15)
        self.aux: dict | None = None
        self.n_rhs_calls = 0
        self._hg_last = None
        self._lam = None
        self._lam_last = None

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
            if isinstance(v, np.ndarray):
                getattr(self.p, k)[idx] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    getattr(self.p, k)[kk][idx] = vv

    def nominal_charge(self, idx=None):
        p = self.p if idx is None else self.params_subset(idx)
        return nominal_charge(p, self.props)

    # --------------------------------------------------------- inventories
    @staticmethod
    def intermediate_split(p, fill_i):
        """Where the liquid of the intermediate section sits, from its liquid
        volume share ``fill_i``.  Returns receiver level, liquid line fill
        (1 = liquid seal at valve 3) and flooded share of the condenser."""
        seg = p.seg_i
        V_L = fill_i * p.V_i
        lower = np.cumsum(seg, axis=1) - seg
        part = clip(V_L[:, None] - lower, 0.0, seg)
        rec_level = (part[:, 0] + part[:, 2]) / p.rec_V
        w_ll = np.maximum(seg[:, 1], 2e-4)          # the seal forms over at least 0.2 L
        ll_fill = clip((V_L - lower[:, 1]) / w_ll, 0.0, 1.0)
        cond_flood = part[:, 4] / np.maximum(seg[:, 4], 1e-9)
        return rec_level, ll_fill, cond_flood

    def _suction_cells(self, P_s, H):
        """Properties of the dynamic suction side cells (P_s shared)."""
        nb, k = H.shape
        S = self.props.state(np.repeat(P_s, k), H.reshape(-1), need_s=True)
        return {a: getattr(S, a).reshape(nb, k) for a in
                ("rho", "T", "x", "drho_dP", "drho_dh", "s", "T_sat", "h_l", "h_v", "rho_l")}

    @staticmethod
    def _pipe(p, key, mdot, rho, mu, share=1.0):
        """Friction and fittings drop of refrigerant line ``key`` (or the ``share`` of it)."""
        D, t, L, K = (getattr(p, f"{a}_{key}") for a in ("D", "t", "L", "K"))
        return pipe_drop(mdot, np.maximum(D - 2.0 * t, 1e-4), share * L, share * K, rho, mu)

    @staticmethod
    def _equiv_C(m0, dP0, rho_ref):
        """Equivalent series coefficient (m = C sqrt(rho_ref dP)) of flow-dependent
        drops ``dP0`` evaluated at the flow ``m0`` a valve passes on its own."""
        dP0 = np.abs(dP0)
        return np.where(dP0 > 1.0, np.abs(m0) / np.sqrt(rho_ref * np.maximum(dP0, 1.0)), 1e3)

    def _quench_drops(self, p, mdot, rho_in, rho_c, x_c, sat_s, tr):
        """Flow-dependent pressure drops of the mixing exchanger's quench side (S3 -> S4)
        at quench flow ``mdot``, static heads excluded.  ``rho_in``: the valve 3 outlet
        mixture; ``rho_c``, ``x_c``: density and quality of the quench cells (top ->
        bottom).  Returns the drop ahead of the channels (distributor, S3 connection and
        port), each cell's friction plus acceleration, and the drop after them (S4 port
        and connection, into the outlet pipe)."""
        n = self.MX
        m, sgn = np.abs(mdot), np.sign(mdot)
        dh = (2.0 * p.mx_b / p.mx_phi)[:, None]
        Gc = (m / ((p.V_mx_q / p.mx_V_ch) * p.mx_b * p.mx_W))[:, None]     # channel mass flux
        dz = (p.mx_H / n)[:, None]
        rl, rv = sat_s["rho_l"][:, None], sat_s["rho_v"][:, None]
        beta = p.mx_beta[:, None]
        # friction: two-phase (Amalfi et al.) in the evaporating cells, Martin's single-phase
        # correlation for vapor (after dry-out) or liquid, blended across the dome
        f_tp = amalfi_ftp(Gc, dh, rho_c, rl, rv, tr["sigma"][:, None], beta)
        dp_tp = 2.0 * f_tp * Gc * Gc * dz / (dh * rho_c)
        mu = np.where(x_c >= 0.5, tr["mu_v"][:, None], tr["mu_l"][:, None])
        dp_sp = martin_xi(Gc * dh / mu, beta) * dz / dh * Gc * Gc / (2.0 * rho_c)
        w_sp = smoothstep((x_c - 0.975) / 0.05) + smoothstep((0.025 - x_c) / 0.05)
        fric = sgn[:, None] * ((1.0 - w_sp) * dp_tp + w_sp * dp_sp)
        # acceleration of the evaporating flow (homogeneous), forward flow only
        v_prev = np.concatenate([1.0 / rho_in[:, None], 1.0 / rho_c[:, :n - 1]], 1)
        acc = (mdot > 0.0)[:, None] * Gc * Gc * (1.0 / rho_c - v_prev)
        # ports (Shah & Focke: 0.75 velocity heads each end) and the 7/8 in connections:
        # expansion from the connection into the port at S3, contraction from the port into
        # the connection and expansion into the outlet pipe at S4
        A_p, A_c = 0.25 * np.pi * p.mx_d_port ** 2, 0.25 * np.pi * p.mx_d_S34 ** 2
        d_mo = np.maximum(p.D_mo - 2.0 * p.t_mo, p.mx_d_S34)
        a_cp = (p.mx_d_S34 / p.mx_d_port) ** 2
        K_in = (1.0 - a_cp) ** 2
        K_out = 0.5 * (1.0 - a_cp) + (1.0 - (p.mx_d_S34 / d_mo) ** 2) ** 2
        vh = lambda A, rho: mdot * m / (A * A * 2.0 * rho)       # signed velocity head
        rho_out = rho_c[:, n - 1]
        dist = mdot * m / (kv_to_C(p.mx_Kv_dist) ** 2 * rho_in)
        inlet = dist + 0.75 * vh(A_p, rho_in) + K_in * vh(A_c, rho_in)
        outlet = 0.75 * vh(A_p, rho_out) + K_out * vh(A_c, rho_out)
        return dict(dist=dist, inlet=inlet, seg=fric + acc, outlet=outlet)

    def _gas_cells(self, P_s, Hg):
        nb, k = Hg.shape
        T, rho = self.props.vapor_props(np.repeat(P_s, k), Hg.reshape(-1))
        rho = rho.reshape(nb, k)
        return T.reshape(nb, k), rho, 1.2 * rho / P_s[:, None]    # drho/dP|h of the vapor (ideal-gas-like)

    # ----------------------------------------------------------------- RHS
    def rhs(self, x, u_cmd, N_cmd, T_amb, T_wi, want_aux: bool = False, p=None, hold_P_s: bool = False):
        """Time derivative of the state matrix ``x`` (shape (n, NX)).

        ``p`` may override the parameter namespace (used by the steady-state
        solver to evaluate stacked copies of a subset of environments);
        ``hold_P_s`` returns the suction cell enthalpy rates at constant
        suction pressure (each cell's own energy balance; the steady-state
        solver uses these as residuals next to dP_s/dt)."""
        p = self.p if p is None else p
        pr = self.props
        n = self.MX
        self.n_rhs_calls += 1
        nb = x.shape[0]
        P_s = x[:, self.P_S]
        H = x[:, self.HS]                               # (nb, n + 2): quench cells, L1, L2
        T_mw = x[:, self.TMW]
        T_sw = x[:, self.T_SW]
        P_d, h_d, T_dw = x[:, self.P_D], x[:, self.H_D], x[:, self.T_DW]
        P_i, h_i, T_cw, T_rw = x[:, self.P_I], x[:, self.H_I], x[:, self.T_CW], x[:, self.T_RW]
        T_sh, N = x[:, self.T_SH], x[:, self.N_]
        u1, u2, u3, u4 = (x[:, i] for i in (self.U1, self.U2, self.U3, self.U4))
        Tm_s, Tm_d, mm, Wm, Tm_co = (x[:, i] for i in (self.TM_S, self.TM_D, self.MM, self.WM, self.TM_CO))
        h_l1, h_l2 = H[:, n], H[:, n + 1]

        # ---- suction side cells (common pressure)
        C = self._suction_cells(P_s, H)
        Vc = p.V_sc
        M_c = C["rho"] * Vc
        T_sat_s, h_ls, h_vs = C["T_sat"][:, 0], C["h_l"][:, 0], C["h_v"][:, 0]
        sat_s = pr.sat(P_s)
        D = pr.state(P_d, h_d)                          # discharge volume
        I = pr.state(P_i, h_i)                          # intermediate section

        # ---- compressor, drawing from its internal suction volume (L2); it takes in at most
        # 1 - comp_x_min liquid by mass, more liquid collects in the shell
        h_cin = np.maximum(h_l2, h_ls + p.comp_x_min * (h_vs - h_ls))
        Cin = pr.state(P_s, h_cin, need_s=True)
        cp = compressor(p, pr, P_s, h_cin, Cin.s, Cin.rho, P_d, N, T_sh)
        mdot_c, h2 = cp["mdot"], cp["h2"]

        # ---- intermediate section inventory: receiver, liquid seal, condenser flooding
        M_i = I.rho * p.V_i
        fill_i = np.minimum((1.0 - clip(I.x, 0.0, 1.0)) * M_i / (I.rho_l * p.V_i), 1.0)
        rec_level, ll_fill, cond_flood = self.intermediate_split(p, fill_i)
        dry = 1.0 - smoothstep(ll_fill)                           # 1: no liquid seal at valve 3
        h_cv_out = np.where(I.x < 0.0, h_i, I.h_l) * (1.0 - dry) + h_i * dry
        rho_co = I.rho_l * (1.0 - dry) + I.rho * dry
        h_iv = I.h_v * (1.0 - dry) + h_i * dry                    # vapor phase of the section

        # ---- condenser, refrigerant -> wall.  Plate area shares: flooded by liquid backing
        # up from a full receiver, condensing (a_c) and vapor-covered (the rest).
        w2 = (1.0 - smoothstep((I.x - 0.85) / 0.15)) * (1.0 - smoothstep(-I.x / 0.05))
        a_c = w2 * (1.0 - cond_flood)
        UA_r = p.cond_A * (p.alpha_r_2ph * a_c + p.alpha_r_1ph * (1.0 - cond_flood - a_c))
        Q_r = UA_r * (I.T - T_cw)                        # refrigerant -> wall

        # ---- condenser pressure drop.  P_i is measured at the outlet (after the receiver,
        # ahead of valve 3); the header at the inlet sits above it by the friction drop of
        # the condensing flow and below it by the static head of the refrigerant column
        # (homogeneous two-phase over the condensing area, liquid where flooded).
        sat_i = pr.sat(P_i)
        rho_v = np.where(I.x >= 1.0, I.rho, sat_i["rho_v"])
        rho_l = sat_i["rho_l"]
        mdot_cr = np.maximum(Q_r, 0.0) / np.maximum(h_d - I.h_l, 1e4)
        rho_fr = w2 * 2.0 / (1.0 / rho_v + 1.0 / rho_l) + (1.0 - w2) * rho_v
        r_lv = np.maximum(rho_l / rho_v, 1.001)
        rho_hm = rho_v * np.log(r_lv) / (1.0 - 1.0 / r_lv)          # column mean, quality 1 -> 0
        rho_col = cond_flood * rho_l + a_c * rho_hm + (1.0 - cond_flood - a_c) * rho_v
        dP_cr = (1.0 - cond_flood) * hx_drop(mdot_cr, kv_to_C(p.cond_Kv_r), rho_fr) - G * p.cond_H * rho_col

        # ---- pressure chain of the intermediate side, from the P_i sensor (receiver outlet)
        # upstream: condensate drain, condenser, the header from the condenser back to the
        # valve 2 branch (taken halfway along the header), which is the header pressure P_h
        tr_i, tr_d, tr_s = pr.transport(P_i), pr.transport(P_d), pr.transport(P_s)
        mu_co = tr_i["mu_l"] * (1.0 - dry) + tr_i["mu_v"] * dry
        dP_drn = self._pipe(p, "drn", mdot_cr, rho_co, mu_co)
        rho_hg = pr.vapor_props(P_i + dP_drn + dP_cr, h_d)[1]      # header gas
        dP_hdr2 = self._pipe(p, "hdr", mdot_cr, rho_hg, tr_i["mu_v"], 0.5)
        P_h = np.maximum(P_i + dP_drn + dP_cr + dP_hdr2, pr.p_min * 1.001)   # hot gas header at the branch

        # ---- suction side: P_s is the compressor suction port; the tee sits above it by the
        # suction line's drop (compressor flow, the line's own mixture)
        mu_l1 = mixture_viscosity(C["x"][:, n], tr_s["mu_l"], tr_s["mu_v"])
        dP_suc = self._pipe(p, "suc", mdot_c, C["rho"][:, n], mu_l1)
        P_tee = P_s + dP_suc

        # ---- valve 1: discharge -> header branch, in series with the discharge line and the
        # first half of the header (equivalent resistance at the valve-alone flow)
        T_g, rho_g = pr.vapor_props(P_h, h_d)                    # header gas after throttling
        f1 = valve_characteristic(u1, p.dpv_char, p.dpv_R)
        C1 = kv_to_C(p.Kv_dpv) * f1
        pipes1 = lambda m: (self._pipe(p, "dis", m, D.rho, tr_d["mu_v"]),
                            self._pipe(p, "hdr", m, rho_g, tr_i["mu_v"], 0.5))
        m0 = gas_valve_flow(C1, P_d, P_h, D.rho, rho_g, p.kappa, p.xT, p.eps_valve)
        mdot_1 = gas_valve_flow(C1, P_d, P_h, D.rho, rho_g, p.kappa, p.xT, p.eps_valve,
                                self._equiv_C(m0, sum(pipes1(m0)), rho_g), rho_g)
        dP_dis, dP_hdr1 = pipes1(mdot_1)
        h_1f = np.where(mdot_1 >= 0.0, h_d, h_iv)

        # ---- valve 2: header branch -> bypass line -> mixing exchanger gas side (S1 at the
        # bottom, rising to S2) -> outlet leg -> tee, all in series
        f2 = valve_characteristic(u2, p.spv_char, p.spv_R)
        C2 = kv_to_C(p.Kv_spv) * f2
        rho_gin = pr.vapor_props(P_s, h_d)[1]                    # bypass gas after throttling
        rho_gm = 2.0 / (1.0 / rho_gin + 1.0 / sat_s["rho_v"])
        head_g = G * p.mx_H * rho_gm
        C_mg = kv_to_C(p.mx_Kv_g)
        pipes2 = lambda m: (self._pipe(p, "bp", m, rho_gin, tr_s["mu_v"]), hx_drop(m, C_mg, rho_gm),
                            self._pipe(p, "mo", m, sat_s["rho_v"], tr_s["mu_v"], 0.5))
        m0 = gas_valve_flow(C2, P_h, P_tee + head_g, rho_g, C["rho"][:, n], p.kappa, p.xT, p.eps_valve)
        mdot_2 = gas_valve_flow(C2, P_h, P_tee + head_g, rho_g, C["rho"][:, n], p.kappa, p.xT, p.eps_valve,
                                self._equiv_C(m0, sum(pipes2(m0)), rho_gm), rho_gm)
        dP_bp, dP_mg_f, dP_mog = pipes2(mdot_2)
        dP_mg = dP_mg_f + head_g                                 # S1 above S2
        m_from_inlet = clip(np.minimum(mdot_2, np.maximum(mdot_1, 0.0)), 0.0, np.inf)
        h_2f_fwd = (m_from_inlet * h_d + (np.maximum(mdot_2, 0.0) - m_from_inlet) * h_iv) \
            / np.maximum(mdot_2, 1e-12)
        fwd2 = mdot_2 > 0.0
        h_2f = np.where(fwd2, h_2f_fwd, h_l1)

        # ---- valve 3: receiver / liquid line -> mixing exchanger quench side (S3 on top), in
        # series with it (distributor, ports, channels; see _quench_drops).  The drop is not
        # quadratic in the flow, so the series pair is solved with the side's equivalent
        # resistance at the valve-alone flow (the valve takes almost all of the difference).
        # Each cell adds its static head and boils at its own pressure; the distributor and
        # the S3 port are ahead of the channels and do not raise the boiling pressure.
        f3 = valve_characteristic(u3, p.stv_char, p.stv_R)
        rho_c, x_c = C["rho"][:, :n], C["x"][:, :n]
        head_q = G * (p.mx_H / n)[:, None] * rho_c
        S3 = pr.state(P_s, h_cv_out)                             # valve 3 outlet, flashed
        rho_3 = S3.rho
        mu_3 = mixture_viscosity(S3.x, tr_s["mu_l"], tr_s["mu_v"])
        mu_qo = mixture_viscosity(x_c[:, n - 1], tr_s["mu_l"], tr_s["mu_v"])
        C3 = kv_to_C(p.Kv_stv) * f3
        P_3dn = P_tee - head_q.sum(1)

        def drops3(m):                    # liquid line, quench line, quench side, outlet leg
            dq = self._quench_drops(p, m, rho_3, rho_c, x_c, sat_s, tr_s)
            dq["liq"] = self._pipe(p, "liq", m, rho_co, mu_co)
            dq["q"] = self._pipe(p, "q", m, rho_3, mu_3)
            dq["mo"] = self._pipe(p, "mo", m, rho_c[:, n - 1], mu_qo, 0.5)
            return dq, dq["liq"] + dq["q"] + dq["inlet"] + dq["seg"].sum(1) + dq["outlet"] + dq["mo"]

        m0 = liquid_valve_flow(C3, P_i, P_3dn, rho_co, rho_c[:, 0], p.f_choke_liq, p.eps_valve)
        mdot_3 = liquid_valve_flow(C3, P_i, P_3dn, rho_co, rho_c[:, 0], p.f_choke_liq, p.eps_valve,
                                   self._equiv_C(m0, drops3(m0)[1], rho_3), rho_3)
        fwd3 = mdot_3 >= 0.0
        dq, _ = drops3(mdot_3)
        seg_q = dq["seg"] - head_q                               # P(top of cell) - P(bottom of cell)
        # cell centres above the compressor port: suction line, outlet leg, S4 port, cells below
        dP_qc = (dP_suc + dq["mo"] + dq["outlet"])[:, None] + np.cumsum(seg_q[:, ::-1], 1)[:, ::-1] - 0.5 * seg_q
        dP_mq = dq["inlet"] + seg_q.sum(1) + dq["outlet"]       # S3 (ahead of the distributor) above S4
        dP_mq_ch = seg_q.sum(1)                                  # across the channels
        # boiling temperature shift of each cell (Clausius-Clapeyron at P_s)
        dTs = (sat_s["T_v"] * (1.0 / sat_s["rho_v"] - 1.0 / sat_s["rho_l"])
               / np.maximum(sat_s["h_v"] - sat_s["h_l"], 1e3))[:, None] * dP_qc

        # ---- walls of suction and discharge lines, receiver shell
        Q_sg = p.UA_sg * (T_sw - C["T"][:, n])           # suction line wall -> refrigerant
        dT_sw = (p.UA_sa * (T_amb - T_sw) - Q_sg) / p.C_sw
        Q_dg = p.UA_dg * (T_dw - D.T)                    # wall -> discharge gas
        dT_dw = (p.UA_da * (T_amb - T_dw) - Q_dg) / p.C_dw
        Q_rw = p.rec_UA_r * (T_rw - I.T)                 # receiver shell -> refrigerant
        dT_rw = (p.rec_UA_a * (T_amb - T_rw) - Q_rw) / p.C_rw

        # ---- condenser, water side.  Water enters at the liquid end: it first subcools the
        # leaving liquid (flooded zone, or the draining condensate film on cond_sc_film of
        # the area), then cools the wall.  The plant loop's supply-to-return difference
        # drives it through valve 4, the piping and the condenser in series.
        f_w = valve_characteristic(u4, p.w_char, p.w_R)
        C_cw = kv_to_C(p.cond_Kv_w)
        mdot_w = water_valve_flow(series_C(kv_to_C(p.Kv_w) * f_w, C_cw, kv_to_C(p.Kv_wpipe)),
                                  p.P_w_sup - p.P_w_ret, p.rho_w)
        dP_cw = hx_drop(mdot_w, C_cw, p.rho_w)
        Cw = mdot_w * p.cp_w
        UA_w = p.alpha_w0 * p.cond_A * np.power(np.maximum(mdot_w / p.mdot_w_ref, 1e-6), 0.8)
        a_sc = np.maximum(cond_flood, p.cond_sc_film) * (1.0 - dry)
        T_l = np.where(I.x < 0.0, I.T, I.T_sat)                   # liquid entering the zone
        C_l = np.maximum(mdot_3, 0.0) * I.cp_l
        C_min, C_max = np.minimum(C_l, Cw), np.maximum(C_l, Cw)
        UA_sc0 = p.alpha_sc * p.cond_A
        UA_sc = a_sc * UA_sc0 * UA_w / (UA_sc0 + UA_w)             # liquid side in series with water side
        eps_sc = counterflow_effectiveness(UA_sc / np.maximum(C_min, 1e-9), C_min / np.maximum(C_max, 1e-9))
        Q_sc = eps_sc * C_min * np.maximum(T_l - T_wi, 0.0)       # liquid -> water
        h_co = h_cv_out - Q_sc / np.maximum(mdot_3, 1e-9)
        T_co = (T_l - Q_sc / np.maximum(C_l, 1e-9)) * (1.0 - dry) + I.T * dry
        SC = I.T_sat - T_co
        h_3i = np.where(fwd3, h_cv_out, H[:, 0])                  # leaving the intermediate section
        h_3f = np.where(fwd3, h_co, H[:, 0])                      # entering the quench side
        T_w1 = T_wi + Q_sc / np.maximum(Cw, 1e-9)        # water leaving the subcooled zone
        eps_w = 1.0 - np.exp(-UA_w * (1.0 - a_sc) / np.maximum(Cw, 1e-9))
        Q_ww = eps_w * Cw * (T_cw - T_w1)                # wall -> water
        Q_w = Q_sc + Q_ww                                # water duty
        T_wo = T_wi + np.where(Cw > 1e-9, Q_w / np.maximum(Cw, 1e-9), 0.0)
        dT_cw = (Q_r - Q_ww + p.UA_ca * (T_amb - T_cw)) / p.C_cw

        # ---- mixing exchanger, gas side: quasi-steady march from S1 (bottom cell n-1)
        # up to S2 against the plate walls; the gas does not condense (walls below
        # saturation are clamped)
        m2p = np.maximum(mdot_2, 0.0)
        Tw_eff = np.maximum(T_mw, T_sat_s[:, None])
        # vapor at the wall temperature (linearized from the dew point; an effectiveness target)
        h_eq = h_vs[:, None] + sat_s["cp_v"][:, None] * np.maximum(Tw_eff - sat_s["T_v"][:, None], 0.0)
        UA_g = p.mx_alpha_g0 * p.A_mx_cell * np.power(
            np.maximum(m2p, 1e-3 * p.mx_mdot_g_ref) / p.mx_mdot_g_ref, 0.8)
        hg = h_2f.copy()
        Q_g = np.zeros((nb, n))
        H_g = np.zeros((nb, n))
        for j in range(n - 1, -1, -1):
            T_in = pr.T_vapor(P_s, hg)
            dT = T_in - Tw_eff[:, j]
            dh = hg - h_eq[:, j]
            big = np.abs(dT) > 0.05
            cp_s = clip(np.where(big, dh / np.where(big, dT, 1.0), 1100.0), 300.0, 1e5)
            Q = (1.0 - np.exp(-UA_g / np.maximum(m2p * cp_s, 1e-9))) * m2p * dh
            hg = hg - Q / np.maximum(m2p, 1e-12)
            Q_g[:, j], H_g[:, j] = Q, hg
        h_go = hg                                        # leaving at S2
        T_gc, rho_gc, rP_gc = self._gas_cells(P_s, H_g)
        self._hg_last = H_g

        # ---- mixing exchanger, quench side: plate -> quench, by flow regime
        m3p = np.maximum(mdot_3, 0.0)
        fq = np.maximum(m3p / p.mx_mdot_q_ref, 0.04)
        UA_e = p.mx_alpha_e * p.A_mx_cell * np.sqrt(fq)
        UA_v = p.mx_alpha_v0 * p.A_mx_cell * np.power(fq, 0.8)
        x_q, T_q = C["x"][:, :n], C["T"][:, :n]
        # stream entering each cell (upwind): valve 3 / the cell above, or from below on reverse flow
        H_up = np.where(fwd3[:, None], np.concatenate([h_3f[:, None], H[:, :n - 1]], 1),
                        np.concatenate([H[:, 1:n], h_l1[:, None]], 1))
        # A cell still holding liquid boils over its whole area.  In the cell where the
        # quench dries out, the evaporating share phi of the area is what the incoming
        # liquid needs at the evaporating coefficient; the rest heats vapor.  Blended over
        # DRYOUT_BAND above the dew point, so the dry-out point moves continuously
        # through the cells and the vapor never leaves hotter than its wall.
        dT_sat = T_mw - (T_sat_s[:, None] + dTs)
        Q_2ph = UA_e[:, None] * dT_sat
        need = np.abs(mdot_3)[:, None] * np.maximum(h_vs[:, None] - H_up, 0.0)
        phi_f = clip(need / np.maximum(UA_e[:, None] * np.maximum(dT_sat, 0.0), 1e-9), 0.0, 1.0)

        def quench_heat(Hq, Tq):
            w_wet = 1.0 - smoothstep((Hq - h_vs[:, None]) / self.DRYOUT_BAND)
            Tq = Tq + w_wet * dTs                        # liquid boils at the cell's own pressure
            # ... never more than the share of the cell's enthalpy rise below the dew point
            phi_h = clip((h_vs[:, None] - H_up) / np.maximum(Hq - H_up, 1.0), 0.0, 1.0)
            phi = np.minimum(phi_f, np.where(Hq > H_up, phi_h, 1.0))
            Q_dry = phi * Q_2ph + (1.0 - phi) * UA_v[:, None] * (T_mw - Tq)
            return w_wet * UA_e[:, None] * (T_mw - Tq) + (1.0 - w_wet) * Q_dry     # wall -> quench

        Q_q = quench_heat(H[:, :n], T_q)
        dT_mw = (Q_g - Q_q + (p.mx_UA_a / n)[:, None] * (T_amb[:, None] - T_mw)) / p.C_mw_cell[:, None]

        # ---- lumped-pressure balances of the suction side.  In every cell
        #   M dh/dt = F_in (h_in - h) + Q + V dP/dt,   dM/dt = V (rho_P dP/dt + rho_h dh/dt)
        # and the flow leaving a cell is F_in - dM/dt; all are affine in dP/dt, which
        # follows from the compressor drawing mdot_c out of the last cell.  For dP/dt the
        # enthalpy rates respond to the pressure only through the compression term V/M
        # (the inflows are taken at their dP/dt = 0 values): every cell then contributes
        # its isentropic capacitance V (rho_P + rho_h / rho) > 0, so the pressure solve
        # stays well posed even where cold liquid enters a vapor-filled cell.
        rho_P, rho_h = C["drho_dP"], C["drho_dh"]
        c = np.zeros((nb, n + 2))
        d = np.zeros((nb, n + 2))
        a_f, b_f = mdot_3.copy(), np.zeros(nb)           # flow entering the current quench cell
        A_face, B_face = np.zeros((nb, n + 1)), np.zeros((nb, n + 1))
        A_face[:, 0] = mdot_3
        for j in range(n):
            h_up = np.where(fwd3, h_3f if j == 0 else H[:, j - 1], H[:, j + 1] if j < n - 1 else h_l1)
            adv_a = np.where(fwd3, a_f, -mdot_3)
            c[:, j] = (adv_a * (h_up - H[:, j]) + Q_q[:, j]) / M_c[:, j]
            d[:, j] = Vc[:, j] / M_c[:, j]
            a_f = a_f - Vc[:, j] * rho_h[:, j] * c[:, j]
            b_f = b_f - Vc[:, j] * (rho_P[:, j] + rho_h[:, j] * d[:, j])
            A_face[:, j + 1], B_face[:, j + 1] = a_f, b_f
        Fq_a, Fq_b = a_f, b_f                            # quench outlet (S4) -> tee
        Fg_a, Fg_b = mdot_2, -(p.V_gc * rP_gc).sum(axis=1)     # gas outlet (S2) -> tee
        wq, wg = fwd3.astype(float), fwd2.astype(float)
        dhq, dhg = H[:, n - 1] - h_l1, h_go - h_l1
        c[:, n] = (wg * Fg_a * dhg + wq * Fq_a * dhq + Q_sg) / M_c[:, n]
        d[:, n] = Vc[:, n] / M_c[:, n]
        F12_a = Fg_a + Fq_a - Vc[:, n] * rho_h[:, n] * c[:, n]
        F12_b = Fg_b + Fq_b - Vc[:, n] * (rho_P[:, n] + rho_h[:, n] * d[:, n])
        w12 = (F12_a > 0.0).astype(float)
        c[:, n + 1] = (w12 * F12_a * (h_l1 - h_l2) - mdot_c * (h_cin - h_l2)) / M_c[:, n + 1]
        d[:, n + 1] = Vc[:, n + 1] / M_c[:, n + 1]
        Fo_a = F12_a - Vc[:, n + 1] * rho_h[:, n + 1] * c[:, n + 1]
        Fo_b = F12_b - Vc[:, n + 1] * (rho_P[:, n + 1] + rho_h[:, n + 1] * d[:, n + 1])
        dP_s = (mdot_c - Fo_a) / np.minimum(Fo_b, -1e-15)
        # With dP/dt known, every face flow follows; the enthalpy rates are evaluated
        # again with the upwind side chosen by each face's actual direction (a fast
        # pressure rise can briefly reverse the flow between quench cells).
        Pr = np.zeros(nb) if hold_P_s else dP_s
        F = A_face + B_face * Pr[:, None]                # into quench cell j through its top face j
        Fg_now = Fg_a + Fg_b * Pr
        F12 = F12_a + F12_b * Pr
        h_above = np.concatenate([h_3f[:, None], H[:, :n - 1]], 1)
        h_below = np.concatenate([H[:, 1:n], h_l1[:, None]], 1)
        dH = np.empty((nb, n + 2))
        dH[:, :n] = (np.maximum(F[:, :n], 0.0) * (h_above - H[:, :n])
                     + np.maximum(-F[:, 1:], 0.0) * (h_below - H[:, :n])
                     + Q_q + Vc[:, :n] * Pr[:, None]) / M_c[:, :n]
        dH[:, n] = (np.maximum(Fg_now, 0.0) * dhg + np.maximum(F[:, n], 0.0) * dhq
                    + np.maximum(-F12, 0.0) * (h_l2 - h_l1) + Q_sg + Vc[:, n] * Pr) / M_c[:, n]
        dH[:, n + 1] = (np.maximum(F12, 0.0) * (h_l1 - h_l2) - mdot_c * (h_cin - h_l2)
                        + Vc[:, n + 1] * Pr) / M_c[:, n + 1]

        # ---- fastest local rates (1/s) for the integrator's step subdivision: valve
        # conductance over the capacitance of each pressure node, and advection plus
        # heat transfer of the quench cells
        k1 = _valve_slope(C1, np.maximum(D.rho, rho_g), P_d - P_h, p.eps_valve)
        k2 = _valve_slope(C2, rho_g, P_h - P_s, p.eps_valve)
        k3 = _valve_slope(kv_to_C(p.Kv_stv) * f3, rho_co, P_i - P_s, p.eps_valve)
        C_d = p.V_d * np.maximum(D.drho_dP + D.drho_dh / D.rho, 1e-12)
        C_i = p.V_i * np.maximum(I.drho_dP + I.drho_dh / I.rho, 1e-12)
        C_s = np.maximum(-Fo_b, 1e-12)
        # (heat flow sensitivity to the cell's own enthalpy, by a finite difference of the
        # heat transfer formula; the temperature moves only where the cell holds vapor)
        dh_fd = 200.0
        dQ = quench_heat(H[:, :n] + dh_fd, T_q + dh_fd / 1100.0 * (x_q >= 1.0)) - Q_q
        F_cell = np.maximum(np.abs(F[:, :n]), np.abs(F[:, 1:]))
        lam_q = ((F_cell + np.abs(dQ) / dh_fd) / M_c[:, :n]).max(1)
        self._lam_last = np.maximum.reduce([k1 / C_d, (k1 + k2 + k3) / C_i,
                                            (k2 + k3 + mdot_c / P_s) / C_s, lam_q])

        # ---- tee and suction line: droplets from the quench outlet evaporate on the way to
        # the probe; the probe reads the vapor temperature
        Fq = np.maximum(Fq_a + Fq_b * dP_s, 0.0) * wq
        Fg = np.maximum(Fg_a + Fg_b * dP_s, 0.0) * wg
        F_in = Fq + Fg
        x_qo = x_q[:, n - 1]
        y_in = clip(1.0 - x_qo, 0.0, 1.0) * Fq / np.maximum(F_in, 1e-9)
        t_res = M_c[:, n] / np.maximum(F_in, 1e-6)
        y_eq = clip(1.0 - C["x"][:, n], 0.0, 1.0)
        y_liq = np.maximum(y_in * np.exp(-t_res / p.tee_tau_evap), y_eq)
        h_vap = np.where(y_liq > 1e-12, (h_l1 - y_liq * h_ls) / np.maximum(1.0 - y_liq, 1e-3), h_l1)
        T_port = np.where(y_liq < 0.999, pr.T_vapor(P_s, h_vap), C["T"][:, n])
        x_out = np.where(y_liq > 0.0, 1.0 - y_liq, C["x"][:, n])

        # ---- compressor shell
        dT_sh = (cp["Q_gs"] + (1.0 - p.f_motor_gas) * cp["Q_motor"]
                 - p.UA_sha * (T_sh - T_amb)) / p.C_shell

        # ---- lumped volumes with their own pressure
        dm_d = mdot_c - mdot_1
        E_d = mdot_c * (h2 - h_d) - mdot_1 * (h_1f - h_d) + Q_dg
        dP_d, dh_d = cv_balance(p.V_d, D.rho, D.drho_dP, D.drho_dh, dm_d, E_d)
        dm_i = mdot_1 - mdot_2 - mdot_3
        E_i = mdot_1 * (h_1f - h_i) - mdot_2 * (h_2f - h_i) - mdot_3 * (h_3i - h_i) - Q_r + Q_rw
        dP_i, dh_i = cv_balance(p.V_i, I.rho, I.drho_dP, I.drho_dh, dm_i, E_i)

        # ---- speed, actuators, sensors
        Nc = np.where(N_cmd > 0.0, clip(N_cmd, p.N_min, p.N_max), 0.0)
        dN = clip((Nc - N) / p.tau_N, -p.ramp_N, p.ramp_N)

        dx = np.zeros_like(x)
        dx[:, self.P_S] = dP_s
        dx[:, self.HS] = dH
        dx[:, self.TMW] = dT_mw
        dx[:, self.T_SW] = dT_sw
        dx[:, self.P_D], dx[:, self.H_D], dx[:, self.T_DW] = dP_d, dh_d, dT_dw
        dx[:, self.P_I], dx[:, self.H_I], dx[:, self.T_CW], dx[:, self.T_RW] = dP_i, dh_i, dT_cw, dT_rw
        dx[:, self.T_SH], dx[:, self.N_] = dT_sh, dN
        dx[:, self.U1] = actuator_rate(u1, u_cmd[:, 0], p.tau_dpv, p.rate_dpv)
        dx[:, self.U2] = actuator_rate(u2, u_cmd[:, 1], p.tau_spv, p.rate_spv)
        dx[:, self.U3] = actuator_rate(u3, u_cmd[:, 2], p.tau_stv, p.rate_stv)
        dx[:, self.U4] = actuator_rate(u4, u_cmd[:, 3], p.tau_w, p.rate_w)
        dx[:, self.TM_S] = (T_port - Tm_s) / p.tau_T
        dx[:, self.TM_D] = (D.T - Tm_d) / p.tau_T
        dx[:, self.TM_CO] = (T_co - Tm_co) / p.tau_T
        dx[:, self.MM] = (mdot_c - mm) / p.tau_m
        dx[:, self.WM] = (cp["W_el"] - Wm) / p.tau_W
        # conserved states: net mass flow and sum(m h) + Q over each boundary
        dx[:, self.M_S] = mdot_2 + mdot_3 - mdot_c
        dx[:, self.U_S] = (mdot_2 * h_2f + mdot_3 * h_3f - mdot_c * h_cin + Q_sg
                           + Q_q.sum(axis=1) - Q_g.sum(axis=1))
        dx[:, self.M_D] = dm_d
        dx[:, self.U_D] = E_d + h_d * dm_d
        dx[:, self.M_I] = dm_i
        dx[:, self.U_I] = E_i + h_i * dm_i
        if not want_aux:
            return dx

        M_g = (rho_gc * p.V_gc).sum(axis=1)
        M_s = M_c.sum(axis=1) + M_g
        M_d = D.rho * p.V_d
        aux = dict(
            t=self.t, P_s=P_s, P_d=P_d, P_i=P_i, T_s=T_port, T_d=D.T, T_i=I.T,
            T_co=T_co, T_sat_s=T_sat_s, T_sat_d=D.T_sat, T_sat_i=I.T_sat,
            SH=T_port - T_sat_s, SC=SC, x_out=x_out, y_liq=y_liq, x_l1=C["x"][:, n], x_i=I.x,
            x_qo=x_qo, T_qo=T_q[:, n - 1], T_go=pr.T_vapor(P_s, h_go), h_go=h_go, T_l1=C["T"][:, n],
            fill_i=fill_i, rec_level=rec_level, ll_fill=ll_fill, cond_flood=cond_flood,
            M_q_liq=((1.0 - clip(x_q, 0.0, 1.0)) * M_c[:, :n]).sum(axis=1),
            h_q=H[:, :n], x_q=x_q,
            T_q=T_q + (1.0 - smoothstep((H[:, :n] - h_vs[:, None]) / self.DRYOUT_BAND)) * dTs, T_g=T_gc, h_g=H_g, T_mw=T_mw,
            Q_mx=Q_g.sum(axis=1), Q_q=Q_q.sum(axis=1),
            rho_s=Cin.rho, h_l1=h_l1, h_l2=h_l2, h_cin=h_cin, x_l2=C["x"][:, n + 1], h_d=h_d, h_i=h_i, h_co=h_co, h2=h2,
            h_2f=h_2f, h_3f=h_3f, T2_ad=cp["T2_ad"],
            P_tee=P_tee, dP_suc=dP_suc, dP_dis=dP_dis, dP_hdr=dP_hdr1 + dP_hdr2, dP_bp=dP_bp, dP_mog=dP_mog,
            dP_moq=dq["mo"], dP_q=dq["q"], dP_liq=dq["liq"], dP_drn=dP_drn,
            P_h=P_h, dP_cr=dP_cr, dP_cw=dP_cw, dP_mg=dP_mg, dP_mq=dP_mq, dP_mq_ch=dP_mq_ch, dP_qc=dP_qc,
            dP_q_dist=dq["dist"], dP_q_in=dq["inlet"], dP_q_out=dq["outlet"], mdot_cr=mdot_cr,
            mdot_c=mdot_c, mdot_1=mdot_1, mdot_2=mdot_2, mdot_3=mdot_3, mdot_w=mdot_w,
            W_el=cp["W_el"], W_shaft=cp["W_shaft"], Pr=cp["Pr"], eta_v=cp["eta_v"], eta_s=cp["eta_s"],
            Q_r=Q_r, Q_w=Q_w, Q_sc=Q_sc, Q_sg=Q_sg, Q_dg=Q_dg, Q_rw=Q_rw, T_wo=T_wo, T_sw=T_sw, T_dw=T_dw,
            T_cw=T_cw, T_rw=T_rw, T_sh=T_sh, N=N, u1=u1, u2=u2, u3=u3, u4=u4,
            M_s=M_s, M_g=M_g, M_d=M_d, M_i=M_i, M_tot=M_s + M_d + M_i, charge=p.charge,
            Tm_s=Tm_s, Tm_d=Tm_d, Tm_co=Tm_co, mm=mm, Wm=Wm, dx=dx,
        )
        return dx, aux

    # ------------------------------------------------------------ integrate
    def _post(self, x, p=None):
        p = self.p if p is None else p
        x[:, self.N_] = np.maximum(x[:, self.N_], 0.0)
        x[:, self.UV] = clip(x[:, self.UV], 0.0, 1.0)
        for iP, ih, iM, iU, vname in self.CV:
            V = getattr(p, vname)
            M = x[:, iM]
            P, h = self._flash_rho_u(M / V, x[:, iU] / M, x[:, iP])
            x[:, iP] = P
            x[:, ih] = h
        self._project_suction(x, p)
        return x

    def _suction_inventory(self, x, p, delta=None):
        """Mass, internal energy and their derivatives with respect to the common
        pressure and a uniform enthalpy shift of the dynamic suction cells."""
        P = x[:, self.P_S]
        H = x[:, self.HS] + (0.0 if delta is None else delta[:, None])
        Hg = x[:, self.HG]
        C = self._suction_cells(P, H)
        _, rho_g, rP_g = self._gas_cells(P, Hg)
        V, Vg = p.V_sc, p.V_gc
        M = (C["rho"] * V).sum(1) + (rho_g * Vg).sum(1)
        U = (C["rho"] * V * H).sum(1) + (rho_g * Vg * Hg).sum(1) - P * p.V_s
        J = dict(
            MP=(C["drho_dP"] * V).sum(1) + (rP_g * Vg).sum(1),
            Md=(C["drho_dh"] * V).sum(1),
            UP=(C["drho_dP"] * V * H).sum(1) + (rP_g * Vg * Hg).sum(1) - p.V_s,
            Ud=((C["rho"] + C["drho_dh"] * H) * V).sum(1),
        )
        return M, U, J, C

    def _project_suction(self, x, p, max_iter: int = 4) -> None:
        """Correct the common suction pressure and shift the dynamic cell
        enthalpies uniformly so that the suction side holds exactly its
        conserved mass and energy (Newton; mass-only step where the joint
        step is ill-conditioned)."""
        Mt, Ut = x[:, self.M_S], x[:, self.U_S]
        delta = np.zeros(x.shape[0])
        for _ in range(max_iter):
            M, U, J, _ = self._suction_inventory(x, p, delta)
            rM, rU = M - Mt, U - Ut
            if (np.abs(rM) <= 1e-9 * Mt).all() and (np.abs(rU) <= 1e-9 * np.abs(Ut) + 1e-3).all():
                break
            det = J["MP"] * J["Ud"] - J["Md"] * J["UP"]
            dP = (rM * J["Ud"] - rU * J["Md"]) / np.where(det != 0.0, det, 1.0)
            dd = (J["MP"] * rU - J["UP"] * rM) / np.where(det != 0.0, det, 1.0)
            P = x[:, self.P_S]
            ok = (det > 0.0) & (np.abs(dP) < 0.05 * P) & (np.abs(dd) < 2e4)
            dP = np.where(ok, dP, rM / J["MP"])
            dd = np.where(ok, dd, 0.0)
            x[:, self.P_S] = clip(P - dP, self.props.p_min * 1.001, self.props.p_max * 0.999)
            delta = delta - dd
        x[:, self.HS] += delta[:, None]

    def _flash_rho_u(self, rho_t, u_t, P0, max_iter: int = 8):
        """(P, h) of a volume with density ``rho_t`` and internal energy ``u_t``.

        The energy constraint fixes h = u + P / rho, and along that path the
        density is a monotonic function of P in every phase region, so a
        bracketed Newton iteration on P converges to the unique solution
        (the previous pressure ``P0`` is the starting point).
        """
        pr = self.props
        lo = np.full_like(P0, pr.p_min * 1.001)
        hi = np.full_like(P0, pr.p_max * 0.999)
        P = clip(P0, lo, hi)
        for _ in range(max_iter):
            h = u_t + P / rho_t
            S = pr.state(P, h)
            r = S.rho - rho_t
            conv = np.abs(r) <= 1e-5 * rho_t
            if conv.all():
                break
            hi = np.where(r > 0.0, np.minimum(hi, P), hi)
            lo = np.where(r < 0.0, np.maximum(lo, P), lo)
            slope = np.maximum(S.drho_dP + S.drho_dh / rho_t, 1e-12)
            Pn = P - r / slope
            outside = ~np.isfinite(Pn) | (Pn <= lo) | (Pn >= hi)
            Pn = np.where(outside, 0.5 * (lo + hi), Pn)
            P = np.where(conv, P, Pn)
        return P, u_t + P / rho_t

    def _sync_mass(self, x, idx=None) -> None:
        """Set the conserved mass and energy states from (P, h) (after direct
        state assignments); ``idx`` selects the environments of the rows of ``x``."""
        p = self.p if idx is None else self.params_subset(idx)
        for iP, ih, iM, iU, vname in self.CV:
            V = getattr(p, vname)
            S = self.props.state(x[:, iP], x[:, ih])
            x[:, iM] = S.rho * V
            x[:, iU] = x[:, iM] * (x[:, ih] - x[:, iP] / S.rho)
        M, U, _, _ = self._suction_inventory(x, p)
        x[:, self.M_S], x[:, self.U_S] = M, U

    def _stiff(self, x, p=None):
        """Per environment: True if a volume is (nearly) liquid-full: subcooled
        liquid, or a two-phase mixture whose liquid volume fraction exceeds
        ``STIFF_FILL`` (lever rule on the saturation properties); the suction
        side counts as one volume."""
        p = self.p if p is None else p
        pr = self.props

        def fill_of(P, h):
            sat = pr.sat(P)
            xq = (h - sat["h_l"]) / (sat["h_v"] - sat["h_l"])
            return xq, (1.0 - xq) / (1.0 + xq * (sat["rho_l"] / sat["rho_v"] - 1.0))

        stiff = np.zeros(x.shape[0], bool)
        for iP, ih, _, _, _ in self.CV:
            xq, fill = fill_of(x[:, iP], x[:, ih])
            stiff |= (xq < 0.0) | (fill > self.STIFF_FILL)
        k = self.MX + 2
        _, fill = fill_of(np.repeat(x[:, self.P_S], k), x[:, self.HS].reshape(-1))
        liq = (np.clip(fill.reshape(-1, k), 0.0, 1.0) * p.V_sc).sum(1) / p.V_s
        return stiff | (liq > self.STIFF_FILL)

    def _subdivisions(self, x):
        """Sub-steps each environment needs for the next step (powers of two):
        its fastest local rate (from the last right-hand side evaluation) times
        the sub-step must stay inside RK4's stable range; liquid-full volumes
        take at least ``STIFF_SUBDIV``."""
        lam = np.where(np.isfinite(self._lam), self._lam, np.inf)
        k = np.ceil(np.minimum(lam * self.dt / self.LAM_DT_MAX, self.MAX_SUBDIV))
        k = np.where(self._stiff(x), np.maximum(k, self.STIFF_SUBDIV), k)
        k = np.clip(k, 1, self.MAX_SUBDIV)
        return np.minimum(2 ** np.ceil(np.log2(k)), self.MAX_SUBDIV).astype(int)

    def _advance(self, x, args, k: int, p, depth: int = 0):
        """One step of ``self.dt`` in ``k`` sub-steps for the environments in
        ``x`` (parameters ``p``).  A sub-step that produces a non-finite state or
        an implausible jump (the local-rate estimate can miss a nonlinear
        transient) is rejected and redone with four times finer steps."""
        dt0 = self.dt
        self.dt = dt0 / k
        try:
            for _ in range(k):
                x_new = self._step_once(x.copy(), args, p)
                if depth < 2 and not self._plausible(x, x_new):
                    x_new = self._advance(x, args, 4, p, depth + 1)
                x = x_new
        finally:
            self.dt = dt0
        return x

    def _plausible(self, x0, x1) -> bool:
        if not np.isfinite(x1).all():
            return False
        dh = np.abs(x1[:, self.HS] - x0[:, self.HS]).max()
        dP = max(np.abs(x1[:, i] / x0[:, i] - 1.0).max() for i in (self.P_S, self.P_D, self.P_I))
        return bool(dh < self.MAX_DH_STEP and dP < self.MAX_DP_STEP)

    def _step_once(self, x, args, p=None):
        dt = self.dt
        f = (lambda xx, *a: self.rhs(xx, *a, p=p))
        if self.integrator == "rk4":
            k1 = f(x, *args)
            k2 = f(x + 0.5 * dt * k1, *args)
            k3 = f(x + 0.5 * dt * k2, *args)
            k4 = f(x + dt * k3, *args)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif self.integrator == "heun":
            k1 = f(x, *args)
            k2 = f(x + dt * k1, *args)
            x = x + 0.5 * dt * (k1 + k2)
        elif self.integrator == "euler":
            x = x + dt * f(x, *args)
        else:
            raise ValueError(f"unknown integrator {self.integrator!r}")
        x[:, self.HG] = self._hg_last            # gas side profile of the last stage
        return self._post(x, p)

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
        """Advance the plant by ``duration`` seconds (or ``n_sub`` sub-steps)
        holding the inputs constant.  Returns the auxiliary output dict
        evaluated at the new state."""
        self.set_inputs(u_cmd, N_cmd, T_amb, T_wi)
        if n_sub is None:
            n_sub = max(1, int(round((duration if duration is not None else self.dt) / self.dt)))
        args = (self.u_cmd, self.N_cmd, self.T_amb, self.T_wi)
        x = self.x
        if self.aux is None or self._lam is None or len(self._lam) != self.n:
            self.rhs(x, *args)               # state was set from outside: fresh rate estimate
            self._lam = self._lam_last
        for _ in range(n_sub):
            k = self._subdivisions(x)
            levels = np.unique(k)
            if len(levels) == 1:
                x = self._advance(x, args, int(levels[0]), self.p)
                self._lam = self._lam_last
                continue
            # environments grouped by the sub-steps they need, each group on its own
            x_new, lam = x.copy(), self._lam.copy()
            for kk in levels:
                idx = np.flatnonzero(k == kk)
                x_new[idx] = self._advance(x[idx], tuple(a[idx] for a in args), int(kk),
                                           self.params_subset(idx))
                lam[idx] = self._lam_last
            x, self._lam = x_new, lam
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
        x[:, self.P_D], x[:, self.H_D] = P, h_d
        x[:, self.P_I], x[:, self.H_I] = P, h_i
        for i in (self.T_SW, self.T_DW, self.T_CW, self.T_RW, self.T_SH, self.TM_S, self.TM_D, self.TM_CO):
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
