"""Dynamic model of a hot-gas-bypass (HGBP) compressor test stand.

Topology (the stand being simulated)
------------------------------------
::

    compressor discharge --> [discharge volume, P_d] --> valve 1 (discharge pressure)
                                                              |
                                        intermediate header, P_i (condensing pressure)
                                            |                        |
                          path 2: valve 2 (suction pressure,   path 1: brazed-plate condenser
                                  hot gas bypass)                      <-- cooling water (valve 4)
                                            |                        |
                                            |               valve 3 (suction temperature, liquid)
                                            v                        v
                               [suction mixer / accumulator tank, P_s] --> compressor suction

Hot gas leaving the compressor is throttled by valve 1 to the intermediate
pressure.  Part of it bypasses through valve 2 to the suction tank; the rest
condenses in a water-cooled brazed-plate condenser (no receiver) and is
injected through valve 3 as liquid to desuperheat the bypass gas.  Valve 4
meters the cooling water and thereby sets the intermediate (condensing)
pressure.  The suction mixer is a tank that also acts as an accumulator:
liquid separates, the compressor draws vapor from the top, excess liquid is
entrained when the tank fills up.

Manipulated variables (0..1 stem commands, in this order):
    u[0]  valve 1  discharge pressure valve
    u[1]  valve 2  suction pressure (HGBP) valve
    u[2]  valve 3  suction temperature (liquid) valve
    u[3]  valve 4  cooling water valve
Exogenous inputs: compressor speed command, ambient temperature, cooling
water inlet temperature.  Total refrigerant charge is a parameter.

States (per environment, see ``HGBPPlant.STATE_NAMES``)
    P_s, h_s, T_sw      suction tank pressure / mean enthalpy, tank wall temp
    P_d, h_d, T_dw      discharge volume, discharge line wall temp
    P_i, h_i, T_cw      intermediate/condenser volume, condenser wall temp
    T_sh                compressor shell temperature
    N                   compressor speed [rpm]
    u1..u4              actual valve positions
    Tm_s, Tm_d, Tm_co   lagged temperature sensors (suction, discharge, condenser outlet)
    mm, Wm              lagged mass-flow and power sensors

Each lumped volume uses (P, h) as states with the standard mass/energy
balance closure through drho/dP|h and drho/dh|P.  Everything is vectorized
over a batch dimension so many environments integrate at once.
"""
from __future__ import annotations

import numpy as np

from .components import (actuator_rate, clip, compressor, cv_balance, gas_valve_flow,
                         liquid_valve_flow, smoothstep, valve_characteristic)
from .params import PlantParams, nominal_charge, sample_params
from .properties import RefrigerantTables, get_tables


class HGBPPlant:
    STATE_NAMES = ("P_s", "h_s", "T_sw", "P_d", "h_d", "T_dw", "P_i", "h_i", "T_cw",
                   "T_sh", "N", "u1", "u2", "u3", "u4", "Tm_s", "Tm_d", "mm", "Wm", "Tm_co")
    (P_S, H_S, T_SW, P_D, H_D, T_DW, P_I, H_I, T_CW, T_SH, N_, U1, U2, U3, U4,
     TM_S, TM_D, MM, WM, TM_CO) = range(20)
    NX = 20
    NU = 4
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
        self._fill_charge(self.p)
        self.x = np.zeros((self.n, self.NX))
        self.t = 0.0
        self.u_cmd = np.zeros((self.n, self.NU))
        self.N_cmd = np.zeros(self.n)
        self.T_amb = np.full(self.n, 298.15)
        self.T_wi = np.full(self.n, 298.15)
        self.aux: dict | None = None
        self.n_rhs_calls = 0

    # ------------------------------------------------------------ parameters
    def _fill_charge(self, p) -> None:
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
                setattr(ns, k, np.repeat(vv, repeat) if repeat > 1 else vv)
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
        self._fill_charge(new)
        for k, v in vars(new).items():
            if isinstance(v, np.ndarray):
                getattr(self.p, k)[idx] = v

    def nominal_charge(self, idx=None):
        p = self.p if idx is None else self.params_subset(idx)
        return nominal_charge(p, self.props)

    # ----------------------------------------------------------------- RHS
    def rhs(self, x, u_cmd, N_cmd, T_amb, T_wi, want_aux: bool = False, p=None):
        """Time derivative of the state matrix ``x`` (shape (n, NX)).

        ``p`` may override the parameter namespace (used by the steady-state
        solver to evaluate stacked copies of a subset of environments)."""
        p = self.p if p is None else p
        pr = self.props
        self.n_rhs_calls += 1
        P_s, h_s, T_sw = x[:, 0], x[:, 1], x[:, 2]
        P_d, h_d, T_dw = x[:, 3], x[:, 4], x[:, 5]
        P_i, h_i, T_cw = x[:, 6], x[:, 7], x[:, 8]
        T_sh, N = x[:, 9], x[:, 10]
        u1, u2, u3, u4 = x[:, 11], x[:, 12], x[:, 13], x[:, 14]
        Tm_s, Tm_d, mm, Wm, Tm_co = x[:, 15], x[:, 16], x[:, 17], x[:, 18], x[:, 19]

        S = pr.state(P_s, h_s)          # suction tank (mean)
        D = pr.state(P_d, h_d)          # discharge volume
        I = pr.state(P_i, h_i)          # intermediate / condenser volume

        # ---- suction accumulator tank: phase separation and outlet stream
        M_s = S.rho * p.V_s
        fill_s = (1.0 - clip(S.x, 0.0, 1.0)) * M_s / (S.rho_l * p.V_s)
        b = smoothstep((S.x - (1.0 - p.acc_blend_dx)) / p.acc_blend_dx)   # 1: superheated tank
        carry = p.acc_carry_max * smoothstep((fill_s - p.acc_carry_fill0) / (1.0 - p.acc_carry_fill0))
        h_out = S.h_v * (1.0 - b) + h_s * b - carry * (S.h_v - S.h_l)
        O = pr.state(P_s, h_out, need_s=True)          # compressor inlet stream

        # ---- compressor
        cp = compressor(p, pr, P_s, h_out, O.s, O.rho, P_d, N, T_sh)
        mdot_c, h2 = cp["mdot"], cp["h2"]

        # ---- condenser inventory, outlet condition
        M_i = I.rho * p.V_i
        fill_i = (1.0 - clip(I.x, 0.0, 1.0)) * M_i / (I.rho_l * p.V_i)
        dry = 1.0 - smoothstep(fill_i / p.cond_dry_fill)          # 1: no liquid seal
        SC = p.SC_max * smoothstep((fill_i - p.SC_fill0) / (1.0 - p.SC_fill0))
        h_co_liq = np.minimum(I.h_l - p.cp_liq * SC, np.where(I.x < 0.0, h_i, I.h_l))
        h_co = h_co_liq * (1.0 - dry) + h_i * dry
        rho_co = I.rho_l * (1.0 - dry) + I.rho * dry
        h_iv = I.h_v * (1.0 - dry) + h_i * dry                    # vapor phase of the CV
        T_co = np.where(I.x < 0.0, np.minimum(I.T, I.T_sat - SC), I.T_sat - SC) * (1.0 - dry) + I.T * dry

        # ---- valve 1: discharge -> intermediate header
        T_g, rho_g = pr.vapor_props(P_i, h_d)                    # header gas after throttling
        f1 = valve_characteristic(u1, p.dpv_char, p.dpv_R)
        mdot_1 = gas_valve_flow(p.C_dpv * f1, P_d, P_i, D.rho, rho_g, p.kappa, p.xT, p.eps_valve)
        h_1f = np.where(mdot_1 >= 0.0, h_d, h_iv)

        # ---- valve 2: hot gas bypass header -> suction tank
        f2 = valve_characteristic(u2, p.spv_char, p.spv_R)
        mdot_2 = gas_valve_flow(p.C_spv * f2, P_i, P_s, rho_g, S.rho, p.kappa, p.xT, p.eps_valve)
        m_from_inlet = clip(np.minimum(mdot_2, np.maximum(mdot_1, 0.0)), 0.0, np.inf)
        h_2f_fwd = (m_from_inlet * h_d + (np.maximum(mdot_2, 0.0) - m_from_inlet) * h_iv) \
            / np.maximum(mdot_2, 1e-12)
        h_2f = np.where(mdot_2 > 0.0, h_2f_fwd, h_s)

        # ---- valve 3: condenser outlet -> suction tank
        f3 = valve_characteristic(u3, p.stv_char, p.stv_R)
        mdot_3 = liquid_valve_flow(p.C_stv * f3, P_i, P_s, rho_co, S.rho, p.f_choke_liq, p.eps_valve)
        h_3f = np.where(mdot_3 >= 0.0, h_co, h_s)

        # ---- pipe / tank walls
        Q_sg = p.UA_sg * (T_sw - S.T)                    # wall -> tank contents
        dT_sw = (p.UA_sa * (T_amb - T_sw) - Q_sg) / p.C_sw
        Q_dg = p.UA_dg * (T_dw - D.T)                    # wall -> discharge gas
        dT_dw = (p.UA_da * (T_amb - T_dw) - Q_dg) / p.C_dw

        # ---- condenser heat transfer (liquid backing up removes condensing area)
        w2 = (1.0 - smoothstep((I.x - 0.85) / 0.15)) * (1.0 - smoothstep(-I.x / 0.05))
        a_c = w2 * (1.0 - fill_i)
        UA_r = p.UA_r_2ph * a_c + p.UA_r_1ph * (1.0 - a_c)
        Q_r = UA_r * (I.T - T_cw)                        # refrigerant -> wall
        f_w = valve_characteristic(u4, p.w_char, p.w_R)
        mdot_w = p.mdot_w_max * f_w
        Cw = mdot_w * p.cp_w
        UA_w = p.UA_w0 * np.power(np.maximum(f_w, 1e-6), 0.8)
        eps_w = 1.0 - np.exp(-UA_w / np.maximum(Cw, 1e-9))
        Q_w = eps_w * Cw * (T_cw - T_wi)                 # wall -> water
        T_wo = T_wi + np.where(Cw > 1e-9, Q_w / np.maximum(Cw, 1e-9), 0.0)
        dT_cw = (Q_r - Q_w + p.UA_ca * (T_amb - T_cw)) / p.C_cw

        # ---- compressor shell
        dT_sh = (cp["Q_gs"] + (1.0 - p.f_motor_gas) * cp["Q_motor"]
                 - p.UA_sha * (T_sh - T_amb)) / p.C_shell

        # ---- control volume balances
        dm_s = mdot_2 + mdot_3 - mdot_c
        E_s = mdot_2 * (h_2f - h_s) + mdot_3 * (h_3f - h_s) - mdot_c * (h_out - h_s) + Q_sg
        dP_s, dh_s = cv_balance(p.V_s, S.rho, S.drho_dP, S.drho_dh, dm_s, E_s)

        dm_d = mdot_c - mdot_1
        E_d = mdot_c * (h2 - h_d) - mdot_1 * (h_1f - h_d) + Q_dg
        dP_d, dh_d = cv_balance(p.V_d, D.rho, D.drho_dP, D.drho_dh, dm_d, E_d)

        dm_i = mdot_1 - mdot_2 - mdot_3
        E_i = mdot_1 * (h_1f - h_i) - mdot_2 * (h_2f - h_i) - mdot_3 * (h_3f - h_i) - Q_r
        dP_i, dh_i = cv_balance(p.V_i, I.rho, I.drho_dP, I.drho_dh, dm_i, E_i)

        # ---- speed, actuators, sensors
        Nc = np.where(N_cmd > 0.0, clip(N_cmd, p.N_min, p.N_max), 0.0)
        dN = clip((Nc - N) / p.tau_N, -p.ramp_N, p.ramp_N)
        du1 = actuator_rate(u1, u_cmd[:, 0], p.tau_dpv, p.rate_dpv)
        du2 = actuator_rate(u2, u_cmd[:, 1], p.tau_spv, p.rate_spv)
        du3 = actuator_rate(u3, u_cmd[:, 2], p.tau_stv, p.rate_stv)
        du4 = actuator_rate(u4, u_cmd[:, 3], p.tau_w, p.rate_w)
        dTm_s = (O.T - Tm_s) / p.tau_T
        dTm_d = (D.T - Tm_d) / p.tau_T
        dTm_co = (T_co - Tm_co) / p.tau_T
        dmm = (mdot_c - mm) / p.tau_m
        dWm = (cp["W_el"] - Wm) / p.tau_W

        dx = np.stack([dP_s, dh_s, dT_sw, dP_d, dh_d, dT_dw, dP_i, dh_i, dT_cw,
                       dT_sh, dN, du1, du2, du3, du4, dTm_s, dTm_d, dmm, dWm, dTm_co], axis=1)
        if not want_aux:
            return dx

        M_d = D.rho * p.V_d
        aux = dict(
            t=self.t, P_s=P_s, P_d=P_d, P_i=P_i, T_s=O.T, T_tank=S.T, T_d=D.T, T_i=I.T,
            T_co=T_co, T_sat_s=S.T_sat, T_sat_d=D.T_sat, T_sat_i=I.T_sat,
            SH=O.T - S.T_sat, SC=SC, x_out=O.x, x_s=S.x, x_i=I.x, fill_s=fill_s, fill_i=fill_i,
            rho_s=O.rho, h_s=h_s, h_out=h_out, h_d=h_d, h_i=h_i, h_co=h_co, h2=h2, T2_ad=cp["T2_ad"],
            mdot_c=mdot_c, mdot_1=mdot_1, mdot_2=mdot_2, mdot_3=mdot_3, mdot_w=mdot_w,
            W_el=cp["W_el"], W_shaft=cp["W_shaft"], Pr=cp["Pr"], eta_v=cp["eta_v"], eta_s=cp["eta_s"],
            Q_r=Q_r, Q_w=Q_w, Q_sg=Q_sg, Q_dg=Q_dg, T_wo=T_wo, T_sw=T_sw, T_dw=T_dw, T_cw=T_cw,
            T_sh=T_sh, N=N, u1=u1, u2=u2, u3=u3, u4=u4, M_s=M_s, M_d=M_d, M_i=M_i,
            M_tot=M_s + M_d + M_i, charge=p.charge, Tm_s=Tm_s, Tm_d=Tm_d, Tm_co=Tm_co, mm=mm, Wm=Wm,
            dx=dx,
        )
        return dx, aux

    # ------------------------------------------------------------ integrate
    def _post(self, x):
        pr = self.props
        x[:, 10] = np.maximum(x[:, 10], 0.0)
        x[:, 11:15] = clip(x[:, 11:15], 0.0, 1.0)
        for k in (0, 3, 6):
            x[:, k] = clip(x[:, k], pr.p_min * 1.001, pr.p_max * 0.999)
        return x

    def _step_once(self, x, args):
        dt, f = self.dt, self.rhs
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
        return self._post(x)

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
        for _ in range(n_sub):
            x = self._step_once(x, args)
        self.x = x
        self.t += n_sub * self.dt
        _, self.aux = self.rhs(self.x, *args, want_aux=True)
        return self.aux

    def outputs(self) -> dict:
        if self.aux is None:
            _, self.aux = self.rhs(self.x, self.u_cmd, self.N_cmd, self.T_amb,
                                   self.T_wi, want_aux=True)
        return self.aux

    # -------------------------------------------------------- initialization
    def cold_start(self, idx=None, T_amb=None, u_pos=None, charge=None,
                   liquid_in_accumulator=None) -> None:
        """Equalized, compressor-off initial condition at ambient temperature.

        The total charge is distributed as saturated vapor in all volumes plus
        liquid split between the accumulator tank (fraction
        ``liquid_in_accumulator``, default ``params.cold_liquid_in_accumulator``)
        and the condenser.  If the charge is too small to reach saturation at
        ambient the whole stand holds superheated vapor at a lower pressure.
        Raises ``ValueError`` if the charge does not fit into the stand.
        """
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx))
        if len(idx) == 0:
            return
        n = len(idx)
        pr, p = self.props, self.p
        T_amb = self.T_amb[idx] if T_amb is None else np.broadcast_to(np.asarray(T_amb, float), (n,))
        self.T_amb[idx] = T_amb
        if charge is not None:
            p.charge[idx] = np.broadcast_to(np.asarray(charge, float), (n,))
        charge = p.charge[idx]
        f_acc = p.cold_liquid_in_accumulator[idx] if liquid_in_accumulator is None \
            else np.broadcast_to(np.asarray(liquid_in_accumulator, float), (n,))
        V_s, V_d, V_i = p.V_s[idx], p.V_d[idx], p.V_i[idx]
        V_tot = V_s + V_d + V_i

        P_eq = pr.P_sat(T_amb)
        sat = pr.sat(P_eq)
        rho_v, rho_l = sat["rho_v"], sat["rho_l"]
        # liquid mass accounting for the vapor volume it displaces
        M_liq = (charge - rho_v * V_tot) / (1.0 - rho_v / rho_l)
        wet = M_liq > 0.0

        # --- liquid distribution (tank first, overflow into the condenser)
        cap_s, cap_i = 0.95 * rho_l * V_s, 0.95 * rho_l * V_i
        M_ls = np.minimum(f_acc * np.maximum(M_liq, 0.0), cap_s)
        M_li = np.minimum(np.maximum(M_liq, 0.0) - M_ls, cap_i)
        rem = np.maximum(M_liq, 0.0) - M_ls - M_li
        M_ls = M_ls + np.minimum(rem, cap_s - M_ls)
        rem = np.maximum(M_liq, 0.0) - M_ls - M_li
        if np.any(rem > 1e-6):
            raise ValueError("refrigerant charge exceeds the stand's liquid capacity")
        M_vs = rho_v * (V_s - M_ls / rho_l)
        M_vi = rho_v * (V_i - M_li / rho_l)
        x_s = M_vs / (M_vs + M_ls)
        x_i = M_vi / (M_vi + M_li)
        h_vap = sat["h_v"] + 300.0
        h_s = np.where(M_ls > 0, sat["h_l"] + x_s * (sat["h_v"] - sat["h_l"]), h_vap)
        h_i = np.where(M_li > 0, sat["h_l"] + x_i * (sat["h_v"] - sat["h_l"]), h_vap)
        h_d = h_vap
        P = P_eq.copy()

        # --- dry stand: superheated vapor at the pressure holding the charge
        if np.any(~wet):
            lo = np.full(n, pr.p_min * 1.01)
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
            h_s = np.where(wet, h_s, h_dry)
            h_i = np.where(wet, h_i, h_dry)
            h_d = np.where(wet, h_d, h_dry)

        x = np.zeros((n, self.NX))
        x[:, 0], x[:, 1], x[:, 2] = P, h_s, T_amb
        x[:, 3], x[:, 4], x[:, 5] = P, h_d, T_amb
        x[:, 6], x[:, 7], x[:, 8] = P, h_i, T_amb
        x[:, 9], x[:, 10] = T_amb, 0.0
        u_pos = np.zeros((n, self.NU)) if u_pos is None else np.broadcast_to(np.asarray(u_pos, float), (n, self.NU))
        x[:, 11:15] = u_pos
        x[:, 15], x[:, 16], x[:, 19] = T_amb, T_amb, T_amb
        x[:, 17], x[:, 18] = 0.0, 0.0
        self.x[idx] = x
        self.u_cmd[idx] = u_pos
        self.N_cmd[idx] = 0.0
        self.aux = None

    def set_state(self, idx, x) -> None:
        idx = np.atleast_1d(np.asarray(idx))
        self.x[idx] = np.asarray(x, float).reshape(len(idx), self.NX)
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
                    T_wi=self.T_wi.copy(), T_wo=a["T_wo"] + z(p.sig_T), T_amb=self.T_amb.copy())

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
            floodback=a["x_out"] < 1.0,                 # liquid reaching the compressor
            accumulator_liquid=a["x_s"] < 1.0,          # liquid stored in the tank
            condenser_dry=a["fill_i"] < p.cond_dry_fill,  # undercharged: no liquid seal
            condenser_flooded=a["fill_i"] > 0.9,        # overcharged
        )
