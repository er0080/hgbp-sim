"""Parameter definitions for the HGBP test-stand model.

All quantities are SI (Pa, J/kg, K, kg/s, m^3, W, W/K, J/K, s).

Defaults describe a ~5 kW-capacity variable-speed semi-hermetic reciprocating
compressor on R134a with a brazed-plate water-cooled condenser.  Every numeric
field can be randomized per environment with :func:`randomize_params`.

Valve numbering follows the stand:
    1  discharge pressure valve      (discharge line, upstream of the split)
    2  suction pressure valve        (hot gas bypass line -> suction mixer)
    3  suction temperature valve     (condenser outlet -> suction mixer)
    4  cooling water valve           (condenser water inlet; sets the
                                      intermediate/condensing pressure)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from types import SimpleNamespace

import numpy as np


@dataclass
class PlantParams:
    # ---------------------------------------------------------------- fluid
    fluid: str = "R134a"

    # ----------------------------------------------------------- compressor
    V_disp: float = 250e-6      # swept volume per revolution [m^3/rev]
    N_nom: float = 1450.0       # nominal speed [rpm]
    N_min: float = 600.0        # minimum VFD speed when running [rpm]
    N_max: float = 2100.0       # maximum VFD speed [rpm]
    eta_v0: float = 0.95        # volumetric efficiency intercept
    c_cl: float = 0.04          # clearance re-expansion coefficient
    kappa: float = 1.10         # polytropic exponent for re-expansion
    eta_s0: float = 0.72        # peak isentropic efficiency
    a_s: float = 0.006          # curvature of eta_s vs pressure ratio
    Pr_opt: float = 4.0         # pressure ratio of peak eta_s
    b_N: float = 0.08           # curvature of eta_s vs relative speed
    eta_motor: float = 0.90     # motor + VFD efficiency
    f_motor_gas: float = 0.7    # fraction of motor loss absorbed by suction gas
    C_shell: float = 20e3       # compressor shell thermal capacity [J/K]
    UA_gs: float = 40.0         # discharge gas -> shell conductance [W/K]
    UA_sha: float = 8.0         # shell -> ambient conductance [W/K]
    cp_gas: float = 1000.0      # vapor cp used in gas/shell effectiveness [J/kg/K]
    tau_N: float = 0.5          # speed loop time constant [s]
    ramp_N: float = 300.0       # VFD ramp limit [rpm/s]

    # ------------------------------------------------------------- volumes
    V_s: float = 12e-3          # suction mixer/accumulator tank + suction line [m^3]
    V_d: float = 1.5e-3         # compressor discharge port -> valve 1 [m^3]
    V_i: float = 3.0e-3         # intermediate section: header + condenser + liquid line [m^3]

    # -------------------------------------------------------------- charge
    charge: float | None = None # total refrigerant mass [kg]; None -> nominal_charge()
    cold_liquid_in_accumulator: float = 0.5   # fraction of liquid sitting in the tank at a cold start

    # --------------------------------------------- suction accumulator tank
    acc_blend_dx: float = 0.02  # quality band over which the outlet switches sat. vapor -> superheated
    acc_carry_fill0: float = 0.5  # liquid fill fraction above which liquid is entrained to the compressor
    acc_carry_max: float = 0.3  # entrained liquid mass fraction at a full tank

    # ------------------------------------------------ pipe / vessel thermal mass
    C_sw: float = 4e3           # tank + suction line wall heat capacity [J/K]
    UA_sg: float = 40.0         # wall <-> gas [W/K]
    UA_sa: float = 6.0          # wall <-> ambient [W/K] (insulated)
    C_dw: float = 3e3           # discharge line wall [J/K]
    UA_dg: float = 40.0
    UA_da: float = 3.0

    # ----------------------------------------------------------- condenser
    UA_r_2ph: float = 3000.0    # refrigerant -> wall, condensing (full area) [W/K]
    UA_r_1ph: float = 250.0     # refrigerant -> wall, single-phase [W/K]
    cond_dry_fill: float = 0.08 # liquid fill below which the outlet loses its liquid seal
    SC_fill0: float = 0.25      # liquid fill above which outlet subcooling builds up
    SC_max: float = 15.0        # outlet subcooling at a liquid-full condenser [K]
    cp_liq: float = 1400.0      # liquid cp for subcooling enthalpy [J/kg/K]
    UA_w0: float = 3000.0       # wall -> water at full water flow [W/K]
    mdot_w_max: float = 0.5     # maximum cooling water flow [kg/s]
    cp_w: float = 4180.0        # water specific heat [J/kg/K]
    C_cw: float = 8e3           # brazed-plate HX metal + water content [J/K]
    UA_ca: float = 5.0          # condenser/liquid line -> ambient [W/K]

    # -------------------------------------------------------------- valves
    # flow coefficient C in kg/s per sqrt(Pa * kg/m^3)
    C_dpv: float = 6.0e-5       # valve 1: discharge pressure valve (fully open)
    dpv_char: str = "eqpct"
    dpv_R: float = 30.0
    tau_dpv: float = 0.3
    rate_dpv: float = 0.2
    C_spv: float = 4.0e-5       # valve 2: suction pressure (hot gas bypass) valve
    spv_char: str = "eqpct"
    spv_R: float = 30.0
    tau_spv: float = 0.3
    rate_spv: float = 0.2
    C_stv: float = 1.0e-6       # valve 3: suction temperature (liquid) valve
    stv_char: str = "linear"
    stv_R: float = 30.0
    tau_stv: float = 0.3
    rate_stv: float = 0.2
    w_char: str = "eqpct"       # valve 4: cooling water valve
    w_R: float = 30.0
    tau_w: float = 2.0
    rate_w: float = 0.1
    xT: float = 0.70            # gas valve terminal pressure-drop ratio
    eps_valve: float = 2e3      # regularization pressure for valves [Pa]
    f_choke_liq: float = 0.7    # max effective dP/P_in for flashing liquid

    # -------------------------------------------------------------- sensors
    tau_T: float = 4.0          # temperature sensor time constant [s]
    tau_m: float = 1.0          # Coriolis flow meter time constant [s]
    tau_W: float = 0.3          # power meter time constant [s]
    sig_T: float = 0.1          # temperature noise std [K]
    sig_P_s: float = 2e3        # suction pressure noise std [Pa]
    sig_P_d: float = 6e3        # discharge / intermediate pressure noise std [Pa]
    sig_m_rel: float = 0.003    # flow meter relative noise
    sig_W_rel: float = 0.005    # power meter relative noise

    # ------------------------------------------------------------- limits
    P_d_max: float = 26e5       # high discharge pressure trip [Pa]
    P_s_min: float = 0.3e5      # low suction pressure trip [Pa]
    P_s_max: float = 9e5        # high suction pressure trip [Pa]
    T_d_max: float = 135.0 + 273.15  # high discharge temperature trip [K]

    # ------------------------------------------------- baseline control aid
    dP_i_margin: float = 2.5e5  # default intermediate pressure setpoint = P_d - margin

    def replace(self, **kw) -> "PlantParams":
        return dataclasses.replace(self, **kw)

    n_act: int = dataclasses.field(default=4, init=False, repr=False)

    @staticmethod
    def numeric_fields() -> list[str]:
        return [f.name for f in fields(PlantParams) if f.type in ("float", float)]


# Default relative randomization (multiplicative, log-uniform +/- fraction)
DEFAULT_RANDOMIZATION: dict[str, float] = {
    "V_disp": 0.10, "eta_v0": 0.03, "c_cl": 0.30, "eta_s0": 0.06, "a_s": 0.3,
    "eta_motor": 0.03, "C_shell": 0.3, "UA_gs": 0.3, "UA_sha": 0.3,
    "V_s": 0.20, "V_d": 0.25, "V_i": 0.20,
    "C_sw": 0.3, "UA_sg": 0.3, "UA_sa": 0.3, "C_dw": 0.3, "UA_dg": 0.3, "UA_da": 0.3,
    "UA_r_2ph": 0.25, "UA_r_1ph": 0.25, "UA_w0": 0.25, "C_cw": 0.3, "UA_ca": 0.3,
    "SC_max": 0.3, "cond_dry_fill": 0.3, "acc_carry_fill0": 0.2,
    "C_dpv": 0.15, "C_spv": 0.15, "C_stv": 0.15,
    "tau_dpv": 0.3, "tau_spv": 0.3, "tau_stv": 0.3, "rate_dpv": 0.3, "rate_spv": 0.3,
    "rate_stv": 0.3, "tau_w": 0.3, "rate_w": 0.3,
    "tau_T": 0.3, "tau_m": 0.3,
}


def params_to_arrays(p: PlantParams, n: int) -> SimpleNamespace:
    """Broadcast scalar parameters to arrays of shape (n,)."""
    ns = SimpleNamespace()
    for f in fields(p):
        v = getattr(p, f.name)
        if isinstance(v, (str, bool)) or v is None:
            setattr(ns, f.name, v)
        else:
            setattr(ns, f.name, np.full(n, float(v)))
    return ns


def randomize_params(p: PlantParams, n: int, rng: np.random.Generator,
                     spec: dict[str, float] | None = None) -> SimpleNamespace:
    """Per-environment log-uniform perturbation of numeric parameters.

    ``spec`` maps field name -> relative half-width (e.g. 0.2 means the value
    is multiplied by a factor in [1/1.2, 1.2] on a log scale).
    """
    spec = DEFAULT_RANDOMIZATION if spec is None else spec
    ns = params_to_arrays(p, n)
    for name, frac in spec.items():
        if not hasattr(ns, name) or frac <= 0:
            continue
        base = getattr(ns, name)
        if not isinstance(base, np.ndarray):
            continue
        lo, hi = -np.log1p(frac), np.log1p(frac)
        setattr(ns, name, base * np.exp(rng.uniform(lo, hi, size=n)))
    return ns


def sample_params(p: PlantParams, n: int, rng, spec=None, randomize: bool = True):
    if randomize:
        return randomize_params(p, n, rng, spec)
    return params_to_arrays(p, n)


def nominal_charge(p, props, T_evap: float = 263.15, T_int: float = 313.15,
                   SH: float = 10.0, fill: float = 0.4):
    """Reference refrigerant charge [kg] for a parameter namespace ``p``
    (arrays): vapor in suction and discharge volumes at a typical medium
    temperature condition plus a condenser with liquid fill fraction ``fill``.
    """
    V_s, V_d, V_i = (np.asarray(getattr(p, k), float) for k in ("V_s", "V_d", "V_i"))
    n = np.broadcast(V_s, V_d, V_i).shape
    P_s = np.broadcast_to(props.P_sat(np.array([T_evap])), n)
    rho_s = props.state(P_s, props.h_PT(P_s, np.full(n, T_evap + SH))).rho
    P_i = np.broadcast_to(props.P_sat(np.array([T_int])), n)
    P_d = P_i + 2.5e5
    rho_d = props.state(P_d, props.h_PT(P_d, np.full(n, T_int + 35.0))).rho
    sat = props.sat(P_i)
    return rho_s * V_s + rho_d * V_d + V_i * (fill * sat["rho_l"] + (1.0 - fill) * sat["rho_v"])
