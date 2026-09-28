"""Parameter definitions for the HGBP test-stand model.

All quantities are SI (Pa, J/kg, K, kg/s, m^3, W, W/K, J/K, s).

Defaults describe the stand: a variable-speed 355 cm3/rev semi-hermetic
compressor on R410A, a water-cooled brazed-plate condenser with a liquid
receiver, and a brazed-plate mixing exchanger in place of a suction mixer
tank.  Components and piping are given as specified (tube sizes and lengths,
plate count, channel volume, heating surface, weights); the model's volumes
and heat capacities are derived from them (:mod:`hgbp_sim.geometry`).  Every
numeric field can be randomized per environment with :func:`randomize_params`.

Valve numbering follows the stand:
    1  discharge pressure valve      (discharge line, upstream of the split)
    2  suction pressure valve        (hot gas bypass line -> mixing exchanger, gas side)
    3  suction temperature valve     (receiver / liquid line -> mixing exchanger, quench side)
    4  cooling water valve           (condenser water inlet; sets the
                                      intermediate/condensing pressure)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from types import SimpleNamespace

import numpy as np

# Refrigerants offered by the UI (any CoolProp fluid works in the library)
FLUIDS = ("R134a", "R1234yf", "R1234ze(E)", "R404A", "R407C", "R410A", "R454B", "R454C", "R32", "R22", "R290", "R600a")


@dataclass
class PlantParams:
    # ---------------------------------------------------------------- fluid
    fluid: str = "R410A"        # refrigerant charged in the stand

    # ----------------------------------------------------------- compressor
    V_disp: float = 3.55e-4     # swept volume per revolution [m^3/rev]
    N_nom: float = 3550.0       # nominal speed, also the default speed setpoint [rpm]
    N_min: float = 1200.0       # minimum VFD speed when running [rpm]
    N_max: float = 5400.0       # maximum VFD speed [rpm]
    eta_v0: float = 0.95        # volumetric efficiency at pressure ratio 1 (mass flow per swept volume)
    c_cl: float = 0.04          # clearance volume fraction: volumetric efficiency falls as the pressure ratio rises
    kappa: float = 1.10         # polytropic exponent of the clearance gas re-expansion
    eta_s0: float = 0.72        # peak isentropic efficiency (sets shaft power and discharge temperature)
    a_s: float = 0.006          # how fast isentropic efficiency falls away from Pr_opt
    Pr_opt: float = 4.0         # pressure ratio at which isentropic efficiency peaks
    b_N: float = 0.08           # how fast isentropic efficiency falls away from nominal speed
    eta_motor: float = 0.90     # motor + VFD efficiency
    f_motor_gas: float = 0.7    # share of the motor losses that heats the suction gas (suction-gas-cooled motor)
    C_shell: float = 20e3       # compressor shell thermal capacity [J/K]
    UA_gs: float = 40.0         # discharge gas -> shell conductance [W/K]
    UA_sha: float = 8.0         # shell -> ambient conductance [W/K]
    cp_gas: float = 1000.0      # vapor cp used for the discharge gas -> shell heat exchange [J/kg/K]
    tau_N: float = 0.5          # speed response time constant of motor and VFD [s]
    ramp_N: float = 300.0       # VFD acceleration / deceleration limit [rpm/s]
    V_comp_suc: float = 5.0e-3  # compressor internal suction volume behind the suction port, excluding oil [m^3]
    V_comp_dis: float = 1.5e-3  # compressor internal discharge volume up to the discharge port (estimate) [m^3]
    comp_x_min: float = 0.7     # lowest quality the compressor draws in; more liquid collects in its shell (internal suction volume)

    # -------------------------------------------------------------- piping
    # Copper refrigerant lines: outside diameter, wall thickness, length and fittings loss
    # (friction: Churchill, smooth drawn copper; fittings in velocity heads).
    # Defaults are ACR type L tube; see geometry.LINES for the routing.
    D_dis: float = 0.028575     # discharge line (compressor -> valve 1) outside diameter, 1-1/8 in [m]
    t_dis: float = 0.00127      # discharge line wall thickness [m]
    L_dis: float = 4.0          # discharge line length [m]
    K_dis: float = 1.2          # discharge line fittings loss (bends; estimate) [velocity heads]
    D_hdr: float = 0.028575     # hot gas header (valve 1 -> condenser and valve 2) outside diameter, 1-1/8 in [m]
    t_hdr: float = 0.00127      # hot gas header wall thickness [m]
    L_hdr: float = 4.0          # hot gas header length [m]
    K_hdr: float = 1.6          # hot gas header fittings loss (bends, run through the valve 2 tee; estimate) [velocity heads]
    D_bp: float = 0.028575      # bypass line (valve 2 -> mixing exchanger S1) outside diameter, 1-1/8 in [m]
    t_bp: float = 0.00127       # bypass line wall thickness [m]
    L_bp: float = 0.5           # bypass line length (estimate) [m]
    K_bp: float = 1.3           # bypass line fittings loss (branch of the header tee, bend; estimate) [velocity heads]
    D_q: float = 0.022225       # quench line (valve 3 -> mixing exchanger S3) outside diameter, 7/8 in [m]
    t_q: float = 0.001143       # quench line wall thickness [m]
    L_q: float = 0.5            # quench line length (estimate) [m]
    K_q: float = 0.6            # quench line fittings loss (bends; estimate) [velocity heads]
    D_mo: float = 0.034925      # mixing exchanger outlets (S2, S4 -> tee) outside diameter, 1-3/8 in [m]
    t_mo: float = 0.001397      # mixing exchanger outlet piping wall thickness [m]
    L_mo: float = 1.0           # mixing exchanger outlet piping, both legs together (estimate) [m]
    K_mo: float = 1.3           # mixing exchanger outlet piping fittings loss per leg (bend, joining at the tee; estimate) [velocity heads]
    D_suc: float = 0.041275     # suction line (tee -> compressor) outside diameter, 1-5/8 in [m]
    t_suc: float = 0.001524     # suction line wall thickness [m]
    L_suc: float = 4.0          # suction line length; the suction temperature probe is at its end [m]
    K_suc: float = 1.2          # suction line fittings loss (bends; estimate) [velocity heads]
    D_drn: float = 0.022225     # condensate drain (condenser -> receiver) outside diameter, 7/8 in [m]
    t_drn: float = 0.001143     # condensate drain wall thickness [m]
    L_drn: float = 0.5          # condensate drain length (estimate) [m]
    K_drn: float = 0.6          # condensate drain fittings loss (bends; estimate) [velocity heads]
    D_liq: float = 0.022225     # liquid line (receiver -> valve 3) outside diameter, 7/8 in [m]
    t_liq: float = 0.001143     # liquid line wall thickness [m]
    L_liq: float = 3.0          # liquid line length [m]
    K_liq: float = 1.8          # liquid line fittings loss (dip tube, bends; filter-drier or sight glass would add; estimate) [velocity heads]

    # -------------------------------------------------------------- charge
    charge: float | None = None  # total refrigerant mass; empty = nominal charge for this stand [kg]
    cold_liquid_in_suction: float = 0.0  # share of the liquid on the suction side at a cold start (migrated into compressor and mixer; rest in the receiver)

    # ---------------------------------------------- condenser (brazed plate)
    # Alfa Laval ACH-70X-78M-F: refrigerant on S3-S4, cooling water on S1-S2
    cond_n_plates: float = 78.0  # condenser plate count (plates - 1 channels, split between the two sides)
    cond_V_ch: float = 0.095e-3  # condenser volume per channel [m^3]
    cond_A: float = 6.588       # condenser heating surface [m^2]
    cond_mass: float = 16.19    # condenser net weight (stainless plates, copper braze) [kg]
    alpha_r_2ph: float = 1594.0  # refrigerant -> plate heat transfer coefficient while condensing [W/m^2/K]
    alpha_r_1ph: float = 133.0  # refrigerant -> plate heat transfer coefficient for vapor (desuperheating) [W/m^2/K]
    alpha_sc: float = 607.0     # liquid -> plate heat transfer coefficient in the subcooled (flooded) zone [W/m^2/K]
    cond_sc_film: float = 0.02  # share of the plate area that subcools the draining condensate while the condenser is not flooded
    alpha_w0: float = 1594.0    # plate -> water heat transfer coefficient at the reference water flow; scales with flow^0.8 [W/m^2/K]
    mdot_w_ref: float = 1.75    # water flow at which alpha_w0 applies [kg/s]
    cp_w: float = 4180.0        # water specific heat [J/kg/K]
    rho_w: float = 1000.0       # cooling water density [kg/m^3]
    UA_ca: float = 5.0          # condenser -> ambient conductance [W/K]

    # ------------------------------------------------------ liquid receiver
    # Standard Refrigeration UR66 (MP): 61 lb R22 pumpdown capacity; vertical, dip tube outlet
    rec_V: float = 26.5e-3      # receiver internal volume (from the pumpdown rating at 90 % full, 90 degF) [m^3]
    rec_dip: float = 0.04       # dip tube inlet height as a share of the receiver volume; below it vapor enters the liquid line
    rec_mass: float = 25.0      # receiver shell weight (estimate) [kg]
    rec_UA_r: float = 150.0     # receiver shell -> refrigerant conductance [W/K]
    rec_UA_a: float = 3.0       # receiver shell -> ambient conductance [W/K]

    # ------------------------------------------ mixing exchanger (brazed plate)
    # Alfa Laval ACH-70X-78M-F, S2/S3 up: quench liquid S3 (top) -> S4, bypass gas S1 (bottom) -> S2
    mx_n_plates: float = 78.0   # mixing exchanger plate count
    mx_V_ch: float = 0.095e-3   # mixing exchanger volume per channel [m^3]
    mx_A: float = 6.588         # mixing exchanger heating surface [m^2]
    mx_mass: float = 16.19      # mixing exchanger net weight [kg]
    mx_alpha_g0: float = 500.0  # bypass gas -> plate heat transfer coefficient at mx_mdot_g_ref; scales with flow^0.8 [W/m^2/K]
    mx_mdot_g_ref: float = 0.5  # bypass gas flow at which mx_alpha_g0 applies [kg/s]
    mx_alpha_e: float = 1500.0  # plate -> evaporating quench heat transfer coefficient at mx_mdot_q_ref; scales with flow^0.5 [W/m^2/K]
    mx_alpha_v0: float = 250.0  # plate -> quench vapor (after dry-out) heat transfer coefficient at mx_mdot_q_ref; scales with flow^0.8 [W/m^2/K]
    mx_mdot_q_ref: float = 0.13  # quench flow at which mx_alpha_e and mx_alpha_v0 apply [kg/s]
    mx_UA_a: float = 2.0        # mixing exchanger -> ambient conductance (insulated) [W/K]
    tee_tau_evap: float = 0.3   # evaporation time constant of liquid droplets from the quench outlet in the gas downstream of the tee [s]

    # ------------------------------------------------------- pressure drops
    # The condenser sides and the mixing exchanger gas side are flow resistances sized like
    # a valve: dP = (mdot / (Kv / 36000))^2 / rho_m, plates and ports together, with the
    # side's mean density rho_m (two-phase: homogeneous).  The evaporating quench side is
    # computed from the plate geometry below.  Estimated from the ACH-70X geometry
    # (docs/pressure-drop.md).  The refrigerant sides add the static head of
    # their column over the port height; the cooling water circuit is closed, so its
    # static heads cancel.
    cond_Kv_r: float = 10.5     # condenser refrigerant side (header -> drain) flow coefficient [m^3/h]
    cond_Kv_w: float = 21.0     # condenser water side flow coefficient [m^3/h]
    cond_H: float = 0.466       # condenser port height (inlet to outlet port centres) [m]
    mx_Kv_g: float = 23.5       # mixing exchanger gas side (S1 -> S2) flow coefficient [m^3/h]
    # quench side (S3 -> S4) from the plate geometry: distributor and ports, then every cell's
    # friction (two-phase Amalfi et al. 2016, vapor / liquid Martin), acceleration and static head
    mx_Kv_dist: float = 4.0     # S3 refrigerant distributor, ahead of the channels (estimate: about 1 bar at the unit's 0.4 kg/s design flow) [m^3/h]
    mx_beta: float = 45.0       # mixing exchanger corrugation angle to the flow (M channels: one high- and one low-angle plate) [deg]
    mx_b: float = 2.0e-3        # mixing exchanger channel gap (pressing depth) [m]
    mx_W: float = 0.090         # mixing exchanger effective channel width [m]
    mx_phi: float = 1.17        # mixing exchanger plate surface enlargement factor (hydraulic diameter 2 b / phi)
    mx_d_port: float = 0.0315   # mixing exchanger port bore [m]
    mx_d_S34: float = 0.020     # mixing exchanger S3 / S4 connection bore (7/8 in) [m]
    mx_H: float = 0.466         # mixing exchanger port height (S1/S4 to S2/S3 port centres) [m]
    P_w_sup: float = 1.5e5      # cooling water supply pressure, gauge [Pa]
    P_w_ret: float = 0.35e5     # cooling water return pressure, gauge; supply minus return drives the water through valve 4, the piping and the condenser [Pa]
    Kv_wpipe: float = 12.0      # cooling water piping, fittings and strainer between supply and return, outside valve 4 and the condenser (estimate) [m^3/h]

    # ------------------------------------------------------------ pipe walls
    UA_sg: float = 40.0         # suction line wall -> refrigerant conductance [W/K]
    UA_sa: float = 6.0          # suction line wall -> ambient conductance (insulated) [W/K]
    UA_dg: float = 40.0         # discharge line wall -> discharge gas conductance [W/K]
    UA_da: float = 3.0          # discharge line wall -> ambient conductance [W/K]

    # -------------------------------------------------------------- valves
    # Every valve is sized by its Kv, the way the hardware is specified: the
    # water flow [m^3/h] the fully open valve passes at a 1 bar pressure drop.
    # Internally mdot = (Kv / 36000) f(u) sqrt(rho dP)  (see components.kv_to_C).
    Kv_dpv: float = 6.3         # valve 1: discharge pressure valve [m^3/h]
    dpv_char: str = "eqpct"     # valve 1 flow characteristic: linear, eqpct (equal percentage) or quick opening
    dpv_R: float = 30.0         # valve 1 rangeability (largest / smallest controllable flow), used by eqpct
    tau_dpv: float = 0.3        # valve 1 actuator time constant [s]
    rate_dpv: float = 0.2       # valve 1 actuator stroke speed limit, full strokes per second [1/s]
    Kv_spv: float = 6.3         # valve 2: suction pressure (hot gas bypass) valve [m^3/h]
    spv_char: str = "eqpct"     # valve 2 flow characteristic: linear, eqpct (equal percentage) or quick opening
    spv_R: float = 30.0         # valve 2 rangeability (largest / smallest controllable flow), used by eqpct
    tau_spv: float = 0.3        # valve 2 actuator time constant [s]
    rate_spv: float = 0.2       # valve 2 actuator stroke speed limit, full strokes per second [1/s]
    Kv_stv: float = 1.0         # valve 3: suction temperature (liquid) valve [m^3/h]
    stv_char: str = "linear"    # valve 3 flow characteristic: linear, eqpct (equal percentage) or quick opening
    stv_R: float = 30.0         # valve 3 rangeability (largest / smallest controllable flow), used by eqpct
    tau_stv: float = 0.3        # valve 3 actuator time constant [s]
    rate_stv: float = 0.2       # valve 3 actuator stroke speed limit, full strokes per second [1/s]
    Kv_w: float = 12.0          # valve 4: cooling water valve [m^3/h]
    w_char: str = "eqpct"       # valve 4 flow characteristic: linear, eqpct (equal percentage) or quick opening
    w_R: float = 30.0           # valve 4 rangeability (largest / smallest controllable flow), used by eqpct
    tau_w: float = 2.0          # valve 4 actuator time constant [s]
    rate_w: float = 0.1         # valve 4 actuator stroke speed limit, full strokes per second [1/s]
    xT: float = 0.70            # gas valve pressure-drop ratio at which the flow chokes (ISA xT)
    eps_valve: float = 2e3      # numerical smoothing of valve flow near zero pressure drop [Pa]
    f_choke_liq: float = 0.7    # largest effective pressure-drop ratio for flashing liquid in valve 3 (liquid choking)

    # -------------------------------------------------------------- sensors
    tau_T: float = 4.0          # temperature sensor time constant [s]
    tau_m: float = 1.0          # Coriolis flow meter time constant [s]
    tau_W: float = 0.3          # power meter time constant [s]
    sig_T: float = 0.1          # temperature measurement noise, standard deviation [K]
    sig_P_s: float = 2e3        # suction pressure measurement noise, standard deviation [Pa]
    sig_P_d: float = 6e3        # discharge / intermediate pressure measurement noise, standard deviation [Pa]
    sig_m_rel: float = 0.003    # mass flow measurement noise, standard deviation relative to the reading
    sig_W_rel: float = 0.005    # power measurement noise, standard deviation relative to the reading

    # ------------------------------------------------------------- limits
    P_d_max: float = 41.4e5     # high discharge pressure trip, 600 psig R410A switch [Pa]
    P_s_min: float = 0.3e5      # low suction pressure trip [Pa]
    P_s_max: float = 30e5       # high suction pressure trip [Pa]
    T_d_max: float = 135.0 + 273.15  # high discharge temperature trip [K]
    y_flood: float = 0.005      # liquid mass share at the compressor suction port counted as floodback

    # ------------------------------------------------- baseline control aid
    dP_i_margin: float = 15.89e5  # liquid pressure setpoint below discharge pressure when a test point gives none [Pa]

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
    "V_comp_suc": 0.2, "V_comp_dis": 0.3,
    "L_dis": 0.25, "L_hdr": 0.25, "L_bp": 0.5, "L_q": 0.5, "L_mo": 0.5, "L_suc": 0.25,
    "L_drn": 0.5, "L_liq": 0.25, "rec_V": 0.1, "rec_dip": 0.3, "rec_mass": 0.2, "rec_UA_r": 0.3,
    "UA_sg": 0.3, "UA_sa": 0.3, "UA_dg": 0.3, "UA_da": 0.3,
    "alpha_r_2ph": 0.25, "alpha_r_1ph": 0.25, "alpha_w0": 0.25, "UA_ca": 0.3,
    "alpha_sc": 0.25, "cond_sc_film": 0.3,
    "mx_alpha_g0": 0.25, "mx_alpha_e": 0.3, "mx_alpha_v0": 0.3, "tee_tau_evap": 0.5,
    "Kv_dpv": 0.15, "Kv_spv": 0.15, "Kv_stv": 0.15, "Kv_w": 0.15,
    "tau_dpv": 0.3, "tau_spv": 0.3, "tau_stv": 0.3, "rate_dpv": 0.3, "rate_spv": 0.3,
    "rate_stv": 0.3, "tau_w": 0.3, "rate_w": 0.3,
    "tau_T": 0.3, "tau_m": 0.3,
}


class ParamSet(SimpleNamespace):
    """Per-environment parameter namespace (arrays of shape (n,), plus the
    derived volumes of :func:`hgbp_sim.geometry.derive`).

    :meth:`packed` returns the values as one structured array, the form the
    compiled model takes.  Packing rebinds the namespace's arrays to views
    into that record, so in-place edits (``p.charge[idx] = ...``) reach the
    model directly; assigning an attribute repacks at the next use."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not name.startswith("_"):
            self.__dict__["_rec"] = None

    def __reduce__(self):
        # a copy packs its own record (views into this one would come back as copies)
        return type(self), (), {k: v for k, v in self.__dict__.items() if k != "_rec"}

    def packed(self) -> np.ndarray:
        rec = self.__dict__.get("_rec")
        if rec is None:
            from .kernel.layout import CHAR_FIELDS, pack
            rec = pack(self)
            d = self.__dict__
            for name in rec.dtype.names:
                if name.startswith("V_line_"):
                    d["V_lines"][name[7:]] = rec[name]
                elif name not in CHAR_FIELDS.values():
                    d[name] = rec[name]
            d["_rec"] = rec
        return rec


def params_to_arrays(p: PlantParams, n: int) -> ParamSet:
    """Broadcast scalar parameters to arrays of shape (n,)."""
    ns = ParamSet()
    for f in fields(p):
        v = getattr(p, f.name)
        if isinstance(v, (str, bool)) or v is None:
            setattr(ns, f.name, v)
        else:
            setattr(ns, f.name, np.full(n, float(v)))
    return ns


def randomize_params(p: PlantParams, n: int, rng: np.random.Generator,
                     spec: dict[str, float] | None = None) -> ParamSet:
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
                   SH: float = 10.0, level: float = 0.4):
    """Reference refrigerant charge [kg] for a parameter namespace ``p``
    (arrays, with the derived volumes of :func:`hgbp_sim.geometry.derive`):
    vapor in the suction and discharge volumes at a typical medium temperature
    condition, a full liquid line and the receiver filled to ``level`` (share
    of its volume); condenser, drain and header hold saturated vapor.
    """
    V_s, V_d, V_i = (np.atleast_1d(np.asarray(getattr(p, k), float)) for k in ("V_s", "V_d", "V_i"))
    n = V_s.shape
    P_s = np.broadcast_to(props.P_sat(np.array([T_evap])), n)
    rho_s = props.state(P_s, props.h_PT(P_s, np.full(n, T_evap + SH))).rho
    P_i = np.broadcast_to(props.P_sat(np.array([T_int])), n)
    P_d = P_i + 2.5e5
    rho_d = props.state(P_d, props.h_PT(P_d, np.full(n, T_int + 35.0))).rho
    sat = props.sat(P_i)
    V_l = np.minimum(liquid_volume_for_level(p, level), V_i)
    return rho_s * V_s + rho_d * V_d + V_l * sat["rho_l"] + (V_i - V_l) * sat["rho_v"]


def liquid_volume_for_level(p, level):
    """Liquid volume of the intermediate section with the receiver filled to
    ``level`` (share of its volume) and the condenser drained: the liquid line
    is full once the level is above the dip tube inlet."""
    level = np.asarray(level, float)
    V_liq = p.V_lines["liq"]
    ramp = np.clip((level - p.rec_dip) / 0.01, 0.0, 1.0)      # liquid line fills just above the dip tube
    return level * p.rec_V + ramp * V_liq


# ---------------------------------------------------------------- metadata
_GROUPS = [
    ("fluid", "Refrigerant"), ("V_disp", "Compressor"), ("D_dis", "Piping"), ("charge", "Charge"),
    ("cond_n_plates", "Condenser"), ("rec_V", "Receiver"), ("mx_n_plates", "Mixing exchanger"), ("cond_Kv_r", "Pressure drops"),
    ("UA_sg", "Pipe walls"), ("Kv_dpv", "Valves"), ("tau_T", "Sensors"), ("P_d_max", "Safety limits"),
    ("dP_i_margin", "Baseline control"),
]
# structural parameters (volumes and heat capacities follow from them): applied at the
# next cold / warm start
_REQUIRE_INIT = {"fluid", "charge", "cold_liquid_in_suction", "V_comp_suc", "V_comp_dis",
                 "cond_n_plates", "cond_V_ch", "cond_mass", "rec_V", "rec_dip", "rec_mass",
                 "mx_n_plates", "mx_V_ch", "mx_A", "mx_mass", "cond_A", "rho_w", "cp_w"} \
    | {f"{a}_{k}" for a in ("D", "t", "L") for k in ("dis", "hdr", "bp", "q", "mo", "suc", "drn", "liq")}
# Editable range (min, max) enforced by the UI and by LiveStand.set_params.
# Parameters that are not listed are unconstrained.
_RANGES: dict[str, tuple[float, float]] = {
    "Kv_dpv": (0.01, 200.0), "Kv_spv": (0.01, 200.0),
    "Kv_stv": (0.01, 200.0), "Kv_w": (0.01, 200.0),
    "P_w_sup": (0.2e5, 10e5), "P_w_ret": (0.0, 9e5),
    "cond_Kv_r": (0.1, 500.0), "cond_Kv_w": (0.1, 500.0), "mx_Kv_g": (0.1, 500.0), "mx_Kv_dist": (0.1, 500.0),
    "mx_beta": (20.0, 70.0), "mx_b": (0.5e-3, 5e-3), "mx_W": (0.02, 0.5), "mx_phi": (1.0, 1.5),
    "mx_d_port": (0.005, 0.2), "mx_d_S34": (0.005, 0.2),
    "Kv_wpipe": (0.1, 500.0), "cond_H": (0.0, 2.0), "mx_H": (0.0, 2.0),
    "cond_n_plates": (4.0, 124.0), "mx_n_plates": (4.0, 124.0), "rec_dip": (0.0, 0.5),
    "cond_sc_film": (0.0, 0.5), "y_flood": (1e-4, 0.5), "tee_tau_evap": (0.01, 10.0), "comp_x_min": (0.0, 1.0),
    "V_comp_suc": (0.0, 0.1), "V_comp_dis": (0.0, 0.1), "rec_V": (1e-3, 1.0),
    **{f"{a}_{k}": (0.0, 0.2 if a != "L" else 50.0) for a in ("D", "t", "L")
       for k in ("dis", "hdr", "bp", "q", "mo", "suc", "drn", "liq")},
    **{f"K_{k}": (0.0, 100.0) for k in ("dis", "hdr", "bp", "q", "mo", "suc", "drn", "liq")},
}


def clamp_to_range(name: str, value: float) -> float:
    """Clamp a numeric parameter to its editable range (identity if unlisted)."""
    lo, hi = _RANGES.get(name, (-np.inf, np.inf))
    return float(min(max(value, lo), hi))


def param_metadata() -> list[dict]:
    """Description, unit, group and default of every parameter, parsed from
    the dataclass source (used by the web UI settings page)."""
    import inspect
    import re

    src = inspect.getsource(PlantParams)
    meta, group = [], "General"
    starts = {k: g for k, g in _GROUPS}
    for line in src.splitlines():
        m = re.match(r"\s{4}(\w+):\s*([\w\[\]| ]+?)\s*=\s*([^#]+?)\s*(#\s*(.*))?$", line)
        if not m or m.group(1) in ("n_act",):
            continue
        name, typ, default, _, comment = m.groups()
        if name in starts:
            group = starts[name]
        comment = comment or ""
        unit = ""
        um = re.search(r"\[([^\]]+)\]", comment)
        if um:
            unit = um.group(1)
            comment = (comment[:um.start()] + comment[um.end():]).strip(" ,;")
        comment = " ".join(comment.split())
        kind = "str" if "str" in typ else ("bool" if "bool" in typ else "float")
        try:
            dflt = eval(default, {"__builtins__": {}}, {})
        except Exception:  # noqa: BLE001
            dflt = default
        lo, hi = _RANGES.get(name, (None, None))
        meta.append(dict(name=name, group=group, unit=unit, description=comment.strip(), default=dflt,
                         kind=kind, requires_init=name in _REQUIRE_INIT, min=lo, max=hi))
    return meta
