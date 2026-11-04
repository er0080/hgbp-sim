"""The tee and series suction exchanger (``mixer`` = 1) in place of the mixing exchanger."""
import numpy as np
import pytest

from hgbp_sim import HGBPPlant, PlantParams, solve_steady_state
from hgbp_sim.geometry import phe_channels, tube_volume
from hgbp_sim.properties import get_tables

C2K = 273.15


def degF(f):
    return (f - 32.0) / 1.8 + C2K


# a 168 cm3/rev machine at 0/110 degF, 10 R superheat at the compressor port, liquid at 30 degC
LT = dict(T_e=degF(0.0), T_c=degF(110.0), T_i=30.0 + C2K, SH=10.0 / 1.8, N=3500.0)


def _lt_state(n=1, **kw):
    pl = HGBPPlant(PlantParams(mixer=1.0, V_disp=1.68e-4, N_nom=3500.0, **kw), n=n)
    pl.set_inputs(T_amb=298.15, T_wi=283.15)
    pr = pl.props
    P_s, P_d, P_i = (float(pr.P_sat(np.array([T]))[0]) for T in (LT["T_e"], LT["T_c"], LT["T_i"]))
    res = solve_steady_state(pl, P_s, P_d, LT["SH"], LT["N"], P_i=P_i)
    return pl, res


def test_vapor_conductivity_in_the_tables():
    CP = pytest.importorskip("CoolProp.CoolProp")
    tab = get_tables("R410A")
    P = np.array([3e5, 10e5, 25e5])
    k = tab.transport(P)["k_v"]
    ref = [CP.PropsSI("L", "P", float(p), "Q", 1.0, "R410A") for p in P]
    assert np.allclose(k, ref, rtol=0.02)


def test_suction_exchanger_geometry():
    p = PlantParams(mixer=1.0)
    d = HGBPPlant(p, n=1).p
    A, B = phe_channels(p.bx_n_plates)
    V_ch = p.bx_b * p.bx_W * p.bx_L
    lines = tube_volume(p.D_q, p.t_q, p.L_q) + tube_volume(p.D_bp, p.t_bp, p.L_bp)
    mo, suc = tube_volume(p.D_mo, p.t_mo, p.L_mo), tube_volume(p.D_suc, p.t_suc, p.L_suc)
    # pass 1 (S1-S2, the larger half) in the quench cells, the tee and its lines in the first;
    # pass 2 in the gas cells, the jumper in its inlet cell (n - 1)
    assert np.isclose(d.V_sc[0, :5].sum(), A * V_ch + lines + 0.5 * mo)
    assert np.isclose(d.V_gc[0].sum(), B * V_ch + 0.25 * np.pi * p.bx_d_conn ** 2 * p.bx_L_jmp)
    assert np.isclose(d.V_sc[0, 5], 0.5 * mo + suc)
    assert np.isclose(d.A_b1_cell[0] * 5, 2.0 * A * p.bx_phi * p.bx_L * p.bx_W)
    assert np.isclose(d.C_mw_cell[0] * 5, p.bx_mass * 500.0)


def test_suction_exchanger_brings_the_tee_stream_to_equilibrium():
    """Out of equilibrium at the tee (the quench's unflashed liquid in superheated gas), dry
    and at the equilibrium state at the outlet; the stand's balances hold."""
    pl, res = _lt_state()
    a = res["aux"]
    assert res["converged"][0]
    assert 0.15 < a["y_bi"][0] < 0.35                       # the quench's liquid at the tee
    assert a["y_bo"][0] < 1e-4 and a["y_liq"][0] < 1e-4     # dry from the exchanger on
    x_path = np.concatenate([a["x_q"][0], a["x_g"][0][::-1]])
    assert np.all(np.diff(x_path) >= -1e-9)                 # the liquid only evaporates along the path
    assert x_path[0] < 0.95 and x_path[-1] == 1.0
    # pass 1 vapor cools from the hot gas towards the equilibrium temperature
    assert a["T_q"][0][0] > a["T_q"][0][-1] and a["T_q"][0][0] > a["T_s"][0] + 10.0
    assert 1e3 < a["dP_bx"][0] < 20e3
    assert a["P_tee"][0] > a["P_s"][0] + a["dP_bx"][0]
    # suction side energy balance (the plates only redistribute heat between the passes)
    Q_amb = (pl.p.mx_UA_a[0] / 5) * (298.15 - a["T_mw"][0]).sum()
    E_in = a["mdot_2"][0] * a["h_2f"][0] + a["mdot_3"][0] * a["h_3f"][0] + a["Q_sg"][0] + Q_amb
    assert abs(E_in - a["mdot_c"][0] * a["h_l1"][0]) < 50.0


def test_liquid_leaves_when_the_wetted_area_is_short_or_the_mixture_is_wet():
    """Less wetted plate leaves liquid at the outlet; a suction line that gains more heat than
    the port superheat holds leaves the stream wet in equilibrium at the exchanger."""
    pl, res = _lt_state(n=3)
    p = pl.p
    p.bx_F_wet[1] = 0.05
    p.UA_sa[2] = 40.0
    pl._prepare(p)
    P = res["aux"]
    res = solve_steady_state(pl, P["P_s"], P["P_d"], LT["SH"], LT["N"], P_i=float(pl.props.P_sat(np.array([LT["T_i"]]))[0]))
    a = res["aux"]
    assert a["y_bo"][0] < 1e-4 and a["y_bo"][1] > 0.01
    h_v = pl.props.sat(a["P_s"])["h_v"]
    assert a["h_go"][2] < h_v[2] and a["y_bo"][2] > 0.0     # wet in equilibrium: no exchanger dries it
    assert a["Q_sg"][2] > 2.0 * a["Q_sg"][0]


def test_suction_exchanger_conserves_mass_and_runs_from_cold():
    pl = HGBPPlant(PlantParams(mixer=1.0, V_disp=1.68e-4, N_nom=3500.0), n=1, dt=0.05)
    pl.cold_start(T_amb=298.15)
    pl.set_inputs(u_cmd=[0.8, 0.4, 0.1, 0.5], N_cmd=3500.0, T_wi=293.15)
    M0 = pl.conserved_mass()[0]
    for _ in range(120):
        a = pl.step(1.0)
    assert abs(pl.conserved_mass()[0] - M0) < 1e-9
    assert abs(a["M_tot"][0] - M0) / M0 < 1e-4
    assert np.isfinite(pl.x).all() and a["mdot_c"][0] > 0.0
