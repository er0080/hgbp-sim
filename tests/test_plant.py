import numpy as np
import pytest

from hgbp_sim import BaselineController, HGBPPlant, PlantParams, named_point, solve_steady_state

C2K = 273.15


def _point(plant, name, N=1450.0):
    return named_point(name, plant.props, N)


def test_cold_start_holds_charge_and_is_near_equilibrium():
    pl = HGBPPlant(n=3, dt=0.05)
    pl.set_inputs(T_amb=298.15)
    charge = pl.nominal_charge() * np.array([0.2, 1.0, 1.6])
    pl.cold_start(T_amb=298.15, charge=charge, liquid_in_accumulator=0.5)
    a = pl.outputs()
    assert np.allclose(a["M_tot"], charge, rtol=2e-3)
    assert a["P_s"][0] < a["P_s"][1] - 0.5e5          # dry stand sits below saturation pressure
    assert np.allclose(a["P_s"][1:], pl.props.P_sat(298.15), rtol=1e-3)
    assert a["fill_s"][2] > a["fill_s"][1] > 0.0
    dx = pl.rhs(pl.x, pl.u_cmd, pl.N_cmd, pl.T_amb, pl.T_wi)
    assert np.abs(dx[:, HGBPPlant.P_S]).max() < 1000.0      # Pa/s (initial superheat relaxing)
    assert np.all(a["mdot_c"] == 0.0)
    with pytest.raises(ValueError):
        pl.cold_start(idx=[0], charge=200.0)


def test_mass_conservation_open_loop():
    pl = HGBPPlant(n=1, dt=0.05)
    pl.cold_start(T_amb=298.15)
    pl.set_inputs(u_cmd=[0.5, 0.6, 0.3, 0.3], N_cmd=1450.0, T_wi=293.15)
    M0 = pl.outputs()["M_tot"][0]
    for _ in range(120):
        a = pl.step(1.0)
    assert abs(a["M_tot"][0] - M0) / M0 < 1e-3
    assert a["mdot_c"][0] > 0.0 and a["P_d"][0] > a["P_i"][0] > a["P_s"][0]
    assert np.isfinite(pl.x).all()


def test_steady_state_solver_and_energy_balance():
    pl = HGBPPlant(n=3, dt=0.05)
    pl.set_inputs(T_amb=298.15, T_wi=293.15)
    pts = [_point(pl, "MT_standard"), _point(pl, "LT_standard"), _point(pl, "HT_standard", 1200.0)]
    kw = {k: [p[k] for p in pts] for k in ("P_s", "P_d", "SH", "N", "P_i")}
    res = solve_steady_state(pl, **kw)
    assert res["converged"].all()
    a = res["aux"]
    for k in ("P_s", "P_d", "P_i"):
        assert np.allclose(a[k], kw[k])
    assert np.allclose(a["SH"], kw["SH"], atol=0.05)
    assert np.allclose(a["M_tot"], pl.p.charge, rtol=1e-3)
    assert np.all(res["u"] > 0.0) and np.all(res["u"] < 1.0)
    # global energy balance: electrical input = water heat + ambient losses
    p = pl.p
    losses = (p.UA_sha * (a["T_sh"] - pl.T_amb) + p.UA_sa * (a["T_sw"] - pl.T_amb)
              + p.UA_da * (a["T_dw"] - pl.T_amb) + p.UA_ca * (a["T_cw"] - pl.T_amb))
    imbalance = a["W_el"] - a["Q_w"] - losses
    assert np.all(np.abs(imbalance) / a["W_el"] < 0.02)
    # simulating from the solution must not drift
    pl.set_state(np.arange(3), res["x"])
    pl.set_inputs(u_cmd=res["u"], N_cmd=kw["N"])
    for _ in range(30):
        b = pl.step(1.0)
    assert np.abs(b["P_s"] - a["P_s"]).max() < 200.0
    assert np.abs(b["P_d"] - a["P_d"]).max() < 500.0
    assert np.abs(b["SH"] - a["SH"]).max() < 0.05


def test_charge_sets_condenser_inventory():
    """More charge -> more liquid in the condenser and more subcooling; too little
    charge -> no liquid seal and the superheat target becomes unreachable."""
    pl = HGBPPlant(n=4, dt=0.05)
    pl.set_inputs(T_amb=298.15, T_wi=293.15)
    pt = _point(pl, "MT_standard")
    fac = np.array([0.25, 0.8, 1.2, 1.6])
    charge = pl.nominal_charge() * fac
    res = solve_steady_state(pl, pt["P_s"], pt["P_d"], pt["SH"], pt["N"], P_i=pt["P_i"], charge=charge)
    a = res["aux"]
    assert res["converged"][1:].all()
    assert a["fill_i"][1] < a["fill_i"][2] < a["fill_i"][3]
    assert a["SC"][3] > a["SC"][1] + 3.0
    assert not res["converged"][0] or a["fill_i"][0] < pl.p.cond_dry_fill[0] * 2


def test_batch_matches_single():
    pA = HGBPPlant(n=1, dt=0.05)
    pB = HGBPPlant(n=4, dt=0.05)
    for pl in (pA, pB):
        pl.cold_start(T_amb=298.15)
        pl.set_inputs(u_cmd=[0.5, 0.5, 0.3, 0.4], N_cmd=1450.0)
        pl.step(30.0)
    assert np.allclose(pA.x[0], pB.x[2])
    assert np.allclose(pB.x[0], pB.x[3])


def test_pid_reaches_test_point():
    pl = HGBPPlant(n=1, dt=0.05, rng=np.random.default_rng(0))
    pl.set_inputs(T_amb=298.15, T_wi=293.15)
    p0 = _point(pl, "MT_standard")
    p1 = _point(pl, "HT_standard", 1200.0)
    res = solve_steady_state(pl, p0["P_s"], p0["P_d"], p0["SH"], p0["N"], P_i=p0["P_i"])
    pl.set_state(0, res["x"])
    ctrl = BaselineController(1)
    ctrl.reset(res["u"])
    sp = dict(P_s=p1["P_s"], P_d=p1["P_d"], SH=p1["SH"], P_i=p1["P_i"])
    for _ in range(900):
        meas = pl.measure(noise=False)
        u = ctrl(meas, sp, 1.0)
        a = pl.step(1.0, u_cmd=u, N_cmd=p1["N"])
    assert abs(a["P_s"][0] - p1["P_s"]) < 0.02e5
    assert abs(a["P_d"][0] - p1["P_d"]) < 0.05e5
    assert abs(a["SH"][0] - p1["SH"]) < 0.5
    assert abs(a["P_i"][0] - p1["P_i"]) < 0.2e5
    assert a["x_out"][0] > 1.0


def test_accumulator_liquid_boils_off_and_overcharge_carries_over():
    pl = HGBPPlant(n=2, dt=0.05)
    pl.set_inputs(T_amb=298.15, T_wi=293.15)
    # 6x nominal charge with 90 % of the liquid in the tank floods the accumulator
    charge = pl.nominal_charge() * np.array([1.0, 6.0])
    pl.cold_start(T_amb=298.15, charge=charge, liquid_in_accumulator=0.9)
    a0 = pl.outputs()
    assert a0["x_s"][0] < 1.0 and a0["x_out"][0] >= 1.0       # liquid stored, vapor drawn
    pl.set_inputs(u_cmd=[0.5, 0.6, 0.0, 0.3], N_cmd=1450.0)
    for _ in range(600):
        a = pl.step(1.0)
    assert a["fill_s"][0] < a0["fill_s"][0] * 0.5             # bypass gas evaporates the liquid
    assert a0["fill_s"][1] > pl.p.acc_carry_fill0[1]           # heavily overcharged: tank above carry-over level
    assert a0["x_out"][1] < 1.0                                # -> liquid entrained to the compressor


def test_trips_on_closed_water_valve():
    """Closing the cooling water with the compressor running must eventually trip."""
    pl = HGBPPlant(n=1, dt=0.05)
    pl.set_inputs(T_amb=298.15, T_wi=293.15)
    p0 = _point(pl, "MT_standard")
    res = solve_steady_state(pl, p0["P_s"], p0["P_d"], p0["SH"], p0["N"], P_i=p0["P_i"])
    pl.set_state(0, res["x"])
    u = res["u"].copy()
    u[:, 3] = 0.0
    tripped = False
    for _ in range(1800):
        pl.step(1.0, u_cmd=u, N_cmd=p0["N"])
        t = pl.trips()
        if t["high_P_d"][0] or t["high_T_d"][0]:
            tripped = True
            break
    assert tripped
