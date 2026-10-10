import numpy as np
import pytest

from hgbp_sim import BaselineController, HGBPPlant, PlantParams, named_point, solve_steady_state
from hgbp_sim.components import churchill_f, kv_to_C, pipe_drop, series_C, valve_characteristic
from hgbp_sim.geometry import tube_volume

C2K = 273.15
RATING = dict(P_s=9.98e5, P_d=33.89e5, P_i=18.0e5, N=3550.0)
# the model tests also run the compressor below the stand's 35-75 Hz VFD range
WIDE = dict(N_min=1000.0, N_max=5400.0)


def _point(plant, name, N=1450.0):
    return named_point(name, plant.props, N)


def _rating_SH(pl, T_s=18.33):
    return (T_s + C2K) - float(pl.props.T_sat(np.array([RATING["P_s"]]))[0])


def _rating_state(n=1, T_wi=283.15, **kw):
    pl = HGBPPlant(PlantParams(**WIDE), n=n, dt=0.05)
    pl.set_inputs(T_amb=298.15, T_wi=T_wi)
    res = solve_steady_state(pl, RATING["P_s"], RATING["P_d"], kw.pop("SH", _rating_SH(pl)), kw.pop("N", RATING["N"]),
                             P_i=RATING["P_i"], **kw)
    return pl, res


def test_volumes_follow_the_piping_and_component_specification():
    p = PlantParams()
    pl = HGBPPlant(p, n=1)
    d = pl.p
    tube = lambda D, t, L: tube_volume(D, t, L)
    # ACH-70X-78: 77 channels, 38 on the S3-S4 side (refrigerant / quench), 39 on S1-S2
    assert np.isclose(d.V_cond[0], 38 * p.cond_V_ch) and np.isclose(d.V_mx_g[0], 39 * p.mx_V_ch)
    V_s = (39 + 38) * p.mx_V_ch + tube(p.D_bp, p.t_bp, p.L_bp) + tube(p.D_q, p.t_q, p.L_q) \
        + tube(p.D_mo, p.t_mo, p.L_mo) + tube(p.D_suc, p.t_suc, p.L_suc) + p.V_comp_suc
    assert np.isclose(d.V_s[0], V_s)
    assert np.isclose(d.V_d[0], p.V_comp_dis + tube(p.D_dis, p.t_dis, p.L_dis))
    assert 0.016 < d.V_s[0] < 0.020 and 0.030 < d.V_i[0] < 0.036
    # 4 m of 1-5/8 in type L: 38.2 mm bore -> 4.59 L
    assert abs(tube(0.041275, 0.001524, 4.0) - 4.59e-3) < 0.02e-3
    # condenser heat capacity: plates plus water content
    assert np.isclose(d.C_cw[0], p.cond_mass * 500.0 + 39 * p.cond_V_ch * p.rho_w * p.cp_w)


def test_cold_start_holds_charge_and_is_near_equilibrium():
    pl = HGBPPlant(PlantParams(**WIDE), n=3, dt=0.05)
    pl.set_inputs(T_amb=298.15)
    charge = pl.nominal_charge() * np.array([0.2, 1.0, 1.6])
    pl.cold_start(T_amb=298.15, charge=charge, liquid_in_suction=np.array([0.0, 0.0, 0.1]))
    a = pl.outputs()
    assert np.allclose(pl.conserved_mass(), charge, rtol=1e-9)
    assert np.allclose(a["M_tot"], charge, rtol=2e-3)
    assert a["P_s"][0] < a["P_s"][1] - 0.5e5          # dry stand sits below saturation pressure
    assert np.allclose(a["P_s"][1:], pl.props.P_sat(298.15), rtol=1e-3)
    assert a["rec_level"][2] > a["rec_level"][1] > 0.0      # the liquid sits in the receiver
    h_v = pl.props.sat(a["P_s"])["h_v"]
    assert a["h_l2"][1] >= h_v[1] and a["h_l2"][2] < h_v[2]  # migrated liquid in the compressor
    dx = pl.rhs(pl.x, pl.u_cmd, pl.N_cmd, pl.T_amb, pl.T_wi)
    # Pa/s: R410A vapor at the dew point sits ~0.1 K above walls at the bubble point and relaxes
    assert np.abs(dx[:, HGBPPlant.P_S]).max() < 2500.0
    assert np.all(a["mdot_c"] == 0.0)
    with pytest.raises(ValueError):
        pl.cold_start(idx=[0], charge=200.0)


def test_mass_and_energy_conservation_open_loop():
    pl = HGBPPlant(PlantParams(**WIDE), n=1, dt=0.05)
    pl.cold_start(T_amb=298.15)
    pl.set_inputs(u_cmd=[0.8, 0.6, 0.15, 0.5], N_cmd=1450.0, T_wi=293.15)
    M0 = pl.conserved_mass()[0]
    for _ in range(120):
        a = pl.step(1.0)
    assert abs(pl.conserved_mass()[0] - M0) < 1e-9
    assert abs(a["M_tot"][0] - M0) / M0 < 1e-4          # the (P, h) states agree with the conserved mass
    assert a["mdot_c"][0] > 0.0 and a["P_d"][0] > a["P_i"][0] > a["P_s"][0]
    assert np.isfinite(pl.x).all()


def test_steady_state_solver_and_energy_balance():
    pl = HGBPPlant(PlantParams(**WIDE), n=3, dt=0.05)
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
    T_amb = pl.T_amb
    losses = (p.UA_sha * (a["T_sh"] - T_amb) + p.UA_sa * (a["T_sw"] - T_amb)
              + p.UA_da * (a["T_dw"] - T_amb) + p.UA_ca * (a["T_cw"] - T_amb)
              + p.rec_UA_a * (a["T_rw"] - T_amb) + p.mx_UA_a / pl.MX * (a["T_mw"] - T_amb[:, None]).sum(1))
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


def test_mixing_exchanger_profile_is_physical():
    """Counterflow: quench evaporates from the top, dries out, and leaves as vapor
    at the bottom; plate walls sit between quench and gas; the exchanger's energy
    balance closes."""
    pl, res = _rating_state(n=2, N=[3550.0, 1200.0])
    assert res["converged"].all()
    a = res["aux"]
    assert np.all(np.diff(a["x_q"], axis=1) > 0.0)                 # quality rises down the quench side
    assert np.all(a["x_q"][:, 0] < 1.0) and np.all(a["x_qo"] > 1.0)
    assert np.all(a["T_q"] <= a["T_mw"] + 1e-6) and np.all(a["T_mw"] <= a["T_g"] + 1e-6)
    assert np.all(a["T_go"] < a["T_s"]) and np.all(a["T_qo"] > a["T_s"])   # the tee mixes them to T_s
    Q_amb = pl.p.mx_UA_a / pl.MX * (pl.T_amb[:, None] - a["T_mw"]).sum(1)
    gas = a["mdot_2"] * (a["h_2f"] - a["h_go"])
    assert np.allclose(gas + Q_amb, a["Q_q"], rtol=1e-3)
    assert np.all(a["y_liq"] == 0.0) and np.all(a["M_q_liq"] > 0.05)


def test_condenser_profile_is_physical():
    """Counterflow: the hot gas enters superheated at S3 and condenses on its way
    down; the water warms on its way up; plate walls sit between water and
    refrigerant; what condenses is what valve 3 passes, and the walls' energy
    balance closes."""
    pl, res = _rating_state(n=2, N=[3550.0, 1200.0])
    assert res["converged"].all()
    a = res["aux"]
    assert np.all(np.diff(a["x_c"], axis=1) < 0.0)                 # quality falls down the refrigerant side
    assert np.all(a["x_c"][:, 0] > 1.0) and np.all((a["x_c"][:, -2] > 0.0) & (a["x_c"][:, -2] < 0.6))
    assert np.all(a["x_c"][:, -1] < 0.0)                           # leaving the last cell: the subcooled drain
    assert np.all(np.diff(a["T_wc"], axis=1) < 0.0)                # water rises from S1 at the bottom
    assert np.all(a["T_wc"] <= a["T_cwc"] + 1e-6) and np.all(a["T_cwc"] <= a["T_c"] + 1e-6)
    assert np.allclose(a["mdot_cr"], a["mdot_3"], rtol=1e-2)
    Q_amb = pl.p.UA_ca * (pl.T_amb - a["T_cw"])
    assert np.allclose(a["Q_r"] + a["Q_sc"] + Q_amb, a["Q_w"], rtol=1e-3)


def test_receiver_pool_lags_the_pressure():
    """The receiver's liquid pool keeps its own temperature: a pressure drop leaves
    it superheated, it flashes (vapor back to the condensing zone, the pool cools)
    and valve 3 loses its subcooling; a pressure rise leaves it subcooled, and the
    subcooling at valve 3 grows until vapor condensing on the pool catches up."""
    for u4, falls in ((1.0, True), (0.3, False)):
        pl, res = _rating_state()
        pl.set_state(0, res["x"])
        u = res["u"].copy()
        u[:, 3] = u4
        SC0, T_L0 = res["aux"]["SC"][0], res["aux"]["T_L"][0]
        a = [pl.step(1.0, u_cmd=u, N_cmd=RATING["N"]) for _ in range(20)]
        SC = np.array([b["SC"][0] for b in a])
        lv = np.array([b["mdot_lv"][0] for b in a])
        if falls:
            assert SC.min() < 0.05 and lv.min() < -0.01 and a[-1]["T_L"][0] < T_L0 - 3.0
        else:
            assert SC[-1] > SC0 + 3.0 and lv.min() > 0.0


def test_condenser_hold_up():
    """The condensing zone holds the condensing film (void fraction on the cell
    profile) plus the condensate on its way down the drain, and the drain passes
    what valve 3 takes from the pool."""
    pl, res = _rating_state(n=2, N=[3550.0, 1200.0])
    a = res["aux"]
    assert np.all(a["M_film"] > 0.1) and np.all(a["M_film"] < pl.p.V_cond * pl.props.sat(a["P_i"])["rho_l"])
    assert np.allclose(a["M_cl"] - a["M_film"], pl.p.cond_tau_drain * a["mdot_drn"], rtol=1e-3)
    assert np.allclose(a["mdot_drn"] + a["mdot_lv"], a["mdot_3"], rtol=1e-3)


def test_receiver_absorbs_charge():
    """The receiver takes up charge variations at unchanged condensing area; only
    a full receiver backs liquid up into the condenser; too little charge loses
    the liquid seal and the point becomes unreachable."""
    pl = HGBPPlant(PlantParams(**WIDE), n=5, dt=0.05)
    pl.set_inputs(T_amb=298.15, T_wi=283.15)
    fac = np.array([0.3, 0.8, 1.0, 1.4, 2.4])
    charge = pl.nominal_charge() * fac
    res = solve_steady_state(pl, RATING["P_s"], RATING["P_d"], _rating_SH(pl), RATING["N"], P_i=RATING["P_i"],
                             charge=charge)
    a = res["aux"]
    assert res["converged"][1:4].all()
    assert np.all(np.diff(a["rec_level"][1:4]) > 0.05)
    assert np.all(a["cond_flood"][1:4] == 0.0)
    assert np.ptp(res["u"][1:4, 3]) < 0.02                         # same water valve: same condensing area
    assert not res["converged"][0] and a["ll_fill"][0] < 1.0       # undercharged: vapor in the liquid line
    assert a["cond_flood"][4] > 0.0 and a["rec_level"][4] > 0.95   # overcharged: receiver full, plates flooding
    assert a["SC"][4] > a["SC"][2]


def test_batch_matches_single():
    pA = HGBPPlant(PlantParams(**WIDE), n=1, dt=0.05)
    pB = HGBPPlant(PlantParams(**WIDE), n=4, dt=0.05)
    for pl in (pA, pB):
        pl.cold_start(T_amb=298.15)
        pl.set_inputs(u_cmd=[0.5, 0.5, 0.15, 0.4], N_cmd=1450.0)
        pl.step(30.0)
    assert np.allclose(pA.x[0], pB.x[2])
    assert np.allclose(pB.x[0], pB.x[3])


def test_pid_reaches_test_point():
    pl = HGBPPlant(PlantParams(**WIDE), n=1, dt=0.05, rng=np.random.default_rng(0))
    pl.set_inputs(T_amb=298.15, T_wi=293.15)
    p0 = _point(pl, "MT_standard")
    p1 = _point(pl, "HT_standard", 1200.0)
    res = solve_steady_state(pl, p0["P_s"], p0["P_d"], p0["SH"], p0["N"], P_i=p0["P_i"])
    pl.set_state(0, res["x"])
    ctrl = BaselineController(1)
    ctrl.reset(res["u"])
    T_s = float(pl.props.T_sat(np.array([p1["P_s"]]))[0]) + p1["SH"]     # the point's return gas temperature
    sp = dict(P_s=p1["P_s"], P_d=p1["P_d"], T_s=T_s, P_i=p1["P_i"])
    for _ in range(3600):          # 900 s at the 0.25 s control interval
        meas = pl.measure(noise=False)
        u = ctrl(meas, sp, 0.25)
        a = pl.step(0.25, u_cmd=u, N_cmd=p1["N"])
    assert abs(a["P_s"][0] - p1["P_s"]) < 0.02e5
    assert abs(a["P_d"][0] - p1["P_d"]) < 0.05e5
    assert abs(a["SH"][0] - p1["SH"]) < 0.5
    assert abs(a["P_i"][0] - p1["P_i"]) < 0.2e5
    assert a["y_liq"][0] == 0.0


def test_overfed_quench_reaches_the_compressor_as_liquid():
    """Opening the quench valve beyond what the exchanger can evaporate sends
    liquid through the tee to the compressor (floodback); the probe then reads
    saturation."""
    pl, res = _rating_state(SH=5.0)
    pl.set_state(0, res["x"])
    u = res["u"].copy()
    u[:, 2] *= 1.6
    for _ in range(60):
        a = pl.step(0.5, u_cmd=u, N_cmd=RATING["N"])
    t = pl.trips()
    assert t["floodback"][0] and t["mixer_wet"][0]
    assert a["x_qo"][0] < 1.0 and a["y_liq"][0] > pl.p.y_flood[0]
    assert a["SH"][0] < 0.5


def test_droplets_from_the_quench_survive_to_the_probe():
    """Wet quench outlet mixed with superheated gas: part of the liquid is still
    in flight at the probe, which reads the (superheated) vapor temperature."""
    pl, res = _rating_state()
    x = res["x"].copy()
    sat = pl.props.sat(np.array([RATING["P_s"]]))
    x[0, HGBPPlant.IX["h_q4"]] = sat["h_l"][0] + 0.8 * (sat["h_v"][0] - sat["h_l"][0])   # 20 % liquid leaving S4
    _, a = pl.rhs(x, res["u"], np.array([RATING["N"]]), pl.T_amb, pl.T_wi, want_aux=True)
    assert a["x_l1"][0] > 1.0                                  # the mixture would be superheated at equilibrium
    assert 0.0 < a["y_liq"][0] < 0.2
    assert a["x_out"][0] < 1.0 and a["SH"][0] > 1.0            # liquid at the compressor, probe sees superheat
    slow = pl.p.tee_tau_evap[0]
    pl.p.tee_tau_evap[:] = 10.0 * slow                         # slower evaporation: more liquid survives
    _, b = pl.rhs(x, res["u"], np.array([RATING["N"]]), pl.T_amb, pl.T_wi, want_aux=True)
    assert b["y_liq"][0] > a["y_liq"][0]


def test_trips_on_closed_water_valve():
    """Closing the cooling water with the compressor running must eventually trip."""
    pl, res = _rating_state()
    pl.set_state(0, res["x"])
    u = res["u"].copy()
    u[:, 3] = 0.0
    tripped = False
    for _ in range(1800):
        pl.step(1.0, u_cmd=u, N_cmd=RATING["N"])
        t = pl.trips()
        if t["high_P_d"][0] or t["high_T_d"][0]:
            tripped = True
            break
    assert tripped


def test_mass_conserved_when_condenser_floods():
    """Overcharging until the receiver is full and the condenser floods must raise
    the pressure, not lose refrigerant."""
    from hgbp_sim.live import LiveStand
    st = LiveStand(dt_ctrl=0.25)
    st.noise = False
    st.params = st.params.replace(charge=float(st.plant.nominal_charge()[0]) * 1.9)
    assert st.warm_start("MT_standard")
    s = st.snapshot()
    m0, P_i0, u4_0 = s["charge"]["kg"], s["meas"]["P_i"], s["meas"]["u4"]
    st.set_sim(charge_rate=0.05)
    st.add_charge(6.0)
    injected = 0.0
    flooded, P_i_max, u4_max = False, 0.0, 0.0
    for _ in range(4800):
        pending = st.charge_pending
        s = st.step()
        injected += pending - st.charge_pending
        assert abs(s["charge"]["kg"] - (m0 + injected)) < 1e-6
        flooded |= s["true"]["cond_flood"] > 0.3
        P_i_max = max(P_i_max, s["meas"]["P_i"])
        u4_max = max(u4_max, s["meas"]["u4"])
        if s["compressor"]["tripped"]:
            break
    assert flooded
    # lost condensing area: the pressure rises and the water loop opens its valve to hold it
    assert s["compressor"]["tripped"] or (P_i_max > P_i0 + 0.1 and u4_max > u4_0 + 0.1)


def test_subcooled_zone_is_bounded_by_the_water_inlet():
    """Subcooling comes from the draining condensate running over the cold bottom of
    the plates (the film share, or the flooded area) on its way to the receiver: the
    liquid never leaves colder than the water enters, colder water subcools more,
    flooded plates subcool more, and the receiver pool carries it to valve 3 (closing
    the valve leaves the liquid at the pool's temperature)."""
    pl = HGBPPlant(PlantParams(**WIDE), n=3, dt=0.05)
    T_wi = np.array([293.15, 293.15, 288.15])
    pl.set_inputs(T_amb=298.15, T_wi=T_wi)
    res = solve_steady_state(pl, RATING["P_s"], RATING["P_d"], _rating_SH(pl), RATING["N"], P_i=RATING["P_i"],
                             charge=pl.nominal_charge() * np.array([1.0, 2.3, 1.0]))
    assert res["converged"].all()
    a = res["aux"]
    assert a["cond_flood"][1] > 0.1 and np.all(a["cond_flood"][[0, 2]] == 0.0)
    assert np.all(a["T_co"] >= T_wi - 1e-6) and np.all(a["SC"] > 0.0)
    assert a["SC"][2] > a["SC"][0] + 0.2                     # 15 degC water vs 20 degC
    assert a["SC"][1] > a["SC"][0]                           # flooded plates subcool more
    assert np.allclose(a["T_co"], a["T_L"])                  # valve 3 takes the pool's liquid
    assert np.allclose(a["mdot_drn"], a["mdot_3"] - a["mdot_lv"], rtol=1e-3)
    x = res["x"].copy()
    x[:, HGBPPlant.U3] = 0.0
    _, b = pl.rhs(x, res["u"] * np.array([1.0, 1.0, 0.0, 1.0]), np.full(3, RATING["N"]), pl.T_amb, pl.T_wi,
                  want_aux=True)
    assert np.allclose(b["mdot_3"], 0.0, atol=1e-6) and np.allclose(b["T_co"], a["T_co"])


def test_plate_exchanger_pressure_drops():
    """Every exchanger side is a flow resistance: the header sits above the
    liquid pressure sensor by the condenser drop, the bypass gas enters S1 and
    the quench S3 above the tee pressure, and the cooling water shares the
    plant's supply-to-return difference with valve 4 and the piping.  The quench
    side splits into the distributor and ports (ahead of the channels, no effect
    on boiling) and the channels, where every cell boils at its own pressure;
    without flow only the static head of its column remains."""
    pl, res = _rating_state()
    assert res["converged"].all()
    a, p = res["aux"], pl.p
    assert 0.0 < a["dP_cr"][0] < 1e4
    assert 1e4 < a["dP_mg"][0] < 1e5
    C_w = series_C(kv_to_C(p.Kv_w) * valve_characteristic(a["u4"], p.w_char, p.w_R), kv_to_C(p.cond_Kv_w),
                   kv_to_C(p.Kv_wpipe))
    assert np.allclose(a["mdot_w"], C_w * np.sqrt(p.rho_w * (p.P_w_sup - p.P_w_ret)))
    assert np.allclose(a["dP_cw"], (a["mdot_w"] / kv_to_C(p.cond_Kv_w)) ** 2 / p.rho_w)
    # quench side: distributor + ports ahead, channels, ports after
    assert np.allclose(a["dP_mq"], a["dP_q_in"] + a["dP_mq_ch"] + a["dP_q_out"])
    assert 0.0 < a["dP_q_dist"][0] < a["dP_q_in"][0] and 0.0 < a["dP_mq_ch"][0] < a["dP_mq"][0] < 5e4
    assert np.all(np.diff(a["dP_qc"], axis=1) < 0.0)                # pressure falls along the quench flow
    assert np.all(a["dP_qc"][:, -1] > a["dP_q_out"])                # channels sit above the outlet port
    wet = a["x_q"][0] < 1.0
    assert np.all(a["T_q"][0][wet] > a["T_sat_s"][0])               # and boil warmer than the tee
    # valve 3 closed: no flow, the quench column's static head only
    x = res["x"].copy()
    x[:, HGBPPlant.U3] = 0.0
    _, b = pl.rhs(x, res["u"] * np.array([1.0, 1.0, 0.0, 1.0]), np.full(1, RATING["N"]), pl.T_amb, pl.T_wi,
                  want_aux=True)
    rho_q = pl._suction_cells(b["P_s"], b["h_q"])["rho"]
    assert np.allclose(b["dP_mq"], -9.81 * p.mx_H / pl.MX * rho_q.sum(1))


def test_pipe_pressure_drops():
    """Refrigerant lines are friction + fittings resistances carrying their own flow:
    Churchill's friction factor reduces to 64/Re laminar and to smooth-pipe values
    turbulent; on the stand the suction line puts the tee (and the exchanger) above
    the compressor port, the discharge line and header sit between P_d and P_i."""
    Re_l, Re_t = 800.0, 1e5
    assert np.isclose(churchill_f(Re_l, 0.0), 64.0 / Re_l, rtol=1e-3)
    assert np.isclose(churchill_f(Re_t, 0.0), 0.0180, rtol=0.03)            # Moody chart, smooth
    d, L, K, rho, mu, m = 0.03, 4.0, 1.5, 30.0, 1.3e-5, 0.5
    u = m / (rho * 0.25 * np.pi * d * d)
    f = churchill_f(rho * u * d / mu, 1.5e-6 / d)
    assert np.isclose(pipe_drop(m, d, L, K, rho, mu), (f * L / d + K) * rho * u * u / 2.0)
    assert np.isclose(pipe_drop(-m, d, L, K, rho, mu), -pipe_drop(m, d, L, K, rho, mu))

    pl, res = _rating_state()
    a, p = res["aux"], pl.p
    assert np.allclose(a["P_tee"], a["P_s"] + a["dP_suc"]) and 2e3 < a["dP_suc"][0] < 5e4
    assert np.all(a["dP_qc"] > a["dP_suc"][:, None])                # the exchanger sits above the tee
    for k in ("dP_dis", "dP_hdr", "dP_bp", "dP_mog", "dP_moq", "dP_q", "dP_liq", "dP_drn"):
        assert np.all(a[k] > 0.0), k
    assert np.all(a["P_h"] > a["P_i"] + a["dP_cr"])                  # drain and header half above P_i
    # compressor stopped: no suction line drop
    x = res["x"].copy()
    x[:, HGBPPlant.N_] = 0.0
    _, b = pl.rhs(x, res["u"], np.zeros(1), pl.T_amb, pl.T_wi, want_aux=True)
    assert np.allclose(b["dP_suc"], 0.0) and np.allclose(b["P_tee"], b["P_s"])


@pytest.mark.parametrize("fluid", ["R454B", "R454C"])
def test_stand_runs_on_r454_blends(fluid):
    """The stand reaches equilibrium at a test point on R454B / R454C and stays there."""
    pl = HGBPPlant(PlantParams(fluid=fluid, **WIDE), n=1)
    pl.set_inputs(T_amb=298.15, T_wi=293.15)
    pt = _point(pl, "MT_standard", 3550.0)
    res = solve_steady_state(pl, pt["P_s"], pt["P_d"], pt["SH"], pt["N"], P_i=pt["P_i"])
    assert res["converged"][0]
    pl.set_state(0, res["x"])
    for _ in range(60):
        a = pl.step(1.0, u_cmd=res["u"], N_cmd=pt["N"])
    assert abs(a["P_s"][0] - pt["P_s"]) < 0.05e5 and abs(a["P_d"][0] - pt["P_d"]) < 0.2e5
