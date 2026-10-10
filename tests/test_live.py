import os

import numpy as np
import pytest

from hgbp_sim.interlock import ST_OFF, ST_RUNNING, ST_STARTING, ST_STOPPING, Interlock
from hgbp_sim.live import HISTORY_CHANNELS, LiveStand


def test_interlock_sequence():
    il = Interlock(1, min_off_time=10.0, min_run_time=20.0)
    perm = np.array([True])
    N_min = np.array([600.0])
    r = il.step(np.array([True]), perm, np.array([0.0]), np.array([1450.0]), N_min, 1.0)
    assert il.state[0] == ST_STARTING and r["N_cmd"][0] == 1450.0
    il.step(np.array([True]), perm, np.array([1400.0]), np.array([1450.0]), N_min, 1.0)
    assert il.state[0] == ST_RUNNING
    r = il.step(np.array([False]), perm, np.array([1450.0]), np.array([1450.0]), N_min, 1.0)
    assert il.state[0] == ST_RUNNING                     # minimum run time not elapsed
    il.t_state[:] = 25.0
    r = il.step(np.array([False]), perm, np.array([1450.0]), np.array([1450.0]), N_min, 1.0)
    assert il.state[0] == ST_STOPPING and r["N_cmd"][0] == 0.0
    il.step(np.array([False]), perm, np.array([100.0]), np.array([1450.0]), N_min, 1.0)
    assert il.state[0] == ST_OFF
    r = il.step(np.array([True]), perm, np.array([0.0]), np.array([1450.0]), N_min, 1.0)
    assert il.state[0] == ST_OFF and r["blocked"][0]     # anti-short-cycle
    il.tripped[:] = True
    il.t_state[:] = 100.0
    r = il.step(np.array([True]), perm, np.array([0.0]), np.array([1450.0]), N_min, 1.0)
    assert il.state[0] == ST_OFF and r["blocked"][0]
    il.acknowledge()
    assert not il.tripped[0]


REST = {"dpv": 0.5, "spv": 0.6, "stv": 0.0, "water": 0.3}      # valve positions for a start


def test_loops_control_while_off_and_short_cycle_timers():
    st = LiveStand(seed=3, dt_ctrl=0.25)
    st.noise = False
    for _ in range(40):            # 10 s off, loops in AUTO: standstill P_d is far below its SP,
        s = st.step()              # so the direct-acting discharge loop closes valve 1
    assert s["compressor"]["state"] == "OFF" and s["meas"]["u1"] < 0.2
    assert not s["compressor"]["permissives"]["valve1_open"]
    # anti-short-cycle timers: a stop right after a start is honoured only without them
    for k, out in REST.items():
        st.set_loop(k, mode="manual", out=out)
    for flag, expect in ((True, "RUNNING"), (False, "OFF")):
        st.cold_start()
        st.set_sim(short_cycle_timers=flag)
        assert st.snapshot()["compressor"]["min_run_time"] == (120.0 if flag else 0.0)
        st.set_compressor(run=True, speed=1450.0)
        for _ in range(40):
            st.step()
        st.set_compressor(run=False)
        for _ in range(80):
            s = st.step()
        assert s["compressor"]["state"] == expect, flag


def test_live_stand_operation():
    st = LiveStand(seed=1, dt_ctrl=0.25)
    s = st.snapshot()
    assert s["compressor"]["state"] == "OFF" and s["compressor"]["permissive_ok"]
    # start procedure: valves at their start positions in MAN, AUTO once the compressor turns
    for k, out in REST.items():
        st.set_loop(k, mode="manual", out=out)
    st.set_compressor(run=True, speed=1450.0)
    for _ in range(1200):          # 300 s
        s = st.step()
        if s["loops"]["dpv"]["mode"] == "manual" and s["meas"]["N"] > 600.0:
            for k in REST:
                st.set_loop(k, mode="auto")
    assert s["compressor"]["state"] == "RUNNING"
    assert abs(s["meas"]["P_s"] - s["loops"]["spv"]["sp"]) < 0.1
    assert abs(s["meas"]["P_d"] - s["loops"]["dpv"]["sp"]) < 0.3
    # manual mode with an explicit output, then back to auto (bumpless)
    st.set_loop("spv", mode="manual", out=0.7)
    for _ in range(120):           # 30 s
        s = st.step()
    assert abs(s["meas"]["u2"] - 0.7) < 0.02
    st.set_loop("spv", mode="auto")
    s = st.step()
    # bumpless: the valve leaves the manual output at most at its slew limit, no jump
    assert abs(s["meas"]["u2"] - 0.7) <= st.params.rate_spv * st.dt_ctrl + 1e-9
    # charging changes the inventory
    m0 = s["charge"]["kg"]
    st.add_charge(0.1)
    for _ in range(160):           # 40 s
        s = st.step()
    assert abs(s["charge"]["kg"] - (m0 + 0.1)) < 5e-3
    # live and deferred parameters
    r = st.set_params({"alpha_r_2ph": 1200.0, "L_liq": 5.0})
    assert r["applied"] == ["alpha_r_2ph"] and r["deferred"] == ["L_liq"]
    assert float(st.plant.p.alpha_r_2ph[0]) == 1200.0
    # closing the water valve in manual trips the stand, reset requires it to be off
    st.set_loop("water", mode="manual", out=0.0)
    for _ in range(4800):          # 1200 s
        s = st.step()
        if s["compressor"]["tripped"]:
            break
    # (with the stand's 700 psia / 320 degF limits the discharge temperature may trip first)
    assert s["compressor"]["tripped"] and {"high_P_d", "high_T_d"} & set(s["compressor"]["trip_reasons"])
    for _ in range(160):           # 40 s
        s = st.step()
    assert s["compressor"]["state"] == "OFF"
    st.set_compressor(reset=True)
    assert not st.snapshot()["compressor"]["tripped"]
    h = st.history_since(-1.0)
    assert len(h["t"]) == st.step_count and np.all(np.diff(h["t"]) > 0)


def test_live_warm_start_applies_pending_fluid():
    st = LiveStand(seed=2, dt_ctrl=1.0)
    st.set_params({"fluid": "R404A"})
    assert st.warm_start("MT_standard")
    s = st.snapshot()
    assert s["fluid"] == "R404A" and s["compressor"]["state"] == "RUNNING"
    assert abs(s["meas"]["SH"] - 10.0) < 0.2


def test_failed_warm_start_leaves_the_stand_running():
    """A warm start without an equilibrium leaves the running stand as it was, with deferred
    parameters still pending (it used to leave a rebuilt, uninitialized plant behind)."""
    st = LiveStand(seed=2, dt_ctrl=0.5)
    st.noise = False
    assert st.warm_start("MT_standard")
    st.step(snapshot=False)
    st.set_params({"charge": 2.0})                     # far too little refrigerant for the point
    plant, x0, t0 = st.plant, st.plant.x.copy(), st.t
    assert not st.warm_start("MT_standard")
    assert st.plant is plant and np.array_equal(st.plant.x, x0) and st.t == t0
    assert st.pending_params == {"charge": 2.0} and "failed" in st.events[0]["msg"]
    s = st.step()
    assert s["compressor"]["state"] == "RUNNING" and np.isfinite(s["meas"]["P_s"])


def test_shipped_defaults_hold_their_setpoints():
    """The stand as configured in webui/config/stand_defaults.json (plant and UT35A
    tuning) warm-starts at its own loop setpoints and holds them in AUTO."""
    import os
    from hgbp_sim.defaults import load_defaults
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "webui", "config", "stand_defaults.json")
    d = load_defaults(path, create=False)
    st = LiveStand(defaults=d)
    st.noise = False
    L, pr = d["loops"], st.props
    T = lambda bar: float(pr.T_sat(np.array([bar * 1e5]))[0]) - 273.15
    T_evap = T(L["spv"]["SP"])
    assert st.warm_start(dict(T_evap=T_evap, T_cond=T(L["dpv"]["SP"]), T_int=T(L["water"]["SP"]),
                              SH=L["stv"]["SP"] - T_evap, N=st.params.N_nom))
    for _ in range(int(300 / st.dt_ctrl)):
        s = st.step()
    assert s["compressor"]["state"] == "RUNNING" and not s["compressor"]["tripped"]
    for k in ("dpv", "spv", "water"):
        assert abs(s["loops"][k]["pv"] - L[k]["SP"]) < 0.05, k
    assert abs(s["loops"]["stv"]["pv"] - L["stv"]["SP"]) < 0.2
    assert s["meas"]["SC"] > 0.0 and s["meas"]["T_co"] >= s["meas"]["T_wi"] - 0.5


def test_incremental_history_rows():
    """The UI stream takes each history row exactly once, however far it falls behind,
    and resumes after the last row a client holds; past the history length only the
    rows still held come back."""
    st = LiveStand(history_len=50)
    st.cold_start()
    count = st.rows_recorded
    got = []
    for burst in (1, 3, 7):
        for _ in range(burst):
            st.step()
        rows, count = st.rows_after(count)
        got += rows["t"]
        assert set(rows) == set(HISTORY_CHANNELS) and all(len(v) == len(rows["t"]) for v in rows.values())
    assert np.allclose(np.diff(got), st.dt_ctrl) and got[-1] == st.t
    assert st.rows_after(count) == ({}, count)
    t_mid = got[5]
    rows, _ = st.rows_after(st.rows_through(t_mid))
    assert rows["t"] == got[6:]
    for _ in range(80):                        # beyond the history length
        st.step()
    rows, count = st.rows_after(count)
    assert len(rows["t"]) == 50 and rows["t"][-1] == st.t


def test_ph_diagram_state_points():
    """The P-h diagram's state points follow the pressure chain and the processes along
    each flow path."""
    st = LiveStand(seed=1, dt_ctrl=0.5)
    st.noise = False
    assert st.warm_start("MT_standard")
    for _ in range(4):
        st.step(snapshot=False)
    s = st.last_snapshot()
    ph = {k: np.array(v) for k, v in s["ph"].items()}
    assert all(np.isfinite(v).all() for v in ph.values())
    P = {k: v[0] for k, v in ph.items()}
    h = {k: v[1] for k, v in ph.items()}
    assert np.isclose(P["suc"], s["meas"]["P_s"], rtol=1e-4) and np.isclose(P["dis"], s["meas"]["P_d"], rtol=1e-4)
    assert np.isclose(P["rec"], s["meas"]["P_i"], rtol=1e-4)
    # pressure falls along the flow: discharge -> valve 1 -> header -> valve 2 / 3 -> tee -> compressor
    assert P["dis"] > P["v1i"] > P["hdr"] >= P["cin"]
    assert P["rec"] > P["v3i"] > P["v3o"] > P["qin"] > P["qout"] > P["tee"] > P["suc"]
    assert P["hdr"] > P["v2o"] > P["gin"] > P["gout"] > P["tee"]
    # valves throttle at constant enthalpy; the condenser and the gas side cool, the quench heats
    assert h["v3i"] == h["v3o"] and h["hdr"] == h["v2o"] == h["gin"]
    assert all(np.diff([h["cin"]] + [h[f"c{j}"] for j in range(5)]) < 0)
    assert all(np.diff([h["gin"]] + [h[f"g{j}"] for j in (4, 3, 2, 1, 0)]) < 0)
    assert all(np.diff([h["qin"]] + [h[f"q{j}"] for j in range(5)]) > 0)
    assert min(h["gout"], h["qout"]) < h["tee"] < max(h["gout"], h["qout"])
    # cells are the state leaving them, where it leaves: the last cell of each exchanger side
    # is its outlet (the quench side's S4, the gas side's S2, the condenser's drain at S4)
    assert np.array_equal(ph["q4"], ph["qout"]) and np.array_equal(ph["g0"], ph["gout"])
    aux = st._last[0]
    assert np.isclose(P["c4"], (aux["P_i"][0] + aux["dP_drn"][0]) / 1e5, rtol=1e-6)   # condenser S4
    assert ph["c4"][3] < 0.0                                                      # the drain, subcooled
    t = s["true"]
    assert np.allclose(t["x_c"], [ph[f"c{j}"][3] for j in range(5)], atol=1e-4)
    assert np.allclose(t["T_q"], [ph[f"q{j}"][2] for j in range(5)], atol=1e-3)
    assert np.allclose(t["T_g"], [ph[f"g{j}"][2] for j in range(5)], atol=1e-3)
    assert np.isclose(t["T_go"], ph["gout"][2], atol=1e-3) and np.isclose(t["T_qo"], ph["qout"][2], atol=1e-3)
    # the compression: motor heating at suction pressure, a polytropic path to the adiabatic
    # end, the shell and the discharge volume at discharge pressure; the isentrope from 1 ends at 2s
    assert P["cmh"] == P["suc"] and h["cmh"] > h["suc"] and P["c2a"] == P["cout"] == P["dis"]
    assert h["suc"] < h["c2s"] < h["c2a"]
    pa = s["ph_paths"]
    comp, isen = np.array(pa["comp"]), np.array(pa["isen"])
    assert np.allclose(comp[0], ph["cmh"][:2], atol=1e-3) and np.allclose(comp[-1], ph["c2a"][:2], atol=1e-3)
    assert np.allclose(isen[0], ph["suc"][:2], atol=1e-3) and np.allclose(isen[-1], ph["c2s"][:2], atol=1e-3)
    assert np.all(np.diff(comp, axis=0) > 0.0) and 0.5 < pa["eta_p"] < 1.0
    s_comp = st.props.state(comp[:, 0] * 1e5, comp[:, 1] * 1e3).s
    assert np.all(np.diff(s_comp) > 0.0)                      # the real path gains entropy
    s_isen = st.props.state(isen[:, 0] * 1e5, isen[:, 1] * 1e3).s
    assert np.ptp(s_isen) < 1e-3 * s_isen[0]
    # receiver liquid subcooled, the quench flashed by valve 3, the suction gas at the setpoint superheat
    assert ph["rec"][3] < 0.0 and np.isclose(ph["rec"][4], s["true"]["SC_L"], atol=0.05)
    assert 0.0 < ph["v3o"][3] < 1.0
    assert ph["suc"][3] > 1.0 and abs(ph["suc"][4] - 10.0) < 0.5


def test_ph_chart():
    st = LiveStand()
    c = st.ph_chart([-20.0, 0.0, 40.0, 120.0])
    assert c["fluid"] == "R410A" and len(c["P"]) == len(c["h_l"]) == len(c["h_v"])
    assert np.all(np.array(c["h_v"]) > np.array(c["h_l"])) and np.all(np.diff(c["P"]) > 0)
    assert c["P_max"] < c["P_crit"] and c["dome_ext"]["P"][-1] == pytest.approx(c["P_crit"])
    iso = {i["T"]: i for i in c["isotherms"]}
    assert set(iso) == {-20.0, 0.0, 40.0, 120.0}
    for T in (0.0, 40.0):                   # meets the bubble line at the bubble and the dew line at the dew pressure
        P_b, P_d = float(st.props.P_sat(T + 273.15)), st.saturation(T)["P"] * 1e5
        Pi, hi = np.array(iso[T]["P"]), np.array(iso[T]["h"])
        assert np.all(np.diff(Pi) <= 0)
        for P_x, h_x in ((P_b, st.props.sat(P_b)["h_l"]), (P_d, st.props.sat(P_d)["h_v"])):
            assert np.min(np.hypot(Pi / (P_x / 1e5) - 1.0, (hi - h_x / 1e3) / 100.0)) < 2e-3
    assert np.all(np.array(iso[120.0]["h"]) > np.interp(iso[120.0]["P"], c["P"], c["h_v"]))   # all vapor
    # isentropes: round entropies in the view, each line at its entropy
    zs = st.ph_chart(view=(400.0, 520.0, 4.0, 40.0), isentropes=True)
    S = np.array([i["s"] for i in zs["isentropes"]])
    assert len(S) >= 4 and np.allclose(S / 1e3 / zs["isen_step"], np.round(S / 1e3 / zs["isen_step"]), atol=1e-6)
    for line in zs["isentropes"][::3]:
        sv = st.props.state(np.array(line["P"]) * 1e5, np.array(line["h"]) * 1e3).s
        assert np.allclose(sv, line["s"], rtol=2e-3)
    # a zoomed view: its own pressure grid and round isotherms across it
    z = st.ph_chart(view=(400.0, 460.0, 5.0, 7.0), unit="C", n_iso=12)
    assert 5.0 / 1.06 < z["P"][0] < z["P"][-1] < 7.0 * 1.06 and "dome_ext" not in z
    Tz = [i["T"] for i in z["isotherms"]]
    assert z["iso_step"] in (2.0, 5.0) and np.allclose(Tz, np.round(np.array(Tz) / z["iso_step"]) * z["iso_step"])
    assert 4 <= len(Tz) <= 14
    zf = st.ph_chart(view=(400.0, 460.0, 5.0, 7.0), unit="F")
    TF = np.array([i["T"] for i in zf["isotherms"]]) * 1.8 + 32.0
    assert np.allclose(TF, np.round(TF / zf["iso_step"]) * zf["iso_step"], atol=1e-4)


def test_saturation_calculator():
    """Dew point (x = 1) pressure of the stand's refrigerant (UI calculator) against CoolProp."""
    CP = pytest.importorskip("CoolProp.CoolProp")
    st = LiveStand()
    for T in (-20.0, 0.0, 40.0):
        s = st.saturation(T)
        assert s["fluid"] == "R410A" and s["in_range"]
        assert np.isclose(s["P"], CP.PropsSI("P", "T", T + 273.15, "Q", 1, "R410A") / 1e5, rtol=2e-3)
    assert not st.saturation(150.0)["in_range"]
    st.params = st.params.replace(fluid="R407C")        # a blend: the dew line, not the bubble line
    st.apply_pending()
    s = st.saturation(0.0)
    assert np.isclose(s["P"], CP.PropsSI("P", "T", 273.15, "Q", 1, "R407C") / 1e5, rtol=2e-3)


def test_pv_input_filter():
    """UT35A FL: a first-order lag on each controller's PV, ahead of the faceplate and
    the PID; the plant's measurements and the trend history stay unfiltered."""
    from hgbp_sim import ut35a
    assert ut35a.normalize_filter(4) == 4 and ut35a.normalize_filter("off") == ut35a.OFF
    with pytest.raises(ValueError):
        ut35a.normalize_filter(121)
    st = LiveStand()
    assert all(st.tuning[k]["FL"] == ut35a.OFF for k in st.tuning)   # built-in gains: tuned without it
    st.set_loop("spv", FL=4)
    # step response of the filter itself: exactly first order with time constant FL
    st.pv_filt["spv"] = 0.0
    y = [st._filter_pv("spv", 1.0, 0.2) for _ in range(20)]         # 4 s
    assert np.isclose(y[-1], 1.0 - np.exp(-1.0))
    # with sensor noise the faceplate PV is smoother than the measurement
    assert st.warm_start("MT_standard")
    st.noise = True
    fp, meas = [], []
    for _ in range(300):
        s = st.step()
        fp.append(s["loops"]["spv"]["pv"])
        meas.append(s["meas"]["P_s"])
    assert np.std(np.diff(fp)) < 0.3 * np.std(np.diff(meas))
    assert st.last_row()["P_s"] != s["loops"]["spv"]["pv"]            # the history is not filtered
    st.set_loop("spv", FL="OFF")
    s = st.step()
    assert s["loops"]["spv"]["pv"] == s["meas"]["P_s"] and s["loops"]["spv"]["FL"] == "OFF"


def test_stand_defaults_set_the_pv_filter():
    """The stand's controllers run with FL = 4 (defaults file), and its tuning copes with it."""
    from hgbp_sim.defaults import load_defaults
    path = os.path.join(os.path.dirname(__file__), "..", "webui", "config", "stand_defaults.json")
    d = load_defaults(path, create=False)
    assert all(d["loops"][k]["FL"] == 4 for k in d["loops"])
