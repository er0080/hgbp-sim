import numpy as np

from hgbp_sim.interlock import ST_OFF, ST_RUNNING, ST_STARTING, ST_STOPPING, Interlock
from hgbp_sim.live import LiveStand


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


def test_live_stand_operation():
    st = LiveStand(seed=1, dt_ctrl=1.0)
    s = st.snapshot()
    assert s["compressor"]["state"] == "OFF" and s["compressor"]["permissive_ok"]
    st.set_compressor(run=True, speed=1450.0)
    for _ in range(300):
        s = st.step()
    assert s["compressor"]["state"] == "RUNNING"
    assert abs(s["meas"]["P_s"] - s["loops"]["spv"]["sp"]) < 0.1
    assert abs(s["meas"]["P_d"] - s["loops"]["dpv"]["sp"]) < 0.3
    # manual mode with an explicit output, then back to auto (bumpless)
    st.set_loop("spv", mode="manual", out=0.7)
    for _ in range(30):
        s = st.step()
    assert abs(s["meas"]["u2"] - 0.7) < 0.02
    st.set_loop("spv", mode="auto")
    s = st.step()
    assert abs(s["meas"]["u2"] - 0.7) < 0.05
    # charging changes the inventory
    m0 = s["charge"]["kg"]
    st.add_charge(0.1)
    for _ in range(40):
        s = st.step()
    assert abs(s["charge"]["kg"] - (m0 + 0.1)) < 5e-3
    # live and deferred parameters
    r = st.set_params({"UA_r_2ph": 2500.0, "V_i": 4e-3})
    assert r["applied"] == ["UA_r_2ph"] and r["deferred"] == ["V_i"]
    assert float(st.plant.p.UA_r_2ph[0]) == 2500.0
    # closing the water valve in manual trips the stand, reset requires it to be off
    st.set_loop("water", mode="manual", out=0.0)
    for _ in range(1200):
        s = st.step()
        if s["compressor"]["tripped"]:
            break
    assert s["compressor"]["tripped"] and "high_P_d" in s["compressor"]["trip_reasons"]
    for _ in range(40):
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
