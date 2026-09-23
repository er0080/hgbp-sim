import json
import os

import numpy as np
import pytest

from hgbp_sim import ut35a
from hgbp_sim.control import DEFAULT_GAINS
from hgbp_sim.defaults import builtin_defaults, load_defaults, merge_defaults
from hgbp_sim.live import LOOPS, LiveStand

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ut35a_conversion():
    # P = 100 % of a 100-unit span: 1 unit of error moves the output 1 %
    kp, ki, kd = ut35a.to_gains(100.0, 20, "OFF", "RVS", 100.0)
    assert kp == pytest.approx(0.01) and ki == pytest.approx(0.01 / 20) and kd == 0.0
    kp, ki, kd = ut35a.to_gains(50.0, "OFF", 4, "DIR", 100.0)      # DIR: output rises with PV
    assert kp == pytest.approx(-0.02) and ki == 0.0 and kd == pytest.approx(-0.08)
    assert ut35a.from_gains(kp, ki, kd, 100.0) == dict(P=50.0, I="OFF", D=4, DR="DIR")
    # controller resolution and ranges
    assert ut35a.normalize_band(12.34) == 12.3 and ut35a.normalize_time(29.6, "I") == 30
    assert ut35a.normalize_time(None, "I") == ut35a.normalize_time(0, "I") == ut35a.normalize_time("off", "I") == "OFF"
    for bad in (lambda: ut35a.normalize_band(0.0), lambda: ut35a.normalize_band(1000.0),
                lambda: ut35a.normalize_time(0.4, "I"), lambda: ut35a.normalize_time(6001, "D"),
                lambda: ut35a.normalize_action("REV")):
        with pytest.raises(ValueError):
            bad()


def test_builtin_tuning_matches_the_tuned_gains():
    d = builtin_defaults()
    for k, info in LOOPS.items():
        L = d["loops"][k]
        gains = ut35a.to_gains(L["P"], L["I"], L["D"], L["DR"], L["RH"] - L["RL"])
        for name, g in zip(("Kp", "Ki", "Kd"), gains):
            assert g / info["scale"] == pytest.approx(DEFAULT_GAINS[k][name], rel=3e-3)


def test_merge_and_validation(tmp_path):
    assert merge_defaults(json.loads(json.dumps(builtin_defaults()))) == merge_defaults({})
    d = merge_defaults({"loops": {"water": {"P": 8.0, "I": "OFF"}}, "plant": {"Kv_w": 10}, "simulation": {"T_wi": 15}})
    assert d["loops"]["water"]["P"] == 8.0 and d["loops"]["water"]["I"] == "OFF"
    assert d["loops"]["water"]["RH"] == builtin_defaults()["loops"]["water"]["RH"]      # untouched keys kept
    assert d["plant"]["Kv_w"] == 10.0 and d["simulation"]["T_wi"] == 15.0
    for bad in ({"plant": {"Kv_ww": 1.0}},                      # typo
                {"loops": {"water": {"Pb": 8.0}}},
                {"loops": {"spv": {"SP": 45.0}}},               # SP outside RL..RH
                {"loops": {"dpv": {"RL": 60.0}}},               # RL above RH
                {"plant": {"Kv_w": 500.0}},                     # outside the editable range
                {"plant": {"fluid": "R999"}},
                {"simulation": {"noise": "yes"}}):
        with pytest.raises(ValueError):
            merge_defaults(bad)
    # a missing file is created with the built-in defaults, then read back
    path = tmp_path / "cfg" / "stand_defaults.json"
    assert load_defaults(str(path)) == merge_defaults({}) and path.exists()
    path.write_text("{ not json")
    with pytest.raises(ValueError):
        load_defaults(str(path))


def test_shipped_defaults_file_is_valid():
    load_defaults(os.path.join(REPO, "webui", "config", "stand_defaults.json"), create=False)


def test_live_stand_tuning_and_defaults():
    d = merge_defaults({"loops": {"water": {"SP": 20.0, "P": 8.0}}, "simulation": {"dt_ctrl": 0.5}})
    st = LiveStand(defaults=d)
    s = st.snapshot()
    assert st.dt_ctrl == 0.5 and s["loops"]["water"]["sp"] == pytest.approx(20.0) and s["loops"]["water"]["P"] == 8.0
    kp = ut35a.to_gains(8.0, 30, "OFF", "DIR", 50.0)[0] / 1e5
    assert float(st.ctrl.pid_4.Kp) == pytest.approx(kp) and float(st.ctrl.pid_4.u_min) == pytest.approx(0.02)
    # a live change goes to the PID; an invalid request changes nothing
    st.set_loop("spv", P=11.1, I=10)
    assert float(st.ctrl.pid_2.Kp) == pytest.approx(100.0 / (11.1 * 30.0) / 1e5)
    assert float(st.ctrl.pid_2.Ki) == pytest.approx(float(st.ctrl.pid_2.Kp) / 10.0)
    with pytest.raises(ValueError):
        st.set_loop("spv", P=20.0, I=9000)
    with pytest.raises(ValueError):
        st.set_loop("spv", sp=45.0)
    assert st.tuning["spv"]["P"] == 11.1 and np.isclose(st.sp["P_s"], 9.98e5)
    # tuning survives a re-initialization
    st.cold_start()
    assert st.tuning["spv"]["P"] == 11.1
