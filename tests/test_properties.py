import numpy as np
import pytest

from hgbp_sim.properties import PHASE_LIQUID, PHASE_TWOPHASE, PHASE_VAPOR, get_tables

C2K = 273.15


@pytest.fixture(scope="module")
def tab():
    return get_tables("R134a")


def test_saturation_monotonic(tab):
    P = np.linspace(tab.p_min * 1.01, tab.p_max * 0.99, 200)
    s = tab.sat(P)
    assert np.all(np.diff(s["T_l"]) > 0)
    assert np.all(s["h_v"] > s["h_l"])
    assert np.all(s["rho_l"] > s["rho_v"])
    T = tab.T_sat(P)
    assert np.allclose(tab.P_sat(T), P, rtol=2e-3)


def test_phase_classification_and_continuity(tab):
    P = np.full(3, 5e5)
    s = tab.sat(P)
    h = np.array([s["h_l"][0] - 5e3, 0.5 * (s["h_l"][0] + s["h_v"][0]), s["h_v"][0] + 5e3])
    st = tab.state(P, h, need_s=True)
    assert list(st.phase) == [PHASE_LIQUID, PHASE_TWOPHASE, PHASE_VAPOR]
    assert abs(st.T[1] - s["T_l"][0]) < 1e-6
    assert st.T[0] < st.T[1] < st.T[2]
    # density derivative signs
    assert np.all(st.drho_dP > 0) and np.all(st.drho_dh < 0)
    # continuity across the dew line
    eps = 20.0
    a = tab.state(np.array([5e5]), np.array([s["h_v"][0] - eps]))
    b = tab.state(np.array([5e5]), np.array([s["h_v"][0] + eps]))
    assert abs(a.rho[0] - b.rho[0]) / a.rho[0] < 2e-3
    assert abs(a.T[0] - b.T[0]) < 0.1


def test_inversions_roundtrip(tab):
    rng = np.random.default_rng(0)
    P = np.exp(rng.uniform(np.log(1e5), np.log(20e5), 300))
    s = tab.sat(P)
    h = s["h_v"] + rng.uniform(2e3, 80e3, 300)
    st = tab.state(P, h, need_s=True)
    assert np.abs(tab.h_Ps_vapor(P, st.s) - h).max() < 200.0
    assert np.abs(tab.h_PT(P, st.T) - h).max() < 200.0
    assert np.abs(tab.T_vapor(P, h) - st.T).max() < 1e-9
    hl = s["h_l"] - rng.uniform(1e3, 40e3, 300)
    stl = tab.state(P, hl)
    assert np.abs(tab.h_PT(P, stl.T) - hl).max() < 200.0


def test_against_coolprop(tab):
    CP = pytest.importorskip("CoolProp.CoolProp")
    rng = np.random.default_rng(1)
    P = np.exp(rng.uniform(np.log(0.5e5), np.log(25e5), 500))
    hl = CP.PropsSI("Hmass", "P", P, "Q", 0, "R134a")
    hv = CP.PropsSI("Hmass", "P", P, "Q", 1, "R134a")
    h = hl - 30e3 + rng.uniform(0, 1, 500) * (hv - hl + 100e3)
    st = tab.state(P, h, need_s=True)
    T = CP.PropsSI("T", "P", P, "Hmass", h, "R134a")
    rho = CP.PropsSI("Dmass", "P", P, "Hmass", h, "R134a")
    assert np.abs(st.T - T).max() < 0.1
    assert (np.abs(st.rho - rho) / rho).max() < 2e-3
    m = st.phase != PHASE_TWOPHASE
    d1 = CP.PropsSI("d(Dmass)/d(P)|Hmass", "P", P[m], "Hmass", h[m], "R134a")
    assert (np.abs(st.drho_dP[m] - d1) / np.abs(d1)).max() < 0.03


def test_two_phase_derivatives_consistent(tab):
    """drho/dP|h and drho/dh|P in the dome must match finite differences of rho."""
    P = np.full(50, 8e5)
    s = tab.sat(P)
    x = np.linspace(0.05, 0.95, 50)
    h = s["h_l"] + x * (s["h_v"] - s["h_l"])
    st = tab.state(P, h)
    dP, dh = 50.0, 20.0
    fd_P = (tab.state(P + dP, h).rho - tab.state(P - dP, h).rho) / (2 * dP)
    fd_h = (tab.state(P, h + dh).rho - tab.state(P, h - dh).rho) / (2 * dh)
    assert np.allclose(st.drho_dP, fd_P, rtol=2e-2)
    assert np.allclose(st.drho_dh, fd_h, rtol=2e-2)
