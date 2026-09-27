"""The model against its reference data (tests/reference/reference.npz, made by
tests/reference/make_reference.py; regenerate it after an intentional change of
the physics).

* pointwise: at the recorded states the right-hand side and the auxiliary
  outputs must agree to rounding;
* trajectories: the scenarios rerun must follow the recorded ones (the step
  size control is per environment, so batched runs may take different sub-steps).
"""
import os
import sys

import numpy as np
import pytest

from hgbp_sim import HGBPPlant, PlantParams

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "reference"))
import make_reference as mr  # noqa: E402

REF = np.load(os.path.join(HERE, "reference", "reference.npz"))
SCENARIOS = sorted({k.split("/")[0] for k in REF.files if not k.startswith("ss_")})


def _ref(sc, k):
    return REF[f"{sc}/{k}"]


def _plant_for(sc):
    if sc == "batch":
        return HGBPPlant(PlantParams(), n=6, dt=0.05, rng=np.random.default_rng(1), randomize=True)
    return HGBPPlant(PlantParams(), n=1)


def _rel_err(a, b):
    """Error relative to each value, floored at 1e-6 of the column's largest magnitude."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    floor = np.abs(b).max(axis=0, keepdims=True) * 1e-6 + 1e-12
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


@pytest.mark.parametrize("sc", SCENARIOS)
def test_rhs_matches_reference_pointwise(sc):
    pl = _plant_for(sc)
    pp = pl.params_subset(_ref(sc, "pt_env"))
    X = _ref(sc, "pt_x")
    args = (_ref(sc, "pt_u"), _ref(sc, "pt_N"), _ref(sc, "pt_Tamb"), _ref(sc, "pt_Twi"))
    dx, a = pl.rhs(X, *args, want_aux=True, p=pp)
    dxh = pl.rhs(X, *args, p=pp, hold_P_s=True)
    assert _rel_err(dx, _ref(sc, "pt_dx")).max() < 1e-9
    assert _rel_err(dxh, _ref(sc, "pt_dxh")).max() < 1e-9
    for k in mr.AUX_POINT:
        assert _rel_err(a[k], _ref(sc, f"pt_aux_{k}")).max() < 1e-9, k


# trajectory tolerance per channel: absolute, or relative to the channel's range over the run
TRAJ_TOL = dict(P_s=2e3, P_d=5e3, P_i=5e3, T_s=0.1, T_d=0.1, T_co=0.1, SH=0.1, SC=0.1, T_wo=0.05,
                T_sh=0.05, T_cw=0.05, N=1.0, M_tot=1e-6)


@pytest.fixture(scope="module")
def reruns():
    return {rec.name: rec for rec, _ in mr.scenarios("hgbp_sim")}


@pytest.mark.parametrize("sc", SCENARIOS)
def test_trajectories_follow_reference(sc, reruns):
    rec = reruns[sc]
    assert np.allclose(np.array(rec.t), _ref(sc, "t"))
    rows = np.array(rec.rows)
    for j, k in enumerate(mr.CHANNELS):
        new, old = rows[:, j], _ref(sc, k)
        rng = float(np.ptp(old))
        tol = TRAJ_TOL.get(k, 0.01 * rng + 1e-9)
        err = np.abs(new - old).max()
        assert err <= tol, f"{sc}/{k}: max deviation {err:.3g} (tolerance {tol:.3g})"


SS_CASES = sorted({k.split("/")[0][3:] for k in REF.files if k.startswith("ss_")})


@pytest.fixture(scope="module")
def steady():
    return mr.steady_cases("hgbp_sim")


@pytest.mark.parametrize("case", SS_CASES)
def test_steady_state_matches_reference(case, steady):
    g = lambda k: REF[f"ss_{case}/{k}"]
    new = lambda k: steady[f"ss_{case}/{k}"]
    assert np.array_equal(new("converged"), g("converged"))
    ok = g("converged") > 0.5
    # single stands follow the reference iteration step by step; a batch's environments
    # used to iterate together and now iterate on their own (same tolerance, other path)
    tol_u, tol_x = (1e-9, 1e-9) if len(ok) == 1 else (1e-4, 1e-3)
    assert np.abs(new("u") - g("u"))[ok].max() < tol_u
    assert _rel_err(new("x"), g("x"))[ok].max() < tol_x
    for k in ("P_s", "P_d", "P_i", "M_tot"):
        assert _rel_err(new(f"aux_{k}"), g(f"aux_{k}"))[ok].max() < 1e-6, k
