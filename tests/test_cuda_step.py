"""The CUDA port of the step (hgbp_sim.kernel.cuda_step) against the CPU kernel.

Opt-in: needs an NVIDIA GPU and the CUDA toolkit, and builds the library on the first
run (about a minute).  Run with ``HGBP_CUDA_TESTS=1 pytest tests/test_cuda_step.py``.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("HGBP_CUDA_TESTS") != "1",
                                reason="CUDA tests are opt-in (HGBP_CUDA_TESTS=1)")


def _replay(n_ctrl=12):
    from hgbp_sim.kernel import model as km
    from hgbp_sim.params import PlantParams
    from hgbp_sim.plant import HGBPPlant

    pl = HGBPPlant(PlantParams(), n=6, rng=np.random.default_rng(3), randomize=True)
    pl.cold_start()                                   # the compressor runs up from rest
    u = np.array([[0.9, 0.3, 0.1, 0.4]] * pl.n)
    u[:, 2] = np.linspace(0.0, 0.3, pl.n)
    N = np.full(pl.n, pl.params.N_nom)
    pl.rhs(pl.x, u, N, pl.T_amb, pl.T_wi)
    x0, lam0 = pl.x.copy(), pl._lam_last.copy()
    x, lam, subs = x0.copy(), lam0.copy(), np.zeros(pl.n, np.int64)
    for _ in range(n_ctrl):
        km.step_batch[False](x, u, N, pl.T_amb, pl.T_wi, pl.p.packed(), pl.props.tab, lam, pl.dt, 5, 0, subs)
    return pl, u, N, x0, lam0, x


def _gpu(pl, u, N, x0, lam0, n_ctrl, **kw):
    from hgbp_sim.kernel.cuda_step import CudaStep

    g = CudaStep.for_plant(pl, **kw)
    g.set_state(x0, lam0)
    g.set_inputs(u, N, pl.T_amb, pl.T_wi)
    g.step(n_ctrl)
    x, _, stats = g.get_state()
    g.close()
    return x, stats


def test_fp64_matches_cpu_kernel():
    pl, u, N, x0, lam0, x_cpu = _replay()
    x_gpu, _ = _gpu(pl, u, N, x0, lam0, 12, mode="fp64", fmad=False)
    scale = np.maximum(np.abs(x_cpu), 1e-3 * np.median(np.abs(x_cpu), 0) + 1e-12)
    assert np.max(np.abs(x_gpu - x_cpu) / scale) < 1e-9


@pytest.mark.parametrize("mode", ["mixed", "fp32"])
def test_reduced_precision_stays_close(mode):
    pl, u, N, x0, lam0, x_cpu = _replay()
    x_gpu, stats = _gpu(pl, u, N, x0, lam0, 12, mode=mode)
    assert np.isfinite(x_gpu).all()
    assert stats["retry2"].sum() == 0
    i = [pl.P_S, pl.P_D]
    assert np.max(np.abs(x_gpu[:, i] - x_cpu[:, i])) < 500.0          # Pa
