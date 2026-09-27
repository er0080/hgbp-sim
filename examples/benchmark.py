"""Simulator speed: the live stand (single plant with its PID loops), the batched
training environment vs. batch size, and the steady-state solver.

    python examples/benchmark.py            # the first run also compiles the model (~1-2 min)

Batches run in parallel on numba's thread pool (all cores by default; set
NUMBA_NUM_THREADS to limit it).
"""
import time

import numba
import numpy as np

from hgbp_sim import EnvConfig, HGBPPlant, HGBPVecEnv, PlantParams, named_point, solve_steady_state
from hgbp_sim.kernel.warmup import warm_up
from hgbp_sim.live import LiveStand


def live_stand():
    st = LiveStand(seed=0)
    st.noise = False
    st.cold_start()
    st.run_request = True
    for label, seconds in (("cold start and pull-down", 120.0), ("settling", 240.0)):
        n = int(seconds / st.dt_ctrl)
        t0 = time.perf_counter()
        for _ in range(n):
            st.step()
        el = time.perf_counter() - t0
        print(f"  {label:26s} {seconds / el:6.0f} x real time  ({1e3 * el / n:.2f} ms per {st.dt_ctrl} s control step)")
    assert st.warm_start("MT_standard")
    n = int(120.0 / st.dt_ctrl)
    t0 = time.perf_counter()
    for _ in range(n):
        st.step()
    el = time.perf_counter() - t0
    print(f"  {'at a test point':26s} {120.0 / el:6.0f} x real time  ({1e3 * el / n:.2f} ms per {st.dt_ctrl} s control step)")


def vec_env(sizes=(1, 16, 64, 256, 1024), steps=100):
    for n in sizes:
        env = HGBPVecEnv(n, EnvConfig(), seed=0)
        for _ in range(5):
            env.step(env.expert_action())
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(env.expert_action())
        el = time.perf_counter() - t0
        print(f"  n={n:5d}: {steps * n / el:8.0f} env-steps/s  ({steps * n * env.cfg.dt_ctrl / el:8.0f} simulated s "
              f"per wall s, {1e3 * el / steps:6.1f} ms per step)")


def steady_state(sizes=(1, 16)):
    for n in sizes:
        pl = HGBPPlant(PlantParams(), n=n, rng=np.random.default_rng(0), randomize=True)
        pt = named_point("MT_standard", pl.props, pl.params.N_nom)
        t0 = time.perf_counter()
        res = solve_steady_state(pl, pt["P_s"], pt["P_d"], pt["SH"], pt["N"], P_i=pt["P_i"])
        el = time.perf_counter() - t0
        print(f"  {n:3d} stand{'s' if n > 1 else ' '}: {1e3 * el:7.1f} ms  ({int(res['converged'].sum())} converged)")


if __name__ == "__main__":
    print(f"compiling / loading the model: {warm_up(verbose=False):.1f} s  "
          f"({numba.get_num_threads()} threads, {numba.config.THREADING_LAYER} layer)")
    print("live stand (web UI engine):")
    live_stand()
    print("batched training environment (baseline controller, random starts):")
    vec_env()
    print("steady-state solver (warm starts, feasibility checks):")
    steady_state()
