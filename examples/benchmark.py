"""Throughput of the batched simulator vs. number of parallel environments."""
import time

import numpy as np

from hgbp_sim import EnvConfig, HGBPVecEnv

for n in (1, 16, 128, 1024):
    env = HGBPVecEnv(n, EnvConfig(start_mode="warm"), seed=0)
    env.step(env.expert_action())
    t0 = time.time()
    k = 20
    for _ in range(k):
        env.step(env.expert_action())
    dt = time.time() - t0
    print(f"n={n:5d}: {k*n/dt:8.0f} env-steps/s  ({k*n*env.cfg.dt_ctrl/dt:9.0f} simulated seconds per wall second)")
