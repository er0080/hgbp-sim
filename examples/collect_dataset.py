"""Collect (observation, expert action, reward) data with the batched env.

The baseline PID controller acts as the expert; optional exploration noise is
added so the dataset covers off-policy states.  Output: an .npz file suitable
for behaviour cloning (see train_bc.py) or offline RL.

Usage: python examples/collect_dataset.py --envs 256 --steps 400 --out data.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from hgbp_sim import EnvConfig, HGBPVecEnv, OBS_NAMES, PlantParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400, help="control steps per environment")
    ap.add_argument("--noise", type=float, default=0.3, help="exploration noise std on expert actions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="hgbp_dataset.npz")
    args = ap.parse_args()

    cfg = EnvConfig(episode_time=1200.0, start_mode="random", action_mode="incremental")
    env = HGBPVecEnv(args.envs, cfg, params=PlantParams(), seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    obs = env.reset()
    O, A, R, D = [], [], [], []
    t0 = time.time()
    for k in range(args.steps):
        a_exp = env.expert_action()
        a = np.clip(a_exp + args.noise * rng.standard_normal(a_exp.shape), -1, 1)
        O.append(obs.copy()); A.append(a_exp.copy())
        obs, r, term, trunc, info = env.step(a)
        R.append(r.copy()); D.append((term | trunc).copy())
        if k % 50 == 0:
            print(f"step {k}: mean reward {r.mean():+.3f}, in-tol {info['in_tol'].mean():.2f}, "
                  f"tripped {info['tripped'].sum()}  ({(k+1)*args.envs/(time.time()-t0):.0f} env-steps/s)")
    np.savez_compressed(args.out, obs=np.array(O), action=np.array(A), reward=np.array(R),
                        done=np.array(D), obs_names=np.array(OBS_NAMES))
    print("saved", args.out, "obs shape", np.array(O).shape)


if __name__ == "__main__":
    main()
