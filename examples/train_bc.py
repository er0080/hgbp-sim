"""Behaviour cloning of the PID expert with a small MLP (PyTorch), then
closed-loop evaluation of the learned policy against the expert.

This is a minimal template for the neural-network control work; RL (e.g.
PPO/SAC through the gymnasium wrapper, or a custom loop on HGBPVecEnv) can
start from the cloned policy.

Usage: python examples/train_bc.py --data hgbp_dataset.npz --epochs 30
"""
from __future__ import annotations

import argparse

import numpy as np

from hgbp_sim import EnvConfig, HGBPVecEnv


def evaluate(policy_fn, n=64, steps=600, seed=123):
    env = HGBPVecEnv(n, EnvConfig(episode_time=steps + 1.0, start_mode="random"), seed=seed)
    obs = env.reset()
    tot, tol, trips = np.zeros(n), np.zeros(n), 0
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(policy_fn(obs, env))
        tot += r
        tol += info["in_tol"]
        trips += int(info["tripped"].sum())
    return tot.mean(), tol.mean() / steps, trips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hgbp_dataset.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise SystemExit("PyTorch is required for this example: pip install torch")

    d = np.load(args.data)
    X = torch.tensor(d["obs"].reshape(-1, d["obs"].shape[-1]), dtype=torch.float32)
    Y = torch.tensor(d["action"].reshape(-1, d["action"].shape[-1]), dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(X.shape[1], args.hidden), nn.Tanh(),
                        nn.Linear(args.hidden, args.hidden), nn.Tanh(),
                        nn.Linear(args.hidden, Y.shape[1]), nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(X)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            loss = ((net((X[idx] - mu) / sd) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep}: mse {tot/n:.4f}")

    def bc_policy(obs, env):
        with torch.no_grad():
            return net((torch.tensor(obs, dtype=torch.float32) - mu) / sd).numpy()

    print("expert :", evaluate(lambda o, e: e.expert_action()))
    print("cloned :", evaluate(bc_policy))


if __name__ == "__main__":
    main()
