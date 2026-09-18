"""Open-loop step responses: from a steady test point, step each valve by +10 %.

Shows the coupled MIMO behaviour of the stand (every valve moves every
controlled variable).  Usage: python examples/open_loop_step.py [--out fig.png]
"""
from __future__ import annotations

import argparse

import numpy as np

from hgbp_sim import HGBPPlant, PlantParams, named_point, solve_steady_state

C2K = 273.15
VALVES = ["1 discharge pressure", "2 suction pressure (HGBP)", "3 suction temperature (liquid)", "4 cooling water"]


def step_responses(T_end: float = 600.0, du: float = 0.10):
    params = PlantParams()
    pl = HGBPPlant(params, n=4, dt=0.05)
    pl.set_inputs(T_amb=25 + C2K, T_wi=20 + C2K)
    p0 = named_point("MT_standard", pl.props, 1450.0)
    res = solve_steady_state(pl, p0["P_s"], p0["P_d"], p0["SH"], p0["N"], P_i=p0["P_i"])
    assert res["converged"].all()
    pl.set_state(np.arange(4), res["x"])
    u = res["u"].copy()
    for k in range(4):            # environment k steps valve k
        u[k, k] = np.clip(u[k, k] + du, 0, 1)
    pl.set_inputs(u_cmd=res["u"], N_cmd=p0["N"])
    pl.step(10.0)
    pl.set_inputs(u_cmd=u)
    log = []
    while pl.t < T_end + 10.0:
        a = pl.step(1.0)
        log.append(np.stack([a["P_s"], a["P_d"], a["SH"], a["P_i"], a["T_d"], a["mdot_c"]], 1))
    return np.array(log), res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="open_loop_step.png")
    args = ap.parse_args()
    log, res = step_responses()
    for k, nm in enumerate(VALVES):
        d = log[-1, k] - log[0, k]
        print(f"+10% valve {nm:32s}: dP_s={d[0]/1e5:+.3f} bar  dP_d={d[1]/1e5:+.3f} bar  dSH={d[2]:+.2f} K  "
              f"dP_i={d[3]/1e5:+.3f} bar  dT_d={d[4]:+.2f} K  dmdot={d[5]*1e3:+.2f} g/s  (after 10 min)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(log.shape[0]) / 60.0
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for k, nm in enumerate(VALVES):
        ax[0].plot(t, (log[:, k, 0] - log[0, k, 0]) / 1e5, label=f"+10% valve {nm}")
        ax[1].plot(t, (log[:, k, 1] - log[0, k, 1]) / 1e5)
        ax[2].plot(t, log[:, k, 2] - log[0, k, 2])
        ax[3].plot(t, (log[:, k, 3] - log[0, k, 3]) / 1e5)
    ax[0].set_ylabel("dP_suc [bar]"); ax[1].set_ylabel("dP_dis [bar]")
    ax[2].set_ylabel("dSH [K]"); ax[3].set_ylabel("dP_int [bar]")
    ax[3].set_xlabel("time [min]"); ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    print("saved", args.out)
