"""Closed-loop run with the baseline 4-loop PID controller through a test schedule.

Usage:  python examples/closed_loop_pid.py [--cold] [--charge 1.0] [--out plot.png]
  --charge  factor on the nominal refrigerant charge (0.3 = badly undercharged,
            1.8 = badly overcharged)
"""
from __future__ import annotations

import argparse

import numpy as np

from hgbp_sim import BaselineController, HGBPPlant, PlantParams, named_point, solve_steady_state

C2K = 273.15


def run(cold: bool = False, seed: int = 0, gains: dict | None = None, T_end: float = 2400.0,
        dt_ctrl: float = 1.0, noise: bool = True, charge_factor: float = 1.0,
        liquid_in_accumulator: float = 0.5):
    params = PlantParams()
    plant = HGBPPlant(params, n=1, dt=0.05, rng=np.random.default_rng(seed))
    plant.p.charge[:] = plant.nominal_charge() * charge_factor
    pr = plant.props
    T_amb, T_wi = 25.0 + C2K, 20.0 + C2K
    plant.set_inputs(T_amb=T_amb, T_wi=T_wi)
    schedule = [
        (0.0, named_point("MT_standard", pr, 1450.0)),
        (600.0, named_point("high_lift", pr, 1750.0)),
        (1200.0, named_point("HT_standard", pr, 1200.0)),
        (1800.0, named_point("LT_standard", pr, 1450.0)),
    ]
    ctrl = BaselineController(1, gains)
    p0 = schedule[0][1]
    if cold:
        plant.cold_start(T_amb=T_amb, u_pos=ctrl.u_off[None, :], liquid_in_accumulator=liquid_in_accumulator)
        ctrl.reset()
        t_start = 10.0
    else:
        res = solve_steady_state(plant, p0["P_s"], p0["P_d"], p0["SH"], p0["N"], P_i=p0["P_i"])
        if not res["converged"].all():
            print("warning: no steady state at the first point with this charge; cold start instead")
            plant.cold_start(T_amb=T_amb, u_pos=ctrl.u_off[None, :], liquid_in_accumulator=liquid_in_accumulator)
            ctrl.reset()
            t_start = 10.0
        else:
            plant.set_state(0, res["x"])
            ctrl.reset(res["u"])
            t_start = -1.0
    log = []
    t = 0.0
    while t < T_end:
        sp = [p for (ts, p) in schedule if ts <= t][-1]
        N_cmd = sp["N"] if t >= t_start else 0.0
        meas = plant.measure(noise=noise)
        running = plant.x[:, HGBPPlant.N_] > 0.5 * params.N_min
        u = ctrl(meas, dict(P_s=sp["P_s"], P_d=sp["P_d"], SH=sp["SH"], P_i=sp["P_i"]), dt_ctrl, running)
        aux = plant.step(dt_ctrl, u_cmd=u, N_cmd=N_cmd)
        t = plant.t
        log.append([t, aux["P_s"][0], aux["P_d"][0], aux["SH"][0], aux["P_i"][0],           # 0-4
                    sp["P_s"], sp["P_d"], sp["SH"], sp["P_i"],                                # 5-8
                    aux["u1"][0], aux["u2"][0], aux["u3"][0], aux["u4"][0],                   # 9-12
                    aux["T_d"][0], aux["Tm_d"][0], aux["T_sh"][0], aux["T_cw"][0], aux["T_co"][0],  # 13-17
                    aux["mdot_c"][0], aux["W_el"][0], aux["N"][0],                            # 18-20
                    aux["x_out"][0], aux["fill_s"][0], aux["fill_i"][0], aux["SC"][0],        # 21-24
                    aux["T_wo"][0], aux["mdot_w"][0],                                         # 25-26
                    meas["P_s"][0], meas["P_d"][0], meas["SH"][0], meas["P_i"][0]])           # 27-30
    return np.array(log)


def plot(log: np.ndarray, out: str, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log[:, 0] / 60.0
    fig, ax = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    ax = ax.ravel()
    ax[0].plot(t, log[:, 27] / 1e5, color="0.7", lw=0.6, label="measured")
    ax[0].plot(t, log[:, 1] / 1e5, label="P_suc")
    ax[0].plot(t, log[:, 5] / 1e5, "k--", label="setpoint")
    ax[0].set_ylabel("suction pressure [bar]")
    ax[0].legend()
    ax[1].plot(t, log[:, 28] / 1e5, color="0.7", lw=0.6)
    ax[1].plot(t, log[:, 2] / 1e5, label="P_dis")
    ax[1].plot(t, log[:, 6] / 1e5, "k--")
    ax[1].plot(t, log[:, 4] / 1e5, label="P_int (condensing)")
    ax[1].plot(t, log[:, 8] / 1e5, "k:", lw=0.8)
    ax[1].set_ylabel("pressure [bar]")
    ax[1].legend()
    ax[2].plot(t, log[:, 29], color="0.7", lw=0.6)
    ax[2].plot(t, log[:, 3], label="superheat")
    ax[2].plot(t, log[:, 7], "k--")
    ax[2].plot(t, log[:, 24], label="cond. outlet subcooling")
    ax[2].set_ylabel("[K]")
    ax[2].legend()
    ax[3].plot(t, log[:, 13] - C2K, label="T_dis (gas)")
    ax[3].plot(t, log[:, 14] - C2K, label="T_dis (sensor)")
    ax[3].plot(t, log[:, 15] - C2K, label="T_shell")
    ax[3].plot(t, log[:, 16] - C2K, label="T_cond wall")
    ax[3].plot(t, log[:, 17] - C2K, label="T_cond outlet")
    ax[3].set_ylabel("temperature [degC]")
    ax[3].legend(fontsize=8)
    ax[4].plot(t, log[:, 9], label="1 discharge pressure")
    ax[4].plot(t, log[:, 10], label="2 suction pressure (HGBP)")
    ax[4].plot(t, log[:, 11], label="3 suction temperature (liquid)")
    ax[4].plot(t, log[:, 12], label="4 cooling water")
    ax[4].set_ylabel("valve position [-]")
    ax[4].legend(fontsize=8)
    ax[5].plot(t, log[:, 18] * 1e3, label="mass flow [g/s]")
    ax[5].plot(t, log[:, 19] / 100.0, label="power [100 W]")
    ax[5].plot(t, log[:, 20] / 100.0, label="speed [100 rpm]")
    ax[5].legend()
    ax[6].plot(t, log[:, 21], label="compressor inlet quality")
    ax[6].plot(t, log[:, 22], label="accumulator liquid fill")
    ax[6].plot(t, log[:, 23], label="condenser liquid fill")
    ax[6].axhline(1.0, color="r", lw=0.5)
    ax[6].legend(fontsize=8)
    ax[6].set_xlabel("time [min]")
    ax[7].plot(t, log[:, 25] - C2K, label="water out [degC]")
    ax[7].plot(t, log[:, 26] * 60.0, label="water flow [kg/min]")
    ax[7].legend()
    ax[7].set_xlabel("time [min]")
    for a in ax:
        a.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", action="store_true", help="start from an equalized, compressor-off state")
    ap.add_argument("--charge", type=float, default=1.0, help="factor on nominal refrigerant charge")
    ap.add_argument("--out", default="closed_loop_pid.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    log = run(cold=args.cold, seed=args.seed, charge_factor=args.charge)
    e = np.abs(log[:, 1:5] - log[:, 5:9])
    print("mean |error| over run: P_s %.3f bar, P_d %.3f bar, SH %.2f K, P_i %.3f bar" % (
        e[:, 0].mean() / 1e5, e[:, 1].mean() / 1e5, e[:, 2].mean(), e[:, 3].mean() / 1e5))
    print("time with liquid at compressor inlet: %.0f s; condenser fill range %.2f-%.2f; subcooling max %.1f K" % (
        (log[:, 21] < 1.0).sum(), log[:, 23].min(), log[:, 23].max(), log[:, 24].max()))
    plot(log, args.out, title=f"charge factor {args.charge:.2f}" + (" (cold start)" if args.cold else ""))
