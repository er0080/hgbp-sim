"""GPU benchmark of the model's step (hgbp_sim.kernel.cuda_step) against the numba CPU kernel.

    python examples/gpu_benchmark.py accuracy   --n 1024 --steps 4800 --out runs/gpu
    python examples/gpu_benchmark.py throughput --sizes 1024,4096,16384,65536 --out runs/gpu
    python examples/gpu_benchmark.py hold       --n 256 --minutes 60 --out runs/gpu
    python examples/gpu_benchmark.py tabletop   --n 256 --out runs/gpu

Inputs come from the training environment: stands at varied points (warm starts) and
cold starts (the compressor started during the run), driven by the baseline controller
plus random valve moves; the valve and speed commands of every control step are
recorded and replayed open loop on the CPU kernel (``km.step_batch``, all threads) and
the GPU, from identical states.  Results go to ``--out`` as JSON (with the commits).
Needs an NVIDIA GPU and the CUDA toolkit.  See docs/gpu-benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from hgbp_sim.env import EnvConfig, HGBPVecEnv
from hgbp_sim.kernel import layout as L
from hgbp_sim.kernel import model as km
from hgbp_sim.kernel.cuda_step import CudaStep

ROOT = Path(__file__).resolve().parents[1]
STAND = str(ROOT / "webui" / "config" / "stand_defaults.json")
BYPASS = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
OUTS = ("P_s", "P_d", "T_s", "T_d", "SH", "P_i")
MODES = ("fp64", "mixed", "fp32")


def provenance() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")), gpu=gpu,
                cpu=platform.processor() or platform.machine(), numba_threads=os.environ.get("NUMBA_NUM_THREADS"),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))


def make_env(n, seed, start_mode="random", p_cold=0.3, fluid=None, **kw):
    cfg = EnvConfig(stand=STAND, p_line=0.9, cold_valves=(0.0, 0.0, 0.0, 0.0), bypass_openings=BYPASS,
                    start_mode=start_mode, p_cold=p_cold, auto_reset=False, noise=False, **kw)
    params = None
    if fluid:
        from hgbp_sim.defaults import load_defaults
        from hgbp_sim.params import PlantParams
        params = PlantParams(**{**load_defaults(STAND, create=False)["plant"], "fluid": fluid})
    return HGBPVecEnv(n, cfg, params=params, seed=seed)


def record(env, pre, steps, sigma, seed):
    """Run the environment (baseline controller plus random valve moves of ``sigma`` in
    action units); returns the state after ``pre`` steps and the inputs of the next ``steps``."""
    rng = np.random.default_rng(seed)
    pl = env.plant

    def act():
        a = env.expert_action()
        return np.clip(a + sigma * rng.standard_normal(a.shape), -1.0, 1.0)

    for _ in range(pre):
        env.step(act())
    snap = dict(x=pl.x.copy(), lam=pl._lam.copy(), T_amb=pl.T_amb.copy(), T_wi=pl.T_wi.copy())
    U = np.empty((steps, env.n, 4))
    N = np.empty((steps, env.n))
    for t in range(steps):
        env.step(act())
        U[t], N[t] = pl.u_cmd, pl.N_cmd
    return snap, U, N


# ------------------------------------------------------------------ replays
def replay_cpu(rec, tab, snap, U, N, dt, checkpoints, n_sub=5):
    x = np.ascontiguousarray(snap["x"]).copy()
    lam = snap["lam"].copy()
    subs = np.zeros(len(x), np.int64)
    out, tot, t_run = {}, np.zeros(len(x), np.int64), 0.0
    for t in range(len(U)):
        u = np.ascontiguousarray(U[t])
        n_ = np.ascontiguousarray(N[t])
        t0 = time.perf_counter()
        km.step_batch[True](x, u, n_, snap["T_amb"], snap["T_wi"], rec, tab, lam, dt, n_sub, 0, subs)
        t_run += time.perf_counter() - t0
        tot += subs
        if t + 1 in checkpoints:
            out[t + 1] = x.copy()
    return out, dict(time=t_run, substeps=tot)


def replay_gpu(rec, tab, snap, U, N, dt, checkpoints, mode, **kw):
    g = CudaStep(rec, tab, dt, mode=mode, **kw)
    g.set_state(snap["x"], snap["lam"])
    out, tot, t_run = {}, {k: np.zeros(len(rec), np.int64) for k in CudaStep.STATS}, 0.0
    for t in range(len(U)):
        g.set_inputs(U[t], N[t], snap["T_amb"], snap["T_wi"])
        t_run += g.step(1)
        st = g.get_stats()
        for i, k in enumerate(CudaStep.STATS):
            tot[k] += st[i]
        if t + 1 in checkpoints:
            out[t + 1] = g.get_state()[0]
    g.close()
    return out, dict(time=t_run, **tot)


# ---------------------------------------------------------------- metrics
def outputs(pl, x, u, N, snap):
    with np.errstate(all="ignore"):
        _, a = pl.rhs(np.nan_to_num(x, nan=1.0), u, N, snap["T_amb"], snap["T_wi"], want_aux=True)
    return {k: np.asarray(a[k]) for k in OUTS + ("N", "x_out", "cond_flood")}


def q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(median=None, p99=None, max=None)
    return dict(median=float(np.median(v)), p99=float(np.percentile(v, 99)), max=float(v.max()))


def compare(pl, xr, xg, u, N, snap):
    """State and output differences of xg against the reference xr (per stand)."""
    finite = np.isfinite(xg).all(1)
    scale = np.maximum(np.median(np.abs(xr), 0), 1e-12)
    rel = np.abs(xg - xr) / np.maximum(np.abs(xr), 1e-3 * scale)
    rel_stand = np.where(finite, rel.max(1), np.inf)
    worst_state = [L.STATE_NAMES[k] for k in np.argsort(-np.nanmax(np.where(finite[:, None], rel, 0), 0))[:3]]
    a_r, a_g = outputs(pl, xr, u, N, snap), outputs(pl, xg, u, N, snap)
    d = dict(nonfinite=int((~finite).sum()), rel_state=q(rel_stand[finite]), worst_states=worst_state)
    for k in OUTS:
        diff = np.abs(a_g[k] - a_r[k])[finite]
        d[k] = q(diff / (1e5 if k.startswith("P") else 1.0))       # bar / K
    run = finite & (a_r["N"] > 1.0)        # compressor running in the reference
    d["running"] = int(run.sum())
    d["running_only"] = {k: q(np.abs(a_g[k] - a_r[k])[run] / (1e5 if k.startswith("P") else 1.0))
                         for k in ("P_s", "P_d", "T_d", "SH")}
    # stands whose trajectories have separated (beyond sensor resolution), and where they are
    sep = ~finite | (np.abs(a_g["P_s"] - a_r["P_s"]) > 1e3) | (np.abs(a_g["P_d"] - a_r["P_d"]) > 1e3) \
        | (np.abs(a_g["SH"] - a_r["SH"]) > 0.1) | (np.abs(a_g["T_d"] - a_r["T_d"]) > 0.1)
    d["separated"] = int(sep.sum())
    d["separated_idx"] = np.flatnonzero(sep)[:50].tolist()
    if sep.any():       # the separated stands in the reference
        d["separated_ref"] = dict(
            running=float((a_r["N"][sep] > 1.0).mean()), started_cold=float((snap["x"][sep, L.IX["N"]] < 1.0).mean()),
            liquid_at_compressor=float((a_r["x_out"][sep] < 0.999).mean()), SH=q(a_r["SH"][sep]),
            P_d_bar=q(a_r["P_d"][sep] / 1e5), cond_flood=float((a_r["cond_flood"][sep] > 0.0).mean()))
    M = lambda x: x[:, L.IX["M_s"]] + x[:, L.IX["M_d"]] + x[:, L.IX["M_i"]]
    U_ = lambda x: x[:, L.IX["U_s"]] + x[:, L.IX["U_d"]] + x[:, L.IX["U_i"]]
    d["rel_mass_vs_ref"] = q((np.abs(M(xg) - M(xr)) / M(xr))[finite])
    d["rel_energy_vs_ref"] = q((np.abs(U_(xg) - U_(xr)) / np.abs(U_(xr)))[finite])
    return d


def mass(x):
    return x[:, L.IX["M_s"]] + x[:, L.IX["M_d"]] + x[:, L.IX["M_i"]]


def drift(x0, x):
    """Relative change of the total refrigerant mass since the start (per stand)."""
    return q(np.abs(mass(x) - mass(x0)) / mass(x0))


def mode_opts(a, mode):
    """GPU options of a mode name such as fp64, fp64-nofmad, fp32-fast (fast math)."""
    opts = gpu_opts(a)
    base, *flags = mode.split("-")
    if "nofmad" in flags:
        opts["fmad"] = False
    if "fast" in flags:
        opts["fastmath"] = True
    return base, opts


def gpu_opts(a):
    return dict(block=a.block, maxrregcount=a.maxrreg or None, fastmath=a.fastmath, fmad=not a.no_fmad,
                device=a.device)


def save(a, name, res):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res["provenance"] = provenance()
    res["args"] = vars(a)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:6000])


# ------------------------------------------------------------------ suites
def run_compare(a, env, snap, U, N, checkpoints, name, extra=None):
    pl = env.plant
    rec, tab, dt = pl.p.packed(), pl.props.tab, pl.dt
    t0 = time.perf_counter()
    ref, ref_info = replay_cpu(rec, tab, snap, U, N, dt, checkpoints)
    res = dict(n=env.n, steps=len(U), cpu_time=ref_info["time"], cpu_wall=time.perf_counter() - t0,
               cpu_substeps_per_step=float(ref_info["substeps"].mean() / len(U)),
               cpu_mass_drift={c: drift(snap["x"], ref[c]) for c in checkpoints}, modes={})
    print(f"cpu replay {res['cpu_time']:.1f} s", flush=True)
    for mode in a.modes.split(","):
        base, opts = mode_opts(a, mode)
        out, info = replay_gpu(rec, tab, snap, U, N, dt, checkpoints, base, **opts)
        r = dict(gpu_time=info["time"], substeps_per_step=float(info["substeps"].mean() / len(U)),
                 retry1=int(info["retry1"].sum()), retry2=int(info["retry2"].sum()),
                 suction_unconverged=int(info["suction_unconverged"].sum()),
                 flash_unconverged=int(info["flash_unconverged"].sum()),
                 stands_with_retry=int((info["retry1"] > 0).sum()), checkpoints={})
        for c in checkpoints:
            cmp = compare(pl, ref[c], out[c], U[c - 1], N[c - 1], snap)
            cmp["mass_drift"] = drift(snap["x"], out[c])
            r["checkpoints"][c] = cmp
        res["modes"][mode] = r
        print(mode, f"{info['time']:.1f} s", flush=True)
    if extra:
        res.update(extra)
    save(a, name, res)


def cmd_accuracy(a):
    env = make_env(a.n, a.seed, fluid=a.fluid)
    t0 = time.perf_counter()
    snap, U, N = record(env, a.pre, a.steps, a.sigma, a.seed + 1)
    print(f"recorded {a.steps} steps in {time.perf_counter() - t0:.0f} s", flush=True)
    cps = sorted({c for c in (40, 400, 1200, 2400, 4800, 9600, 14400) if c <= a.steps} | {a.steps})
    cold = int((snap["x"][:, L.IX["N"]] < 1.0).sum())
    run_compare(a, env, snap, U, N, cps, f"accuracy_{a.fluid or 'stand'}_{a.n}_{a.steps}",
                extra=dict(stands_off_at_start=cold))


def settle(env, pre):
    """``pre`` control steps of the baseline controller; the state and its rate estimate."""
    pl = env.plant
    for _ in range(pre):
        env.step(env.expert_action())
    pl.rhs(pl.x, pl.u_cmd, pl.N_cmd, pl.T_amb, pl.T_wi)
    return dict(x=pl.x.copy(), lam=pl._lam_last.copy(), T_amb=pl.T_amb.copy(), T_wi=pl.T_wi.copy())


def cmd_hold(a):
    """Warm starts at their set points, inputs held: drift of a steady state."""
    env = make_env(a.n, a.seed, start_mode="warm", p_warm_at_setpoint=1.0, fluid=a.fluid)
    pl = env.plant
    steps = int(a.minutes * 60 / 0.25)
    snap = settle(env, 4)
    U = np.broadcast_to(pl.u_cmd, (steps, a.n, 4))
    N = np.broadcast_to(pl.N_cmd, (steps, a.n))
    cps = sorted({c for c in (240, 1200, 2400, 4800, 7200, 14400) if c <= steps} | {steps})
    # equilibrium check of the start: how far the CPU kernel moves in the first minute
    run_compare(a, env, snap, U, N, cps, f"hold_{a.fluid or 'stand'}_{a.n}_{a.minutes}min")


def cmd_tabletop(a):
    """Warm stands; valve 1 closed step by step: the discharge pressure runs up to the top
    of the property tables (0.99 P_crit for R410A) with the compressor still running."""
    env = make_env(a.n, a.seed, start_mode="warm", fluid=a.fluid)
    pl = env.plant
    steps = a.steps
    snap = settle(env, 4)
    U = np.repeat(pl.u_cmd[None], steps, 0)
    ramp = np.minimum(np.arange(1, steps + 1) * 0.25 / a.close_time, 1.0)        # 0 -> 1 over close_time
    U[:, :, 0] = pl.u_cmd[None, :, 0] * (1.0 - ramp[:, None]) + 0.02 * ramp[:, None]
    N = np.broadcast_to(pl.N_cmd, (steps, a.n))
    cps = sorted({c for c in (40, 120, 200, 400, 800, 1200) if c <= steps} | {steps})
    # how close the reference gets to the table top
    rec, tab, dt = pl.p.packed(), pl.props.tab, pl.dt
    ref, _ = replay_cpu(rec, tab, snap, U[:steps], N[:steps], dt, cps)
    top = {c: q(ref[c][:, L.IX["P_d"]] / tab.p_max) for c in cps}
    run_compare(a, env, snap, U, N, cps, f"tabletop_{a.fluid or 'stand'}_{a.n}",
                extra=dict(P_d_over_table_top=top, table_p_max=float(tab.p_max)))


def cmd_throughput(a):
    env = make_env(a.n, a.seed, fluid=a.fluid)
    snap, U, N = record(env, a.pre, a.steps + 1, a.sigma, a.seed + 1)
    pl = env.plant
    rec0, tab, dt = pl.p.packed(), pl.props.tab, pl.dt
    res = dict(base_stands=a.n, steps=a.steps, cpu={}, gpu={})
    for n in [int(s) for s in a.sizes.split(",")]:
        reps = -(-n // a.n)
        tile = lambda v: np.ascontiguousarray(np.concatenate([v] * reps)[:n])
        rec = tile(rec0)
        s = dict(x=tile(snap["x"]), lam=tile(snap["lam"]), T_amb=tile(snap["T_amb"]), T_wi=tile(snap["T_wi"]))
        Us = [tile(U[t]) for t in range(a.steps + 1)]
        Ns = [tile(N[t]) for t in range(a.steps + 1)]
        if not a.skip_cpu and n <= a.cpu_max:
            x = s["x"].copy()
            lam = s["lam"].copy()
            subs = np.zeros(n, np.int64)
            km.step_batch[True](x, Us[0], Ns[0], s["T_amb"], s["T_wi"], rec, tab, lam, dt, 5, 0, subs)   # warm-up
            x, lam, tot = s["x"].copy(), s["lam"].copy(), 0
            t0 = time.perf_counter()
            for t in range(a.steps):
                km.step_batch[True](x, Us[t], Ns[t], s["T_amb"], s["T_wi"], rec, tab, lam, dt, 5, 0, subs)
                tot += subs.sum()
            el = time.perf_counter() - t0
            res["cpu"][n] = dict(time=el, steps_per_s=n * a.steps / el, substeps_per_step=tot / (n * a.steps))
            print(f"cpu n={n}: {n * a.steps / el:,.0f} env-steps/s", flush=True)
        for mode in [m for m in a.modes.split(",") if m]:
            for block in [int(b) for b in a.blocks.split(",")]:
                for sort in ([False, True] if a.sort else [False]):
                    opts = gpu_opts(a)
                    opts["block"] = block
                    t0 = time.perf_counter()
                    g = CudaStep(rec, tab, dt, mode=mode, **opts)
                    t_init = time.perf_counter() - t0
                    g.set_state(s["x"], s["lam"])
                    g.set_inputs(Us[0], Ns[0], s["T_amb"], s["T_wi"])
                    g.step(1)                                    # warm-up launch
                    g.set_state(s["x"], s["lam"])
                    k_gpu, wall, tot = 0.0, time.perf_counter(), 0
                    for t in range(a.steps):
                        g.set_inputs(Us[t], Ns[t], s["T_amb"], s["T_wi"])
                        if sort and t > 0:
                            g.set_order(np.argsort(st[0], kind="stable"))
                        k_gpu += g.step(1)
                        st = g.get_stats()
                        tot += st[0].sum()
                    wall = time.perf_counter() - wall
                    key = f"{mode}/b{block}" + ("/sorted" if sort else "")
                    res["gpu"].setdefault(key, {})[n] = dict(
                        kernel_time=k_gpu, wall=wall, steps_per_s=n * a.steps / k_gpu,
                        steps_per_s_wall=n * a.steps / wall, substeps_per_step=tot / (n * a.steps), init=t_init,
                        **g.kernel_info())
                    print(f"gpu {key} n={n}: {n * a.steps / k_gpu:,.0f} env-steps/s (kernel), "
                          f"{n * a.steps / wall:,.0f} (wall) {g.kernel_info()}", flush=True)
                    g.close()
    save(a, f"throughput{a.tag}", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["accuracy", "throughput", "hold", "tabletop"])
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--pre", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fluid", default=None)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--close-time", type=float, default=60.0)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--sizes", default="1024,4096,16384,65536")
    ap.add_argument("--blocks", default="128")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--maxrreg", type=int, default=0)
    ap.add_argument("--fastmath", action="store_true")
    ap.add_argument("--no-fmad", action="store_true")
    ap.add_argument("--sort", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--cpu-max", type=int, default=1 << 30)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="runs/gpu")
    a = ap.parse_args()
    {"accuracy": cmd_accuracy, "throughput": cmd_throughput, "hold": cmd_hold, "tabletop": cmd_tabletop}[a.cmd](a)


if __name__ == "__main__":
    main()
