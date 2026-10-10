"""Reference data for the model regression test (tests/test_reference.py).

Runs a fixed set of scenarios and stores, per scenario, sampled trajectories of
the main outputs, and a set of states visited on the way with the model's
right-hand side and auxiliary outputs at those states; and the results of a
set of steady-state solves.  The test compares the
current model against this file: pointwise (same state -> same derivatives,
tight tolerance) and along the trajectories (loose tolerance, the integrator's
step control may differ).

Regenerate after an intentional change of the model's physics:

    python tests/reference/make_reference.py            # writes tests/reference/reference.npz

``--package`` runs another build of the package (e.g. a copy of an older
version under a different name on ``--path``) to compare implementations.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reference.npz")

# sampled outputs (aux keys)
CHANNELS = ("P_s", "P_d", "P_i", "T_s", "T_d", "T_co", "SH", "SC", "mdot_c", "mdot_1", "mdot_2", "mdot_3",
            "mdot_w", "W_el", "N", "u1", "u2", "u3", "u4", "T_wo", "rec_level", "x_qo", "y_liq", "M_tot",
            "T_sh", "T_cw")
# aux outputs compared pointwise (scalars per environment)
AUX_POINT = CHANNELS + ("P_h", "P_tee", "dP_suc", "dP_dis", "dP_hdr", "dP_bp", "dP_mg", "dP_mq", "dP_cr",
                        "dP_cw", "Q_r", "Q_w", "Q_mx", "Q_q", "T_go", "x_out", "fill_i", "cond_flood",
                        "M_s", "M_d", "M_i", "h2", "h_2f", "h_3f", "T_sat_s", "T_sat_d", "T_sat_i")
SAMPLE_DT = 1.0        # trajectory sampling [s]
POINT_EVERY = 20.0     # a pointwise sample every ... [s]


class Recorder:
    def __init__(self, name):
        self.name = name
        self.t, self.rows, self.pts = [], [], []

    def sample(self, t, aux):
        self.t.append(t)
        self.rows.append([np.asarray(aux[k], float).copy() for k in CHANNELS])

    def point(self, plant):
        self.pts.append(dict(x=plant.x.copy(), u=plant.u_cmd.copy(), N=plant.N_cmd.copy(),
                             T_amb=plant.T_amb.copy(), T_wi=plant.T_wi.copy()))

    def arrays(self, plant):
        out = {f"{self.name}/t": np.array(self.t)}
        rows = np.array(self.rows)                     # (T, channels, n)
        for j, k in enumerate(CHANNELS):
            out[f"{self.name}/{k}"] = rows[:, j]
        # pointwise: the plant's right-hand side at the recorded states (its own parameters)
        if self.pts:
            X = np.concatenate([p["x"] for p in self.pts])
            U = np.concatenate([p["u"] for p in self.pts])
            N = np.concatenate([p["N"] for p in self.pts])
            Ta = np.concatenate([p["T_amb"] for p in self.pts])
            Tw = np.concatenate([p["T_wi"] for p in self.pts])
            reps = len(self.pts)
            pp = plant.params_subset(np.tile(np.arange(plant.n), reps))
            dx, a = plant.rhs(X, U, N, Ta, Tw, want_aux=True, p=pp)
            dxh = plant.rhs(X, U, N, Ta, Tw, p=pp, hold_P_s=True)
            out[f"{self.name}/pt_x"] = X
            out[f"{self.name}/pt_u"], out[f"{self.name}/pt_N"] = U, N
            out[f"{self.name}/pt_Tamb"], out[f"{self.name}/pt_Twi"] = Ta, Tw
            out[f"{self.name}/pt_env"] = np.tile(np.arange(plant.n), reps)
            out[f"{self.name}/pt_dx"], out[f"{self.name}/pt_dxh"] = dx, dxh
            for k in AUX_POINT:
                out[f"{self.name}/pt_aux_{k}"] = np.asarray(a[k], float)
        return out


def _live_run(st, rec, seconds, t0=0.0):
    n = int(round(seconds / st.dt_ctrl))
    every = int(round(SAMPLE_DT / st.dt_ctrl))
    pevery = int(round(POINT_EVERY / st.dt_ctrl))
    for i in range(n):
        st.step()
        if (i + 1) % every == 0:
            rec.sample(t0 + (i + 1) * st.dt_ctrl, st.plant.outputs())
        if (i + 1) % pevery == 0:
            rec.point(st.plant)
    return t0 + seconds


def scenarios(pkg):
    live = importlib.import_module(pkg + ".live")
    plant_mod = importlib.import_module(pkg + ".plant")
    params_mod = importlib.import_module(pkg + ".params")
    LiveStand, HGBPPlant, PlantParams = live.LiveStand, plant_mod.HGBPPlant, params_mod.PlantParams

    def stand(**params):
        st = LiveStand(PlantParams(**params) if params else None, seed=0)
        st.noise = False
        return st

    # trip limits in force when the reference was made; the trip scenario depends on them
    # (the defaults now hold the stand's 700 psia / 320 degF / 30 psia hard limits)
    ref_limits = dict(P_d_max=41.4e5, P_s_min=0.3e5, T_d_max=135.0 + 273.15)

    # 1. cold start, pull-down and settling under the baseline loops
    st = stand()
    rec = Recorder("cold")
    st.cold_start()
    rec.point(st.plant)
    st.run_request = True
    _live_run(st, rec, 360.0)
    yield rec, st.plant

    # 2. warm start at MT_standard, compressor speed step, refrigerant recovery
    st = stand()
    rec = Recorder("mt")
    assert st.warm_start("MT_standard")
    rec.point(st.plant)
    t = _live_run(st, rec, 30.0)
    st.set_compressor(speed=0.7 * st.params.N_nom)
    t = _live_run(st, rec, 60.0, t)
    st.set_sim(charge_rate=0.05)
    st.add_charge(-1.0)
    _live_run(st, rec, 60.0, t)
    yield rec, st.plant

    # 3. warm start at LT_standard, suction pressure setpoint step
    st = stand()
    rec = Recorder("lt")
    assert st.warm_start("LT_standard")
    t = _live_run(st, rec, 20.0)
    st.set_loop("spv", sp=st.sp["P_s"] / 1e5 + 0.5)
    _live_run(st, rec, 60.0, t)
    yield rec, st.plant

    # 4. overfed quench: valve 3 in manual well above its equilibrium opening (floodback)
    st = stand()
    rec = Recorder("flood")
    assert st.warm_start("HT_standard")
    st.set_loop("stv", mode="manual")
    st.set_loop("stv", out=min(1.0, 1.8 * st.manual_out["stv"]))
    _live_run(st, rec, 60.0)
    yield rec, st.plant

    # 5. cooling water closed: discharge pressure rises until the stand trips
    st = stand(**ref_limits)
    rec = Recorder("trip")
    assert st.warm_start("MT_standard")
    st.set_loop("water", mode="manual")
    st.set_loop("water", out=0.0)
    _live_run(st, rec, 150.0)
    yield rec, st.plant

    # 6. batch of randomized stands, open loop from a cold start (valve and speed steps)
    rng = np.random.default_rng(1)
    pl = HGBPPlant(PlantParams(), n=6, dt=0.05, rng=rng, randomize=True)
    pl.set_inputs(T_amb=np.linspace(288.0, 308.0, 6), T_wi=np.linspace(285.0, 300.0, 6))
    pl.cold_start()
    rec = Recorder("batch")
    rec.point(pl)
    u = np.array([[0.35, 0.3, 0.15, 0.5]] * 6) * np.linspace(0.8, 1.2, 6)[:, None]
    N = np.full(6, 3550.0)
    t = 0.0
    for seg, (du, Nf) in enumerate([(1.0, 1.0), (1.3, 1.0), (0.8, 0.8)]):
        for i in range(80):
            a = pl.step(0.5, u_cmd=np.clip(u * du, 0, 1), N_cmd=N * Nf)
            t += 0.5
            if i % 2 == 1:
                rec.sample(t, a)
            if i % 40 == 39:
                rec.point(pl)
    yield rec, pl


def steady_cases(pkg):
    """Direct steady-state solves: every named point for the default stand, one at a
    receiver level instead of a charge, and a batch of randomized stands."""
    plant_mod = importlib.import_module(pkg + ".plant")
    params_mod = importlib.import_module(pkg + ".params")
    sc = importlib.import_module(pkg + ".scenarios")
    ss = importlib.import_module(pkg + ".steady_state")
    HGBPPlant, PlantParams = plant_mod.HGBPPlant, params_mod.PlantParams
    out = {}

    def store(name, res):
        for k in ("x", "u", "converged", "resid"):
            out[f"ss_{name}/{k}"] = np.asarray(res[k], float)
        for k in ("P_s", "P_d", "P_i", "SH", "mdot_c", "mdot_2", "mdot_3", "M_tot", "T_d"):
            out[f"ss_{name}/aux_{k}"] = np.asarray(res["aux"][k], float)

    pl = HGBPPlant(PlantParams(), n=1)
    pl.set_inputs(T_amb=298.15, T_wi=293.15)
    for name in sc.NAMED_POINTS:
        pt = sc.named_point(name, pl.props, 3550.0)
        store(name, ss.solve_steady_state(pl, pt["P_s"], pt["P_d"], pt["SH"], pt["N"], P_i=pt["P_i"]))
    pt = sc.named_point("MT_standard", pl.props, 3000.0)
    store("fill", ss.solve_steady_state(pl, pt["P_s"], pt["P_d"], pt["SH"], pt["N"], P_i=pt["P_i"], fill=0.4))
    pl = HGBPPlant(PlantParams(), n=6, rng=np.random.default_rng(2), randomize=True)
    pl.set_inputs(T_amb=np.linspace(290.0, 305.0, 6), T_wi=np.linspace(286.0, 298.0, 6))
    names = list(sc.NAMED_POINTS)
    pts = [sc.named_point(names[k % len(names)], pl.props, 2500.0 + 300.0 * k) for k in range(6)]
    col = lambda f: np.array([q[f] for q in pts])
    store("batch", ss.solve_steady_state(pl, col("P_s"), col("P_d"), col("SH"), col("N"), P_i=col("P_i")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="hgbp_sim")
    ap.add_argument("--path", default=None, help="prepended to sys.path")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.path:
        sys.path.insert(0, args.path)
    data = {}
    for rec, plant in scenarios(args.package):
        t0 = time.perf_counter()
        data.update(rec.arrays(plant))
        print(f"{rec.name}: {len(rec.t)} samples, {len(rec.pts)} states ({time.perf_counter() - t0:.1f} s)")
    t0 = time.perf_counter()
    data.update(steady_cases(args.package))
    print(f"steady-state solves ({time.perf_counter() - t0:.1f} s)")
    np.savez_compressed(args.out, **data)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
