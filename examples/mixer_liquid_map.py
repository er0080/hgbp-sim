"""Liquid leaving a poor mixing exchanger across the compressor's operating envelope.

A planning map for the suction exchanger (docs/suction-exchanger.md, branch
``suction-bphe``): where would an old-style mixer, one that transfers only a fixed share
of the heat it could, let quench liquid through to the suction line?

At each evaporating / condensing point inside the envelope the whole stand is brought to
its steady state (the stand definition file; superheat held at the compressor port), which
gives the flows and states the controller sets at the mixer: hot gas ``m2`` at ``h_2f``,
quench ``m3`` at ``h_3f``, both at the mixer pressure ``P_tee``.  The mixer is then replaced
by a fixed effectiveness on the hot gas (the boiling quench has no temperature glide, so
the gas is the minimum capacity rate):

    Q = eps * m2 * (h_2f - h_v)        h_v: saturated vapor at the mixer pressure

The quench liquid that heat does not evaporate leaves with the gas, out of equilibrium:

    y = max(0, m3 (1 - x3) - Q / h_lv) / (m2 + m3)
      = max(0, y_eq + (1 - eps) m2 (h_2f - h_v) / ((m2 + m3) h_lv))

``y_eq`` is the liquid share of the mixed stream in equilibrium.  Where ``y_eq`` > 0 the
suction line's heat gain exceeds what the port superheat holds, so no mixer or exchanger
dries the stream (hatched on the maps).  Elsewhere the effectiveness that just dries it,
``eps_required = m3 (1 - x3) h_lv / (m2 (h_2f - h_v))``, depends on the operating point
only, not on an assumed mixer: it has its own map.

Each point is reached in up to four stages:

1. the steady-state solver with the stand as defined;
2. the solver again with the mixing exchanger's evaporating coefficient ``mx_alpha_e``
   times MX_EVAP.  At small flows the stand's mixing exchanger (sized for large machines)
   holds a liquid pool whose profile the solver does not bring to rest; that exchanger is
   not the mixer this map studies (it only passes the stand's flows to it), and where both
   solves converge the flows and the liquid on the map agree within 0.3 % and 0.05 points;
3. a valve the solution pins wide open (u > 0.98) gets ENLARGE times its Kv (a manual
   bypass or a larger valve), and the solver runs again;
4. what is still unsolved is settled dynamically under the stand's own PID loops from the
   solver's best state, the quench valve replaced by a smaller one (SMALL_V3 times its Kv)
   where it would run under 5 % open.  Settled: on its setpoints, flows steady.

Valves change no steady state's thermodynamics (they throttle at constant enthalpy), only
whether the stand can hold it; the CSV records how each point was reached (``method``) and
the maps mark the points that needed resized valves.

    python examples/mixer_liquid_map.py                            # eps 0.25, three machines, 10 R
    python examples/mixer_liquid_map.py --eps 0.25 0.5 --SH-R 10 30
    python examples/mixer_liquid_map.py --machine "32T 60 Hz=3.55e-4@3550" --set UA_sa=4

Writes two PNGs (<out>.png: liquid leaving the mixer, one panel per machine, superheat
and eps; <out>_eps.png: the effectiveness required) and a CSV of every point.  Inside the
envelope, points below 20 K lift are outside the stand's limits (left blank).  The first
run compiles the model (about a minute); then a few minutes.
"""
import argparse
import csv
import itertools

import numpy as np

from hgbp_sim.control import BaselineController
from hgbp_sim.defaults import controller_settings, load_defaults
from hgbp_sim.params import PlantParams
from hgbp_sim.plant import HGBPPlant
from hgbp_sim.scenarios import R410A_ENVELOPE, Envelope, inside_polygon
from hgbp_sim.steady_state import solve_steady_state

C2K = 273.15
# swept volume [m^3/rev] @ speed [rpm]
MACHINES = ("small 35 Hz=1.05e-4@2071", "12T 60 Hz=1.68e-4@3500", "32T 60 Hz=3.55e-4@3550")
VALVE_KV = ("Kv_dpv", "Kv_spv", "Kv_stv", "Kv_w")       # valves 1..4
MX_EVAP = 1.0 / 3.0       # the mixing exchanger's evaporating coefficient for the retry
ENLARGE = 3.0              # a valve pinned open: manual bypass / larger valve, times its Kv
SMALL_V3 = 0.3             # the quench valve under U3_SMALL: a smaller valve, times its Kv
U_PINNED, U3_SMALL = 0.98, 0.05
T_SETTLE, T_CHECK, DT_CTRL = 3000.0, 300.0, 0.25     # PID settling run, steadiness window, control step [s]
KEYS = ("mdot_c", "mdot_2", "mdot_3", "h_2f", "h_3f", "P_tee", "Q_sg")


def f2c(f):
    return (np.asarray(f, float) - 32.0) / 1.8


def c2f(c):
    return np.asarray(c, float) * 1.8 + 32.0


def parse_machine(s):
    name, spec = s.split("=", 1)
    vd, rpm = spec.split("@")
    return name.strip(), float(vd), float(rpm)


class Grid:
    """The cases (machine, superheat, envelope point) and their targets."""

    def __init__(self, doc, over, machines, SH_list, Te_f, Tc_f, T_amb, T_int):
        env = Envelope()
        TE, TC = np.meshgrid(f2c(Te_f), f2c(Tc_f))
        self.ok = inside_polygon(TE, TC, R410A_ENVELOPE) & (TC - TE >= env.dT_lift_min)
        self.pts = np.argwhere(self.ok)
        self.cases = list(itertools.product(range(len(machines)), range(len(SH_list)), range(len(self.pts))))
        self.doc, self.over, self.machines, self.T_amb = doc, over, machines, T_amb + C2K
        self.T_wi = doc["simulation"]["T_wi"] + C2K
        n = len(self.cases)
        self.kv = np.ones((n, 4))                         # valve Kv factors per case
        self.mx_evap = np.ones(n)                         # mixing exchanger evaporating coefficient factor
        pl = self.plant(np.arange(n))
        self.props = pr = pl.props
        self.Te = np.array([TE[tuple(self.pts[k])] for _, _, k in self.cases])
        self.Tc = np.array([TC[tuple(self.pts[k])] for _, _, k in self.cases])
        self.P_s, self.P_d = pr.P_sat(self.Te + C2K), pr.P_sat(self.Tc + C2K)
        # the liquid at T_int, at least dT_int_above_water over the cooling water and
        # dT_int_below_cond under the condensing temperature
        Ti = np.clip(T_int, self.T_wi - C2K + env.dT_int_above_water, self.Tc - env.dT_int_below_cond)
        self.P_i = pr.P_sat(Ti + C2K)
        self.feasible = self.P_d / self.P_s <= env.Pr_max
        self.SH = np.array([SH_list[i] / 1.8 for _, i, _ in self.cases])
        self.N = np.array([machines[im][2] for im, _, _ in self.cases])

    def plant(self, idx, dt=None):
        """A plant for cases ``idx`` with their machines and valve sizes."""
        kw = {} if dt is None else dict(dt=dt)
        pl = HGBPPlant(PlantParams(**{**self.doc["plant"], **self.over}), n=len(idx), **kw)
        p = pl.p
        for i, e in enumerate(idx):
            p.V_disp[i] = self.machines[self.cases[e][0]][1]
            for v, name in enumerate(VALVE_KV):
                getattr(p, name)[i] *= self.kv[e, v]
            p.mx_alpha_e[i] *= self.mx_evap[e]
        pl._prepare(p)
        pl.set_inputs(T_amb=np.full(pl.n, self.T_amb), T_wi=np.full(pl.n, self.T_wi))
        return pl

    def solve(self, idx):
        pl = self.plant(idx)
        return solve_steady_state(pl, self.P_s[idx], self.P_d[idx], self.SH[idx], self.N[idx], P_i=self.P_i[idx],
                                  rounds=25, max_iter=80)

    def settle(self, idx, x0, u0):
        """Closed loop under the stand's PID loops from ``x0``; returns (settled, aux)."""
        pl = self.plant(idx, dt=0.05)
        pl.set_state(np.arange(len(idx)), x0)
        ctrl = BaselineController(len(idx), **controller_settings(self.doc))
        ctrl.reset(u0)
        pr = self.props
        sp = dict(P_s=self.P_s[idx], P_d=self.P_d[idx], T_s=pr.T_sat(self.P_s[idx]) + self.SH[idx], P_i=self.P_i[idx])
        N = self.N[idx]
        n_chk = int(T_CHECK / DT_CTRL)
        lo = hi = None
        for k in range(int(T_SETTLE / DT_CTRL)):
            u = ctrl(pl.measure(noise=False), sp, DT_CTRL)
            a = pl.step(DT_CTRL, u_cmd=u, N_cmd=N)
            if k >= int(T_SETTLE / DT_CTRL) - n_chk:
                m = np.stack([a["mdot_2"], a["mdot_3"]])
                lo = m if lo is None else np.minimum(lo, m)
                hi = m if hi is None else np.maximum(hi, m)
        on_sp = ((np.abs(a["P_s"] - sp["P_s"]) < 2e3) & (np.abs(a["P_d"] - sp["P_d"]) < 5e3)
                 & (np.abs(a["P_i"] - sp["P_i"]) < 5e3) & (np.abs(a["SH"] - self.SH[idx]) < 0.3))
        steady = ((hi - lo) < 0.01 * np.maximum(a["mdot_c"], 1e-6)).all(axis=0)
        return on_sp & steady, a


def reach(g):
    """Every case's steady state in up to four stages; returns (A, ok, method, resized)."""
    n = len(g.cases)
    A = {k: np.full(n, np.nan) for k in KEYS}
    ok = np.zeros(n, bool)
    method = np.array(["no steady state"] * n, dtype=object)

    def take(idx, a, sel, how):
        for k in KEYS:
            A[k][idx[sel]] = a[k][sel]
        ok[idx[sel]] = True
        method[idx[sel]] = how

    idx = np.arange(n)
    r = g.solve(idx)
    conv = r["converged"] & g.feasible
    take(idx, r["aux"], conv, "solved")
    best_x, best_u = r["x"].copy(), r["u"].copy()
    print(f"  stage 1, steady-state solver: {int(conv.sum())}/{n}")

    # 2. the mixing exchanger's liquid pool: a solver aid
    todo = np.flatnonzero(~ok & g.feasible)
    if len(todo):
        g.mx_evap[todo] = MX_EVAP
        r = g.solve(todo)
        take(todo, r["aux"], r["converged"], "solved, mixing exchanger aid")
        best_x[todo], best_u[todo] = r["x"], r["u"]
        print(f"  stage 2, mixing exchanger evaporating coefficient x{MX_EVAP:.2f}: "
              f"{int(r['converged'].sum())}/{len(todo)}")

    # 3. valves pinned open: enlarged
    todo = np.flatnonzero(~ok & g.feasible & (best_u >= U_PINNED).any(axis=1))
    if len(todo):
        g.kv[todo] *= np.where(best_u[todo] >= U_PINNED, ENLARGE, 1.0)
        r = g.solve(todo)
        take(todo, r["aux"], r["converged"], "solved, valve(s) enlarged")
        best_x[todo], best_u[todo] = r["x"], r["u"]
        print(f"  stage 3, pinned valves enlarged: {int(r['converged'].sum())}/{len(todo)}")

    # 4. PID settling, the quench valve smaller where it runs nearly shut
    todo = np.flatnonzero(~ok & g.feasible)
    if len(todo):
        small = best_u[todo, 2] < U3_SMALL
        g.kv[todo[small], 2] *= SMALL_V3
        r = g.solve(todo)                                  # a start with the new valves
        best_x[todo], best_u[todo] = r["x"], r["u"]
        take(todo, r["aux"], r["converged"], "solved, quench valve smaller")
        left = todo[~r["converged"]]
        if len(left):
            settled, a = g.settle(left, best_x[left], best_u[left])
            take(left, a, settled, "PID settled")
        print(f"  stage 4, PID settling (smaller quench valve where nearly shut): "
              f"{int(ok[todo].sum())}/{len(todo)}")
    resized = (g.kv != 1.0).any(axis=1) & ok
    return A, ok, method, resized


def mixer_liquid(A, pr, eps):
    """Liquid share leaving a mixer of effectiveness ``eps``, in equilibrium, entering it,
    and the effectiveness that just dries it."""
    m2, m3 = np.maximum(A["mdot_2"], 0.0), np.maximum(A["mdot_3"], 0.0)
    m = np.maximum(m2 + m3, 1e-9)
    P = np.where(np.isfinite(A["P_tee"]), A["P_tee"], 1e6)
    sat = pr.sat(P)
    h_v, h_l = sat["h_v"], sat["h_l"]
    h_lv = h_v - h_l
    x3 = np.clip((A["h_3f"] - h_l) / h_lv, 0.0, 1.0)            # quench flashed at the mixer
    q_gas = m2 * np.maximum(A["h_2f"] - h_v, 0.0)               # the gas's heat down to saturation
    liq_in = m3 * (1.0 - x3)
    y = np.maximum(liq_in - eps * q_gas / h_lv, 0.0) / m
    h_mix = (m2 * A["h_2f"] + m3 * A["h_3f"]) / m
    y_eq = np.maximum((h_v - h_mix) / h_lv, 0.0)
    eps_req = np.where(y_eq > 0.0, np.nan, liq_in * h_lv / np.maximum(q_gas, 1e-9))
    return y, y_eq, liq_in / m, m, eps_req


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stand", default="webui/config/stand_defaults.json", help="stand definition (defaults file)")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.25], help="mixer effectiveness(es)")
    ap.add_argument("--SH-R", type=float, nargs="+", default=[10.0], help="superheat at the compressor port [R]")
    ap.add_argument("--machine", nargs="+", default=list(MACHINES), metavar="NAME=VDISP@RPM",
                    help="machines: swept volume [m^3/rev] at speed [rpm]")
    ap.add_argument("--step", type=float, default=2.5, help="grid step [degF]")
    ap.add_argument("--T-amb", type=float, default=25.0, help="ambient temperature [degC]")
    ap.add_argument("--T-int", type=float, default=30.0,
                    help="liquid (receiver) saturation temperature [degC], kept 4 K under condensing")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="plant parameter overrides")
    ap.add_argument("--out", default="mixer_liquid_map", help="output file stem (.png, _eps.png, .csv)")
    args = ap.parse_args()

    doc = load_defaults(args.stand, create=False)
    over = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    machines = [parse_machine(s) for s in args.machine]
    lo_e, hi_e = c2f(Envelope().T_evap)
    lo_c, hi_c = c2f(Envelope().T_cond)
    st = args.step                                     # (a grid through 0/110 and 45/130 degF)
    Te_f = np.arange(st * np.floor(lo_e / st), hi_e + 1e-9, st)
    Tc_f = np.arange(st * np.floor(lo_c / st), hi_c + 1e-9, st)
    g = Grid(doc, over, machines, args.SH_R, Te_f, Tc_f, args.T_amb, args.T_int)
    print(f"{len(g.cases)} points")
    A, ok, method, resized = reach(g)
    print(f"{int(ok.sum())}/{len(g.cases)} reached, {int(resized.sum())} with resized valves")
    pr = g.props

    def grid_of(v, im, isx):
        Z = np.full(g.ok.shape, np.nan)
        for e, (jm, js, k) in enumerate(g.cases):
            if jm == im and js == isx and ok[e]:
                Z[tuple(g.pts[k])] = v[e]
        return Z

    def marks(im, isx, sel):
        return np.array([(c2f(g.Te[e]), c2f(g.Tc[e])) for e, (jm, js, _) in enumerate(g.cases)
                         if jm == im and js == isx and sel[e]]).reshape(-1, 2)

    rows, liq_panels, eps_panels = [], [], []
    for ie, eps in enumerate(args.eps):
        y, y_eq, y_in, m, eps_req = mixer_liquid(A, pr, eps)
        for im, (name, _, _) in enumerate(machines):
            for isx, sh in enumerate(args.SH_R):
                pan = dict(YE=grid_of(y_eq, im, isx), miss=marks(im, isx, ~ok & g.feasible),
                           resized=marks(im, isx, resized))
                liq_panels.append(dict(pan, title=f"{name}, eps {eps:g}, {sh:g} R at the port",
                                       Z=100 * grid_of(y, im, isx)))
                if ie == 0:
                    eps_panels.append(dict(pan, title=f"{name}, {sh:g} R at the port", Z=grid_of(eps_req, im, isx)))
        for e, (jm, js, _) in enumerate(g.cases):
            rows.append(dict(machine=machines[jm][0], V_disp=machines[jm][1], rpm=machines[jm][2],
                             SH_R=args.SH_R[js], eps=eps, T_evap_F=c2f(g.Te[e]), T_cond_F=c2f(g.Tc[e]),
                             T_int_F=c2f(pr.T_sat(np.array([g.P_i[e]]))[0] - C2K), reached=bool(ok[e]),
                             method=method[e], **{f"kv{v + 1}_factor": g.kv[e, v] for v in range(4)}, mx_alpha_e_factor=g.mx_evap[e],
                             mdot=m[e], mdot_2=A["mdot_2"][e], mdot_3=A["mdot_3"][e],
                             liquid_in_pct=100 * y_in[e], liquid_out_pct=100 * y[e], liquid_eq_pct=100 * y_eq[e],
                             eps_required=eps_req[e], Q_line_W=A["Q_sg"][e]))
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    plot(liq_panels, Te_f, Tc_f, args.out + ".png", np.arange(0.0, 30.0 + 1e-9, 2.5),
         "liquid leaving the mixer [% of the suction mass flow]", "{:.1f}")
    plot(eps_panels, Te_f, Tc_f, args.out + "_eps.png", np.round(np.arange(0.4, 1.0 + 1e-9, 0.05), 2),
         "mixer effectiveness needed for a dry outlet (below the band: 0.4 or less)", "{:.2f}")
    print(f"wrote {args.out}.png, {args.out}_eps.png, {args.out}.csv")


# sequential blue ramp, light (low) to dark (high)
RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#104281", "#0d366b")
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#d9d8d4"


def plot(panels, Te_f, Tc_f, path, levels, label, fmt):
    """Filled contours of one quantity, one panel per case."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon

    nb = len(levels) - 1
    cmap = ListedColormap([RAMP[round(i * (len(RAMP) - 1) / max(nb - 1, 1))] for i in range(nb)])
    norm = BoundaryNorm(levels, cmap.N)
    ncol = min(len(panels), 3)
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow + 0.9), squeeze=False,
                            sharex=True, sharey=True)
    poly_f = np.column_stack([c2f([v[0] for v in R410A_ENVELOPE]), c2f([v[1] for v in R410A_ENVELOPE])])
    TE, TC = np.meshgrid(Te_f, Tc_f)
    for ax, P in zip(axs.flat, panels):
        Z = np.clip(P["Z"], levels[0], levels[-1] - 1e-6)
        cs = ax.contourf(TE, TC, Z, levels=levels, cmap=cmap, norm=norm)
        ax.contour(TE, TC, Z, levels=levels[1:-1], colors="white", linewidths=0.6)
        wet = np.where(np.isnan(P["YE"]), np.nan, (P["YE"] > 1e-4).astype(float))
        if np.nansum(wet) > 0:
            ax.contourf(TE, TC, wet, levels=[0.5, 1.5], colors="none", hatches=["////"])
            ax.contour(TE, TC, wet, levels=[0.5], colors=INK, linewidths=1.2)
        if len(P["miss"]):
            ax.plot(*P["miss"].T, "s", ms=4, mfc=GRID, mec="none")
        if len(P["resized"]):
            ax.plot(*P["resized"].T, ".", ms=3, color=INK, alpha=0.55)
        ax.add_patch(Polygon(poly_f, closed=True, fill=False, ec=MUTED, lw=1.0))
        for te, tc, lab in ((0, 110, "0/110"), (45, 130, "45/130")):
            ax.plot(te, tc, "o", ms=6, mfc="white", mec=INK, mew=1.2)
            ax.annotate(lab, (te, tc), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_title(P["title"], fontsize=10, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axs.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("evaporating (dew) temperature [°F]", fontsize=9, color=MUTED)
    for ax in axs[:, 0]:
        ax.set_ylabel("condensing (dew) temperature [°F]", fontsize=9, color=MUTED)
    cb = fig.colorbar(cs, ax=axs.ravel().tolist(), orientation="horizontal", fraction=0.04, pad=0.09,
                      aspect=50, ticks=levels)
    cb.set_label(label, fontsize=9, color=INK)
    cb.ax.set_xticklabels([fmt.format(v) for v in levels])
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_visible(False)
    fig.legend(handles=[Patch(fc="white", ec=INK, hatch="////",
                              label="wet even in equilibrium (the suction line gains more than the port superheat holds)"),
                        Line2D([], [], ls="none", marker=".", ms=6, color=INK, alpha=0.55,
                               label="valves resized (bypass opened / smaller quench valve)"),
                        Line2D([], [], ls="none", marker="s", ms=6, mfc=GRID, mec="none", label="not reached"),
                        Patch(fc="white", ec=MUTED, label="envelope (blank inside: lift under 20 K)"),
                        Line2D([], [], ls="none", marker="o", ms=6, mfc="white", mec=INK, label="test points")],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
