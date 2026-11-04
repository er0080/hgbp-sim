"""Sizing sweep for the suction exchanger (``mixer`` = 1): a tee where the hot gas and the
quench join, then a small brazed plate exchanger with both circuits piped in series
(docs/suction-exchanger.md).

Every candidate geometry is solved to its steady state at each operating point with the
full stand model (the rest of the stand from a stand definition file), so the exchanger
sees the tee's worst-case inlet: the quench's unflashed liquid in the superheated gas.
Reported per candidate: heating surface, the exchanger's pressure drop (S1 connection to
S4 connection), the liquid share entering it and leaving it, where along the flow path
the liquid is gone, and the superheat at its outlet and at the compressor port.

    python examples/bphe_sizing.py                      # the stand's defaults file
    python examples/bphe_sizing.py --stand my_stand.json --F-wet 0.3 0.5
    python examples/bphe_sizing.py --set UA_sa=4      # any plant parameter, e.g. a better insulated suction line

The first run compiles the model (about a minute).
"""
import argparse
import itertools

import numpy as np

from hgbp_sim.defaults import load_defaults
from hgbp_sim.geometry import derive
from hgbp_sim.params import PlantParams
from hgbp_sim.plant import HGBPPlant
from hgbp_sim.steady_state import solve_steady_state

PSI = 6894.757


def degF(f):
    return (f - 32.0) / 1.8 + 273.15


# operating points: name, swept volume [m^3/rev], speed [rpm], evaporating / condensing [degF],
# superheat at the compressor port [R]
POINTS = (
    ("12T 60 Hz", 1.68e-4, 3500.0, 0.0, 110.0, 10.0),
    ("small 35 Hz", 1.05e-4, 2071.0, 0.0, 110.0, 10.0),
)
# candidate plates: effective length, width [m], port bore [m]; corrugation angles [deg]; plate counts
PLATES = ((0.29, 0.12, 0.033), (0.47, 0.12, 0.033))
BETAS = (30.0, 45.0)
COUNTS = (20, 30, 40, 50)


def dry_point(x_q, x_g, tol=1e-4):
    """Share of the flow path (pass 1 then pass 2, cell by cell) after which the stream
    carries no liquid; None if it leaves wet."""
    path = list(x_q) + list(x_g[::-1])
    for k, xv in enumerate(path):
        if xv >= 1.0 - tol:
            return (k + 1) / len(path)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stand", default="webui/config/stand_defaults.json", help="stand definition (defaults file)")
    ap.add_argument("--F-wet", type=float, nargs="+", default=[0.5], help="wetted plate share(s) bx_F_wet")
    ap.add_argument("--T-amb", type=float, default=25.0, help="ambient temperature [degC]")
    ap.add_argument("--T-int", type=float, default=30.0, help="liquid (receiver) saturation temperature [degC]")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="plant parameter overrides")
    args = ap.parse_args()

    doc = load_defaults(args.stand, create=False)
    T_wi = doc["simulation"]["T_wi"] + 273.15
    over = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    base = PlantParams(**{**doc["plant"], "mixer": 1.0, "N_nom": 3500.0, **over})
    cases = list(itertools.product(range(len(POINTS)), PLATES, BETAS, COUNTS, args.F_wet))
    plant = HGBPPlant(base, n=len(cases))
    p = plant.p
    for e, (ip, (L, W, d_port), beta, N_pl, fw) in enumerate(cases):
        p.V_disp[e] = POINTS[ip][1]
        p.bx_L[e], p.bx_W[e], p.bx_d_port[e], p.bx_beta[e], p.bx_n_plates[e], p.bx_F_wet[e] = L, W, d_port, beta, N_pl, fw
        # (the plate weight with the plate area: 8 kg for 30 plates of 0.47 m x 0.12 m)
        p.bx_mass[e] = 8.0 * (N_pl / 30.0) * (L * W) / (0.47 * 0.12)
    derive(p, plant.MX)
    p.__dict__["_rec"] = None
    plant.set_inputs(T_amb=np.full(plant.n, args.T_amb + 273.15), T_wi=np.full(plant.n, T_wi))
    pr = plant.props
    P_s = np.array([float(pr.P_sat(degF(POINTS[c[0]][3]))) for c in cases])
    P_d = np.array([float(pr.P_sat(degF(POINTS[c[0]][4]))) for c in cases])
    P_i = np.full(len(cases), float(pr.P_sat(np.array([args.T_int + 273.15]))[0]))
    SH = np.array([POINTS[c[0]][5] / 1.8 for c in cases])
    N = np.array([POINTS[c[0]][2] for c in cases])
    res = solve_steady_state(plant, P_s, P_d, SH, N, P_i=P_i)     # the stand's (nominal) charge
    a = res["aux"]

    last = None
    for e, (ip, (L, W, d_port), beta, N_pl, fw) in enumerate(cases):
        if ip != last:
            name, Vd, rpm, Te, Tc, sh = POINTS[ip]
            print(f"\n{name}: {Vd * 1e6:.0f} cm3/rev at {rpm:.0f} rpm, {Te:.0f}/{Tc:.0f} degF, {sh:.0f} R at the port, "
                  f"liquid at {args.T_int:.0f} degC, T_amb {args.T_amb:.0f} degC")
            print(f"  {'plate':>12} {'beta':>4} {'N':>3} {'Fw':>4} {'A m2':>5} {'dP kPa':>6} {'psi':>5} {'mdot':>6} "
                  f"{'y_in':>5} {'y_out':>6} {'dry at':>7} {'SH_out R':>8} {'SH_port R':>9} {'Q_line W':>8}")
            last = ip
        g = lambda k: float(np.asarray(a[k])[e])
        dp = dry_point(a["x_q"][e], a["x_g"][e])
        T_sat_out = float(pr.T_sat(np.array([a["P_gf"][e][0]]))[0])     # leaving pass 2 (S4)
        SH_out = (g("T_go") - T_sat_out) * 1.8
        flag = "" if res["converged"][e] else "  (not converged)"
        print(f"  {L:5.2f}x{W:4.2f}/{d_port * 1e3:2.0f} {beta:4.0f} {N_pl:3.0f} {fw:4.2f} {float(p.bx_A[e]):5.2f} "
              f"{g('dP_bx') / 1e3:6.2f} {g('dP_bx') / PSI:5.2f} {g('mdot_c'):6.4f} {g('y_bi'):5.3f} {g('y_bo'):6.4f} "
              f"{'wet' if dp is None else f'{dp:.0%}':>7} {SH_out:8.1f} {(g('T_s') - g('T_sat_s')) * 1.8:9.1f} "
              f"{g('Q_sg'):8.0f}{flag}")


if __name__ == "__main__":
    main()
