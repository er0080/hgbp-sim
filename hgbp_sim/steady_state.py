"""Steady-state (equilibrium) solver for the HGBP plant.

Given target suction pressure, discharge pressure, suction superheat,
intermediate (condensing) pressure and compressor speed, find the four valve
positions and the remaining states for which all time derivatives vanish.
Used for warm-start initialization, for checking whether a test point is
reachable with the installed charge, and for steady-state performance data.

In steady state the liquid inventory of the intermediate section (receiver
level, condenser flooding) is fixed by the total refrigerant charge (mass
conservation), so the charge (or, alternatively, a receiver level) is an
input of the solve.

The solve is split in two, because the mixing exchanger's internal profile
barely changes its outlet state (a generously sized counterflow exchanger is
insensitive to where the quench dries out), which makes one coupled Newton
problem ill-conditioned:

1. the stand as a whole, with the suction side's overall steady mass and
   energy balance as its equations (a batched Levenberg-Marquardt iteration
   with finite-difference Jacobians over valve positions, discharge enthalpy
   and wall temperatures), holding the exchanger profile fixed;
2. the exchanger profile for the resulting flows, by marching its own
   (stable) quench-cell and plate-wall dynamics to rest.

The two are repeated until the liquid held up in the exchanger (which enters
the charge balance) stops changing.  The solver is compiled
(:mod:`hgbp_sim.kernel.steady`); every environment is solved on its own, the
environments of a batch in parallel.
"""
from __future__ import annotations

import numpy as np

from .kernel import layout as L
from .kernel import steady as ks


def fill_enthalpy(props, P, V, fill):
    """Mean enthalpy of a volume V at pressure P holding liquid fill fraction ``fill``."""
    sat = props.sat(P)
    M_l = sat["rho_l"] * V * fill
    M_v = sat["rho_v"] * V * (1.0 - fill)
    xq = M_v / (M_l + M_v)
    return sat["h_l"] + xq * (sat["h_v"] - sat["h_l"])


def solve_steady_state(plant, P_s, P_d, SH, N, P_i=None, charge=None, fill=None,
                       T_amb=None, T_wi=None, idx=None, max_iter=40, tol=2e-3,
                       verbose=False, rounds=10):
    """Solve for the equilibrium of environments ``idx`` at the given targets.

    Parameters
    ----------
    plant : HGBPPlant
    P_s, P_d : target suction / discharge pressure [Pa]
    SH : suction superheat at the probe [K]      N : compressor speed [rpm]
    P_i : intermediate (condensing) pressure [Pa]; default P_d - dP_i_margin
    charge : total refrigerant mass [kg] (default: the plant's ``p.charge``);
        ignored if ``fill`` is given
    fill : receiver level (share of its volume, condenser drained) instead of
        a charge (the resulting total mass is returned as ``charge``)
    T_amb, T_wi : ambient / cooling-water inlet temperature [K] (default: the
        plant's current inputs)

    Returns
    -------
    dict with ``x`` (m, NX) full state matrix (sensor states set to their
    steady values), ``u`` (m, 4) valve positions, ``converged`` (m,) bool,
    ``resid`` (m,) final scaled residual norm, ``aux`` outputs at the
    solution, ``charge`` (m,) total mass at the solution.
    """
    idx = np.arange(plant.n) if idx is None else np.atleast_1d(np.asarray(idx))
    m = len(idx)
    bc = lambda v, default: np.ascontiguousarray(
        np.broadcast_to(np.asarray(default if v is None else v, float), (m,)))
    P_s, P_d, SH, N = bc(P_s, 0), bc(P_d, 0), bc(SH, 0), bc(N, 0)
    T_amb = bc(T_amb, plant.T_amb[idx])
    T_wi = bc(T_wi, plant.T_wi[idx])
    rec = plant.p.packed()[idx]
    P_i = bc(P_i, P_d - rec["dP_i_margin"])
    use_fill = fill is not None
    fill = bc(fill if use_fill else 0.0, 0.0)
    charge = bc(None if use_fill else charge, rec["charge"])
    x = np.zeros((m, plant.NX))
    aux = np.empty((m, L.NA))
    info = np.empty((m, 3))
    ks.solve_batch[plant._parallel(m)](rec, plant.props.tab, P_s, P_d, SH, N, P_i, charge, fill, use_fill,
                                       T_amb, T_wi, int(max_iter), float(tol), int(rounds), x, aux, info)
    a = plant._aux_dict(aux, plant.p.charge[idx])
    if verbose:
        print(f"steady state: {int(info[:, 0].sum())}/{m} converged, residual max {info[:, 1].max():.2e}, "
              f"{int(info[:, 2].max())} iterations at most")
    return dict(x=x, u=x[:, plant.UV].copy(), converged=info[:, 0] > 0.5, resid=info[:, 1], aux=a,
                charge=a["M_tot"], iterations=int(info[:, 2].sum()))
