"""Compile every kernel ahead of use (numba caches the machine code next to the
modules, so later runs start at once):

    python -m hgbp_sim.kernel.warmup

The first run takes a minute or two; afterwards (and in any later process) it
takes seconds.  The web UI's Docker image runs it at build time.
"""
from __future__ import annotations

import time

import numpy as np


def warm_up(verbose: bool = True) -> float:
    """Run the serial (single environment) and parallel (batch) builds of the
    integrator, the right-hand side and the steady-state solver once; returns
    the seconds it took."""
    from ..params import PlantParams
    from ..plant import HGBPPlant
    from ..scenarios import named_point
    from ..steady_state import solve_steady_state

    t0 = time.perf_counter()
    for n in (1, 2):
        pl = HGBPPlant(PlantParams(), n=n)
        pt = named_point("MT_standard", pl.props, pl.params.N_nom)
        res = solve_steady_state(pl, pt["P_s"], pt["P_d"], pt["SH"], pt["N"], P_i=pt["P_i"])
        solve_steady_state(pl, pt["P_s"], pt["P_d"], pt["SH"], pt["N"], P_i=pt["P_i"], fill=0.4)
        pl.set_state(np.arange(n), res["x"])
        pl.step(0.2, u_cmd=res["u"], N_cmd=pt["N"])
        pl.cold_start()
        pl.step(0.2)
        pl.rhs(pl.x, pl.u_cmd, pl.N_cmd, pl.T_amb, pl.T_wi, hold_P_s=True)
        pl._stiff(pl.x)
        if verbose:
            print(f"compiled for {'a single environment' if n == 1 else 'batches'} "
                  f"({time.perf_counter() - t0:.1f} s)")
    return time.perf_counter() - t0


if __name__ == "__main__":
    warm_up()
