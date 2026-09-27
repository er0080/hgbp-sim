"""Compiled (numba) model kernel: property lookups, right-hand side,
integrator and steady-state solver."""
from __future__ import annotations

import os
import types

import numba
from numba import njit

if "NUMBA_THREADING_LAYER" not in os.environ:
    # Batches run on numba's thread pool.  Its OpenMP layer keeps idle workers spinning
    # between launches, which slows down the Python code in between (the environment's
    # own step, the learner) and oversubscribes the cores; the workqueue layer does not.
    # (workqueue must not be launched from several Python threads at once; a single
    # environment runs serially, without the pool.)
    numba.config.THREADING_LAYER = "workqueue"


def batch_variants(fn):
    """Serial and parallel builds ``{False: ..., True: ...}`` of a loop over the
    environments of a batch (written with ``prange``).  Each build gets its own
    function object and name: numba's on-disk cache does not tell builds of one
    function with different options apart."""
    out = {}
    for par in (False, True):
        name = fn.__name__ + ("_parallel" if par else "_serial")
        f = types.FunctionType(fn.__code__, fn.__globals__, name, fn.__defaults__, fn.__closure__)
        f.__qualname__ = name
        f.__doc__ = fn.__doc__
        out[par] = njit(cache=True, parallel=par)(f)
    return out
