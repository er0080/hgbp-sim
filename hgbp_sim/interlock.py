"""Compressor start/stop interlock state machine (vectorized).

The same logic is used by the training environment and by the live stand so
that a controller trained in simulation meets exactly the sequencing it will
find on the PLC:

    OFF      -> STARTING  run request AND permissives AND off time >= min_off_time
    STARTING -> RUNNING   speed >= 90 % of the speed setpoint
    STARTING/RUNNING -> STOPPING   stop request AND run time >= min_run_time
    STOPPING -> OFF       speed below half the minimum speed
    any      -> OFF       trip (latched until reset)

Permissives (start allowed): suction pressure between 1.5x the low trip and
0.9x the high trip, discharge pressure below 0.8x its trip, valve 1 at least
10 % open (no dead-heading), valve 2 at least 5 % open (bypass path), no
latched trip.
"""
from __future__ import annotations

import numpy as np

ST_OFF, ST_STARTING, ST_RUNNING, ST_STOPPING = 0, 1, 2, 3
STATE_NAMES = ("OFF", "STARTING", "RUNNING", "STOPPING")


def permissives(P_s, P_d, u1, u2, p, tripped=None) -> np.ndarray:
    """Start permissives from measured pressures, valve positions and the
    parameter namespace ``p`` (limits)."""
    ok = ((P_s > 1.5 * p.P_s_min) & (P_s < 0.9 * p.P_s_max) & (P_d < 0.8 * p.P_d_max)
          & (u1 >= 0.1) & (u2 >= 0.05))
    if tripped is not None:
        ok = ok & ~np.asarray(tripped, bool)
    return ok


class Interlock:
    """Batch of ``n`` state machines.  ``state`` and ``t_state`` (time since
    the last transition) are public arrays."""

    def __init__(self, n: int, min_off_time: float = 60.0, min_run_time: float = 120.0):
        self.n = n
        self.min_off_time = float(min_off_time)
        self.min_run_time = float(min_run_time)
        self.state = np.zeros(n, int)
        self.t_state = np.full(n, self.min_off_time)
        self.tripped = np.zeros(n, bool)

    def reset(self, idx=None, running=False) -> None:
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx))
        self.state[idx] = ST_RUNNING if running else ST_OFF
        self.t_state[idx] = self.min_run_time if running else self.min_off_time
        self.tripped[idx] = False

    def step(self, run_req, perm, N, N_sp, N_min, dt: float, trip=None) -> dict:
        """Advance by one control interval.  Returns dict with ``N_cmd``
        (speed command), ``blocked`` (start request refused), ``switched``."""
        run_req = np.asarray(run_req, bool)
        perm = np.asarray(perm, bool)
        st = self.state
        if trip is not None:
            self.tripped = self.tripped | np.asarray(trip, bool)
        perm = perm & ~self.tripped
        off_ok = (st == ST_OFF) & (self.t_state >= self.min_off_time)
        blocked = (st == ST_OFF) & run_req & ~(perm & off_ok)
        start = (st == ST_OFF) & run_req & perm & off_ok
        stop = np.isin(st, (ST_STARTING, ST_RUNNING)) & ~run_req & (self.t_state >= self.min_run_time)
        up = (st == ST_STARTING) & (N >= 0.9 * N_sp)
        down = (st == ST_STOPPING) & (N < 0.5 * N_min)
        new = st.copy()
        new[start] = ST_STARTING
        new[up] = ST_RUNNING
        new[stop] = ST_STOPPING
        new[down] = ST_OFF
        # a latched trip forces the compressor off immediately
        forced = self.tripped & (new != ST_OFF)
        new[forced & (N >= 0.5 * N_min)] = ST_STOPPING
        new[forced & (N < 0.5 * N_min)] = ST_OFF
        switched = new != st
        self.state = new
        self.t_state = np.where(switched, 0.0, self.t_state + dt)
        N_cmd = np.where(np.isin(new, (ST_STARTING, ST_RUNNING)), N_sp, 0.0)
        return dict(N_cmd=N_cmd, blocked=blocked, switched=switched, start=start, stop=stop)

    def acknowledge(self, idx=None) -> None:
        """Operator trip reset (allowed once the compressor is off)."""
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx))
        can = self.state[idx] == ST_OFF
        self.tripped[idx[can]] = False
