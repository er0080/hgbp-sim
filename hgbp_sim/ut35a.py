"""PID tuning in the units of the Yokogawa UT35A/UT32A controllers on the stand.

The UT35A (IM 05P01D31-01EN, sections 6.4 and 8.3) computes, in % of span,

    OUT = 100 / P * ( e + 1/Ti * integral(e dt) + Td * d(PV)/dt )

with e the deviation in % of the PV input span, and these parameters:

    P    proportional band, % of the PV input span (RL..RH)   0.1 .. 999.9 %, 0.1 % steps
    I    integral time [s]                                    OFF or 1 .. 6000 s, 1 s steps
    D    derivative time [s] (on PV, standard mode)           OFF or 1 .. 6000 s, 1 s steps
    DR   DIR: output rises when PV rises (e = PV - SP)
         RVS: output falls when PV rises (e = SP - PV)
    FL   PV input filter [s] (section 7.1): first-order lag on the PV input,
         ahead of both the PV display and the control computation
                                                              OFF or 1 .. 120 s, 1 s steps

The simulator's :class:`hgbp_sim.control.PID` works in the parallel form
u = Kp e + Ki integral(e dt) - Kd dy/dt with e = SP - PV in engineering units
and u in 0..1, so

    |Kp| = 100 / (P * span),   Ki = Kp / Ti,   Kd = Kp * Td,   Kp < 0 for DIR.

The PV-derivative form matches the UT35A's standard PID mode (ALG = 0) in
AUTO + LOCAL.  The simulator stores the integral in output units, so a tuning
change does not bump the output.
"""
from __future__ import annotations

OFF = "OFF"
P_RANGE = (0.1, 999.9)
T_RANGE = (1, 6000)
FL_RANGE = (1, 120)
ACTIONS = ("DIR", "RVS")


def normalize_time(v, name: str):
    """Integral or derivative time: ``"OFF"`` (also None, 0, "" or "off") or
    whole seconds in 1..6000.  Raises ValueError outside the range."""
    if v is None or (isinstance(v, str) and v.strip().upper() in ("", OFF)):
        return OFF
    t = float(v)
    if t == 0.0:
        return OFF
    t = int(round(t))
    if not T_RANGE[0] <= t <= T_RANGE[1]:
        raise ValueError(f"{name} must be OFF or {T_RANGE[0]}..{T_RANGE[1]} s, got {v!r}")
    return t


def normalize_filter(v):
    """PV input filter time constant FL: ``"OFF"`` (also None, 0, "" or "off") or whole
    seconds in 1..120.  Raises ValueError outside the range."""
    if v is None or (isinstance(v, str) and v.strip().upper() in ("", OFF)):
        return OFF
    t = float(v)
    if t == 0.0:
        return OFF
    t = int(round(t))
    if not FL_RANGE[0] <= t <= FL_RANGE[1]:
        raise ValueError(f"FL must be OFF or {FL_RANGE[0]}..{FL_RANGE[1]} s, got {v!r}")
    return t


def normalize_band(v) -> float:
    """Proportional band in % rounded to the controller's 0.1 % resolution."""
    p = round(float(v), 1)
    if not P_RANGE[0] <= p <= P_RANGE[1]:
        raise ValueError(f"P must be {P_RANGE[0]}..{P_RANGE[1]} %, got {v!r}")
    return p


def normalize_action(v) -> str:
    a = str(v).strip().upper()
    if a not in ACTIONS:
        raise ValueError(f"DR must be one of {ACTIONS}, got {v!r}")
    return a


def to_gains(P, I, D, DR, span):
    """(Kp, Ki, Kd) per engineering unit of PV (e = SP - PV, output 0..1)."""
    kp = 100.0 / (normalize_band(P) * float(span))
    if normalize_action(DR) == "DIR":
        kp = -kp
    ti, td = normalize_time(I, "I"), normalize_time(D, "D")
    ki = 0.0 if ti == OFF else kp / ti
    kd = 0.0 if td == OFF else kp * td
    return kp, ki, kd


def from_gains(Kp, Ki, Kd, span) -> dict:
    """Nearest UT35A setting (P, I, D, DR) to parallel-form gains per
    engineering unit.  ``Kp`` must be nonzero."""
    if Kp == 0.0:
        raise ValueError("Kp = 0 has no proportional band")
    return dict(P=normalize_band(100.0 / (abs(Kp) * float(span))),
                I=OFF if Ki == 0.0 else normalize_time(Kp / Ki, "I"),
                D=OFF if Kd == 0.0 else normalize_time(Kd / Kp, "D"),
                DR="DIR" if Kp < 0.0 else "RVS")
