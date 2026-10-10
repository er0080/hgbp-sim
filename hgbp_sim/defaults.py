"""Stand defaults as one JSON document: simulation settings, the four
controller loops (in UT35A units, see :mod:`hgbp_sim.ut35a`) and every plant
parameter.

The built-in defaults come from the code (``PlantParams`` field defaults,
``control.DEFAULT_GAINS`` and the loop configuration below).  A defaults file
overrides them: sections or keys it leaves out keep the built-in value, keys
it does not know are an error (so a typo cannot be silently ignored), and
keys starting with ``_`` are comments.  Write the built-in document with

    python -m hgbp_sim.defaults stand_defaults.json
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import fields

from . import ut35a
from .control import DEFAULT_GAINS
from .params import _RANGES, FLUIDS, PlantParams

LOOP_NAMES = ("dpv", "spv", "stv", "water")
VALVE_CHARS = ("linear", "eqpct", "quick")

# SP and PV input range RL..RH in the loop's display unit (bar absolute or degC),
# output limits OL..OH in %.  P, I, D and DR are derived from DEFAULT_GAINS.
# UT35A PV input filter FL: OFF with the built-in gains, which were tuned without it (a 4 s
# filter makes their discharge pressure loop overshoot to the trip at a cold start); the
# stand's defaults file sets FL = 4 with the stand's own tuning, as on its controllers
PV_FILTER = "OFF"
_LOOPS = dict(
    dpv=dict(SP=33.89, RL=0.0, RH=50.0, OL=2.0, OH=100.0),
    spv=dict(SP=9.98, RL=0.0, RH=30.0, OL=0.0, OH=100.0),
    stv=dict(SP=18.33, RL=-50.0, RH=150.0, OL=0.0, OH=100.0),
    water=dict(SP=18.00, RL=0.0, RH=50.0, OL=2.0, OH=100.0),
)
_GAIN_SCALE = dict(dpv=1e5, spv=1e5, stv=1.0, water=1e5)     # SI gain -> gain per display unit

_SIMULATION = dict(speed_factor=1.0, noise=True, dt_ctrl=0.2, T_amb=25.0, T_wi=20.0, charge_rate_g_s=5.0,
                   short_cycle_timers=True)
_SIM_BOOLS = ("noise", "short_cycle_timers")
_SIM_RANGES = dict(speed_factor=(0.1, 100.0), dt_ctrl=(0.1, 10.0), T_amb=(-20.0, 60.0), T_wi=(0.0, 50.0),
                   charge_rate_g_s=(1.0, 250.0))

_ABOUT = [
    "Defaults of the simulated test stand, read when the backend starts (restart to apply edits).",
    "simulation: speed_factor [-], noise [bool], dt_ctrl control interval [s] (the UT35A runs at 0.2 s),",
    "  T_amb ambient / T_wi cooling water inlet [degC], charge_rate_g_s refrigerant charging rate [g/s],",
    "  short_cycle_timers [bool] compressor minimum off time 60 s / minimum run time 120 s.",
    "loops: one Yokogawa UT35A per loop. SP setpoint and RL..RH PV input range in bar absolute (dpv, spv,",
    "  water) or degC (stv); P proportional band [% of RL..RH]; I integral and D derivative time [s] or",
    "  \"OFF\"; DR \"DIR\" (output rises with PV) or \"RVS\"; OL..OH output limits [%]; FL PV input filter",
    "  (first-order lag ahead of the PV display and the PID) [s] or \"OFF\".",
    "plant: model parameters in SI units as listed on the Settings tab (units and descriptions there).",
]


def _plant_fields():
    return [f for f in fields(PlantParams) if f.init]


def builtin_defaults() -> dict:
    """The defaults defined in code, as a JSON-serializable dict."""
    loops = {}
    for k in LOOP_NAMES:
        cfg = _LOOPS[k]
        g = {n: DEFAULT_GAINS[k][n] * _GAIN_SCALE[k] for n in ("Kp", "Ki", "Kd")}
        tune = ut35a.from_gains(g["Kp"], g["Ki"], g["Kd"], cfg["RH"] - cfg["RL"])
        loops[k] = dict(SP=cfg["SP"], P=tune["P"], I=tune["I"], D=tune["D"], DR=tune["DR"],
                        RL=cfg["RL"], RH=cfg["RH"], OL=cfg["OL"], OH=cfg["OH"], FL=PV_FILTER)
    p = PlantParams()
    return dict(_about=list(_ABOUT), simulation=dict(_SIMULATION), loops=loops,
                plant={f.name: getattr(p, f.name) for f in _plant_fields()})


# ------------------------------------------------------------- validation
def _num(v, where: str, rng=None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: expected a number, got {v!r}")
    v = float(v)
    if rng is not None and not rng[0] <= v <= rng[1]:
        raise ValueError(f"{where}: {v} is outside {rng[0]}..{rng[1]}")
    return v


def _check_keys(given: dict, known, where: str) -> None:
    if not isinstance(given, dict):
        raise ValueError(f"{where}: expected an object")
    unknown = sorted(k for k in given if not k.startswith("_") and k not in known)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {', '.join(unknown)}")


def validate_loop(cfg: dict, where: str) -> dict:
    """Normalized copy of one loop configuration (raises ValueError)."""
    out = dict(SP=_num(cfg["SP"], f"{where}.SP"), P=ut35a.normalize_band(cfg["P"]),
               I=ut35a.normalize_time(cfg["I"], f"{where}.I"), D=ut35a.normalize_time(cfg["D"], f"{where}.D"),
               DR=ut35a.normalize_action(cfg["DR"]), RL=_num(cfg["RL"], f"{where}.RL"),
               RH=_num(cfg["RH"], f"{where}.RH"), OL=_num(cfg["OL"], f"{where}.OL", (0.0, 100.0)),
               OH=_num(cfg["OH"], f"{where}.OH", (0.0, 100.0)), FL=ut35a.normalize_filter(cfg["FL"]))
    if out["RL"] >= out["RH"]:
        raise ValueError(f"{where}: RL must be below RH")
    if not out["RL"] <= out["SP"] <= out["RH"]:
        raise ValueError(f"{where}: SP {out['SP']} is outside the input range {out['RL']}..{out['RH']}")
    if out["OL"] >= out["OH"]:
        raise ValueError(f"{where}: OL must be below OH")
    return out


def _validate_plant_value(f, v):
    where = f"plant.{f.name}"
    if f.name == "fluid":
        if v not in FLUIDS:
            raise ValueError(f"{where}: {v!r} is not one of {', '.join(FLUIDS)}")
        return v
    if f.name.endswith("_char"):
        if v not in VALVE_CHARS:
            raise ValueError(f"{where}: {v!r} is not one of {', '.join(VALVE_CHARS)}")
        return v
    if f.name == "charge" and v is None:
        return None
    return _num(v, where, _RANGES.get(f.name))


def merge_defaults(doc: dict) -> dict:
    """Built-in defaults overridden by ``doc`` (a parsed defaults file),
    validated and normalized.  Raises ValueError with the offending key."""
    base = builtin_defaults()
    _check_keys(doc, ("simulation", "loops", "plant"), "defaults")

    sim = doc.get("simulation", {})
    _check_keys(sim, _SIMULATION, "simulation")
    for k, v in sim.items():
        if k.startswith("_"):
            continue
        if k in _SIM_BOOLS:
            if not isinstance(v, bool):
                raise ValueError(f"simulation.{k}: expected true or false, got {v!r}")
            base["simulation"][k] = v
        else:
            base["simulation"][k] = _num(v, f"simulation.{k}", _SIM_RANGES.get(k))

    loops = doc.get("loops", {})
    _check_keys(loops, LOOP_NAMES, "loops")
    for k in LOOP_NAMES:
        given = loops.get(k, {})
        _check_keys(given, base["loops"][k], f"loops.{k}")
        base["loops"][k] = validate_loop({**base["loops"][k], **given}, f"loops.{k}")

    plant = doc.get("plant", {})
    by_name = {f.name: f for f in _plant_fields()}
    _check_keys(plant, by_name, "plant")
    for k, v in plant.items():
        if not k.startswith("_"):
            base["plant"][k] = _validate_plant_value(by_name[k], v)
    return base


def write_defaults(path: str, doc: dict | None = None) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(builtin_defaults() if doc is None else doc, fh, indent=2)
        fh.write("\n")


def load_defaults(path: str | None, create: bool = True) -> dict:
    """Validated defaults from ``path``.  No path: the built-in defaults.  A
    missing file is created with the built-in defaults when ``create``."""
    if not path:
        return merge_defaults({})
    if not os.path.exists(path):
        if not create:
            raise FileNotFoundError(path)
        write_defaults(path)
    with open(path, encoding="utf-8") as fh:
        try:
            doc = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    try:
        return merge_defaults(doc)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def controller_settings(doc: dict) -> dict:
    """Keyword arguments for :class:`hgbp_sim.control.BaselineController` from the
    ``loops`` section of a defaults document: the stand's UT35A tuning as SI gains,
    the output limits OL..OH and the PV input filters FL."""
    gains, limits, pv_filter = {}, {}, {}
    for k in LOOP_NAMES:
        t = doc["loops"][k]
        Kp, Ki, Kd = ut35a.to_gains(t["P"], t["I"], t["D"], t["DR"], t["RH"] - t["RL"])
        gains[k] = dict(Kp=Kp / _GAIN_SCALE[k], Ki=Ki / _GAIN_SCALE[k], Kd=Kd / _GAIN_SCALE[k])
        limits[k] = (t["OL"] / 100.0, t["OH"] / 100.0)
        pv_filter[k] = None if t.get("FL", ut35a.OFF) == ut35a.OFF else float(t["FL"])
    return dict(gains=gains, limits=limits, pv_filter=pv_filter)


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) > 1:
        write_defaults(sys.argv[1])
    else:
        json.dump(builtin_defaults(), sys.stdout, indent=2)
        sys.stdout.write("\n")
