import { createContext, useContext, useEffect, useRef, useState } from "react";
import { fmt } from "./api";

/** Display unit systems.  The backend (and the simulator) never change: the API speaks
 *  the base units below, and the UI converts for display and back for input. */
export type UnitSystem = "metric" | "english";
export const SYSTEMS: { key: UnitSystem; label: string }[] = [
  { key: "metric", label: "Metric (SI)" },
  { key: "english", label: "US (English)" },
];

/** Kinds of quantity, with the base unit the API uses for each. */
export type Qty =
  | "P"       // pressure, bar (absolute)
  | "dP"      // pressure drop, kPa
  | "T"       // temperature, °C
  | "dT"      // temperature difference (superheat, subcooling), K
  | "mdot"    // refrigerant mass flow, g/s
  | "flow_w"  // cooling water flow, kg/min
  | "heat"    // heat flow, W
  | "heatk"   // heat flow, kW
  | "power"   // electrical power, W
  | "mass"    // refrigerant mass, kg
  | "h"       // specific enthalpy, kJ/kg (the property tables' reference state)
  | "s"       // specific entropy, kJ/(kg K) (the property tables' reference state)
  | "rate";   // charging rate, g/s

interface Conv { unit: string; k: number; o?: number; dec: number; step: number }
// display = base * k + o
const TABLE: Record<Qty, Record<UnitSystem, Conv>> = {
  P: { metric: { unit: "bar", k: 1, dec: 2, step: 0.1 }, english: { unit: "psia", k: 14.5037738, dec: 1, step: 1 } },
  dP: { metric: { unit: "kPa", k: 1, dec: 1, step: 1 }, english: { unit: "psi", k: 0.145037738, dec: 2, step: 0.1 } },
  T: { metric: { unit: "°C", k: 1, dec: 1, step: 0.5 }, english: { unit: "°F", k: 1.8, o: 32, dec: 1, step: 1 } },
  dT: { metric: { unit: "K", k: 1, dec: 1, step: 0.5 }, english: { unit: "°F", k: 1.8, dec: 1, step: 1 } },
  mdot: { metric: { unit: "g/s", k: 1, dec: 1, step: 1 }, english: { unit: "lb/h", k: 7.93664144, dec: 0, step: 10 } },
  // water at 1000 kg/m³ (the model's cooling water density)
  flow_w: { metric: { unit: "kg/min", k: 1, dec: 1, step: 1 }, english: { unit: "gpm", k: 0.264172052, dec: 1, step: 1 } },
  heat: { metric: { unit: "kW", k: 1e-3, dec: 1, step: 1 }, english: { unit: "Btu/h", k: 3.41214163, dec: 0, step: 1000 } },
  heatk: { metric: { unit: "kW", k: 1, dec: 1, step: 1 }, english: { unit: "kBtu/h", k: 3.41214163, dec: 1, step: 1 } },
  power: { metric: { unit: "kW", k: 1e-3, dec: 2, step: 1 }, english: { unit: "kW", k: 1e-3, dec: 2, step: 1 } },
  mass: { metric: { unit: "kg", k: 1, dec: 3, step: 0.01 }, english: { unit: "lb", k: 2.20462262, dec: 2, step: 0.05 } },
  // a difference scale only: the English values keep the tables' reference state
  h: { metric: { unit: "kJ/kg", k: 1, dec: 1, step: 1 }, english: { unit: "Btu/lb", k: 0.429922614, dec: 1, step: 1 } },
  s: { metric: { unit: "kJ/(kg·K)", k: 1, dec: 3, step: 0.01 }, english: { unit: "Btu/(lb·°F)", k: 0.238845897, dec: 4, step: 0.001 } },
  rate: { metric: { unit: "g/s", k: 1, dec: 0, step: 1 }, english: { unit: "lb/min", k: 0.132277357, dec: 2, step: 0.1 } },
};

export interface Units {
  sys: UnitSystem;
  unit: (q: Qty) => string;
  /** base -> display */
  to: (q: Qty, v: number) => number;
  /** display -> base (for values entered by the operator) */
  from: (q: Qty, v: number) => number;
  /** base -> display for a difference (no offset: °C -> °F differences scale by 1.8 only) */
  diff: (q: Qty, v: number) => number;
  /** formatted display value (``dec`` overrides the default decimals; large whole numbers
   *  get thousands separators) */
  fmt: (q: Qty, v: number | undefined, dec?: number) => string;
  /** plain display value for an input field (no separators) */
  text: (q: Qty, v: number) => string;
  /** formatted display value with its unit */
  fmtU: (q: Qty, v: number | undefined, dec?: number) => string;
  dec: (q: Qty) => number;
  /** a sensible input step in display units */
  step: (q: Qty) => number;
}

export function makeUnits(sys: UnitSystem): Units {
  const c = (q: Qty) => TABLE[q][sys];
  const to = (q: Qty, v: number) => v * c(q).k + (c(q).o ?? 0);
  const f = (q: Qty, v: number | undefined, dec?: number) => {
    const d = dec ?? c(q).dec;
    if (v === undefined || v === null || Number.isNaN(v)) return fmt(undefined, d);
    const x = to(q, v);
    return d === 0 && Math.abs(x) >= 10000 ? Math.round(x).toLocaleString("en-US") : fmt(x, d);
  };
  return {
    sys,
    unit: (q) => c(q).unit,
    to,
    from: (q, v) => (v - (c(q).o ?? 0)) / c(q).k,
    diff: (q, v) => v * c(q).k,
    fmt: f,
    text: (q, v) => fmt(to(q, v), c(q).dec),
    fmtU: (q, v, dec) => `${f(q, v, dec)} ${c(q).unit}`,
    dec: (q) => c(q).dec,
    step: (q) => c(q).step,
  };
}

/** Quantity of a control loop's process value, from the loop's (base) unit. */
export const loopQty = (unit: string): Qty => (unit === "bar" ? "P" : "T");

const STORE = "hgbp.units";
export function loadSystem(): UnitSystem {
  try {
    const v = localStorage.getItem(STORE);
    if (v === "metric" || v === "english") return v;
  } catch { /* storage unavailable */ }
  return "metric";
}
export function saveSystem(sys: UnitSystem) {
  try { localStorage.setItem(STORE, sys); } catch { /* storage unavailable */ }
}

export const UnitsContext = createContext<Units>(makeUnits("metric"));
export const useUnits = () => useContext(UnitsContext);

/** An input field for quantity ``q``: the text is in display units, starts at ``base``
 *  (base units), and is converted when the unit system changes.  Returns the text, its
 *  setter and a function giving the entered value in base units (NaN if not a number). */
export function useQtyInput(q: Qty, base: number): [string, (s: string) => void, () => number] {
  const u = useUnits();
  const [text, setText] = useState(() => u.text(q, base));
  const prev = useRef(u);
  useEffect(() => {
    if (prev.current.sys === u.sys) return;
    const v = Number(text);
    if (text.trim() !== "" && Number.isFinite(v)) setText(u.text(q, prev.current.from(q, v)));
    prev.current = u;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [u]);
  return [text, setText, () => (text.trim() === "" ? NaN : u.from(q, Number(text)))];
}
