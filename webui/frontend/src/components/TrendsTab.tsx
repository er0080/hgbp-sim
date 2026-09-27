import { useEffect, useMemo, useRef, useState } from "react";
import type { History, Snapshot } from "../types";
import UPlotChart, { SeriesDef } from "./UPlotChart";
import { useUnits, type Qty, type Units } from "../units";

const C = { blue: "#4fc3f7", red: "#ef5350", green: "#66bb6a", orange: "#ffa726", yellow: "#ffd54f", purple: "#ba68c8", cyan: "#26c6da", grey: "#b0bec5", pink: "#f06292" };

type Def = { key: string; label: string; color: string; dash?: number[]; width?: number; qty?: Qty; unit?: string };
const SP = { dash: [6, 4], width: 1 };

// history channels are in the API's base units; ``qty`` converts them to the display units
const CHART_DEFS: { title: string; series: Def[] }[] = [
  { title: "Pressures", series: [
    { key: "P_d", label: "discharge", color: C.red, qty: "P" }, { key: "sp_P_d", label: "discharge SP", color: C.red, qty: "P", ...SP },
    { key: "P_i", label: "intermediate", color: C.orange, qty: "P" }, { key: "sp_P_i", label: "intermediate SP", color: C.orange, qty: "P", ...SP },
    { key: "P_s", label: "suction", color: C.blue, qty: "P" }, { key: "sp_P_s", label: "suction SP", color: C.blue, qty: "P", ...SP } ] },
  { title: "Superheat and subcooling", series: [
    { key: "SH", label: "superheat", color: C.green, qty: "dT" }, { key: "SC", label: "subcooling", color: C.cyan, qty: "dT" } ] },
  { title: "Valve positions", series: [
    { key: "u1", label: "1 discharge pressure", color: C.red, unit: "-" }, { key: "u2", label: "2 suction pressure", color: C.blue, unit: "-" },
    { key: "u3", label: "3 suction temperature", color: C.green, unit: "-" }, { key: "u4", label: "4 cooling water", color: C.cyan, unit: "-" } ] },
  { title: "Temperatures", series: [
    { key: "T_d", label: "discharge", color: C.red, qty: "T" }, { key: "T_s", label: "suction", color: C.blue, qty: "T" },
    { key: "sp_T_s", label: "suction SP", color: C.blue, qty: "T", ...SP },
    { key: "T_co", label: "liquid to valve 3", color: C.cyan, qty: "T" }, { key: "T_sh", label: "shell", color: C.orange, qty: "T" },
    { key: "T_cw", label: "condenser wall", color: C.purple, qty: "T" }, { key: "T_wo", label: "water out", color: C.grey, qty: "T" } ] },
  { title: "Mixing exchanger", series: [
    { key: "T_go", label: "gas outlet S2", color: C.red, qty: "T" }, { key: "T_qo", label: "quench outlet S4", color: C.cyan, qty: "T" },
    { key: "T_s", label: "suction probe", color: C.blue, qty: "T" }, { key: "Q_mx", label: "duty", color: C.orange, qty: "heatk" } ] },
  { title: "Flow, power, speed", series: [
    { key: "mdot", label: "mass flow", color: C.blue, qty: "mdot" }, { key: "W", label: "power", color: C.orange, unit: "W" },
    { key: "N", label: "speed", color: C.green, unit: "rpm" }, { key: "mdot_w", label: "water", color: C.cyan, qty: "flow_w" } ] },
  { title: "Inventory and charge", series: [
    { key: "rec_level", label: "receiver level", color: C.cyan, unit: "-" }, { key: "cond_flood", label: "condenser flooded", color: C.purple, unit: "-" },
    { key: "M_q_liq", label: "liquid in mixing exch.", color: C.blue, qty: "mass" }, { key: "y_liq", label: "liquid share at compressor", color: C.red, unit: "-" },
    { key: "charge", label: "charge", color: C.yellow, qty: "mass" } ] },
];

/** Chart definitions in display units: the axis carries the unit when all series share it,
 *  otherwise every series label does. */
function charts(u: Units) {
  return CHART_DEFS.map((c) => {
    const unitOf = (s: Def) => (s.qty ? u.unit(s.qty) : s.unit ?? "");
    const units = [...new Set(c.series.map(unitOf))];
    const mixed = units.length > 1;
    const series: SeriesDef[] = c.series.map((s) => ({
      key: s.key, color: s.color, dash: s.dash, width: s.width,
      label: mixed && unitOf(s) !== "-" ? `${s.label} [${unitOf(s)}]` : s.label,
      conv: s.qty ? ((q: Qty) => (v: number) => u.to(q, v))(s.qty) : undefined,
    }));
    return { title: c.title, unit: units.filter((x) => x !== "-").join(" · ") || "-", series };
  });
}

export default function TrendsTab({ history, version, snap }: { history: React.MutableRefObject<History>; version: number; snap: Snapshot }) {
  const u = useUnits();
  const CHARTS = useMemo(() => charts(u), [u]);
  const [windowS, setWindowS] = useState(900);
  const [frozen, setFrozen] = useState(false);
  const [shown, setShown] = useState(0);
  const lastDraw = useRef(0);
  // redraw the charts at most every 250 ms, always with the newest data (a throttle: new
  // data keeps the pending redraw at its time instead of postponing it)
  useEffect(() => {
    if (frozen) return;
    const wait = Math.max(0, lastDraw.current + 250 - Date.now());
    const id = window.setTimeout(() => { lastDraw.current = Date.now(); setShown(version); }, wait);
    return () => clearTimeout(id);
  }, [version, frozen]);
  return (
    <div>
      <div className="toolbar">
        <label className="note">window</label>
        <select value={windowS} onChange={(e) => setWindowS(Number(e.target.value))}>
          <option value={300}>5 min</option><option value={900}>15 min</option><option value={1800}>30 min</option>
          <option value={3600}>1 h</option><option value={0}>all</option>
        </select>
        <button onClick={() => setFrozen((f) => !f)} className={frozen ? "warn" : ""}>{frozen ? "Resume" : "Freeze"}</button>
        <a href={`/api/export.csv?units=${u.sys}`} download><button>Export CSV</button></a>
        <span className="note">sim time {snap.t.toFixed(0)} s · {history.current.t?.length ?? 0} samples in buffer</span>
      </div>
      <div className="trend-grid">
        {CHARTS.map((c) => (
          <div className="card" key={c.title}>
            <UPlotChart title={c.title} unit={c.unit} series={c.series} history={history} version={shown} windowS={windowS} />
          </div>
        ))}
      </div>
    </div>
  );
}
