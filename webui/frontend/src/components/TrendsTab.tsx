import { useEffect, useState } from "react";
import type { History, Snapshot } from "../types";
import UPlotChart, { SeriesDef } from "./UPlotChart";

const C = { blue: "#4fc3f7", red: "#ef5350", green: "#66bb6a", orange: "#ffa726", yellow: "#ffd54f", purple: "#ba68c8", cyan: "#26c6da", grey: "#b0bec5", pink: "#f06292" };

const CHARTS: { title: string; unit: string; series: SeriesDef[] }[] = [
  { title: "Pressures", unit: "bar", series: [
    { key: "P_d", label: "discharge", color: C.red }, { key: "sp_P_d", label: "discharge SP", color: C.red, dash: [6, 4], width: 1 },
    { key: "P_i", label: "intermediate", color: C.orange }, { key: "sp_P_i", label: "intermediate SP", color: C.orange, dash: [6, 4], width: 1 },
    { key: "P_s", label: "suction", color: C.blue }, { key: "sp_P_s", label: "suction SP", color: C.blue, dash: [6, 4], width: 1 } ] },
  { title: "Superheat and subcooling", unit: "K", series: [
    { key: "SH", label: "superheat", color: C.green }, { key: "SC", label: "subcooling", color: C.cyan } ] },
  { title: "Valve positions", unit: "-", series: [
    { key: "u1", label: "1 discharge pressure", color: C.red }, { key: "u2", label: "2 suction pressure", color: C.blue },
    { key: "u3", label: "3 suction temperature", color: C.green }, { key: "u4", label: "4 cooling water", color: C.cyan } ] },
  { title: "Temperatures", unit: "°C", series: [
    { key: "T_d", label: "discharge", color: C.red }, { key: "T_s", label: "suction", color: C.blue },
    { key: "sp_T_s", label: "suction SP", color: C.blue, dash: [6, 4], width: 1 },
    { key: "T_co", label: "condenser outlet", color: C.cyan }, { key: "T_sh", label: "shell", color: C.orange },
    { key: "T_cw", label: "condenser wall", color: C.purple }, { key: "T_wo", label: "water out", color: C.grey } ] },
  { title: "Flow, power, speed", unit: "g/s · W/100 · rpm/100", series: [
    { key: "mdot", label: "mass flow [g/s]", color: C.blue }, { key: "W", label: "power [W]", color: C.orange },
    { key: "N", label: "speed [rpm]", color: C.green }, { key: "mdot_w", label: "water [kg/min]", color: C.cyan } ] },
  { title: "Inventory and charge", unit: "-", series: [
    { key: "fill_i", label: "condenser liquid fill", color: C.cyan }, { key: "fill_s", label: "accumulator liquid fill", color: C.blue },
    { key: "x_out", label: "compressor inlet quality", color: C.red }, { key: "charge", label: "charge [kg]", color: C.yellow } ] },
];

export default function TrendsTab({ history, version, snap }: { history: React.MutableRefObject<History>; version: number; snap: Snapshot }) {
  const [windowS, setWindowS] = useState(900);
  const [frozen, setFrozen] = useState(false);
  const [shown, setShown] = useState(0);
  // throttle chart updates to ~5 Hz
  useEffect(() => {
    if (frozen) return;
    const id = window.setTimeout(() => setShown(version), 200);
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
        <a href="/api/export.csv" download><button>Export CSV</button></a>
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
