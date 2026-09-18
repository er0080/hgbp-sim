import { useState } from "react";
import { api, fmt } from "../api";
import type { Snapshot } from "../types";

export default function KpiTiles({ snap }: { snap: Snapshot }) {
  const m = snap.meas, t = snap.true, ch = snap.charge;
  const [dq, setDq] = useState("0.05");
  const charge = (sign: number) => api("/api/charge", { delta_kg: sign * Number(dq) });
  return (
    <div className="card">
      <h2>Process values</h2>
      <div className="kpis">
        <div className="kpi"><div className="l">mass flow</div><div className="v">{fmt(m.mdot, 1)}<small>g/s</small></div></div>
        <div className="kpi"><div className="l">power</div><div className="v">{fmt(m.W / 1000, 2)}<small>kW</small></div></div>
        <div className="kpi"><div className="l">superheat</div><div className="v">{fmt(m.SH, 1)}<small>K</small></div></div>
        <div className="kpi"><div className="l">subcooling</div><div className="v">{fmt(m.SC, 1)}<small>K</small></div></div>
        <div className="kpi"><div className="l">pressure ratio</div><div className="v">{fmt(t.Pr, 2)}</div></div>
        <div className="kpi"><div className="l">volumetric eff.</div><div className="v">{fmt(t.eta_v * 100, 0)}<small>%</small></div></div>
        <div className="kpi"><div className="l">condenser liquid</div><div className="v">{fmt(t.fill_i * 100, 0)}<small>%</small></div></div>
        <div className="kpi"><div className="l">accumulator liquid</div><div className="v">{fmt(t.fill_s * 100, 1)}<small>%</small></div></div>
        <div className="kpi"><div className="l">inlet quality</div><div className="v" style={{ color: t.x_out < 1 ? "var(--bad)" : undefined }}>{fmt(t.x_out, 3)}</div></div>
        <div className="kpi"><div className="l">heat to water</div><div className="v">{fmt(t.Q_w / 1000, 2)}<small>kW</small></div></div>
        <div className="kpi"><div className="l">charge</div><div className="v">{fmt(ch.kg, 3)}<small>kg</small></div><div className="l">{fmt(ch.kg / ch.nominal_kg * 100, 0)} % of nominal</div></div>
        <div className="kpi">
          <div className="l">charge / recover</div>
          <div style={{ display: "flex", gap: 4, alignItems: "center", marginTop: 4 }}>
            <button className="small" onClick={() => charge(-1)}>-</button>
            <input type="number" step={0.01} min={0} value={dq} onChange={(e) => setDq(e.target.value)} style={{ width: "5em" }} />
            <button className="small" onClick={() => charge(+1)}>+</button>
          </div>
          <div className="l">kg at {fmt(ch.rate_kg_s * 1000, 0)} g/s{ch.pending_kg !== 0 ? ` · pending ${ch.pending_kg.toFixed(3)}` : ""}</div>
        </div>
      </div>
    </div>
  );
}
