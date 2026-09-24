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
        <div className="kpi"><div className="l">receiver level</div><div className="v">{fmt(t.rec_level * 100, 0)}<small>%</small></div></div>
        <div className="kpi"><div className="l">condenser flooded</div><div className="v">{fmt(t.cond_flood * 100, 0)}<small>%</small></div></div>
        <div className="kpi"><div className="l">quench outlet</div><div className="v" style={{ color: t.x_qo < 1 ? "var(--bad)" : undefined }}>
          {t.x_qo < 1 ? <>x {fmt(t.x_qo, 2)}</> : <>{fmt(t.T_qo, 1)}<small>°C</small></>}</div></div>
        <div className="kpi"><div className="l">liquid at compressor</div><div className="v" style={{ color: t.y_liq > 0.0005 ? "var(--bad)" : undefined }}>{fmt(t.y_liq * 100, 1)}<small>%</small></div></div>
        <div className="kpi"><div className="l">heat to water</div><div className="v">{fmt(t.Q_w / 1000, 2)}<small>kW</small></div></div>
      </div>
      <div className="chargebar">
        <div className="kpi">
          <div className="l">refrigerant charge</div>
          <div className="v">{fmt(ch.kg, 3)}<small>kg</small><small>· {fmt(ch.kg / ch.nominal_kg * 100, 0)} % of nominal {fmt(ch.nominal_kg, 3)} kg</small></div>
        </div>
        <div className="chargectl">
          <span className="l">amount</span>
          <input type="number" step={0.01} min={0} value={dq} onChange={(e) => setDq(e.target.value)} />
          <span className="l">kg</span>
          <button onClick={() => charge(-1)}>Recover</button>
          <button onClick={() => charge(+1)}>Add</button>
          <span className="l">at {fmt(ch.rate_kg_s * 1000, 0)} g/s{ch.pending_kg !== 0 ? ` · pending ${ch.pending_kg > 0 ? "+" : ""}${ch.pending_kg.toFixed(3)} kg` : ""}</span>
        </div>
      </div>
    </div>
  );
}
