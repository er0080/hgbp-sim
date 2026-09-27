import { api, fmt } from "../api";
import type { Snapshot } from "../types";
import { useQtyInput, useUnits } from "../units";

export default function KpiTiles({ snap }: { snap: Snapshot }) {
  const m = snap.meas, t = snap.true, ch = snap.charge;
  const u = useUnits();
  const [dq, setDq, dqBase] = useQtyInput("mass", 0.05);
  const qw = u.fmt("heat", t.Q_w, u.sys === "metric" ? 2 : 0);
  const charge = (sign: number) => { const v = dqBase(); if (Number.isFinite(v)) api("/api/charge", { delta_kg: sign * v }); };
  return (
    <div className="card">
      <h2>Process values</h2>
      <div className="kpis">
        <div className="kpi"><div className="l">mass flow</div><div className="v">{u.fmt("mdot", m.mdot)}<small>{u.unit("mdot")}</small></div></div>
        <div className="kpi"><div className="l">power</div><div className="v">{u.fmt("power", m.W)}<small>{u.unit("power")}</small></div></div>
        <div className="kpi"><div className="l">superheat</div><div className="v">{u.fmt("dT", m.SH)}<small>{u.unit("dT")}</small></div></div>
        <div className="kpi"><div className="l">subcooling</div><div className="v">{u.fmt("dT", m.SC)}<small>{u.unit("dT")}</small></div></div>
        <div className="kpi"><div className="l">pressure ratio</div><div className="v">{fmt(t.Pr, 2)}</div></div>
        <div className="kpi"><div className="l">volumetric eff.</div><div className="v">{fmt(t.eta_v * 100, 0)}<small>%</small></div></div>
        <div className="kpi"><div className="l">receiver level</div><div className="v">{fmt(t.rec_level * 100, 0)}<small>%</small></div></div>
        <div className="kpi"><div className="l">condenser flooded</div><div className="v">{fmt(t.cond_flood * 100, 0)}<small>%</small></div></div>
        <div className="kpi"><div className="l">quench outlet</div><div className="v" style={{ color: t.x_qo < 1 ? "var(--bad)" : undefined }}>
          {t.x_qo < 1 ? <>x {fmt(t.x_qo, 2)}</> : <>{u.fmt("T", t.T_qo)}<small>{u.unit("T")}</small></>}</div></div>
        <div className="kpi"><div className="l">liquid at compressor</div><div className="v" style={{ color: t.y_liq > 0.0005 ? "var(--bad)" : undefined }}>{fmt(t.y_liq * 100, 1)}<small>%</small></div></div>
        <div className="kpi"><div className="l">heat to water</div><div className={`v${qw.length > 6 ? " long" : ""}`}>{qw}<small>{u.unit("heat")}</small></div></div>
      </div>
      <div className="chargebar">
        <div className="kpi">
          <div className="l">refrigerant charge</div>
          <div className="v">{u.fmt("mass", ch.kg)}<small>{u.unit("mass")}</small><small>· {fmt(ch.kg / ch.nominal_kg * 100, 0)} % of nominal {u.fmtU("mass", ch.nominal_kg)}</small></div>
        </div>
        <div className="chargectl">
          <span className="l">amount</span>
          <input type="number" step={u.step("mass")} min={0} value={dq} onChange={(e) => setDq(e.target.value)} />
          <span className="l">{u.unit("mass")}</span>
          <button onClick={() => charge(-1)}>Recover</button>
          <button onClick={() => charge(+1)}>Add</button>
          <span className="l">at {u.fmtU("rate", ch.rate_kg_s * 1000)}{ch.pending_kg !== 0 ? ` · pending ${ch.pending_kg > 0 ? "+" : ""}${u.fmtU("mass", ch.pending_kg)}` : ""}</span>
        </div>
      </div>
    </div>
  );
}
