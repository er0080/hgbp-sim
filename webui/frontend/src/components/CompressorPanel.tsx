import { useEffect, useState } from "react";
import { api, fmt } from "../api";
import type { Snapshot } from "../types";

const PERM_LABELS: Record<string, string> = {
  P_s_above_min: "suction pressure above low limit",
  P_s_below_max: "suction pressure below high limit",
  P_d_below_max: "discharge pressure below limit",
  valve1_open: "valve 1 (discharge) at least 10 % open",
  valve2_open: "valve 2 (bypass) at least 5 % open",
  no_trip: "no trip latched",
  off_time_elapsed: "anti-short-cycle off time elapsed",
};

export default function CompressorPanel({ snap }: { snap: Snapshot }) {
  const c = snap.compressor;
  const [speed, setSpeed] = useState(String(Math.round(c.speed_sp)));
  const [editing, setEditing] = useState(false);
  useEffect(() => { if (!editing) setSpeed(String(Math.round(c.speed_sp))); }, [c.speed_sp, editing]);
  const cmd = (body: any) => api("/api/compressor", body);
  const stateClass = c.tripped ? "TRIPPED" : c.state;
  const canStart = !c.run_request && !c.tripped && c.state === "OFF";
  const canStop = c.run_request || c.state === "RUNNING" || c.state === "STARTING";
  return (
    <div className="card">
      <h2>Compressor</h2>
      <div className="comp">
        <div>
          <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 10 }}>
            <span className={`state ${stateClass}`}>{c.tripped ? "TRIPPED" : c.state}</span>
            <span className="note">{fmt(c.t_state, 0)} s in state</span>
          </div>
          <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
            <button className="ok" disabled={!canStart} onClick={() => cmd({ run: true })}>START</button>
            <button className="bad" disabled={!canStop} onClick={() => cmd({ run: false })}>STOP</button>
            {c.tripped && <button className="warn" disabled={c.state !== "OFF"} onClick={() => cmd({ reset: true })}>RESET TRIP</button>}
          </div>
          <div className="row" style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span className="note">speed SP</span>
            <input type="number" min={c.N_min} max={c.N_max} step={10} value={speed} onFocus={() => setEditing(true)}
              onChange={(e) => setSpeed(e.target.value)}
              onBlur={() => { cmd({ speed: Number(speed) }); setEditing(false); }}
              onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
            <span className="note">rpm ({c.N_min}-{c.N_max})</span>
          </div>
          <div className="kpis" style={{ marginTop: 10 }}>
            <div className="kpi"><div className="l">speed</div><div className="v">{fmt(c.speed, 0)}<small>rpm</small></div></div>
            <div className="kpi"><div className="l">power</div><div className="v">{fmt(snap.meas.W / 1000, 2)}<small>kW</small></div></div>
            <div className="kpi"><div className="l">shell temp.</div><div className="v">{fmt(snap.true.T_sh, 0)}<small>°C</small></div></div>
          </div>
          {c.tripped && <p className="err">Trip: {c.trip_reasons.join(", ") || "latched"}. Stop condition cleared? Press RESET TRIP, then START.</p>}
        </div>
        <div>
          <div className="note">start permissives</div>
          <ul className="perm">
            {Object.entries(c.permissives).map(([k, v]) => (
              <li key={k} className={v ? "ok" : ""}>{PERM_LABELS[k] ?? k}</li>
            ))}
          </ul>
          <div className="note" style={{ marginTop: 8 }}>min off {c.min_off_time} s · min run {c.min_run_time} s · run request: {c.run_request ? "ON" : "off"}</div>
        </div>
      </div>
    </div>
  );
}
