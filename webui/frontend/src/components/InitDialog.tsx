import { useEffect, useState } from "react";
import { api } from "../api";
import type { ParamsView, Snapshot } from "../types";

export default function InitDialog({ mode, snap, onClose }: { mode: "cold" | "warm"; snap: Snapshot; onClose: () => void }) {
  const [T_amb, setTamb] = useState(snap.meas.T_amb.toFixed(1));
  const [T_wi, setTwi] = useState(snap.meas.T_wi.toFixed(1));
  const [liq, setLiq] = useState("0.5");
  const [named, setNamed] = useState("MT_standard");
  const [custom, setCustom] = useState(false);
  const [pt, setPt] = useState({ T_evap: "-10", T_cond: "45", T_int: "38", SH: "10", N: String(Math.round(snap.compressor.speed_sp)) });
  const [points, setPoints] = useState<ParamsView["named_points"]>({});
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { api<ParamsView>("/api/params").then((v) => setPoints(v.named_points)).catch(() => {}); }, []);
  const go = async () => {
    setBusy(true); setErr("");
    try {
      if (mode === "cold") {
        await api("/api/init", { mode: "cold", T_amb: Number(T_amb), T_wi: Number(T_wi), liquid_in_accumulator: Number(liq) });
      } else {
        const point = custom ? { T_evap: Number(pt.T_evap), T_cond: Number(pt.T_cond), T_int: Number(pt.T_int), SH: Number(pt.SH), N: Number(pt.N) } : named;
        await api("/api/init", { mode: "warm", T_amb: Number(T_amb), T_wi: Number(T_wi), point });
      }
      onClose();
    } catch (e: any) { setErr(String(e.message ?? e)); }
    setBusy(false);
  };
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>{mode === "cold" ? "Cold start (equalized stand, compressor off)" : "Warm start (equilibrium at a test point)"}</h3>
        {snap.pending_params.length > 0 && <p className="note">Pending parameter changes will be applied: {snap.pending_params.join(", ")}</p>}
        <div className="form-row"><label>ambient temperature [°C]</label><input type="number" value={T_amb} onChange={(e) => setTamb(e.target.value)} /></div>
        <div className="form-row"><label>cooling water inlet [°C]</label><input type="number" value={T_wi} onChange={(e) => setTwi(e.target.value)} /></div>
        {mode === "cold" && (
          <div className="form-row"><label>share of the liquid charge placed in the accumulator (rest in the condenser)</label><input type="number" min={0} max={1} step={0.1} value={liq} onChange={(e) => setLiq(e.target.value)} /></div>
          <p className="note">This is a mass split of the liquid, not a quality or a level. With the nominal charge, half of the liquid is about 0.6 kg, which fills roughly 4 % of the 12 L accumulator; the panel's "liquid %" readouts are liquid volume divided by vessel volume.</p>
        )}
        {mode === "warm" && (
          <>
            <div className="form-row"><label>test point</label>
              <select value={custom ? "custom" : named} onChange={(e) => { if (e.target.value === "custom") setCustom(true); else { setCustom(false); setNamed(e.target.value); } }}>
                {Object.entries(points).map(([k, p]) => <option key={k} value={k}>{k}: {p.T_evap}/{p.T_cond} °C, int {p.T_int} °C, SH {p.SH} K</option>)}
                <option value="custom">custom...</option>
              </select>
            </div>
            {custom && (["T_evap", "T_cond", "T_int", "SH", "N"] as const).map((k) => (
              <div className="form-row" key={k}><label>{{ T_evap: "evaporating (suction sat.) [°C]", T_cond: "condensing (discharge sat.) [°C]", T_int: "intermediate sat. [°C]", SH: "superheat [K]", N: "speed [rpm]" }[k]}</label>
                <input type="number" value={pt[k]} onChange={(e) => setPt({ ...pt, [k]: e.target.value })} /></div>
            ))}
            <p className="note">The warm start solves the equilibrium with the current charge; an unreachable point is reported.</p>
          </>
        )}
        {err && <p className="err">{err}</p>}
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 12 }}>
          <button onClick={onClose}>Cancel</button>
          <button className="primary" disabled={busy} onClick={go}>{busy ? "working..." : "Initialize"}</button>
        </div>
      </div>
    </div>
  );
}
