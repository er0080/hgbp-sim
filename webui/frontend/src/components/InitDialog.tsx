import { useEffect, useState } from "react";
import { api } from "../api";
import type { ParamsView, Snapshot } from "../types";
import { useQtyInput, useUnits } from "../units";

export default function InitDialog({ mode, snap, onClose }: { mode: "cold" | "warm"; snap: Snapshot; onClose: () => void }) {
  const u = useUnits();
  const [T_amb, setTamb, T_ambBase] = useQtyInput("T", snap.meas.T_amb);
  const [T_wi, setTwi, T_wiBase] = useQtyInput("T", snap.meas.T_wi);
  const [liq, setLiq] = useState("0");
  const [named, setNamed] = useState("MT_standard");
  const [custom, setCustom] = useState(false);
  // custom test point, in display units
  const ptIn = {
    T_evap: useQtyInput("T", -10), T_cond: useQtyInput("T", 45), T_int: useQtyInput("T", 38), SH: useQtyInput("dT", 10),
  };
  const [N, setN] = useState(String(Math.round(snap.compressor.speed_sp)));
  const [points, setPoints] = useState<ParamsView["named_points"]>({});
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { api<ParamsView>("/api/params").then((v) => setPoints(v.named_points)).catch(() => {}); }, []);
  const go = async () => {
    setBusy(true); setErr("");
    try {
      if (mode === "cold") {
        await api("/api/init", { mode: "cold", T_amb: T_ambBase(), T_wi: T_wiBase(), liquid_in_suction: Number(liq) });
      } else {
        const point = custom ? { T_evap: ptIn.T_evap[2](), T_cond: ptIn.T_cond[2](), T_int: ptIn.T_int[2](), SH: ptIn.SH[2](), N: Number(N) } : named;
        await api("/api/init", { mode: "warm", T_amb: T_ambBase(), T_wi: T_wiBase(), point });
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
        <div className="form-row"><label>ambient temperature [{u.unit("T")}]</label><input type="number" value={T_amb} onChange={(e) => setTamb(e.target.value)} /></div>
        <div className="form-row"><label>cooling water inlet [{u.unit("T")}]</label><input type="number" value={T_wi} onChange={(e) => setTwi(e.target.value)} /></div>
        {mode === "cold" && (
          <>
            <div className="form-row"><label>share of the liquid charge migrated to the suction side (rest in the receiver)</label><input type="number" min={0} max={1} step={0.05} value={liq} onChange={(e) => setLiq(e.target.value)} /></div>
            <p className="note">A mass split of the liquid, not a level. Migrated liquid fills the compressor's internal suction volume first, then the suction line and the mixing exchanger, as after a long off cycle; with the nominal charge, 0.05 is about 0.5 kg. The receiver level is liquid volume over receiver volume, as on the sight glass.</p>
          </>
        )}
        {mode === "warm" && (
          <>
            <div className="form-row"><label>test point</label>
              <select value={custom ? "custom" : named} onChange={(e) => { if (e.target.value === "custom") setCustom(true); else { setCustom(false); setNamed(e.target.value); } }}>
                {Object.entries(points).map(([k, p]) => <option key={k} value={k}>{k}: {u.fmt("T", p.T_evap, 0)}/{u.fmt("T", p.T_cond, 0)} {u.unit("T")}, int {u.fmt("T", p.T_int, 0)} {u.unit("T")}, SH {u.fmt("dT", p.SH, 0)} {u.unit("dT")}</option>)}
                <option value="custom">custom...</option>
              </select>
            </div>
            {custom && (["T_evap", "T_cond", "T_int", "SH"] as const).map((k) => (
              <div className="form-row" key={k}><label>{{ T_evap: "evaporating (suction sat.)", T_cond: "condensing (discharge sat.)", T_int: "intermediate sat.", SH: "superheat" }[k]} [{u.unit(k === "SH" ? "dT" : "T")}]</label>
                <input type="number" value={ptIn[k][0]} onChange={(e) => ptIn[k][1](e.target.value)} /></div>
            ))}
            {custom && <div className="form-row"><label>speed [rpm]</label>
              <input type="number" value={N} onChange={(e) => setN(e.target.value)} /></div>}
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
