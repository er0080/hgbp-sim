import { useEffect, useState } from "react";
import { api } from "../api";
import { useQtyInput, useUnits } from "../units";

interface Sat { fluid: string; P: number; T_min: number; T_max: number; in_range: boolean }

/** Dew point (saturated vapor, x = 1) pressure of the stand's refrigerant for a saturation
 *  temperature: the pressure of a compressor state point at that evaporating / condensing
 *  temperature. */
export default function SatCalc({ fluid }: { fluid: string }) {
  const u = useUnits();
  const [T, setT, Tbase] = useQtyInput("T", 0);
  const [sat, setSat] = useState<Sat | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    const v = Tbase();
    if (!Number.isFinite(v)) { setSat(null); return; }
    const id = window.setTimeout(() => {
      api<Sat>(`/api/saturation?T=${v}`).then((r) => { setSat(r); setErr(""); })
        .catch((e) => setErr(String(e.message ?? e)));
    }, 150);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [T, fluid, u]);
  return (
    <div className="card">
      <h2>Saturation pressure (dew point, x = 1) · {fluid}</h2>
      <div className="row" style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <span className="note">saturation temperature</span>
        <input type="number" step={u.step("T")} value={T} onChange={(e) => setT(e.target.value)} />
        <span className="note">{u.unit("T")}</span>
        <span className="note">→</span>
        {sat && sat.in_range && <b>{u.fmtU("P", sat.P)}</b>}
        {sat && !sat.in_range && <span className="err">outside the tables ({u.fmt("T", sat.T_min)} to {u.fmt("T", sat.T_max)} {u.unit("T")})</span>}
      </div>
      {err && <p className="err">{err}</p>}
    </div>
  );
}
