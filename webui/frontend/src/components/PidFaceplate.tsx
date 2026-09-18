import { useEffect, useState } from "react";
import { api, fmt } from "../api";
import type { Loop } from "../types";

export default function PidFaceplate({ name, loop }: { name: string; loop: Loop }) {
  const [sp, setSp] = useState(String(loop.sp.toFixed(2)));
  const [out, setOut] = useState(String((loop.manual_out * 100).toFixed(0)));
  const [editingSp, setEditingSp] = useState(false);
  const [editingOut, setEditingOut] = useState(false);
  useEffect(() => { if (!editingSp) setSp(loop.sp.toFixed(2)); }, [loop.sp, editingSp]);
  useEffect(() => { if (!editingOut) setOut((loop.manual_out * 100).toFixed(0)); }, [loop.manual_out, editingOut]);

  const send = (body: any) => api(`/api/loop/${name}`, body);
  const applySp = () => { const v = Number(sp); if (Number.isFinite(v)) send({ sp: v }); setEditingSp(false); };
  const applyOut = () => { const v = Number(out); if (Number.isFinite(v)) send({ out: Math.min(100, Math.max(0, v)) / 100 }); setEditingOut(false); };
  const dec = loop.unit === "bar" ? 2 : 1;
  const step = loop.unit === "bar" ? 0.1 : 0.5;
  const bump = (d: number) => send({ sp: Math.round((loop.sp + d) * 100) / 100 });
  const err = loop.sp - loop.pv;

  return (
    <div className="card fp">
      <div className="title">
        <b>{loop.label}</b>
        <span>{loop.valve_label}</span>
      </div>
      <div className="pv">{fmt(loop.pv, dec)}<small>{loop.unit}</small>
        <small style={{ color: Math.abs(err) > (loop.unit === "bar" ? 0.05 : 0.5) ? "var(--warn)" : "var(--ok)" }}>
          {err >= 0 ? "+" : ""}{fmt(err, dec)} to SP
        </small>
      </div>
      <div className="row">
        <label>SP</label>
        <input type="number" step={step} value={sp} onFocus={() => setEditingSp(true)}
          onChange={(e) => setSp(e.target.value)} onBlur={applySp}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
        <button className="small" onClick={() => bump(-step)}>-{step}</button>
        <button className="small" onClick={() => bump(step)}>+{step}</button>
        <span className="note">{loop.unit}</span>
      </div>
      <div className="row">
        <label>OUT</label>
        <div className="bar"><i style={{ width: `${loop.out * 100}%` }} /></div>
        <span style={{ fontFamily: "var(--mono)", width: 52, textAlign: "right" }}>{fmt(loop.out * 100, 1)} %</span>
      </div>
      <div className="row">
        <label>mode</label>
        <div className="mode">
          <button className={loop.mode === "auto" ? "active" : ""} onClick={() => send({ mode: "auto" })}>AUTO</button>
          <button className={loop.mode === "manual" ? "active" : ""} onClick={() => send({ mode: "manual" })}>MAN</button>
        </div>
        {loop.mode === "manual" && (
          <>
            <input type="number" min={0} max={100} step={1} value={out} onFocus={() => setEditingOut(true)}
              onChange={(e) => setOut(e.target.value)} onBlur={applyOut}
              onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
            <span className="note">% open</span>
            <input type="range" min={0} max={100} step={1} value={Number(out)} style={{ flex: 1, minWidth: 80 }}
              onChange={(e) => { setOut(e.target.value); send({ out: Number(e.target.value) / 100 }); }} />
          </>
        )}
      </div>
      <div className="gains">Kp {loop.Kp.toPrecision(3)} /{loop.unit} · Ki {loop.Ki.toPrecision(3)} /({loop.unit}·s) · Kd {loop.Kd.toPrecision(3)}</div>
    </div>
  );
}
