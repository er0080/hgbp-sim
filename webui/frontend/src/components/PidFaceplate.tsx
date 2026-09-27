import { useEffect, useState } from "react";
import { api, fmt } from "../api";
import type { Loop } from "../types";
import { loopQty, useUnits } from "../units";

export default function PidFaceplate({ name, loop }: { name: string; loop: Loop }) {
  // the loop's values are in its base unit (bar or °C); shown and entered in the chosen units
  const u = useUnits();
  const q = loopQty(loop.unit);
  const dec = u.dec(q);
  const [sp, setSp] = useState(u.text(q, loop.sp));
  const [out, setOut] = useState(String((loop.manual_out * 100).toFixed(0)));
  const [editingSp, setEditingSp] = useState(false);
  const [editingOut, setEditingOut] = useState(false);
  useEffect(() => { if (!editingSp) setSp(u.text(q, loop.sp)); }, [loop.sp, editingSp, u, q]);
  useEffect(() => { if (!editingOut) setOut((loop.manual_out * 100).toFixed(0)); }, [loop.manual_out, editingOut]);

  const send = (body: any) => api(`/api/loop/${name}`, body);
  const clampSp = (v: number) => Math.min(loop.RH, Math.max(loop.RL, v));      // the UT35A keeps SP inside RL..RH
  const applySp = () => {
    const v = Number(sp);
    if (sp.trim() !== "" && Number.isFinite(v)) send({ sp: clampSp(u.from(q, v)) });
    setEditingSp(false);
  };
  const applyOut = () => { const v = Number(out); if (Number.isFinite(v)) send({ out: Math.min(100, Math.max(0, v)) / 100 }); setEditingOut(false); };
  const step = u.step(q);
  const bump = (d: number) => send({ sp: clampSp(u.from(q, Math.round((u.to(q, loop.sp) + d) * 100) / 100)) });
  const err = loop.sp - loop.pv;                                              // base units
  const errD = u.diff(q, err);
  const errS = fmt(Math.abs(errD) < 0.5 * 10 ** -dec ? 0 : errD, dec);        // no "-0.00"
  const rng = (v: number) => String(Number(u.to(q, v).toFixed(1)));

  return (
    <div className="card fp">
      <div className="title">
        <b>{loop.label}</b>
        <span>{loop.valve_label}</span>
      </div>
      <div className="pv">{u.fmt(q, loop.pv)}<small>{u.unit(q)}</small>
        <small style={{ color: Math.abs(err) > (q === "P" ? 0.05 : 0.5) ? "var(--warn)" : "var(--ok)" }}>
          {errS.startsWith("-") ? "" : "+"}{errS} to SP
        </small>
      </div>
      <div className="row">
        <label>SP</label>
        <input type="number" step={step} min={u.to(q, loop.RL)} max={u.to(q, loop.RH)} value={sp} onFocus={() => setEditingSp(true)}
          onChange={(e) => setSp(e.target.value)} onBlur={applySp}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
        <button className="small" onClick={() => bump(-step)}>-{step}</button>
        <button className="small" onClick={() => bump(step)}>+{step}</button>
        <span className="note">{u.unit(q)}</span>
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
      <div className="gains">P {loop.P.toFixed(1)} % · I {loop.I === "OFF" ? "OFF" : `${loop.I} s`} · D {loop.D === "OFF" ? "OFF" : `${loop.D} s`} · FL {loop.FL === "OFF" ? "OFF" : `${loop.FL} s`} · {loop.DR} · range {rng(loop.RL)}..{rng(loop.RH)} {u.unit(q)}</div>
    </div>
  );
}
