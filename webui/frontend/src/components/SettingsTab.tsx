import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { ParamMeta, ParamsView, Snapshot, Time, Tuning } from "../types";
import { loopQty, useQtyInput, useUnits, type Qty } from "../units";

const GROUP_ORDER = ["Refrigerant", "Charge", "Compressor", "Piping", "Condenser", "Receiver", "Mixing exchanger",
  "Pressure drops", "Valves", "Pipe walls", "Sensors", "Safety limits", "Baseline control"];
const LOOPS = ["dpv", "spv", "stv", "water"];

type TuneEdit = { P: string; I: string; D: string; FL: string };
const timeStr = (t: Time) => (t === "OFF" ? "" : String(t));
const timeVal = (s: string): Time => (s.trim() === "" || s.trim().toUpperCase() === "OFF" ? "OFF" : Number(s));
const tuneEdit = (t: Tuning): TuneEdit => ({ P: t.P.toFixed(1), I: timeStr(t.I), D: timeStr(t.D), FL: timeStr(t.FL) });

function fmtVal(v: any) {
  if (typeof v !== "number") return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e4 || a < 1e-3 ? v.toExponential(3) : String(Number(v.toPrecision(5)));
}

export default function SettingsTab({ snap }: { snap: Snapshot }) {
  const [view, setView] = useState<ParamsView | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [msg, setMsg] = useState("");
  const [tune, setTune] = useState<Record<string, TuneEdit>>({});
  const u = useUnits();
  const [T_amb, setTamb, T_ambBase] = useQtyInput("T", snap.meas.T_amb);
  const [T_wi, setTwi, T_wiBase] = useQtyInput("T", snap.meas.T_wi);
  const [rate, setRate, rateBase] = useQtyInput("rate", snap.charge.rate_kg_s * 1000);
  const rng = (q: Qty, v: number) => String(Number(u.to(q, v).toFixed(2)));
  const reload = () => api<ParamsView>("/api/params").then((v) => { setView(v); setEdits({}); });
  useEffect(() => { reload(); }, []);
  useEffect(() => {
    const t: Record<string, TuneEdit> = {};
    for (const k of LOOPS) t[k] = tuneEdit(snap.loops[k]);
    setTune(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snap.fluid]);

  const groups = useMemo(() => {
    if (!view) return [] as [string, ParamMeta[]][];
    const by: Record<string, ParamMeta[]> = {};
    for (const m of view.meta) (by[m.group] ??= []).push(m);
    // groups the backend adds later than this list still show, at the end
    const order = [...GROUP_ORDER.filter((g) => by[g]), ...Object.keys(by).filter((g) => !GROUP_ORDER.includes(g))];
    return order.map((g) => [g, by[g]] as [string, ParamMeta[]]);
  }, [view]);

  const apply = async (andInit: boolean) => {
    if (!view) return;
    const values: Record<string, any> = {};
    for (const [k, s] of Object.entries(edits)) {
      const m = view.meta.find((x) => x.name === k)!;
      values[k] = k === "charge" && s.trim() === "" ? null
        : m.kind === "float" ? Number(s) : m.kind === "bool" ? s === "true" : s;
    }
    try {
      const r = await api("/api/params", { values });
      let text = `applied: ${r.applied.join(", ") || "none"}`;
      if (r.deferred.length) text += ` · deferred to next start: ${r.deferred.join(", ")}`;
      if (andInit) { await api("/api/init", { mode: "cold" }); text += " · cold start done"; }
      setMsg(text);
      await reload();
    } catch (e: any) { setMsg("error: " + e.message); }
  };
  const resetGroup = (ms: ParamMeta[]) => {
    const e = { ...edits };
    for (const m of ms) e[m.name] = m.default === null ? "" : String(m.default);
    setEdits(e);
  };
  const applyTune = async (k: string) => {
    const t = tune[k];
    try {
      const r = await api(`/api/loop/${k}`, { P: Number(t.P), I: timeVal(t.I), D: timeVal(t.D), FL: timeVal(t.FL) });
      setTune({ ...tune, [k]: tuneEdit(r) });            // show the values as the controller rounded them
      setMsg(`${snap.loops[k].label} tuning applied`);
    } catch (e: any) { setMsg("error: " + e.message); }
  };
  const defaultTunes = () => {
    if (!view) return;
    const t: Record<string, TuneEdit> = {};
    for (const k of LOOPS) t[k] = tuneEdit(view.defaults.loops[k]);
    setTune(t);
    setMsg("default tuning loaded into the fields; press Apply on each loop to use it");
  };
  const applySim = async () => {
    const body: Record<string, number> = {};
    for (const [k, v] of [["T_amb", T_ambBase()], ["T_wi", T_wiBase()], ["charge_rate_g_s", rateBase()]] as const)
      if (Number.isFinite(v)) body[k] = k === "charge_rate_g_s" ? Math.min(250, Math.max(1, v)) : v;
    await api("/api/sim", body);
    setMsg("conditions and charge rate applied");
  };
  if (!view) return <p className="note">loading parameters...</p>;
  const dirty = Object.keys(edits).length;
  return (
    <div>
      <div className="settings">
        <div className="card">
          <h2>Simulation and conditions</h2>
          <table><tbody>
            <tr><td className="n">speed factor</td><td className="d">wall-clock speed-up</td><td className="i">
              <select value={snap.speed_factor} onChange={(e) => api("/api/sim", { speed_factor: Number(e.target.value) })}>
                {[0.5, 1, 2, 5, 10, 20, 50, 100].map((v) => <option key={v} value={v}>{v}x</option>)}</select></td></tr>
            <tr><td className="n">sensor noise</td><td className="d">measurement noise on/off</td><td className="i">
              <input type="checkbox" checked={snap.noise} onChange={(e) => api("/api/sim", { noise: e.target.checked })} /></td></tr>
            <tr><td className="n">short-cycle timers</td><td className="d">compressor minimum off time 60 s and minimum run time 120 s (live)</td><td className="i">
              <input type="checkbox" checked={snap.compressor.short_cycle_timers} onChange={(e) => api("/api/sim", { short_cycle_timers: e.target.checked })} /></td></tr>
            <tr><td className="n">control interval</td><td className="d">PID execution and display update period (applied immediately)</td><td className="i">
              <select value={snap.dt_ctrl} onChange={(e) => api("/api/sim", { dt_ctrl: Number(e.target.value) })}>
                {[0.1, 0.2, 0.25, 0.5, 1, 2].map((v) => <option key={v} value={v}>{v} s</option>)}</select></td></tr>
            <tr><td className="n">T_amb</td><td className="d">ambient temperature (live)</td><td className="i"><input type="number" step={u.step("T")} value={T_amb} onChange={(e) => setTamb(e.target.value)} /> {u.unit("T")}</td></tr>
            <tr><td className="n">T_wi</td><td className="d">cooling water inlet temperature (live)</td><td className="i"><input type="number" step={u.step("T")} value={T_wi} onChange={(e) => setTwi(e.target.value)} /> {u.unit("T")}</td></tr>
            <tr><td className="n">charge rate</td><td className="d">rate at which refrigerant is added or recovered ({rng("rate", 1)}-{rng("rate", 250)})</td><td className="i"><input type="number" min={u.to("rate", 1)} max={u.to("rate", 250)} step={u.step("rate")} value={rate} onChange={(e) => setRate(e.target.value)} /> {u.unit("rate")}</td></tr>
          </tbody></table>
          <div style={{ marginTop: 8 }}><button className="primary" onClick={applySim}>Apply conditions</button></div>
        </div>
        <div className="card">
          <h2>PID tuning, UT35A units (applied live) <button className="small" style={{ float: "right" }} onClick={defaultTunes}>defaults</button></h2>
          <p className="note">As on the Yokogawa UT35A: P = proportional band in % of the PV input range (0.1-999.9), I = integral time and D = derivative time in seconds (1-6000, empty = OFF), FL = PV input filter, a first-order lag in seconds on the PV ahead of both the display and the PID (1-120, empty = OFF). Action (DIR/RVS), input range and output limits come from the defaults file.</p>
          {LOOPS.map((k) => tune[k] && (
            <div className="gainrow" key={k}>
              <div className="gainname"><b>{snap.loops[k].label}</b>
                <span>{snap.loops[k].valve_label} · {snap.loops[k].DR} · range {rng(loopQty(snap.loops[k].unit), snap.loops[k].RL)}..{rng(loopQty(snap.loops[k].unit), snap.loops[k].RH)} {u.unit(loopQty(snap.loops[k].unit))} · output {snap.loops[k].OL}..{snap.loops[k].OH} %</span></div>
              <div className="gaininputs">
                <label>P<input type="number" min={0.1} max={999.9} step={0.1} value={tune[k].P}
                  onChange={(e) => setTune({ ...tune, [k]: { ...tune[k], P: e.target.value } })} />%</label>
                {(["I", "D"] as const).map((g) => (
                  <label key={g}>{g}<input type="number" min={1} max={6000} step={1} placeholder="OFF" value={tune[k][g]}
                    onChange={(e) => setTune({ ...tune, [k]: { ...tune[k], [g]: e.target.value } })} />s</label>
                ))}
                <label title="PV input filter: first-order lag on the PV, ahead of the display and the PID">FL<input type="number" min={1} max={120} step={1} placeholder="OFF" value={tune[k].FL}
                  onChange={(e) => setTune({ ...tune, [k]: { ...tune[k], FL: e.target.value } })} />s</label>
                <button className="small" onClick={() => applyTune(k)}>Apply</button>
              </div>
            </div>
          ))}
        </div>
        {groups.map(([g, ms]) => (
          <div className="card" key={g}>
            <h2>{g} <button className="small" style={{ float: "right" }} onClick={() => resetGroup(ms)}>defaults</button></h2>
            {u.sys !== "metric" && g === GROUP_ORDER[0] && <p className="note">Model parameters are shown in the simulator's SI units whatever the display units.</p>}
            {g === "Charge" && <p className="note">Current charge {view.values.charge.toFixed(3)} kg ({u.fmtU("mass", view.values.charge)}), nominal for these volumes {view.nominal_charge.toFixed(3)} kg (receiver 40 % full). The value set here applies at the next cold/warm start; use the Add / Recover buttons on the operator panel to charge or recover at the receiver while running. Leave empty for nominal.</p>}
            {g === "Piping" && view.derived && (
              <details className="note" style={{ marginBottom: 8 }}>
                <summary>Refrigerant volumes derived from piping and components: suction {(view.derived.V_s * 1e3).toFixed(1)} L,
                  discharge {(view.derived.V_d * 1e3).toFixed(1)} L, intermediate {(view.derived.V_i * 1e3).toFixed(1)} L</summary>
                <table><tbody>
                  {view.derived.volumes.map((r) => (
                    <tr key={r.item}><td className="n">{r.section}</td><td className="d">{r.item}</td><td className="u">{r.L.toFixed(2)} L</td></tr>
                  ))}
                </tbody></table>
              </details>
            )}
            <table><tbody>
              {ms.map((m) => {
                const cur = m.name in edits ? edits[m.name] : (view.values[m.name] === null ? "" : fmtVal(view.values[m.name]));
                const pending = view.pending.includes(m.name);
                return (
                  <tr key={m.name}>
                    <td className="n">{m.name}{m.requires_init && <span title="applied at next cold/warm start" style={{ color: "var(--warn)" }}> *</span>}</td>
                    <td className="d">{m.description}
                      {(m.min != null || m.max != null) && <span className="note"> ({m.min} .. {m.max})</span>}</td>
                    <td className="u">{m.unit}</td>
                    <td className="i">
                      {m.name === "fluid" ? (
                        <select value={cur} className={pending ? "pending" : ""} onChange={(e) => setEdits({ ...edits, fluid: e.target.value })}>
                          {view.fluids.map((f) => <option key={f} value={f}>{f}</option>)}
                        </select>
                      ) : m.kind === "str" ? (
                        <select value={cur} onChange={(e) => setEdits({ ...edits, [m.name]: e.target.value })}>
                          {["linear", "eqpct", "quick"].map((f) => <option key={f} value={f}>{f}</option>)}
                        </select>
                      ) : (
                        <input type="number" step="any" value={cur} className={m.name in edits || pending ? "pending" : ""}
                          min={m.min ?? undefined} max={m.max ?? undefined}
                          title={m.min != null || m.max != null ? `allowed range ${m.min} .. ${m.max}` : undefined}
                          placeholder={m.name === "charge" ? "nominal" : String(m.default)}
                          onChange={(e) => setEdits({ ...edits, [m.name]: e.target.value })} />
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody></table>
          </div>
        ))}
      </div>
      <div className="sticky-actions">
        <button className="primary" disabled={!dirty} onClick={() => apply(false)}>Apply {dirty ? `(${dirty})` : ""}</button>
        <button className="warn" disabled={!dirty} onClick={() => apply(true)}>Apply and cold start</button>
        <button disabled={!dirty} onClick={() => setEdits({})}>Discard</button>
        <span className="note">* structural parameters (refrigerant, volumes, charge) take effect at the next cold/warm start · defaults: {view.defaults_path ?? "built-in (HGBP_DEFAULTS not set)"} · {msg}</span>
      </div>
    </div>
  );
}
