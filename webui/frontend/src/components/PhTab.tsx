import { useEffect, useRef, useState } from "react";
import { api, fmt } from "../api";
import type { PhChart, Snapshot } from "../types";
import { useUnits, type Units } from "../units";

/** Live pressure-enthalpy diagram: the stand's state points along every flow path over
 *  the refrigerant's saturation dome, lines of constant quality and isotherms.  Legs
 *  carry the schematic's pipe colors; the exchangers blend from inlet to outlet color.
 *  The view fits the cycle, shows the whole dome, or zooms to a box dragged on the chart;
 *  the backend draws the background for each view (its own pressure grid, isotherms at a
 *  spacing to suit it). */

const COL = { hot: "#ef5350", liq: "#42a5f5", suc: "#66bb6a" };
type Pt = [number, number, number, number, number];           // P bar, h kJ/kg, T °C, x, SC / SH K
type Pts = Record<string, Pt>;
const cells = (p: string, order: number[]) => order.map((j) => `${p}${j}`);
const N_CELL = [0, 1, 2, 3, 4];

/** Legs in flow order: points, start / end color, label of a valve on the leg. */
const LEGS: { key: string; name: string; pts: string[]; from: keyof typeof COL; to: keyof typeof COL; valve?: string; arrow?: boolean }[] = [
  { key: "comp", name: "compression", pts: ["suc", "dis"], from: "suc", to: "hot", arrow: true },
  { key: "dis", name: "discharge line", pts: ["dis", "v1i"], from: "hot", to: "hot" },
  { key: "v1", name: "valve 1, header", pts: ["v1i", "hdr"], from: "hot", to: "hot", valve: "V1", arrow: true },
  { key: "hdr", name: "header to condenser", pts: ["hdr", "cin"], from: "hot", to: "hot" },
  { key: "cond", name: "condenser", pts: ["cin", ...cells("c", N_CELL)], from: "hot", to: "liq", arrow: true },
  { key: "drain", name: "drain, receiver", pts: ["c4", "rec"], from: "liq", to: "liq" },
  { key: "liq", name: "liquid line", pts: ["rec", "v3i"], from: "liq", to: "liq" },
  { key: "v3", name: "valve 3, quench line", pts: ["v3i", "v3o", "qin"], from: "liq", to: "liq", valve: "V3", arrow: true },
  { key: "quench", name: "mixing exchanger, quench side", pts: ["qin", ...cells("q", N_CELL), "qout"], from: "liq", to: "suc", arrow: true },
  { key: "qo", name: "quench outlet", pts: ["qout", "tee"], from: "suc", to: "suc" },
  { key: "v2", name: "valve 2, bypass line", pts: ["hdr", "v2o", "gin"], from: "hot", to: "hot", valve: "V2", arrow: true },
  { key: "gas", name: "mixing exchanger, gas side", pts: ["gin", ...cells("g", [4, 3, 2, 1, 0]), "gout"], from: "hot", to: "suc", arrow: true },
  { key: "go", name: "gas outlet", pts: ["gout", "tee"], from: "suc", to: "suc" },
  { key: "sucl", name: "suction line", pts: ["tee", "suc"], from: "suc", to: "suc", arrow: true },
];

/** Numbered state points (the exchanger cells are listed under their exchanger). */
const KEY_POINTS: [string, string][] = [
  ["suc", "compressor suction"], ["dis", "discharge (probe)"], ["v1i", "valve 1 inlet"], ["hdr", "hot gas header"],
  ["cin", "condenser in (S3)"], ["rec", "receiver liquid"], ["v3i", "valve 3 inlet"], ["v3o", "valve 3 outlet"],
  ["qin", "quench in (S3)"], ["qout", "quench out (S4)"], ["v2o", "valve 2 outlet"], ["gin", "bypass gas in (S1)"],
  ["gout", "bypass gas out (S2)"], ["tee", "tee (mixed)"],
];
const CELL_GROUPS: [string, string, string[]][] = [
  ["c", "condenser cells (top → bottom)", cells("c", N_CELL)],
  ["q", "quench cells (top → bottom)", cells("q", N_CELL)],
  ["g", "gas cells (bottom → top)", cells("g", [4, 3, 2, 1, 0])],
];
const NUM: Record<string, number> = Object.fromEntries(KEY_POINTS.map(([k], i) => [k, i + 1]));
const NAME: Record<string, string> = Object.fromEntries([
  ...KEY_POINTS,
  ...CELL_GROUPS.flatMap(([, label, ks]) => ks.map((k, i) => [k, `${label.split(" (")[0].replace(" cells", "")} cell ${i + 1}`])),
]);

// plot geometry (viewBox units)
const W = 1000, H = 640, M = { l: 66, r: 92, t: 16, b: 46 };          // setpoint labels in the right margin
const PW = W - M.l - M.r, PH = H - M.t - M.b;
const TRAIL_N = 150;                                  // snapshots kept for the trail (about 15 s at 10/s)
const TRAIL_PTS = ["suc", "dis", "hdr", "rec", "qout", "gout"];

interface Range { h0: number; h1: number; P0: number; P1: number }   // base units: kJ/kg, bar

function niceStep(span: number, n: number) {
  const raw = span / n, p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  return (f < 1.5 ? 1 : f < 2.25 ? 2 : f < 3.5 ? 2.5 : f < 7.5 ? 5 : 10) * p;
}

/** Pressure ticks: 1-2-5 and finer decades, linear ticks over a narrow (zoomed) range.
 *  Returns the ticks and the decimals to show. */
function pressureTicks(lo: number, hi: number): [number[], (v: number) => number] {
  if (hi / lo < 2.5) {
    const st = niceStep(hi - lo, 6), t: number[] = [];
    for (let v = Math.ceil(lo / st) * st; v <= hi; v += st) t.push(v);
    return [t, () => Math.max(0, -Math.floor(Math.log10(st) + 1e-9))];
  }
  return [logTicks(lo, hi), (v) => (v >= 10 ? 0 : v < 1 ? 2 : 1)];
}

function logTicks(lo: number, hi: number) {
  let t: number[] = [];
  for (const mult of [[1, 2, 5], [1, 2, 3, 5], [1, 1.5, 2, 3, 4, 5, 6, 8], [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9]]) {
    t = [];
    for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++)
      for (const m of mult) { const v = m * 10 ** e; if (v >= lo && v <= hi) t.push(v); }
    if (t.length >= 5) return t;
  }
  return t.length >= 2 ? t : [lo, hi];
}

function fullRange(c: PhChart): Range {
  const hl = Math.min(...c.h_l), hv = Math.max(...c.h_v);
  return { h0: hl - 10, h1: hv + 0.45 * (hv - hl), P0: Math.max(c.P_min, 0.3), P1: c.P_crit * 1.3 };
}

function cycleRange(pts: Pts): Range {
  const P = Object.values(pts).map((p) => p[0]), h = Object.values(pts).map((p) => p[1]);
  const h0 = Math.min(...h), h1 = Math.max(...h), span = Math.max(h1 - h0, 50);
  // at least a factor 4 in pressure (a stopped, equalized stand), centred on the points
  const p0 = Math.min(...P) / 1.6, p1 = Math.max(...P) * 1.45, widen = Math.sqrt(Math.max(4 / (p1 / p0), 1));
  return { h0: h0 - 0.12 * span - 5, h1: h1 + 0.12 * span + 5, P0: Math.max(p0 / widen, 0.05), P1: p1 * widen };
}

/** Quality lines across the view: the coarsest of 0.1, 0.05, 0.02, 0.01 that still gives
 *  about a dozen lines over the view's width of the dome. */
function qualityStep(c: PhChart, R: Range) {
  const Pm = Math.sqrt(R.P0 * R.P1);
  const hfg = Math.max(interpLog(Pm, c.P, c.h_v) - interpLog(Pm, c.P, c.h_l), 1);
  return [0.01, 0.02, 0.05, 0.1].find((s) => (R.h1 - R.h0) / (hfg * s) <= 12) ?? 0.1;
}

/** The point label: SC / SH / quality by phase. */
function stateText(u: Units, p: Pt) {
  const [, , , x, dT] = p;
  if (x < 0) return `SC ${u.fmtU("dT", dT)}`;
  if (x > 1) return `SH ${u.fmtU("dT", dT)}`;
  return `x ${fmt(x, 2)}`;
}

export default function PhTab({ snap }: { snap: Snapshot }) {
  const u = useUnits();
  const [base, setBase] = useState<PhChart | null>(null);       // whole dome (its extent; drawn until a view arrives)
  const [view, setView] = useState<PhChart | null>(null);       // background of the present view
  const [err, setErr] = useState<string | null>(null);
  const [auto, setAuto] = useState<"cycle" | "full">("cycle");
  const [zooms, setZooms] = useState<Range[]>([]);              // box zooms, innermost last
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [showIso, setShowIso] = useState(true);
  const [showQ, setShowQ] = useState(true);
  const [showCells, setShowCells] = useState(true);
  const [showTrail, setShowTrail] = useState(false);
  const [showSp, setShowSp] = useState(true);
  const [ref, setRef] = useState<{ t: number; pts: Pts } | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const range = useRef<Range | null>(null);
  const trail = useRef<Pts[]>([]);
  const pts = snap.ph;

  useEffect(() => {
    let live = true;
    setView(null);
    setZooms([]);
    api<PhChart>("/api/ph_chart")
      .then((c) => { if (live) { setBase(c); setErr(null); } })
      .catch((e) => live && setErr(String(e)));
    return () => { live = false; };
  }, [snap.fluid]);

  // axis range: a box zoom, the whole dome, or the cycle with a margin (kept while the
  // cycle fits, so that the axes do not jitter)
  let R: Range | null = null;
  if (zooms.length) R = zooms[zooms.length - 1];
  else if (base && auto === "full") R = fullRange(base);
  else if (pts) {
    const want = cycleRange(pts), cur = range.current;
    const fits = cur && want.h0 >= cur.h0 && want.h1 <= cur.h1 && want.P0 >= cur.P0 && want.P1 <= cur.P1
      && (cur.h1 - cur.h0) < 1.6 * (want.h1 - want.h0) && Math.log(cur.P1 / cur.P0) < 1.6 * Math.log(want.P1 / want.P0);
    R = fits && cur ? cur : want;
  }
  range.current = !zooms.length && auto === "cycle" ? R : null;

  // the background of this view, with isotherms at round values of the display unit
  const viewKey = R ? [R.h0, R.h1, R.P0, R.P1].map((v) => v.toPrecision(6)).join(",") : "";
  useEffect(() => {
    if (!viewKey) return;
    let live = true;
    const timer = window.setTimeout(() => {
      api<PhChart>(`/api/ph_chart?view=${viewKey}&unit=${u.sys === "metric" ? "C" : "F"}&n=20`)
        .then((c) => { if (live) setView(c); })
        .catch(() => { /* keep the previous background */ });
    }, 120);
    return () => { live = false; clearTimeout(timer); };
  }, [viewKey, u.sys, snap.fluid]);

  // Escape steps out of a box zoom
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setZooms((z) => z.slice(0, -1)); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // trail of the main points (newest last)
  if (showTrail && pts && trail.current[trail.current.length - 1] !== pts) {
    trail.current.push(pts);
    if (trail.current.length > TRAIL_N) trail.current.splice(0, trail.current.length - TRAIL_N);
  }
  useEffect(() => { if (!showTrail) trail.current = []; }, [showTrail]);

  if (err) return <div className="card"><p className="err">P-h chart unavailable: {err}</p></div>;
  if (!base || !pts || !R) return <div className="card"><p className="note">Loading the P-h chart...</p></div>;
  const chart = view && view.fluid === base.fluid ? view : base;
  const zoomed = zooms.length > 0;

  const hx0 = u.to("h", R.h0), hx1 = u.to("h", R.h1);
  const ly0 = Math.log10(u.to("P", R.P0)), ly1 = Math.log10(u.to("P", R.P1));
  const X = (h: number) => M.l + ((u.to("h", h) - hx0) / (hx1 - hx0)) * PW;
  const Y = (P: number) => M.t + PH - ((Math.log10(u.to("P", Math.max(P, 1e-6))) - ly0) / (ly1 - ly0)) * PH;
  const line = (P: number[], h: number[]) => P.map((p, i) => `${X(h[i]).toFixed(1)},${Y(p).toFixed(1)}`).join(" ");
  const inPlot = (x: number, y: number) => x >= M.l && x <= W - M.r && y >= M.t && y <= H - M.b;

  // ticks in display units
  const hStep = niceStep(hx1 - hx0, 10);
  const hDec = Math.max(0, -Math.floor(Math.log10(hStep) + 1e-9));
  const hTicks: number[] = [];
  for (let v = Math.ceil(hx0 / hStep) * hStep; v <= hx1; v += hStep) hTicks.push(v);
  const [pTicks, pDec] = pressureTicks(10 ** ly0, 10 ** ly1);
  // viewBox position -> base units (the box zoom and the cursor readout)
  const hAt = (x: number) => u.from("h", hx0 + ((x - M.l) / PW) * (hx1 - hx0));
  const PAt = (y: number) => u.from("P", 10 ** (ly0 + ((M.t + PH - y) / PH) * (ly1 - ly0)));
  const qStep = qualityStep(chart, R);
  const qualities: number[] = [];
  for (let k = 1; k * qStep < 1 - 1e-9; k++) qualities.push(Math.round(k * qStep * 100) / 100);
  const isoDec = chart.iso_step !== undefined && chart.iso_step !== null && chart.iso_step < 1 ? 1 : 0;
  const xOfDisp = (v: number) => M.l + ((v - hx0) / (hx1 - hx0)) * PW;
  const yOfDisp = (v: number) => M.t + PH - ((Math.log10(v) - ly0) / (ly1 - ly0)) * PH;

  // dome: two-phase region outline (bubble line up, estimated closure, dew line down)
  const ext = chart.dome_ext;
  const domeP = [...chart.P, ...(ext ? ext.P : []), ...(ext ? [...ext.P].reverse() : []), ...[...chart.P].reverse()];
  const domeH = [...chart.h_l, ...(ext ? ext.h_l : []), ...(ext ? [...ext.h_v].reverse() : []), ...[...chart.h_v].reverse()];

  // a point's position, and whether the point exists (reverse flows can leave gaps)
  const at = (k: string, src: Pts = pts) => (src[k] ? { x: X(src[k][1]), y: Y(src[k][0]) } : null);
  const legPath = (pk: string[], src: Pts = pts) =>
    pk.map((k) => at(k, src)).filter(Boolean).map((p) => `${p!.x.toFixed(1)},${p!.y.toFixed(1)}`).join(" ");

  // numbered labels: points closer than 16 px share one label, placed at the first free
  // corner (above left, above right, below left, below right)
  const groups: { x: number; y: number; nums: number[] }[] = [];
  for (const [k] of KEY_POINTS) {
    const p = at(k);
    if (!p) continue;
    const g = groups.find((g) => Math.hypot(g.x - p.x, g.y - p.y) < 16);
    if (g) g.nums.push(NUM[k]); else groups.push({ x: p.x, y: p.y, nums: [NUM[k]] });
  }
  const boxes: { x0: number; x1: number; y0: number; y1: number }[] = [];
  const labels = groups.map((g) => {
    const text = g.nums.join(","), w = 7 * text.length, h = 11;
    let pick = { x: g.x - 7, y: g.y - 8, anchor: "end" as "end" | "start" };
    for (const [dx, dy, anchor] of [[-7, -8, "end"], [7, -8, "start"], [-7, 17, "end"], [7, 17, "start"]] as const) {
      const x0 = anchor === "end" ? g.x + dx - w : g.x + dx, b = { x0, x1: x0 + w, y0: g.y + dy - h, y1: g.y + dy };
      if (!boxes.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0)) {
        pick = { x: g.x + dx, y: g.y + dy, anchor };
        boxes.push(b);
        break;
      }
    }
    return { text, ...pick, cell: false };
  });
  // zoomed in, the cells get labels too where there is room (C, Q, G and their number in flow order)
  if (zoomed && showCells) {
    for (const [g, , ks] of CELL_GROUPS) ks.forEach((k, i) => {
      const p = at(k);
      if (!p || !inPlot(p.x, p.y)) return;
      const text = `${g.toUpperCase()}${i + 1}`, w = 7 * text.length, h = 10;
      for (const [dx, dy] of [[5, -5], [5, 14], [-5 - w, -5], [-5 - w, 14]]) {
        const b = { x0: p.x + dx, x1: p.x + dx + w, y0: p.y + dy - h, y1: p.y + dy };
        if (!boxes.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0)) {
          boxes.push(b);
          labels.push({ text, x: b.x0, y: b.y1, anchor: "start", cell: true });
          break;
        }
      }
    });
  }

  // mouse: viewBox position, box zoom by dragging, hover on the nearest point
  const toView = (e: React.MouseEvent) => {
    const m = svgRef.current?.getScreenCTM();
    return m ? new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse()) : null;
  };
  const clampX = (x: number) => Math.min(Math.max(x, M.l), W - M.r);
  const clampY = (y: number) => Math.min(Math.max(y, M.t), H - M.b);
  const onDown = (e: React.MouseEvent) => {
    const q = toView(e);
    if (e.button !== 0 || !q || !inPlot(q.x, q.y)) return;
    e.preventDefault();
    setDrag({ x0: q.x, y0: q.y, x1: q.x, y1: q.y });
  };
  const onUp = () => {
    if (!drag) return;
    const xa = Math.min(drag.x0, drag.x1), xb = Math.max(drag.x0, drag.x1);
    const ya = Math.min(drag.y0, drag.y1), yb = Math.max(drag.y0, drag.y1);
    setDrag(null);
    if (xb - xa < 8 || yb - ya < 8) return;                  // a click, not a box
    const z = { h0: hAt(xa), h1: hAt(xb), P0: PAt(yb), P1: PAt(ya) };
    // no finer than 0.2 kJ/kg and 0.2 % in pressure
    if (z.h1 - z.h0 < 0.2) { const c = (z.h0 + z.h1) / 2; z.h0 = c - 0.1; z.h1 = c + 0.1; }
    if (z.P1 / z.P0 < 1.002) { const c = Math.sqrt(z.P0 * z.P1); z.P0 = c / 1.001; z.P1 = c * 1.001; }
    setZooms((zs) => [...zs, z]);
  };
  const onMove = (e: React.MouseEvent) => {
    const q = toView(e);
    if (!q) return;
    setCursor(inPlot(q.x, q.y) ? { x: q.x, y: q.y } : null);
    if (drag) {
      setDrag({ ...drag, x1: clampX(q.x), y1: clampY(q.y) });
      setHover(null);
      return;
    }
    let best: string | null = null, bd = 16;
    for (const k of Object.keys(pts)) {
      if (!showCells && /^[cqg]\d$/.test(k)) continue;
      const p = at(k)!;
      const d = Math.hypot(p.x - q.x, p.y - q.y);
      if (d < bd) { bd = d; best = k; }
    }
    setHover(best);
  };

  const sp = [
    { P: snap.loops.dpv.sp, label: "discharge SP", c: COL.hot },
    { P: snap.loops.water.sp, label: "intermediate SP", c: COL.liq },
    { P: snap.loops.spv.sp, label: "suction SP", c: COL.suc },
  ];
  const isoLabel = (iso: PhChart["isotherms"][number]) => {
    // at the dew line crossing (two-phase -> vapor), else where the line enters the plot
    let i = iso.P.findIndex((p, j) => iso.h[j] >= interpLog(p, chart.P, chart.h_v) - 0.5);
    if (i < 0 || !inPlot(X(iso.h[i]), Y(iso.P[i]))) i = iso.P.findIndex((p, j) => inPlot(X(iso.h[j]), Y(p)));
    return i < 0 ? null : { x: X(iso.h[i]), y: Y(iso.P[i]) };
  };

  const hp = hover ? pts[hover] : null;
  const m = snap.true;
  const d = (a: string, b: string) => (pts[a] && pts[b] ? pts[b][1] - pts[a][1] : NaN);
  const hU = u.unit("h");
  const stats: [string, string][] = [
    ["pressure ratio", fmt(pts.dis[0] / Math.max(pts.suc[0], 1e-3), 2)],
    ["compression Δh", `${fmt(u.diff("h", d("suc", "dis")), 1)} ${hU}`],
    ["condenser Δh (S3 → receiver)", `${fmt(u.diff("h", d("cin", "rec")), 1)} ${hU}`],
    ["bypass gas Δh (S1 → S2)", `${fmt(u.diff("h", d("gin", "gout")), 1)} ${hU}`],
    ["quench Δh (S3 → S4)", `${fmt(u.diff("h", d("qin", "qout")), 1)} ${hU}`],
    ["valve 1 flow", u.fmtU("mdot", m.mdot_1)],
    ["valve 2 (bypass) flow", u.fmtU("mdot", m.mdot_2)],
    ["valve 3 (quench) flow", u.fmtU("mdot", m.mdot_3)],
  ];

  return (
    <div className="ph-layout">
      <div className="card ph-chart">
        <h2>P-h diagram · {chart.fluid}
          {cursor && <span className="ph-cursor">cursor P {u.fmtU("P", PAt(cursor.y), u.dec("P") + 1)} · h {u.fmtU("h", hAt(cursor.x), u.dec("h") + 1)}</span>}
        </h2>
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className={`ph${drag ? " dragging" : ""}`}
          style={{ width: "100%", height: "auto" }} onMouseDown={onDown} onMouseUp={onUp}
          onMouseMove={onMove} onMouseLeave={() => { setHover(null); setCursor(null); setDrag(null); }}
          onDoubleClick={() => setZooms((z) => z.slice(0, -1))}>
          <defs>
            <clipPath id="phClip"><rect x={M.l} y={M.t} width={PW} height={PH} /></clipPath>
            {LEGS.filter((l) => l.from !== l.to).map((l) => {
              const a = at(l.pts[0]), b = at(l.pts[l.pts.length - 1]);
              if (!a || !b) return null;
              // along the leg's chord (a degenerate chord falls back to the end color)
              const same = Math.hypot(a.x - b.x, a.y - b.y) < 1;
              return (
                <linearGradient key={l.key} id={`ph-${l.key}`} gradientUnits="userSpaceOnUse"
                  x1={a.x} y1={a.y} x2={same ? a.x + 1 : b.x} y2={same ? a.y : b.y}>
                  <stop offset="0" stopColor={COL[l.from]} /><stop offset="1" stopColor={COL[l.to]} />
                </linearGradient>
              );
            })}
          </defs>

          {/* grid and axes */}
          <rect x={M.l} y={M.t} width={PW} height={PH} className="ph-frame" />
          {hTicks.map((v) => (
            <g key={`h${v}`}>
              <line x1={xOfDisp(v)} x2={xOfDisp(v)} y1={M.t} y2={H - M.b} className="ph-grid" />
              <text x={xOfDisp(v)} y={H - M.b + 16} textAnchor="middle">{fmt(v, hDec)}</text>
            </g>
          ))}
          {pTicks.map((v) => (
            <g key={`p${v}`}>
              <line x1={M.l} x2={W - M.r} y1={yOfDisp(v)} y2={yOfDisp(v)} className="ph-grid" />
              <text x={M.l - 6} y={yOfDisp(v) + 4} textAnchor="end">{fmt(v, pDec(v))}</text>
            </g>
          ))}
          <text x={M.l + PW / 2} y={H - 8} textAnchor="middle" className="lbl">specific enthalpy h [{hU}]</text>
          <text x={16} y={M.t + PH / 2} textAnchor="middle" className="lbl" transform={`rotate(-90 16 ${M.t + PH / 2})`}>
            pressure P [{u.unit("P")}] (log scale)</text>

          <g clipPath="url(#phClip)">
            {/* two-phase region */}
            <polygon points={line(domeP, domeH)} className="ph-dome" />
            {/* isotherms */}
            {showIso && chart.isotherms.map((iso) => (
              <polyline key={`t${iso.T}`} points={line(iso.P, iso.h)} className="ph-iso" />
            ))}
            {/* lines of constant quality */}
            {showQ && qualities.map((q) => (
              <polyline key={`x${q}`} points={line(chart.P, chart.h_l.map((hl, i) => hl + q * (chart.h_v[i] - hl)))} className="ph-qual" />
            ))}
            {/* saturation lines: x = 0 (bubble) and x = 1 (dew); dashed: estimated to the critical point */}
            <polyline points={line(chart.P, chart.h_l)} className="ph-sat liq" />
            <polyline points={line(chart.P, chart.h_v)} className="ph-sat vap" />
            {ext && <>
              <polyline points={line(ext.P, ext.h_l)} className="ph-sat liq ext" />
              <polyline points={line(ext.P, ext.h_v)} className="ph-sat vap ext" />
              {chart.h_crit !== undefined && <circle cx={X(chart.h_crit)} cy={Y(chart.P_crit)} r={4} className="ph-crit" />}
            </>}
            {/* setpoint pressures */}
            {showSp && sp.map((s) => (
              <line key={s.label} x1={M.l} x2={W - M.r} y1={Y(s.P)} y2={Y(s.P)} stroke={s.c} strokeOpacity={0.55}
                strokeDasharray="6 5" strokeWidth={1} />
            ))}
            {/* reference cycle */}
            {ref && LEGS.map((l) => (
              <polyline key={`r${l.key}`} points={legPath(l.pts, ref.pts)} className="ph-ref" />
            ))}
            {/* trail of the main points */}
            {showTrail && trail.current.map((tp, i) => TRAIL_PTS.map((k) => {
              const p = at(k, tp);
              return p ? <circle key={`${i}${k}`} cx={p.x} cy={p.y} r={1.8} fill="#e6edf3"
                fillOpacity={0.05 + 0.45 * (i / trail.current.length)} /> : null;
            }))}
            {/* the live cycle */}
            {LEGS.map((l) => (
              <polyline key={l.key} points={legPath(l.pts)} className="ph-leg"
                stroke={l.from === l.to ? COL[l.from] : `url(#ph-${l.key})`}
                strokeDasharray={l.valve ? "7 4" : undefined} />
            ))}
            {/* flow direction: an arrow halfway along each main leg */}
            {LEGS.filter((l) => l.arrow).map((l) => {
              const ps = l.pts.map((k) => at(k)).filter(Boolean) as { x: number; y: number }[];
              let len = 0;
              for (let i = 1; i < ps.length; i++) len += Math.hypot(ps[i].x - ps[i - 1].x, ps[i].y - ps[i - 1].y);
              if (len < 30) return null;
              let s = len / 2;
              for (let i = 1; i < ps.length; i++) {
                const seg = Math.hypot(ps[i].x - ps[i - 1].x, ps[i].y - ps[i - 1].y);
                if (s <= seg && seg > 0.5) {
                  const f = s / seg, x = ps[i - 1].x + f * (ps[i].x - ps[i - 1].x), y = ps[i - 1].y + f * (ps[i].y - ps[i - 1].y);
                  const ang = Math.atan2(ps[i].y - ps[i - 1].y, ps[i].x - ps[i - 1].x) * 180 / Math.PI;
                  const c = mix(COL[l.from], COL[l.to], 0.5);
                  return <path key={`a${l.key}`} d="M -6 -5 L 6 0 L -6 5 z" fill={c}
                    transform={`translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${ang.toFixed(1)})`} />;
                }
                s -= seg;
              }
              return null;
            })}
            {/* valve labels at the middle of the throttling */}
            {LEGS.filter((l) => l.valve).map((l) => {
              const a = at(l.pts[0]), b = at(l.pts[1]);
              if (!a || !b) return null;
              return <text key={`v${l.key}`} x={(a.x + b.x) / 2 + 8} y={(a.y + b.y) / 2} className="ph-valve">{l.valve}</text>;
            })}
            {/* cells and state points */}
            {showCells && CELL_GROUPS.flatMap(([, , ks]) => ks).map((k) => {
              const p = at(k);
              return p ? <circle key={k} cx={p.x} cy={p.y} r={2.6} className="ph-cell" /> : null;
            })}
            {KEY_POINTS.map(([k]) => {
              const p = at(k);
              return p ? <circle key={k} cx={p.x} cy={p.y} r={4} className="ph-pt" /> : null;
            })}
            {labels.map((l) => (
              <text key={l.text} x={l.x} y={l.y} textAnchor={l.anchor} className={l.cell ? "ph-celllbl" : "ph-num"}>{l.text}</text>
            ))}
          </g>

          {/* labels outside the clip: setpoints, isotherms, qualities, saturation lines */}
          {showSp && (() => {
            // two-line labels at their lines, pushed apart where setpoints lie close together
            const ls = sp.map((s) => ({ ...s, y: Y(s.P) })).filter((s) => s.y > M.t && s.y < H - M.b)
              .sort((a, b) => a.y - b.y);
            for (let i = 1; i < ls.length; i++) ls[i].y = Math.max(ls[i].y, ls[i - 1].y + 28);
            return ls.map((s) => (
              <g key={`sl${s.label}`}>
                <text x={W - M.r + 6} y={s.y - 2} className="ph-splbl" style={{ fill: s.c }}>{s.label}</text>
                <text x={W - M.r + 6} y={s.y + 11} className="ph-splbl" style={{ fill: s.c }}>{u.fmtU("P", s.P)}</text>
              </g>));
          })()}
          {showIso && (() => {
            const placed: { x: number; y: number }[] = [];
            return chart.isotherms.map((iso) => {
              const p = isoLabel(iso);
              if (!p || placed.some((q) => Math.abs(q.x - p.x) < 40 && Math.abs(q.y - p.y) < 13)) return null;
              placed.push(p);
              return <text key={`tl${iso.T}`} x={p.x + 4} y={p.y - 3} className="ph-isolbl">
                {u.fmt("T", iso.T, isoDec)}{u.unit("T")}</text>;
            });
          })()}
          {showQ && (() => {
            // where the line leaves the plot at the bottom, where there is room
            let last = -1e9;
            return qualities.map((q) => {
              const hq = chart.h_l.map((hl, i) => hl + q * (chart.h_v[i] - hl));
              const i = chart.P.findIndex((p, j) => inPlot(X(hq[j]), Y(p)));
              if (i < 0 || Y(chart.P[i]) < H - M.b - 12 || X(hq[i]) - last < 44) return null;
              last = X(hq[i]);
              return <text key={`ql${q}`} x={last} y={H - M.b - 4} textAnchor="middle" className="ph-qlbl">
                x {fmt(q, qStep < 0.1 ? 2 : 1)}</text>;
            });
          })()}
          {(() => {
            // inside the dome, 40 % up the plot (or wherever the line is in view)
            const label = (hs: number[], dx: number, anchor: "start" | "end", text: string) => {
              const vis = chart.P.map((p, j) => j).filter((j) => inPlot(X(hs[j]), Y(chart.P[j])));
              if (!vis.length) return null;
              const yt = M.t + 0.6 * PH;
              const j = vis.reduce((a, b) => (Math.abs(Y(chart.P[b]) - yt) < Math.abs(Y(chart.P[a]) - yt) ? b : a));
              return <text x={X(hs[j]) + dx} y={Y(chart.P[j])} textAnchor={anchor} className="ph-satlbl">{text}</text>;
            };
            return <>{label(chart.h_l, 8, "start", "x = 0")}{label(chart.h_v, -8, "end", "x = 1")}</>;
          })()}

          {/* box being dragged */}
          {drag && <rect x={Math.min(drag.x0, drag.x1)} y={Math.min(drag.y0, drag.y1)} width={Math.abs(drag.x1 - drag.x0)}
            height={Math.abs(drag.y1 - drag.y0)} className="ph-zoombox" />}

          {/* hover readout */}
          {hp && hover && !drag && (() => {
            const p = at(hover)!;
            const lines = [`${NUM[hover] ? NUM[hover] + " · " : ""}${NAME[hover] ?? hover}`,
              `P ${u.fmtU("P", hp[0])}   T ${u.fmtU("T", hp[2])}`, `h ${u.fmtU("h", hp[1])}   ${stateText(u, hp)}`];
            const bw = 230, bh = 58, left = p.x + bw + 16 > W - M.r;
            const bx = left ? p.x - bw - 12 : p.x + 12, by = Math.max(M.t, Math.min(p.y - bh - 8, H - M.b - bh));
            return (
              <g pointerEvents="none">
                <circle cx={p.x} cy={p.y} r={7} className="ph-hover" />
                <rect x={bx} y={by} width={bw} height={bh} rx={6} className="ph-tip" />
                {lines.map((s, i) => <text key={i} x={bx + 10} y={by + 18 + i * 16} className={i === 0 ? "ph-tiptitle" : undefined}>{s}</text>)}
              </g>
            );
          })()}
        </svg>
        <div className="ph-legend">
          <span><i style={{ background: COL.hot }} />discharge and hot gas</span>
          <span><i style={{ background: COL.liq }} />liquid</span>
          <span><i style={{ background: COL.suc }} />suction gas</span>
          <span><i style={{ background: `linear-gradient(90deg, ${COL.hot}, ${COL.liq})` }} />exchangers and compression: inlet → outlet color</span>
          <span><i className="dash" />valve (throttling)</span>
          {ref && <span><i className="ref" />reference at {fmt(ref.t, 0)} s</span>}
          <span className="ph-hint">drag a box to zoom · Esc or double click: back</span>
        </div>
      </div>

      <div className="ph-side">
        <div className="card">
          <h2>Display</h2>
          <div className="ph-ctl">
            <div className="mode">
              <button className={!zoomed && auto === "cycle" ? "active" : ""} onClick={() => { setAuto("cycle"); setZooms([]); }}>Fit cycle</button>
              <button className={!zoomed && auto === "full" ? "active" : ""} onClick={() => { setAuto("full"); setZooms([]); }}>Full dome</button>
              <button className={zoomed ? "active" : ""} disabled={!zoomed} title="drag a box on the chart to zoom into it">
                Zoom{zooms.length > 1 ? ` ×${zooms.length}` : ""}</button>
            </div>
            <button disabled={!zoomed} onClick={() => setZooms((z) => z.slice(0, -1))}
              title="back to the previous view (also Esc or a double click)">Back</button>
            <label><input type="checkbox" checked={showIso} onChange={(e) => setShowIso(e.target.checked)} /> isotherms</label>
            <label><input type="checkbox" checked={showQ} onChange={(e) => setShowQ(e.target.checked)} /> quality lines</label>
            <label><input type="checkbox" checked={showCells} onChange={(e) => setShowCells(e.target.checked)} /> exchanger cells</label>
            <label><input type="checkbox" checked={showSp} onChange={(e) => setShowSp(e.target.checked)} /> setpoints</label>
            <label title="recent positions of the suction, discharge, header, receiver and exchanger outlet points">
              <input type="checkbox" checked={showTrail} onChange={(e) => setShowTrail(e.target.checked)} /> trail</label>
            <div>
              <button onClick={() => setRef({ t: snap.t, pts })} title="keep the present cycle as a dashed reference">Hold reference</button>
              {ref && <button onClick={() => setRef(null)} style={{ marginLeft: 6 }}>Clear</button>}
            </div>
          </div>
        </div>
        <div className="card">
          <h2>Cycle</h2>
          <table className="ph-table">
            <tbody>{stats.map(([k, v]) => <tr key={k}><td>{k}</td><td className="v">{v}</td></tr>)}</tbody>
          </table>
        </div>
        <div className="card">
          <h2>State points</h2>
          <table className="ph-table">
            <thead><tr><th>#</th><th>point</th><th>P <small>{u.unit("P")}</small></th><th>T <small>{u.unit("T")}</small></th>
              <th>h <small>{hU}</small></th><th>state</th></tr></thead>
            <tbody>
              {KEY_POINTS.map(([k, name]) => pts[k] && (
                <tr key={k} className={hover === k ? "hl" : ""} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)}>
                  <td>{NUM[k]}</td><td className="n">{name}</td><td className="v">{u.fmt("P", pts[k][0])}</td>
                  <td className="v">{u.fmt("T", pts[k][2])}</td><td className="v">{u.fmt("h", pts[k][1])}</td>
                  <td className="v">{stateText(u, pts[k])}</td>
                </tr>
              ))}
              {showCells && CELL_GROUPS.map(([g, label, ks]) => [
                <tr key={g} className="grp"><td /><td colSpan={5}>{label}</td></tr>,
                ...ks.map((k, i) => pts[k] && (
                  <tr key={k} className={`cell ${hover === k ? "hl" : ""}`} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)}>
                    <td /><td className="n">{i + 1}</td><td className="v">{u.fmt("P", pts[k][0])}</td>
                    <td className="v">{u.fmt("T", pts[k][2])}</td><td className="v">{u.fmt("h", pts[k][1])}</td>
                    <td className="v">{stateText(u, pts[k])}</td>
                  </tr>
                )),
              ])}
            </tbody>
          </table>
          <p className="note">Model values (no sensor noise or lag). Exchanger cells sit at their centres' pressures;
            the {u.sys === "english" ? "Btu/lb values keep the property tables' reference state" : "enthalpy reference is the property tables'"}.</p>
        </div>
      </div>
    </div>
  );
}

/** Blend of two #rrggbb colors (``f`` = 0: ``a``). */
function mix(a: string, b: string, f: number) {
  const c = (s: string, i: number) => parseInt(s.slice(1 + 2 * i, 3 + 2 * i), 16);
  return "#" + [0, 1, 2].map((i) => Math.round(c(a, i) + f * (c(b, i) - c(a, i))).toString(16).padStart(2, "0")).join("");
}

/** Linear interpolation of ``y`` at ``x`` over log-spaced ``xs`` (ascending). */
function interpLog(x: number, xs: number[], ys: number[]) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  const f = (Math.log(x) - Math.log(xs[lo])) / (Math.log(xs[hi]) - Math.log(xs[lo]));
  return ys[lo] + f * (ys[hi] - ys[lo]);
}
