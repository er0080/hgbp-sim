import { fmt } from "../api";
import type { Snapshot } from "../types";
import { useUnits } from "../units";

const VS = 13;   // valve half size

/** Valve symbol.  ``vertical``: flow runs vertically through the valve (hourglass);
 *  otherwise horizontally (bowtie).  Labels sit beside the valve on ``side``.  The
 *  body is outline only and the pipes end at its faces (see ``pipeCut``). */
function Valve({ x, y, open, label, vertical = false, side = "right" }:
  { x: number; y: number; open: number; label: string; vertical?: boolean; side?: "left" | "right" }) {
  const s = VS;
  const pts = vertical
    ? `${x - s},${y - s} ${x + s},${y - s} ${x - s},${y + s} ${x + s},${y + s}`
    : `${x - s},${y - s} ${x - s},${y + s} ${x + s},${y - s} ${x + s},${y + s}`;
  const dir = side === "right" ? 1 : -1;
  // stem + actuator: sideways for vertical flow, upward for horizontal flow
  const stem = vertical
    ? <><line x1={x} y1={y} x2={x + 22 * dir} y2={y} stroke="#cfd8dc" strokeWidth={2} />
        <rect x={dir > 0 ? x + 22 : x - 30} y={y - 9} width={8} height={18} fill="#cfd8dc" /></>
    : <><line x1={x} y1={y} x2={x} y2={y - 22} stroke="#cfd8dc" strokeWidth={2} />
        <rect x={x - 9} y={y - 30} width={18} height={8} fill="#cfd8dc" /></>;
  const lx = vertical ? x + 38 * dir : x;
  const anchor = vertical ? (dir > 0 ? "start" : "end") : "middle";
  return (
    <g>
      <polygon points={pts} className="valve" />
      {stem}
      <text x={lx} y={vertical ? y + 4 : y + 32} textAnchor={anchor} className="lbl">{label}</text>
      <text x={lx} y={vertical ? y + 18 : y + 46} textAnchor={anchor}>{fmt(open * 100, 0)} %</text>
    </g>
  );
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));


export default function Schematic({ snap }: { snap: Snapshot }) {
  const m = snap.meas, t = snap.true;
  const u = useUnits();
  const running = snap.compressor.running;
  const rec = clamp01(t.rec_level), flood = clamp01(t.cond_flood);
  const xq: number[] = t.x_q ?? [], Tq: number[] = t.T_q ?? [], Tg: number[] = t.T_g ?? [];
  const xc: number[] = t.x_c ?? [], Tc: number[] = t.T_c ?? [], Twc: number[] = t.T_wc ?? [];
  // refrigerant cell: quality while two-phase, temperature as vapor or liquid
  const cellText = (x: number, T: number) => (x >= 0 && x < 1 ? `x ${fmt(x, 2)}` : `${u.fmt("T", T, 0)}°`);
  // geometry
  const CX = 110, CY = 250, CR = 38;                    // compressor
  const TOP = 40;                                       // discharge / header level
  const HX = { w: 110, h: 175 };                        // brazed plate heat exchanger outline (both units)
  const CD = { x: 750, y: 72, ...HX };                  // condenser (S2, S3 up; drawn mirrored)
  const RX = CD.x + 18, WCX = CD.x + CD.w - 18;         // refrigerant (S3 -> S4) and water (S1 -> S2) port columns
  const CB = CD.y + CD.h;
  const REC = { x: RX - 30, y: 280, w: 60, h: 140 };    // receiver
  const MX = { x: 440, y: 205, ...HX };                 // mixing exchanger (S2, S3 up)
  const GX = MX.x + 18, QX = MX.x + MX.w - 18;          // gas (S1 -> S2) and quench (S3 -> S4) port columns
  const MB = MX.y + MX.h;
  const BX = 370;                                       // bypass line x
  const LX = 660;                                       // liquid line riser x
  const TX = 280, TY = 440;                             // tee
  const OUT_Y = MX.y - 25;                              // gas outlet run to the tee
  const WX = 975;                                       // water pipe x
  const W_IN = CB + 28, W_OUT = CD.y - 20;              // water supply run to S1, return run from S2
  const cellH = MX.h / Math.max(1, xq.length), cellHc = CD.h / Math.max(1, xc.length);
  const liquidY = REC.y + REC.h - 14;                   // receiver outlet level
  const PTX = 720;                                      // liquid pressure transmitter x
  const HOP = 7;                                        // radius of the gas outlet's hop over the bypass line
  const PROBE = CR + 28;                                // probe distance from the compressor centre
  // compressor symbol: two chords converging on the discharge (top), suction enters at the wide end
  const cp = (deg: number) => `${CX + CR * Math.cos(deg * Math.PI / 180)},${CY - CR * Math.sin(deg * Math.PI / 180)}`;
  const compChords = `M ${cp(114)} L ${cp(214)} M ${cp(66)} L ${cp(-34)}`;
  const compStroke = running ? "var(--ok)" : "#90a4ae";
  const V = {                                           // valve positions (symbols and pipe cuts)
    v1: { x: CX, y: 125, vertical: true }, v2: { x: BX, y: 125, vertical: true },
    v3: { x: 610, y: MX.y - 45, vertical: false }, v4: { x: WX, y: 370, vertical: true },
  };
  return (
    <div className="card">
      <h2>Stand schematic</h2>
      <svg viewBox="0 0 1000 500" className="schem" style={{ width: "100%", height: "auto" }}>
        <defs>
          {/* pipes end at the valve faces */}
          <mask id="pipeCut" maskUnits="userSpaceOnUse" x={0} y={0} width={1000} height={500}>
            <rect width={1000} height={500} fill="white" />
            {Object.values(V).map((v, i) => v.vertical
              ? <rect key={i} x={v.x - 6} y={v.y - VS} width={12} height={2 * VS} fill="black" />
              : <rect key={i} x={v.x - VS} y={v.y - 6} width={2 * VS} height={12} fill="black" />)}
          </mask>
          <clipPath id="recClip"><rect x={REC.x + 2} y={REC.y + 2} width={REC.w - 4} height={REC.h - 4} rx={REC.w / 2 - 2} ry={16} /></clipPath>
        </defs>

        <g mask="url(#pipeCut)">
          {/* discharge: compressor -> valve 1 -> hot gas header -> condenser S3 (top) */}
          <polyline points={`${CX},${CY - CR} ${CX},${TOP} ${RX},${TOP} ${RX},${CD.y}`} className="pipe hot" />
          {/* cooling water: valve 4 -> condenser S1 (bottom) -> S2 (top) -> return */}
          <polyline points={`${WX},470 ${WX},${W_IN} ${WCX},${W_IN} ${WCX},${CB}`} className="pipe water" />
          <polyline points={`${WCX},${CD.y} ${WCX},${W_OUT} ${WX},${W_OUT} ${WX},14`} className="pipe water" />
          {/* condensate drain S4 -> receiver; liquid line: receiver -> valve 3 -> mixing exchanger S3 (top) */}
          <line x1={RX} y1={CB} x2={RX} y2={REC.y + 8} className="pipe liq" />
          <polyline points={`${REC.x + 10},${liquidY} ${LX},${liquidY} ${LX},${MX.y - 45} ${QX},${MX.y - 45} ${QX},${MX.y}`} className="pipe liq" />
          {/* bypass: header -> valve 2 -> mixing exchanger S1 (bottom) */}
          <polyline points={`${BX},${TOP} ${BX},${MB + 30} ${GX},${MB + 30} ${GX},${MB}`} className="pipe hot" />
          {/* mixing exchanger outlets -> tee (the gas outlet hops over the bypass line) -> compressor */}
          <path d={`M ${GX},${MX.y} V ${OUT_Y} H ${BX + HOP} A ${HOP} ${HOP} 0 0 0 ${BX - HOP},${OUT_Y} H ${TX} V ${TY}`} className="pipe suc" />
          <polyline points={`${QX},${MB} ${QX},${TY} ${TX},${TY}`} className="pipe suc" />
          <polyline points={`${TX},${TY} ${CX},${TY} ${CX},${CY + CR}`} className="pipe suc" />
        </g>

        <Valve {...V.v1} open={m.u1} label="1 discharge press." />
        {/* discharge probe, as close to the compressor port as the suction probe */}
        <circle cx={CX} cy={CY - PROBE} r={6} fill="#263238" stroke="#cfd8dc" strokeWidth={1.5} />
        <text x={CX + 12} y={CY - PROBE - 4}>{u.fmtU("T", m.T_d)} · {u.fmtU("P", m.P_d)}</text>
        <text x={CX + 12} y={CY - PROBE + 10} className="lbl">discharge probe · sat {u.fmtU("T", m.Tsat_d)}</text>
        <text x={CX + 12} y={CY - PROBE - 22} className="lbl">discharge line ΔP {u.fmtU("dP", t.dP_dis)}</text>
        <text x={560} y={TOP - 14} textAnchor="middle" className="lbl">hot gas header · line ΔP {u.fmtU("dP", t.dP_hdr)}</text>
        <text x={560} y={TOP + 22} textAnchor="middle">{u.fmtU("P", t.P_h)} · sat {u.fmtU("T", t.Tsat_h)}</text>

        {/* condenser: refrigerant column (left, S3 top -> S4 bottom), water column (right, S1 bottom -> S2 top);
            liquid backs up from a full receiver into the bottom cells */}
        <rect x={CD.x} y={CD.y} width={CD.w} height={CD.h} rx={6} className="vessel" />
        {xc.map((x, j) => {
          const y = CD.y + j * cellHc;
          return (
            <g key={j}>
              <rect x={CD.x + 2} y={y + 1} width={CD.w / 2 - 3} height={cellHc - 2} className="liquid" style={{ opacity: 0.1 + 0.75 * clamp01(1 - x) }} />
              <text x={CD.x + CD.w * 0.25} y={y + cellHc / 2 + 4} textAnchor="middle">{cellText(x, Tc[j])}</text>
              <text x={CD.x + CD.w * 0.75} y={y + cellHc / 2 + 4} textAnchor="middle" className="lbl">{u.fmt("T", Twc[j], 0)}°</text>
            </g>
          );
        })}
        <line x1={CD.x + CD.w / 2} y1={CD.y + 4} x2={CD.x + CD.w / 2} y2={CB - 4} stroke="#90a4ae" strokeWidth={1} />
        <text x={RX - 10} y={CD.y - 6} textAnchor="end" className="lbl">S3</text>
        <text x={RX - 8} y={CB + 12} textAnchor="end" className="lbl">S4</text>
        <text x={WCX + 10} y={CD.y - 6} className="lbl">S2</text>
        <text x={WCX + 8} y={CB + 13} className="lbl">S1</text>
        <text x={CD.x + CD.w + 8} y={CD.y + 30} className="lbl">condenser</text>
        <text x={CD.x + CD.w + 8} y={CD.y + 48}>{u.fmtU("heat", t.Q_w)}</text>
        <text x={CD.x + CD.w + 8} y={CD.y + 70} className="lbl">refrigerant (S3 → S4)</text>
        <text x={CD.x + CD.w + 8} y={CD.y + 84}>SC {u.fmtU("dT", m.SC)}</text>
        <text x={CD.x + CD.w + 8} y={CD.y + 98} style={{ fill: flood > 0.05 ? "var(--warn)" : undefined }}>flooded {fmt(flood * 100, 0)} %</text>
        <text x={CD.x + CD.w + 8} y={CD.y + 112}>ΔP {u.fmtU("dP", t.dP_cr)}</text>
        <text x={CD.x + CD.w + 8} y={CD.y + 134} className="lbl">water (S1 → S2)</text>
        <text x={CD.x + CD.w + 8} y={CD.y + 148}>{u.fmtU("flow_w", t.mdot_w)}</text>
        <text x={CD.x + CD.w + 8} y={CD.y + 162}>ΔP {u.fmtU("dP", t.dP_cw)}</text>
        <Valve {...V.v4} open={m.u4} label="4 cooling water" side="left" />
        <text x={WX - 8} y={462} textAnchor="end" className="lbl">water in</text>
        <text x={WX - 8} y={476} textAnchor="end" className="lbl">{u.fmtU("T", m.T_wi)}</text>
        <text x={WX - 8} y={W_OUT - 18} textAnchor="end" className="lbl">water out</text>
        <text x={WX - 8} y={W_OUT - 5} textAnchor="end" className="lbl">{u.fmtU("T", m.T_wo)}</text>

        {/* receiver (dished heads); the level fills the vessel outline */}
        <rect x={REC.x} y={REC.y} width={REC.w} height={REC.h} rx={REC.w / 2} ry={18} className="vessel" />
        <rect x={REC.x} y={REC.y + REC.h - 2 - (REC.h - 4) * rec} width={REC.w} height={(REC.h - 4) * rec + 2} className="liquid" clipPath="url(#recClip)" />
        <text x={REC.x + REC.w + 8} y={REC.y + 18} className="lbl">receiver</text>
        <text x={REC.x + REC.w + 8} y={REC.y + 34}>level {fmt(rec * 100, 0)} %</text>
        <text x={REC.x + REC.w + 8} y={REC.y + 48} className="lbl">{t.ll_fill < 1 ? "vapor in liquid line" : "liquid seal"}</text>

        <Valve {...V.v3} open={m.u3} label="3 suction temp." />
        <text x={LX + 8} y={liquidY - 36} className="lbl">quench</text>
        <text x={LX + 8} y={liquidY - 22}>{u.fmtU("mdot", t.mdot_3)}</text>
        <text x={LX + 8} y={liquidY - 8} className="lbl">{u.fmtU("T", m.T_co)}</text>
        {/* liquid (intermediate) pressure transmitter, after the receiver */}
        <circle cx={PTX} cy={liquidY} r={6} fill="#263238" stroke="#cfd8dc" strokeWidth={1.5} />
        <text x={PTX} y={liquidY + 22} textAnchor="middle">{u.fmtU("P", m.P_i)}</text>
        <text x={PTX} y={liquidY + 36} textAnchor="middle" className="lbl">liquid pressure · sat {u.fmtU("T", m.Tsat_i)}</text>
        <Valve {...V.v2} open={m.u2} label="2 suction press. (HGBP)" />
        <text x={BX - 10} y={300} textAnchor="end" className="lbl">bypass</text>
        <text x={BX - 10} y={314} textAnchor="end">{u.fmtU("mdot", t.mdot_2)}</text>
        <text x={BX - 10} y={328} textAnchor="end" className="lbl">line ΔP {u.fmtU("dP", t.dP_bp)}</text>

        <circle cx={TX} cy={TY} r={5} fill="#66bb6a" />

        {/* mixing exchanger: gas column (left, S1 bottom -> S2 top), quench column (right, S3 top -> S4 bottom) */}
        <rect x={MX.x} y={MX.y} width={MX.w} height={MX.h} rx={6} className="vessel" />
        {xq.map((x, j) => {
          const wet = clamp01(1 - x);
          const y = MX.y + j * cellH;
          return (
            <g key={j}>
              <rect x={MX.x + MX.w / 2 + 1} y={y + 1} width={MX.w / 2 - 3} height={cellH - 2} className="liquid" style={{ opacity: 0.1 + 0.75 * wet }} />
              <text x={MX.x + MX.w * 0.75} y={y + cellH / 2 + 4} textAnchor="middle">{cellText(x, Tq[j])}</text>
              <text x={MX.x + MX.w * 0.25} y={y + cellH / 2 + 4} textAnchor="middle" className="lbl">{u.fmt("T", Tg[j], 0)}°</text>
            </g>
          );
        })}
        <line x1={MX.x + MX.w / 2} y1={MX.y + 4} x2={MX.x + MX.w / 2} y2={MB - 4} stroke="#90a4ae" strokeWidth={1} />
        <text x={GX - 8} y={MB + 13} textAnchor="end" className="lbl">S1</text>
        <text x={GX - 12} y={MX.y - 6} textAnchor="end" className="lbl">S2</text>
        <text x={QX + 10} y={MX.y - 6} className="lbl">S3</text>
        <text x={QX + 8} y={MB + 12} className="lbl">S4</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 30} className="lbl">mixing</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 44} className="lbl">exchanger</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 62}>{u.fmtU("heat", t.Q_mx)}</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 84} className="lbl">gas (S1 → S2)</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 98}>out {u.fmtU("T", t.T_go)}</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 112}>ΔP {u.fmtU("dP", t.dP_mg)}</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 132} className="lbl">quench (S3 → S4)</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 146} style={{ fill: t.x_qo < 1 ? "var(--bad)" : undefined }}>
          out {t.x_qo < 1 ? `x ${fmt(t.x_qo, 2)}` : u.fmtU("T", t.T_qo)}</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 160}>ΔP {u.fmtU("dP", t.dP_mq)}</text>

        {/* suction line: probe at the compressor port */}
        <circle cx={CX} cy={CY + PROBE} r={6} fill="#263238" stroke="#cfd8dc" strokeWidth={1.5} />
        <text x={CX + 12} y={CY + PROBE - 4}>{u.fmtU("T", m.T_s)} · SH {u.fmtU("dT", m.SH)}</text>
        <text x={CX + 12} y={CY + PROBE + 10} className="lbl">suction probe</text>
        <text x={(TX + CX) / 2} y={TY - 26} textAnchor="middle">{u.fmtU("P", m.P_s)} · sat {u.fmtU("T", m.Tsat_s)}</text>
        <text x={(TX + CX) / 2} y={TY - 10} textAnchor="middle" className="lbl">suction {u.fmtU("mdot", m.mdot)}</text>
        <text x={(TX + CX) / 2} y={TY - 42} textAnchor="middle" className="lbl">suction line ΔP {u.fmtU("dP", t.dP_suc)}</text>
        {t.y_liq > 0.0005 && (
          <text x={(TX + CX) / 2} y={TY + 20} textAnchor="middle" style={{ fill: "var(--bad)" }}>
            liquid at compressor {fmt(t.y_liq * 100, 1)} %</text>
        )}
        {/* compressor */}
        <circle cx={CX} cy={CY} r={CR} className="comp-body" style={{ stroke: compStroke }} />
        <path d={compChords} className="comp-body" style={{ fill: "none", stroke: compStroke }} />
        <text x={CX - CR - 8} y={CY + 4} textAnchor="end" className="lbl">compressor</text>
        <text x={CX + CR + 8} y={CY - 12}>{fmt(m.N, 0)} rpm</text>
        <text x={CX + CR + 8} y={CY + 3}>{u.fmtU("power", m.W)}</text>
        <text x={CX + CR + 8} y={CY + 18} className="lbl">shell {u.fmtU("T", t.T_sh, 0)}</text>
        <text x={20} y={492} className="lbl">charge {u.fmtU("mass", snap.charge.kg)} (nominal {u.fmtU("mass", snap.charge.nominal_kg)}) · {snap.fluid} · ambient {u.fmtU("T", m.T_amb)}</text>
      </svg>
    </div>
  );
}
