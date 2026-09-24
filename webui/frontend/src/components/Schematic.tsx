import { fmt } from "../api";
import type { Snapshot } from "../types";

/** Valve symbol.  ``vertical``: flow runs vertically through the valve (hourglass);
 *  otherwise horizontally (bowtie).  Labels sit beside the valve on ``side``. */
function Valve({ x, y, open, label, vertical = false, side = "right" }:
  { x: number; y: number; open: number; label: string; vertical?: boolean; side?: "left" | "right" }) {
  const s = 13;
  const pts = vertical
    ? `${x - s},${y - s} ${x + s},${y - s} ${x - s},${y + s} ${x + s},${y + s}`
    : `${x - s},${y - s} ${x - s},${y + s} ${x + s},${y - s} ${x + s},${y + s}`;
  const col = `rgba(79,195,247,${0.15 + 0.85 * Math.min(1, Math.max(0, open))})`;
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
      <polygon points={pts} className="valve" style={{ fill: col }} />
      {stem}
      <text x={lx} y={vertical ? y + 4 : y + 32} textAnchor={anchor} className="lbl">{label}</text>
      <text x={lx} y={vertical ? y + 18 : y + 46} textAnchor={anchor}>{fmt(open * 100, 0)} %</text>
    </g>
  );
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

export default function Schematic({ snap }: { snap: Snapshot }) {
  const m = snap.meas, t = snap.true;
  const running = snap.compressor.running;
  const rec = clamp01(t.rec_level), flood = clamp01(t.cond_flood);
  const xq: number[] = t.x_q ?? [], Tq: number[] = t.T_q ?? [], Tg: number[] = t.T_g ?? [];
  const nCells = Math.max(1, xq.length);
  // geometry
  const CX = 110, CY = 250, CR = 38;                    // compressor
  const TOP = 60;                                       // discharge / header level
  const COND = { x: 690, y: 95, w: 170, h: 80 };        // condenser
  const CIN = COND.x + COND.w / 2;                      // condenser inlet / drain x
  const REC = { x: CIN - 30, y: 230, w: 60, h: 160 };   // receiver
  const MX = { x: 440, y: 205, w: 110, h: 175 };        // mixing exchanger (S2, S3 up)
  const GX = MX.x + 18, QX = MX.x + MX.w - 18;          // gas (S1 -> S2) and quench (S3 -> S4) port columns
  const MB = MX.y + MX.h;
  const BX = 370;                                       // bypass line x
  const LX = 660;                                       // liquid line riser x
  const TX = 300, TY = 440;                             // tee
  const OUT_Y = MX.y - 25;                              // gas outlet run to the tee
  const WX = 930;                                       // water pipe x
  const cellH = MX.h / nCells;
  const liquidY = REC.y + REC.h - 12;                   // receiver outlet (dip tube) level
  const dipY = REC.y + REC.h - 2 - (REC.h - 4) * 0.04;
  return (
    <div className="card">
      <h2>Stand schematic</h2>
      <svg viewBox="0 0 1000 500" className="schem" style={{ width: "100%", height: "auto" }}>
        {/* discharge: compressor -> valve 1 -> hot gas header -> condenser */}
        <polyline points={`${CX},${CY - CR} ${CX},${TOP} ${CIN},${TOP} ${CIN},${COND.y}`} className="pipe hot" />
        <Valve x={CX} y={140} open={m.u1} label="1 discharge press." vertical />
        <text x={CX + 20} y={185} className="lbl">discharge</text>
        <text x={CX + 20} y={200}>{fmt(m.P_d, 2)} bar</text>
        <text x={CX + 20} y={214}>{fmt(m.T_d, 0)} °C</text>
        <text x={CX + 20} y={228} className="lbl">sat {fmt(m.Tsat_d, 1)} °C</text>
        <text x={560} y={TOP - 14} textAnchor="middle" className="lbl">hot gas header · intermediate pressure</text>
        <text x={560} y={TOP + 22} textAnchor="middle">{fmt(m.P_i, 2)} bar · sat {fmt(m.Tsat_i, 1)} °C</text>

        {/* condenser: liquid backs up from a full receiver */}
        <rect x={COND.x} y={COND.y} width={COND.w} height={COND.h} rx={6} className="vessel" />
        <rect x={COND.x + 2} y={COND.y + COND.h - 2 - (COND.h - 4) * flood} width={COND.w - 4} height={(COND.h - 4) * flood} className="liquid" />
        <text x={CIN} y={COND.y + 18} textAnchor="middle" className="lbl">condenser (BPHE)</text>
        <text x={CIN} y={COND.y + 35} textAnchor="middle">flooded {fmt(flood * 100, 0)} % · SC {fmt(m.SC, 1)} K</text>
        <text x={CIN} y={COND.y + 50} textAnchor="middle">wall {fmt(t.T_cw, 0)} °C</text>
        <text x={CIN} y={COND.y + 65} textAnchor="middle" className="lbl">{fmt(t.Q_w / 1000, 1)} kW to water</text>
        <polyline points={`${WX},470 ${WX},${COND.y + 62} ${COND.x + COND.w},${COND.y + 62}`} className="pipe water" />
        <polyline points={`${COND.x + COND.w},${COND.y + 18} ${WX},${COND.y + 18} ${WX},${TOP + 20}`} className="pipe water" />
        <Valve x={WX} y={320} open={m.u4} label="4 cooling water" vertical side="left" />
        <text x={WX + 12} y={462} className="lbl">water in</text>
        <text x={WX + 12} y={476} className="lbl">{fmt(m.T_wi, 1)} °C</text>
        <text x={WX} y={TOP - 4} textAnchor="middle" className="lbl">water out</text>
        <text x={WX} y={TOP + 11} textAnchor="middle" className="lbl">{fmt(m.T_wo, 1)} °C · {fmt(t.mdot_w, 1)} kg/min</text>

        {/* condensate drain -> receiver (sight glass level, dip tube outlet) */}
        <line x1={CIN} y1={COND.y + COND.h} x2={CIN} y2={REC.y} className="pipe liq" />
        <rect x={REC.x} y={REC.y} width={REC.w} height={REC.h} rx={22} className="vessel" />
        <rect x={REC.x + 2} y={REC.y + REC.h - 2 - (REC.h - 4) * rec} width={REC.w - 4} height={(REC.h - 4) * rec} rx={rec > 0.9 ? 20 : 4} className="liquid" />
        <line x1={REC.x + 14} y1={REC.y + 14} x2={REC.x + 14} y2={dipY} stroke="#cfd8dc" strokeWidth={2} />
        <text x={REC.x + REC.w + 8} y={REC.y + 18} className="lbl">receiver</text>
        <text x={REC.x + REC.w + 8} y={REC.y + 34}>level {fmt(rec * 100, 0)} %</text>
        <text x={REC.x + REC.w + 8} y={REC.y + 48} className="lbl">{t.ll_fill < 1 ? "vapor in liquid line" : "liquid seal"}</text>

        {/* liquid line: receiver -> valve 3 -> mixing exchanger S3 (top) */}
        <polyline points={`${REC.x},${liquidY} ${LX},${liquidY} ${LX},${MX.y - 45} ${QX},${MX.y - 45} ${QX},${MX.y}`} className="pipe liq" />
        <Valve x={610} y={MX.y - 45} open={m.u3} label="3 suction temp." />
        <text x={LX + 8} y={liquidY - 36} className="lbl">quench</text>
        <text x={LX + 8} y={liquidY - 22}>{fmt(t.mdot_3, 1)} g/s</text>
        <text x={LX + 8} y={liquidY - 8} className="lbl">{fmt(m.T_co, 1)} °C</text>

        {/* bypass: header -> valve 2 -> mixing exchanger S1 (bottom) */}
        <polyline points={`${BX},${TOP} ${BX},${MB + 22} ${GX},${MB + 22} ${GX},${MB}`} className="pipe hot" />
        <Valve x={BX} y={130} open={m.u2} label="2 suction press. (HGBP)" vertical />
        <text x={BX - 10} y={300} textAnchor="end" className="lbl">bypass</text>
        <text x={BX - 10} y={314} textAnchor="end">{fmt(t.mdot_2, 1)} g/s</text>

        {/* mixing exchanger outlets -> tee (the gas outlet run hops over the bypass line) */}
        <polyline points={`${GX},${MX.y} ${GX},${OUT_Y} ${TX},${OUT_Y} ${TX},${TY}`} className="pipe suc" />
        <line x1={BX} y1={OUT_Y - 8} x2={BX} y2={OUT_Y + 8} stroke="var(--panel)" strokeWidth={8} />
        <line x1={BX} y1={OUT_Y - 9} x2={BX} y2={OUT_Y + 9} className="pipe hot" strokeDasharray="3 3" />
        <polyline points={`${QX},${MB} ${QX},${TY} ${TX},${TY}`} className="pipe suc" />
        <circle cx={TX} cy={TY} r={5} fill="#66bb6a" />
        <text x={TX + 8} y={TY - 8} className="lbl">tee</text>

        {/* mixing exchanger: gas column (left, S1 bottom -> S2 top), quench column (right, S3 top -> S4 bottom) */}
        <rect x={MX.x} y={MX.y} width={MX.w} height={MX.h} rx={6} className="vessel" />
        {xq.map((x, j) => {
          const wet = clamp01(1 - x);
          const y = MX.y + j * cellH;
          return (
            <g key={j}>
              <rect x={MX.x + MX.w / 2 + 1} y={y + 1} width={MX.w / 2 - 3} height={cellH - 2} className="liquid" style={{ opacity: 0.1 + 0.75 * wet }} />
              <text x={MX.x + MX.w * 0.75} y={y + cellH / 2 + 4} textAnchor="middle">{x < 1 ? `x ${fmt(x, 2)}` : `${fmt(Tq[j], 0)}°`}</text>
              <text x={MX.x + MX.w * 0.25} y={y + cellH / 2 + 4} textAnchor="middle" className="lbl">{fmt(Tg[j], 0)}°</text>
            </g>
          );
        })}
        <line x1={MX.x + MX.w / 2} y1={MX.y + 4} x2={MX.x + MX.w / 2} y2={MB - 4} stroke="#90a4ae" strokeWidth={1} />
        <text x={GX} y={MB + 12} textAnchor="middle" className="lbl">S1</text>
        <text x={GX - 12} y={MX.y - 6} textAnchor="end" className="lbl">S2</text>
        <text x={QX + 10} y={MX.y - 6} className="lbl">S3</text>
        <text x={QX + 8} y={MB + 12} className="lbl">S4</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 30} className="lbl">mixing</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 44} className="lbl">exchanger</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 62}>{fmt(t.Q_mx / 1000, 1)} kW</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 90} className="lbl">gas out</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 104}>{fmt(t.T_go, 1)} °C</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 126} className="lbl">quench out</text>
        <text x={MX.x + MX.w + 8} y={MX.y + 140} style={{ fill: t.x_qo < 1 ? "var(--bad)" : undefined }}>
          {t.x_qo < 1 ? `x ${fmt(t.x_qo, 2)}` : `${fmt(t.T_qo, 1)} °C`}</text>

        {/* suction line: tee -> elbow -> vertical lead-in into the compressor, probe at the port */}
        <polyline points={`${TX},${TY} ${CX},${TY} ${CX},${CY + CR}`} className="pipe suc" />
        <circle cx={CX} cy={CY + CR + 28} r={6} fill="#263238" stroke="#cfd8dc" strokeWidth={1.5} />
        <text x={CX + 12} y={CY + CR + 24}>{fmt(m.T_s, 1)} °C · SH {fmt(m.SH, 1)} K</text>
        <text x={CX + 12} y={CY + CR + 38} className="lbl">suction probe</text>
        <text x={(TX + CX) / 2 + 20} y={TY - 26} textAnchor="middle">{fmt(m.P_s, 2)} bar · sat {fmt(m.Tsat_s, 1)} °C</text>
        <text x={(TX + CX) / 2 + 20} y={TY - 10} textAnchor="middle" className="lbl">suction {fmt(m.mdot, 1)} g/s</text>
        {t.y_liq > 0.0005 && (
          <text x={(TX + CX) / 2 + 20} y={TY + 20} textAnchor="middle" style={{ fill: "var(--bad)" }}>
            liquid at compressor {fmt(t.y_liq * 100, 1)} %</text>
        )}
        {/* compressor */}
        <circle cx={CX} cy={CY} r={CR} className="comp-body" style={{ stroke: running ? "var(--ok)" : "#90a4ae" }} />
        <text x={CX} y={CY - 4} textAnchor="middle" className="lbl">compressor</text>
        <text x={CX} y={CY + 12} textAnchor="middle">{fmt(m.N, 0)} rpm</text>
        <text x={CX + CR + 8} y={CY - 2}>{fmt(m.W / 1000, 2)} kW</text>
        <text x={CX + CR + 8} y={CY + 13} className="lbl">shell {fmt(t.T_sh, 0)} °C</text>
        <text x={20} y={492} className="lbl">charge {fmt(snap.charge.kg, 3)} kg (nominal {fmt(snap.charge.nominal_kg, 3)} kg) · {snap.fluid} · ambient {fmt(m.T_amb, 1)} °C</text>
      </svg>
    </div>
  );
}
