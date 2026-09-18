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

export default function Schematic({ snap }: { snap: Snapshot }) {
  const m = snap.meas, t = snap.true;
  const running = snap.compressor.running;
  const fillS = Math.min(1, Math.max(0, t.fill_s)), fillI = Math.min(1, Math.max(0, t.fill_i));
  // geometry
  const CX = 140, CY = 300, CR = 38;            // compressor
  const TOP = 60;                                // discharge / intermediate pipe level
  const COND = { x: 700, y: 120, w: 160, h: 90 }; // condenser
  const CIN = COND.x + COND.w / 2;               // condenser inlet / outlet x
  const TANK = { x: 400, y: 330, w: 230, h: 120 };
  const TCX = TANK.x + TANK.w / 2;               // tank centre x: bypass enters here
  const TCY = TANK.y + TANK.h / 2;               // tank centre y: liquid in / suction out here
  const BY = TCX;                                // bypass branch x
  const LIQY = TCY;                              // liquid line level
  const SUCY = TCY;                              // suction line level
  const WX = 930;                                // water pipe x
  return (
    <div className="card">
      <h2>Stand schematic</h2>
      <svg viewBox="0 0 1000 495" className="schem" style={{ width: "100%", height: "auto" }}>
        {/* discharge: compressor -> valve 1 -> intermediate pipe -> condenser */}
        <polyline points={`${CX},${CY - CR} ${CX},${TOP} ${CIN},${TOP} ${CIN},${COND.y}`} className="pipe hot" />
        <Valve x={CX} y={130} open={m.u1} label="1 discharge press." vertical />
        <text x={CX + 20} y={185} className="lbl">discharge</text>
        <text x={CX + 20} y={200}>{fmt(m.P_d, 2)} bar</text>
        <text x={CX + 20} y={214}>{fmt(m.T_d, 0)} °C</text>
        <text x={CX + 20} y={228} className="lbl">sat {fmt(m.Tsat_d, 1)} °C</text>
        <text x={300} y={TOP - 14} textAnchor="middle" className="lbl">intermediate pressure</text>
        <text x={300} y={TOP + 22} textAnchor="middle">{fmt(m.P_i, 2)} bar · sat {fmt(m.Tsat_i, 1)} °C</text>
        {/* condenser with liquid level and water connections */}
        <rect x={COND.x} y={COND.y} width={COND.w} height={COND.h} rx={6} className="vessel" />
        <rect x={COND.x + 2} y={COND.y + COND.h - 2 - (COND.h - 4) * fillI} width={COND.w - 4} height={(COND.h - 4) * fillI} className="liquid" />
        <text x={CIN} y={COND.y + 20} textAnchor="middle" className="lbl">condenser (BPHE)</text>
        <text x={CIN} y={COND.y + 38} textAnchor="middle">fill {fmt(fillI * 100, 0)} %</text>
        <text x={CIN} y={COND.y + 52} textAnchor="middle">SC {fmt(m.SC, 1)} K</text>
        <text x={CIN} y={COND.y + 66} textAnchor="middle">wall {fmt(t.T_cw, 0)} °C</text>
        <polyline points={`${WX},${TANK.y + TANK.h} ${WX},${COND.y + 70} ${COND.x + COND.w},${COND.y + 70}`} className="pipe water" />
        <polyline points={`${COND.x + COND.w},${COND.y + 20} ${WX},${COND.y + 20} ${WX},${TOP + 20}`} className="pipe water" />
        <Valve x={WX} y={300} open={m.u4} label="4 cooling water" vertical side="left" />
        <text x={WX + 12} y={TANK.y + TANK.h + 4} className="lbl">water in</text>
        <text x={WX + 12} y={TANK.y + TANK.h + 18} className="lbl">{fmt(m.T_wi, 1)} °C</text>
        <text x={WX - 12} y={TOP + 8} textAnchor="end" className="lbl">water out {fmt(m.T_wo, 1)} °C · {fmt(t.mdot_w, 1)} kg/min</text>
        {/* condenser outlet -> valve 3 -> tank */}
        <polyline points={`${CIN},${COND.y + COND.h} ${CIN},${LIQY} ${TANK.x + TANK.w},${LIQY}`} className="pipe liq" />
        <Valve x={CIN} y={290} open={m.u3} label="3 suction temp." vertical side="left" />
        <text x={(TANK.x + TANK.w + CIN) / 2} y={LIQY - 10} textAnchor="middle" className="lbl">liquid {fmt(t.mdot_3, 1)} g/s · {fmt(m.T_co, 0)} °C</text>
        {/* bypass branch -> valve 2 -> tank */}
        <polyline points={`${BY},${TOP} ${BY},${TANK.y}`} className="pipe hot" />
        <Valve x={BY} y={190} open={m.u2} label="2 suction press. (HGBP)" vertical />
        <text x={BY - 12} y={TANK.y - 12} textAnchor="end" className="lbl">bypass {fmt(t.mdot_2, 1)} g/s</text>
        {/* suction mixer / accumulator */}
        <rect x={TANK.x} y={TANK.y} width={TANK.w} height={TANK.h} rx={10} className="vessel" />
        <rect x={TANK.x + 2} y={TANK.y + TANK.h - 2 - (TANK.h - 4) * fillS} width={TANK.w - 4} height={(TANK.h - 4) * fillS} className="liquid" />
        <text x={TANK.x + TANK.w / 2} y={TANK.y + 22} textAnchor="middle" className="lbl">suction mixer / accumulator</text>
        <text x={TANK.x + TANK.w / 2} y={TANK.y + 42} textAnchor="middle">{fmt(m.P_s, 2)} bar · sat {fmt(m.Tsat_s, 1)} °C</text>
        <text x={TANK.x + TANK.w / 2} y={TANK.y + 60} textAnchor="middle">T {fmt(m.T_s, 1)} °C · SH {fmt(m.SH, 1)} K</text>
        <text x={TANK.x + TANK.w / 2} y={TANK.y + 78} textAnchor="middle">liquid {fmt(fillS * 100, 1)} % · x {fmt(t.x_out, 3)}</text>
        {/* suction line: tank -> elbow -> vertical lead-in into the compressor */}
        <polyline points={`${TANK.x},${SUCY} ${CX},${SUCY} ${CX},${CY + CR}`} className="pipe suc" />
        <text x={(TANK.x + CX) / 2} y={SUCY + 18} textAnchor="middle" className="lbl">suction {fmt(m.mdot, 1)} g/s</text>
        {/* compressor */}
        <circle cx={CX} cy={CY} r={CR} className="comp-body" style={{ stroke: running ? "var(--ok)" : "#90a4ae" }} />
        <text x={CX} y={CY - 4} textAnchor="middle" className="lbl">compressor</text>
        <text x={CX} y={CY + 12} textAnchor="middle">{fmt(m.N, 0)} rpm</text>
        <text x={CX - CR - 8} y={CY - 2} textAnchor="end">{fmt(m.W / 1000, 2)} kW</text>
        <text x={CX - CR - 8} y={CY + 13} textAnchor="end" className="lbl">shell {fmt(t.T_sh, 0)} °C</text>
        <text x={20} y={482} className="lbl">charge {fmt(snap.charge.kg, 3)} kg (nominal {fmt(snap.charge.nominal_kg, 3)} kg) · {snap.fluid} · ambient {fmt(m.T_amb, 1)} °C</text>
      </svg>
    </div>
  );
}
