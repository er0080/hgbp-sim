import { fmt } from "../api";
import type { Snapshot } from "../types";

function Valve({ x, y, open, label, vertical = false }: { x: number; y: number; open: number; label: string; vertical?: boolean }) {
  const s = 13;
  const pts = vertical
    ? `${x - s},${y - s} ${x + s},${y - s} ${x - s},${y + s} ${x + s},${y + s}`
    : `${x - s},${y - s} ${x - s},${y + s} ${x + s},${y - s} ${x + s},${y + s}`;
  const col = `rgba(79,195,247,${0.15 + 0.85 * Math.min(1, Math.max(0, open))})`;
  return (
    <g>
      <polygon points={pts} className="valve" style={{ fill: col }} />
      <line x1={x} y1={y} x2={x + (vertical ? 22 : 0)} y2={y - (vertical ? 0 : 22)} stroke="#cfd8dc" strokeWidth={2} />
      <rect x={x + (vertical ? 22 : -9)} y={y - (vertical ? 9 : 30)} width={vertical ? 8 : 18} height={vertical ? 18 : 8} fill="#cfd8dc" />
      <text x={vertical ? x + 36 : x} y={vertical ? y + 4 : y + 32} textAnchor={vertical ? "start" : "middle"} className="lbl">{label}</text>
      <text x={vertical ? x + 36 : x} y={vertical ? y + 17 : y + 45} textAnchor={vertical ? "start" : "middle"}>{fmt(open * 100, 0)} %</text>
    </g>
  );
}

export default function Schematic({ snap }: { snap: Snapshot }) {
  const m = snap.meas, t = snap.true;
  const running = snap.compressor.running;
  const fillS = Math.min(1, Math.max(0, t.fill_s)), fillI = Math.min(1, Math.max(0, t.fill_i));
  return (
    <div className="card">
      <h2>Stand schematic</h2>
      <svg viewBox="0 0 900 440" className="schem" style={{ width: "100%", height: "auto" }}>
        {/* discharge line: compressor -> valve 1 -> header */}
        <polyline points="150,300 150,60 870,60" className="pipe hot" />
        <text x={100} y={175} className="lbl">discharge</text>
        <text x={100} y={190}>{fmt(m.P_d, 2)} bar</text>
        <text x={100} y={203}>{fmt(m.T_d, 0)} °C</text>
        <text x={100} y={216} className="lbl">sat {fmt(m.Tsat_d, 1)} °C</text>
        <Valve x={150} y={110} open={m.u1} label="1 discharge press." vertical />
        <text x={480} y={50} textAnchor="middle" className="lbl">intermediate header</text>
        <text x={480} y={80} textAnchor="middle">{fmt(m.P_i, 2)} bar · sat {fmt(m.Tsat_i, 1)} °C</text>
        {/* condenser branch */}
        <polyline points="800,60 800,120" className="pipe hot" />
        <rect x={720} y={120} width={160} height={90} rx={6} className="vessel" />
        <rect x={722} y={210 - 86 * fillI} width={156} height={86 * fillI} className="liquid" />
        <text x={800} y={140} textAnchor="middle" className="lbl">condenser (BPHE)</text>
        <text x={800} y={158} textAnchor="middle">fill {fmt(fillI * 100, 0)} %</text>
        <text x={800} y={172} textAnchor="middle">SC {fmt(m.SC, 1)} K</text>
        <text x={800} y={186} textAnchor="middle">wall {fmt(t.T_cw, 0)} °C</text>
        <polyline points="880,190 895,190 895,235" className="pipe water" />
        <polyline points="895,95 895,140 880,140" className="pipe water" />
        <text x={860} y={250} textAnchor="end" className="lbl">water in {fmt(m.T_wi, 1)} °C</text>
        <text x={860} y={90} textAnchor="end" className="lbl">out {fmt(m.T_wo, 1)} °C · {fmt(t.mdot_w, 1)} kg/min</text>
        <Valve x={870} y={270} open={m.u4} label="4 cooling water" />
        <polyline points="800,210 800,330 630,330" className="pipe liq" />
        <Valve x={800} y={265} open={m.u3} label="3 suction temp." vertical />
        <text x={700} y={318} textAnchor="middle" className="lbl">liquid {fmt(t.mdot_3, 1)} g/s · {fmt(m.T_co, 0)} °C</text>
        {/* bypass branch */}
        <polyline points="480,60 480,270" className="pipe hot" />
        <Valve x={480} y={160} open={m.u2} label="2 suction press. (HGBP)" vertical />
        <text x={470} y={250} textAnchor="end" className="lbl">bypass {fmt(t.mdot_2, 1)} g/s</text>
        {/* accumulator tank */}
        <rect x={400} y={270} width={230} height={120} rx={10} className="vessel" />
        <rect x={402} y={388 - 116 * fillS} width={226} height={116 * fillS} className="liquid" />
        <text x={515} y={292} textAnchor="middle" className="lbl">suction mixer / accumulator</text>
        <text x={515} y={312} textAnchor="middle">{fmt(m.P_s, 2)} bar · sat {fmt(m.Tsat_s, 1)} °C</text>
        <text x={515} y={330} textAnchor="middle">T {fmt(m.T_s, 1)} °C · SH {fmt(m.SH, 1)} K</text>
        <text x={515} y={348} textAnchor="middle">liquid {fmt(fillS * 100, 1)} % · x {fmt(t.x_out, 3)}</text>
        {/* suction line to compressor */}
        <polyline points="400,360 150,360 150,340" className="pipe suc" />
        <text x={275} y={350} textAnchor="middle" className="lbl">suction {fmt(m.mdot, 1)} g/s</text>
        {/* compressor */}
        <circle cx={150} cy={320} r={38} className="comp-body" style={{ stroke: running ? "var(--ok)" : "#90a4ae" }} />
        <text x={150} y={316} textAnchor="middle" className="lbl">compressor</text>
        <text x={150} y={332} textAnchor="middle">{fmt(m.N, 0)} rpm</text>
        <text x={150} y={385} textAnchor="middle">{fmt(m.W / 1000, 2)} kW</text>
        <text x={150} y={400} textAnchor="middle" className="lbl">shell {fmt(t.T_sh, 0)} °C</text>
        <text x={20} y={425} className="lbl">charge {fmt(snap.charge.kg, 3)} kg (nominal {fmt(snap.charge.nominal_kg, 3)} kg) · {snap.fluid} · ambient {fmt(m.T_amb, 1)} °C</text>
      </svg>
    </div>
  );
}
