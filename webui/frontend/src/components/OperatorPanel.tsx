import type { Snapshot } from "../types";
import Schematic from "./Schematic";
import PidFaceplate from "./PidFaceplate";
import CompressorPanel from "./CompressorPanel";
import KpiTiles from "./KpiTiles";

const ORDER = ["dpv", "spv", "stv", "water"];

export default function OperatorPanel({ snap }: { snap: Snapshot }) {
  return (
    <div className="grid-op">
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <Schematic snap={snap} />
        <CompressorPanel snap={snap} />
        <KpiTiles snap={snap} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="faceplates">
          {ORDER.map((k) => <PidFaceplate key={k} name={k} loop={snap.loops[k]} />)}
        </div>
        <div className="card">
          <h2>Event log</h2>
          <ul className="events">
            {snap.events.map((e, i) => <li key={i}><span>{e.t.toFixed(0)} s</span>{e.msg}</li>)}
          </ul>
        </div>
      </div>
    </div>
  );
}
