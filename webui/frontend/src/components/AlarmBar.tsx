import type { Snapshot } from "../types";

const LABELS: Record<string, [string, "bad" | "warn"]> = {
  floodback: ["LIQUID AT COMPRESSOR INLET", "bad"],
  high_P_d_warning: ["DISCHARGE PRESSURE HIGH", "warn"],
  high_T_d_warning: ["DISCHARGE TEMPERATURE HIGH", "warn"],
  accumulator_liquid: ["liquid in accumulator", "warn"],
  condenser_dry: ["condenser dry (undercharged)", "warn"],
  condenser_flooded: ["condenser flooded (overcharged)", "warn"],
};

export default function AlarmBar({ snap }: { snap: Snapshot }) {
  const c = snap.compressor;
  const active = Object.entries(snap.alarms).filter(([, v]) => v);
  return (
    <div className="alarms">
      {c.tripped && <span className="chip bad">TRIP: {c.trip_reasons.join(", ") || "latched"} (reset on the compressor panel)</span>}
      {active.map(([k]) => (
        <span key={k} className={`chip ${LABELS[k]?.[1] ?? "warn"}`}>{LABELS[k]?.[0] ?? k}</span>
      ))}
      {snap.pending_params.length > 0 && (
        <span className="chip warn">parameter changes pending: {snap.pending_params.join(", ")} (applied at next cold/warm start)</span>
      )}
      {snap.charge.pending_kg !== 0 && <span className="chip">charging {snap.charge.pending_kg > 0 ? "+" : ""}{snap.charge.pending_kg.toFixed(3)} kg</span>}
      {snap.paused && <span className="chip">PAUSED</span>}
      {!c.tripped && active.length === 0 && <span className="chip ok">no alarms</span>}
    </div>
  );
}
