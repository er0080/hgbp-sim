import type { Snapshot } from "../types";
import { useUnits } from "../units";

const LABELS: Record<string, [string, "bad" | "warn"]> = {
  floodback: ["LIQUID AT COMPRESSOR INLET", "bad"],
  high_P_d_warning: ["DISCHARGE PRESSURE HIGH", "warn"],
  high_T_d_warning: ["DISCHARGE TEMPERATURE HIGH", "warn"],
  mixer_wet: ["quench liquid leaving the mixing exchanger", "warn"],
  no_liquid_seal: ["receiver below dip tube: vapor in liquid line (undercharged)", "warn"],
  receiver_full: ["receiver above 90 % (overcharged)", "warn"],
  condenser_flooded: ["condenser flooding (overcharged)", "warn"],
};

export default function AlarmBar({ snap }: { snap: Snapshot }) {
  const c = snap.compressor;
  const u = useUnits();
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
      {snap.charge.pending_kg !== 0 && <span className="chip">charging {snap.charge.pending_kg > 0 ? "+" : ""}{u.fmtU("mass", snap.charge.pending_kg)}</span>}
      {snap.paused && <span className="chip">PAUSED</span>}
      {!c.tripped && active.length === 0 && <span className="chip ok">no alarms</span>}
    </div>
  );
}
