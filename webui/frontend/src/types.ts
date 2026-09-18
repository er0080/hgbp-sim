export interface Loop {
  label: string; valve_label: string; unit: string;
  pv: number; sp: number; out: number; mode: "auto" | "manual"; manual_out: number;
  Kp: number; Ki: number; Kd: number;
}
export interface Snapshot {
  t: number; step: number; paused: boolean; speed_factor: number; noise: boolean; dt_ctrl: number; fluid: string;
  compressor: {
    state: string; tripped: boolean; trip_reasons: string[]; run_request: boolean; speed_sp: number; speed: number;
    running: boolean; permissive_ok: boolean; permissives: Record<string, boolean>; t_state: number;
    min_off_time: number; min_run_time: number; N_min: number; N_max: number;
  };
  loops: Record<string, Loop>;
  meas: Record<string, number>;
  true: Record<string, number>;
  alarms: Record<string, boolean>;
  limits: Record<string, number>;
  charge: { kg: number; nominal_kg: number; pending_kg: number; rate_kg_s: number };
  pending_params: string[];
  events: { t: number; msg: string }[];
}
export type History = Record<string, number[]>;
export interface ParamMeta {
  name: string; group: string; unit: string; description: string; default: any; kind: "float" | "str" | "bool";
  requires_init: boolean;
}
export interface ParamsView {
  values: Record<string, any>; meta: ParamMeta[]; pending: string[]; fluids: string[];
  named_points: Record<string, { T_evap: number; T_cond: number; T_int: number; SH: number }>;
  nominal_charge: number;
}
