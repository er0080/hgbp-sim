export type Time = number | "OFF";
// UT35A settings: P proportional band [% of RL..RH], I / D [s] or OFF, DR action, OL..OH output limits [%],
// FL PV input filter [s] or OFF
export interface Tuning { P: number; I: Time; D: Time; DR: "DIR" | "RVS"; RL: number; RH: number; OL: number; OH: number; FL: Time }
export interface Loop extends Tuning {
  label: string; valve_label: string; unit: string;
  pv: number; sp: number; out: number; mode: "auto" | "manual"; manual_out: number;
}
export interface Snapshot {
  t: number; step: number; paused: boolean; speed_factor: number; achieved_speed: number; noise: boolean; dt_ctrl: number; fluid: string;
  compressor: {
    state: string; tripped: boolean; trip_reasons: string[]; run_request: boolean; speed_sp: number; speed: number;
    running: boolean; permissive_ok: boolean; permissives: Record<string, boolean>; t_state: number;
    min_off_time: number; min_run_time: number; short_cycle_timers: boolean; N_min: number; N_max: number;
  };
  loops: Record<string, Loop>;
  meas: Record<string, number>;
  // true (model) values; x_q, T_q, T_g, T_mw are per mixing exchanger cell and x_c, T_c (refrigerant),
  // T_wc (water), T_cwc (wall) per condenser cell, top -> bottom
  true: Record<string, number> & { x_q: number[]; T_q: number[]; T_g: number[]; T_mw: number[];
    x_c: number[]; T_c: number[]; T_wc: number[]; T_cwc: number[] };
  alarms: Record<string, boolean>;
  limits: Record<string, number>;
  charge: { kg: number; nominal_kg: number; pending_kg: number; rate_kg_s: number };
  pending_params: string[];
  events: { t: number; msg: string }[];
}
export type History = Record<string, number[]>;
export interface ParamMeta {
  name: string; group: string; unit: string; description: string; default: any; kind: "float" | "str" | "bool";
  requires_init: boolean; min: number | null; max: number | null;
}
export interface Defaults {
  simulation: { speed_factor: number; noise: boolean; dt_ctrl: number; T_amb: number; T_wi: number; charge_rate_g_s: number;
    short_cycle_timers: boolean };
  loops: Record<string, Tuning & { SP: number }>;
  plant: Record<string, any>;
}
export interface ParamsView {
  values: Record<string, any>; meta: ParamMeta[]; pending: string[]; fluids: string[];
  named_points: Record<string, { T_evap: number; T_cond: number; T_int: number; SH: number }>;
  nominal_charge: number; defaults: Defaults; defaults_path: string | null;
  derived: { volumes: { section: string; item: string; L: number }[]; V_s: number; V_d: number; V_i: number;
    C_mw: number; C_cw: number; C_sw: number; C_dw: number; C_rw: number };
}
