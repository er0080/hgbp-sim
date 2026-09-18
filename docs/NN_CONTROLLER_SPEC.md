# Neural-network controller for the HGBP compressor test stand: specification

Version 0.1, 2026-09-17. Companion to the `hgbp-sim` simulator (v0.2). This
document is written so that the controller can be developed and trained in a
separate repository, using the simulator only through its environment interface.

## 1. Purpose and scope

The controller automates a hot-gas-bypass compressor test stand: it drives the stand
to a sequence of test points, holds each point inside the tolerance band long enough
for a valid measurement, moves on, and shuts the stand down, without an operator in the
loop. It must do so across different compressors (swept volume, speed), different
refrigerants and different (possibly incorrect) refrigerant charges, and it must
respect the stand's safety limits at all times.

Goals, in priority order:

1. never trip the stand or flood the compressor with liquid;
2. reach every test point and hold it inside tolerance for the required dwell;
3. minimize cycle time (time to tolerance, number of unstable dwells);
4. minimize actuator wear (valve travel);
5. tell the operator whether the stand is under- or over-charged.

Out of scope for the network: safety interlocks and trips (they live in the PLC and are
mirrored in the simulator), the test schedule itself (an input), and data acquisition.

## 2. The control problem

The stand has four manipulated valves and four controlled variables:

| controlled variable | primary valve | secondary effects |
|---|---|---|
| discharge pressure | 1, discharge pressure valve | lowers the intermediate pressure and the bypass flow |
| suction pressure | 2, hot gas bypass valve | raises discharge pressure, superheat and mass flow |
| suction superheat | 3, liquid (suction temperature) valve | raises suction and discharge pressure |
| intermediate (condensing) pressure | 4, cooling water valve | moves every other variable |

Open-loop step responses from the medium-temperature point (`examples/open_loop_step.py`):
a 10 % opening of valve 2 changes suction pressure by +0.6 bar, discharge pressure by
+1.8 bar and superheat by +11 K. The plant is therefore a strongly coupled, nonlinear
4x4 system; single-loop PID (the baseline) works but produces large superheat excursions
at every setpoint change and needs several minutes to settle.

Properties that shape the network design:

* **Hidden state.** The condenser liquid inventory, the liquid stored in the suction
  accumulator, the compressor shell temperature and the condenser wall temperature are
  not measured but determine the plant gains and the time constants (tens of seconds to
  ~20 min). The charge is also hidden.
* **Wide operating range.** Valve gains vary by more than an order of magnitude over
  their stroke; gas valves choke; the liquid valve flashes.
* **Integral action needed.** Sensor bias, model mismatch and slow drifts require
  offset-free tracking.
* **Constraints.** Discharge pressure and temperature limits, suction pressure limits,
  no liquid at the compressor inlet, actuator rate limits, anti-short-cycle timers.
* **Hybrid decisions.** Compressor start/stop and the charge advisory are discrete.

## 3. Interface to the simulator

All arrays below come from `hgbp_sim.HGBPVecEnv` (batched) or `hgbp_sim.HGBPEnv`
(gymnasium). One environment instance is one refrigerant; train across refrigerants by
running several instances (one per fluid) and sampling batches across them.

### 3.1 Observation vector (51 entries, `hgbp_sim.OBS_NAMES`, groups in `OBS_GROUPS`)

All entries are scaled to order one. Temperatures in degC / 50 unless noted, pressures
are expressed as saturation temperatures (refrigerant-agnostic), errors in kelvin.

| group | entries | notes |
|---|---|---|
| measurements (18) | `Tsat_s`, `Tsat_d`, `Tsat_i` (saturation temperature at suction, discharge, intermediate pressure); `T_s` (compressor inlet), `T_d` (/150), `T_co` (condenser outlet, /80); `SH` (/30), `SC` (/20); `mdot_norm` = mass flow / (saturated vapor density at P_suc x swept volume x speed), a volumetric-efficiency proxy in 0..1; `W_norm` = power / (swept volume x speed x P_suc) (/3); `T_wi`, `T_wo`, `T_amb`; `N_rel` = speed / nominal; `u1..u4` valve positions | sensor lag and noise included; `mdot_norm`, `W_norm` are 0 while the compressor is off |
| setpoints (5) | `Tsat_s_sp`, `Tsat_d_sp`, `SH_sp`, `Tsat_i_sp`, `N_sp_rel` | current test point |
| errors (8) | `e_Tsat_s`, `e_Tsat_d`, `e_SH`, `e_Tsat_i` (setpoint minus measurement, K, /10); `ie_*` clipped integrals of error/scale in minutes (+-5, /5) | integrals reset at each point change and while stopped |
| context (10) | `V_disp_rel` = swept volume / 250 cm3; `N_nom_rel` = nominal speed / 1500 rpm; refrigerant descriptors `rf_T_crit` (/400 K), `rf_P_crit` (/50 bar), `rf_M_molar` (/0.1 kg/mol), `rf_P_sat_ref` (/5 bar at 0 degC), `rf_h_fg_ref` (/200 kJ/kg), `rf_rho_v_ref` (/20 kg/m3), `rf_rho_l_ref` (/1200 kg/m3), `rf_dPsat_dT_ref` (/0.1 bar/K) | constant within an episode; descriptors let the policy interpolate to refrigerants not seen in training |
| status (10) | `running`, `run_required` (schedule wants the compressor on), `permissive_ok` (start permissives hold), one-hot interlock state `st_off/st_starting/st_running/st_stopping`, `t_point`, `t_in_tol`, `t_since_switch` (s, /600, capped) | |

### 3.2 Action vector (`[-1, 1]`)

| index | meaning | mapping |
|---|---|---|
| 0..3 | valves 1..4 | incremental (default): command += a x `max_rate` (0.05 per second); absolute: u = (a + 1) / 2 |
| 4 | run request (only with `EnvConfig(start_stop_action=True)`) | > 0 requests run, < 0 requests stop; the interlock state machine decides |

Use incremental mode. It gives PID-like robustness, cannot slam a valve, and makes the
policy's output a rate that the PLC can clamp.

### 3.3 Interlock state machine (in the environment; to be replicated in the PLC)

* `OFF -> STARTING`: run request AND permissives (suction pressure between 1.5x the low
  trip and 0.9x the high trip, discharge pressure below 0.8x its trip, valve 1 >= 10 %
  open, valve 2 >= 5 % open, no trip) AND off time >= `min_off_time` (60 s).
* `STARTING -> RUNNING`: speed >= 90 % of setpoint.
* `RUNNING -> STOPPING`: stop request AND run time >= `min_run_time` (120 s).
* `STOPPING -> OFF`: speed below half the minimum speed.
* Any trip: episode terminates with the trip penalty (the PLC would lock out).

### 3.4 Episode structure and labels

* 1..3 test points from the envelope (evaporating -30..12 degC, condensing 30..65 degC,
  superheat 3..25 K, intermediate temperature between water inlet and condensing,
  50..140 % speed, estimated discharge temperature below the trip), each checked
  against the equilibrium solver at nominal charge (every valve below 95 % at steady
  state; unreachable points are resampled), each with a maximum hold of 5..15 min. A point is **completed** when the three test variables have been
  inside the tolerance band (0.3 K, 0.2 K, 0.5 K on Tsat_suc, Tsat_dis, SH) for
  `dwell_required` = 180 s; then the next point is loaded. After the last point
  `run_required` drops to 0 and the episode terminates successfully once the compressor
  is off.
* Start modes: warm (at or near a point, compressor running), cold (equalized stand at
  ambient with liquid in the accumulator, compressor off), sampled 70/30.
* Per episode randomization: plant parameters (+-10..30 %), ambient 15..35 degC, water
  inlet 12..30 degC, charge factor 0.85..1.15 (85 % of episodes) or 0.3..1.8 (15 %),
  liquid distribution at cold start, sensor noise.
* `info` provides training labels and privileged states: `charge_factor` (true charge /
  nominal), `steady` (in tolerance for >= 120 s), `true` (noise-free pressures,
  temperatures, superheat, subcooling, condenser and accumulator liquid fills, inlet
  quality, shell and condenser wall temperatures, flows, power, charge), interlock
  `state`, `permissive_ok`, `point_done`, `schedule_complete`, per-term reward
  components.

### 3.5 Reward (per 1 s control step)

| term | value |
|---|---|
| tracking | - mean over the four errors of min(|e| / scale, 10), weights (1, 1, 1, 0.5), scales (1.0, 0.7, 1.0, 1.0) K, only while running and required |
| in tolerance | +0.5 per step |
| point completed by dwell | +5 |
| actuator movement | -0.05 x sum |du| |
| liquid at compressor inlet | -2 per step |
| discharge temperature margin | -(T_d - (T_d,max - 15 K)) / 15 K when positive |
| idle while start possible | -0.5 per step (start/stop mode) |
| blocked start request | -0.2 |
| running during shutdown | -0.5 per step |
| shutdown complete | +5 (episode ends) |
| trip | -50 (episode ends) |

The weights are a starting point; keep the ordering of Section 1 when changing them.

## 4. Network architecture

### 4.1 Overview

```
observation (51) --> [feature MLP 2 x 128, ELU] --> [GRU 128] --> hidden h_t
                                                                   |-- valve head: 4 x tanh (incremental commands)
                                                                   |-- run head:   1 logit (run request)
                                                                   |-- charge head: 1 regression (log charge factor)
privileged states (training only) + h_t --> [value MLP 2 x 128] --> V(s)
```

About 120 k parameters. Inference cost is negligible at 1 Hz.

### 4.2 Design decisions and rationale

* **Recurrent core (GRU).** Required by the hidden thermal and inventory states and by
  the charge estimation task; also supplies integral action. A window of the last 60..120
  observations feeding an MLP or a small temporal convolution is an acceptable
  alternative if recurrent inference is inconvenient on the target hardware; a
  transformer is not warranted at this state size and rate.
* **Incremental valve outputs** with the tanh output scaled to the rate limit.
* **Run head** produces a request only; the state machine has authority. Train it with a
  supervised target (`run_required` from the schedule, gated by permissives) and let RL
  fine-tune the timing.
* **Charge head** regresses `log(charge / nominal)` with a supervised loss against
  `info["charge_factor"]`, masked to steps where `info["steady"]` is true. The advisory
  logic (Section 6) sits outside the network.
* **Asymmetric critic.** The value network sees the privileged states listed in 3.4 in
  addition to the policy's hidden state; only the policy must work from measurements.
* **Context conditioning.** Swept volume, nominal speed and refrigerant descriptors are
  part of the observation. Consider a FiLM-style modulation (context -> per-layer scale
  and shift) if plain concatenation underfits across refrigerants.
* **Output smoothing.** Add a penalty on the change of the valve output between steps
  during training (already part of the reward) and optionally a small low-pass on the
  action at deployment.

### 4.3 Input handling

* Observations are already scaled; clip to [-5, 5] as a guard.
* Reset the GRU hidden state at episode start only, not at test-point changes.
* For deployment, the same observation vector must be assembled from the stand's DAQ
  (Section 7); saturation temperatures come from the refrigerant property library, the
  flow and power normalizations from the compressor data sheet.

## 5. Training plan

1. **Data and baseline.** Collect 5..10 M steps with `examples/collect_dataset.py`
   (expert + Gaussian exploration noise, sigma 0.3) across all start modes, charges and
   several refrigerants. Record the expert action, `charge_factor`, `steady`.
2. **Behaviour cloning.** Train the policy (valve and run heads) on the expert actions
   with MSE / cross-entropy, sequence length 256 steps, truncated BPTT. Train the charge
   head jointly. Then run 3..5 DAgger rounds: roll out the cloned policy, relabel with
   the expert, retrain. Target: match the expert's tracking with no trips.
3. **RL fine-tuning.** PPO with the recurrent policy, 512..1024 parallel environments,
   rollouts of 128..256 steps, GAE lambda 0.95, gamma 0.995 (time constants of minutes at
   1 s steps), entropy bonus small, learning rate 3e-4 decaying, KL penalty against the
   cloned policy for the first iterations to avoid unlearning. Keep the supervised charge
   loss as an auxiliary term.
4. **Curriculum.** Nominal charge, warm starts, single points first; then multiple
   points, cold starts, wider charge range, start/stop action, then multiple
   refrigerants and the full compressor size range.
5. **Evaluation** on held-out seeds and refrigerants, 1000 episodes each: trips per 1000
   episodes (target 0), fraction of points completed by dwell, mean time to tolerance
   after a setpoint change, superheat undershoot events (inlet quality < 1), valve travel
   per point, charge estimate error (target +-10 % once steady), and the same metrics for
   the PID baseline as the reference.

Suggested rewards to log separately during training: tracking, tolerance bonus, point
completion, movement, floodback, trips. A policy that improves cycle time by sacrificing
floodback margin is not acceptable.

## 6. Predicates and logic around the network

### 6.1 Action shield (PLC / supervisory layer)

Applied to every network output before it reaches the actuators:

* clamp increments to the actuator rate limits and positions to [0, 1];
* valve 1 never below 10 % while the compressor runs (no dead-heading);
* valve 2 never below 5 % while the compressor runs (bypass path always open);
* freeze valve 3 (liquid) closed and open valve 2 if the measured superheat is below
  1 K or the compressor inlet temperature is at saturation for more than 30 s;
* if the discharge temperature exceeds the trip minus 10 K: open valve 3 by a fixed rate
  regardless of the network; if the discharge pressure exceeds the trip minus 1 bar: open
  valve 1 and valve 4 by a fixed rate;
* on any sensor validity failure (stuck, out of range, rate of change beyond physical
  limits) hand over to the PID baseline and alarm.

### 6.2 Start/stop

The state machine of Section 3.3 has authority. The network's run head is one input; the
schedule's `run_required` is another; permissives and timers are non-negotiable. Start
sequence before the request is honoured: valve 2 to 60 %, valve 1 to 50 %, valve 3
closed, valve 4 to 30 % (the expert's rest positions).

### 6.3 Charge advisory

* Evaluate the charge head only while `steady` is true; average over the dwell of each
  completed point; report once per point.
* Thresholds with hysteresis: estimate < 0.8 -> "add charge", > 1.25 -> "remove charge",
  otherwise "ok"; require two consecutive points before changing the advisory.
* Supporting physical indicators that must agree before the advisory is shown (both are
  visible to an operator and make the estimate auditable): undercharge = valve 3 near
  fully open with superheat above setpoint and subcooling near 0 K; overcharge =
  subcooling above 8 K with valve 4 unusually open for the condensing temperature.

### 6.4 Steady-state (measurement valid) predicate

Reuse the simulator's definition: the three test variables inside the tolerance band
for `dwell_required` seconds with the compressor running and no liquid at the inlet.
Add on the real stand: mass flow and power readings within +-1 % of their trailing
5-minute mean.

## 7. Deployment on the real stand

* **System identification first.** Run the open-loop step protocol on the real stand
  (valves stepped +-10 % from two or three test points, cold start logged) and fit the
  simulator parameters (volumes, valve coefficients, condenser UA, shell thermal mass,
  sensor time constants, compressor map). Keep the fitted parameters as the center of the
  randomization ranges; keep the ranges (the policy must tolerate residual mismatch).
* **Same observation pipeline.** Assemble the observation on the stand from the same
  definitions (Section 3.1), including the property conversions and the normalizations.
  Log the assembled vector so simulator and stand data are directly comparable.
* **Timing.** 1 s control interval; measurement filtering identical to the simulator's
  sensor model (first-order, 4 s temperature, 1 s flow). Verify end-to-end latency
  (DAQ -> network -> actuator command) below 200 ms.
* **Staged rollout.** (1) Shadow mode: the network runs alongside the PID, its actions
  are logged, not applied; compare. (2) Advisory mode: the network's increments are
  applied with reduced authority (rate limit at 25 %) and the PID can override.
  (3) Full authority on the valves with the action shield; start/stop and charge advisory
  last. Keep the PID baseline as a one-button fallback at every stage.
* **Data loop.** Keep every real run (observations, actions, outcomes). Use it to refit
  the simulator, to fine-tune with real trajectories (offline RL or BC on real data
  mixed with simulated), and to validate the charge estimator against known charge
  changes (weigh the charge in and out).
* **Acceptance criteria** before removing operator supervision: 100 consecutive test
  points with zero trips and zero floodback events, cycle time at least 20 % below the
  manual/PID reference, charge advisory correct on deliberate +-20 % charge changes.

## 8. Recommendations beyond the controller

* Add a **liquid line sight glass or a light-transmission sensor** and a **condenser
  outlet temperature** sensor if the stand does not have them; subcooling is the most
  informative signal for charge.
* Add an **accumulator level indication** (capacitive or differential pressure) if
  possible; it turns the hardest hidden state into a measurement and reduces floodback
  risk at startup.
* Log the **water inlet temperature and flow**; the water loop is the slowest and least
  observable part of the plant.
* Consider a **feed-forward table** (steady-state valve positions as a function of the
  test point, compressor and refrigerant, from `solve_steady_state`) as the first thing
  applied at a setpoint change, with the network correcting around it. It is a cheap,
  auditable way to cut the transition time and it makes the learned part smaller.
* Keep test-point transitions **ordered by lift** where the schedule allows: moving the
  intermediate pressure with the condensing pressure avoids the largest superheat
  excursions.

## 9. Open questions for the stand owner

1. Which refrigerants and compressor sizes must be covered in the first release?
2. Is the intermediate pressure a setpoint given by the test procedure or a value the
   controller may choose within limits? (The simulator supports both.)
3. What is the required dwell and tolerance band per test standard used?
4. Which safety actions are already implemented in the PLC, and can the shield of
   Section 6.1 be added there?
5. Can charge be measured (scale on the charging cylinder) to validate the advisory?
