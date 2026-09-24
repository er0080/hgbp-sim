# Suction mixer / accumulator: model review and known limitations

> **Superseded (2026-09-24).** The stand no longer has a suction mixer tank: the model
> now represents the brazed-plate mixing exchanger, the liquid receiver and the specified
> piping (`STAND_MODEL.md`). This note is kept as the record of why the tank model was
> replaced; the volumes, parameters and line numbers below refer to the former model.

2026-09-22. Findings from an investigation into accumulator boil-off with the liquid
injection valve (valve 3) closed. No model changes were made; this note records what the
model does, what was measured, how the result should be read, and where the model departs
from the real vessel.

## 1. Observation that prompted the review

Running the live stand with liquid in the suction mixer/accumulator and valve 3 closed
manually (no liquid leaving the condenser), the suction pressure and temperature stayed
nearly constant while the liquid in the tank boiled off, and the boil-off seemed to take
unnaturally long given compressor power of several tens of kilowatts. The expectation was
that, with the condenser outlet blocked, heat rejection to the cooling water would stop and
the system pressures would climb.

## 2. How the model treats the tank today

`hgbp_sim/plant.py`, "suction accumulator tank" block (around lines 167-171), with the energy
balance `E_s` further down:

* One lumped control volume `V_s` (tank plus suction line) at pressure `P_s` and a single
  mean enthalpy `h_s`. Liquid and vapor share one saturation state (homogeneous equilibrium).
* Everything entering the tank mixes into that volume instantly: the hot-gas bypass stream
  (valve 2, taken from the discharge gas first, so it carries the discharge superheat) and the
  liquid from valve 3. Perfect thermal and mechanical mixing.
* Outlet to the compressor: saturated vapor while the tank holds liquid, blending to the tank's
  superheated state over the quality band `acc_blend_dx`. Liquid is entrained into the outlet
  only once the liquid fill exceeds `acc_carry_fill0` (0.5), rising smoothly to `acc_carry_max`
  (30 % liquid in the outlet stream) at a full tank.
* Tank and suction line walls: heat capacity `C_sw` with small conductances to the refrigerant
  (`UA_sg`, 40 W/K) and to ambient (`UA_sa`, 6 W/K). These are negligible next to the
  refrigerant streams.

Consequences: while liquid is present, the compressor draws saturated vapor (measured
superheat 0), and every watt of superheat in the bypass stream goes into boiling liquid
immediately. The boil-off rate is

    boil-off [kg/s] = mdot_2 * (h_bypass - h_v(P_s)) / h_fg(P_s)

## 3. What was measured

Configuration: the defaults file at the time (`webui/config/stand_defaults.json`: `V_s` 120 L,
`V_d` 15 L, `V_i` 30 L, 17.4 kg charge, 10 degC cooling water, UT35A tuning P 150 %, I 50 s),
condenser model with the subcooled zone (branch `condenser-subcooled-zone`).

Scenario: cold start with all of the liquid in the accumulator, valve 3 closed in MAN,
compressor started with loops 1, 2 and 4 in MAN at start positions, then handed to AUTO.

| Speed | Liquid in tank | Time to boil off | Rate (model) | Rate (energy balance above) |
|---|---|---|---|---|
| 3550 rpm | 6.9 kg | ~60 s | 0.10 kg/s | 0.09 kg/s |
| 1200 rpm | 6.9 kg | ~210 s | 0.027-0.040 kg/s | same within a few % |

Other observations from the same runs:

* Liquid injected into a dry, superheated tank during normal operation (3 kg at 50 g/s,
  loops in AUTO) boiled off as fast as it arrived; the vapor ended up as liquid in the
  condenser (condenser liquid fill 0.32 -> 0.43).
* Once the tank ran dry with valve 3 still closed, nothing cooled the recirculating bypass
  gas: suction temperature rose to about 110 degC and the stand tripped on high discharge
  temperature (152 degC in that defaults file). Closing valve 3 on an already dry tank trips
  within about 30 s. Both are correct physics.

## 4. Interpretation

**The boil-off rate is at the energy limit, not artificially slow.** R410A has a latent heat
of 212 kJ/kg at 9.98 bar. Even if all of the compressor's electrical power went into boiling
liquid, the ceiling would be:

* about 0.05 kg/s (3 kg/min) at 1200 rpm, roughly 11 kW;
* about 0.15 kg/s (9 kg/min) at 3550 rpm, 32.5 kW.

At 1200 rpm the model already puts about 70 % of the compressor power into boiling. Tens of
kilowatts correspond to only a tenth of a kilogram of refrigerant per second.

**Closing valve 3 blocks the liquid return, not the heat rejection.** Discharge gas that is
not bypassed still flows into the condenser, condenses and simply stays there. The
condenser behaves as a receiver: the stand runs as a small refrigeration cycle that moves
the charge from the tank into the condenser, and the cooling water loop keeps holding the
liquid (intermediate) pressure.

**Pressures would climb once the condenser floods**, because flooding removes condensing
area. With `V_i` = 30 L the intermediate section (header, condenser, liquid line) holds about
31 kg of liquid at 18 bar, nearly twice the whole 17.4 kg charge, so it cannot flood in this
configuration. If the real header, condenser and liquid line hold only a few litres, the real
stand floods much sooner and its head pressure rises as expected. `V_i` should be checked
against the hardware.

**The volumes also decide how much of the charge is liquid at all.** At a 25 degC standstill,
165 L of total volume at 16.5 bar holds 10.9 kg of the 17.4 kg charge as vapor, leaving 6.9 kg
as liquid.

**Constant suction pressure and temperature while liquid remains** is what the model is
designed to do: a saturated pool at the pressure held by the suction pressure loop, measured
superheat 0.

## 5. Main limitation: the real vessel is not a mixing tank

The real suction mixer/accumulator is a finned-coil heat exchanger inside a tank:

* The hot-gas bypass stream flows **through the coil**; it never mixes with the tank contents
  directly.
* The liquid from valve 3 is brought in at the **top of the tank** and falls onto the coil.
* Liquid that does not evaporate settles at the bottom of the tank. It leaves only by
  **spilling over once it reaches a certain level**; below that level only vapor exits.
* The tank's outlet (vapor, plus liquid once it spills over) and the coil outlet (bypass gas)
  join at a **simple tee**, which feeds the compressor suction port directly.

So thermal and mechanical mixing are far from perfect. Differences from the model:

| Aspect | Model | Real vessel |
|---|---|---|
| Bypass gas -> liquid heat transfer | Instantaneous, all superheat used | Limited by the coil's UA (fins wetted by falling liquid, or dipping into the pool) |
| Bypass gas leaving | Mixed into the tank | Leaves the coil still superheated, joins at the tee |
| Suction state with liquid in the tank | Saturated vapor, superheat 0 | Mix of warm coil gas and saturated tank vapor at the tee: superheated even with liquid in the tank |
| Boil-off rate | Upper bound (energy limit) | Slower: `Q_coil / h_fg`, with `Q_coil <= eps * C_gas * (T_gas - T_sat)` |
| Liquid leaving the tank | Smooth carry-over from 50 % fill | Weir-like: none below the spill level, then liquid goes straight to the suction |
| Effect of injected liquid on suction temperature | Immediate (mixes into the tank) | Through evaporation on the coil; liquid that reaches the pool acts only after it boils or spills |

Implications worth keeping in mind:

* With liquid in the tank the model reports zero suction superheat, whereas the real stand can
  show superheat from the coil gas at the tee. Signals built on measured superheat (the
  superheat loop, the training environment's observations) are therefore more pessimistic in
  the model while the tank holds liquid below the spill level. The floodback alarm itself
  (`x_out < 1`, liquid actually reaching the compressor) fires only on carry-over, which
  roughly corresponds to the real spill-over.
* The suction temperature loop's dynamics differ. In the model, liquid injection acts on the
  suction state at once. In the real vessel, liquid that evaporates on the coil acts quickly,
  liquid that falls into the pool acts only as the pool boils, and above the spill level it
  goes straight to the compressor. Expect a slower, more nonlinear response on the real stand,
  and a sharper transition to liquid carry-over.
* Boil-off of accumulated liquid (for example after a cold start with liquid in the tank) is
  faster in the model than on the real stand.

## 6. If the vessel is modelled in more detail later

A representation closer to the hardware would split the current single volume into:

1. **Coil gas path**: the bypass stream through a heat exchanger (effectiveness-NTU) against
   the tank contents at saturation, with a coil UA that depends on wetting: liquid film from
   the injection, immersion in the pool, or dry.
2. **Tank volume** (liquid and vapor): receives the injected liquid and the coil heat, and
   discharges vapor plus liquid above a spill level (a weir fill fraction instead of the smooth
   `acc_carry_*` ramp).
3. **Tee**: adiabatic mixing of the coil outlet gas and the tank outlet into the compressor
   inlet state, which the suction temperature sensor sees.

Parameters it would need: coil UA (wet and dry, or per unit wetted area), the spill level,
the tank volume and the coil's internal volume.

## 7. Open questions / data that would pin this down

* Real refrigerant-side volumes: header + condenser + liquid line (`V_i`) and the
  mixer/accumulator (`V_s`, tank and coil separately).
* Coil geometry or a measured coil duty (for the coil UA), and the spill level.
* Whether the tank has any bleed or oil-return path that meters liquid into the suction line
  below the spill level.
* One timed boil-off on the real stand (known liquid mass, speed, bypass valve position) would
  show directly how far the real rate is below the model's energy limit.
