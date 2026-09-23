# Nominal refrigerant charge: how it is calculated and what it means

2026-09-22. Reference note on `nominal_charge()` (`hgbp_sim/params.py`).

## 1. What it is

The simulator does not compute an *optimal* charge. It computes a **nominal reference
charge**: the refrigerant mass the stand holds at one fixed, typical operating condition
with the condenser partly filled with liquid. It is used in three places:

* **Plant parameter `charge`.** Left empty (`null` in the defaults file, blank on the
  Settings tab), the stand is charged with the nominal charge for its volumes. The Settings
  tab, the operator tab's charge tile and the schematic show the nominal value next to the
  actual charge.
* **Training environment.** Episodes are charged with a random factor times the nominal
  charge: 0.85-1.15 normally (`EnvConfig.charge_range`), and 0.3-1.8 in 15 % of episodes
  (`charge_extreme_range`, `p_charge_extreme`). The factor is reported as the
  `charge_factor` label for charge estimation.
* **Parameter randomization.** When volumes are re-randomized, the charge of those
  environments is reset to the nominal charge of the new volumes.

## 2. Calculation

The mass in each of the three lumped volumes is added up at a fixed reference condition:

| Volume | State assumed |
|---|---|
| Suction `V_s` (tank + suction line) | vapor at the saturation pressure of -10 degC, 10 K superheat |
| Discharge `V_d` (compressor port to valve 1) | vapor at the intermediate pressure + 2.5 bar, 75 degC (40 + 35 K) |
| Intermediate `V_i` (header + condenser + liquid line) | saturated at 40 degC, 40 % of the volume liquid, 60 % vapor |

    charge = rho_s * V_s + rho_d * V_d + V_i * (0.4 * rho_liquid + 0.6 * rho_vapor)

All densities come from the refrigerant property tables, so the result follows the
refrigerant and the volumes. The liquid term dominates: in practice the nominal charge is
"enough to leave the condenser 40 % full of liquid", plus the vapor in the rest of the stand.

The reference condition is set by the function's keyword arguments (`T_evap=263.15 K`,
`T_int=313.15 K`, `SH=10 K`, `fill=0.4`); the discharge offset (2.5 bar) and the 35 K
discharge superheat are fixed inside the function.

## 3. Relation to the stand's own operating point

The reference condition dates from the original R134a medium-temperature configuration and
is not tied to the stand's rating point:

| | Nominal-charge reference | Stand rating point (loop setpoints) |
|---|---|---|
| Evaporating (suction saturation) | -10 degC | about 7 degC (9.98 bar) |
| Intermediate (condensing) | 40 degC | about 28 degC (18.00 bar) |
| Discharge pressure | intermediate + 2.5 bar | intermediate + 15.9 bar (33.89 bar) |

Because the liquid term dominates, the effect is modest: at the rating point the nominal
charge gives a condenser liquid fill of about 0.32 instead of the intended 0.40.

The actual liquid inventory at any operating point follows from mass conservation: whatever
the suction and discharge volumes do not hold as vapor ends up as liquid in the condenser
(and in the suction tank, if it floods). A different operating point with the same charge
therefore shows a different condenser fill and, through the subcooled zone and the
condensing area, different subcooling and condensing capacity.

## 4. Options if "nominal" should mean "correct for this stand"

* Enter the known real charge in `charge` (defaults file or Settings tab). This is the most
  direct way to represent the real stand.
* Compute the nominal charge at the stand's rating point (the loop setpoints in the defaults
  file) instead of the fixed reference condition, possibly with the target condenser fill as
  a setting.

## 5. Volumes matter as much as the condition

Because every volume contributes, the volume parameters decide how the charge splits between
vapor and liquid. For example, with `V_s` 120 L, `V_d` 15 L and `V_i` 30 L (the defaults file
at the time of writing), the stand at a 25 degC standstill holds 10.9 kg of its 17.4 kg charge
as vapor, and the intermediate section alone could hold about 31 kg of liquid, more than the
whole charge (see `SUCTION_MIXER_ANALYSIS.md`, section 4). Checking `V_s`, `V_d` and `V_i`
against the hardware is therefore as important as the charge itself.
