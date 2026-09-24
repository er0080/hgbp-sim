# Nominal refrigerant charge: how it is calculated and what it means

2026-09-24 (updated for the receiver). Reference note on `nominal_charge()`
(`hgbp_sim/params.py`).

## 1. What it is

The simulator does not compute an *optimal* charge. It computes a **nominal reference
charge**: the refrigerant mass the stand holds at one fixed, typical operating condition
with the receiver partly filled. It is used in three places:

* **Plant parameter `charge`.** Left empty (`null` in the defaults file, blank on the
  Settings tab), the stand is charged with the nominal charge. The Settings tab, the
  operator tab's charge tile and the schematic show the nominal value next to the actual
  charge.
* **Training environment.** Episodes are charged with a random factor times the nominal
  charge: 0.85-1.15 normally (`EnvConfig.charge_range`), and 0.3-1.8 in 15 % of episodes
  (`charge_extreme_range`, `p_charge_extreme`). The factor is reported as the
  `charge_factor` label for charge estimation.
* **Parameter randomization.** When the piping or receiver is re-randomized, the charge
  of those environments is reset to the nominal charge of the new volumes.

## 2. Calculation

The mass in each volume is added up at a fixed reference condition:

| Volume | State assumed |
|---|---|
| Suction side (mixing exchanger, lines, compressor inside) | vapor at the saturation pressure of -10 degC, 10 K superheat |
| Discharge (compressor inside, discharge line) | vapor at the intermediate pressure + 2.5 bar, 75 degC |
| Intermediate section | saturated at 40 degC: liquid line full, receiver filled to `level` (40 %), the rest (condenser, drain, header, receiver vapor space) vapor |

    charge = rho_s V_s + rho_d V_d + V_liquid rho_liquid + (V_i - V_liquid) rho_vapor
    V_liquid = level * V_receiver + V_liquid_line

All densities come from the refrigerant property tables, so the result follows the
refrigerant, the piping and the receiver. With the defaults (R410A, 26.5 L receiver) it
is 14.2 kg: 10.4 kg of liquid in the receiver, 0.9 kg in the liquid line, and 2.9 kg of
vapor in the whole stand.

The reference condition is set by the function's keyword arguments (`T_evap=263.15 K`,
`T_int=313.15 K`, `SH=10 K`, `level=0.4`).

## 3. Relation to the stand's own operating point

The reference condition is not tied to the stand's rating point (7 degC evaporating,
28 degC intermediate). With the receiver this barely matters: at the rating point the
nominal charge gives a receiver level of about 38 % instead of 40 %.

The actual liquid inventory at any operating point follows from mass conservation:
whatever the suction side, the discharge volume and the vapor space do not hold ends up
as liquid in the receiver, and in the condenser once the receiver is full. Different
operating points with the same charge therefore show different receiver levels. The
condensing area and the subcooling stay the same as long as the receiver neither
overflows nor drops below its dip tube.

## 4. Charge window

With the defaults at the rating point:

* **Undercharge:** at about 0.4 times the nominal charge the receiver level reaches the
  dip tube inlet. Vapor enters the liquid line, valve 3 passes flashing two-phase
  fluid, and the suction temperature loop loses authority.
* **Normal range:** between about 0.4 and 2.1 times nominal only the receiver level
  changes (5 % to 99 %).
* **Overcharge:** above about 2.1 times the nominal charge the receiver is full (about
  27.6 kg of liquid). Liquid backs up into the condenser, removes condensing area and
  subcools the liquid. The water valve then opens further to hold the intermediate
  pressure, and eventually the discharge pressure trips.

The exact limits depend on the operating point, because the vapor inventory does.

## 5. Options if "nominal" should mean "correct for this stand"

* Enter the known real charge in `charge` (defaults file or Settings tab). This is the most
  direct way to represent the real stand.
* Compute the nominal charge at the stand's rating point instead of the fixed reference
  condition, or change the reference `level`.
