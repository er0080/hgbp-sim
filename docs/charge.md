# Refrigerant charge

`PlantParams.charge` is the stand's total refrigerant mass [kg]. Left empty (`None`), the
stand is charged with the **nominal charge**, `nominal_charge()` in `hgbp_sim/params.py`.
The charge is conserved exactly while the stand runs; charging and recovery (web UI) change
it at a set rate.

## 1. The nominal charge

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
| Intermediate section | saturated at 40 degC: liquid line full, receiver filled to `level` (40 %), the rest (condenser, drain, header, receiver vapor space) vapor; the condenser's running hold-up is not included |

```math
m = \rho_s V_s + \rho_d V_d + \rho_l V_\mathrm{liq} + \rho_v \left( V_i - V_\mathrm{liq} \right),
\qquad V_\mathrm{liq} = \ell \, V_\mathrm{receiver} + V_\mathrm{liquid\,line}
```

with $`\ell`$ the receiver level (`level`) and $`\rho_l`$, $`\rho_v`$ the saturated liquid and
vapor densities in the intermediate section.

All densities come from the refrigerant property tables, so the result follows the
refrigerant, the piping and the receiver. With the defaults (R410A, 26.5 L receiver) it
is 14.2 kg: 10.4 kg of liquid in the receiver, 0.9 kg in the liquid line, and 2.9 kg of
vapor in the whole stand.

The reference condition is set by the function's keyword arguments (`T_evap=263.15 K`,
`T_int=313.15 K`, `SH=10 K`, `level=0.4`).

## 3. Relation to the stand's own operating point

The reference condition is not tied to the stand's rating point (7 degC evaporating,
28 degC intermediate). At the rating point the nominal charge gives a receiver level of
about 35 % instead of 40 %. The difference is mostly the condenser's own liquid, about
0.6 kg: the condensing film and the condensate draining to the receiver.

The actual liquid inventory at any operating point follows from mass conservation. Whatever
the suction side, the discharge volume, the vapor space and the condenser's hold-up do not
hold ends up as liquid in the receiver, and in the condenser once the receiver is full.
Different operating points with the same charge therefore show different receiver levels.

The receiver pool keeps its own temperature, so its density, and with it the level, also
follows the subcooling: a subcooled pool is denser and holds more. The condensing area and
the steady subcooling stay the same as long as the receiver neither overflows nor drops
below its dip tube.

## 4. Charge window

With the defaults at the rating point:

* **Undercharge:** below about 0.42 times the nominal charge the receiver level reaches
  the dip tube inlet. Vapor enters the liquid line, valve 3 passes flashing two-phase
  fluid, and the suction temperature loop loses authority.
* **Normal range:** between about 0.45 and 2.2 times nominal only the receiver level
  changes (5 % to 100 %). The subcooling stays at about 3 K.
* **Overcharge:** above about 2.2 times the nominal charge the receiver is full (about
  29 kg of liquid). The drain backs up and liquid floods the condenser from the bottom,
  removing condensing area. The water valve opens further to hold the intermediate
  pressure. The condensate draining through the flooded plates subcools strongly: at 2.25,
  2.3 and 2.4 times nominal, 6, 13 and 30 % of the plates are flooded, with 4, 7 and 13 K
  of subcooling. The colder, denser pool then takes back some of the liquid. Further up,
  the water valve saturates and the discharge pressure trips.

The exact limits depend on the operating point, because the vapor inventory does.

## 5. Cold start

At a cold start (compressor off, stand equalized at ambient) the liquid sits in the
receiver; what it cannot hold stays in the condenser. A share `cold_liquid_in_suction` can be placed on the suction side instead
(compressor first, then suction line and mixing exchanger), as after refrigerant migration
during a long off cycle. A charge too small to reach saturation at ambient leaves the whole
stand with superheated vapor at a lower pressure.

## 6. Representing the real stand

Enter the stand's known charge in `charge` (defaults file or Settings tab). To make
"nominal" mean something else, change the reference condition of `nominal_charge()`
(section 2).
