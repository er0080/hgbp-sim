# Pressure drops: plate heat exchangers and refrigerant lines

How the model treats the four sides of the two brazed plate heat exchangers (condenser and
mixing exchanger, both Alfa Laval ACH-70X-78M-F) and the refrigerant lines as flow
resistances, where the default coefficients come from, and what they change.

## 1. Model

Each side is a quasi-steady resistance sized like a valve, for plates and ports together:

```math
\Delta P_\mathrm{friction} = \frac{1}{\rho_m} \left( \frac{\dot m}{C} \right)^2, \qquad C = \frac{K_v}{36000}
```

$`\rho_m`$ is the side's mean density. For two-phase sides it is the homogeneous mixture density,
averaged over the specific volume along the side. Refrigerant sides also carry the static head of
their column over the port height `H` (`cond_H`, `mx_H`, 0.466 m). The cooling water circuit is
closed, so its static heads cancel and are not modelled.

The exchanger volumes hold only a few litres, so a pressure difference across them settles within
milliseconds. The drops are therefore algebraic: no pressure states are added, and the integration step
is unchanged.

| Side | Flow | Treatment |
|---|---|---|
| Condenser, refrigerant (header -> drain, downward) | condensing flow `Q_r / (h_d - h_l)` | The liquid pressure `P_i` is measured after the receiver, ahead of valve 3, and taken as the condenser outlet pressure. The hot gas header (valve 1 outlet, valve 2 inlet) sits at $`P_h = P_i + \Delta P_\mathrm{friction} - g H \rho_\mathrm{col}`$. `rho_col` weights liquid (flooded area), the homogeneous column mean (condensing area) and vapor (the rest). Friction acts on the non-flooded share. |
| Condenser, water | `mdot_w` | Plant loop at 1.5 barg supply and 0.35 barg return (`P_w_sup`, `P_w_ret`). The 1.15 bar difference drives the water through valve 4, the condenser and the piping (`Kv_wpipe`) in series: $`1/C^2 = 1/C_\mathrm{valve}^2 + 1/C_\mathrm{hx}^2 + 1/C_\mathrm{pipe}^2`$. |
| Mixing exchanger, gas (S1 bottom -> S2 top, upward) | `mdot_2` | Valve 2 discharges into the gas side in series (below). S1 sits above S2 by the friction drop plus the head of the rising gas column. Gas density is the mean of the bypass gas at `P_s` and saturated vapor. |
| Mixing exchanger, quench (S3 top -> S4 bottom, downward) | `mdot_3` | Computed from the plate geometry, not a Kv (section 3). In flow order: distributor, S3 connection and port; then each of the five cells with its own friction (two-phase or single-phase by its quality), acceleration and static head; then the S4 port and connection. Each cell boils at its own pressure: its saturation temperature is shifted by `dT/dP` (Clausius-Clapeyron at `P_s`) times its offset. The distributor and inlet port are ahead of the channels and do not raise the boiling pressure. The cells' mass and energy stay at `P_s`, so conservation is unchanged. |

**Valves in series with a side (and with their lines, section 4).** The pair shares the pressure difference in the ratio of
their resistances. The valve takes the share

```math
\frac{1}{1 + \left( C_\mathrm{valve} / C_\mathrm{hx} \right)^2 \rho_\mathrm{up} / \rho_\mathrm{hx}}
```

of the difference. It chokes
on its own share, so a choked valve passes the same flow with or without the exchanger behind it. For
valve 2 the expansion factor `Y` is evaluated at the valve's own share of the pressure ratio (two fixed-point
passes). Closed-form, no iteration over the plant state. The quench side's drop is not quadratic
in the flow. The valve 3 pair therefore uses the side's equivalent resistance at the flow valve 3 would pass on
its own. Valve 3 takes about 99 % of the difference, so the resulting flow error is well under 1 %.

**Where the effect shows.**
* Valve 1 works against the header pressure, slightly above `P_i`, through the discharge line and header.
* Valve 2's flow at small pressure differences is limited, for example at start-up or with an equalized stand.
* The quench boils warmer at the top of the exchanger.
* The water valve opens further for the same water flow.

At the rating point the drops are:

| Side | Rating flow | Drop |
|---|---|---|
| Condenser, refrigerant | 0.13 kg/s | +0.5 kPa net (about 2 kPa friction less 1.4 kPa head) |
| Condenser, water | 0.60 kg/s | 1.1 kPa |
| Mixing exchanger, gas | 0.52 kg/s | 20 kPa |
| Mixing exchanger, quench | 0.13 kg/s | 10.9 kPa: distributor 6.7, ports and connections 2.4 (S3 0.2, S4 2.2), channels 1.8 net of 0.25 static gain |

Together with the suction line and the outlet leg (section 4), the quench cells sit 13.6-15.1 kPa above
the compressor suction port, where `P_s` is measured. They boil about 0.5 K warmer than the suction
saturation temperature the stand displays.

**Why some of these are small.** Both exchangers are large for the flows they carry. The ACH-70X-78
is rated about 18 TR (63 kW), and the condenser runs at 31 kW. The cooling water is 0.6 kg/s, with a
12.5 K rise, over 39 channels: about 55 kg/h per channel. Alfa Laval's design examples run at
200-2000 kg/h per channel (section 2). Condensing R410A reaches only about 19 kg/(m2 s) in the channels.
The gas side is the exception: 0.52 kg/s of vapor at 10 bar reaches about 21 m/s in the ports. On the
quench side most of the drop is in the distributor and the 7/8 in connections, not in the channels.

At the rating point the water valve sits at about 53 %. Fully open it passes about 2.3 kg/s: the
1.15 bar loop difference across valve, condenser and piping, mostly taken by the piping placeholder
(`Kv_wpipe`) and the valve itself.

## 2. Default coefficients: estimate from the geometry

No selection data is available for these units, so the Kv values are estimates. They use the plate
geometry from the Alfa Laval AC70X product leaflet (CHE00004EN 2016-09) and the product page for the
ACH-70X-78M:

| Quantity | Value | Source |
|---|---|---|
| Plate size | 111 x 526 mm | leaflet |
| Port centres | 466 mm vertical, 50 mm horizontal | leaflet |
| Plate pitch | 2.3 mm (A = 11 + 2.3 n) | leaflet |
| Channel gap `b` | 2.0 mm (pitch less about 0.3 mm plate) | assumption |
| Flow width `W` | 0.090 m (0.095 L per channel / b / height) | derived |
| Hydraulic diameter | 3.4 mm (2 b / 1.17 area enlargement) | assumption |
| Channel type | M (mixed: one high-angle H plate, one low-angle L plate) | model designation "78M" |
| Chevron angle | 45 deg to the flow (mean of H about 60-65 deg and L about 25-30 deg) | assumption |
| Port bore | 31.5 mm (14 m3/h at 5 m/s connection velocity) | leaflet |
| S3/S4 connections | 7/8 in, about 20 mm bore | product page |
| Channels | 39 on S1-S2, 38 on S3-S4 | drawing |

**Single-phase channel friction** (water, bypass gas) follows the Martin (VDI Heat Atlas) correlation
for chevron plates at 45 deg. **Port loss** is 1.5 velocity heads at the port bore. The S3/S4 sides add
about 1.5 velocity heads at the 7/8 in connection bore. These are split between inlet (dense) and outlet
(vapor) on the two-phase sides.

**Cross-check against Alfa Laval data.** Alfa Laval's refrigeration reference manual tabulates measured
water-side drops per channel for CB76 plates: M channels give 2.14 / 9.75 / 34.7 / 129 kPa at
203.5 / 500 / 1000 / 2000 kg/h per channel. Scaled to the AC70X, whose channel is narrower (about 1.8
times the mass flux) and slightly shorter, that gives 0.5 / 3.6 / 5.5 kPa at 0.6 / 1.75 / 2.2 kg/s. Martin
at 45 deg gives 0.7 / 4.2 / 6.3 kPa, a good match (60 deg, H-channel behaviour, would give about twice as
much).

**Condensing refrigerant.** At about 19 kg/(m2 s) the channel friction is small, even with a two-phase
correlation (Yan, Lio & Lin 1999 gives less than the homogeneous estimate). The side is dominated by the
7/8 in connections. Alfa Laval states that its distributor, if it sits in the condensate outlet, takes
under 0.1 bar at condensing.

| Side | Estimated at | Drop | Kv (m3/h) | Default |
|---|---|---|---|---|
| Condenser, water | 0.60 / 1.75 / 2.2 kg/s | 1.2 / 8.0 / 12.3 kPa (channels + ports) | 20 / 22 / 22.5 | `cond_Kv_w` = 21 |
| Condenser, refrigerant | 0.13 kg/s, mean quality 0.5 at 18 bar | 1.5 kPa | 10.3 | `cond_Kv_r` = 10.5 |
| Mixing exchanger, gas | 0.52 kg/s vapor, 10 bar, 40 degC | 20 kPa (9 channels + 11 ports) | 23.6 | `mx_Kv_g` = 23.5 |

A Kv implies a drop that rises with the square of the flow, which holds well for these three sides. The
quench side is modelled from the geometry instead (section 3).

## 3. Quench side: model from the plate geometry

The quench side cannot be measured on the stand, so it is computed from the geometry with the
best-validated published methods. A single Kv cannot capture it. Its drop depends on how far the quench
evaporates in each cell (dry-out moves with the operating point), its flow dependence is far from quadratic,
and only the part inside the channels shifts the boiling temperature.

**Channels, evaporating: Amalfi, Vakili-Farahani & Thome (2016).**
* The local two-phase Fanning friction factor and the pressure drop over a length $`dz`$ are

```math
f = C \cdot 15.698 \, \mathrm{We}_m^{-0.475} \, \mathrm{Bd}^{0.255} \left( \frac{\rho_l}{\rho_v} \right)^{-0.571},
\qquad C = 2.125 \left( \frac{\beta}{70^\circ} \right)^{9.993} + 0.955,
\qquad dP = \frac{2 f \, G^2}{d_h \, \rho_m} \, dz
```

  with the homogeneous Weber number $`\mathrm{We}_m = G^2 d_h / (\rho_m \sigma)`$, the Bond number
  $`\mathrm{Bd} = (\rho_l - \rho_v) g d_h^2 / \sigma`$, mass flux $`G`$, hydraulic diameter $`d_h`$ and
  corrugation angle $`\beta`$.
* It was fitted to 1513 frictional pressure drop points from 13 independent studies of plate exchangers,
  covering many fluids (including ammonia) and plate designs. The R410A data of Hsieh & Lin are among the
  studies it draws on. It predicts 74 % of the points within +-30 % and 91 % within +-50 %, and it worked
  better than any of the 29 published methods it was compared with.
* At the rating point the quench side is inside its ranges: `We_m` 1.1-4.1 (database 0.03-150), `Bd` 16
  (2.4-49), density ratio 30 (19-1350), `beta` 45 deg (30-65).

**Channels, single-phase** (vapor after dry-out, liquid if the quench floods): Martin's correlation. The
two blend over a quality band of +-0.025 at each end of the dome.

**Acceleration** uses the homogeneous model, per cell. It is small at these mass fluxes.

**Static head** uses each cell's density. Downflow gains pressure towards S4.

**Ports:** 0.75 velocity heads at each end (Shah & Focke, as in Amalfi's reduction of their own data).

**7/8 in connections:** sudden expansion into the port at S3, and contraction into the connection plus
expansion into the 1-3/8 in outlet pipe at S4, from the bores (`mx_d_port`, `mx_d_S34`, `D_mo`/`t_mo`).
The S4 outlet carries vapor at about 13 m/s, so it takes about 2 kPa at the rating flow.

**Distributor (S3):** an orifice, `mx_Kv_dist`, evaluated at the density of the valve 3 outlet mixture.
* SWEP puts the drop of BPHE evaporator distribution systems at normally 0.5-2 bar and recommends more than
  1 bar for good distribution. That drop is ahead of the heat transfer area, so it does not change the
  evaporation temperature.
* Alfa Laval sizes the nozzle in the S3 connection to the refrigerant and operating conditions, and states
  that its drop is negligible when the unit condenses.
* The default assumes 1 bar at the unit's design duty: 18 TR is about 63 kW, or about 0.4 kg/s of R410A
  at typical chiller conditions. That gives `mx_Kv_dist` = 4.0 m3/h.
* The stand's quench flow is about a third of that, so the distributor takes about 7 kPa.
* What "N60" on the drawing encodes is not known. If it is a nozzle size or capacity class, it would pin
  this value down.

**Uncertainty at the rating point:**

| Assumption | Total S3 -> tee | Channels | Top-cell boiling shift |
|---|---|---|---|
| Default (distributor 1 bar at design, 45 deg) | 10.7 kPa | 1.5 kPa | +0.14 K |
| Distributor 0.5 / 2 bar at design | 7.3 / 16.8 kPa | 1.5 kPa | +0.14 K |
| Corrugation angle 60 / 65 deg | 11.7 / 12.8 kPa | 2.6 / 3.7 kPa | +0.18 / +0.21 K |
| Amalfi +-50 % (91 % of the database) | 10-11.5 kPa | 0.8-2.3 kPa | +0.1-0.2 K |

The total range is set almost entirely by the distributor, and it only matters for valve 3's capacity at
small pressure differences. The part that acts on the physics, the boiling pressure in the channels, is
bounded to about 1-4 kPa, or 0.1-0.2 K. At the rating point valve 3's flow changes by under 0.5 % across
the whole table.

**Other correlations.** For the same conditions, Hsieh & Lin (2002, R410A, one 60 deg plate) gives about
3 times Amalfi's channel friction, and Yan & Lin (1999, R134a) about 10 times. Each was fitted to a single
plate. Amalfi's comparison found that most such correlations should not be used outside their original
test conditions.

**Downward evaporation.** All of these methods come from upward flow. Alfa Laval notes that downward
evaporation in BPHEs has been little studied, and that it needs a comparably high channel drop and a low
nozzle drop to distribute the liquid. The quench side here flows downward, so its distribution, and hence
its effective heat transfer, is a larger uncertainty than its pressure drop.

`Kv_wpipe` = 12 m3/h (the plant piping, fittings and strainer) is a placeholder, with no data behind it.
It takes about 0.03 bar at the rating water flow and 0.45 bar at 2.2 kg/s.

The Kv values do not follow `cond_n_plates` / `mx_n_plates` automatically. Changing the plate count
means re-estimating them.

## 4. Refrigerant lines

Every copper line from the piping specification (`D_*`, `t_*`, `L_*`) is a quasi-steady resistance, like the
exchanger sides: no pressure states, and the drop is taken by the flow the line actually carries.
* **Friction:** Churchill's (1977) Darcy friction factor for all regimes, drawn copper (1.5 um roughness).
  It reduces to 64/Re when laminar and gives the smooth-pipe values when turbulent.
* **Fittings:** `K_*` velocity heads, about 0.3 per long-radius bend, 1.0 for a tee branch and 0.4 for a
  run through a tee. These are estimates from a typical routing, not the actual one.
* **Two-phase lines** (quench line, a wet suction line) are homogeneous: mixture density and McAdams
  viscosity.
* **Static heads** between components are not modelled, because the elevations are not known.

Pressure chain:
* **Suction side.** `P_s` is the compressor suction port, where the suction pressure is measured. The
  suction line (compressor flow) puts the tee above it. The two exchanger outlet legs (half of `L_mo`
  each, `K_mo` per leg) put S2 and S4 above the tee.
* **Discharge side.** `P_d` is the compressor discharge. Valve 1 sees it through the discharge line
  (valve 1 flow).
* **Header.** The valve 2 branch is taken halfway along the header: the first half carries valve 1's flow,
  the second half only the condensing flow. Valve 1 discharges through the first half. The branch pressure
  is the header pressure `P_h` on the schematic.
* **Liquid side.** `P_i` is measured after the receiver. The drain (condensing flow) puts the condenser
  outlet above it. Valve 3 draws through the liquid line and discharges through the quench line.

Each valve is solved in series with everything its flow passes through, with the equivalent resistance
evaluated at the valve-alone flow:
* valve 1: discharge line, first half of the header;
* valve 2: bypass line, gas side, gas outlet leg;
* valve 3: liquid line, quench line, quench side, quench outlet leg.

At the rating point:

| Line | Size, length | Flow | Velocity | Drop |
|---|---|---|---|---|
| Discharge (compressor -> valve 1) | 1-1/8 in, 4 m | 0.65 kg/s hot gas, 34 bar | ~12 m/s | 22 kPa |
| Header (valve 1 -> branch -> condenser) | 1-1/8 in, 4 m | 0.65 / 0.13 kg/s, 18 bar | ~23 m/s first half | 25 kPa |
| Bypass (valve 2 -> S1) | 1-1/8 in, 0.5 m | 0.52 kg/s gas, 10 bar | ~35 m/s | 26 kPa |
| Gas outlet leg (S2 -> tee) | 1-3/8 in, 0.5 m | 0.52 kg/s | ~17 m/s | 4.5 kPa |
| Suction (tee -> compressor) | 1-5/8 in, 4 m | 0.65 kg/s vapor | ~16 m/s | 11 kPa |
| Quench outlet leg (S4 -> tee) | 1-3/8 in, 0.5 m | 0.13 kg/s | ~5 m/s | 0.4 kPa |
| Quench (valve 3 -> S3) | 7/8 in, 0.5 m | 0.13 kg/s flashing | ~2 m/s | 0.4 kPa |
| Liquid (receiver -> valve 3) | 7/8 in, 3 m | 0.13 kg/s liquid | ~0.4 m/s | 0.4 kPa |
| Drain (condenser -> receiver) | 7/8 in, 0.5 m | 0.13 kg/s | | 0.1 kPa |

The gas lines are the large ones: hot gas at 12-35 m/s in 1-1/8 in tube. Valves 1-3 each take several
bar, so the valve positions barely change. The measurable consequences are these:
* **Mixing exchanger pressure.** The exchanger runs about 15 kPa above the measured suction pressure,
  so the quench boils about 0.5 K warmer than the displayed suction saturation temperature.
* **Header pressure.** The header sits above `P_i` by less than 2 kPa, but valve 1's outlet sits about
  25 kPa above the branch.

A filter-drier, sight glass or solenoid valve in the liquid line would add to `K_liq`, typically
0.1-0.3 bar for a filter-drier.

## 5. Calibrating against the real stand

* **Condenser water side.** Measure the water flow and the pressure difference across the condenser water
  connections. $`K_v = 3.6 \, \dot m / \sqrt{\rho \, \Delta P / 1000}`$ (flow in kg/s, drop in bar) gives `cond_Kv_w` in
  the model's convention ($`K_v = Q / \sqrt{\Delta P}`$ for water, Q in m³/h). Measure the supply-to-return difference at full valve
  opening to fit `Kv_wpipe`.
* **Mixing exchanger gas side.** Measure the pressure at the valve 2 outlet (S1) against suction
  pressure, at a known bypass flow (from the suction mass flow and the quench flow).
* **Mixing exchanger quench side.** Not measurable on the stand. The model computes it from the geometry
  (section 3). If an Alfa Laval selection or the meaning of "N60" becomes available, set `mx_Kv_dist`
  from the distributor's rated drop. A pressure tap at S3, if one is ever added, would check the total.
* **Condenser refrigerant side.** Its drop is small next to the sensors' noise (6 kPa standard deviation on
  the intermediate pressure). Only a differential transmitter between the header and the liquid line
  would resolve it.

Selection printouts from Alfa Laval (CAS / Hexact) for these two units would give each side's drop
at a design point directly. Divide the flow by the square root of the drop and density ratio to get the
Kv.

## 6. Limitations

* No maldistribution between channels. On the refrigerant sides two-phase friction enters only through
  the Kv, except on the quench side, which uses the geometry model.
* The quench distributor's rated drop is an assumption (1 bar at an estimated design flow). It sets most
  of the quench side's total, but not the boiling pressure.
* The correlations were developed for upward evaporation; the quench side flows downward.
* Line fittings (`K_*`) are estimates from a typical routing; line static heads are not modelled.
* The condenser's condensing temperature uses the measured `P_i`. The mean condenser pressure differs by
  under 0.1 K of saturation temperature at the default Kv.
* Only the quench cells' boiling temperature follows the local pressure. The cells' densities and the
  suction side's mass and energy use the common `P_s`.
* Static heads between components, such as the condenser outlet to the receiver and the P_i transmitter,
  are not modelled: `P_i` is taken at the condenser outlet level.

## Sources

* [Alfa Laval AC70X / ACH70X / ACP70X product leaflet](https://www.alfalaval.com/globalassets/documents/microsites/heating-and-cooling-hub/pd-leaflets/ac70_product-leaflet.pdf)
* [Alfa Laval ACH-70X-78M product page](https://shop.alfalaval.com/en-us/brazed-plate-heat-exchanger-cross-reference-tool--ru_bhe/ach-70x-78m--1259553-)
* R.L. Amalfi, F. Vakili-Farahani, J.R. Thome, "Flow boiling and frictional pressure gradients in plate heat
  exchangers. Part 1: Review and experimental database; Part 2: Comparison of literature methods to
  database and new prediction methods", Int. J. Refrigeration 61 (2016) 166-184 and 185-203; and
  [R.L. Amalfi, PhD thesis EPFL no. 6856 (2016)](https://infoscience.epfl.ch/record/214760/files/EPFL_TH6856.pdf)
  (equations 3.10-3.14, 7.7-7.8)
* SWEP refrigerant handbook, [6.8 Brazed plate heat exchanger evaporators](https://www.swep.net/refrigerant-handbook/6.-evaporators/asas3/)
  and [4.2 Thermal expansion valves](https://www.swep.net/refrigerant-handbook/4.-expansion-valves/adf6/)
  (distribution system pressure drop)
* [Alfa Laval heating and cooling FAQ](https://www.alfalaval.us/industries/hvac/hvac-consultant-portal/faq/)
  (S3 connection as pre-distributor, evaporators used as condensers)
* R.K. Shah, W.W. Focke, "Plate heat exchangers and their design theory", in Heat Transfer Equipment
  Design (1988) (port pressure drop)
* [Alfa Laval, Plate heat exchangers for refrigeration applications, technical reference manual](https://www.alfalaval.com/globalassets/documents/microsites/heating-and-cooling-hub/technical-reference-manual-refrigeration.pdf)
  (CB76 L / M / H channel pressure drop tables; distributors)
* Y.Y. Hsieh, T.F. Lin, "Saturated flow boiling heat transfer and pressure drop of refrigerant R-410A in a
  vertical plate heat exchanger", Int. J. Heat Mass Transfer 45 (2002)
* Y.Y. Yan, T.F. Lin, "Evaporation heat transfer and pressure drop of refrigerant R-134a in a plate heat
  exchanger", J. Heat Transfer 121 (1999); Y.Y. Yan, H.C. Lio, T.F. Lin, "Condensation heat transfer and
  pressure drop of refrigerant R-134a in a plate heat exchanger", Int. J. Heat Mass Transfer 42 (1999)
* H. Martin, "A theoretical approach to predict the performance of chevron-type plate heat exchangers",
  Chem. Eng. Process. 35 (1996); VDI Heat Atlas, chapter on plate heat exchangers.
