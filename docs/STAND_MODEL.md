# Stand model: mixing exchanger, liquid receiver and specified piping

2026-09-24. This note describes the stand model as of branch `bphe-mixer-receiver`. It
replaces the suction mixer tank model reviewed in `SUCTION_MIXER_ANALYSIS.md`.

## 1. Hardware represented

```
compressor --> discharge line --> valve 1 --> hot gas header (P_i + condenser pressure drop)
                                                 |                         |
                                              valve 2              condenser (BPHE) <-- water, valve 4
                                                 |                         | drain
                                                 |                  receiver (UR66, dip tube) -> P_i sensor
                                                 |                         | liquid line
                                                 |                      valve 3
                                                 v                         v
                                   mixing exchanger (BPHE), S2 / S3 up
                            gas: S1 (bottom) -> S2 (top)   quench: S3 (top) -> S4 (bottom)
                                                 \__________ tee __________/
                                                             |
                                          suction line, 4 m -> probe -> compressor (5 L inside)
```

* **Condenser and mixing exchanger:** both are Alfa Laval ACH-70X-78M-F (drawing
  `3287083725`). There are 78 plates and 77 channels: 39 on S1-S2 and 38 on S3-S4, with
  0.095 L per channel, so 3.71 L and 3.61 L per side. Net weight is 16.19 kg.
  * The drawing states a "heating surface 6.588 ft²". That can only be m²: the weight
    formula gives 0.18 kg per plate, which at 0.25 mm of stainless steel is about
    0.087 m² per plate, and 76 × 0.087 = 6.6 m². The model uses 6.588 m².
  * Port use: the S3-S4 side (S3 with the 7/8 in N60 distributor) carries the refrigerant
    in the condenser and the quench in the mixing exchanger. S1-S2 carries the water and
    the bypass gas respectively.
* **Receiver:** Standard Refrigeration UR66, MP version. The pumpdown rating is 61 lb of
  R22 at 90 % of the shell volume at 90 °F, which gives about 26.5 L inside. That is
  about 27.6 kg of R410A liquid at 18 bar if completely full. The outlet is a dip tube
  whose inlet sits at `rec_dip` (4 %) of the volume.
* **Piping:** copper, ACR type L. Each line is a parameter triple (`D_*` outside
  diameter, `t_*` wall, `L_*` length):

| line | size | length | volume |
|---|---|---|---|
| discharge (compressor -> valve 1) | 1-1/8 in | 4 m | 2.13 L |
| hot gas header (valve 1 -> condenser, valve 2) | 1-1/8 in | 4 m | 2.13 L |
| bypass (valve 2 -> S1) | 1-1/8 in | 0.5 m (estimate) | 0.27 L |
| quench (valve 3 -> S3) | 7/8 in | 0.5 m (estimate) | 0.16 L |
| exchanger outlets (S2, S4 -> tee) | 1-3/8 in | 1 m (estimate) | 0.81 L |
| suction (tee -> compressor) | 1-5/8 in | 4 m | 4.59 L |
| condensate drain (condenser -> receiver) | 7/8 in | 0.5 m (estimate) | 0.16 L |
| liquid line (receiver -> valve 3) | 7/8 in | 3 m | 0.94 L |

* **Compressor:** 5 L internal suction volume, behind the suction port and the probe.
  The internal discharge volume of 1.5 L is an estimate. The compressor takes in at most
  30 % liquid by mass (`comp_x_min` = 0.7); more liquid collects in its shell, the
  internal suction volume, instead of being pumped at liquid density.

These give the lumped volumes (`hgbp_sim/geometry.py`, shown on the Settings tab under
Piping):

| section | volume | before |
|---|---|---|
| suction (exchanger, tee, suction line, compressor inside) | 18.1 L | 120 L tank |
| discharge (compressor inside + discharge line) | 3.6 L | 15 L |
| intermediate (header, condenser, drain, receiver, liquid line) | 33.3 L | 30 L |

Heat capacities are derived from the same data. The condenser holds its plates plus
water content (23.6 kJ/K). The exchanger plates hold 8.1 kJ/K, split over the cells.
The suction and discharge line copper hold 3.1 and 1.5 kJ/K. The receiver shell is
25 kg of steel (estimate), 12 kJ/K.

## 2. Model structure

### Intermediate section with receiver
The section is one (P, h) volume in thermodynamic equilibrium. Its liquid volume is
assigned to where the liquid actually sits, in this order:
1. the receiver up to the dip tube inlet;
2. the liquid line;
3. the rest of the receiver;
4. the drain;
5. the condenser plates;
6. the header.

Consequences:
* The condenser keeps its full condensing area until the receiver is full. Only then does
  liquid back up and flood the plates, which removes condensing area and grows a
  subcooled zone.
* Below the dip tube the liquid line fills with vapor and valve 3 loses its liquid seal
  (flag `no_liquid_seal`).
* The liquid leaving the receiver is saturated. The draining condensate is subcooled on a
  small share `cond_sc_film` (2 %) of the plate area at the water inlet end; this is a
  calibration knob for the subcooling reading at valve 3.
* A receiver shell temperature state exchanges heat with the refrigerant and with
  ambient. The receiver level is reported as a sight glass reading (`rec_level`).

### Suction side
The whole suction side shares one pressure P_s for mass and energy; P_s is the compressor suction port,
and the suction line, outlet legs and exchanger sides sit above it by their pressure drops. The quench cells boil at their own
pressure along the S3 -> S4 column, and valves 2 and 3 discharge through their exchanger sides in series
(`PRESSURE_DROP.md`). The cells are:
* five quench-side cells of the mixing exchanger (enthalpy states, top to bottom), each
  against its own plate-wall temperature;
* the tee plus suction line;
* the compressor's internal suction volume, from which the compressor draws.

The bypass gas side has a residence time of about 0.2 s. It is quasi-steady: marched
cell by cell from S1 up to S2 against the same walls. Flows between the cells follow from
the common pressure (lumped-pressure finite volumes). The rate dP_s/dt comes from the
compressor drawing its flow out of the last cell, and each face flow is then upwinded by
its actual direction.

Heat transfer per cell, as a coefficient per m² times the cell's share of the heating
surface:
* **Gas side:** `mx_alpha_g0` (500 W/m²K at 0.5 kg/s, scaling with flow^0.8).
* **Quench side, evaporating:** `mx_alpha_e` (1500 W/m²K at 0.13 kg/s, scaling with
  flow^0.5).
* **Quench side, vapor after dry-out:** `mx_alpha_v0` (250 W/m²K, scaling with
  flow^0.8).
* **Dry-out cell:** the evaporating share of its area is what the incoming liquid needs
  at the evaporating coefficient. The rest heats vapor against the wall, so the vapor
  cannot leave hotter than the plate. A cell that still holds liquid boils over its whole
  area, including at zero flow. The two cases blend over 15 kJ/kg above the dew point,
  so the dry-out point moves continuously through the cells.

### Tee and probe
The tee mixes the gas outlet (S2) and the quench outlet (S4). Liquid leaving S4 travels
as droplets that evaporate with the time constant `tee_tau_evap` (0.3 s) over the
suction line's residence time (about 0.3 s).
* The probe at the compressor reads the vapor temperature. While droplets are still in
  flight it can read superheat even though liquid is reaching the compressor.
* `y_liq` is the liquid mass share at the compressor port. Floodback is flagged above
  `y_flood` (0.5 %).
* In this exchanger the bypass gas leaves pinched close to saturation whenever the quench
  side runs wet. So in practice the tee mixture turns wet together with the quench outlet,
  and the probe then reads saturation.

### Conservation and numerics
* Mass and internal energy of the suction side (as one group), the discharge volume and
  the intermediate section are integrated as conserved states. After every sub-step:
  * the discharge and intermediate volumes are flashed back from density and energy;
  * the suction side's common pressure and a uniform enthalpy shift of its cells are
    corrected by Newton iteration.
  The charge is exact to machine precision.
* Step subdivision is automatic. The right-hand side estimates the fastest local rate:
  valve conductance over the capacitance of each pressure node, plus advection, heat
  transfer and dry-out switching in the quench cells. Each 0.05 s step is split until
  rate × sub-step ≤ 1.8 (up to 32 sub-steps).
  * This matters because the real discharge volume (3.6 L) with a wide-open valve 1 at a
    small pressure drop has millisecond dynamics.
* Sub-stepping is decided per environment: environments are grouped by the sub-steps
  they need (a power of two), so one stiff environment does not slow the whole batch.
  A sub-step that still produces a non-finite state or an implausible jump is rejected
  and redone with four times finer steps.
* Throughput: 10-20 times real time for a single live stand, and about 1 000 control
  steps per second for 1024 environments (the tank model ran about 4 500). The extra
  cost comes from the five exchanger cells, the gas-side march and the pressure
  projection.

### Steady-state solver
A single Newton problem over the whole stand is ill-conditioned here, because a
generously sized counterflow exchanger's outlet state is insensitive to where the quench
dries out. The solver therefore alternates two steps:
1. The stand with the suction side's overall mass and energy balance as its equations
   (Levenberg-Marquardt over valve positions, discharge enthalpy and wall temperatures).
2. The exchanger profile for the resulting flows: local pseudo-time stepping of its own
   cell and wall dynamics, then a Newton polish.

It repeats until the liquid held up on the suction side stops changing, typically 3-6
rounds (about 0.5 s per batch).

## 3. Behaviour at the rating point

The rating point is R410A, 3550 rpm, 9.98 / 33.89 / 18.0 bar, suction probe at 18.33 °C
(about 11 K superheat), 10 °C water, nominal charge 14.2 kg.

| quantity | value |
|---|---|
| compressor / bypass / quench flow | 0.648 / 0.519 / 0.129 kg/s |
| mixing exchanger duty | 30 kW |
| quench quality, cells top -> bottom | 0.22, 0.43, 0.96, 1.15, 1.23 (dries out in the middle cell) |
| gas outlet S2 / quench outlet S4 | 10 °C / 54 °C, mixing to 18.3 °C at the tee |
| liquid held up in the exchanger | 0.14 kg |
| receiver level | 38 % |
| subcooling at valve 3 | 3.1 K |

Dynamics:
* A +10 % step on valve 3 lowers the suction temperature over about a minute. The hot
  lower plates have to cool down, so the plate mass dominates the response (the old tank
  responded almost at once).
* At a 5 K superheat setpoint the same step wets the quench outlet after about 35 s, and
  floodback follows.
* The suction pressure responds far faster than with the 120 L tank.

Controllers:
* With the UT35A tuning in the defaults file (P 150 %, I 50 s on all loops) the stand
  starts from cold, holds its setpoints and follows suction temperature steps of ±4 to
  7 K without oscillation.
* The baseline controller's superheat loop was retuned for the slower response: 0.003 per
  K with an integral time of 60 s (previously 10 s).

Charge:
* From about 0.4× to 2.1× nominal, the receiver absorbs the difference. The water valve
  position, the condensing area and the subcooling (3.1 K) stay the same (receiver 5 % to
  99 %).
* At about 2.2× nominal the receiver is full and the condenser starts to flood. The water
  valve opens further and subcooling jumps to 17 K.
* Below about 0.4× the level falls to the dip tube and the liquid seal is lost.

## 4. Calibration knobs and open points

* Estimated inputs, worth checking on the hardware: the bypass, quench, outlet and drain
  line lengths, the compressor's internal discharge volume, the receiver shell weight,
  and the dip tube height.
* The exchanger coefficients (`mx_alpha_*`) are correlation estimates for the stated
  channel geometry. A measured valve 3 -> suction temperature step would pin
  `mx_alpha_e`, `mx_alpha_v0` and the plate heat capacity. The condenser coefficients
  keep the previous calibration (equal to the former UA values over 6.588 m²).
* `cond_sc_film` sets the subcooling seen at valve 3 when the condenser is not flooded.
* The receiver is in equilibrium with the vapor space. A stratified, subcooled pool
  would make the intermediate-pressure loop respond differently; this would be the next
  refinement if that loop's feel is off.
* The droplet time constant `tee_tau_evap` only matters when liquid leaves the exchanger
  while the gas outlet still carries superheat.
