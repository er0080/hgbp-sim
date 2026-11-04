# Suction exchanger: a tee and a series brazed plate exchanger

The older stands mix the hot gas bypass and the liquid quench in a vertical coil-in-shell
vessel (a Standard Refrigeration VSE 15T water-cooled condenser): the quench enters the top
of the shell at suction pressure and rains over the coil, which carries the hot gas. At low
return gas temperatures and small flows, liquid collects at the bottom of the shell without
touching the coil. It then leaves with the gas, out of equilibrium. Saturated liquid travels
in superheated vapor, so the flow meter's inlet reads saturation while the compressor port
reads the superheat setpoint.

The fix studied here keeps that mixer and adds a small brazed plate heat exchanger (BPHE)
downstream of it, ahead of the flow meter, with **both circuits piped in series**:

* pass 1: S1 (bottom) up to S2;
* a jumper from S2 to S3;
* pass 2: S3 down to S4, then the suction line.

The plates do not exchange heat with another fluid. The stream sweeps both faces of every
plate, and the plates bring its liquid and vapor to equilibrium.

`mixer = 1` simulates the worst case this exchanger can see. The mixer upstream is replaced by
a **bare tee**, where the hot gas (valve 2) and the quench (valve 3) join with no heat
exchanged before the plates. Everything else is the stand as defined (`mixer = 0` is the
brazed-plate mixing exchanger of the newer stand).

## The model

**Topology.**
* The suction exchanger takes the mixing exchanger's places in the state vector:
  * pass 1 uses the five dynamic quench cells (`h_q`), from the inlet;
  * pass 2 uses the five quasi-steady gas cells (`h_g`), marched from its inlet (cell 4) to S4 (cell 0);
  * the five plate walls (`T_mw`) sit between pass 1 cell *j* and pass 2 cell *j*, so the passes run counterflow.
* The bypass and quench lines and the tee join pass 1's inlet cell, and the jumper joins pass 2's inlet cell.
* The outlet piping (`D_mo`, `L_mo`) is split between the tee leg and the S4 leg.

**Out of equilibrium.** Each cell holds its mixture enthalpy $`h`$, which the conservation
laws fix. In addition, the cell carries the vapor share $`s`$ (1 minus the liquid share) of
the stream leaving it. The liquid boils at the cell's own saturation temperature
$`T_\mathrm{sat}`$, and the vapor is superheated by

```math
\theta = \frac{(h - h_l)/s - h_{lv}}{c_p}
```

over the dew point. The tee's stream starts with the liquid of the quench that did not flash
in valve 3, $`1 - s = \dot m_3 (1 - x_3) / (\dot m_2 + \dot m_3)`$, at the vapor's mixed
temperature. In a well-mixed cell with flow $`\dot m`$, liquid evaporates by two paths: the
vapor's heat to the surface of the liquid film on the wetted share $`F_w`$ of the plates,
and the wall's heat to the film:

```math
\dot m (s - s_\mathrm{in}) = \frac{\alpha_v F_w A\, \theta(s) + \alpha_e F_w A\, (T_w - T_\mathrm{sat})}{h_{lv}}
```

This is a quadratic in $`s`$, solved in closed form, with $`s`$ capped at its equilibrium
value. On the dry share $`1 - F_w`$ the wall exchanges heat with the vapor. The two passes
share the walls: in the counterflow arrangement, pass 1's hot vapor heats the plates, and
the plates boil off liquid on either side. The plates only move heat between the passes,
so the stream leaves with the enthalpy it came in with, less losses to ambient. Once the
liquid is gone, the wetted area recedes over the last 0.2 % of liquid.

**Heat transfer.**
* Vapor to plate: Martin's chevron correlation (VDI Heat Atlas), with the channel mass flux from the geometry:
  ```math
  Nu = 0.122\, Pr^{1/3} (\xi Re^2 \sin 2\beta)^{0.374}
  ```
  It is evaluated with saturated-vapor properties (conductivity from the property tables). That is conservative for the hot vapor near the inlet.
* Film: `bx_alpha_e`.

**Pressure drops.**
* Martin's friction factor per cell, at the cell's homogeneous density. Out of equilibrium the stream occupies almost the volume it would in equilibrium: evaporating liquid cools the vapor while adding to it.
* Static heads: pass 1 rises, pass 2 falls.
* Ports: Shah & Focke, 0.75 velocity heads each, plus the expansion from the connection into the port and the contraction back.
* The jumper as a pipe at the connection bore.

The drops are taken at the compressor flow, and every cell boils at its own pressure. Valves
2 and 3 both discharge into the tee, at the suction port pressure plus the suction line, the
exchanger and the outlet piping.

**Outputs.**
* `x_q`, `T_q` (pass 1) and `x_g`, `T_g` (pass 2): the vapor share actually leaving each cell and its vapor's temperature.
* `y_bi`, `y_bo`: the liquid share at the tee and leaving the exchanger.
* `dP_bx`: the exchanger's drop, S1 connection to S4 connection.
* The droplet model of the suction line (`tee_tau_evap`) takes over from the exchanger's outlet.

## Parameters (Settings tab, group "Suction exchanger")

| parameter | default | meaning |
|---|---|---|
| `mixer` | 0 | 1: tee and suction exchanger in place of the mixing exchanger |
| `bx_n_plates` | 30 | plates (N - 1 channels; pass 1 on S1-S2 takes the larger half) |
| `bx_L`, `bx_W` | 0.47 m, 0.12 m | effective channel length (port to port, also the port height) and width |
| `bx_b`, `bx_phi` | 2 mm, 1.17 | pressing depth, surface enlargement (hydraulic diameter 2 b / phi) |
| `bx_beta` | 30° | corrugation angle to the flow (Martin: about 30° low-theta, 60° high-theta) |
| `bx_d_port`, `bx_d_conn` | 33 mm, 32.1 mm | port bore, connection bore (1-3/8 in) |
| `bx_mass` | 8 kg | net weight: the plates' heat capacity, which also buffers liquid slugs |
| `bx_F_wet` | 0.5 | wetted share of the plate area while liquid is present: **the main uncertainty** |
| `bx_alpha_e` | 3000 W/m²K | plate to evaporating film |
| `bx_L_jmp`, `bx_K_jmp` | 0.5 m, 1.5 | jumper length and fittings |

Heating surface: $`(N - 2)\, \phi\, L\, W`$, 1.85 m² with the defaults. The ambient losses
use `mx_UA_a`. Pass 1 is taken to rise and pass 2 to fall (see the recommendations).

## Sizing

[`examples/bphe_sizing.py`](../examples/bphe_sizing.py) solves every candidate to its steady
state with the whole stand. The rest of the stand comes from a stand definition file (the
web UI's defaults file). The candidates are run at:
* the 12T machine (168 cm³/rev, 60 Hz);
* the smallest machine (105 cm³/rev, 35 Hz);

both at 0/110 °F, with 10 °F superheat at the compressor port and the liquid at 30 °C.

```
python examples/bphe_sizing.py                          # stand file webui/config/stand_defaults.json
python examples/bphe_sizing.py --F-wet 0.3 0.5 --set UA_sa=4
```

It reports, per candidate:
* heating surface;
* pressure drop;
* liquid share in and out;
* where along the flow path the liquid is gone;
* the superheat at the exchanger outlet and at the port;
* the suction line's heat gain.

### What sets the outcome: the heat the suction line gains after the exchanger

The exchanger is adiabatic, so it can only dry the stream if the stream's equilibrium state
at its outlet is superheated. The superheat at the compressor port includes everything the
line gains between the exchanger and the port:
* heat through the insulation;
* the flow meter's housing;
* fittings.

The condition is

```math
\dot Q_\mathrm{line} < \dot m \left( h_\mathrm{port} - h_\mathrm{dew} \right)
```

With the stand file's suction line (`UA_sa` = 12 W/K, about 440 W at 25 °C ambient), 0/110 °F, 10 °F at the port:

| machine | flow | line gain per kg | equilibrium superheat at the exchanger outlet | result |
|---|---|---|---|---|
| 12T, 60 Hz | 0.127 kg/s | 3.5 kJ/kg | about 3.3 °F | dries |
| 105 cm³/rev, 35 Hz | 0.046 kg/s | 9.6 kJ/kg | none: wet in equilibrium | about 1.9 % liquid leaves whatever the plate count |

At the small machine, the port's 10 °F superheat comes entirely from the suction line. That
also explains a flow meter inlet at saturation with superheat at the port, without any
non-equilibrium. With a better insulated line the same exchanger dries both machines:
* `UA_sa` = 6 W/K: the small machine dries, just;
* `UA_sa` = 3 W/K: about 5 °F to spare.

Before sizing, check the old stand's line by an energy balance:
* the measured hot gas and liquid states and flows at the mixer inlets;
* against the enthalpy at the port.

Alternatively, control the superheat at the exchanger outlet or at the flow meter inlet
rather than at the port.

### Candidates at the 12T point (stand file's line, `bx_F_wet` = 0.5)

| plate (L × W, port) | β | plates | area | ΔP | liquid gone at |
|---|---|---|---|---|---|
| 0.29 × 0.12 m, 33 mm | 30° | 30 | 1.14 m² | 5.8 kPa (0.84 psi) | 100 % of the path |
| 0.29 × 0.12 m, 33 mm | 30° | 40 | 1.55 m² | 4.7 kPa (0.68 psi) | 90 % |
| 0.47 × 0.12 m, 33 mm | 30° | 30 | 1.85 m² | 7.5 kPa (1.08 psi) | 90 % |
| 0.47 × 0.12 m, 33 mm | 30° | 40 | 2.51 m² | 5.7 kPa (0.82 psi) | 80 % |
| 0.47 × 0.12 m, 33 mm | 45° | 30 | 1.85 m² | 11.6 kPa (1.68 psi) | 100 % |

With only about 3 °F of equilibrium superheat at the outlet, the liquid is gone late along
the path. The margin comes from the line's heat gain more than from the plates. Within the
2 psid budget:
* low-angle (L) plates and long plates give the most heat transfer per pressure drop;
* the ports and connections take a fixed share (about 0.8 psi at 33 mm), so the connections should be at least 1-3/8 in.

## Recommendations for the hardware

* **Orientation:** feed pass 1 from the bottom (S1) upward, jumper at the top, pass 2 downward to an outlet at the bottom (S4).
  * The rising pass holds the liquid in the plates until it boils off.
  * The falling pass drains any rest toward the outlet instead of collecting it at a low port (the series connection's alternative, down then up, makes a liquid trap at the bottom ports).
  * Keep the jumper free of traps.
* **Connections:** at least 1-3/8 in. Prefer low-theta (L) or mixed plates and a long plate over more plates.
* **Inlet distribution:** a two-phase inlet distributes unevenly across the channels. The port between the passes mixes the stream again.

## Limitations

* **Homogeneous flow:**
  * no slip, hold-up or gravity separation of the liquid;
  * no maldistribution across the channels;
  * the wetted share `bx_F_wet` lumps all of these and the film's coverage. Measure it, or bracket it (`--F-wet`).
* **Vapor coefficient:** from saturated-vapor properties. Hotter vapor transfers somewhat more.
* **Cells:** five per pass, well mixed. Compared with plug flow this slightly understates evaporation (conservative).
* **The upstream mixer:** not modelled. The tee is its worst case. A mixer that already evaporates part of the liquid only helps.
