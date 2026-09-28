# Examples

Scripts in `examples/`:

| script | what it does |
|---|---|
| `closed_loop_pid.py [--cold] [--charge F]` | baseline PID through a 4-point schedule at charge factor F; plots everything |
| `open_loop_step.py` | +10 % step on each valve from a steady point |
| `collect_dataset.py` | expert plus exploration noise data from the batched environment -> `.npz` |
| `train_bc.py` | behaviour-cloning MLP (PyTorch) and closed-loop evaluation against the expert |
| `benchmark.py` | speed of the live stand, the batched environment and the steady-state solver |

## Direct plant access

For custom experiments, MPC or system identification:

```python
from hgbp_sim import HGBPPlant, PlantParams, solve_steady_state, named_point

plant = HGBPPlant(PlantParams(), n=1, dt=0.05)
plant.p.charge[:] = plant.nominal_charge() * 1.2                # 20 % overcharged
plant.set_inputs(T_amb=298.15, T_wi=293.15)
pt = named_point("MT_standard", plant.props, N=1450.0)         # -10 / 45 °C, 10 K SH, 38 °C condensing
ss = solve_steady_state(plant, pt["P_s"], pt["P_d"], pt["SH"], pt["N"], P_i=pt["P_i"])
plant.set_state(0, ss["x"])
aux = plant.step(1.0, u_cmd=ss["u"], N_cmd=pt["N"])             # advance 1 s
meas = plant.measure()                                           # noisy, lagged sensor readings
```

## Closed loop with the baseline PID

Four test points (MT standard -> high lift at 1750 rpm -> HT standard at 1200 rpm -> LT
standard), nominal charge, warm start. Grey traces are the noisy sensor readings, dashed
lines the setpoints (`python examples/closed_loop_pid.py`):

![closed loop, nominal charge](../figures/closed_loop_charge_1.0.png)

The same schedule from a cold, equalized stand, compressor started at 10 s. The shell and
discharge temperatures take tens of minutes to settle (`--cold`):

![closed loop, cold start](../figures/closed_loop_cold_start.png)

Undercharged (30 % of nominal): the receiver level falls below the dip tube and valve 3
passes vapor (the negative "subcooling" is that vapor's superheat). The quench cannot cool
the bypass gas, so superheat runs high and the suction pressure loop loses the point
(`--charge 0.3`):

![closed loop, undercharged](../figures/closed_loop_charge_0.3.png)

Overcharged (210 % of nominal): the receiver is full and liquid floods part of the
condenser from the start. At the high-lift point the water loop cycles; after about
6 minutes there the condenser is liquid-logged, the water valve saturates and the
intermediate and discharge pressures run past the 41.4 bar trip (the example does not
enforce trips). The same happens at the last point (`--charge 2.1`):

![closed loop, overcharged](../figures/closed_loop_charge_2.1.png)

## Open-loop steps

+10 % on each valve from the MT standard point: every valve moves every controlled
variable (`python examples/open_loop_step.py`):

![open loop step responses](../figures/open_loop_step.png)
