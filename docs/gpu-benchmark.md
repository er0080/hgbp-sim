# GPU benchmark of the model step

**Question.** On an RTX 3090, is a CUDA version of the model's step faster than the numba
CPU kernel on the same machine in float64? How much faster, and how much less accurate,
is it in float32 or mixed precision?

**Short answer** (one GPU, physics step only):
* **fp64.** Faster only for large batches: 2.5x the 16-thread CPU at 64 k stands,
  slower below about 10 k.
* **Mixed precision.** 14x the CPU (21x with fast-math), with differences of the order
  that FMA rounding alone gives in fp64.
* **fp32 throughout.** 20x (36x with fast-math), but held steady states and the
  conserved mass drift.

Details are in [Conclusions](#conclusions).

## What was built

`hgbp_sim/kernel/cuda/hgbp_step.cu` is a CUDA C++ port of `kernel/model.step_batch`.
It includes the property lookups (`kernel/props.py`) and the scalar components
(`components.py`). It is a benchmark prototype: nothing in the package imports it, and
the CPU kernel is unchanged.

* **Threads.** One thread advances one stand by one control step (5 steps of 0.05 s).
* **Same algorithm as the CPU kernel:**
  * sub-steps from the fastest local rate;
  * liquid-full volumes take at least 4 sub-steps;
  * an implausible sub-step is redone at 4x finer, at most twice;
  * after every sub-step, `post()` runs the bracketed Newton flash of the discharge
    volume and the intermediate section, and the Newton projection of the suction side.
  * RK4, Heun and Euler are all ported.
* **Device data.** State, parameters (structure of arrays) and the property tables stay
  on the device between launches. Each control step uploads the inputs and launches the
  kernel. The kernel writes per-stand counters: sub-steps, retries at each level, and
  projections that ran out of iterations.
* **Not ported.** The auxiliary outputs (`rhs(want_aux=True)`), the steady-state solver,
  cold starts and the environment logic.
* **Wrapper.** `hgbp_sim/kernel/cuda_step.py`:
  * generates the index header from `kernel/layout.py`;
  * builds the library with `nvcc` (cached in `~/.cache/hgbp_sim/cuda`);
  * wraps it with ctypes (`CudaStep`).
* **Benchmark script.** `examples/gpu_benchmark.py`.

### Precision modes

Every device function is a template on its compute type. Before compiling, the wrapper
writes every floating-point literal as `R(literal)`, so a float32 build stays in float32
(C++ would otherwise promote to double). The float32 kernel's SASS has 81 double-precision
instructions, all input conversions. The float64 kernel has 35 000.

| mode | state, sub-step accumulation, projections, step control | right-hand side | property tables |
|---|---|---|---|
| `fp64` | double | double | double |
| `mixed` | double | float | float |
| `fp32` | float | float | float |

In `mixed`, each RK stage state is rounded to float for the right-hand side. The stage
derivatives are accumulated in double. The projections (`post()`) run in double on the
float tables.

## Method

**Hardware.** One machine: an Intel Core i9-9900K (8 cores, 16 threads) and two RTX 3090
(24 GB, sm_86, FP64 at 1/64 of the FP32 rate). One GPU is used per run.

**Software.** CUDA 13.3, numba 0.68 (workqueue layer, `NUMBA_NUM_THREADS=16`),
Python 3.12, hgbp-sim at a740b24 plus this branch.

**Stands.** `HGBPVecEnv` with the stand definition (`webui/config/stand_defaults.json`):
* 90 % of the episodes across the line;
* idle valves closed;
* random starts (30 % cold, the compressor starting 5-30 s in);
* randomized parameters;
* R410A.

The baseline controller drives the stands, plus Gaussian noise on its actions (σ = 0.5
of the action range: random valve moves of up to a few % per step). The valve and speed
commands of every control step are recorded. They are then replayed open loop, from
identical states, on:
* the CPU kernel (`km.step_batch`, parallel build);
* the GPU kernel, in each mode.

The rate estimate `lam` is carried from step to step as `step_batch` does. The
environment instead refreshes it from its output evaluation; both sides here do the same.

**Throughput.**
* **Physics step only.** One `step_batch` call per control step on the CPU, one kernel
  launch on the GPU, 40 control steps (10 s).
* **Starting state.** A snapshot 30 s into the episodes, when every cold start is
  running up. The stands average 6.5 sub-steps per control step (26 right-hand side
  evaluations).
* **Batch size.** 1024 distinct stands, tiled to the batch size.
* **GPU timing.** CUDA events around the launches. The "wall" figures add the per-step
  input upload and counter download.

**Accuracy.** States and outputs are compared to the CPU reference:
* outputs P_s, P_d, T_s, T_d, superheat, evaluated by the CPU `rhs` on each state;
* total refrigerant mass M_s + M_d + M_i (conserved by construction);
* internal energy U_s + U_d + U_i.

**Builds.** The 60-minute hold and the table-top runs used the build before the
fast-math fix (below). The fix does not change the non-fast builds, since
pow(-x, 16) = pow(x, 16) exactly.

Run with:

```bash
NUMBA_NUM_THREADS=16 python examples/gpu_benchmark.py throughput --n 1024 --pre 120 --steps 40 \
    --sizes 1024,4096,16384,65536,131072 --modes fp64 --blocks 32 --sort
python examples/gpu_benchmark.py accuracy --n 1024 --steps 4800       # 20 min simulated
python examples/gpu_benchmark.py hold --n 256 --minutes 60
python examples/gpu_benchmark.py tabletop --n 256 --steps 1200
```

## Results

### Throughput

**Units.** Stand-control-steps per second of the physics step alone: one 0.25 s step of
one stand. Each row's workload has the same mean of 6.5 sub-steps per control step.

**GPU configuration.** One GPU, best configuration per mode:
* fp64: blocks of 32, 255 registers;
* mixed: blocks of 64, at most 64 registers;
* fp32: blocks of 128, at most 128 registers.

The GPU columns take the better of unsorted and sorted. "Sorted" means each launch takes
the stands in order of the sub-steps they needed in the previous step. Sorting wins from
16k stands up and loses below.

| stands | CPU, 16 threads | GPU fp64 | GPU mixed | GPU mixed, fast-math | GPU fp32 | GPU fp32, fast-math |
|---:|---:|---:|---:|---:|---:|---:|
| 1 024 | 50 800 | 7 700 | 32 400 | 47 200 | 43 800 | 91 400 |
| 4 096 | 58 800 | 29 000 | 117 300 | 171 600 | 147 900 | 318 400 |
| 16 384 | **61 700** | 90 500 | 358 100 | 525 500 | 542 500 | 1 022 200 |
| 65 536 | 58 500 | 146 300 | 800 500 | 1 219 500 | 1 178 000 | 2 108 900 |
| 131 072 | 58 100 | 167 500 | 931 700 | 1 450 600 | 1 386 300 | 2 600 700 |
| *vs CPU at 65 536* | 1 | 2.5x | 13.7x | 20.8x | 20.1x | 36x |

**CPU.** Best at 16 384 stands (61 700). With 8 threads it reaches 43 000. It
scales little with the batch above 4 096 stands. The workload matters: a snapshot 5 s
into the episodes (9.1 sub-steps per control step) runs at 38 000 on the CPU.

**GPU per launch.** One launch, which is one control step for all stands, takes at least
0.12 s in fp64, 25 ms in mixed and 16 ms in fp32, at any batch size. One thread runs a
whole stand: about 9 sub-steps of 4 right-hand sides each, in series. Below about 10 000
stands the GPU cannot hide this, and the fp64 kernel is slower than the CPU.

**Two GPUs.** The machine's two GPUs should double the GPU columns, each running its own
batch (not measured).

**What the tuning did** (at 65 536 stands):
* Sorting by sub-step count: fp64 87 k → 146 k, mixed 551 k → 800 k, fp32 869 k →
  1 178 k. Warp divergence is the largest loss.
* Register cap: mixed 400 k → 550 k at 64 registers, despite the spills (4.4 KB of local
  memory per thread). fp32 gains 15 % at 128. fp64 does not change: it is bound by the
  3090's FP64 units (1:64).
* Block size: 32-128 within 5 %. 256 is slower.
* `--fmad=false` (exact CPU arithmetic): about 5 % slower in fp64 (82 k vs 87 k unsorted).
* `--use_fast_math`: float only.
  * As first written, it failed. Fast `powf` returns NaN for a negative base, and the
    Churchill friction factor raises a negative number to the 16th power at Reynolds
    numbers below about 7 (stagnant lines). Idle and starting stands then went into retry
    storms: 64 000 level-1 and 255 000 level-2 retries in 8 control steps for 128
    stands, 6x the sub-steps, and 50-150x slower.
  * Taking the magnitude of that even power fixes it. The value is unchanged; fp64 still
    matches the CPU to 1e-14.
  * With the fix, fast-math gains +50 % (mixed) and +80 % (fp32).

**Warm-up.** These are separate from the figures above:
* `nvcc` build: about 65 s per option set, all three modes in one library, cached in
  `~/.cache/hgbp_sim/cuda`.
* Creating a `CudaStep` (allocations, uploading the 3 MB of tables and the parameters):
  0.15-0.3 s.
* The numba CPU kernels compile in about 60-75 s on the first import, then come from
  numba's cache.

### Accuracy, 20 minutes simulated

**Setup.**
* 1024 stands, 4800 control steps, inputs recorded from the controller with random valve
  moves.
* About 30 % of the stands start cold during the run.
* Many episodes trip or finish their schedule and stop the compressor. 776 stands are
  running after 10 s, 499 after 20 min.

**Errors.** Differences from the CPU reference, as the maximum over the stands that are
running in the reference:

| mode | P_s [bar] 100 s / 20 min | P_d [bar] 100 s / 20 min | T_d [K] 100 s / 20 min | superheat [K] 100 s / 20 min |
|---|---|---|---|---|
| fp64, `--fmad=false` (all stands) | 1e-13 / 2e-13 | 6e-13 / 4e-13 | 1e-12 / 5e-12 | 1e-12 / 3e-11 |
| fp64 | 3.0e-5 / 8.9e-6 | 1.4e-4 / 2.7e-5 | 2.4e-3 / 9.5e-4 | 2.0e-3 / 4.7e-4 |
| mixed | 6.2e-5 / 1.4e-4 | 3.0e-4 / 5.9e-5 | 5.0e-4 / 3.7e-4 | 4.2e-3 / 1.4e-3 |
| fp32 | 5.0e-4 / 3.6e-4 | 4.6e-3 / 4.7e-4 | 7.8e-3 / 1.4e-2 | 4.0e-3 / 1.4e-2 |
| mixed, fast-math | 3.9e-5 / 1.5e-4 | 3.4e-4 / 7.7e-5 | 4.0e-4 / 3.5e-3 | 3.5e-4 / 3.9e-3 |
| fp32, fast-math | 1.1e-4 / 3.7e-4 | 6.6e-4 / 5.1e-4 | 1.9e-3 / 1.4e-2 | 4.0e-3 / 1.4e-2 |

**Fast-math** (after the fix below) is as accurate as the plain float builds. Mass drift is
1.4e-8 (mixed) and 2.0e-4 (fp32), with the same retries and the same stopped stands
separating.

**Medians** over the running stands after 20 min are much smaller: the superheat differs
by 4e-5 K in fp64 (FMA), 7e-5 K in mixed and 6e-4 K in fp32.

**Mixed against fp64.** Mixed precision's differences from the CPU are of the same order
as the default fp64 build's own, which come only from FMA rounding.

**Exactness of the port.** Built with `--fmad=false`, the fp64 kernel matches the CPU
kernel to 1e-13 in every output and 3e-10 in the worst state, over 20 minutes and all 1024
stands. It also takes the same 9.094 sub-steps per control step. The port is faithful.
The default build contracts multiply-adds to FMA instructions. These round differently,
which is where its differences come from.

**Stopped stands.** These separate in every mode except `--fmad=false`, by up to 5-7 K
in superheat and 0.02-0.04 bar after 20 min:

| fp64 | mixed | fp32 |
|---|---|---|
| 56 at 10 min, 318 at 20 min | 54 / 310 | 60 / 346 |

* **Which stands.** A stand separates when its P_s or P_d differs by more than 0.01 bar,
  or its T_d or superheat by more than 0.1 K. Every separated stand is stopped in the
  reference: no compressor flow, pressures equalizing through the valves near zero
  pressure difference.
* **Running stands.** No running stand separated in any mode in the 20-min and 60-min
  runs. At the table top (below), one running stand pinned at the clamp reached 0.14 K
  in T_d for a while, in fp64 (FMA) and mixed.
* **Cause.** The dynamics of an idle stand amplify round-off by about 10 orders of
  magnitude in 10 minutes. FMA contraction alone is enough to change them.
* **This is a property of the model, not of the GPU.** A different CPU compiler would
  diverge the same way.

**Conservation and the solvers over the 20-minute run:**

| | CPU | fp64 | mixed | fp32 |
|---|---|---|---|---|
| total mass, max relative drift | 4.7e-14 | 4.2e-14 | 1.6e-8 | 2.3e-4 (median 3.8e-6) |
| internal energy vs CPU, median | | 4e-7 | 4e-7 | 7e-6 |
| sub-steps per control step | 9.094 | 9.084 | 9.082 | 9.081 |
| level-1 / level-2 retries | | 8 / 0 | 9 / 0 | 9 / 0 |
| suction projection not converged in 4 iterations (share of sub-steps) | | 11.6 % (7.5 % without FMA) | 11.4 % | 92.7 % |
| flash not converged in 8 iterations (per sub-step) | | 0.06 % | 0.06 % | 0.06 % |

**Mixed.** The mass drift comes from the three mass rates, rounded to float separately,
so they no longer cancel exactly. It is 1.6e-8 of the charge after 20 minutes.

**fp32 projections.** The suction projection's tolerance is 1e-9 of the mass. In float
it is never met, so the projection always runs its 4 iterations. It still converges to
float resolution, and nothing diverges.

**Retries.** The plausibility retries do not grow in any mode.

### Steady state held for 60 minutes

**Setup.** 256 warm starts at their set point, inputs held for 60 minutes. 244 stands
are running; 12 are idle and separate as above.

**Errors.** Maximum over the running stands after 5 / 20 / 60 min:

| mode | P_s [bar] | P_d [bar] | T_d [K] | superheat [K] | mass drift, max |
|---|---|---|---|---|---|
| fp64, `--fmad=false` (all stands) | 3e-9 at 60 min | 2e-9 | 8e-8 | 9e-8 | 4.9e-14 |
| mixed | 1.4e-5 / 3.5e-5 / 3.0e-5 | 5.1e-5 / 6.1e-5 / 8.6e-5 | 4.1e-4 / 5.0e-4 / 7.2e-4 | 2.7e-4 / 3.9e-4 / 5.7e-4 | 1.5e-6 |
| fp32 | 4.4e-4 / 1.4e-3 / 2.0e-3 | 7.9e-4 / 2.6e-3 / 3.6e-3 | 0.015 / 0.053 / 0.078 | 0.011 / 0.040 / 0.061 | 3.1e-4 |

**mixed** holds the steady state of the fp64 model to well below a sensor's resolution.
It shows no trend.

**fp32 drifts.** The superheat moves away from the fp64 steady state: 0.04 K after 20
minutes, 0.06 K after an hour. The pressures move by a few mbar. The cause is the drift of the
conserved states: mass and energy accumulated in float.

### Discharge pressure at the top of the R410A tables

**Setup.**
* 256 warm stands, R410A; valve 1 closed to 2 % over 60 s, the compressor kept running,
  no trip.
* The tables end at p_max = 0.99 P_crit = 48.5 bar. The flash clamps P_d at
  0.999 p_max = 48.47 bar.
* The median stand reaches the clamp within 50 s (`P_d / p_max` = 0.999) and stays
  pinned for the remaining 4 minutes.
* 232 stands are running.

**Errors.** Maximum over the running stands after 5 min:

| mode | P_s [bar] | T_d [K] | superheat [K] | mass drift, max |
|---|---|---|---|---|
| fp64, `--fmad=false` | 1.9e-5 | 2e-12 | 2.3e-3 | 1.0e-14 |
| fp64 | 2.2e-4 | 3.2e-4 | 0.021 | 1.3e-14 |
| mixed | 1.8e-4 | 4.9e-4 | 0.022 | 2.2e-9 |
| fp32 | 2.8e-3 | 3.2e-3 | 0.060 | 1.8e-5 |

* **P_d** is the clamp value in all modes.
* **No new failures.** No mode produces a non-finite state. Each mode has the same 4
  level-1 retries in one stand, and no level-2 retries.
* **The flash at the clamp.** It cannot reach its target density and runs out of
  iterations about 1.57 million times in every mode. This is a property of the clamped
  tables (hgbp-sim #15), not of the precision. The fp64 build without FMA, which
  reproduces the CPU kernel, counts the same.
* **Divergence at the top.** Even the exact fp64 build diverges slightly here (2e-3 K
  in superheat): the pinned state is sensitive to the last bit of the math library.
* **fp32 at the top.** The worst differences are 3-15x those of mixed and fp64. It does
  not break.

## Conclusions

1. **fp64 on the GPU is faster than the CPU only for large batches.** With one RTX 3090:
   * slower than the 16-thread CPU below about 10 000 stands (7.7 k vs 50.8 k
     env-steps/s at 1 024);
   * 1.5x at 16 k;
   * 2.5x at 64 k;
   * 2.7x at 128 k (167 k vs the CPU's best of 62 k).

   The GeForce card's FP64 rate (1:64) caps it. Both GPUs together would make it about
   5x.
2. **Mixed precision** (float tables and right-hand side; double state and projections)
   runs at 14x the CPU at 64 k stands, and 21x with fast-math. On running stands, the
   worst stand at any checkpoint of the 20-minute run is within 1.2e-3 bar and 7e-3 K of
   the CPU. Typical differences are 1e-5 bar and 1e-4 K. The default fp64 build differs
   by the same order, from FMA rounding alone. It holds steady states without drift
   (6e-4 K in an hour) and conserves mass to 1e-6 per hour.
   Retries, step control and projections behave as in fp64.
3. **fp32 throughout** runs at 20x (36x with fast-math). It is not broken: no NaNs, no
   retry storms, sane at the table top. But it drifts:
   * the conserved mass by up to 3e-4 per hour;
   * held steady states by 0.06 K and a few mbar per hour;
   * its suction projection never meets its tolerance.

   For RL episodes of 20-40 minutes this is probably harmless. It is a systematic error
   that mixed does not have, and fp32 is only 1.5-1.7x faster than mixed.
4. **Stopped stands are chaotic in this model.** Round-off differences of any origin,
   including FMA contraction on the same hardware, grow to several kelvin of superheat
   in 10-20 minutes. Comparisons and reproducibility tests should look at running stands
   or use exact arithmetic.

**Recommendation.**
* **Default.** Keep the numba CPU kernel as the reference and the default.
* **Mixed precision with fast-math on the GPU.** Choose it if the training needs more
  than about 60 k env-steps/s and can run batches of 16 k-64 k stands per GPU. That gives
  8-21x the CPU per GPU at accuracy indistinguishable for training. In that case:
  * use sorted launches and the fp64 `--fmad=false` build as a correctness check
    against the CPU;
  * port the auxiliary outputs and the environment's per-step logic as well: at these
    rates, the Python environment and the CPU output evaluation become the bottleneck.
* **Not fp64 on this card.** fp64 on a 3090 is not worth the second code path.
* **Not fp32 throughout.** Its drift is avoidable at a modest cost.

## Limitations and open points

* **Physics only.** The figures are for the physics step alone. Not measured or not
  ported:
  * the environment's own step: interlock, rewards, observations, the auxiliary output
    evaluation (`rhs(want_aux=True)`, one more right-hand side per control step);
  * transfers of observations.

  On the CPU, the whole environment ran at about 25 000-28 000 env-steps/s at 1 024
  stands (recording runs) against 50 800 for its physics step alone.
* **Second implementation.** The port must be kept in step with `kernel/model.py` by
  hand. The literal wrapping, the generated index header and the `--fmad=false`
  comparison make that checkable, not automatic.
* **Coverage.** One stand definition (R410A) and one workload mix. R454B was not run.
* **Untried tuning.** Shared-memory tables, warp-per-stand layouts and fp64-only-where-needed
  variants.
