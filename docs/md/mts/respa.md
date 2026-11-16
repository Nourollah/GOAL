# RESPA integrator

`RespaIntegrator` is a velocity-Verlet **r**eference **s**ystem **p**ropagator **a**lgorithm (Tuckerman, Berne & Martyna, *J. Chem. Phys.* **97**, 1990 (1992)) with two force levels and an optional Langevin thermostat. It is an ordinary ASE `MolecularDynamics` subclass, so trajectories, loggers, and observers work as usual. `MTSSimulation` is a thin, config-friendly wrapper around it.

## Theory in brief

Split the force into a fast part, which is cheap and rapidly varying, and a slow part, which is expensive and smooth:

$$
\mathbf F(\mathbf x) = \mathbf F_\text{fast}(\mathbf x) + \mathbf F_\text{slow}(\mathbf x),
\qquad
\mathbf F_\text{fast} = \mathbf F_\text{ML},\quad
\mathbf F_\text{slow} = \mathbf F_\text{QM} - \mathbf F_\text{ML}.
$$

A symmetric Trotter factorisation of the Liouville propagator over one outer step $\Delta t = n\,\delta t$ gives

$$
e^{iL\Delta t} \approx
e^{iL_\text{slow}\Delta t/2}\,
\Big[e^{iL_\text{fast}\delta t/2}\, e^{iL_x\delta t}\, e^{iL_\text{fast}\delta t/2}\Big]^{n}\,
e^{iL_\text{slow}\Delta t/2},
$$

which is time-reversible and symplectic. The dynamics is therefore that of the **full** force $\mathbf F_\text{QM}$, but $\mathbf F_\text{QM}$ is evaluated only once per $n$ inner steps. The ML model never has to be exact. It only has to absorb the high-frequency part of the force, so that the residual $\mathbf F_\text{QM}-\mathbf F_\text{ML}$ varies slowly on the scale of $\Delta t$.

One outer step as implemented (momenta $\mathbf p$, masses $m$):

```text
p += F_slow * Δt/2                  # slow half-kick        [+ Langevin O-step if thermostatted]
repeat n times:
    p += F_fast * δt/2              # fast half-kick
    x += p/m * δt                   # drift
    F_fast = F_ML(x)                # one ML call
    p += F_fast * δt/2              # fast half-kick
F_slow = F_QM(x) - F_fast           # one QM call (reuses the ML forces just computed)
p += F_slow * Δt/2                  # slow half-kick        [+ Langevin O-step if thermostatted]
```

Cost per outer step: $n$ ML calls and **1** QM call. With `slow_calculator=None`, `F_slow = 0` and the scheme reduces to plain velocity Verlet on the fast potential, which is useful for debugging.

## API

```python
from ase import units
from goal.md.mts import RespaIntegrator

dyn = RespaIntegrator(
    atoms,
    fast_calculator=ml_calc,
    timestep=0.5 * units.fs,     # INNER step, ASE time units
    mts_ratio=4,                 # n -> outer step 2 fs
    slow_calculator=qm_calc,     # None -> fast-only Verlet
    temperature_K=300.0,         # None -> NVE
    friction=0.01,               # 1/fs, only with temperature_K
    trajectory="respa.traj",
    logfile="respa.log",
    loginterval=1,               # in OUTER steps
)
dyn.run(1000)                    # 1000 OUTER steps = 1000 * 4 * 0.5 fs = 2 ps
print(dyn.mts_stats)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `atoms` | required | System (masses must be set) |
| `fast_calculator` | required | Cheap calculator, called once per inner step. |
| `timestep` | required | **Inner** timestep in ASE units (`n_fs * ase.units.fs`) |
| `mts_ratio` | `4` | Inner steps per outer step; outer step = `mts_ratio * timestep` |
| `slow_calculator` | `None` | Expensive calculator. Its correction `F_QM − F_fast` is applied once per outer step. |
| `temperature_K` | `None` | Enables the Langevin (Ornstein–Uhlenbeck) thermostat |
| `friction` | `0.01` | Langevin friction γ in **1/fs** |
| `trajectory`, `logfile`, `loginterval` | `None`, `None`, `1` | Standard ASE `MolecularDynamics` options, counted in outer steps |

Methods and properties:

- `step()` advances **one outer step**. `run(steps)` runs `steps` outer steps. As in ASE, observers see the initial state (step 0) and then each outer step. Forces are computed once at step 0 and carried over, so running in chunks costs no extra calls.
- `dt` and `get_time()` count **outer** steps, so ASE's `MDLogger` time column is the simulated time.
- `mts_stats` returns `{"fast_calculator_calls", "slow_calculator_calls", "mts_ratio", "inner_timestep_fs", "outer_timestep_fs"}`. The integrator counts the calls itself.

### Through `MTSSimulation`

`MTSSimulation(atoms, fast_calculator, config=MTSConfig(...), slow_calculator=...)` builds the integrator. It draws Maxwell–Boltzmann velocities when `temperature_K` is set, attaches trajectory and log observers, and runs in chunks with a progress bar. `MTSConfig` fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `timestep_fs` | `1.0` | inner step, **fs** (converted for you) |
| `mts_ratio` | `4` | inner steps per outer step |
| `total_steps` | `1000` | **outer** steps |
| `temperature_K` | `300.0` | `None` = NVE |
| `friction` | `0.01` | 1/fs |
| `log_interval` | `10` | outer steps between log/trajectory writes |
| `trajectory_file`, `log_file` | `None` | relative paths are placed under `output_dir` |
| `output_dir` | `"outputs/md/mts"` | created if missing |
| `chunk_size` | `50` | outer steps per internal `run()` call |
| `show_progress` | `True` | progress bar |

Unknown `mts:` keys raise `TypeError`.

## Choosing the inner and outer steps

**Inner step δt** is set by the fastest motion the *fast* force describes, exactly as in ordinary MD:

- With explicit hydrogens use 0.5 fs, or up to 1 fs if you can tolerate some drift.
- With heavy atoms only, 1–2 fs.

**Outer step Δt = n·δt** is limited by how quickly $\mathbf F_\text{QM}-\mathbf F_\text{ML}$ changes, and by RESPA's **resonance instability**. When Δt approaches half the period of a vibration that the slow force still couples to, energy is pumped into that mode. For an X–H stretch (period ≈ 9–11 fs), keep Δt ≲ 3–4 fs.

Practical recipe:

1. Start at `mts_ratio=4` with δt = 0.5 fs (Δt = 2 fs).
2. Run a short **NVE** test (`temperature_K=None`) and monitor the conserved energy at outer steps, $E_\text{QM}(\mathbf x) + K$ (see below).
3. If drift is acceptable, try `mts_ratio=6–8`. If not, improve the ML model, which shrinks the correction, before touching the timesteps.

| Fast-level quality | Typical `mts_ratio` |
|--------------------|---------------------|
| Foundation model, not trained on your system | 2–4 |
| GOAL model trained on in-distribution data | 4–8 |
| Same level of theory, just a cheaper basis or grid | 8–16 |

### Monitoring the right energy

With a slow calculator, `atoms.calc` is the slow calculator after every outer step, so ASE's `MDLogger` and trajectory writers report $E_\text{QM}$ and $\mathbf F_\text{QM}$ at the current positions. $E_\text{QM} + K$ is the conserved quantity of the RESPA dynamics, so the logged `Etot` is the right one to watch in an NVE test, and each trajectory frame is a QM-labelled structure. Without a slow calculator the log shows the fast level.

## Thermostat details

With `temperature_K` set, an Ornstein–Uhlenbeck update $\mathbf p \leftarrow c_1\mathbf p + c_2\boldsymbol\xi$ is applied after each **outer** half-kick, where $c_1 = e^{-\gamma\,\Delta t/2}$ (half an outer step, since it runs twice per outer step) and $c_2 = \sqrt{m k_B T (1-c_1^2)}$.

## Testing without QM

Use two cheap ASE calculators to check your setup before spending QM time:

```python
from ase import units
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from goal.md.mts import RespaIntegrator

atoms = molecule("H2O"); atoms.center(vacuum=5.0)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = RespaIntegrator(atoms, EMT(), timestep=0.5 * units.fs, mts_ratio=4,
                      slow_calculator=EMT(), temperature_K=300, friction=0.01)
dyn.run(20)
print(dyn.nsteps, dyn.mts_stats["outer_timestep_fs"])   # 20 2.0
```

With identical fast and slow calculators the slow correction is exactly zero, so this exercises the bookkeeping only.
