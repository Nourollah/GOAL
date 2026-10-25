# Dynamics

This page covers how `goal.md` integrates the equations of motion: the `DynamicsFactory` integrators, the `Simulation` orchestrator, the observers that write trajectories and logs, and the chunked run loop. It ends with practical advice on thermostats and time steps.

`goal.md` gives you two levels to work at:

| Level | What you get | Use it when |
| --- | --- | --- |
| `Simulation` | Molecule + calculator + Langevin NVT + trajectory/log/energy collection in one object; also buildable from a Hydra config | Standard NVT runs, config-driven batches, the CLI |
| `DynamicsFactory` + `setup_observers` + `run_dynamics` | You choose the integrator (any ASE dynamics, FlashMD, or your own), and the observers and progress loop come from GOAL | Non-Langevin ensembles, FlashMD, custom integrators, fine control |

## Simulation

`goal.md.core.simulation.Simulation` (also exported from `goal.md`) is the simplest way to run MD.

```python
from goal.md import CalculatorFactory, MoleculeFactory, Simulation

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
calc = CalculatorFactory.create("goal_model", checkpoint="path/to/best.simurgh", device="cuda")

sim = Simulation(
    atoms, calc,
    dynamics_key='langevin_ase', steps=20_000, temperature_K=300.0,
    timestep_fs=0.5, friction=0.01, log_interval=20,
    trajectory_file="md.traj", log_file="md.log",
    output_dir="outputs/md/ethanol_300K",
)
result = sim.run()
```

### Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `atoms` | — | The system. `atoms.calc` is set to `calculator`. |
| `calculator` | — | Any ASE calculator. |
| `dynamics_key` | `None` | The integrator used for simulations (either `"langevin_ase"` or `"langevin_flashmd"`). It will be set to `"langevin_ase"` if no value is porvided. |
| `dynamics_cfg` | `dict()` | The configuration parameters placeholder. |
| `steps` | `1000` | Number of integration steps per `run()` call. |
| `temperature_K` | `300.0` | Thermostat target and initial-velocity temperature (K). |
| `chunk_size` | `100` | Steps per inner `dynamics.run()` call (progress-bar granularity, velocity restarts). |
| `restart_thermostat` | `False` | Re-draw Maxwell–Boltzmann velocities after every chunk (see [run loop](#the-run-loop-run_dynamics)). |
| `log_interval` | `10` | Interval, in steps, for trajectory frames, log lines and energy samples. |
| `trajectory_file` | `None` | ASE `.traj` output. Relative paths are placed under `output_dir`. |
| `log_file` | `None` | ASE `MDLogger` text log. Relative paths are placed under `output_dir`. |
| `output_dir` | `"outputs/md"` | Created if needed. Relative to the current working directory. |
| `show_progress` | `True` | Rich progress bar (needs `rich`, a core dependency). |
| `initialise_velocities` | `True` | Draw Maxwell–Boltzmann velocities at `temperature_K` before the first step. |
| `timestep` | `None` | Time step in **ASE units** (converted with `ase.units.fs`). * It must not be used along with **timestep_fs**. |
| `timestep_fs` | `1.0` | Time step in **femtoseconds** - for `"langevin_ase"` integrator. * It must not be used along with **timestep**. |
| `fixcm` | `None` | Fix Centre of Mass - for `"langevin_ase"` integrator. |
| `friction` | `None` | Langevin friction in **ASE units** (converted as `friction / units.fs`) - for `"langevin_ase"` integrator. * It must not be used along with **friction_per_fs**. |
| `friction_per_fs` | `None` | Langevin friction in **1/fs** - for `"langevin_ase"` integrator. * It must not be used along with **friction**. |
| `time_constant` | `None` | Thermostat time constant in fs - for `"langevin_flashmd"` integrator. |
| `model_name` | `None` | FlashMD model name - for `"langevin_flashmd"` integrator. |
| `device` | `None` | Torch device ("cuda" or "cpu") - for `"langevin_flashmd"` integrator. |
| `**kwargs` | --- | Additional arguments for building the dynamics integrator. |

`SimulationConfig` is a dataclass with the same fields (minus `atoms`/`calculator`). It documents the schema and supplies the key set that `from_config` accepts.

### What `run()` does

1. On the first call, it draws initial velocities if `initialise_velocities=True`, then builds an ASE `Langevin(atoms, timestep_fs * fs, temperature_K=..., friction=friction / fs)`, creates `output_dir` and attaches the observers (trajectory, log, and always an in-memory energy collector).
2. It calls [`run_dynamics`](#the-run-loop-run_dynamics) for `steps` steps and returns a `RunResult`.

Setup happens only once, so calling `run()` again **continues** the same trajectory: the integrator, its random state and the observers are reused, and new frames are appended. `sim.dynamics` gives you the underlying ASE object (built on first access), so you can attach extra observers before running:

```python
sim.dynamics.attach(lambda: print(sim.atoms.get_temperature()), interval=1000)
sim.run()      # 20 000 steps
sim.run()      # 20 000 more, same trajectory file
```

> [!IMPORTANT]
> `Simulation` uses the `dynamics` section in a config and `DynamicsFactory` to integrate with ASE's or FlashMD's Langevin thermostat. For NVE, Nosé–Hoover, Bussi or NPT, use the [lower-level path](#building-your-own-loop).

### `RunResult`

| Field | Meaning |
| --- | --- |
| `steps_completed` | Steps run in **this** `run()` call (equals `steps`). |
| `final_energy` | Potential energy (eV) after the last step, or `None` if it could not be evaluated. |
| `final_temperature` | Instantaneous kinetic temperature, $T = 2E_\mathrm{kin} / (3 N k_\mathrm{B})$. It uses $3N$ degrees of freedom, with no correction for the fixed centre of mass. |
| `energy_history` | Potential energy at every `log_interval` step, including step 0, accumulated over all `run()` calls. |

The instantaneous temperature of a small molecule fluctuates by tens of kelvin from step to step. Judge equilibration from the log, not from `final_temperature`.

### From a config

`Simulation.from_config(cfg)` and its wrapper `simulate_from_config(cfg)` (build then run) read a **flat** mapping:

```yaml
molecule:              # builder from `key` (or `type`/`method`; default from_smiles)
  key: from_smiles
  smiles: CCO
calculator:            # builder from `key` (or `type`; default goal_model)
  key: goal_model
  checkpoint: path/to/best.simurgh
  device: cuda
dynamics:              # builder from `key` (or `method`; default langevin_ase)
  key: langevin_ase
  timestep_fs: 1.0
  temperature_K: 300.0
  friction_per_fs: 0.05
  fixcm: true
# any SimulationConfig field at the top level:
steps: 20000
trajectory_file: md.traj
log_file: md.log
output_dir: outputs/md/ethanol
```

Top-level keys that are not `SimulationConfig` fields are silently ignored; `num_steps` is accepted as an alias of `steps`. Hydra's `_target_`/`_convert_` keys inside `molecule:`/`calculator:`/`dynamics:` are dropped, so group templates can be composed in. The [Configuration reference](../reference/configuration.md#writing-a-simulation-config-that-works) shows a template and how the bundled `langevin_with_model.yaml` is built.

## DynamicsFactory

```python
from goal.md import DynamicsFactory
dyn = DynamicsFactory.create(key, atoms, **kwargs)
```

`DynamicsFactory` (`goal.md.core.md_factory`) looks up a registered `DynamicsBuilder` and returns the integrator it builds. An unknown key raises `ValueError` with the list of registered keys. Two are built in:

| Key | Returns | Extra dependency |
| --- | --- | --- |
| `langevin_ase` | `ase.md.langevin.Langevin` | none |
| `langevin_flashmd` | `flashmd.ase.langevin.Langevin` | `flashmd` |

### `langevin_ase`

```python
from ase import units
from goal.md import DynamicsFactory

dyn = DynamicsFactory.create(
    "langevin_ase", atoms,
    timestep_fs=0.5,                # femtoseconds
    temperature_K=300.0,
    friction_per_fs=0.01,           # 1/fs
    fixcm=True,
)
# equivalent, in ASE's own units:
dyn = DynamicsFactory.create("langevin_ase", atoms, timestep=0.5 * units.fs,
                             temperature_K=300.0, friction=0.01 / units.fs)
```

| Argument | Default | Notes |
| --- | --- | --- |
| `atoms` | — | Positional or `atoms=`. (Older docs used `molecule=`, which no longer works.) |
| `timestep_fs` | — | Time step in fs (converted with `units.fs`). |
| `friction_per_fs` | — | Friction in fs⁻¹ (converted with `units.fs`). |
| `timestep` | `1.0 * units.fs` when neither is given | Passed to ASE **unchanged**, i.e. in ASE time units. Mutually exclusive with `timestep_fs`. |
| `temperature_K` | `300.0` | |
| `friction` | `0.5` when neither is given | Passed to ASE **unchanged**: 0.5 per ASE time unit ≈ 0.049 fs⁻¹. Mutually exclusive with `friction_per_fs`. |
| `fixcm` | `True` | Remove centre-of-mass drift. Deprecated in ASE ≥ 3.28 (see below). |
| `**kwargs` | | Forwarded to `Langevin` (`logfile`, `trajectory`, `rng`, …). |

> [!WARNING]
> **Units.** The bare `timestep` / `friction` arguments go to ASE unchanged, exactly like `ase.md.langevin.Langevin`: `timestep=1.0` means 1 ASE time unit ≈ **10.18 fs**, which is unstable for any molecule with hydrogens. Use `timestep_fs` / `friction_per_fs`, or multiply / divide by `units.fs`. The template [`src/goal/configs/md/dynamics/langevin.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/dynamics/langevin.yaml) uses `timestep_fs: 1.0` and `friction_per_fs: 0.05` (before October 2026 it passed `timestep: 1.0` raw).

On ASE ≥ 3.28, `fixcm=True` emits a `FutureWarning`: ASE's in-integrator centre-of-mass fix does not sample NVT exactly. The deviation is small for large systems but noticeable for small molecules. The ASE-recommended replacement is shown below:

```python
from ase.constraints import FixCom
atoms.set_constraint(FixCom())
dyn = DynamicsFactory.create("langevin_ase", atoms, timestep=0.5 * units.fs,
                             temperature_K=300, friction=0.01 / units.fs, fixcm=False)
```

`Simulation` does not pass `fixcm`, so it inherits ASE's default (`True`) and you will see the same warning there.

### `langevin_flashmd`

[FlashMD](https://github.com/lab-cosmo/flashmd) predicts positions and momenta several femtoseconds ahead in one step, instead of integrating forces with a small time step. The builder fetches the pretrained pair `get_pretrained(model_name, timestep)` and wraps the FlashMD Langevin integrator:

```python
from goal.md import CalculatorFactory, DynamicsFactory, MoleculeFactory

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
atoms.calc = CalculatorFactory.create("flashmd", model_name="pet-omatpes-v2")   # energy model

dyn = DynamicsFactory.create(
    "langevin_flashmd", atoms,
    timestep=16,               # fs (plain number; converted internally)
    temperature_K=300.0,
    time_constant=100.0,       # fs
    model_name="pet-omatpes-v2",
    device=None,               # auto: cuda if available
)
```

| Argument | Default | Notes |
| --- | --- | --- |
| `timestep` | `16` | In **fs**; multiplied by `units.fs` inside the builder. Must match a step size FlashMD publishes for `model_name`. |
| `temperature_K` | `300.0` | |
| `time_constant` | `100.0` | Thermostat time constant, fs. |
| `model_name` | `"pet-omatpes-v2"` | FlashMD model family. |
| `device` | `None` | `"cuda"` if available, else `"cpu"`. |

The units here differ from `langevin_ase`: FlashMD takes plain femtoseconds. Validate a FlashMD trajectory against conventional MD with the same energy model (structure, RDF, kinetic temperature) before trusting it for production. See [MACE, NequIP & FlashMD](../calculators/foundation.md#flashmd) for the calculator side.

## Building your own loop

Any ASE dynamics object works with GOAL's observers and run loop. This is how you run NVE, Nosé–Hoover chains, Bussi or NPT:

```python
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from goal.md import CalculatorFactory, MoleculeFactory, run_dynamics, setup_observers

atoms = MoleculeFactory.create("from_file", path="equilibrated.traj")    # last frame, with velocities
atoms.calc = CalculatorFactory.create("goal_model", checkpoint="path/to/best.simurgh")

dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)                    # NVE
obs = setup_observers(
    dyn, atoms,
    trajectory_file="outputs/md/nve/nve.traj",
    log_file="outputs/md/nve/nve.log",
    log_interval=10,
    collect_energies=True,
)
result = run_dynamics(dyn, steps=20_000, chunk_size=500, label="NVE",
                      energy_collector=obs["energy_collector"])
obs["trajectory"].close()
```

An NVE run from an equilibrated NVT state is the standard check of a force field's energy conservation. Drift in `Etot` in the log points to a time step that is too large, or to a non-conservative model (for example a direct-force head).

### Observers

`goal.md.core.observers` provides three callables that you attach with `dyn.attach(obs, interval=n)`, plus one function that wires them up for you.

#### `setup_observers(...)`

| Argument | Default | Meaning |
| --- | --- | --- |
| `dynamics`, `atoms` | — | The integrator and its system. |
| `trajectory_file` | `None` | Attach a `TrajectoryObserver` if set. Parent directories are created. |
| `log_file` | `None` | Attach an `MDLogObserver` if set. Parent directories are created. |
| `log_interval` | `10` | Interval for the log and energy collector, and the trajectory's default interval. |
| `trajectory_interval` | `None` | Separate trajectory interval (falls back to `log_interval`). |
| `log_stress` | `False` | Add the stress tensor to the log (requires a calculator that returns stress and a cell). |
| `log_per_atom` | `True` | Log energies per atom. |
| `trajectory_mode` | `"w"` | `"w"` overwrites, `"a"` appends. |
| `log_mode` | `"a"` | The log **appends** by default, so re-running in the same directory adds a new header and continues the file. |
| `collect_energies` | `False` | Attach an `EnergyCollector`. |

It returns a dict containing whichever of `"trajectory"`, `"log"` and `"energy_collector"` were created.

Use `trajectory_interval` to write frames less often than log lines. Long ML runs at 0.5 fs with a frame every step produce very large files.

```python
obs = setup_observers(dyn, atoms, trajectory_file="md.traj", log_file="md.log",
                      log_interval=10, trajectory_interval=200)
```

#### The observer classes

- **`TrajectoryObserver(atoms, path, mode="w")`** writes the current frame to an ASE `Trajectory`, including positions, momenta, cell and any calculator results already computed (energy, forces). Call `.close()` when you finish, or rely on garbage collection.
- **`MDLogObserver(dynamics, atoms, path, header=True, stress=False, per_atom=True, mode="a")`** wraps `ase.md.MDLogger`. With `per_atom=True` the columns are:

  ```text
  Time[ps]    Etot/N[eV]   Epot/N[eV]   Ekin/N[eV]    T[K]
  0.0000        -438.8616    -438.8900       0.0284   219.4
  ```

- **`EnergyCollector(atoms)`** appends `atoms.get_potential_energy()` (or `nan` if that fails) to `.energies` and a running call counter to `.steps`. `.steps` counts calls, not MD steps; multiply by the interval to get MD steps.

### The run loop: `run_dynamics`

```python
run_dynamics(dynamics, steps, chunk_size=100, restart_thermostat=False,
             temperature_K=300.0, progress=True, label="MD", energy_collector=None) -> RunResult
```

The loop calls `dynamics.run(n)` in chunks of `chunk_size` until `steps` steps are done. With `rich` installed and `progress=True`, it updates a progress bar (elapsed and remaining time) after each chunk. Chunking does not change the trajectory, because ASE observers keep their step counting across `run()` calls.

`restart_thermostat=True` re-draws all velocities from a Maxwell–Boltzmann distribution at `temperature_K` after every chunk except the last. This is a heavy-handed, Andersen-like reset. It is useful for driving a badly-started structure quickly towards the target temperature, or for decorrelated sampling of configurations. It **destroys dynamical information** (velocity autocorrelations, diffusion, vibrational spectra), so keep it off for production.

## Thermostats, friction and time steps

### Choosing a time step

| System | Typical time step |
| --- | --- |
| Organic molecules with explicit H, ML or DFT forces | 0.5 fs (safe); 1.0 fs is often fine at ≤ 300 K |
| No hydrogens / heavy atoms only (metals, oxides) | 1–2 fs, more for very heavy elements (check with NVE) |
| High temperature (≥ 600 K) or bond breaking | 0.25–0.5 fs |
| Hydrogen masses repartitioned / constrained | up to 2 fs |
| FlashMD | the model's trained stride (e.g. 16 fs) |

The fastest motion (X–H stretches, period ≈ 10 fs) needs roughly 10–20 steps per period. The practical test is a short NVE run: if `Etot/N` in the log drifts or jumps, halve the time step. ML potentials can be *less* forgiving than DFT here, because a model may behave badly in compressed geometries it never saw in training. Atoms that come too close is a common failure mode of an over-large time step with an ML potential.

### Choosing the Langevin friction

Langevin dynamics adds a friction force $-\gamma m v$ and matching random kicks. The coupling time is $\tau = 1/\gamma$.

- **`Simulation` defaults to `friction=0.5` fs⁻¹** ($\tau$ = 2 fs). That is very strong coupling: the system reaches temperature quickly and stays robustly NVT, but the motion is heavily damped. Use it only for fast equilibration or pure configurational sampling.
- For **sampling with realistic dynamics**, use 0.001–0.01 fs⁻¹ ($\tau$ = 100 fs–1 ps). 0.01 fs⁻¹ is a common choice for molecules, and it is what the MTS configs use.
- For **dynamical properties** (diffusion, IR/VDOS spectra), equilibrate with Langevin and then switch to NVE, or use a very weak friction.

A typical two-stage protocol:

```python
sim = Simulation(atoms, calc, steps=4000, timestep_fs=0.5, friction=0.1,   # 2 ps, strong coupling
                 output_dir="outputs/md/run", trajectory_file="equil.traj", log_file="equil.log")
sim.run()

prod = Simulation(atoms, calc, steps=40_000, timestep_fs=0.5, friction=0.01,  # 20 ps production
                  initialise_velocities=False,                              # keep equilibrated velocities
                  output_dir="outputs/md/run", trajectory_file="prod.traj", log_file="prod.log",
                  log_interval=20)
prod.run()
```

### Other ensembles

ASE ships the alternatives below. Combine them with `setup_observers` + `run_dynamics` as in [Building your own loop](#building-your-own-loop), or [register them](extending.md#custom-dynamics) under a key:

| Ensemble | ASE class |
| --- | --- |
| NVE | `ase.md.verlet.VelocityVerlet` |
| NVT, stochastic velocity rescaling | `ase.md.bussi.Bussi` |
| NVT, Nosé–Hoover chain | `ase.md.nose_hoover_chain.NoseHooverChainNVT` |
| NVT, Andersen | `ase.md.andersen.Andersen` |
| NPT | `ase.md.npt.NPT`, `ase.md.nptberendsen.NPTBerendsen`, `ase.md.melchionna.MelchionnaNPT` |

NPT needs a calculator that returns stress. Check the calculator page for the backend you use.

## Configs

| File | Purpose |
| --- | --- |
| [`src/goal/configs/md/dynamics/langevin_ase.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/dynamics/langevin_ase.yaml) | `langevin_ase` template (`timestep_fs: 1.0`, `temperature_K: 300`, `friction_per_fs: 0.05`, `fixcm: true`). Instantiate with `atoms=`. |
| [`src/goal/configs/md/dynamics/langevin_flashmd.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/dynamics/langevin_flashmd.yaml) | `langevin_flashmd` template (`timestep: 16`, `temperature_K: 300`, `time_constant: 100.0`, `model_name: pet-omatpes-v2`, `device: null`). Instantiate with `atoms=`. |
| [`src/goal/configs/md/simulations/langevin_with_model.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/simulations/langevin_with_model.yaml) | Default config of `goal-simulate`. |

For the multiple-time-step integrator (RESPA), see [MTS overview](../mts/overview.md) and [RESPA integrator](../mts/respa.md).
