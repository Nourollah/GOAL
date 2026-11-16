---
name: goal-run-md
description: Runs molecular dynamics, geometry optimisation and batched simulations with GOAL's MD module (goal.md). Covers the goal-simulate, goal-simulate-batch and goal-simulate-mts commands, the Simulation and CalculatorFactory APIs, calculator backends (trained GOAL checkpoints, GFN2-xTB through tblite, ORCA, CP2K, Quantum ESPRESSO, VASP, foundation models), thermostats, NVE runs, time steps, friction and observers. Use when the user wants to simulate or relax a molecule, compare calculators on trajectories, or an MD run explodes.
license: MIT
---

# Molecular dynamics with GOAL

References: `docs/md/guide/usage-patterns.md`, `docs/md/guide/dynamics.md`, `docs/md/reference/cli.md`, `docs/integrations/batched_simulation.md`, `docs/md/calculators/overview.md`, `docs/md/reference/troubleshooting.md`.

Use the `-full` environments (`pixi run -e cuda-full ...`); they add the MD extras such as tblite.

## Library pattern

```python
from goal.md import CalculatorFactory, MoleculeFactory, Simulation

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
calc = CalculatorFactory.create("goal_model", checkpoint="models/my_model.simurgh", device="cuda")
# calc = CalculatorFactory.create("tblite", method="GFN2-xTB", charge=0)
sim = Simulation(atoms, calc, dynamics_key="langevin_ase", steps=20_000, temperature_K=300.0,
                 timestep_fs=0.5, friction_per_fs=0.01, log_interval=20,
                 trajectory_file="md.traj", log_file="md.log", output_dir="outputs/md/ethanol_300K")
result = sim.run()
```

`checkpoint` accepts a `.simurgh` archive, a checkpoint directory or a `.ckpt` file. Units: `timestep_fs` in fs and `friction_per_fs` in fs⁻¹; the bare `timestep` and `friction` arguments are passed to ASE unchanged, in ASE units (1 ASE time unit ≈ 10.18 fs), so `friction=0.01` is about 0.001 fs⁻¹.

## CLI pattern

```bash
pixi run -e cuda-full goal-simulate calculator.checkpoint=models/my_model.simurgh calculator.device=cuda \
    'molecule.smiles="CC(=O)O"' steps=5000 output_dir=outputs/md/acetic_acid
pixi run -e cuda-full goal-simulate-batch --checkpoint models/my_model.simurgh \
    --structures frames.xyz --steps 2000 --temperature 300 --timestep 0.5      # many structures at once
pixi run -e cuda-full goal-simulate-batch --checkpoint models/my_model.simurgh \
    --structures a.xyz b.xyz --mode relax --fmax 0.02 --backend ase             # batched relaxation
```

Quote SMILES that contain parentheses or `=`. `goal-simulate` takes its thermostat from the `dynamics:` section, which defaults to the `langevin_ase` template (`src/goal/configs/md/dynamics/langevin_ase.yaml`: 1 fs, `friction_per_fs: 0.05`, 300 K); override it with `dynamics.timestep_fs=0.5 dynamics.friction_per_fs=0.01 dynamics.temperature_K=300`. Hydra multirun (`-m dynamics.temperature_K=250,300,350`) gives temperature sweeps; a top-level `temperature_K=` is rejected.

## Choosing settings

- Time step: 0.5 fs for organic molecules with explicit hydrogens; 0.25–0.5 fs above 600 K or when bonds break. If `Etot/N` drifts or jumps in a short NVE run, halve it.
- Friction: with no friction argument, `Simulation` (key `langevin_ase`) uses 0.5 ASE units ≈ 0.049 fs⁻¹ and a 1 fs time step; the `langevin_ase` template sets `friction_per_fs: 0.05`. Both are strong coupling. Use 0.001–0.01 fs⁻¹ for sampling with realistic dynamics.
- Dynamical properties (diffusion, spectra): equilibrate with Langevin, then continue in NVE.
- NVE: the `velocity_verlet` key is registered in `DynamicsFactory`: `Simulation(..., dynamics_key="velocity_verlet", timestep_fs=0.5)` (pass `initialise_velocities=False` to keep the velocities of an equilibrated frame), or `goal-simulate dynamics.key=velocity_verlet '~dynamics.friction_per_fs' '~dynamics.fixcm'` (it rejects the Langevin-only keys). `ase.md.verlet.VelocityVerlet` with `setup_observers` and `run_dynamics` ("Building your own loop" in `docs/md/guide/dynamics.md`) also works.
- NPT needs a calculator that returns stress.

## Multiple time stepping and QM calculators

`goal-simulate-mts` pairs a fast model with a slow QM calculator (ORCA). Do a fast-only dry run first (`slow_calculator=null mts.total_steps=100`). QM calculators are for simulation and offline data generation; never call them from inside a model or a training loop.

## When a run fails

- Temperature explodes or atoms collapse: halve the time step, test the model on compressed geometries, and check the friction and its units.
- `expected scalar type Double but found Float`: model and input dtypes differ; see the troubleshooting page.
- Nonsensical energies: check the element references and that the checkpoint was trained on this chemistry.
- Robustness comparisons between models follow a fixed protocol: `goal-md-stability-benchmark`.

## Done when

- [ ] The run wrote its trajectory and log to `output_dir`.
- [ ] Temperature and total energy behave as expected for the ensemble.
