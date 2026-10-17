# Usage patterns

`goal.md` can be driven in several ways, from a few lines in a notebook to a version-controlled YAML file run on a cluster. This page shows the common patterns end to end, with scripts you can copy. Each pattern says when it is the right choice.

| Pattern | Best for |
| --- | --- |
| [1. Library mode](#pattern-1-library-mode) | Notebooks, exploration, custom protocols |
| [2. A trained goal.ml model](#pattern-2-using-a-trained-goalml-model) | Running MD with your own potential |
| [3. Config-first](#pattern-3-config-first) | Reproducible production runs, batches, the CLI |
| [4. Data conversion](#pattern-4-data-conversion-workflows) | Moving structures between MD and goal.ml |
| [5. Multi-model comparison](#pattern-5-multi-model-comparison) | Choosing between checkpoints or force fields |

All snippets assume you are in the repository root with a GOAL environment that includes the MD module active (`pixi shell -e cuda-full`, or prefix commands with `pixi run -e cuda-full`; `cpu-full` / `rocm-full` work the same way).

## Pattern 1: library mode

In library mode you call the factories and classes directly. Hydra and the CLI are not involved, and every intermediate object is a normal ASE object you can inspect.

### With `Simulation` (Langevin NVT)

```python
from goal.md import CalculatorFactory, MoleculeFactory, Simulation

atoms = MoleculeFactory.create("from_smiles", smiles="CC(C)=O")          # acetone
calc = CalculatorFactory.create("tblite", method="GFN2-xTB", charge=0)    # GFN2-xTB, in the -full envs

sim = Simulation(
    atoms, calc,
    steps=10_000, temperature_K=300.0, timestep_fs=0.5, friction=0.01,
    log_interval=20, trajectory_file="acetone.traj", log_file="acetone.log",
    output_dir="outputs/md/acetone_gfn2",
)
result = sim.run()

import matplotlib.pyplot as plt
plt.plot(result.energy_history)
plt.xlabel(f"sample (every {sim.log_interval} steps)")
plt.ylabel("E_pot / eV")
plt.show()
```

### With an explicit integrator

`Simulation` always uses ASE Langevin. When you want a different ensemble, or the FlashMD integrator, build the dynamics yourself and reuse GOAL's observers and progress loop:

```python
from ase import units
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from goal.md import CalculatorFactory, MoleculeFactory, run_dynamics, setup_observers

atoms = MoleculeFactory.create("from_smiles", smiles="CC(C)=O")
atoms.calc = CalculatorFactory.create("tblite")

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)                                     # remove net momentum

dyn = Bussi(atoms, timestep=0.5 * units.fs, temperature_K=300, taut=100 * units.fs)
obs = setup_observers(dyn, atoms,
                      trajectory_file="outputs/md/acetone_bussi/md.traj",
                      log_file="outputs/md/acetone_bussi/md.log",
                      log_interval=20, collect_energies=True)
result = run_dynamics(dyn, steps=10_000, chunk_size=500, label="Bussi NVT",
                      energy_collector=obs["energy_collector"])
```

The registered integrators are available the same way through `DynamicsFactory.create("langevin_ase" | "langevin_flashmd", atoms, ...)`. Note the [unit convention](dynamics.md#langevin_ase) of `langevin_ase`.

**When to use library mode:** prototyping, protocols that change between stages (heat → equilibrate → produce), anything that needs Python logic between runs, and integrators other than Langevin.

## Pattern 2: using a trained goal.ml model

Point the `goal_model` calculator at the `.simurgh` archive (or checkpoint directory) that training produced:

```python
from goal.md import CalculatorFactory, MoleculeFactory, Simulation
from ase.optimize import BFGS

CKPT = "logs/simurgh_gmd26/runs/2026-10-01_12-00-00_run/checkpoints/best_val_forces_mae.simurgh"

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
atoms.calc = CalculatorFactory.create("goal_model", checkpoint=CKPT, device="cuda")

# Single point
print("E =", atoms.get_potential_energy(), "eV")
print("max |F| =", abs(atoms.get_forces()).max(), "eV/Å")

# Relax, then run MD from the relaxed structure
BFGS(atoms, logfile=None).run(fmax=0.05)
Simulation(atoms, atoms.calc, steps=20_000, timestep_fs=0.5, friction=0.01,
           trajectory_file="md.traj", log_file="md.log",
           output_dir="outputs/md/ethanol_goal").run()
```

The same calculator can be obtained with `goal.md.adapters.load_goal_calculator(CKPT, device="cuda")`. For checkpoint formats, `dtype`, multi-head models and other details, see [Bridge to goal.ml](ml-bridge.md) and [Trained GOAL models](../calculators/goal-model.md).

## Pattern 3: config-first

In the config-first pattern, one YAML file fully describes a run: the molecule, the calculator, and every `SimulationConfig` field. You run it from Python or from the CLI and override values on the command line. The file is your record of what was run.

### Write the config

`Simulation.from_config` reads a **flat** schema: a `molecule:` mapping, a `calculator:` mapping, and the `SimulationConfig` fields at the top level. Put the file in your own config directory, for example `configs/md_runs/ethanol_goal.yaml`:

```yaml
# @package _global_
# Ethanol, 300 K Langevin NVT with a trained GOAL model.

molecule:
  method: from_smiles          # popped; the rest goes to the builder
  smiles: CCO

calculator:
  type: goal_model             # popped (`key` also accepted); the rest goes to the builder
  checkpoint: ???              # mandatory, given on the command line
  device: cuda

steps: 20000
temperature_K: 300.0
timestep_fs: 0.5
friction: 0.01                 # 1/fs
log_interval: 20
trajectory_file: md.traj
log_file: md.log
output_dir: outputs/md/ethanol_goal
show_progress: true
```

The first line, `# @package _global_`, matters. Hydra places a config loaded by a path such as `md_runs/ethanol_goal` under the key `md_runs:` unless the file declares itself global. Without that line, `from_config` finds no `molecule` and fails with `missing 1 required positional argument: 'smiles'`. The bundled `src/goal/configs/md/simulations/langevin_with_model.yaml` has this problem; see [Configuration](../reference/configuration.md#writing-a-simulation-config-that-works).

### Run it from Python

```python
from pathlib import Path
from hydra import compose, initialize_config_dir
from goal.md import simulate_from_config

with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(
        "md_runs/ethanol_goal",
        overrides=[
            "calculator.checkpoint=logs/my_exp/runs/2026-10-01_12-00-00_run/checkpoints/best_val_forces_mae.simurgh",
            "temperature_K=400",
        ],
    )

result = simulate_from_config(cfg)          # = Simulation.from_config(cfg).run()
print(result)
```

`initialize_config_dir` requires an **absolute** path, hence `.resolve()`.

You can skip YAML entirely and build the same structure in code. This is handy for parameter sweeps:

```python
from omegaconf import OmegaConf
from goal.md import simulate_from_config

for T in (250, 300, 350, 400):
    cfg = OmegaConf.create({
        "molecule": {"method": "from_smiles", "smiles": "CCO"},
        "calculator": {"type": "goal_model", "checkpoint": "path/to/best.simurgh", "device": "cuda"},
        "steps": 20_000, "temperature_K": T, "timestep_fs": 0.5, "friction": 0.01,
        "trajectory_file": "md.traj", "log_file": "md.log",
        "output_dir": f"outputs/md/ethanol_T{T}",
    })
    simulate_from_config(cfg)
```

### Run it from the CLI

```bash
pixi run -e cuda-full goal-simulate --config-path "$PWD/configs/md_runs" --config-name ethanol_goal \
    calculator.checkpoint=logs/my_exp/runs/2026-10-01_12-00-00_run/checkpoints/best_val_forces_mae.simurgh \
    temperature_K=400 output_dir=outputs/md/ethanol_400K
```

Hydra multirun gives you a sweep with one command:

```bash
pixi run -e cuda-full goal-simulate --config-path "$PWD/configs/md_runs" --config-name ethanol_goal -m \
    calculator.checkpoint=path/to/best.simurgh \
    temperature_K=250,300,350 'output_dir=outputs/md/ethanol_${temperature_K}K'
```

See [CLI](../reference/cli.md) for why `--config-path` must be absolute and for the other options.

**When to use config-first:** production runs, anything you will report or re-run, batch jobs on a cluster, and sweeps.

## Pattern 4: data conversion workflows

`goal.md.adapters` converts between ASE `Atoms` and goal.ml `AtomicGraph`. This lets you take MD frames into goal.ml (for evaluation, feature extraction or dataset building) and take goal.ml structures back into MD.

```python
from ase.io import read
from goal.md.adapters import ase_atoms_from_atomic_graph, atomic_graph_from_ase

frames = read("outputs/md/ethanol_goal/md.traj", index="::10")    # every 10th frame

# ASE → goal.ml: radius graph with the model's cutoff, carrying labels if you have them
graphs = [
    atomic_graph_from_ase(f, cutoff=5.0, energy=f.get_potential_energy(), forces=f.get_forces())
    for f in frames
]
print(graphs[0])

# goal.ml → ASE: positions, numbers, cell and PBC only (no labels, no velocities)
atoms_back = ase_atoms_from_atomic_graph(graphs[0])
```

The most common conversion is **MD frames → training data** for fine-tuning. For that you usually do not need `AtomicGraph` at all: label the frames with a reference method and write `extxyz`, which goal.ml loads directly. See the complete example in [Bridge to goal.ml](ml-bridge.md#3-analyse) and the formats in [Loading data](../../ml/data/loading.md).

## Pattern 5: multi-model comparison

When you have several checkpoints (different architectures, seeds or fine-tuning rounds) or want to compare a trained model with a foundation model, compare them on the same reference frames **and** on MD stability. Low force error on a test set does not guarantee a stable trajectory.

```python
import numpy as np
from ase.io import read
from goal.md import CalculatorFactory, MoleculeFactory, Simulation

# Reference frames with DFT energies/forces stored (extxyz or traj)
ref_frames = read("data/ethanol_test.extxyz", index=":")
E_ref = np.array([f.get_potential_energy() for f in ref_frames])
F_ref = [f.get_forces() for f in ref_frames]

candidates = {
    "simurgh_v1": dict(key="goal_model", checkpoint="logs/a/runs/r1/checkpoints/best_val_forces_mae.simurgh", device="cuda"),
    "simurgh_v2": dict(key="goal_model", checkpoint="logs/a/runs/r2/checkpoints/best_val_forces_mae.simurgh", device="cuda"),
    "gfn2_xtb":   dict(key="tblite", method="GFN2-xTB"),
}

for name, spec in candidates.items():
    spec = dict(spec)
    calc = CalculatorFactory.create(spec.pop("key"), **spec)

    # 1) accuracy on reference frames
    E, F_err = [], []
    for f, f_ref in zip(ref_frames, F_ref):
        g = f.copy(); g.calc = calc
        E.append(g.get_potential_energy())
        F_err.append(g.get_forces() - f_ref)
    E = np.array(E)
    dE = (E - E.mean()) - (E_ref - E_ref.mean())         # compare relative energies
    e_mae = np.mean(np.abs(dE)) / len(ref_frames[0]) * 1000
    f_rmse = np.sqrt(np.mean(np.concatenate(F_err) ** 2))

    # 2) stability: short NVT run at an elevated temperature
    atoms = ref_frames[0].copy()
    res = Simulation(atoms, calc, steps=4_000, temperature_K=500, timestep_fs=0.5,
                     friction=0.01, show_progress=False,
                     trajectory_file=f"{name}.traj", output_dir="outputs/md/compare").run()
    traj = read(f"outputs/md/compare/{name}.traj", index=":")
    dmin = min(np.min(a.get_all_distances()[np.triu_indices(len(a), 1)]) for a in traj)

    print(f"{name:12s}  E MAE {e_mae:6.2f} meV/atom   F RMSE {f_rmse:.3f} eV/Å   "
          f"min d {dmin:.2f} Å   final T {res.final_temperature:.0f} K")
```

Notes on this comparison:

- Energies are compared after subtracting each method's mean, because different methods (or models with different atomic reference energies) are offset by a constant.
- A minimum interatomic distance far below a physical bond length (for example < 0.7 Å for X–H) means the trajectory has collapsed, whatever the test-set error.
- Run the stability test at a temperature higher than you plan to use for production. This exposes holes in the potential's training coverage more quickly.
- To compare foundation models, swap the specs, for example `dict(key="mace", model_name_or_path="medium")` or `dict(key="upet", model="pet-mad-s")`. See [Calculators](../calculators/overview.md) for each builder's arguments.
