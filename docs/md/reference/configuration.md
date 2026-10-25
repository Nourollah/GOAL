# Configuration

This page documents every YAML file under [`src/goal/configs/md/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/configs/md): what each one sets, which of them work as-is with the current code, and how to write a simulation config that `goal-simulate` and `simulate_from_config` accept. It also covers composition and command-line overrides.

> [!IMPORTANT]
> The code is the reference. `Simulation.from_config` (and `MTSSimulation.from_config` for MTS) defines the schema actually read. Several bundled files predate that schema. They are still useful as **templates** for each backend's arguments, but they cannot all be composed or instantiated as written. Each section below says which files work.

## Layout

The configs ship inside the package (`goal/configs/md/` when installed). `goal-simulate` and `goal-simulate-mts` find them from any directory. The Python snippets on these pages load templates by their clone path (`"src/goal/configs/md/..."`); outside a clone, build the path from `goal.CONFIGS_DIR` instead, for example `Path(goal.CONFIGS_DIR) / "md/calculators/vasp.yaml"`.

```text
src/goal/configs/md/
├── calculators/            # one file per CalculatorFactory backend
│   ├── goal_model.yaml     ├── mace.yaml     ├── flashmd.yaml   ├── upet.yaml   ├── tblite.yaml
│   ├── cp2k.yaml           ├── cp2k_pw.yaml  ├── espresso.yaml  └── vasp.yaml
├── dynamics/
│   └── langevin_ase.yaml   └── langevin_flashmd.yaml
├── molecules/
│   ├── ethanol.yaml
│   └── from_file.yaml
├── simulations/
│   ├── langevin_with_model.yaml      # default config of goal-simulate
│   └── batch.yaml                    # defaults for goal-simulate-batch --config (not Hydra)
└── mts/                              # multiple-time-step templates (see below)
    ├── base_calculators/orca_reference.yaml
    ├── ml_models/learner.yaml
    ├── dynamics/langevin_mts.yaml
    └── simulations/{mts_respa, online_learner, orca_ml_learner}.yaml
```

## The schema `Simulation.from_config` reads

```yaml
molecule:                  # mapping; builder from `key` (or `type`/`method`; default: from_smiles)
  key: from_smiles
  smiles: CCO

calculator:                # mapping; builder from `key` (or `type`; default: goal_model)
  key: goal_model
  checkpoint: path/to/best.simurgh

dynamics:                  # mapping; builder from `key` (or `method`; default: langevin_ase)
  temperature_K: 300.0
  timestep_fs: 1.0           # fs
  friction: 0.5              # 1/fs

# SimulationConfig fields, all optional, at the TOP level
steps: 1000
chunk_size: 100
restart_thermostat: false
log_interval: 10
trajectory_file: null      # relative → under output_dir
log_file: null             # relative → under output_dir
output_dir: outputs/md
show_progress: true
initialise_velocities: true
```

Rules worth knowing:

- Any other top-level key is **silently ignored**, including `output_file`; `num_steps` is accepted as an alias of `steps`.
- `molecule:` / `calculator:` missing at the top level is an error that names the keys it did find (usually a missing `# @package _global_`).
- Hydra's `_target_` / `_convert_` / `_recursive_` / `_partial_` keys are dropped, so the group templates can be placed under `molecule:` / `calculator:` / `dynamics:` unchanged (that is how the default config composes them).
- After the builder name is taken from `calculator`, every remaining key must be a keyword argument of the builder's `build` method. `method` is never the builder name for calculators: it is a real argument of `tblite` and `cp2k`. A misspelt key fails loudly with `TypeError: ...build() got an unexpected keyword argument`. Each backend's arguments are listed on its [calculator page](../calculators/overview.md).

### Writing a simulation config that works

Hydra places a config by its path: `md/simulations/foo.yaml` loaded by name ends up under `cfg.md.simulations`, not at the root. The schema above needs the keys at the root, so a simulation config in a sub-directory must start with `# @package _global_`:

```yaml
# @package _global_
# configs/md_runs/ethanol_goal_nvt.yaml  (your own config directory)

molecule:
  key: from_smiles
  smiles: CCO

calculator:
  key: goal_model
  checkpoint: ???              # required on the command line
  device: cuda

dynamics:
  key: langevin_ase
  temperature_K: 300.0
  timestep_fs: 0.5
  friction: 0.01
  

steps: 20000
log_interval: 20
trajectory_file: md.traj
log_file: md.log
output_dir: outputs/md/ethanol_goal_nvt
```

```bash
pixi run -e cuda-full goal-simulate --config-path "$PWD/configs" --config-name md_runs/ethanol_goal_nvt \
    calculator.checkpoint=path/to/best.simurgh
```

You can check what Hydra composed without running anything:

```bash
pixi run -e cuda-full goal-simulate --config-path "$PWD/configs" --config-name md_runs/ethanol_goal_nvt --cfg job
```

The keys you expect (`molecule`, `calculator`, `dynamics`, …) must appear at the top level of the printout.

### How the bundled `langevin_with_model.yaml` is built

[`src/goal/configs/md/simulations/langevin_with_model.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/simulations/langevin_with_model.yaml), the default config of `goal-simulate`, composes the group templates into the schema with package overrides:

```yaml
# @package _global_
defaults:
  - /md/molecules@molecule: ethanol
  - /md/calculators@calculator: goal_model
  - /md/dynamics@dynamics: langevin_ase
  - _self_

steps: 1000
chunk_size: 100
log_interval: 10
...
```

`/md/molecules@molecule: ethanol` places `molecules/ethanol.yaml` under `molecule:` (`goal_model.yaml` under `calculator:` and `langevin_ase.yaml` under `dynamics:`); their `_target_` keys are ignored by `Simulation.from_config`. Use the same pattern to reuse templates in your own configs. Before October 2026 this file lacked the header and the package overrides, so `goal-simulate` with its default config failed.

## Calculator templates

Each file is a `_target_: goal.md.core.calculator_factory.CalculatorFactory.create` template with `_convert_: none` (`tblite.yaml` uses `_convert_: all`). Every file names its backend with `key:`, so each one can be passed straight to `hydra.utils.instantiate` or placed under `calculator:` in a simulation config. The [calculator pages](../calculators/overview.md) document every argument.

| File | Backend key | Key field | Main settings | Works with `instantiate` |
| --- | --- | --- | --- | --- |
| `goal_model.yaml` | `goal_model` | `key` | `checkpoint: ???`, `select: best`, `use_ema: true`, `device: cpu`, `head: null` | Yes |
| `flashmd.yaml` | `flashmd` | `key` | `model_name: pet-omatpes-v2`, `device: null` | Yes |
| `mace.yaml` | `mace` | `key` | `model_name_or_path: small` (`small`/`medium`/`large` or a path) | Yes |
| `upet.yaml` | `upet` | `key` | `model: pet-mad-s`, `version`, `checkpoint_path`, `variants`, `device: cuda`, `non_conservative: false` | Yes |
| `tblite.yaml` | `tblite` | `key` | `method: GFN2-xTB` (here `method` is the xTB Hamiltonian), `charge`, `multiplicity`, `accuracy`, `electronic_temperature`, `max_iterations`, `solvation`, `verbosity` | Yes |
| `cp2k.yaml` | `cp2k` | `key` | GPW (Quickstep): `cutoff_ry: 400`, `xc: PBE`, `basis_set: DZVP-MOLOPT-SR-GTH`, `pseudo_potential: GTH-PBE`, `preset: molecular`, `set_pos_file: true`, MPI/OMP | Yes |
| `cp2k_pw.yaml` | `cp2k` | `key` | Pure plane-wave via SIRIUS: `method: sirius`, `pw_cutoff`/`gk_cutoff` (bohr⁻¹), `kpts: [1,1,1]`, `smearing_width` | Yes |
| `espresso.yaml` | `espresso` | `key` | `pseudo_dir`, `pseudopotentials: {}` (**required**), `preset`, `kpts`, `input_data` (`ecutwfc: 80`, `ecutrho: 640` Ry) | Yes |
| `vasp.yaml` | `vasp` | `key` | `pp_path`, `preset`, `parameters` (`xc: PBE`, `encut: 520`, `ediff: 1e-6`, single point) | Yes |

Calculators with no template file (`orca`, `nequip`) are still available through `type:` in a simulation config. The ORCA template for MTS lives in `mts/base_calculators/`.

Use the templates from Python with `instantiate`:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

calc = instantiate(OmegaConf.load("src/goal/configs/md/calculators/mace.yaml"), model_name_or_path="medium")
```

To put a template into a simulation config, compose it with `- /md/calculators@calculator: upet` in `defaults:`, or write the mapping inline (`_target_`/`_convert_` are optional there):

```yaml
calculator:
  key: upet
  model: pet-mad-s
  device: cuda
```

## Molecule templates

| File | Contents | Notes |
| --- | --- | --- |
| `molecules/ethanol.yaml` | `key: from_smiles`, `smiles: CCO` | `_target_` template |
| `molecules/from_file.yaml` | `key: from_file`, `path: ???` | `_target_` template |

Both work with `instantiate` and can be placed under `molecule:` unchanged (the default config does exactly that with `ethanol.yaml`). The molecule-related keys are documented in [Molecules](../guide/molecules.md#molecule-configs).

## Dynamics templates

| File | Contents | Notes |
| --- | --- | --- |
| `dynamics/langevin_ase.yaml` | `key: langevin_ase`, `timestep_fs: 1.0`, `temperature_K: 300.0`, `friction_per_fs: 0.05`, `fixcm: true` | Instantiate it with `atoms=`. Femtoseconds throughout (before October 2026 it passed `timestep: 1.0` raw, ≈ 10.18 fs). See [Dynamics](../guide/dynamics.md#langevin_ase). |
| `dynamics/langevin_flashmd.yaml` | `key: langevin_flashmd`, `timestep: 16`, `temperature_K: 300.0`, `time_constant: 100.0`, `model_name: pet-omatpes-v2`, `device: null` | Instantiate it with `atoms=`. See [Dynamics](../guide/dynamics.md#langevin_ase). |

### Loading any group template from Python

Every template names its factory with `key:`, so `hydra.utils.instantiate` builds it directly; pass the dynamics template its `atoms` and override anything as keyword arguments:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

def template(path):
    return OmegaConf.load(f"src/goal/configs/md/{path}")   # outside a clone: goal.CONFIGS_DIR

atoms = instantiate(template("molecules/ethanol.yaml"))
atoms.calc = instantiate(template("calculators/goal_model.yaml"), checkpoint="path/to/best.simurgh")
dyn = instantiate(template("dynamics/langevin_ase.yaml"), atoms=atoms, timestep_fs=0.5)
dyn.run(1000)
```

`MoleculeFactory.from_config(mapping)` and `CalculatorFactory.from_config(mapping)` do the same for plain dicts and accept `key`, `type` (and, for molecules, `method`).

## Composition and overrides

The configs use Hydra 1.3. These are the override forms you will use most with `goal-simulate` (see [CLI](cli.md)):

| Syntax | Effect | Example |
| --- | --- | --- |
| `key=value` | Change an existing key | `temperature_K=400` |
| `a.b=value` | Change a nested key | `calculator.device=cuda:1` |
| `+key=value` | Add a key that is not in the config | `+calculator.head=dft` |
| `++key=value` | Add or override | `++initialise_velocities=false` |
| `'+k={a: 1}'` | Add a whole mapping (after `'~k'` to replace one) | `'~calculator' '+calculator={type: tblite}'` |
| `~key` | Delete a key (quote it in zsh/bash) | `'~calculator.head'` |
| `-m k=a,b,c` | Multirun sweep (one job per value) | `-m temperature_K=250,300,350` |
| `'k=${other}'` | Interpolation (single-quote it in the shell) | `'output_dir=outputs/md/T${temperature_K}'` |
| `--cfg job` | Print the composed config and exit | |

Notes:

- **Quote values that contain `=`, spaces or parentheses.** Lightning checkpoint names (`best_val_forces_mae=0.0110_epoch=1204.ckpt`) and SMILES with branches need quoting, for example `'molecule.smiles="CC(=O)O"'`.
- **`???` marks a mandatory value.** Composition succeeds, but reading the value raises `MissingMandatoryValue` until you set it.
- **Hydra working directory.** Each run gets a Hydra output directory (`outputs/<date>/<time>/` by default, containing `.hydra/` with the composed config and the log). The process does **not** `chdir` into it, so relative `output_dir`, `trajectory_file` and checkpoint paths are resolved against the directory you launched from. The `.hydra/config.yaml` copy is your exact record of the run.

To reuse group templates through `defaults:` in your own config, place them under the package the schema expects with an `@` override, as the bundled default does (`- /md/calculators@calculator: tblite`). A single self-contained `# @package _global_` file, as shown above, works just as well.

## MTS configs

`src/goal/configs/md/mts/` holds the multiple-time-step templates. `goal-simulate-mts` uses `md/mts/simulations/mts_respa` as its default config. The full description, including the RESPA and online-learning parameters, is in [MTS overview](../mts/overview.md).

| File | Purpose | Status with current code |
| --- | --- | --- |
| `simulations/mts_respa.yaml` | RESPA with an ML `fast_calculator` (`goal_model`) and an ORCA `slow_calculator`; `mts:` block (`timestep_fs: 1.0`, `mts_ratio: 4`, `total_steps: 10000`, `friction: 0.01`, outputs) | Default of `goal-simulate-mts`; needs `fast_calculator.checkpoint` and `slow_calculator.orca_path` (or `slow_calculator=null`) |
| `simulations/online_learner.yaml` | Online learning: `base_calculator` (ORCA), `ml_calculator`, `learner:` (buffer, fine-tuning, `policy: threshold`), `mts:` block (`mts_ratio: 1`) | Run with `goal-simulate-mts --config-name md/mts/simulations/online_learner`; builds a `QMLearnerCalculator` |
| `simulations/orca_ml_learner.yaml` | The same online-learning run composed from the three templates below via `defaults:` | Run like `online_learner.yaml` |
| `base_calculators/orca_reference.yaml` | ORCA template (`key: orca`, `orca_path: ???`, ωB97M-V/def2-TZVPD `keywords`, charge, multiplicity) | `_target_` template |
| `ml_models/learner.yaml` | `goal_model` template (`checkpoint: ???`, `device: cuda`) | `_target_` template |
| `dynamics/langevin_mts.yaml` | `timestep: 1.0`, `temperature_K: 300.0`, `friction: 0.5` | Plain values, not read by any entry point |
