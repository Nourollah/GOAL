# Troubleshooting

This page lists the known limitations of `goal.md` and the errors you are most likely to hit, with their causes and fixes. Errors are grouped by where they occur: installation, the CLI and configs, structures, calculators and checkpoints, devices, and the simulation itself.

## Known limitations

| Limitation | Workaround |
| --- | --- |
| A config in your own directory is not found with a relative `--config-path` | Pass an absolute `--config-path` (`"$PWD/..."`), or use `--config-dir` |
| `Simulation` always uses ASE Langevin; the `dynamics:` config section and `DynamicsFactory` are not used by it | Build the integrator yourself and use `setup_observers` + `run_dynamics` ([Dynamics](../guide/dynamics.md#building-your-own-loop)) |
| `from_db` molecule builder is a placeholder | Use `ase.db` directly |
| SMILES embedding is not seeded, so each call gives a different conformer | Generate once, save, and use `from_file` |
| `box_molecule(**kwargs)` forwards kwargs to both `set_cell` and `center` | Do not pass kwargs; adjust with ASE afterwards |
| With `torch_geometric` imported first, `flashmd`'s `metatensor` stack fails TorchScript compilation in this environment (torch 2.14, Python 3.14) | `import goal.md` still works and reports FlashMD as unavailable; import `goal.md` (or `flashmd`) before `goal.ml` in a process that needs FlashMD |

> [!NOTE]
> **Fixed in October 2026.** Earlier versions of this page listed more limitations, all now resolved:
> `goal-simulate` / `goal-simulate-mts` no longer need `typer`;
> the shipped simulation configs compose to the schema the code reads (`# @package _global_`, `key:` instead of `method:`);
> `pixi run -e cuda-full simulate` / `simulate-mts` run the commands;
> `langevin_ase` accepts `timestep_fs` / `friction_per_fs`;
> `goal_model` / `load_goal_calculator` forward `select` (best/last/epoch=N), match the model's dtype and load the EMA weights;
> `get_model_cutoff` reads archives, directories and `.ckpt` files without building the model;
> `goal-simulate-mts` runs the online learner (`online_learner.yaml`, `orca_ml_learner.yaml`), and the learner no longer calls QM on every step after switching to ML.

## Installation and environment

### `ModuleNotFoundError` for `flashmd` or `ipi`

The MD module's optional dependencies are only in the `-full` pixi environments (`cuda-full`, `cpu-full`, `rocm-full`) and in the `[md]` pip extra. A plain `cuda` / `cpu` / `rocm` environment contains the ML module only. Switch environments:

```bash
pixi install -e cuda-full
pixi run -e cuda-full goal-simulate …
```

or, in a pip/uv virtualenv, `pip install "goal-atomistic[md]"`. `goal-simulate` and `goal-simulate-mts` need nothing beyond the core install themselves; only the backends you select do. (The placeholder Typer app moved to `python -m goal.md.cli.main` and is the only part that needs `typer`.)

### `tblite not installed` / `pw.x` not found

tblite and Quantum ESPRESSO (`pw.x`, `mpirun`) are only in the `-full` pixi environments. In a plain `cuda` / `cpu` / `rocm` environment, `CalculatorFactory.create("tblite")` raises `ImportError`, and `espresso` cannot find `pw.x`. Use `pixi run -e cuda-full …`. With pip, the `md` extra installs `tblite` on x86-64 only (PyPI has no aarch64 wheels) and does not install `pw.x`. See [tblite](../calculators/tblite.md#installation) and [Quantum ESPRESSO](../calculators/espresso.md#installation).

### `MACE not installed` / e3nn errors with MACE

`mace-torch` pins `e3nn==0.4.4`, while GOAL's own models need `e3nn>=0.5`, so MACE cannot live in the main environment. Use the separate `.venv-mace` environment described in [Installation](../../getting-started/installation.md) and [MACE, NequIP & FlashMD](../calculators/foundation.md).

## CLI and configuration

### `Primary config module 'goal.md.cli.configs…' not found`

The command was run with a relative `--config-path`. Hydra interprets a relative `--config-path` relative to `simulate.py`, not your shell. The bundled configs need no `--config-path` at all (they are found from any directory); for your own config directory, pass an absolute path:

```bash
pixi run -e cuda-full goal-simulate --config-path "$PWD/configs/md_runs" --config-name <your_config> ...
```

### `A simulation config needs top-level 'molecule:' and 'calculator:' sections`

`Simulation.from_config` (and `MTSSimulation.from_config`, which needs `molecule:`) found the sections nested somewhere else. The usual cause is a config in a sub-directory (`md/simulations/...`) without `# @package _global_` on its first line, so Hydra nested everything under `md.simulations`. Every shipped simulation config has that line; add it to your own. Run with `--cfg job` to see the composed structure, and see [Configuration](configuration.md#writing-a-simulation-config-that-works).

### `got an unexpected keyword argument ...`

`Simulation.from_config` / `MTSSimulation.from_config` drop Hydra's `_target_` / `_convert_` keys and take the builder name from `key` or `type` (molecules also accept `method`), so the group templates can be placed under `calculator:` / `molecule:` unchanged. An unexpected keyword argument is therefore an argument the builder does not take, usually a misspelling. Check the [calculator page](../calculators/overview.md) for the backend. `method` is a real argument for `tblite` and `cp2k`, never a calculator's builder name.

### `MoleculeFactory.create() missing 1 required positional argument: 'key'`

You called `hydra.utils.instantiate` on a mapping that names its builder with `method:` or not at all. The shipped templates use `key:`, which `instantiate` passes on as the factory's `key` argument. Use `key:` in your own templates, or the `from_config` helpers (`MoleculeFactory.from_config`, `CalculatorFactory.from_config`), which accept every spelling.

### `mismatched input '=' expecting <EOF>` / `Could not override 'calculator'`

- The first error means an override value contains `=` (Lightning checkpoint names do) or other special characters. Quote the value: `'calculator.checkpoint="path/best=0.011_epoch=12.ckpt"'`. For SMILES: `'molecule.smiles="CC(=O)O"'`.
- `Could not override ... Key 'x' is not in struct` means you assigned a mapping (`calculator={...}`) that would merge in new keys, or set a key that does not exist. Prefix new keys with `+`, and replace a whole mapping with `'~calculator' '+calculator={type: tblite}'`.



### `MissingMandatoryValue: ... checkpoint`

A `???` field was not set. Pass it on the command line (`calculator.checkpoint=...`).

## Molecules

### `ImportError: RDKit not installed`

`from_smiles` needs RDKit. RDKit is a core dependency, so this means a broken or minimal environment: re-run `pixi install -e cuda-full`, or `pip install rdkit`.

### `ValueError: Invalid SMILES` / `Could not generate 3D conformation for SMILES`

The first error means RDKit could not parse the string; check aromaticity, charges and brackets. The second means ETKDG embedding failed, which is typical for large, strained, caged or metal-containing species. Build the structure another way (a crystal structure, `ase.build`, a QM-optimised XYZ) and load it with `from_file`.

### `AttributeError: 'list' object has no attribute ...`

`from_file` returned a list because you passed `index=":"` (or a slice). `Simulation` needs one `Atoms`; pass `index=-1` or pick a frame.

### `NotImplementedError: Database loading not yet implemented`

`from_db` is a placeholder. Use `ase.db.connect(path).get_atoms(id=...)`, or the `from_ase_db` builder from [Extending goal.md](../guide/extending.md#custom-molecule-builders).

## Calculators and checkpoints

### Missing backend packages or programs

| Message | Cause | Fix |
| --- | --- | --- |
| `FlashMD not installed. Install with: pip install flashmd` | `flashmd` missing (raised by both the `flashmd` calculator and `langevin_flashmd`) | `pip install "flashmd>=0.2.5,<0.3"` |
| `tblite not installed. Use a -full pixi environment ...` | `tblite` missing | `pixi run -e cuda-full …`, or `pip install tblite` (x86-64) |
| `CalculationFailed: Total number of electrons (N) and number unpaired electrons (M) is not compatible` | tblite read a spin from the atoms' initial magnetic moments that does not fit the charge | Pass `multiplicity=` explicitly, or zero the magnetic moments ([tblite](../calculators/tblite.md#charge-and-spin)) |
| `MACE not installed` | `mace-torch` missing | Separate MACE environment (above) |
| `NequIP not installed` | `nequip` missing | `pip install nequip` |
| `GOALCalculatorNotFound: ...` | An external code (ORCA, CP2K, pw.x, VASP), its bindings, or required data (pseudopotentials, `$VASP_PP_PATH`, `upet`) is missing | The message names what is missing. See the backend's page: [CP2K](../calculators/cp2k.md), [Quantum ESPRESSO](../calculators/espresso.md), [VASP](../calculators/vasp.md), [ORCA](../calculators/orca.md), [UPET](../calculators/upet.md) |
| `Calculator builder 'x' not registered. Available: ...` | Typo, or a custom builder whose module was not imported | Check the list; `import` your plugin module ([Extending goal.md](../guide/extending.md#making-your-keys-available-to-configs-and-the-cli)) |

### Checkpoint not found or not loadable

`GOALCalculator` chooses how to load based on the path. The messages below tell you which case you hit:

| Message | Meaning | Fix |
| --- | --- | --- |
| `Archive not found: ...` | A `.simurgh` path that does not exist | Check the path. Relative paths are resolved from the launch directory, not the Hydra run directory. |
| `Directory ... is missing config.yaml, metadata.json, or frozen_source/` | A directory that was not written by `GOALCheckpointManager` (for example the run directory instead of its `checkpoints/` sub-directory) | Point at `.../checkpoints/`, or at the `.simurgh` file inside it |
| `Unrecognised checkpoint path: ...` | Not a `.simurgh`, a directory or a `.ckpt` | Use one of the three formats ([Bridge to goal.ml](../guide/ml-bridge.md#what-you-can-load)) |
| `KeyError: 'hyper_parameters'` | A `.ckpt` that is not a full Lightning checkpoint (for example a weights-only file) | Use the run's `.simurgh` archive or checkpoint directory |
| Import or shape errors when loading a bare `.ckpt` | Legacy `.ckpt` files are rebuilt with the *current* source code, which may have changed since training | Load the `.simurgh` archive or checkpoint directory, which use the frozen source saved with the run |
| `ValueError` from the model config during reconstruction (for example `atomic_energies.mode='dataset' requires 'values' ...`) | A checkpoint written before October 2026: training injected dataset-derived values without saving them in the config | Load the checkpoint directory or `.simurgh` archive (frozen source). Runs trained since then save `elements`, `scale`, `avg_num_neighbors` and atomic references in their config; see [Checkpoints & archives](../../ml/deployment/archives.md) |

A warning such as `Could not reconstruct loss components from frozen source ... safe for inference only` is harmless for MD. The loss is not used at inference.

### Wrong or nonsensical energies

- **Elements outside the training set.** A model only knows the elements it was trained on. Check the composition of the training data.
- **Large absolute offsets between methods are normal** (atomic reference energies differ). Compare relative energies and forces.
- **Charge/spin not set.** `from_smiles` drops formal charges. Pass `charge`/`multiplicity` (`tblite`, `orca`), or the code's own input keys (`tot_charge`/`nspin` for QE, and so on) to the calculator.
- **Head selection.** Multi-head models need `head=...`.

## Devices and precision

### CUDA errors

| Message | Cause | Fix |
| --- | --- | --- |
| `Torch not compiled with CUDA enabled` | You are in a `cpu` / `cpu-full` environment (`pytorch-cpu`) | Run in a CUDA environment: `pixi run -e cuda-full ...` / `pixi shell -e cuda-full` |
| `No CUDA GPUs are available` / `CUDA error: invalid device ordinal` | No GPU visible, or the index is wrong | Check `nvidia-smi` and `CUDA_VISIBLE_DEVICES`; use `device=cpu` or a valid `cuda:N` |
| `CUDA out of memory` | System too large for the model/GPU, or several models resident on one GPU | Use a smaller system or a GPU with more memory, free other calculators, or run on CPU |

Defaults differ between backends. `goal_model` defaults to `cpu`; `flashmd`, `mace` and `nequip` auto-detect (`None` → CUDA if available); `src/goal/configs/md/calculators/upet.yaml` sets `device: cuda`. Set `device` explicitly in production configs.

### `expected scalar type Double but found Float` (or similar dtype mismatches)

`GOALCalculator`, `goal_model` and `load_goal_calculator` build the model in the dtype its checkpoint was saved in and feed it inputs of that dtype, so GOAL checkpoints should not hit this any more. For a module you pass in yourself, `GOALCalculator(module=..., dtype=None)` matches the module's parameters; an explicit `dtype=` in Python overrides it.

## During the simulation

### Temperature explodes, atoms fly apart or collapse

Work through these causes in order:

1. **Time-step units.** `DynamicsFactory.create("langevin_ase", ..., timestep=1.0)` passes `timestep` to ASE raw, which is ≈ 10 fs. Use `timestep_fs=0.5` (or `timestep=0.5 * units.fs`). `Simulation`, `goal-simulate`, `goal-simulate-batch` and the shipped `dynamics/langevin.yaml` template all work in femtoseconds.
2. **Time step too large for the chemistry.** Use 0.5 fs for molecules with hydrogens, less at high temperature ([Dynamics](../guide/dynamics.md#choosing-a-time-step)).
3. **Unrelaxed start.** The MMFF94 geometry from `from_smiles` is not a minimum of your potential. Relax with the production calculator first.
4. **Model extrapolation.** The potential is being evaluated outside its training distribution. Check the minimum interatomic distance along the trajectory, compare forces on a few frames with a reference method, and add training data from those regions (see the [train → simulate → analyse loop](../guide/ml-bridge.md#the-train-simulate-analyse-loop)).
5. **Non-conservative forces.** Direct-force models (for example UPET with `non_conservative: true`) do not conserve energy. Do not use them for production MD.

### Temperature reported at the end is far from the target

`RunResult.final_temperature` is a single instantaneous value, and for a 9-atom molecule it fluctuates by ±100 K or more. Average the `T[K]` column of the log over the production part instead. Note also that `Simulation` defaults to a strong friction (0.5 fs⁻¹); with weak friction, equilibration takes several coupling times.

### `FutureWarning: The implementation of fixcm=True in Langevin ...`

ASE ≥ 3.28 deprecates Langevin's built-in centre-of-mass fixing. It is harmless for large systems. For small molecules, prefer `fixcm=False` together with `ase.constraints.FixCom()`, which requires building the integrator yourself ([Dynamics](../guide/dynamics.md#langevin_ase)).

### Output files are not where I expected, or are overwritten or duplicated

- Relative `output_dir`, `trajectory_file` and `log_file` resolve against the directory you **launched from**. Hydra's own `outputs/<date>/<time>/` directory only holds `.hydra/` and the run log.
- The trajectory is opened in `"w"` mode: a second run in the same `output_dir` overwrites it. The MD log is opened in `"a"` mode: a second run appends a new header and new lines to the old file. Give every run its own `output_dir` (for example with `'output_dir=outputs/md/T${temperature_K}'`).
- Calling `Simulation.run()` twice on the same object continues the same trajectory and appends to the same files. That is intended.

### The run is slow

- Check that the calculator is on the GPU (`device`) and that you are in a CUDA environment.
- Write frames less often: a large `log_interval`, or `trajectory_interval` when using `setup_observers` directly.
- QM backends: set MPI ranks and threads (`n_mpi`, `omp_threads`, `num_threads`) on the calculator.
- For QM-accuracy runs, consider [multiple time-stepping](../mts/overview.md), which calls the expensive method only every `mts_ratio` steps.
