# Batched simulation

`goal-simulate-batch` runs Langevin MD or a geometry relaxation for **many structures with one GOAL model**: every frame of every input file. On an NVIDIA GPU with nvalchemi-toolkit installed, all structures go into one GPU-resident batch and are integrated together by nvalchemi. Everywhere else, the same command runs the structures one after another with ASE and `GOALCalculator`. The interface, units and output files are the same for both backends.

Source:
[`goal.md.core.batched`](https://github.com/Nourollah/GOAL/blob/main/src/goal/md/core/batched.py) (library) and
[`goal.md.cli.simulate_batch`](https://github.com/Nourollah/GOAL/blob/main/src/goal/md/cli/simulate_batch.py) (CLI).
For a single long trajectory with full control over the integrator and observers, use `goal-simulate` instead ([MD CLI](../md/reference/cli.md), [Dynamics](../md/guide/dynamics.md)).

## What it is

Two backends behind one interface:

| Backend | How | Runs on |
|---|---|---|
| `nvalchemi` | All structures in **one** GPU batch through the [nvalchemi model wrapper](nvalchemi_wrapper.md): `NVTLangevin` for MD, `FIRE2` for relaxation, with a `NaNDetectorHook` | NVIDIA CUDA with nvalchemi-toolkit |
| `ase` | Structures one after another: ASE `Langevin` for MD, ASE `FIRE` for relaxation, one `GOALCalculator` shared by all structures | NVIDIA, AMD ROCm, Apple MPS, CPU |

`backend: auto` (the default) picks `nvalchemi` when nvalchemi-toolkit and an NVIDIA GPU are available, and `ase` otherwise, and logs the choice.

Typical uses: screening hundreds of conformers or crystal candidates with short MD runs, relaxing a set of structures before DFT, or generating many independent trajectories for training-set expansion.

The two backends agree. With the same small checkpoint, a 16-atom diamond supercell run for 40 steps at 0.25 fs goes from 27.565 to 27.562 eV with the ASE backend and from 27.565 to 27.563 eV with the nvalchemi backend. The two thermostats draw different random numbers, so trajectories agree statistically, not step by step. `FIRE2` relaxations converge.

## Hardware requirements

| Hardware | Backend `auto` resolves to | Model device (`device: auto`) |
|---|---|---|
| NVIDIA GPU, nvalchemi-toolkit installed | `nvalchemi` | `cuda` |
| NVIDIA GPU, no nvalchemi-toolkit (e.g. `pixi run -e cuda-full`) | `ase` | `cuda` |
| AMD GPU (ROCm build of PyTorch) | `ase` | `cuda` (PyTorch's name for the ROCm device) |
| Apple Silicon | `ase` | `cpu` (`auto` never selects MPS) |
| CPU | `ase` | `cpu` |

The nvalchemi backend pays off when many structures share the GPU: one batched force evaluation replaces one evaluation per structure per step.

## Installation

The command is part of the MD module. Use a `-full` pixi environment:

```bash
pixi install -e cuda-full     # or cpu-full / rocm-full
pixi run -e cuda-full goal-simulate-batch --help
```

That is enough for the ASE backend. The nvalchemi backend additionally needs `nvalchemi-toolkit`, which is not in the pixi environments. Install it into the layered `.venv-nvalchemi` described in [NVIDIA nvalchemi integration](nvalchemi_overview.md#installation) and run the CLI module with that interpreter:

```bash
.venv-nvalchemi/bin/python -m goal.md.cli.simulate_batch --checkpoint ... --structures ...
```

Writing trajectories as Zarr (`--trajectory-format zarr`) needs `zarr>=3`, which is in every pixi environment ([Zarr data](zarr_data.md)).

## Quick start

2000 Langevin steps at 300 K for every frame in `frames.xyz`:

```bash
pixi run -e cuda-full goal-simulate-batch \
    --checkpoint logs/simurgh_gmd26/runs/<run>/checkpoints \
    --structures frames.xyz --steps 2000 --temperature 300 --timestep 0.5
```

Relax everything in two files to `fmax` 0.02 eV/Å with the portable backend:

```bash
pixi run -e cuda-full goal-simulate-batch --checkpoint models/water.simurgh \
    --structures a.xyz b.xyz --mode relax --fmax 0.02 --backend ase
```

Start from the example config and override some of its values:

```bash
pixi run -e cuda-full goal-simulate-batch --config src/goal/configs/md/simulations/batch.yaml \
    --checkpoint models/water.simurgh --structures frames.xyz --steps 500
```

The command ends with a summary:

```text
Done: 3 structure(s), 2000 steps, backend=ase, <wall time> s
  wrote outputs/md/batch/traj_0000.extxyz
  ...
  wrote outputs/md/batch/final.extxyz
```

From Python:

```python
from goal.md.core.batched import BatchSimulationConfig, run_batched_simulation

result = run_batched_simulation(
    BatchSimulationConfig(
        checkpoint="logs/simurgh_gmd26/runs/<run>/checkpoints",
        structures=["frames.xyz"],
        steps=2000, temperature_K=300.0, timestep_fs=0.5,
    )
)
print(result.backend, result.final_energies)
result.final_structures[0].write("first_final.xyz")
```

## API reference

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--config` | — | YAML file whose keys are `BatchSimulationConfig` fields; flags override it. Unknown keys are an error |
| `--checkpoint` | required | `.simurgh` archive, checkpoint directory or `.ckpt` |
| `--checkpoint-select` | `best` | `best`, `last` or `epoch=N` inside a checkpoint directory |
| `--structures` | required | One or more structure files; every frame of every file is simulated (`ase.io.read(path, index=":")`) |
| `--steps` | `1000` | MD steps, or the maximum number of optimiser steps for `relax` |
| `--temperature` | `300` | Langevin target temperature and initial Maxwell–Boltzmann temperature, K |
| `--timestep` | `0.5` | MD time step, fs |
| `--friction` | `0.01` | Langevin friction, 1/fs |
| `--fmax` | `0.05` | Force convergence threshold for `relax`, eV/Å |
| `--mode` | `md` | `md` or `relax` |
| `--backend` | `auto` | `auto`, `nvalchemi` or `ase` |
| `--sampler` | `langevin` | `langevin`; `metadynamics` and `remd` are [roadmap stubs](roadmap_stubs.md) |
| `--output-dir` | `outputs/md/batch` | Output directory, relative to the launch directory |
| `--trajectory-format` | `extxyz` | `extxyz` (one file per structure) or `zarr` (one store) |
| `--no-trajectory` | off | Write only `final.extxyz` |
| `--log-every` | `100` | Trajectory and log interval, steps |
| `--device` | `auto` | Torch device for the model; `auto` = `cuda` when `torch.cuda.is_available()`, else `cpu` |
| `--seed` | `42` | Seed for the initial velocities and the thermostat |

Exit codes: `0` success; `1` invalid input (bad value, missing checkpoint or structure file, unknown YAML key); `2` a feature that is not implemented (roadmap sampler) or an argparse usage error. A fragment-aware checkpoint on the nvalchemi path falls back to ASE with a warning under `--backend auto`, and is an input error (exit 1) under an explicit `--backend nvalchemi`. The errors listed under [Troubleshooting](#troubleshooting) are printed as one line, without a traceback.

### `BatchSimulationConfig`

A dataclass in `goal.md.core.batched` with the same settings. CLI flag names differ only for `--temperature` (`temperature_K`), `--timestep` (`timestep_fs`) and `--no-trajectory` (`write_trajectory: false`).

| Field | Type | Default |
|---|---|---|
| `checkpoint` | `str` | `""` (required) |
| `structures` | `list[str]` | `[]` (required) |
| `steps` | `int` | `1000` |
| `timestep_fs` | `float` | `0.5` |
| `temperature_K` | `float` | `300.0` |
| `friction` | `float` | `0.01` (1/fs) |
| `fmax` | `float` | `0.05` (eV/Å) |
| `mode` | `str` | `"md"` |
| `backend` | `str` | `"auto"` |
| `sampler` | `str` | `"langevin"` |
| `output_dir` | `str` | `"outputs/md/batch"` |
| `write_trajectory` | `bool` | `True` |
| `trajectory_format` | `str` | `"extxyz"` |
| `log_every` | `int` | `100` |
| `device` | `str` | `"auto"` |
| `checkpoint_select` | `str` | `"best"` |
| `seed` | `int` | `42` |

`validate()` checks the choices, that `checkpoint` and `structures` are set, and that `steps` and `log_every` are at least 1. `run_batched_simulation` calls it.

### Functions and result

| Name | Meaning |
|---|---|
| `run_batched_simulation(config)` | Runs everything and returns a `BatchSimulationResult`. Raises `NotImplementedError` for the roadmap samplers |
| `resolve_backend(requested)` | `"auto"` → `"nvalchemi"` when `nvalchemi_available()`, else `"ase"`; an explicit unavailable `"nvalchemi"` → `"ase"` with a warning |
| `load_structures(paths)` | Every frame of every file; `FileNotFoundError` for a missing file |

`BatchSimulationResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `backend` | `str` | The backend that ran (`"nvalchemi"` or `"ase"`) |
| `final_structures` | `list[ase.Atoms]` | Final structures, with `info["energy"]`, `info["step"]` and `arrays["forces"]` |
| `final_energies` | `list[float]` | Final potential energies, eV |
| `steps_completed` | `int` | Steps actually taken: for ASE the largest per-structure count (a relaxation stops at convergence), for nvalchemi the integrated count |
| `wall_time_s` | `float` | Wall time of the run, s |
| `output_files` | `list[str]` | Every file written |

## How it works

1. **Read** every frame of every structure file and **resolve the backend**.
2. **Initial velocities** (MD only): Maxwell–Boltzmann at `temperature_K`, drawn structure by structure from one random generator seeded with `seed`. ASE's velocity unit (Å per 10.18 fs) is also nvalchemi's internal velocity unit, so the ASE-generated velocities are handed to nvalchemi unchanged.
3. **Integrate.**
   - *nvalchemi:* each structure becomes an `AtomicData` with positions, atomic numbers, masses, velocities and preallocated `energy`/`forces` buffers. Every structure gets `cell` and `pbc`, with an identity cell and `pbc=False` for molecules, because upstream `Batch.from_data_list` keeps only the first structure's fields and would otherwise drop the cells of periodic structures that follow a molecule. MD uses `NVTLangevin(dt=timestep_fs, temperature=temperature_K, friction=friction, random_seed=seed)`, with `dt` in fs and friction in 1/fs. Relaxation uses `FIRE2(dt=0.05, convergence_hook=ConvergenceHook.from_fmax(fmax))`; `FIRE2`'s `dt` is a raw optimiser step, not fs. The engine runs in chunks of `log_every` steps, and frames are written between chunks.
   - *ase:* `Langevin(timestep=timestep_fs * units.fs, temperature_K=..., friction=friction / units.fs)` or `FIRE`, run for `steps` (relaxations stop early once `fmax` is reached). An observer writes a frame every `log_every` steps.
4. **Write** the outputs and `final.extxyz`.

**Units.** Time in fs, friction in 1/fs, temperature in K, energies in eV, forces in eV/Å, throughout. This differs from GOAL's older `langevin_ase` dynamics builder, which passed `timestep` to ASE unconverted (1.0 ≈ 10 fs). That builder now also accepts `timestep_fs` and `friction_per_fs`; see [Dynamics](../md/guide/dynamics.md).

**Output files** in `output_dir`:

| File | Content |
|---|---|
| `traj_0000.extxyz`, `traj_0001.extxyz`, ... | One per input structure (extxyz format): a frame every `log_every` steps, with `info["step"]`, `info["energy"]` and `arrays["forces"]`. Existing files of the same name are replaced at the start of a run |
| `trajectory.zarr` | Instead of the extxyz trajectories with `--trajectory-format zarr`: one [GOAL Zarr store](zarr_data.md), split (group) `trajectory`, positions, cells, energies and forces of every frame, ordered by structure and then step, with `structure_index` and `step` stored per frame (`ds.structure_info(i)`) |
| `final.extxyz` | All final structures with energy and forces, in input order |

> [!NOTE]
> Both backends write `trajectory.zarr` in the same order — all frames of structure 0, then structure 1, … — and record `structure_index` and `step` for every frame via `write_zarr(..., info_keys=...)`. Read them back with `ZarrDataset(root=..., cutoff=..., split="trajectory").structure_info(i)`.

## Configuration reference

[`src/goal/configs/md/simulations/batch.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/simulations/batch.yaml) lists every key with its default. It is a plain YAML file for `--config`, not a Hydra config: no `defaults`, no `_target_`, and command-line flags (not `key=value` overrides) take precedence over it.

```yaml
checkpoint: ""               # required: .simurgh archive, checkpoint directory or .ckpt
checkpoint_select: best      # best | last | epoch=N (checkpoint directories)
structures: []               # required: structure files; every frame is simulated

mode: md                     # md | relax
steps: 1000
timestep_fs: 0.5             # fs
temperature_K: 300.0
friction: 0.01               # 1/fs
fmax: 0.05                   # eV/Å, relax only

backend: auto                # auto | nvalchemi | ase
sampler: langevin            # langevin | metadynamics | remd (the last two are roadmap stubs)
device: auto                 # auto | cpu | cuda | cuda:N

output_dir: outputs/md/batch
write_trajectory: true
trajectory_format: extxyz    # extxyz (one file per structure) | zarr (one store)
log_every: 100
seed: 42
```

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `GOAL batched simulation: backend 'nvalchemi' requested but it needs nvalchemi-toolkit and an NVIDIA CUDA GPU; falling back to 'ase'.` (WARNING) | `--backend nvalchemi` under `pixi run` (no nvalchemi-toolkit), or on CPU, ROCm or MPS | Expected on non-NVIDIA hardware. On NVIDIA, run `.venv-nvalchemi/bin/python -m goal.md.cli.simulate_batch` ([installation](nvalchemi_overview.md#installation)) |
| `goal-simulate-batch: Structure file not found: ...` (exit 1) | Wrong path; relative paths resolve from the launch directory | Fix the path |
| `goal-simulate-batch: checkpoint is required.` / `structures is required (one or more structure files).` (exit 1) | Neither the flag nor the `--config` file sets it | Pass `--checkpoint` / `--structures` |
| `goal-simulate-batch: unknown keys in <file>: [...]` (exit 1) | A key in the `--config` YAML that is not a `BatchSimulationConfig` field, e.g. `temperature` instead of `temperature_K` | Use the field names from the table above |
| `goal-simulate-batch: Metadynamics is on the GOAL roadmap but not implemented yet. ...` (exit 2) | `--sampler metadynamics` or `--sampler remd` | Not available yet; see [Roadmap stubs](roadmap_stubs.md) |
| `goal-simulate-batch: backend 'nvalchemi' cannot run this checkpoint: This checkpoint was trained with fragment labels ...` (exit 1) | Fragment-aware checkpoints cannot run on the nvalchemi backend; `--backend auto` falls back to ASE by itself | `--backend auto` or `--backend ase` |
| `goal-simulate-batch: command not found` in `cpu-full`, `rocm-full` or an older environment | The environment was installed before the command existed, so its entry point is missing | `pixi install -e <env>`, or run `python -m goal.md.cli.simulate_batch` |
| Temperature explodes or atoms fly apart | Time step too large for the chemistry, or an unrelaxed start | 0.5 fs or less for molecules with hydrogen; relax first (`--mode relax`) |
