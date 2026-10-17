# Roadmap stubs

Three features are planned on top of the nvalchemi integration but not implemented yet: well-tempered **metadynamics**, **replica-exchange MD** (REMD), and **knowledge distillation** from a large teacher model into a small GOAL student (`goal-distill`). Their entry points already exist, so scripts, configs and documentation can refer to them, and selecting one fails with a clear message instead of an obscure error. This page lists what exists, what each stub does when called, and how to get the same results today with the tools that work.

Source:
[`goal.ml.utils.enhanced_sampling`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/utils/enhanced_sampling.py) and
[`goal.ml.cli.distill`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/cli/distill.py).

## What it is

| Stub | Entry point | Behaviour today |
|---|---|---|
| Metadynamics | `goal.ml.utils.enhanced_sampling.run_metadynamics`, `goal-simulate-batch --sampler metadynamics` | `NotImplementedError` naming the upstream project; the CLI prints the message and exits with code 2 |
| Replica-exchange MD | `goal.ml.utils.enhanced_sampling.run_remd`, `goal-simulate-batch --sampler remd` | Same |
| Distillation | `goal-distill`, `goal.ml.cli.distill.run_distillation` | Prints the roadmap message and exits with code 1; `run_distillation()` raises `NotImplementedError` |

**Upstream status (October 2026).** nvalchemi-toolkit's `main` branch has an enhanced-sampling package, `nvalchemi.enhanced_sampling`, with an `EnhancedSampling` runner, `WellTemperedMetaDynamicsBias`, `ReplicaExchange`, `HarmonicUmbrellaBias`, walls and `AdaptiveBiasingForce`. It is not in a release yet: release 0.2.0 only has the generic `BiasedPotentialHook`. GOAL's metadynamics and REMD entry points will drive that engine once its API is released. The plan for distillation is to label structures with a large teacher (a foundation model or a big SIMURGH) using nvalchemi's batched inference on NVIDIA GPUs, and train a small, fast GOAL student on those labels.

## Hardware requirements

The stubs themselves run anywhere: they only raise or print a message. The planned implementations will build on nvalchemi-toolkit and therefore target NVIDIA CUDA GPUs; the workarounds below run on any hardware unless they use the [nvalchemi model wrapper](nvalchemi_wrapper.md).

## Installation

Nothing to install. The stubs ship with GOAL; `goal-simulate-batch` needs a `-full` pixi environment ([Batched simulation](batched_simulation.md)), `goal-distill` works in every environment. Running a GOAL model inside upstream's enhanced-sampling engine yourself needs nvalchemi-toolkit from git `main`, installed as in [NVIDIA nvalchemi integration](nvalchemi_overview.md#installation).

## Quick start

What the stubs do:

```bash
pixi run -e cuda-full goal-simulate-batch --checkpoint model.simurgh --structures frames.xyz --sampler remd
# goal-simulate-batch: Replica-exchange MD (REMD) is on the GOAL roadmap but not implemented yet.  It will be
# built on nvalchemi-toolkit's enhanced-sampling engine (nvalchemi.enhanced_sampling, unreleased on its main
# branch): https://github.com/NVIDIA/nvalchemi-toolkit
# Today you can already run a GOAL model there yourself through
# goal.ml.adapters.nvalchemi_wrapper.SIMURGHNvalchemiWrapper.
echo $?    # 2

pixi run -e cuda-full goal-distill --teacher big_model.simurgh
# goal-distill is on the GOAL roadmap but not implemented yet.
# Today: label your structures with the teacher yourself (e.g. goal-simulate-batch or a GOALCalculator / ASE
# loop writing extxyz), then train the student with goal-train on those files.
echo $?    # 1
```

### Distillation today

Label the structures with the teacher through any ASE calculator, write ExtXYZ, and train the student on it with `goal-train`. With a GOAL teacher:

```python
from ase.io import iread, write
from goal.ml.utils.calculator import GOALCalculator

teacher = GOALCalculator(checkpoint_path="logs/big_run/runs/<run>/checkpoints", device="cuda")

labelled = []
for atoms in iread("data/unlabelled.xyz", index=":"):
    atoms.calc = teacher
    atoms.info["energy"] = atoms.get_potential_energy()
    atoms.arrays["forces"] = atoms.get_forces()
    atoms.calc = None
    labelled.append(atoms)
write("data/distill/train.xyz", labelled, format="extxyz")
```

A foundation-model teacher works the same way with its own ASE calculator (for MACE, from the `.venv-mace` interpreter; see [Fine-tuning MACE](../ml/finetuning/mace.md)). To generate the structures to label, run short teacher trajectories with `goal-simulate-batch` and use the frames of `traj_*.extxyz`. Then train the student with an experiment config whose `data.train_paths` / `data.val_paths` point at the labelled files ([Loading data](../ml/data/loading.md)), and package it as usual ([Checkpoints & archives](../ml/deployment/archives.md)).

### Enhanced sampling today

`SIMURGHNvalchemiWrapper` makes a GOAL checkpoint an nvalchemi `BaseModelMixin` model, and upstream's biases are themselves `BaseModelMixin` models. You can therefore combine them in nvalchemi's engine from `main` following upstream's documentation, with the GOAL model as the physical potential. GOAL does not wrap that API yet because it may still change before release.

## API reference

### `goal.ml.utils.enhanced_sampling`

| Name | Type | Meaning |
|---|---|---|
| `run_metadynamics(*args, **kwargs)` | function | Raises `NotImplementedError("Metadynamics is on the GOAL roadmap but not implemented yet. ...")` |
| `run_remd(*args, **kwargs)` | function | Raises `NotImplementedError("Replica-exchange MD (REMD) is on the GOAL roadmap ...")` |
| `SAMPLERS` | `dict[str, callable]` | `{"metadynamics": run_metadynamics, "remd": run_remd}`; `run_batched_simulation` dispatches non-Langevin samplers through it |
| `UPSTREAM_URL` | `str` | `"https://github.com/NVIDIA/nvalchemi-toolkit"`, contained in both messages |

Both functions accept and ignore any arguments, so the call signature used by callers today will not have to change.

### `goal-distill`

| Flag | Meaning (planned) |
|---|---|
| `--teacher` | Teacher checkpoint or foundation-model name |
| `--student-config` | `goal-train` experiment for the student |
| `--structures` | Structures to label (zero or more files) |

The flags are parsed (so a misspelt flag is still a usage error) and otherwise ignored. Exit code 1 after printing `ROADMAP_MESSAGE`. In Python, `goal.ml.cli.distill.run_distillation(*args, **kwargs)` raises `NotImplementedError(ROADMAP_MESSAGE)`.

## How it works

- `run_batched_simulation` validates the config, and for any `sampler` other than `langevin` calls `SAMPLERS[sampler](config)` **before** it reads structures or loads the checkpoint. The stub raises `NotImplementedError`; `goal-simulate-batch` catches it, prints `goal-simulate-batch: <message>` to stderr without a traceback, and returns exit code 2. Because the check comes first, the message appears even when the checkpoint or structure paths are wrong.
- `goal-distill` parses its flags, calls `run_distillation()`, catches the `NotImplementedError`, prints the message to stderr and returns 1.

## Configuration reference

| Feature | YAML |
|---|---|
| Metadynamics, REMD | `sampler: metadynamics` or `sampler: remd` in a `goal-simulate-batch --config` file such as [`src/goal/configs/md/simulations/batch.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/simulations/batch.yaml). Allowed values: `langevin` (default), `metadynamics`, `remd`. There are no bias or replica parameters yet |
| Distillation | No YAML configuration; `goal-distill` takes command-line flags only |

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `goal-simulate-batch: Metadynamics is on the GOAL roadmap but not implemented yet. ...` (exit 2) | `--sampler metadynamics` (or `sampler: metadynamics` in the `--config` file) | Use `--sampler langevin`; see [Enhanced sampling today](#enhanced-sampling-today) |
| `goal-simulate-batch: Replica-exchange MD (REMD) is on the GOAL roadmap but not implemented yet. ...` (exit 2) | `--sampler remd` | As above |
| `goal-distill is on the GOAL roadmap but not implemented yet.` (exit 1) | `goal-distill` has no implementation yet | Follow [Distillation today](#distillation-today) |
| `NotImplementedError: ... on the GOAL roadmap ...` with a traceback | Calling `run_metadynamics`, `run_remd` or `run_distillation`, or `run_batched_simulation` with `sampler` set to `metadynamics` or `remd`, from Python | Catch it, or use the workarounds above |
| `goal-simulate-batch: error: argument --sampler: invalid choice: ...` (exit 2) | A sampler name other than `langevin`, `metadynamics` or `remd` | Use one of the three |
| `ValueError: sampler must be one of ('langevin', 'metadynamics', 'remd'), got ...` | Same, through `BatchSimulationConfig` in Python | Same |
| `goal-distill: command not found` | The environment was installed before the command existed | `pixi install -e <env>`, or `python -m goal.ml.cli.distill` |
