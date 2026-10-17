# Fine-tuning UPET / PET

UPET (the renamed `pet-mad` package, `lab-cosmo/upet`) is **not** fine-tuned
in-process. GOAL writes a metatrain `options.yaml` and runs `mtt train` as a
subprocess, through the `goal-finetune-upet` CLI. The fine-tuned checkpoint
then comes back into GOAL as an MD calculator.

Source:
[`goal.ml.cli.finetune_upet`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/cli/finetune_upet.py).

## Why a subprocess

UPET represents everything internally as metatensor `TensorMap` objects.
Building valid, gradient-carrying `TensorMap`s from `AtomicGraph` would mean
reimplementing a large part of metatensor's machinery. metatrain, on the other
hand, is **file-based**: no gradients need to flow through GOAL. So the right
tool is a thin wrapper around `mtt train`, not a backbone.

There is also no ml-side UPET backbone, so UPET has no `AtomicGraph` path
and no batched inference in GOAL. It is used through ASE, one frame at a
time.

> [!NOTE]
> **Future work:** in-process UPET fine-tuning through a metatensor
> `TensorMap` adapter, which would build `TensorMap` systems from
> `AtomicGraph` so UPET trains inside `GOALModule` like MACE and UMA.

## Install

```bash
pip install upet metatrain
```

`mtt` must be on `PATH`. The CLI checks for it before launching.

## Workflow

1. **Export training data** to ExtXYZ, with energy in `atoms.info["energy"]`
   (eV) and forces in `atoms.arrays["forces"]` (eV/Å). These key names are
   hard-coded in the generated `options.yaml`:

   ```python
   from goal.ml.data.export import write_extxyz
   write_extxyz(train_dataset, "data/train.xyz")
   write_extxyz(val_dataset, "data/val.xyz")
   ```

2. **Generate the options and inspect them** with `--dry-run`:

   ```bash
   pixi run -e cuda goal-finetune-upet \
       --model pet-omat-l \
       --train "$PWD/data/train.xyz" \
       --val   "$PWD/data/val.xyz" \
       --output runs/upet_finetuned/ \
       --epochs 100 --lr 1e-4 \
       --dry-run
   ```

3. **Run.** Drop `--dry-run`. The CLI runs
   `mtt train options.yaml -o model.ckpt` with the output directory as the
   working directory, and returns `mtt`'s exit code.

4. **Use the fine-tuned model** as an MD calculator:

   ```python
   from goal.md import CalculatorFactory

   calc = CalculatorFactory.create(
       "upet",
       checkpoint_path="runs/upet_finetuned/model.ckpt",
       variants={"energy": "finetune"},
   )
   ```

   See the [UPET calculator docs](../../md/calculators/upet.md).

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | base model, `<family>-<size>` |
| `--train` | required | training ExtXYZ (must exist) |
| `--val` | none | validation ExtXYZ. Without it, metatrain holds out 10% of `--train` |
| `--output` | required | run directory (created). Receives `options.yaml` and `model.ckpt` |
| `--epochs` | `100` | `architecture.training.num_epochs` |
| `--lr` | `1e-4` | `architecture.training.learning_rate` |
| `--dry-run` | off | write `options.yaml` only |

## What the generated `options.yaml` contains

```yaml
device: cuda
base_precision: 64

architecture:
  name: pet
  training:
    num_epochs: 100
    learning_rate: 0.0001
  finetune:
    method: full
    read_from: pet-omat-l   # the --model value

training_set:
  systems: {read_from: /abs/path/data/train.xyz, length_unit: angstrom}
  targets:
    energy: {key: energy, unit: eV, forces: {key: forces}}
validation_set: ...         # same shape, or 0.1 without --val
test_set: 0.0
```

> [!WARNING]
> Check the generated file before a long run:
>
> - **Use absolute paths for `--train`/`--val`.** `mtt train` runs with the
>   output directory as its working directory, so a relative path such as
>   `data/train.xyz` resolves against `runs/upet_finetuned/` and is not
>   found.
> - **`finetune.read_from` receives the model *name*.** Depending on your
>   metatrain version it may expect a path to a checkpoint file. If so,
>   download the base checkpoint and edit `read_from`.
> - `device: cuda`, `base_precision: 64` and `method: full` are fixed. Edit
>   the file for CPU runs, other precisions or other fine-tuning methods.
> - Field names differ across metatrain and upet versions. If `mtt train`
>   rejects the file, edit it and run `mtt train` yourself.

> [!NOTE]
> `--model` is checked against a fixed grid: families `pet-mad`, `pet-oam`,
> `pet-omat`, `pet-spice` crossed with sizes `xs`, `s`, `m`, `l`, `xl`. That
> grid is not what upet ships. Real releases are not a full cross product
> (for example `pet-oam` comes only in `l`/`xl`), and newer families such as
> `pet-omad` and `pet-omatpes` are rejected. The MD calculator already reads
> the valid list from `upet._version.UPET_AVAILABLE_MODELS`. For models
> outside the grid, run with a grid-valid name and `--dry-run`, then edit
> `read_from`.
