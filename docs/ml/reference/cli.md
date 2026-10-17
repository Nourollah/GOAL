# CLI

This page is the reference for every `goal.ml` console script declared in `[project.scripts]`: what each one does, how it is configured (Hydra or argparse), its real options, and worked examples. The MD commands `goal-simulate`, `goal-simulate-mts` and `goal-simulate-batch` are documented in the [MD CLI reference](../../md/reference/cli.md) and [Batched simulation](../../integrations/batched_simulation.md).

## Overview

| Command | Entry point | Configured by | Default config | Purpose |
|---|---|---|---|---|
| `goal-train` | `goal.ml.cli.train:train_ml` | Hydra | `src/goal/configs/ml/simurgh_gmd26.yaml` | Train (and fine-tune) any model |
| `goal-train-ml` | same function | Hydra | same | Alias of `goal-train` |
| `goal-eval` | `goal.ml.cli.evaluate:evaluate` | Hydra | `src/goal/configs/ml/eval.yaml` | Test a checkpoint on the test split |
| `goal-finetune` | `goal.ml.cli.finetune:finetune` | Hydra | `src/goal/configs/ml/finetune/simurgh_lora.yaml` | Fine-tune a pre-trained backbone (default: LoRA on a SIMURGH checkpoint) |
| `goal-finetune-upet` | `goal.ml.cli.finetune_upet:main` | argparse | — | Fine-tune UPET/PET through metatrain |
| `goal-tune` | `goal.ml.cli.tune:tune` | Hydra | `src/goal/configs/ml/simurgh_gmd26_hpo.yaml` | LR finder, Ray Tune, W&B Sweeps |
| `goal-inspect` | `goal.ml.cli.inspect_ckpt:main` | argparse | — | Describe a checkpoint without loading the model |
| `goal-pack-archive` | `goal.ml.cli.pack:main` | argparse | — | Pack a self-contained `.simurgh` archive |
| `goal-pack-zarr` | `goal.ml.cli.pack_zarr:main` | argparse | — | Convert structure files to a GOAL Zarr store ([Zarr data](../../integrations/zarr_data.md)) |
| `goal-distill` | `goal.ml.cli.distill:main` | argparse | — | Roadmap stub; exits 1 with a message ([Roadmap stubs](../../integrations/roadmap_stubs.md)) |

Every module also runs with `python -m`, for example `python -m goal.ml.cli.train` or `python -m goal.ml.cli.pack --help`.

## Common behaviour of the Hydra commands

`goal-train`, `goal-eval`, `goal-finetune` and `goal-tune` share Hydra's command-line interface:

| Flag | Meaning |
|---|---|
| `--config-name NAME` / `-cn NAME` | Config file to load, relative to the command's config directory, without `.yaml` (for example `nf_simurgh_gmd26`, `ablations/01_gmd26_base`) |
| `--config-dir DIR` / `-cd DIR` | Add a directory to the search path |
| `key=value`, `+key=value`, `++key=value`, `~key` | Override, add, force or delete a config key |
| `-m` / `--multirun` | Sweep over comma-separated values |
| `--cfg job` (`-c job`) | Print the composed config and exit, without training. Add `-p model.backbone` to print one node. |
| `--info` | Print the search path, defaults list and so on |
| `--help` | Hydra help, including the available config groups |

The config directories are absolute paths into the configs bundled with the package (`goal/configs/`, which is `src/goal/configs/` in a clone; the path is available as `goal.CONFIGS_DIR`). They are computed in `goal/ml/cli/__init__.py`, so every command finds them from any working directory, in a clone or in a pip/conda install:

- all four search **`goal/configs/ml/`**, so `--config-name <experiment>` picks any experiment file there;
- each has a default config that exists: `simurgh_gmd26` (`goal-train`), `eval` (`goal-eval`), `finetune/simurgh_lora` (`goal-finetune`), `simurgh_gmd26_hpo` (`goal-tune`).

The shared presets in `goal/configs/debug/` and `goal/configs/hparams_search/` sit one level up, so attaching one needs that directory on the search path. In a clone:

```bash
<command> --config-dir src/goal/configs --config-name <experiment> +debug=fdr
```

From any directory (including a pip or conda install), use the bundled path:

```bash
<command> --config-dir "$(python -c 'import goal; print(goal.CONFIGS_DIR)')" --config-name <experiment> +debug=fdr
```

Beyond that, `--config-dir` is for your **own** config directories: put your experiment files there and they are found alongside the bundled ones.

> [!NOTE]
> Before October 2026 `goal-eval`, `goal-finetune` and `goal-tune` looked for `eval.yaml` / `train.yaml` at the top of the bundled configs, which never existed, so every invocation without `--config-name` failed with `MissingConfigException`.

> [!NOTE]
> Importing `goal.ml.cli` also does two things for every command. It populates all registries (backbones, heads, datasets, adapters), including the benchmark adapters in [`goal.ml.data.benchmarks`](../data/benchmarks.md). And it patches Lightning's checkpoint loader to use `weights_only=False`, because GOAL checkpoints store an OmegaConf config. Only load checkpoints you trust.

## `goal-train`

The main training command (`goal-train-ml` is an identical alias). It builds the datamodule, derives dataset statistics, builds the backbone and head from the registries, and runs `lightning.Trainer.fit`. See [Training a model](../training/training.md) for the full walkthrough.

What it does, in order:

1. Seeds everything (`seed`) and applies the performance settings: TF32 matmul on Ampere+ (`training.performance.float32_matmul_precision`, default `high` on Ampere+ and `highest` otherwise), plus cuDNN benchmark and deterministic flags.
2. **Resolves resume.** If `training.checkpoint_dir/last.ckpt` exists it resumes. If `resume.from` points at a finished run and `trainer.max_epochs` was raised, it extends that run. If nothing is left to train, it prints why and exits (this is what stops SLURM requeues of finished jobs).
3. Prepares the data and computes the dataset-derived values: elements, atomic references, scale, `avg_num_neighbors` and pair counts (see [Configuration](configuration.md#values-goal-train-fills-in-from-the-data)).
4. Builds the backbone and head and checks that they pair correctly (a monolithic backbone needs `head: null`).
5. Instantiates callbacks, loggers and `GOALCheckpointManager`. If `ModelCheckpoint` and the manager are both active, it refuses to start.
6. Builds the strategy (`build_strategy(cfg)`), and adds `SLURMEnvironment(auto_requeue=True)` when `training.slurm_mode=true`.
7. Fits, then writes `TRAINING_COMPLETE` into the checkpoint directory on rank 0.

```bash
# Default: SIMURGH ARACE on GMD-26 (src/goal/configs/ml/simurgh_gmd26.yaml)
pixi run -e cuda goal-train

# Another experiment file
pixi run -e cuda goal-train --config-name nf_simurgh_md17
pixi run -e cuda goal-train --config-name ablations/05_gmd26_body_order
pixi run -e cuda goal-train --config-name finetune/uma

# Overrides
pixi run -e cuda goal-train trainer.max_epochs=500 training.optimizer.lr=5e-3 data.batch_size=32

# Multi-GPU DDP on one node
pixi run -e cuda goal-train trainer.devices=4 strategy.name=ddp

# Smoke test with a debug preset (presets live in src/goal/configs/debug/, outside goal-train's ml/ dir, hence --config-dir)
pixi run -e cuda goal-train --config-dir src/goal/configs +debug=fdr
# outside a clone: goal-train --config-dir "$(python -c 'import goal; print(goal.CONFIGS_DIR)')" +debug=fdr

# Extend a finished run to 3000 epochs
pixi run -e cuda goal-train trainer.max_epochs=3000 \
    resume.from=logs/simurgh_gmd26/runs/<old_run>/checkpoints

# Check the composed config without training
pixi run -e cuda goal-train --config-name nf_simurgh_gmd26 --cfg job -p model
```

Related pages: [Distributed training](../training/distributed.md), [Resuming & extending](../training/resuming.md), [Performance engineering](../training/performance.md).

## `goal-eval`

Tests a checkpoint on a test split. The **model comes from the checkpoint** (frozen source, saved config, EMA weights) through the same loader as `GOALCalculator`; the config only chooses the data and the hardware. Minimal invocation:

```bash
pixi run -e cuda goal-eval ckpt_path=logs/simurgh_gmd26/runs/<run>/checkpoints
```

| Key | Meaning |
|---|---|
| `ckpt_path` | **Required.** `.simurgh` archive, checkpoint directory or `.ckpt` |
| `checkpoint` | Selector inside a directory: `best` (default), `last`, `epoch=N`, `<name>.ckpt` |
| `use_ema` | Evaluate the EMA weights when present (default `true`) |
| `data` | Merged onto the checkpoint's own `data:`; empty = the run's test split. Add keys with `+`, e.g. `'+data.test_paths=[x.xyz]'` |
| `trainer.*` | `accelerator`, `devices`, `precision` (`null` = the checkpoint's) for the test loop |
| `output_file` | Write the metrics to this JSON file |

`--config-name <experiment>` uses that experiment's `data:` / `trainer:` / `logger:` sections instead of `eval.yaml`. Details and the Python equivalent: [Evaluation](../deployment/evaluation.md).

## `goal-finetune`

Fine-tunes a pre-trained backbone. Its default config, [`finetune/simurgh_lora.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/simurgh_lora.yaml), adapts a trained **SIMURGH checkpoint with LoRA**. Minimal invocation:

```bash
pixi run -e cuda goal-finetune model.backbone.checkpoint_path=/abs/path/logs/simurgh_gmd26/runs/<run>/checkpoints \
    'data.train_paths=[data/dicarboxylic/train.xyz]' 'data.val_paths=[data/dicarboxylic/val.xyz]'
```

`goal-train --config-name finetune/simurgh_lora` runs the same config through the full training path (`GOALCheckpointManager`, archives, resume) and is the recommended route; see [LoRA fine-tuning of SIMURGH](../../integrations/lora_finetune.md).

Other configs build the backbone in one of three modes:

| Mode | Trigger | What happens |
|---|---|---|
| Local checkpoint | `model.backbone.local_checkpoint=/path` | `adapter_cls.from_local(path)` |
| Hub weights | `model.backbone.pretrained=true` (+ `model.backbone.variant=large`) | `adapter_cls.from_pretrained(variant=...)` |
| Fresh | neither | `adapter_cls(**backbone_kwargs)` (how `simurgh_lora` loads its checkpoint) |

`training.freeze_backbone=true` sets `requires_grad=False` on every backbone parameter (linear probing). Modular backbones need a `model.head` block; monolithic ones (`head: null`, such as `simurgh_lora`) are supported. Checkpointing uses a fixed `GOALCheckpoint` (top-3 on `val/total` plus `last.ckpt` in `training.checkpoint_dir`), and the command auto-resumes from `last.ckpt`.

```bash
pixi run -e cuda goal-finetune --config-name my_adapter_experiment \
    model.backbone.name=mace-medium +model.backbone.pretrained=true \
    +training.freeze_backbone=true
```

> [!TIP]
> For MACE and UMA, prefer the in-process trainable backbones run through `goal-train`: `goal-train --config-name finetune/mace` or `finetune/uma`. They use the same `GOALModule`, `GOALCheckpointManager`, resume and archive machinery as native models, and support `head_only`, `full` and `lora` strategies. See [Fine-tuning overview](../finetuning/overview.md).

## `goal-finetune-upet`

UPET/PET models are fine-tuned through [metatrain](https://github.com/metatensor/metatrain), not inside GOAL. This command writes a metatrain `options.yaml` and runs `mtt train` as a subprocess. Plain argparse; no Hydra.

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--model` | yes | — | Base model `<family>-<size>`: family `pet-mad`, `pet-oam`, `pet-omat` or `pet-spice`; size `xs`, `s`, `m`, `l` or `xl` (for example `pet-omat-l`) |
| `--train` | yes | — | Training ExtXYZ file (must exist) |
| `--val` | no | none (10% validation split) | Validation ExtXYZ file |
| `--output` | yes | — | Run directory (created). `options.yaml` and `model.ckpt` go here. |
| `--epochs` | no | `100` | `architecture.training.num_epochs` |
| `--lr` | no | `1e-4` | `architecture.training.learning_rate` |
| `--dry-run` | no | off | Write `options.yaml` only and do not launch `mtt` |

```bash
uv pip install --python .pixi/envs/cuda/bin/python metatrain upet   # not part of the pixi environments

# Inspect the generated options first
pixi run -e cuda goal-finetune-upet --model pet-mad-s --train data/train.xyz --val data/val.xyz \
    --output runs/upet_ft --dry-run
cat runs/upet_ft/options.yaml

# Then run it
pixi run -e cuda goal-finetune-upet --model pet-mad-s --train data/train.xyz --val data/val.xyz \
    --output runs/upet_ft --epochs 200 --lr 5e-5
```

The generated file expects `energy` and `forces` keys in the ExtXYZ, sets `device: cuda` and `base_precision: 64`, and uses `finetune.method: full`. Edit `options.yaml` and run `mtt train` yourself if your metatrain version needs different field names. Use the resulting checkpoint for MD with `CalculatorFactory.create("upet", checkpoint_path="runs/upet_ft/model.ckpt", ...)`. See [UPET fine-tuning](../finetuning/upet.md).

## `goal-tune`

Hyperparameter search, dispatched on `hparams_search.method`:

| `method` | Backend | Extra deps |
|---|---|---|
| `tuner` (default) | Lightning `Tuner`: LR finder and batch-size scaling | none |
| `ray` | Ray Tune with ASHA/PBT scheduler, Optuna/HyperOpt/random search | `[tune]` (see [Installation](../../getting-started/installation.md#ray-tune-and-optuna)) |
| `wandb` | W&B Sweeps (bayes, grid or random, Hyperband early termination) | none (`wandb` is core) |

Its default config is the ready-made ARACE search, so the minimal invocation is:

```bash
pixi run -e cuda goal-tune          # = --config-name simurgh_gmd26_hpo (Ray Tune + Optuna, needs [tune])
```

To use the generic presets in `src/goal/configs/hparams_search/` with any experiment, put `goal/configs` on the search path and append them **into the global package**:

```bash
# Lightning LR finder on the default experiment
pixi run -e cuda goal-tune --config-dir src/goal/configs --config-name simurgh_gmd26 +hparams_search@_global_=basic

# Ray Tune (ASHA + Optuna) over lr / weight_decay / loss weight / EMA decay
pixi run -e cuda goal-tune --config-dir src/goal/configs --config-name simurgh_gmd26 +hparams_search@_global_=ray_tune

# W&B Sweep, or resume one
pixi run -e cuda goal-tune --config-dir src/goal/configs --config-name simurgh_gmd26 +hparams_search@_global_=wandb_sweep
pixi run -e cuda goal-tune --config-dir src/goal/configs --config-name simurgh_gmd26 +hparams_search@_global_=wandb_sweep \
    hparams_search.sweep_id=<SWEEP_ID>
```

> [!WARNING]
> The preset files wrap their content in `hparams_search:`, so `+hparams_search=ray_tune` (without `@_global_`) nests it twice (`cfg.hparams_search.hparams_search.method`), and `goal-tune` silently falls back to `method: tuner`. Use `@_global_` as shown, and confirm with `--cfg job -p hparams_search`.

Caveats of the tune path:

- It builds the backbone and head **directly from the config**, with no dataset-derived injection. Pin `model.backbone.elements` (as `simurgh_gmd26_hpo.yaml` does) and do not rely on `atomic_energies.mode: dataset`.
- Monolithic configs (`head: null`) are supported; the head is built with the same rules as `goal-train`.
- Ray and W&B trials build their own `L.Trainer` (`accelerator="auto"`, no checkpointing) and use `hparams_search.max_epochs` (Ray) or `training.max_epochs` (W&B, default 500) as the epoch budget, not `trainer.max_epochs`.

See [Hyperparameter tuning](../training/hyperparameter-tuning.md).

## `goal-inspect`

This command prints provenance and model facts for a checkpoint **without building the model**, so it works even when the current source can no longer construct it.

```text
usage: goal-inspect [-h] path
```

`path` can be any of the three checkpoint formats:

```bash
pixi run -e cuda goal-inspect logs/simurgh_gmd26/runs/<run>/checkpoints/best_val_forces_mae.simurgh   # archive
pixi run -e cuda goal-inspect logs/simurgh_gmd26/runs/<run>/checkpoints/                              # managed directory
pixi run -e cuda goal-inspect logs/simurgh_gmd26/runs/<run>/checkpoints/last.ckpt                      # bare .ckpt
```

The report contains the format, whether the model source is frozen (self-contained), the GOAL version, git commit, save time, model class, elements, the number of dedicated and rare artisan pairs, average neighbours, scale, and, from the `.json` sidecar, the epoch, pool and `val/*` metrics. For a directory it also lists the available checkpoints. It exits with code 1 if the path does not exist.

## `goal-pack-archive`

This command packs one checkpoint from a `GOALCheckpointManager` directory into a single shippable `.simurgh` file containing the weights, resolved config, metadata and frozen model source. You can run it during training to ship the current best model, or afterwards. (The manager already packs `best_<metric>.simurgh` automatically at the end of training when `checkpoint_manager.save_archive: true`.)

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--checkpoint-dir` | yes | — | Checkpoint directory written by `GOALCheckpointManager` |
| `--checkpoint` | no | `best` | `best`, `last`, `epoch=N`, or a `.ckpt` filename or path |
| `--output` | no | `<ckpt stem>.simurgh` inside the checkpoint directory | Output path |

```bash
pixi run -e cuda goal-pack-archive --checkpoint-dir logs/simurgh_gmd26/runs/<run>/checkpoints \
    --checkpoint best --output models/simurgh_gmd26.simurgh

pixi run -e cuda goal-pack-archive --checkpoint-dir logs/simurgh_gmd26/runs/<run>/checkpoints --checkpoint epoch=1124
```

It prints `Packed <ckpt> -> <archive>` on success and exits with code 1 on a missing or invalid directory. Load the archive with `GOALCalculator(checkpoint_path="model.simurgh")`. See [Checkpoints & archives](../deployment/archives.md).

## `goal-pack-zarr`

Converts structure files (any `ase.io` format) into a GOAL Zarr store for `data.dataset_type: zarr`:

```bash
pixi run -e cuda goal-pack-zarr --input data/train.xyz --output data/gmd.zarr --split train
pixi run -e cuda goal-pack-zarr --input data/val.xyz --output data/gmd.zarr --split val --compute-edges --cutoff 8.0
```

Options: `--input` (one or more files), `--output`, `--split` (default `train`), `--compute-edges` with `--cutoff` (store neighbour lists), `--neighbor-list-backend`, `--overwrite`. See [Zarr data](../../integrations/zarr_data.md).

## `goal-distill`

A roadmap placeholder for teacher → student distillation. It prints what to do today and exits with status 1. See [Roadmap stubs](../../integrations/roadmap_stubs.md).

## MD commands

`goal-simulate`, `goal-simulate-mts` and `goal-simulate-batch` belong to `goal.md` and are documented in the [MD CLI reference](../../md/reference/cli.md) and [Batched simulation](../../integrations/batched_simulation.md).
