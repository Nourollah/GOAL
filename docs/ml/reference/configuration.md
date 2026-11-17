# Configuration

This page is the reference for how `goal.ml` is configured with [Hydra](https://hydra.cc/). It covers the layout of the bundled `configs/` directory, every ready-made experiment config, the default SIMURGH model block, command-line overrides, multiruns, the output directory, and the Hydra pitfalls that matter in this repository. Read it before you write your own experiment file.

## The design: one file per experiment

GOAL uses **self-contained experiment configs**. Every training run is described by a **single YAML file** in [`src/goal/configs/ml/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/configs/ml). That one file holds the data, model, trainer, losses and training stages, checkpointing, callbacks, loggers, distributed strategy, resume policy and the Hydra run directory, with every available option documented inline as comments.

There are no `model=`, `data=` or `trainer=` config groups to compose. That was the old multi-file layout, and command lines such as `goal-train trainer=ddp data=hdf5 model=invariant_gnn` from older docs **no longer work**. Instead:

- you **pick a file** with `--config-name`;
- you **change values** with dotted overrides (`trainer.max_epochs=100`).

The benefit is reproducibility and readability. The file you launched is the experiment, and `diff a.yaml b.yaml` is the complete difference between two runs.

`goal-train` (and its alias `goal-train-ml`) is declared as

```python
@hydra.main(version_base=None, config_path=CONFIGS_ML_DIR, config_name="simurgh_dev_gmd26_FD")
```

in [`src/goal/ml/cli/train.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/cli/train.py). `CONFIGS_ML_DIR` is the **absolute** path of `src/goal/configs/ml/`, so the configs are found no matter which directory you launch from or whether you run the console script or `python -m goal.ml.cli.train`.

## Layout of `configs/`

The configs ship inside the package: `src/goal/configs/` in a clone, `goal/configs/` in an installed wheel or conda package. `goal.CONFIGS_DIR` holds the absolute path (`python -c 'import goal; print(goal.CONFIGS_DIR)'`), and every CLI finds it on its own, so no clone is needed to run the bundled experiments.

```text
src/goal/configs/
├── ml/                     # goal-train's config directory: one file = one experiment
│   ├── *.yaml              #   experiment configs (table below)
│   ├── simurgh_dev/        #   shared SIMURGH bases (_base, _rmd17, _md22), not runs
│   ├── splits/             #   GMD split presets (+splits@data=...)
│   ├── finetune/           #   foundation-model fine-tuning (mace, uma)
│   └── ablations/simurgh_dev/ #  SIMURGH ablation arms, one key each
├── hparams_search/         # goal-tune schemas: basic, ray_tune, wandb_sweep
├── debug/                  # debug presets: default, fdr, limit, overfit, profiler
├── md/                     # goal.md configs (see MD → Reference → Configuration)
├── hydra/, paths/, extras/ # legacy defaults from the multi-file layout (ml configs inline their own)
└── local/                  # machine-specific, untracked
```

For the `goal.md` side, see the [MD configuration reference](../../md/reference/configuration.md).

## Experiment configs in `src/goal/configs/ml/`

Every config below can be run as `goal-train --config-name <name>`. The *Backbone* column is the `model.backbone.name` registry key, and *Dataset* is `data.dataset_type`.

Every `simurgh_dev_<task>.yaml` composes `simurgh_dev/_base.yaml` (through `_rmd17` or `_md22` for those benchmarks) and sets only its task data, so two task configs differ only in `data`. The model is described on the [SIMURGH (development)](../models/simurgh-dev.md) page.

### SIMURGH on GMD-26

| Config | Backbone | Dataset | Description |
|---|---|---|---|
| [`simurgh_dev_gmd26_FD`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_dev_gmd26_FD.yaml) | `simurgh_dev` | `gmd` | **Default.** FragmentDuplication: C4–C10 monoacids for training, the matching diacids as OOD test. MFC on. |
| [`simurgh_dev_gmd26_FF`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_dev_gmd26_FF.yaml) | `simurgh_dev` | `gmd` | FragmentFusion. |
| [`simurgh_dev_gmd26_FCE`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_dev_gmd26_FCE.yaml) | `simurgh_dev` | `gmd` | FragmentChainExtension: C2–C6 alkanes for training, longer alkanes as OOD test. |
| [`simurgh_dev_gmd26_FCB`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_dev_gmd26_FCB.yaml) | `simurgh_dev` | `gmd` | FragmentCombination. |
| [`simurgh_dev_gmd26_FD_plus_alkane`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_dev_gmd26_FD_plus_alkane.yaml) | `simurgh_dev` | `gmd` | FragmentDuplication plus one alkane (C8H18) from the chain-extension set (split preset `splits/gmd26_FD_plus_alkane`). |

### Standard benchmarks (rMD17 / MD22 / 3BPA)

| Config | Backbone | Dataset | Description |
|---|---|---|---|
| `simurgh_dev_rmd17_<molecule>` | `simurgh_dev` | `rmd17` | One file per rMD17 molecule (aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic_acid, toluene, uracil); 950 training / 50 validation structures. MFC off. |
| `simurgh_dev_md22_<system>` | `simurgh_dev` | `md22` | One file per MD22 system (Ac-Ala3-NHMe, DHA, stachyose, AT-AT, AT-AT-CG-CG, buckyball-catcher, double-walled_nanotube) with its published training size. MFC off. |
| [`simurgh_dev_3bpa`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_dev_3bpa.yaml) | `simurgh_dev` | `xyz` | 3BPA: 500 training structures at 300 K; tests at 300, 600 and 1200 K and on the dihedral slices. |
| [`hyperspec_md17`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/hyperspec_md17.yaml) | `hyperspec` | `md17` | HyperSpec equivariant GNN baseline on MD17 aspirin (950 / 50). |

The `md17`, `rmd17`, `md22` and `qm9` dataset types come from [`src/goal/ml/data/benchmarks/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/ml/data/benchmarks) (module `goal.ml.data.benchmarks`), which every CLI registers automatically. See [Benchmark datasets](../data/benchmarks.md).

### Fine-tuning: `src/goal/configs/ml/finetune/`

| Config | Backbone | Description |
|---|---|---|
| [`finetune/mace`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/mace.yaml) | `mace_finetune` | MACE-MP fine-tuning in-process through `GOALModule` (`head: null`). `model.backbone.strategy` is `head_only`, `full` or `lora`. Run it from the `.venv-mace` interpreter. |
| [`finetune/uma`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/uma.yaml) | `uma_finetune` | UMA / FairChem fine-tuning (pixi `cuda` environment plus FairChem, see [Installation](../../getting-started/installation.md#fairchem-uma)). |

`finetune/mace` is also the default config of `goal-finetune`. `eval.yaml` in `src/goal/configs/ml/` is `goal-eval`'s default config (`ckpt_path`, `checkpoint`, `use_ema`, `data` overrides, test-loop `trainer`); see [Evaluation](../deployment/evaluation.md).

```bash
.venv-mace/bin/goal-train --config-name finetune/mace          # MACE needs its own venv
pixi run -e cuda goal-train --config-name finetune/uma data.root=/path/to/my_xyz_data
```

Both files start with `# @package _global_` because they live in a sub-directory (see [below](#sub-directory-configs-and-_global_)). See [Fine-tuning overview](../finetuning/overview.md).

### Ablation arms: `src/goal/configs/ml/ablations/simurgh_dev/`

One file per arm of the SIMURGH ablation programme. Each composes a task config (GMD FragmentDuplication by default) and changes one key, so the experiment *is* the config diff; the `sNN_` prefix gives the stage.

```bash
pixi run -e cuda goal-train --config-name ablations/simurgh_dev/s05_num_rbf_16
```

The protocol (seeds, selection rule, gates) is in [`src/goal/configs/ml/ablations/simurgh_dev/README.md`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/ablations/simurgh_dev/README.md).

## Anatomy of an experiment file

Top-level sections of a composed SIMURGH config (`src/goal/configs/ml/simurgh_dev/_base.yaml` plus a task file), in file order:

| Section | Consumed by | Purpose |
|---|---|---|
| `experiment` | loggers | name, description, tags |
| `seed` | `L.seed_everything` | global seed (`seed: 42`) |
| `task_name`, `run_name_suffix`, `run_name` | Hydra run dir, W&B | output naming (see [below](#output-directory-layout)) |
| `hydra` | Hydra | run/sweep directories and log file |
| `paths` | interpolations | `root_dir` (`$PROJECT_ROOT`, else `$PWD`), `data_dir`, `log_dir`, `output_dir` |
| `data` | `GOALDataModule` | dataset type, loading mode, cutoff, batching, key mapping, fragmentation |
| `model.backbone`, `model.head` | registries | backbone kwargs (`name` = registry key) and head (`null` for monolithic models) |
| `trainer` | `lightning.Trainer` via `_target_` | epochs, devices, precision, validation cadence, … |
| `training` | `GOALModule` / `train.py` | optimiser, scheduler, EMA, `compile_model`, `slurm_mode`, `performance`, `stages`, `losses` |
| `dataset_stats` | `train.py` | reuse or recompute the dataset-derived backbone values on resume (see [Resuming](../training/resuming.md)) |
| `checkpoint_manager` | `GOALCheckpointManager` | top-k / interval / last pools, `.simurgh` archive packing |
| `callbacks` | instantiated by `_target_` | progress bar, model summary, early stopping, … |
| `logger` | instantiated by `_target_` | CSV (always on), W&B, plus commented TensorBoard / MLflow / Neptune / Aim / Comet |
| `strategy` (not in the SIMURGH configs; add with `++strategy.name=ddp`) | `build_strategy(cfg)` | `name` (`auto`, `single`, `ddp`, `fsdp`, `fsdp2`, `deepspeed_zero1/2/3`, `deepspeed_zero3_offload`) and its knobs |
| `resume`, `ckpt_path` (add with `+`) | `resolve_resume` / `goal-eval` | continuing finished runs; checkpoint to evaluate |
| `hparams_search` (add a preset) | `goal-tune` | search definition |

> [!NOTE]
> `trainer.max_epochs` is authoritative. Do not set `training.max_epochs`: it has no effect on `goal-train`.

### Values `goal-train` fills in from the data

Some backbone keys are deliberately left `null` or omitted, because `train.py` computes them from the training split before it builds the model:

| Key | When it is filled in |
|---|---|
| `model.backbone.elements` | Always, for `simurgh_dev` and its `simurgh_dev_abl_*` arms. The dataset is authoritative and any value in the config is overwritten. |
| `atomic_energies.values` | When `mode: dataset` (or `compute_from_dataset: true`) and `values` is null: least-squares fit of per-element references. |
| `scale` | `scale: force_rms` (SIMURGH): RMS of the training force components. Otherwise, when `mode` is `dataset` or `provided` and `scale: null`. |
| `avg_num_neighbors` | `simurgh_dev` with `avg_num_neighbors: null`. |
| `atom_statistics` | When the backbone requests descriptor statistics (SIMURGH's `attention.pan_gate`). |

The values are recorded in the run's `dataset_stats` block and reused on resume. These injections happen **only in `goal-train`**. `goal-eval` rebuilds the model from the checkpoint's saved config, and `goal-tune` builds the backbone straight from the experiment config, so pin `model.backbone.elements` (and the other values) in a config you tune.

## The default model config

The start of the `model:` block of `simurgh_dev/_base.yaml` (print the whole block with `goal-train --cfg job -p model`):

```yaml
model:
  backbone:
    name: simurgh_dev
    core: ace # ace | ece
    cutoff: ${data.cutoff}
    elements: null # derived from the training set by goal-train
    embedding_dim: 32
    num_channels: 128
    message_irreps: "128x0e + 128x1o + 128x2e"
    lmax_sh: 3
    num_interactions: 2
    correlation: 3
    product_basis: symmetric # symmetric | ladder | ladder_reduced
    num_rbf: 8 # stage 5 sweeps 8 | 16 | 32
    envelope_p: 5
    radial_mlp: [64, 64, 64]
    pair_conditioned_radial: true
    element_weights: index # index | embedding
    avg_num_neighbors: null # derived from the training set by goal-train
    node_norm: none # none | rms (Muon arm only)
    kernels: auto # auto | cuequivariance | e3nn
    dtype: float32 # float64 only in stage 13, on A100/H100/H200
    # ... radial, a_basis, attention, readout, pair_line, mfc, electrostatics, d3, aux
  head: null
```

SIMURGH is monolithic (`head: null`): its forces are −∂E/∂R inside the model. Every optional capability defaults to off and unknown keys raise `ValueError`. The key-by-key description is on the [SIMURGH (development)](../models/simurgh-dev.md) page.

`train.py` enforces the head pairing. A monolithic backbone with a `head:` block, or a modular backbone without one, raises a `ValueError` before training starts.

## Overriding values from the command line

Any key can be overridden with Hydra's dotted syntax. Lists use brackets, and strings containing spaces or `+` need quoting.

```bash
# Choose an experiment file
pixi run -e cuda goal-train --config-name simurgh_dev_rmd17_aspirin

# Training budget and optimiser
pixi run -e cuda goal-train trainer.max_epochs=300 training.optimizer.lr=5e-3 training.ema.decay=0.999

# Architecture
pixi run -e cuda goal-train model.backbone.num_interactions=3 model.backbone.correlation=4
pixi run -e cuda goal-train model.backbone.num_channels=64 'model.backbone.message_irreps="64x0e + 64x1o + 64x2e"'

# Loss weights (list index into training.losses)
pixi run -e cuda goal-train training.losses.0.weight=1.0 training.losses.1.weight=10.0

# Cutoff: model.backbone.cutoff follows data.cutoff automatically
pixi run -e cuda goal-train data.cutoff=6.0

# Your own data (one file, split by data.split_ratio)
pixi run -e cuda goal-train data.dataset_type=xyz '~data.task' data.root=/data/mysys/all.xyz

# Hardware
pixi run -e cuda goal-train trainer.accelerator=cpu
pixi run -e cuda goal-train trainer.devices=4 ++strategy.name=ddp
pixi run -e cuda goal-train trainer.precision=bf16-mixed

# Disable W&B for a quick local run
pixi run -e cuda goal-train '~logger.wandb'

# Name the run
pixi run -e cuda goal-train run_name_suffix=_lr5e-3
```

Hydra's prefix operators:

| Syntax | Meaning |
|---|---|
| `key=value` | Override an existing key. Fails if the key does not exist, which catches typos. |
| `+key=value` | Add a key that is not in the config. |
| `++key=value` | Add or override. |
| `~key` | Delete a key (for example `~logger.wandb`, `~data.task`). |

> [!TIP]
> Preview the composed config without starting a run: `goal-train <overrides> --cfg job` prints everything, and `--cfg job -p model.backbone` prints one node. `--info` shows the search path and defaults list.

### Using the shared presets (`debug/`, `hparams_search/`)

The preset groups live in `configs/`, not in `src/goal/configs/ml/`, so add `configs/` to the search path with `--config-dir` and **append** them with `+` (the experiment files have no defaults list to override):

```bash
pixi run -e cuda goal-train --config-dir src/goal/configs +debug=fdr       # 1 train/val/test step (fast_dev_run), CPU, loggers off
pixi run -e cuda goal-train --config-dir src/goal/configs +debug=limit     # 3 epochs on 1% train / 5% val+test
pixi run -e cuda goal-train --config-dir src/goal/configs +debug=overfit   # overfit 3 batches for 20 epochs
pixi run -e cuda goal-train --config-dir src/goal/configs +debug=profiler  # 1 epoch with the simple profiler
```

The `debug/*.yaml` files carry `# @package _global_`, so they merge at the top level as intended. The `hparams_search/*.yaml` files do **not**, and they need the `@_global_` package override. See [CLI → goal-tune](cli.md#goal-tune).

## Multirun sweeps

`-m` / `--multirun` launches one job per combination of comma-separated values:

```bash
pixi run -e cuda goal-train -m training.optimizer.lr=1e-2,5e-3,1e-3
pixi run -e cuda goal-train -m model.backbone.num_interactions=1,2 seed=42,43,44   # 6 jobs
```

Jobs run sequentially with Hydra's default basic launcher. Each gets its own numbered sub-directory under `logs/<task_name>/multiruns/<run_name>/`. For model-based search with early stopping, use [`goal-tune`](../training/hyperparameter-tuning.md).

> [!WARNING]
> Auto-resume keys off `training.checkpoint_dir`, which sits inside the per-job output directory. Each multirun job therefore starts fresh. Do not point several jobs at one shared `resume.from`.

## Output directory layout

`hydra.run.dir` is set in every experiment file to

```text
${paths.log_dir}/${task_name}/runs/${run_name}
run_name = ${now:%Y-%m-%d_%H-%M-%S}_${data.dataset_type}_${model.backbone.name}${run_name_suffix}
```

`paths.log_dir` is `<root>/logs`, and `<root>` is `$PROJECT_ROOT` if set, otherwise the directory you launched from. A finished default run looks like this:

```text
logs/simurgh_dev_gmd26_FD/runs/2026-11-16_09-54-59_gmd_simurgh_dev/
├── simurgh_dev_gmd26_FD.log          # Hydra job log (${task_name}.log)
├── .hydra/
│   ├── config.yaml                   # composed config (unresolved)
│   ├── hydra.yaml
│   └── overrides.yaml                # your command-line overrides
├── csv_logs/version_0/metrics.csv    # CSVLogger (always on)
├── wandb/                            # W&B local files (if enabled)
└── checkpoints/                      # GOALCheckpointManager dirpath
    ├── config.yaml                   # fully RESOLVED config, incl. the dataset-derived values
    ├── metadata.json                 # GOAL version, git commit, elements, scale, …
    ├── frozen_source/                # snapshot of every goal .py file the model imports
    ├── checkpoint_state.json         # pool membership, used on resume
    ├── best_val_forces_forces_mae_atomw=0.0100_epoch=1124.ckpt   # top-k pool (+ .json sidecar each)
    ├── interval_epoch=1124.ckpt      # interval pool (+ .json sidecar)
    ├── last.ckpt / last.json         # crash recovery / auto-resume
    ├── best_val_forces_forces_mae_atomw.simurgh   # self-contained archive of the best checkpoint
    └── TRAINING_COMPLETE             # sentinel: stops a SLURM requeue from retraining
```

`task_name` groups runs of one experiment, and the timestamped `run_name` keeps each launch separate. Override either one:

```bash
pixi run -e cuda goal-train task_name=my_study run_name=baseline_v1
```

The same `run_name` is used as the W&B run name, so the directory and the dashboard always agree. The checkpoint layout and the `.simurgh` archive are covered in [Checkpoints & archives](../deployment/archives.md), and resuming in [Resuming & extending](../training/resuming.md).

## Sub-directory configs and `_global_`

Hydra places a config that lives in a **sub-directory** of the search path under a key named after that directory. A file `src/goal/configs/ml/simurgh_dev/_base.yaml` is treated as belonging to the `simurgh_dev` group, so without a directive its whole content lands under `cfg.ablations.*` and the top-level `model`, `data` and `trainer` keys are never set.

This is why every file in `src/goal/configs/ml/simurgh_dev/`, `src/goal/configs/ml/ablations/`, `src/goal/configs/ml/finetune/` and `src/goal/configs/debug/` begins with:

```yaml
# @package _global_
```

For a shared base, a missing header fails silently: its contents are filed under `simurgh_dev:` and every task config composes to an empty experiment. For an ablation arm, the arm's own override is filed under its group, the task config still composes, and **the arm quietly trains the base model**. When you add a config in any sub-directory:

1. Make `# @package _global_` the first line (comments above it are fine; the shared bases keep their explanatory comments below it).
2. Check the composition before you spend GPU hours on it:

   ```bash
   pixi run -e cuda goal-train --config-name ablations/simurgh_dev/my_new_arm --cfg job -p model.backbone
   ```

Top-level files in `src/goal/configs/ml/` (for example `simurgh_dev_gmd26_FD.yaml`) are in the root package and do not need the header.

The opposite problem affects the `src/goal/configs/hparams_search/*.yaml` presets. They have no header *and* already wrap their content in an `hparams_search:` key, so `+hparams_search=ray_tune` produces `cfg.hparams_search.hparams_search.method`, and `goal-tune` falls back to the default `method: tuner`. Append them with `+hparams_search@_global_=ray_tune` instead. See [CLI → goal-tune](cli.md#goal-tune).

## Writing your own experiment config

Copy the closest existing file rather than starting from scratch. A SIMURGH task file is a few lines over `simurgh_dev/_base.yaml`, which documents every key inline.

```bash
cp src/goal/configs/ml/simurgh_dev_gmd26_FD.yaml src/goal/configs/ml/my_system.yaml
# edit: experiment.*, task_name, data.*, atomic_energies, …
pixi run -e cuda goal-train --config-name my_system --cfg job -p data   # check
pixi run -e cuda goal-train --config-name my_system
```

Things to change every time:

- `experiment.name` / `description` / `tags` and `task_name`, so runs do not mix in `logs/` or on W&B;
- the `data` loading mode and paths (four modes, documented at the top of the `data:` block; see [Loading data](../data/loading.md));
- `data.energy_key` / `forces_key` if your files use different labels;
- `model.backbone.atomic_energies` (isolated-atom references for your elements and level of theory, or `mode: dataset`); SIMURGH takes `elements` from the training set;
- `checkpoint_manager.top_k.metric`, if forces are not your selection criterion.
