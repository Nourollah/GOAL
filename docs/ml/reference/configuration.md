# Configuration

This page is the reference for how `goal.ml` is configured with [Hydra](https://hydra.cc/). It covers the layout of the bundled `configs/` directory, every ready-made experiment config, the default SIMURGH ARACE model block, command-line overrides, multiruns, the output directory, and the Hydra pitfalls that matter in this repository. Read it before you write your own experiment file.

## The design: one file per experiment

GOAL uses **self-contained experiment configs**. Every training run is described by a **single YAML file** in [`src/goal/configs/ml/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/configs/ml). That one file holds the data, model, trainer, losses and training stages, checkpointing, callbacks, loggers, distributed strategy, resume policy and the Hydra run directory, with every available option documented inline as comments.

There are no `model=`, `data=` or `trainer=` config groups to compose. That was the old multi-file layout, and command lines such as `goal-train trainer=ddp data=hdf5 model=invariant_gnn` from older docs **no longer work**. Instead:

- you **pick a file** with `--config-name`;
- you **change values** with dotted overrides (`trainer.max_epochs=100`).

The benefit is reproducibility and readability. The file you launched is the experiment, and `diff a.yaml b.yaml` is the complete difference between two runs.

`goal-train` (and its alias `goal-train-ml`) is declared as

```python
@hydra.main(version_base=None, config_path=CONFIGS_ML_DIR, config_name="simurgh_gmd26")
```

in [`src/goal/ml/cli/train.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/cli/train.py). `CONFIGS_ML_DIR` is the **absolute** path of `src/goal/configs/ml/`, so the configs are found no matter which directory you launch from or whether you run the console script or `python -m goal.ml.cli.train`.

## Layout of `configs/`

The configs ship inside the package: `src/goal/configs/` in a clone, `goal/configs/` in an installed wheel or conda package. `goal.CONFIGS_DIR` holds the absolute path (`python -c 'import goal; print(goal.CONFIGS_DIR)'`), and every CLI finds it on its own, so no clone is needed to run the bundled experiments.

```text
src/goal/configs/
├── ml/                     # goal-train's config directory: one file = one experiment
│   ├── *.yaml              #   experiment configs (table below)
│   ├── finetune/           #   foundation-model fine-tuning (mace, uma)
│   └── ablations/          #   SIMURGH ablation study rungs + shared base
├── hparams_search/         # goal-tune schemas: basic, ray_tune, wandb_sweep
├── debug/                  # debug presets: default, fdr, limit, overfit, profiler
├── md/                     # goal.md configs (see MD → Reference → Configuration)
├── hydra/, paths/, extras/ # legacy defaults from the multi-file layout (ml configs inline their own)
└── local/                  # machine-specific, untracked
```

For the `goal.md` side, see the [MD configuration reference](../../md/reference/configuration.md).

## Experiment configs in `src/goal/configs/ml/`

Every config below can be run as `goal-train --config-name <name>`. The *Backbone* column is the `model.backbone.name` registry key, and *Dataset* is `data.dataset_type`.

### SIMURGH family on GMD-26

| Config | Backbone | Dataset | Description |
|---|---|---|---|
| [`simurgh_gmd26`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_gmd26.yaml) | `simurgh_arace` | `trajectory` | **Default.** Modular SIMURGH ARACE (artisan layer + ACE block per round) with an `energy_forces` head on GMD-26 FragmentDuplication (H, C, O). Fragment interaction add-on enabled. |
| [`simurgh_gmd26_hpo`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_gmd26_hpo.yaml) | `simurgh_arace` | `trajectory` | Composes `simurgh_gmd26` and adds a Ray Tune search space over ARACE width, depth and rounds plus LR and weight decay. Run with `goal-tune`, not `goal-train`. |
| [`simurgh_ace_first_gmd26`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_ace_first_gmd26.yaml) | `simurgh` | `trajectory` | Legacy ACE-first SIMURGH: environment dressing first, artisan bank as the final readout. Uses `dressing_kwargs` / `artisan_config` keys. |
| [`monolithic_arace_gmd26`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/monolithic_arace_gmd26.yaml) | `monolithic_arace` | `trajectory` | Self-contained ARACE that computes its own forces (`head: null`). Research and ablation baseline. |
| [`nf_simurgh_gmd26`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/nf_simurgh_gmd26.yaml) | `nf_simurgh` | `trajectory` | NF_SIMURGH: ARACE rounds whose per-atom energies are pooled onto fragment virtual nodes that interact pairwise. A/B partner of `monolithic_arace_gmd26`. |
| [`nf_simurgh_gmd26_typecond`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/nf_simurgh_gmd26_typecond.yaml) | `nf_simurgh` | `trajectory` | NF_SIMURGH with type-conditioned virtual nodes, compared against the `nf_simurgh_gmd26` baseline (FragmentDuplication). |
| [`nf_simurgh_gmd26_FF`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/nf_simurgh_gmd26_FF.yaml) | `nf_simurgh` | `trajectory` | The type-conditioned model on the GMD **FragmentFusion** split (`task_name: nf_simurgh_gmd26_fusion`). |
| [`simurgh_simplified_routed_gmd26`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_simplified_routed_gmd26.yaml) | `simurgh_simplified_routed` | `trajectory` | SIMURGH Simplified, Routed: three parallel lines (atom / pair / fragment) with frequency-routed artisans and a fragment mixture of experts, additive readout. |
| [`simurgh_simplified_monolithic_gmd26`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_simplified_monolithic_gmd26.yaml) | `simurgh_simplified_monolithic` | `trajectory` | SIMURGH Simplified, Monolithic: the same three lines, each a single stack of simple equivariant layers. Ablation baseline for the routed version. |

### Benchmarks (MD17 / MD22 / QM9 / OMol25)

| Config | Backbone | Dataset | Description |
|---|---|---|---|
| [`nf_simurgh_md17`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/nf_simurgh_md17.yaml) | `nf_simurgh` | `rmd17` | NF_SIMURGH (type-conditioned virtual nodes) on rMD17 aspirin. |
| [`nf_simurgh_md17_benzene`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/nf_simurgh_md17_benzene.yaml) | `nf_simurgh` | `rmd17` | rMD17 benzene, an atom-level baseline. Benzene is one fragment (K=1), so the fragment channel is inert. |
| [`nf_simurgh_md22`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/nf_simurgh_md22.yaml) | `nf_simurgh` | `md22` | NF_SIMURGH on MD22 Ac-Ala3-NHMe. Switching molecule also means changing elements and `avg_num_neighbors` (see the table in the file). |
| [`nf_simurgh_qm9`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/nf_simurgh_qm9.yaml) | `nf_simurgh` | `qm9` | NF_SIMURGH on QM9 U0 (energy only; QM9 has no forces). |
| [`nf_simurgh_omol25`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/nf_simurgh_omol25.yaml) | `nf_simurgh` | `aselmdb` | NF_SIMURGH on a filtered organic subset of OMol25 `train_4M`. |
| [`simurgh_simplified_routed_md17`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_simplified_routed_md17.yaml) | `simurgh_simplified_routed` | `rmd17` | SIMURGH Simplified Routed on rMD17 aspirin. |
| [`simurgh_simplified_monolithic_md17`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_simplified_monolithic_md17.yaml) | `simurgh_simplified_monolithic` | `rmd17` | SIMURGH Simplified Monolithic on rMD17 aspirin (ablation baseline). |
| [`simurgh_md17`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_md17.yaml) | `simurgh` | `md17` | Legacy ACE-first SIMURGH on MD17 aspirin (rMD17, 950 train / 50 val). |
| [`hyperspec_md17`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/hyperspec_md17.yaml) | `hyperspec` | `md17` | HyperSpec equivariant GNN baseline on MD17 aspirin (950 / 50). |

The `md17`, `rmd17`, `md22` and `qm9` dataset types come from [`src/goal/ml/data/benchmarks/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/ml/data/benchmarks) (module `goal.ml.data.benchmarks`), which every CLI registers automatically. See [Benchmark datasets](../data/benchmarks.md).

### Fine-tuning: `src/goal/configs/ml/finetune/`

| Config | Backbone | Description |
|---|---|---|
| [`finetune/mace`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/mace.yaml) | `mace_finetune` | MACE-MP fine-tuning in-process through `GOALModule` (`head: null`). `model.backbone.strategy` is `head_only`, `full` or `lora`. Run it from the `.venv-mace` interpreter. |
| [`finetune/uma`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/uma.yaml) | `uma_finetune` | UMA / FairChem fine-tuning (pixi `cuda` environment plus FairChem, see [Installation](../../getting-started/installation.md#fairchem-uma)). |
| [`finetune/simurgh_lora`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/simurgh_lora.yaml) | `simurgh_lora` | LoRA on a trained SIMURGH checkpoint (`model.backbone.checkpoint_path`), any pixi environment. Default config of `goal-finetune`. See [LoRA fine-tuning](../../integrations/lora_finetune.md). |

`eval.yaml` in the same directory is `goal-eval`'s default config (`ckpt_path`, `checkpoint`, `use_ema`, `data` overrides, test-loop `trainer`); see [Evaluation](../deployment/evaluation.md).

```bash
.venv-mace/bin/goal-train --config-name finetune/mace          # MACE needs its own venv
pixi run -e cuda goal-train --config-name finetune/uma data.root=/path/to/my_xyz_data
```

Both files start with `# @package _global_` because they live in a sub-directory (see [below](#sub-directory-configs-and-_global_)). See [Fine-tuning overview](../finetuning/overview.md).

### Ablation study: `src/goal/configs/ml/ablations/`

These configs drive the isolated `simurgh_ablation` model: one stripped base model (`01_gmd26_base`), one rung per module (`02`–`14`), controls (`90`–`93`: two extra seeds, a width-matched base and a 3-round base), and phase-2 pairings of `body_order` with each other module (`201`–`208`). Every rung composes `_ablation_base.yaml` and overrides exactly one thing, so the experiment *is* the config diff.

```bash
pixi run -e cuda goal-train --config-name ablations/01_gmd26_base
pixi run -e cuda goal-train --config-name ablations/05_gmd26_body_order
```

The study design, parameter counts and run order are in [`src/goal/configs/ml/ablations/README.md`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/ablations/README.md) and on the [Ablation study](../models/ablation-study.md) page.

## Anatomy of an experiment file

Top-level sections of `src/goal/configs/ml/simurgh_gmd26.yaml`, in file order:

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
| `checkpoint_manager` | `GOALCheckpointManager` | top-k / interval / last pools, `.simurgh` archive packing |
| `callbacks` | instantiated by `_target_` | progress bar, model summary, early stopping, … |
| `logger` | instantiated by `_target_` | CSV (always on), W&B, plus commented TensorBoard / MLflow / Neptune / Aim / Comet |
| `strategy` | `build_strategy(cfg)` | `name` (`auto`, `single`, `ddp`, `fsdp`, `fsdp2`, `deepspeed_zero1/2/3`, `deepspeed_zero3_offload`) and its knobs |
| `resume`, `ckpt_path` | `resolve_resume` / `goal-eval` | continuing finished runs; checkpoint to evaluate |
| `hparams_search` (commented) | `goal-tune` | search definition |

> [!NOTE]
> `trainer.max_epochs` is authoritative. Do not set `training.max_epochs`: it has no effect on `goal-train`.

### Values `goal-train` fills in from the data

Some backbone keys are deliberately left `null` or omitted, because `train.py` computes them from the training split before it builds the model:

| Key | When it is filled in |
|---|---|
| `model.backbone.elements` | Always, for `simurgh`, `simurgh_ace_first`, `simurgh_arace`, `simurgh_monolithic` and `monolithic_arace`. The dataset is authoritative and any value in the config is overwritten. |
| `atomic_energies.values` | When `mode: dataset` (or `compute_from_dataset: true`) and `values` is null: least-squares fit of per-element references. |
| `scale` | When `mode` is `dataset` or `provided` and `scale: null`. |
| `avg_num_neighbors` | `simurgh_arace` with `avg_num_neighbors: null`, and legacy backbones with `dressing_kwargs.avg_num_neighbors: null`. |
| `artisan.pair_counts` / `artisan_config.pair_counts` | When `rare_artisan.enabled: true`. The counts are also written back into the saved config, so the checkpoint can be rebuilt. |

These injections happen **only in `goal-train`**. `goal-eval` and `goal-tune` build the backbone straight from the config, which is why `simurgh_gmd26_hpo.yaml` pins `model.backbone.elements: [1, 6, 8]`.

## The default model config (ARACE)

The `model:` block of `simurgh_gmd26.yaml`, as composed (`goal-train --cfg job -p model`):

```yaml
model:
  backbone:
    name: simurgh_arace
    num_rounds: 2                 # artisan layer + ACE block, repeated
    share_artisan_weights: false  # true = one artisan bank shared by all rounds
    cutoff: ${data.cutoff}        # always interpolated from the data cutoff (8.0 Å)
    embedding_dim: 64
    num_elements: 9               # embedding table size; must exceed max(Z) in the data
    avg_num_neighbors: null       # computed from the training set by goal-train
    artisan:                      # per-element-pair equivariant artisans
      architecture: equivariant
      hidden_irreps: "16x0e + 16x1o + 16x2e"
      num_layers: 1               # 1 = tensor product only; >1 adds residual layers
      num_rbf: 8
      radial_hidden: 32
      n_scalar_out: 16
      final_hidden: 16
      element_conditioned: true
    atomic_energies:
      mode: provided              # learned | dataset | provided
      values:                     # isolated-atom PBE/def2-SVP references (eV)
        1: -13.595298
        6: -1028.481145
        7: -1483.895483
        8: -2041.117735
      compute_from_dataset: false
    scale: null                   # computed for dataset/provided modes
    fragment_interaction:         # optional add-on; ENABLED by the section's presence
      irreps_hidden: null         # null = reuse artisan.hidden_irreps
      num_rbf: 8
      radial_hidden: 32
      num_layers: 1
      init_zero: true
      max_fragments: 512
      cutoff: 12.0                # max fragment-centroid separation (Å)
  head:
    name: energy_forces           # conservative forces, F = -dE/dr
    irreps_in: ${model.backbone.artisan.hidden_irreps}
    hidden_dim: 128
```

Two optional ARACE add-ons, `fragment_interaction` and `adaptive_gate`, are switched on **by the presence of their section**. Comment the block out (or leave it empty) and the module is never built: zero parameters and a forward pass bit-identical to plain ARACE. The shipped default has `fragment_interaction` enabled, which also requires `data.compute_fragment_index: true` (set in the same file). See [SIMURGH](../models/simurgh.md) for the physics behind each key.

To switch architecture, prefer the ready-made files over editing `name`. The legacy ACE-first backbone uses different keys (`dressing_kwargs`, `artisan_config`):

```bash
pixi run -e cuda goal-train --config-name simurgh_ace_first_gmd26   # legacy ACE-first
pixi run -e cuda goal-train --config-name monolithic_arace_gmd26    # monolithic ARACE, head: null
```

`train.py` enforces the head pairing. A monolithic backbone with a `head:` block, or a modular backbone without one, raises a `ValueError` before training starts.

## Overriding values from the command line

Any key can be overridden with Hydra's dotted syntax. Lists use brackets, and strings containing spaces or `+` need quoting.

```bash
# Choose an experiment file
pixi run -e cuda goal-train --config-name nf_simurgh_gmd26

# Training budget and optimiser
pixi run -e cuda goal-train trainer.max_epochs=300 training.optimizer.lr=5e-3 training.ema.decay=0.999

# Architecture
pixi run -e cuda goal-train model.backbone.num_rounds=3 model.backbone.share_artisan_weights=true
pixi run -e cuda goal-train 'model.backbone.artisan.hidden_irreps="32x0e + 16x1o + 8x2e"'

# Loss weights (list index into training.losses)
pixi run -e cuda goal-train training.losses.0.weight=1.0 training.losses.1.weight=10.0

# Cutoff: model.backbone.cutoff follows data.cutoff automatically
pixi run -e cuda goal-train data.cutoff=6.0

# Your own data (directory per split)
pixi run -e cuda goal-train data.dataset_type=xyz \
    data.train_dir=/data/mysys/train data.val_dir=/data/mysys/val data.test_dir=/data/mysys/test

# Hardware
pixi run -e cuda goal-train trainer.accelerator=cpu
pixi run -e cuda goal-train trainer.devices=4 strategy.name=ddp
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
| `~key` | Delete a key (for example `~logger.wandb`, `~model.backbone.fragment_interaction`). |

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
pixi run -e cuda goal-train -m model.backbone.num_rounds=2,3 seed=42,43,44   # 6 jobs
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
logs/simurgh_gmd26/runs/2026-08-07_09-54-59_trajectory_simurgh_arace_fragment_duplication/
├── simurgh_gmd26.log                 # Hydra job log (${task_name}.log)
├── .hydra/
│   ├── config.yaml                   # composed config (unresolved)
│   ├── hydra.yaml
│   └── overrides.yaml                # your command-line overrides
├── csv_logs/version_0/metrics.csv    # CSVLogger (always on)
├── wandb/                            # W&B local files (if enabled)
└── checkpoints/                      # GOALCheckpointManager dirpath
    ├── config.yaml                   # fully RESOLVED config, incl. injected pair_counts
    ├── metadata.json                 # GOAL version, git commit, elements, scale, …
    ├── frozen_source/                # snapshot of every goal .py file the model imports
    ├── checkpoint_state.json         # pool membership, used on resume
    ├── best_val_forces_mae=0.0100_epoch=1124.ckpt   # top-k pool (+ .json sidecar each)
    ├── interval_epoch=1124.ckpt      # interval pool (+ .json sidecar)
    ├── last.ckpt / last.json         # crash recovery / auto-resume
    ├── best_val_forces_mae.simurgh   # self-contained archive of the best checkpoint
    └── TRAINING_COMPLETE             # sentinel: stops a SLURM requeue from retraining
```

`task_name` groups runs of one experiment, and the timestamped `run_name` keeps each launch separate. Override either one:

```bash
pixi run -e cuda goal-train task_name=my_study run_name=baseline_v1
```

The same `run_name` is used as the W&B run name, so the directory and the dashboard always agree. The checkpoint layout and the `.simurgh` archive are covered in [Checkpoints & archives](../deployment/archives.md), and resuming in [Resuming & extending](../training/resuming.md).

## Sub-directory configs and `_global_`

Hydra places a config that lives in a **sub-directory** of the search path under a key named after that directory. A file `src/goal/configs/ml/ablations/05_gmd26_body_order.yaml` is treated as belonging to the `ablations` group, so without a directive its whole content lands under `cfg.ablations.*` and the top-level `model`, `data` and `trainer` keys are never set.

This is why every file in `src/goal/configs/ml/ablations/`, `src/goal/configs/ml/finetune/` and `src/goal/configs/debug/` begins with:

```yaml
# @package _global_
```

For the ablation rungs, a missing header fails silently: the rung's own override is filed under `ablations:`, the shared `_ablation_base` still composes, and **the rung quietly trains the base model**. When you add a config in any sub-directory:

1. Make `# @package _global_` the first line (comments above it are fine; the ablation files keep their explanatory comments below it).
2. Check the composition before you spend GPU hours on it:

   ```bash
   pixi run -e cuda goal-train --config-name ablations/my_new_rung --cfg job -p model.backbone
   ```

Top-level files in `src/goal/configs/ml/` (for example `simurgh_gmd26.yaml`) are in the root package and do not need the header.

The opposite problem affects the `src/goal/configs/hparams_search/*.yaml` presets. They have no header *and* already wrap their content in an `hparams_search:` key, so `+hparams_search=ray_tune` produces `cfg.hparams_search.hparams_search.method`, and `goal-tune` falls back to the default `method: tuner`. Append them with `+hparams_search@_global_=ray_tune` instead. See [CLI → goal-tune](cli.md#goal-tune).

## Writing your own experiment config

Copy the closest existing file rather than starting from scratch. Every file is complete and documents its alternatives inline.

```bash
cp src/goal/configs/ml/simurgh_gmd26.yaml src/goal/configs/ml/my_system.yaml
# edit: experiment.*, task_name, data.*, atomic_energies, …
pixi run -e cuda goal-train --config-name my_system --cfg job -p data   # check
pixi run -e cuda goal-train --config-name my_system
```

Things to change every time:

- `experiment.name` / `description` / `tags` and `task_name`, so runs do not mix in `logs/` or on W&B;
- the `data` loading mode and paths (four modes, documented at the top of the `data:` block; see [Loading data](../data/loading.md));
- `data.energy_key` / `forces_key` if your files use different labels;
- `model.backbone.num_elements` (must exceed the largest Z) and `atomic_energies` (references for your elements and level of theory, or `mode: dataset`);
- `checkpoint_manager.top_k.metric`, if forces are not your selection criterion.
