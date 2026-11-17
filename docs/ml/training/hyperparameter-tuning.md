# Hyperparameter tuning

This page covers the ways to search hyperparameters in GOAL: what to search for SIMURGH, the `goal-tune` entry point and its three back-ends (Lightning Tuner, Ray Tune, W&B Sweeps), and approaches that drive the full `goal-train` pipeline (Hydra multirun and W&B sweeps over `goal-train`). It also states which paths currently work end to end and which have known gaps.

## Which approach to use

| Approach | Extra dependencies | Runs the full `goal-train` pipeline | Pruning | Best for |
| --- | --- | --- | --- | --- |
| [Hydra multirun](#hydra-multirun-grids-and-seeds) | none | yes | no | small grids, seed repeats |
| [W&B sweep over `goal-train`](#wb-sweeps-over-goal-train-recommended) | `wandb` (core dependency) | yes | Hyperband | real searches, many GPUs or nodes |
| [`goal-tune`, method `ray`](#ray-tune-goal-tune-method-ray) | `ray[tune]`, `optuna` | no (see limitations) | ASHA / PBT | (see limitations) |
| [`goal-tune`, method `wandb`](#wb-sweeps-goal-tune-method-wandb) | `wandb` | no (see limitations) | Hyperband | (see limitations) |
| [`goal-tune`, method `tuner`](#lightning-tuner-goal-tune-method-tuner) | none | no | — | LR range test |

"Full pipeline" means the dataset-derived quantities (elements, atomic reference energies, scale, `avg_num_neighbors`), the `trainer:` block from your config (including `inference_mode: false`), stage curricula, checkpointing and loggers. `goal-tune` builds its models with a simplified path that skips most of this. The details are under each method.

> [!IMPORTANT]
> Whatever tool you use, **optimise a validation metric, never a `test…/` one.** With `data.monitor_test_during_training: true`, the OOD set is logged every epoch. Tuning against it turns it into a second validation set. Use `val_forces/forces_mae_atomw` (what the SIMURGH configs select on), `val_energy/energy_mae_per_atom` or a combination of the two.

## What to search

For SIMURGH, the dimensions that matter most (all keys exist in [`simurgh_dev/_base.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_dev/_base.yaml)):

| Key | Suggested range | What it controls |
| --- | --- | --- |
| `model.backbone.num_interactions` | `{1, 2, 3}` | Message-passing layers, which set the receptive field (≈ layers × cutoff) and cost |
| `model.backbone.correlation` | `{2, 3, 4}` | Body order of the product basis |
| `model.backbone.num_channels` with `model.backbone.message_irreps` | 64 or 128 channels | Equivariant width; the `message_irreps` multiplicity must equal `num_channels` |
| `model.backbone.num_rbf` | `{8, 16, 32}` | Radial basis size |
| `training.optimizer.lr` | log-uniform 1e-4 … 1e-2 | The most important single knob |
| `training.optimizer.weight_decay` | log-uniform 1e-8 … 1e-4 | Decoupled; not applied to embeddings or pair-line coefficients |

For one-key comparisons with a fixed protocol, use the ablation arms in `src/goal/configs/ml/ablations/simurgh_dev/` instead of a search.

Some other keys worth considering are the loss weights (`training.losses.1.weight` for forces relative to energy), `data.batch_size`, `training.ema.decay`, `model.backbone.cutoff` (kept in sync with `data.cutoff` through interpolation), and `training.optimizer.warmup_epochs`.

> [!TIP]
> **Search in stages.** First tune `lr` (and the force/energy weight ratio) on the default architecture with short runs. Then fix them and search the architecture. Finally confirm the top few configurations with full-length runs and several seeds. Architecture rankings at 100 epochs usually, but not always, carry over to 1500, which is why the last step matters.

## Hydra multirun: grids and seeds

This needs no extra packages and uses the exact production pipeline:

```bash
pixi run -e cuda goal-train -m \
    training.optimizer.lr=2e-3,5e-3,1e-2 \
    model.backbone.num_interactions=1,2 \
    trainer.max_epochs=100 \
    run_name_suffix=_grid
```

Hydra runs the Cartesian product (6 jobs) one after another. Each job gets its own directory, `logs/<task_name>/multiruns/<run_name>/<job_num>/`, with checkpoints, CSV logs and a W&B run. Compare them in W&B (group with `logger.wandb.group=<name>`) or from the CSVs. For seed repeats, add `seed=1,2,3`. To run jobs in parallel on a cluster, use one SLURM array task per configuration instead (see [Training a model](training.md#many-runs-on-a-cluster)).

## W&B sweeps over `goal-train` (recommended)

A W&B sweep agent can launch `goal-train` itself, passing the sampled values as Hydra overrides. Each trial is then an ordinary GOAL run, with dataset statistics, `inference_mode: false`, the stage curriculum, the checkpoint manager and the W&B logger all working, while the W&B controller handles Bayesian search and Hyperband early termination.

`sweep_simurgh.yaml`:

```yaml
program: goal-train
method: bayes
metric:
  name: val_forces/forces_mae_atomw # validation metric keys have no _epoch suffix
  goal: minimize
early_terminate:
  type: hyperband
  min_iter: 10 # number of metric reports (validation epochs) before a run can be stopped
  eta: 3
parameters:
  training.optimizer.lr:
    distribution: log_uniform_values
    min: 1.0e-4
    max: 1.0e-2
  training.optimizer.weight_decay:
    distribution: log_uniform_values
    min: 1.0e-8
    max: 1.0e-4
  model.backbone.num_interactions:
    values: [1, 2, 3]
  model.backbone.correlation:
    values: [2, 3, 4]
command:
  - ${env}
  - goal-train
  - ${args_no_hyphens} # -> key=value, exactly Hydra's override syntax
  - trainer.max_epochs=100
  - run_name_suffix=_sweep
  - data.monitor_test_during_training=false # no need to pay for the OOD pass while searching
```

```bash
wandb sweep --project goal sweep_simurgh.yaml      # prints: wandb agent <entity>/goal/<sweep_id>
wandb agent <entity>/goal/<sweep_id>               # run on as many GPUs / nodes as you like
```

Notes:

- **Metric name.** W&B sees what was *logged*. Validation **metrics** are logged as plain keys (`val_forces/forces_mae`, `val_energy/energy_mae_per_atom`), but validation **losses** are suffixed, so the loss total is `val/total_epoch`, not `val/total` (see [Callbacks & logging](callbacks-logging.md#_step-and-_epoch-suffixes)).
- **One agent per GPU.** Start each agent with its own `CUDA_VISIBLE_DEVICES`. On SLURM, submit an array where each task runs `wandb agent --count 1 <sweep>` (or more trials per task).
- **Project.** Pass the same project to `wandb sweep` as the config's `logger.wandb.project`, or override it in the command list.
- Hyperband stops a run early by killing the process. The checkpoints written up to that point stay on disk.

## `goal-tune`

`goal-tune` ([`src/goal/ml/cli/tune.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/cli/tune.py) and [`src/goal/ml/training/tuning.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/training/tuning.py)) dispatches on `hparams_search.method`: `tuner`, `ray` or `wandb`.

### Invoking it

`goal-tune` reads experiment files from `src/goal/configs/ml/` (like `goal-train`), and its default experiment is `simurgh_dev_gmd26_FD`, which has no `hparams_search:` section. The presets in `src/goal/configs/hparams_search/` live one level up, so put that directory on the search path to use them (outside a clone, replace `src/goal/configs` with `"$(python -c 'import goal; print(goal.CONFIGS_DIR)')"`):

```bash
# Any experiment file + one of the presets in src/goal/configs/hparams_search/
pixi run -e cuda goal-tune --config-dir src/goal/configs --config-name simurgh_dev_gmd26_FD '+hparams_search@_global_=basic'
pixi run -e cuda goal-tune --config-dir src/goal/configs --config-name simurgh_dev_gmd26_FD '+hparams_search@_global_=ray_tune'
pixi run -e cuda goal-tune --config-dir src/goal/configs --config-name simurgh_dev_gmd26_FD '+hparams_search@_global_=wandb_sweep'

# Or inline
pixi run -e cuda goal-tune --config-name simurgh_dev_gmd26_FD '+hparams_search={method: tuner, tuner: {lr_find: true}}'
```

`@_global_` is needed because the preset files start with a top-level `hparams_search:` key and have no `# @package` header. (Before October 2026 the default config was a non-existent `train.yaml`, so a bare `goal-tune` failed with *Cannot find primary config 'train'*.)

### How `goal-tune` builds a trial

Unlike `goal-train`, each trial is built directly from the config:

- `backbone_cls(**cfg.model.backbone)` with **no dataset-derived injection**. `elements`, `avg_num_neighbors`, a numeric `scale` (SIMURGH's `scale: force_rms` is resolved by `goal-train` only) and `atomic_energies` in `dataset` mode must all be set in the config. [Benchmark datasets](../data/benchmarks.md#switching-molecules) shows how to measure them.
- The head and the losses are built with the same helpers as `goal-train` (`goal.ml.cli.builders`): monolithic backbones (`head: null`) are supported, and each loss receives exactly the keyword arguments it accepts (`normalize_by_n_atoms`, `property_name`, …).
- No callbacks, loggers or checkpoint manager are attached (except the W&B logger in `wandb` mode).

### Lightning Tuner (`goal-tune`, method `tuner`)

```yaml
hparams_search:
  method: tuner
  tuner:
    lr_find: true
    min_lr: 1.0e-8
    max_lr: 1.0
    num_training_steps: 100
    lr_find_mode: exponential # or linear
    auto_apply: true
    scale_batch_size: false
    batch_size_mode: power # or binsearch
    steps_per_trial: 3
    init_batch_size: 2
    max_trials: 25
```

> [!WARNING]
> As shipped, this method does not complete. Lightning's `Tuner.lr_find` looks for an `lr` or `learning_rate` attribute on the module, and `scale_batch_size` looks for a `batch_size` attribute on the module or datamodule. `GOALModule` and `GOALDataModule` expose neither (the LR lives in `config.training.optimizer.lr`, the batch size in `cfg.data.batch_size`), so both raise `AttributeError`. `auto_apply` would also only set `module.hparams.lr`, which `configure_optimizers` never reads.

To run an LR range test, call Lightning's tuner on a module you build yourself. Building components from a config is shown in [Custom training loops](custom-loops.md#building-goal-components-from-a-config):

```python
import lightning as L
from lightning.pytorch.tuner import Tuner

from goal.ml.training.module import GOALModule

# cfg, dm, backbone, head, loss built as in custom-loops.md
module = GOALModule(backbone=backbone, head=head, loss=loss, config=cfg)
module.lr = float(cfg.training.optimizer.lr)  # attribute Lightning's lr_find looks for

trainer = L.Trainer(accelerator="gpu", devices=1, inference_mode=False, logger=False,
                    enable_checkpointing=False, max_epochs=1)
finder = Tuner(trainer).lr_find(module, datamodule=dm, min_lr=1e-6, max_lr=1e-1, num_training=200)
print("suggested lr:", finder.suggestion())
finder.plot(suggest=True).savefig("lr_find.png")
```

`lr_find` replaces the scheduler with its own exponential ramp, so the result does not depend on `scheduler_type`. Use the suggestion as `training.optimizer.lr`, and treat it as an upper bound: the steepest-descent point is usually somewhat too aggressive for the long cosine runs typical of MLIPs.

### Ray Tune (`goal-tune`, method `ray`)

```bash
# ray[tune] + optuna are in every pixi environment (pip: pip install "goal-atomistic[tune]")
pixi run -e cuda goal-tune --config-dir src/goal/configs --config-name simurgh_dev_gmd26_FD '+hparams_search@_global_=ray_tune'
```

The `hparams_search` schema:

| Key | Meaning |
| --- | --- |
| `num_samples` | number of trials |
| `max_epochs` | epochs per trial (also ASHA's `max_t`) |
| `metric`, `mode` | target and direction (default `val/total`, `min`) |
| `scheduler` | `asha` (`grace_period`, `reduction_factor`) or `pbt` (`perturbation_interval`) |
| `search_algorithm` | `null` (random), `optuna`, `hyperopt` |
| `search_space` | dotted config keys mapped to `{type: uniform/loguniform/quniform/choice/randint/grid, …}` |
| `experiment_name`, `storage_path` | Ray results location (default `<checkpoint_dir>/../ray_tune`) |

A search-space entry looks like this:

```yaml
search_space:
  training.optimizer.lr: { type: loguniform, lower: 1.0e-4, upper: 1.0e-2 }
  model.backbone.num_interactions: { type: choice, values: [1, 2, 3] }
  training.losses.1.weight: { type: uniform, lower: 1.0, upper: 20.0 }
```

> [!WARNING]
> Known limitations of the current Ray trial function (`_ray_train_fn`), from reading the code:
>
> - Each trial creates its own `L.Trainer(max_epochs=…, accelerator="auto", devices="auto", logger=False, …)` instead of using your `trainer:` block. Lightning's default `inference_mode=True` therefore applies, and validation of autograd-force models (every SIMURGH config) cannot compute forces.
> - Metrics are reported to Ray **once, after `trainer.fit` ends**, and only keys starting with `val/`. ASHA and PBT therefore have no intermediate results to prune on, and a target such as `val_forces/forces_mae_atomw` is never reported. Only `val/total` is.
> - Sampled values are merged into a resolved copy of the config. Interpolations such as `model.backbone.cutoff: ${data.cutoff}` keep the **base** value.
>
> Until these are fixed, prefer a W&B sweep over `goal-train`, or Hydra multirun.

### W&B Sweeps (`goal-tune`, method `wandb`)

```yaml
hparams_search:
  method: wandb
  project: goal
  entity: null
  sweep_id: null # set to attach more agents to an existing sweep
  sweep_method: bayes # bayes | grid | random
  metric: val/total
  mode: min
  count: 20 # trials this agent runs
  early_terminate: { type: hyperband, min_iter: 10, eta: 3 }
  parameters:
    training.optimizer.lr: { distribution: log_uniform_values, min: 1.0e-5, max: 1.0e-2 }
```

`goal-tune` creates the sweep (or attaches to `sweep_id`) and runs `wandb.agent` in-process. `parameters` uses W&B's own sweep schema.

> [!WARNING]
> The in-process W&B trial has similar gaps to the Ray one: a bare `L.Trainer` with `inference_mode=True`, and `max_epochs` taken from `training.max_epochs` (default 500) rather than `trainer.max_epochs`. In addition, the shipped preset's `metric: val/total` is never logged under that name, because the loss total appears in W&B as `val/total_epoch`. Use `val_forces/forces_mae` as the metric. The [W&B sweep over `goal-train`](#wb-sweeps-over-goal-train-recommended) has none of these problems.

## After the search

1. Take the best few configurations and train them at **full length** with **3+ seeds** (`seed=1,2,3`). The search used short budgets.
2. Report the in-distribution validation metric that was tuned on, and the OOD (`test…/`) numbers that were not.
3. Write the chosen values into a new experiment file in `src/goal/configs/ml/` (for example `simurgh_dev_gmd26_FD_tuned.yaml`, starting with `defaults: [simurgh_dev_gmd26_FD, _self_]`), so the final model is reproducible from a single `--config-name`.
