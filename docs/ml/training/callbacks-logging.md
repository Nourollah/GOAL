# Callbacks & logging

This page lists the callbacks a `goal-train` run uses (the ones always added in code, the ones in the config, and optional ones you can enable), how experiment loggers are configured, how logged metric names are grouped into sections, and how to monitor an out-of-distribution set while training without contaminating model selection.

## Where callbacks come from

A run's callbacks come from three places:

| Source | Callbacks | How to change them |
| --- | --- | --- |
| Hard-wired in `goal.ml.cli.train` | `LearningRateMonitor(logging_interval="epoch")`, `RichLoggingCallback`, plus `LRScheduleRetarget` on resume | Not configurable. Write your own entry script to change them (see [Custom training loops](custom-loops.md)). |
| Top-level `checkpoint_manager:` block | `GOALCheckpointManager` | Edit or override `checkpoint_manager.*`, or disable it with `checkpoint_manager=null`. |
| `callbacks:` block | Every entry with a `_target_` is instantiated through Hydra. The entry names (`early_stopping`, `swa`, …) are only labels. | Edit the YAML, override `callbacks.<name>.<key>=…`, add with `+callbacks.<name>=…`, remove with `~callbacks.<name>`. |

### Always active

- **`LearningRateMonitor`** logs the learning rate of each parameter group once per epoch, as `lr-AdamW/pg1` (equivariant parameters, no weight decay) and `lr-AdamW/pg2` (everything else). Use these curves to confirm warmup, cosine decay and [extension LR policies](resuming.md#the-learning-rate-schedule-needs-a-policy).
- **`RichLoggingCallback`** prints a Rich table of every entry in `trainer.callback_metrics` at the end of each validation epoch, on rank 0 only. This gives you a full readout in the SLURM `.out` file without opening W&B.
- **`LRScheduleRetarget`** is added only when resuming, and re-aims the restored LR schedule (see [Resuming & extending](resuming.md)).

### `GOALCheckpointManager`

This manager keeps three checkpoint pools (top-k by a validation metric, a rolling interval pool, and `last.ckpt`). It freezes the model source at training start, writes JSON sidecars, and packs the best checkpoint into a `.simurgh` archive at the end. Its behaviour, configuration and resume semantics are covered in [Resuming & extending](resuming.md#checkpoint-manager-behaviour). Common overrides:

```bash
pixi run -e cuda goal-train checkpoint_manager.top_k.k=3
pixi run -e cuda goal-train checkpoint_manager.top_k.metric=val_energy/energy_mae_per_atom
pixi run -e cuda goal-train checkpoint_manager.top_k.metric=val_forces/forces_cosine_similarity checkpoint_manager.top_k.mode=max
pixi run -e cuda goal-train checkpoint_manager.interval.every_n_epochs=10 checkpoint_manager.interval.keep_last_n=3
```

> [!IMPORTANT]
> `GOALCheckpointManager` and Lightning's `ModelCheckpoint` cannot be active together. They would save twice and fight over deletions, so `goal-train` raises a `ValueError` if it finds both. Use the manager for SIMURGH-style experiments. If you need `ModelCheckpoint` (or GOAL's `GOALCheckpoint` subclass, which also writes `TRAINING_COMPLETE`), set `checkpoint_manager=null` and add the callback under `callbacks:`. For multi-GPU runs, see the note in [Distributed training](distributed.md#checkpointing-with-more-than-one-gpu).

### Configured in the shipped experiment files

```yaml
callbacks:
  rich_progress_bar:
    _target_: goal.ml.training.callbacks.progress.GOALRichProgressBar

  model_summary:
    _target_: lightning.pytorch.callbacks.RichModelSummary
    max_depth: 1 # -1 = whole tree

  early_stopping:
    _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: "val_forces/forces_mae"
    patience: 50
    mode: "min"
    min_delta: 1.0e-4
    strict: true # crash if the metric is missing (catches typos)
    check_finite: true # stop on NaN/Inf
    stopping_threshold: null
    divergence_threshold: null
```

- **`GOALRichProgressBar`** is Lightning's `RichProgressBar` with a stage-aware description. While a `training.stages` curriculum is active, the bar reads `[energy_forces 12/∞] Epoch 17/1499` instead of only `Epoch 17/1499`. Lightning allows only one progress bar, so this one is configured here rather than in code. Remove it (`~callbacks.rich_progress_bar`) to get Lightning's default TQDM bar, which is easier to read in plain-text SLURM logs.
- **`RichModelSummary`** prints the module tree and parameter counts once at startup.
- **`EarlyStopping`** stops when `val_forces/forces_mae` has not improved by `min_delta` for `patience` *validation* epochs. Two interactions to keep in mind:
    - With `trainer.check_val_every_n_epoch: k`, patience counts validation runs, which is $k\times$ more epochs.
    - `trainer.min_epochs` (10 in the shipped configs) prevents stopping before that epoch.

    Its state is saved in checkpoints. An extension of an early-stopped run therefore stops again quickly unless you raise `patience` (see [Resuming & extending](resuming.md#what-carries-over-and-what-to-watch)).

### Optional callbacks (commented out in the configs)

| Name in config | Callback | When to use |
| --- | --- | --- |
| `backbone_finetuning` | `BackboneFinetuning` | Transfer learning: keep the backbone frozen for `unfreeze_backbone_at_epoch` epochs, then train it at a reduced LR. Needs a module exposing `.backbone`, which `GOALModule` does. |
| `ema_weight_averaging` | `EMAWeightAveraging` | Lightning's EMA. Set `training.ema.enabled=false` first (see [Performance engineering](performance.md#exponential-moving-average-ema)). |
| `grad_accumulation` | `GradientAccumulationScheduler` | Increase gradient accumulation over epochs, e.g. `{0: 1, 10: 2, 20: 4}`. |
| `swa` | `StochasticWeightAveraging` | Weight averaging over the end of training. Do not combine with EMA. |
| `throughput_monitor` | `ThroughputMonitor` | Samples per second. Needs a `batch_size_fn` (see [Performance engineering](performance.md#profiling)). |

Enable one from the command line without editing the file:

```bash
pixi run -e cuda goal-train '+callbacks.swa={_target_: lightning.pytorch.callbacks.StochasticWeightAveraging, swa_lrs: 1.0e-4, swa_epoch_start: 0.75}' training.ema.enabled=false
```

Other GOAL callbacks:

- **`FoundationE0Callback`** (`goal.ml.training.callbacks.foundation`) re-estimates a foundation backbone's atomic reference energies on your training set at `on_fit_start`. It is used by fine-tuning configs (see [Fine-tuning](../finetuning/overview.md)).
- **`GOALCheckpoint`** (`goal.ml.training.callbacks.checkpoint`) is a `ModelCheckpoint` that writes the `TRAINING_COMPLETE` sentinel on a normal finish.

### Writing your own callback

Any Lightning `Callback` works. Make it importable and reference it with `_target_`:

```python
# my_project/callbacks.py
import torch
from lightning import Callback


class GradNormMonitor(Callback):
    """Log the global gradient norm before every optimiser step."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        norms = [p.grad.detach().norm() for p in pl_module.parameters() if p.grad is not None]
        if norms:
            pl_module.log("train_gate/grad_norm", torch.stack(norms).norm(), on_step=True, on_epoch=False)
```

```bash
pixi run -e cuda goal-train '+callbacks.grad_norm={_target_: my_project.callbacks.GradNormMonitor}'
```

`goal-train` imports callbacks by dotted path, so `my_project` has to be on `PYTHONPATH` (or installed). Because `on_before_optimizer_step` runs before GOAL's clipping, this logs the **unclipped** norm, which is the number you need for choosing `training.gradient_clip`.

## Loggers

Loggers are declared in the `logger:` block. Every entry with a `_target_` is instantiated, and all of them receive the same metrics. The shipped configs enable CSV and Weights & Biases:

```yaml
logger:
  csv:
    _target_: lightning.pytorch.loggers.csv_logs.CSVLogger
    save_dir: "${paths.output_dir}"
    name: "csv_logs"
    prefix: ""

  wandb:
    _target_: lightning.pytorch.loggers.wandb.WandbLogger
    name: ${run_name} # same string as the output directory
    project: "goal"
    entity: null
    save_dir: "${paths.output_dir}"
    offline: false
    id: null # set to continue an existing W&B run
    log_model: false # true uploads checkpoints to W&B
    group: ""
    tags: ["simurgh", "gmd"]
```

These are also in the files, commented out:

| Logger | `_target_` | Requirements / viewing |
| --- | --- | --- |
| TensorBoard | `lightning.pytorch.loggers.tensorboard.TensorBoardLogger` | `tensorboard --logdir logs/` |
| MLflow | `lightning.pytorch.loggers.mlflow.MLFlowLogger` | a tracking server (`tracking_uri`), `mlflow ui` |
| Neptune | `lightning.pytorch.loggers.neptune.NeptuneLogger` | `NEPTUNE_API_TOKEN` |
| Aim | `aim.pytorch_lightning.AimLogger` | `pip install aim`, then `aim up` |
| Comet | `lightning.pytorch.loggers.comet.CometLogger` | `COMET_API_TOKEN` |

Common adjustments:

```bash
# No network on compute nodes: log locally and sync later with `wandb sync`
pixi run -e cuda goal-train logger.wandb.offline=true          # or: WANDB_MODE=offline goal-train

# Drop W&B entirely, keep CSV
pixi run -e cuda goal-train '~logger.wandb'

# Add TensorBoard alongside
pixi run -e cuda goal-train '+logger.tensorboard={_target_: lightning.pytorch.loggers.tensorboard.TensorBoardLogger, save_dir: ${paths.output_dir}/tensorboard, name: tb}'

# Group seeds of one experiment in W&B
pixi run -e cuda goal-train logger.wandb.group=arace_lr5e-3 seed=1
```

If `logger:` is `null` or empty (as in the `debug` presets), Lightning falls back to its default logger.

The CSV logger writes `csv_logs/version_0/metrics.csv` in the run directory. It is the easiest file to load into pandas for paper plots:

```python
import pandas as pd

df = pd.read_csv("logs/simurgh_gmd26/runs/<run>/csv_logs/version_0/metrics.csv")
val = df.dropna(subset=["val_forces/forces_mae"]).set_index("epoch")
val[["val_forces/forces_mae", "val_energy/energy_mae_per_atom"]].plot(logy=True)
```

### Logger sections

W&B and TensorBoard group panels by the text before the first `/`. GOAL chooses that prefix deliberately, so each group answers one question. The rule is defined once, in `metric_section()` in [`training/metrics.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/training/metrics.py), and is used for both losses and metrics:

| Section | Contents |
| --- | --- |
| `{split}/` | `total`, the loss actually optimised, plus any loss component that is neither energy nor forces (`stress`, `dipole`, `scalar_property`, custom losses). |
| `{split}_energy/` | The energy loss term (and composite sub-terms like `energy_huber`), `energy_mae_per_atom`, `energy_rmse_per_atom`, and `energy_round_{L}`: ARACE's per-round energy contribution. A round that collapses to zero has stopped contributing. |
| `{split}_forces/` | The forces loss term (and `forces_mae`/`forces_rmse`… sub-terms), `forces_mae`, `forces_rmse`, `forces_cosine_similarity`, `forces_magnitude_mae`, and `newton_violation` with its `_x`, `_y`, `_z` components. |
| `{split}_gate/` | All gate diagnostics, which are numerous and kept off the progress bar: `gate_mean_round_{L}` (ARACE adaptive depth gate), `aux_loss` (its sparsity penalty), `gate_reg`, `artisan/round{L}/{pair}` (per element-pair artisan gates, $K(K+1)/2 \times$ rounds curves), and on the legacy ACE-first backbone only, `artisan_load/{pair}` and `artisan_load_variance` (validation only). |

`{split}` is `train`, `val` or `test`. The reason for this split: per-atom energy errors and force cosines are on different scales and answer different questions, and the gate curves can number in the dozens. Kept apart, they do not bury `val_forces/forces_mae` in a wall of panels.

**Reading the gate curves.** An `artisan/round{L}/{pair}` gate drifting to ~0 means that element pair has been switched off, the artisan equivalent of expert collapse. For `gate_mean_round_{L}`, ~1.0 means the gate is always open (not selective: raise `aux_loss_weight`), ~0.5 means it discriminates between atoms, and a fall towards 0 means whole rounds are being skipped (lower `aux_loss_weight`).

### `_step` and `_epoch` suffixes

Lightning adds suffixes when a value is logged both per step and per epoch:

| Logged by | Keys in W&B / CSV |
| --- | --- |
| Training losses and metrics (`on_step` and `on_epoch`) | `train/total_step`, `train/total_epoch`, `train_forces/forces_mae_step`, `train_forces/forces_mae_epoch`, … |
| Validation/test **losses** (`on_step` and `on_epoch`) | `val/total_epoch`, `val/total_step`, `val_forces/forces_epoch`, … |
| Validation/test **metrics** (`on_epoch` only) | `val_forces/forces_mae`, `val_energy/energy_mae_per_atom`, … (no suffix) |

Inside the run (`trainer.callback_metrics`, used by `EarlyStopping`, `GOALCheckpointManager` and `ReduceLROnPlateau`), the epoch value is also available **without** a suffix. That is why `monitor: val_forces/forces_mae` and the scheduler's `val/total` both work. External tools see only what was logged, so in a W&B sweep or dashboard the loss total is called `val/total_epoch`.

## Watching an OOD set during training

GMD-26 ships an in-distribution test set (`TestID`) and an out-of-distribution one (`TestOOD`). The default config uses them as:

```yaml
data:
  train_dir: ${paths.data_dir}/GMD/FragmentDuplication/Training
  val_dir: ${paths.data_dir}/GMD/FragmentDuplication/TestID # select on this
  test_dir: ${paths.data_dir}/GMD/FragmentDuplication/TestOOD # only watch this
  monitor_test_during_training: true
```

With `monitor_test_during_training: true`, `GOALDataModule.val_dataloader()` returns `[val, test]`, and `GOALModule` logs the second loader under its own sections: `test/`, `test_energy/` and `test_forces/`. Every validation epoch then gives in-distribution and OOD curves side by side, and they are **never averaged together**. The generalisation gap becomes a curve you can watch, instead of a single number at the end.

Details:

- The OOD set is evaluated with the same EMA weights and the same active stage weights as `val`, so `test/total` and `val/total` are directly comparable epoch by epoch.
- The OOD loader stays off the progress bar and skips the diagnostics that would only duplicate `val` (artisan gate values) or that cost an extra forward pass (artisan load balance).
- The cost is one extra forward pass (no backward) over `test_dir` per validation epoch.
- If no test split is configured, the flag only triggers a warning at setup, and only `val` is reported.
- A standalone `trainer.test()` run (for example `goal-eval`) also logs under `test…/` but uses the **base** loss weights, so its loss values are comparable only within the same stage. The metrics (MAE, RMSE, cosine, Newton violation) are weight-independent and always comparable.

> [!WARNING]
> **Select on `val_*`, read `test_*`.** Never point `checkpoint_manager.top_k.metric`, `callbacks.early_stopping.monitor` or a sweep's target metric at a `test…/` curve. Choosing checkpoints or hyperparameters by OOD performance turns the OOD set into a second validation set, and the OOD number stops measuring generalisation.

## Related pages

- [Loss functions](losses.md): what each loss term and metric measures.
- [Resuming & extending](resuming.md): checkpoint pools in detail.
- [Hyperparameter tuning](hyperparameter-tuning.md): picking the metric a sweep optimises.
