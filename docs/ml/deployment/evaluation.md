# Evaluation

Once a model is trained you want numbers on data it has never seen: per-atom energy error, force error, how well force directions are reproduced, and whether the forces sum to zero. This page covers what GOAL measures, where the test split comes from, the `goal-eval` command, and the equivalent Python route for notebooks.

Both routes rebuild the model from the checkpoint itself, the same way the [ASE calculator](ase-calculator.md) does: from the run's frozen source, with the saved config and the EMA weights.

## What gets measured

Every evaluation path (validation during training, the test-set monitor, `goal-eval` and the Python route below) calls the same function, `goal.ml.training.metrics.mlip_metrics`. It compares the model's predictions with the reference labels on each batch:

| Metric | Logged as | Definition | Units |
| --- | --- | --- | --- |
| `energy_mae_per_atom` | `test_energy/energy_mae_per_atom` | $\langle \lvert E^{\text{pred}}_m - E^{\text{ref}}_m \rvert / N_m \rangle_m$ | eV/atom |
| `energy_rmse_per_atom` | `test_energy/energy_rmse_per_atom` | $\sqrt{\langle ((E^{\text{pred}}_m - E^{\text{ref}}_m)/N_m)^2 \rangle_m}$ | eV/atom |
| `forces_mae` | `test_forces/forces_mae` | mean absolute error over every Cartesian force component | eV/Å |
| `forces_rmse` | `test_forces/forces_rmse` | RMS error over every Cartesian force component | eV/Å |
| `forces_cosine_similarity` | `test_forces/forces_cosine_similarity` | mean per-atom $\cos\angle(\mathbf F^{\text{pred}}_i, \mathbf F^{\text{ref}}_i)$, in $[-1, 1]$ | — |
| `forces_magnitude_mae` | `test_forces/forces_magnitude_mae` | $\langle \lvert\, \lVert\mathbf F^{\text{pred}}_i\rVert - \lVert\mathbf F^{\text{ref}}_i\rVert \,\rvert \rangle_i$ | eV/Å |
| `newton_violation_{x,y,z}` | `test_forces/newton_violation_x` … | $\langle \lvert \sum_{i \in m} F^{\text{pred}}_{i,\alpha} \rvert \rangle_m$ per axis | eV/Å |
| `newton_violation` | `test_forces/newton_violation` | sum of the three axes | eV/Å |

How to read them:

- **Per-atom energy** errors make structures of different sizes comparable. A per-structure error would be dominated by the largest molecules.
- **Force MAE and RMSE** are the headline numbers in most MLIP papers. A large RMSE/MAE ratio means the error is concentrated in a few outlier atoms, which are often strained geometries or bond-breaking regions.
- **Cosine similarity and magnitude MAE** split the force error into a direction part and a size part. A model can have a good MAE but a poor cosine on weak forces near equilibrium. That matters for geometry optimisation.
- **Newton violation** checks translational invariance: the total force on an isolated structure should be zero. Conservative (autograd) force heads give about $10^{-15}$ (floating-point noise). Direct force heads give finite values, and this metric shows how far off they are.

Metrics are only computed when the labels exist. A dataset without forces (for example QM9) gives only the energy rows.

The loss terms are also logged under the same sections (`test/total`, `test_energy/energy`, `test_forces/forces`, …). Unlike validation, `test_step` uses the **base** loss weights and ignores any `training.stages` curriculum, so test losses are not comparable to validation losses from a staged run. The metrics above do not depend on the weights, so they are comparable.

> [!NOTE]
> Sectioned names (`test_energy/…`, `test_forces/…`) come from the current code. A checkpoint loaded from its **frozen source** (see [Checkpoints & archives](archives.md)) logs with the naming its snapshot used. Older snapshots put everything under `test/` (`test/forces_mae`, …).

### EMA weights

When `training.ema.enabled: true`, `GOALModule.test_step` evaluates inside `ema.average_parameters()`, so you get the exponential-moving-average weights. Validation and checkpoint selection use the same weights. Checkpoints written by the current code store the EMA shadow under the `goal_ema` key, and the loader puts it into the parameters. A checkpoint without a usable shadow is evaluated on its raw weights; the module's EMA is re-seeded from them, so `test_step` never averages in a randomly initialised shadow (it did before October 2026).

## Where the test set comes from

The test split is whatever the `data:` block of the config defines. The four loading modes are described in [Loading data](../data/loading.md). In short:

| Mode | Config keys | Test split |
| --- | --- | --- |
| Single root, auto-split | `data.root`, `data.split_ratio: [0.8, 0.1, 0.1]` | the third fraction (seeded by `data.split_seed`) |
| Explicit file lists | `data.train_paths`, `data.val_paths`, `data.test_paths` | `test_paths` |
| Merged sources, auto-split | `data.root: [/path/A, /path/B]`, `split_ratio`, `merge_strategy` | the third fraction |
| Directory per split | `data.train_dir`, `data.val_dir`, `data.test_dir` | `test_dir` |

The GMD-26 configs use directory mode with `val_dir: …/TestID` (in-distribution) and `test_dir: …/TestOOD` (out-of-distribution). The gap between the two is the generalisation measurement those experiments exist for.

### Watching the test set during training

With `data.monitor_test_during_training: true` (the default in `simurgh_gmd26.yaml`) the test split runs through the **validation** loop as a second dataloader every validation epoch. You get `test/`, `test_energy/` and `test_forces/` curves next to the `val_*` ones without running a separate evaluation. Two rules apply:

- Select checkpoints and early-stop on `val_*`, never on `test_*`. If you select on the test set it becomes a second validation set, and its out-of-distribution number no longer means anything.
- Those curves use the stage-weighted loss (they come from `validation_step`). Again, only the metrics are comparable with a standalone test run.

## Evaluate from Python

`goal-eval` is a thin wrapper around these steps; use them directly in a notebook:

```python
import lightning as L
from omegaconf import open_dict

from goal.ml.data.datamodule import GOALDataModule
from goal.ml.utils.calculator import load_goal_checkpoint

CKPT_DIR = "logs/simurgh_gmd26/runs/<run>/checkpoints"   # or a .simurgh archive or a .ckpt

# 1. Rebuild the GOALModule from the frozen source + saved config, with the EMA weights.
loaded = load_goal_checkpoint(CKPT_DIR, checkpoint="best")   # "best" | "last" | "epoch=N" | "<file>.ckpt"
print("evaluating with", loaded.weights, "weights")         # "ema" or "raw"
module = loaded.module

# 2. Optionally point the test split somewhere else.
cfg = loaded.config
with open_dict(cfg):
    cfg.data.test_dir = "/path/to/another/test/set"   # or test_paths / root, per your data mode

# 3. Run Lightning's test loop.  inference_mode=False is mandatory:
#    autograd force heads need the gradient tape during evaluation.
trainer = L.Trainer(accelerator="auto", devices=1, logger=False, inference_mode=False)
results = trainer.test(module, datamodule=GOALDataModule(cfg))
print(results[0]["test_forces/forces_mae"])
```

`trainer.test` prints a table and returns a list with one dict per dataloader, for example:

```text
{'test_energy/energy_mae_per_atom': 9.19e-05, 'test_energy/energy_rmse_per_atom': 1.23e-04,
 'test_forces/forces_mae': 0.00544, 'test_forces/forces_rmse': 0.01092,
 'test_forces/forces_cosine_similarity': 0.99993, 'test_forces/forces_magnitude_mae': 0.00604,
 'test_forces/newton_violation': 2.3e-15, ..., 'test/total_epoch': 0.0}
```

> [!NOTE]
> The loader rebuilds the loss with the frozen source. If that fails (for example, a loss keyword the snapshot does not know), it falls back to an empty `CompositeLoss`, so `test/total` reads `0.0`. The metrics are unaffected. Only the loss rows are meaningless in that case.

Notes and variations:

- **Accelerator.** Pass `accelerator="gpu"` or `"cpu"` explicitly to override auto-detection. Add `limit_test_batches=…` for a quick smoke test.
- **Precision.** The module is built in the dtype its weights were saved in. Use `precision="64-true"` for a float64 run if you also want Lightning to cast the batches.
- **Logging to disk.** Swap `logger=False` for `lightning.pytorch.loggers.CSVLogger("eval_logs")` to get a `metrics.csv`.
- **Fragment-aware models.** Models that need `fragment_index` (fragment interaction, SIMURGH Simplified's fragment line) work here because `GOALDataModule` attaches fragment labels from the saved `data.compute_fragment_index` settings.

### Per-structure errors and parity plots

The Lightning test loop averages over batches. For parity plots or a per-structure error table, run the calculator over the structures directly:

```python
import numpy as np
from ase.io import read

from goal.ml.utils.calculator import GOALCalculator

calc = GOALCalculator(checkpoint_path="model.simurgh", device="cuda")
frames = read("test.extxyz", index=":")

e_ref, e_pred, f_err = [], [], []
for atoms in frames:
    ref_e = atoms.get_potential_energy()       # from the file's SinglePointCalculator
    ref_f = atoms.get_forces()
    atoms.calc = calc
    e_ref.append(ref_e / len(atoms))
    e_pred.append(atoms.get_potential_energy() / len(atoms))
    f_err.append(np.abs(atoms.get_forces() - ref_f).mean())

print(f"energy MAE {np.mean(np.abs(np.subtract(e_pred, e_ref))) * 1000:.2f} meV/atom")
print(f"forces MAE {np.mean(f_err) * 1000:.1f} meV/Å")
```

The calculator uses the same weights as the test loop (EMA when the checkpoint has them; `calc.weights` says which) and attaches fragment labels for fragment-aware models.

## The `goal-eval` command

`goal-eval` (`goal.ml.cli.evaluate:evaluate`, also `pixi run -e cuda eval`) is a Hydra entry point. It:

1. rebuilds the model from `ckpt_path` with `load_goal_checkpoint` — any of the three checkpoint formats, frozen source, saved config, EMA weights (`use_ema`);
2. merges the command line's `data:` section onto the checkpoint's own `data:` section, so by default it tests on exactly the split the run used;
3. builds a `Trainer` from `trainer:` (`inference_mode: false`; `precision: null` takes the checkpoint's `trainer.precision`);
4. calls `trainer.test(module, datamodule=…)` and, with `output_file=…`, writes the metrics to JSON.

### Minimal invocation

```bash
pixi run -e cuda goal-eval ckpt_path=logs/simurgh_gmd26/runs/<run>/checkpoints
```

The default config is [`src/goal/configs/ml/eval.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/eval.yaml):

| Key | Default | Meaning |
|---|---|---|
| `ckpt_path` | `???` (required) | `.simurgh` archive, checkpoint directory or `.ckpt` |
| `checkpoint` | `best` | selector inside a directory: `best`, `last`, `epoch=N`, `<name>.ckpt` |
| `use_ema` | `true` | evaluate the EMA weights when the checkpoint has them |
| `data` | `{}` | merged onto the checkpoint's `data:`; add keys with `+` |
| `trainer.accelerator` / `devices` / `precision` | `auto` / `1` / `null` | test-loop hardware; `precision: null` = the checkpoint's |
| `output_file` | `null` | JSON file for the metrics (`checkpoint`, `weights`, `metrics`) |
| `hydra.run.dir` | `logs/eval/<date>_<time>` | Hydra's run directory |

### More invocations

```bash
CKPT_DIR=logs/simurgh_gmd26/runs/<run>/checkpoints

# last instead of best, on the GPU, metrics to JSON
pixi run -e cuda goal-eval ckpt_path=$CKPT_DIR checkpoint=last trainer.accelerator=gpu output_file=eval/last.json

# another test set (the key set of `data` is open, hence the +)
pixi run -e cuda goal-eval ckpt_path=$CKPT_DIR '+data.test_paths=[data/ood_test.xyz]'

# a single .ckpt file (quote it: checkpoint names contain '=')
pixi run -e cuda goal-eval "ckpt_path='$CKPT_DIR/best_val_forces_forces_mae=0.0052_epoch=1440.ckpt'"

# an experiment's whole data/trainer/logger sections instead of eval.yaml
pixi run -e cuda goal-eval --config-name simurgh_gmd26 ckpt_path=$CKPT_DIR
```

With an experiment config, its `trainer:` block (with `_target_`) and any `logger:` entries are used as they are; the model still comes from the checkpoint, so the experiment config only has to agree on the data. Strategy and devices come from the `trainer` block, so `trainer.devices=4 trainer.strategy=ddp` runs a distributed evaluation, as during training.

> [!NOTE]
> **Fixed in October 2026.** `goal-eval` used to name a primary config (`eval`) that did not exist, forward loss keywords the loss classes reject (`normalize_by_n_atoms`), rebuild the model from the *live* source and the command-line config (missing the dataset-derived `elements` / `scale` / atomic references, so some models did not build), and evaluate a random EMA shadow for checkpoints without `goal_ema`. All five are gone: the model now comes from the checkpoint as described above.

- Quote a `.ckpt` path as shown: checkpoint filenames contain `=`, which Hydra's override parser would otherwise reject.
- Any other override works as usual, for example `trainer.accelerator=cpu` or `+data.batch_size=4`.

## See also

- [ASE calculator](ase-calculator.md): single points, optimisation and MD with a trained model.
- [Checkpoints & archives](archives.md): what is in a checkpoint directory and how `best`, `last` and `epoch=N` are resolved.
- [Callbacks & logging](../training/callbacks-logging.md): logger sections and metric naming during training.
- [CLI reference](../reference/cli.md): every `goal-*` entry point.
