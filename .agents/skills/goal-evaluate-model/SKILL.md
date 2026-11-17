---
name: goal-evaluate-model
description: Evaluates trained GOAL checkpoints and archives. Runs goal-eval on the saved test split or another test set, loads models with GOALCalculator or load_goal_checkpoint, packs and inspects .simurgh archives, and reports energy and force errors with explicit units (per atom or per molecule, MAE or RMSE, tail errors, per-element forces) and fair-comparison rules. Use when the user asks for test metrics, compares checkpoints or models, exports a model, or the numbers look wrong.
license: MIT
---

# Evaluating a GOAL model

References: `docs/ml/deployment/evaluation.md`, `docs/ml/deployment/archives.md`, `docs/ml/deployment/ase-calculator.md`.

## Commands

```bash
CKPT_DIR=logs/<task>/runs/<run>/checkpoints
pixi run -e cuda goal-eval ckpt_path=$CKPT_DIR                                       # best checkpoint, saved test split
pixi run -e cuda goal-eval ckpt_path=$CKPT_DIR checkpoint=last trainer.accelerator=gpu output_file=eval/last.json
pixi run -e cuda goal-eval ckpt_path=$CKPT_DIR '+data.test_paths=[data/ood_test.xyz]'  # another test set
pixi run -e cuda goal-eval "ckpt_path='$CKPT_DIR/<file name with = signs>.ckpt'"     # a single file: quote it
pixi run -e cuda goal-inspect $CKPT_DIR                                              # checkpoints and metadata
pixi run -e cuda goal-pack-archive --checkpoint-dir $CKPT_DIR --checkpoint best --output models/my_model.simurgh
```

Python:

```python
from goal.ml.utils.calculator import GOALCalculator, load_goal_checkpoint

loaded = load_goal_checkpoint(CKPT_DIR, checkpoint="best")   # "best" | "last" | "epoch=N" | "<file>.ckpt"
print(loaded.weights)                                         # "ema" or "raw"; also .module, .cutoff (Å), .config
calc = GOALCalculator(checkpoint_path="models/my_model.simurgh", device="cuda")   # ASE calculator
```

A Lightning test loop needs `inference_mode=False`, because forces come from autograd.

## What the logged numbers mean

| Metric | Meaning | Unit |
|---|---|---|
| `energy_{mae,mse,rmse}_per_atom` | per-structure energy error divided by the atom count; exact epoch values (RMSE = √ of the dataset MSE) | eV/atom (MSE eV²/atom²) |
| `energy_{mae,mse,rmse}_per_molecule` | total-energy error of each structure | eV (MSE eV²) |
| `forces_mae`, `forces_rmse` | over every Cartesian force component | eV/Å |
| `forces_cosine_similarity`, `forces_magnitude_mae` | direction and size parts of the force error | none, eV/Å |
| `newton_violation` | net force on an isolated structure; floating-point noise for conservative models | eV/Å |

- The step metrics `forces_mae`, `forces_rmse` and `forces_cosine_similarity` (logger and CSV) are batch averages weighted by the number of structures, not atoms, so they under-report the dataset MAE when molecule sizes vary (an earlier SIMURGH model, GMD Duplication OOD: 0.050 eV/Å logged, 0.057 eV/Å at dataset level). `GOALModule` also logs exact atom-weighted epoch values `forces_mae_atomw`, `forces_rmse_atomw` and `forces_cosine_similarity_atomw` (eV/Å, cosine unitless) in the `{split}_forces/` section next to them; the `simurgh_dev` configs select checkpoints on `val_forces/forces_mae_atomw`. Report dataset-level numbers from `goal-eval ... predictions_file=...` or the `*_atomw` metrics.
- Papers usually report total energy per molecule (meV) on fixed-composition benchmarks. Convert a per-atom error exactly by multiplying by the atom count (aspirin has 21 atoms).
- rMD17 and MD22 tables report MAE; 3BPA tables report RMSE (`goal-standard-benchmarks`).
- A large RMSE/MAE ratio means a few atoms carry the error. Report the tail (Q95, Q99 and max of the per-atom force-error norm) and per-element force MAE next to every force MAE. The implemented `forces_q95`, `forces_q99` and `forces_max` are taken over the per-component absolute error instead, so compute the per-atom-norm tail from the predictions file and say which one you report.
- Logged per batch (`{split}_forces/`): per-element force MAE `forces_mae_<element>` (e.g. `forces_mae_O`, eV/Å) and `newton_violation_x`, `_y`, `_z` next to their sum `newton_violation`.
- `goal-eval ... predictions_file=preds.npz` writes per-structure predictions (keys `energy`, `energy_ref`, `num_atoms`, `atomic_numbers`, `forces`, `forces_ref`) and prints dataset-level metrics, also stored under `evaluation` in `output_file`. Energies, in eV and with a `_per_atom` variant in eV/atom: `energy_mae`, `energy_rmse`, `energy_centred_mae`, `energy_offset_mae`, plus `per_molecule` mean signed errors. Forces, in eV/Å: `forces_mae`, `forces_rmse`, `forces_q95`, `forces_q99`, `forces_max`, `forces_magnitude_mae`, `forces_cosine_similarity` (unitless) and `forces_mae_<element>`. If a metric you need is missing, compute it from these predictions; never estimate it.
- Benchmarks with OOD energy offsets need the centred and offset energy errors (`energy_centred_mae`, `energy_offset_mae`; `goal-gmd-benchmark`).

## Comparing models fairly

- Same splits, same element-reference handling, same units, same checkpoint rule (best validation, never OOD).
- Report mean ± s.d. over seeds 42, 43 and 44, or say that only one seed was run.
- With EMA enabled, evaluation and checkpoint selection use the EMA weights; `load_goal_checkpoint` reports which weights it loaded.
- Report the parameter count and the time per training step next to the errors.
- A checkpoint loaded from its frozen source logs the metric names of that snapshot (older ones use `test/forces_mae`).

## Pitfalls

- Checkpoint file names contain `=`; quote them for Hydra.
- A model rebuilt from a checkpoint takes dataset-derived values (elements, element energies, scale, neighbour count) from the saved config, not from the new test set.
- Models that need fragment labels, and checkpoints trained with `atomic_energies.mode: dataset`, have calculator caveats: read "Limitations" in `docs/ml/deployment/ase-calculator.md`.

## Done when

- [ ] Every number states its unit, its split (ID, OOD, temperature) and its checkpoint (best or final).
- [ ] Energy errors say per atom or per molecule.
- [ ] Seeds and their s.d. are reported.
