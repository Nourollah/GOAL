---
name: goal-standard-benchmarks
description: Runs and reports the standard molecular MLIP benchmarks in GOAL with their published protocols. Covers rMD17 (1000 structures as 950 training plus 50 validation, or 50 in the low-data setting; MAE), original MD17, MD22 (per-system training sizes; MAE) and 3BPA (500 structures at 300 K; RMSE at 300, 600 and 1200 K and on dihedral slices), with unit conversions, the gates used in GOAL ablations and a dated table of best published values. Use when training on, evaluating or comparing results for rMD17, MD17, MD22 or 3BPA.
license: MIT
---

# Standard benchmarks: rMD17, MD17, MD22 and 3BPA

References: `docs/ml/data/benchmarks.md`. Best published values: `references/published_targets.csv` (compiled on 2026-10-30; re-check the literature before claiming a state of the art).

## Loaders and configs

- `rmd17`, `md17` and `md22` are registered dataset adapters (`DATASET_REGISTRY`). They download on first use and cache processed `.pt` files.
- Switch molecules with `data.molecule=...` (for example `ethanol` or `AT-AT`); the original MD17 with `data.revised=false` (both `rmd17` and `md17` default to `revised: true`). MD22 identifiers: `Ac-Ala3-NHMe`, `DHA`, `stachyose`, `AT-AT`, `AT-AT-CG-CG`, `buckyball-catcher`, `double-walled_nanotube`.
- Training-set size: `data.train_size` (rMD17 default 950 with `val_size` 50; low-data setting `data.train_size=50 data.val_size=50`). The md22 adapter's default `train_size` per system matches the MD22 row below, with `val_size` 500; `data.test_size` caps the test set.
- One config per benchmark task in `src/goal/configs/ml/` (`simurgh_dev_rmd17_<molecule>.yaml`, `simurgh_dev_md22_<system>.yaml`, `simurgh_dev_3bpa.yaml`); list them with `ls src/goal/configs/ml/`.
- MD22 systems differ in elements and neighbour density. Where a config hard-codes `elements` or `avg_num_neighbors`, recompute both per system (`compute_unique_elements` and `compute_avg_num_neighbors` in `goal.ml.data.statistics`). The `simurgh_dev` configs leave both null and `goal-train` derives them from the training split.
- 3BPA has no built-in adapter. `src/goal/configs/ml/simurgh_dev_3bpa.yaml` reads the standard extxyz files from `data/3BPA/` with `dataset_type: xyz` and file lists (train `train_300K.xyz`, validation `test_300K.xyz`, test `test_1200K.xyz`); evaluate the other test sets with `goal-eval ... '+data.test_paths=[...]'`.

## Protocols

| Benchmark | Training data | Test | Papers report |
|---|---|---|---|
| rMD17 | 1000 structures per molecule (950 training + 50 validation in GOAL) | the remaining structures | MAE: energy in meV per molecule, forces in meV/Å |
| rMD17, low data | 50 structures per molecule | the remaining structures | MAE |
| MD17 (original) | as rMD17 | as rMD17 | noisier labels; never mix with rMD17 numbers |
| MD22 | Ac-Ala3-NHMe 6000, DHA 8000, stachyose 8000, AT-AT 3000, AT-AT-CG-CG 2000, buckyball catcher 600, double-walled nanotube 800 | the remaining structures | MAE, energy per molecule and forces |
| 3BPA | 500 structures sampled at 300 K | 300 K, 600 K and 1200 K sets, dihedral slices | RMSE |

## Levels of theory and element references

| Benchmark | Labels | Isolated-atom energies |
|---|---|---|
| rMD17 | PBE/def2-SVP, ORCA, tight SCF and dense grid | not in the standard release |
| MD22 | PBE+MBD, FHI-aims | not in the standard release |
| 3BPA | ωB97X/6-31G(d), ORCA | H, C, N and O atoms ship with the dataset |

Each model sees one composition, so element energies only shift a constant per molecule and leave the metrics unchanged. Use the shipped isolated atoms for 3BPA (`simurgh_dev_3bpa.yaml` sets `atomic_energies.mode: provided` with them) and the mean-energy reference (`atomic_energies.mode: dataset`, as in the `simurgh_dev` rMD17 and MD22 configs) for rMD17 and MD22, so that the network does not have to learn a large constant. Never reuse GMD's element energies here: they are PBE-D3(BJ)/def2-TZVP. The 3BPA test sets hold 1669 (300 K), 2138 (600 K) and 2139 (1200 K) structures; check the loaded sizes against these.

## Reporting

- Convert GOAL's per-atom energy error to per molecule by multiplying by the atom count; this is exact for a single-molecule dataset (aspirin: 21 atoms).
- 3BPA results are RMSE and rMD17 and MD22 results are MAE; compare like with like.
- Report mean ± s.d. over seeds 42, 43 and 44, with models as rows.

## Gates used in GOAL ablations

- MD22 per-system gate: a change counts only if force MAE improves on at least 4 of the 7 systems and none worsens beyond seed noise.
- 3BPA extrapolation gate: report 1200 K and dihedral-slice errors over 3 seeds for any change aimed at extrapolation.
- MD stability at the MD17 effective temperatures (`goal-md-stability-benchmark`).

## Published targets

`references/published_targets.csv` has the columns system, value, unit, model, source and table: per molecule or system, the best value found in a literature survey on 2026-10-30, with its arXiv number and table. New methods appear constantly, so treat the file as a starting point.

## Done when

- [ ] Every number states its metric (MAE or RMSE), unit and training-set size.
- [ ] Per-atom values are converted before they are compared with published tables.
