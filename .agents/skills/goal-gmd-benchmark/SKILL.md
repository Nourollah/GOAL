---
name: goal-gmd-benchmark
description: Runs and reports the GMD compositional-generalisation benchmark (arXiv:2605.08988) in GOAL. Covers the data layout, the Chain Extension, Fusion, Duplication and Combination tasks and the auxiliary Fission split, TestID and TestOOD evaluation, isolated-atom element references and why least-squares references fail on GMD, centred and offset energy errors, per-element force errors, the reference-handling probe, published targets and the rules that keep OOD numbers honest. Use when training, evaluating or comparing models on GMD, or interpreting GMD energy errors.
license: MIT
---

# GMD benchmark

Paper: GMD, arXiv:2605.08988 (NeurIPS 2026 Datasets and Benchmarks). Labels: PBE-D3(BJ)/def2-TZVP from ORCA (keyword line `!EnGrad PBE D3BJ def2-TZVP RI TightSCF`), two-body D3 only (no `ABC` keyword). Older GOAL config comments said def2-SVP (the tracked configs and docs were corrected); the paper's keyword line is authoritative. Detailed evidence for the energy-reference rules: `references/energy_references.md`.

## Data and configs

- Data: `dataset_type: gmd` downloads GMD from the Hugging Face Hub (https://huggingface.co/datasets/AmirMasoud/GMD) on first use and caches the graphs under `data.root`; never copy GMD files into the repository. Options (task, format, seed, sizes, `val_split`/`test_split`, `train_molecules`/`extra_train_molecules`, revision, `local_root`): `docs/ml/data/benchmarks.md`.
- Tasks and config suffixes: FragmentChainExtension (`FCE`), FragmentFusion (`FF`), FragmentDuplication (`FD`), FragmentCombination (`FCB`), plus the auxiliary FragmentFission (`FFI`), which reuses the Duplication models. The molecule lists are `goal.ml.data.benchmarks.gmd.TASKS` (Table 1 of the paper).
- Duplication trains on the C5–C10 monocarboxylic acids and tests OOD on the C5–C10 dicarboxylic acids (the published protocol, which is the reference). Add trajectories with `data.extra_train_molecules` / `data.extra_test_molecules`.
- The paper protocol (`seed: null`): frames [0, 1000) train, [1400, 1600) TestID of the training molecules, [1400, 1600) TestOOD of the unseen ones. GOAL validates, stops early and selects checkpoints on TestID only; TestOOD is the test set.
- `simurgh_dev_gmd26_FD_plus_alkane` (split `src/goal/configs/ml/splits/gmd26_FD_plus_alkane.yaml`): the Duplication training set plus one alkane trajectory (octane) from the Chain Extension OOD split. It makes every diacid OOD energy identifiable, so comparing it with plain Duplication separates missing information from genuine compositional failure. Models trained on it are never evaluated on Chain Extension.

## Element references

Use the isolated-atom energies that the GOAL GMD configs already contain:

```yaml
atomic_energies: {mode: provided, values: {1: -13.595298, 6: -1028.481145, 7: -1483.895483, 8: -2041.117735}}   # eV
```

Never fit references by least squares on GMD (`mode: dataset`, or a foundation model's E0 re-estimation). Monoacid compositions (n, 2n, 2) have rank 2, so least squares pins only E0(C) + 2 E0(H) = −1055.67 eV, and the Duplication OOD energies are not identifiable at any level. Duplication OOD total-energy MAE therefore measures reference handling, not learning: never select or rank models on it, and always report it next to the centred and offset errors.

## Metrics to report

- Report dataset-level numbers from `goal-eval` with `predictions_file=...`. The `forces_mae` logged during training is a batch average weighted by molecules, not atoms: an earlier SIMURGH model on Duplication OOD logged 0.050 eV/Å against a dataset-level 0.057 eV/Å. GOAL also logs the exact atom-weighted `forces_mae_atomw`, `forces_rmse_atomw` and `forces_cosine_similarity_atomw`; the SIMURGH configs select checkpoints on `val_forces/forces_mae_atomw`.
- Headline: ID and OOD force MAE (meV/Å), with force RMSE, Q95, Q99 and max of the per-atom error norm, cosine similarity, and per-element force MAE (O-resolved errors are the GMD failure signature).
- Energy per molecule (as in the paper) and per atom. For each OOD molecule m with residuals r_mj = E_pred − E_ref over its frames j:
  - offset δ_m = mean_j r_mj, and **offset MAE** = mean_m |δ_m| (`energy_offset_mae`, and `energy_offset_mae_per_atom`);
  - **centred MAE** = mean over m and j of |r_mj − δ_m| (`energy_centred_mae`, `energy_centred_mae_per_atom`);
  - mean signed error per molecule against chain length.
- Seeds 42, 43 and 44 as mean ± s.d.; from-scratch and fine-tuned models in separate blocks.

## Reference-handling probe

Run it on every checkpoint before comparing energies (`scripts/simurgh_dev/probe_reference.py`):

1. Predict isolated H, C, N and O atoms. A model with isolated-atom references returns the values above.
2. A least-squares-referenced model returns roughly C −211.1, H −422.3 and O −2041.1 eV (H below C). An unreferenced model with per-atom mean scaling returns about the same value for every element.
3. Regress OOD energy residuals on (n_C, n_H, n_O). Large coefficients, linear in composition, mean a referencing error rather than a learning failure.

## Published targets

GMD arXiv:2605.08988v1 Table 7 (Table 6 in later versions):

| Task | Best ID force MAE (meV/Å) | Best ID energy MAE (meV per molecule) | Best from-scratch OOD force MAE (meV/Å) |
|---|---|---|---|
| Chain Extension | 1.5 (EquiformerV2) | 1.6 | 4.0 (EquiformerV2) |
| Fusion | 4.7 | 3.6 | 88.9 |
| Duplication | 4.9 | 2.9 | 39.0 |
| Combination | 5.3 | 2.9 | 13.3 |

Unnamed entries are MACE trained from scratch (B-REF in SIMURGH's ablation programme). Fine-tuned foundation models do better OOD on two tasks: 57.4 meV/Å on Fusion (UMA-Small-110) and 32.0 meV/Å on Duplication. Re-check the current version of the paper before quoting these.

## Reporting format

- LaTeX: booktabs, one table, every metric split into ID and OOD sub-columns, best in bold, second best underlined.
- Markdown: models as rows, best in bold, second best in italics.

## Done when

- [ ] The probe passes for every compared checkpoint.
- [ ] Every energy says per molecule or per atom, and centred and offset MAE accompany every Duplication energy.
- [ ] OOD was evaluated once, at the end, on the checkpoint selected on TestID.
