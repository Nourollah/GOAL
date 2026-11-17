---
name: goal-ablation-study
description: Plans, runs and reports ablation studies of GOAL models so that every comparison is a one-key config diff. Covers base configurations, the noise floor from seeds 42, 43 and 44, seed-42 screening, gates that must pass before accuracy counts, selection rules in units of seed standard deviation, matched-parameter controls, never selecting on OOD data, run naming, and booktabs or Markdown result tables. Use when the user asks to ablate or compare model variants, decide whether a component helps, or write an ablation table.
license: MIT
---

# Ablation studies in GOAL

The SIMURGH programme itself: `goal-simurgh`.

## Design

- **One idea per arm.** Every arm differs from its parent by one config key, or by one ablation file that overrides one section of the model. The experiment is the diff.
- **Defaults are the base.** Every optional capability defaults to off or to the base setting, and unknown keys raise, so a typo cannot train the base under an arm's name.
- **Where things go.** Configs in `src/goal/configs/ml/ablations/<model>/`. Mechanism files: one per mechanism, named after the mechanism rather than a paper, registered as `<model>_abl_<mechanism>`.
- **Controls.** When an arm adds parameters, add a matched-parameter control (a wider base, or an equal-parameter MLP in place of the new component). Report the parameter count and the time per step for every arm.
- **Stages.** Settle the optimiser first, because it changes every later comparison. Each stage starts from the best configuration of the previous one.

## Noise floor first

Train the base with seeds 42, 43 and 44 before any arm. Without its s.d., no difference means anything.

## Protocol

1. Screen each arm with seed 42.
2. Run seeds 43 and 44 only for arms whose seed-42 change exceeds 2× the base s.d.
3. Gates before accuracy counts: the invariant battery (`goal-mlip-invariant-tests`, including cutoff crossing, smoothness and tail errors) and the MD-stability screen (`goal-md-stability-benchmark`).
4. Arm-specific gates: depth, attention and long-range arms pass the MD22 per-system gate (force MAE improves on at least 4 of 7 systems and none worsens beyond seed noise); arms aimed at extrapolation pass the 3BPA 1200 K and dihedral gate over 3 seeds.

## Selection rules

- **Additive arm:** keep it if OOD force MAE improves by more than 2 s.d. on at least 2 of the 4 GMD tasks while ID force MAE worsens by no more than 1 s.d. For arms aimed at the standard benchmarks, keep it if ID force MAE improves by more than 2 s.d. on the target benchmark and its gates pass. Otherwise drop it.
- **Removal of an existing component:** drop the component only if the removal arm is not worse on any metric beyond 1 s.d. and is better on at least one beyond 1 s.d.
- Never select on OOD data during training, and never on GMD Duplication total energy.

## Running and naming

- Run names `<stage>_<arm>_s<seed>`; group runs by stage in the logger.
- `-m seed=42,43,44` for replicates; `CUDA_VISIBLE_DEVICES` to run arms side by side.
- Keep one results CSV per stage: run name, config diff, parameter count, time per step and every metric.

## Reporting

- LaTeX (papers): booktabs, one table, every metric (energy MAE, force MAE, force cosine similarity, force-magnitude MAE) split into ID and OOD sub-columns, mean ± s.d., best in bold, second best underlined.
- Markdown (notes): models as rows, best in bold, second best in italics, no "change from base" column.
- List the known confounds under the table: unmatched parameter counts, one learning rate for every arm, arms that move two things at once, a single dataset.

## Done when

- [ ] The base s.d. exists for every metric used in a decision.
- [ ] Every kept or dropped arm cites the rule and the numbers that decided it.
