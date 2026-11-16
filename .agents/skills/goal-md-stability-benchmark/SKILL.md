---
name: goal-md-stability-benchmark
description: Measures molecular-dynamics stability and energy conservation of GOAL interatomic potentials with a fixed protocol. Covers MD17 effective temperatures (552.6 K aspirin, 642.9 K ethanol), batched Langevin screening of 32 replicas for 100 ps, 300 ps finalist runs over 3 seeds and two checkpoints, bond-deviation instability detection at 0.5 Å, NVE drift on Ac-Ala3-NHMe and on GMD OOD molecules, and wall-time estimates. Use when comparing models for simulation robustness or checking that accuracy gains survive MD.
license: MIT
---

# MD stability benchmark

Accuracy on static test sets does not guarantee stable dynamics. Run this protocol for every configuration that passes the accuracy screen.

## Temperatures

The original MD17 thermostat counted 3N degrees of freedom instead of 3N − 6, so the trajectories are hotter than their nominal 500 K (arXiv:2606.31660). Simulate at T_eff = 500 K × 3N / (3N − 6):

| Molecule | N | T_eff |
|---|---|---|
| aspirin | 21 | 552.6 K |
| ethanol | 9 | 642.9 K |

Use the same formula for any other MD17 molecule. That paper's three estimators give 553–581 K for aspirin and 643–667 K for ethanol; this protocol uses the degrees-of-freedom correction above.

## Protocol

1. **Screening.** 32 test frames each of aspirin (553 K) and ethanol (643 K), one batched run per molecule because `goal-simulate-batch` takes a single `--temperature` (nvalchemi backend where it is installed; see `docs/integrations/batched_simulation.md`): Langevin, 0.5 fs, friction 0.01 fs⁻¹, 100 ps (aspirin: `--temperature 552.6 --timestep 0.5 --friction 0.01 --steps 200000`; `--timestep` is in fs and `--friction` in fs⁻¹).
2. **Finalists.** 300 ps, 3 seeds, for both the best-validation and the final checkpoint.
3. **NVE drift.** Ac-Ala3-NHMe (MD22), 100 ps at 0.5 fs, started from the same equilibrated frame for every model.
4. **GMD.** 100 ps NVE on one OOD molecule per task.

## Instability criterion

Bonded pairs come from the first frame (covalent radii × 1.2). A replica becomes unstable the first time any bonded distance deviates by more than 0.5 Å from its starting value, or any non-bonded pair comes closer than 0.7 Å.

Tooling:
- `BondDeviationObserver` in `goal.md.core.observers` (not exported from `goal.md`); defaults `bond_factor=1.2`, `max_bond_deviation=0.5` Å, `min_nonbonded_distance=0.7` Å.
- `goal-simulate-batch --mode md` writes `stability.json` in `--output-dir`: per structure `unstable_step` and `onset_ps` (null while stable) and `mean_temperature_K`, checked every `--log-every` steps (default 100) with no extra force calls.
- NVE through the `velocity_verlet` dynamics key, registered in `DynamicsFactory` (`goal-run-md`).
- The summary script `scripts/simurgh_dev/md_stability.py`: `--batch <output dirs>` gives survival fraction, onset median and IQR in ps (failed replicas) and mean temperature; `--nve-log <ASE MD logs>` gives the drift slope in meV/atom/ps from a per-atom log (`Time[ps] Etot/N[eV] ...`); `--md17-temperature N` prints T_eff for N atoms.

## Report

- survival fraction, and onset time per replica as median and interquartile range;
- mean temperature, checked against the target;
- NVE drift slope in meV/atom/ps;
- precision (float32 or float64), checkpoint (best or final) and seed.

## Cost

100 ps at 0.5 fs is 200,000 steps, so wall time is 200,000 × the time per step. Measure the time per step once per GPU and model, and report it. For a 21-atom molecule the step is dominated by kernel-launch overhead, so 32 replicas in one batch should cost about as much as one; verify this on the hardware before planning runs.

## Pitfalls

- `Simulation` without a friction argument, and the `goal-simulate` default (`langevin_ase` template), use about 0.05 fs⁻¹ (0.5 in ASE units) at 1 fs; this protocol uses 0.01 fs⁻¹ at 0.5 fs (the `goal-simulate-batch` defaults).
- Do not run MD17 molecules at the nominal 500 K.
- Start NVE runs from equilibrated frames with velocities, the same frame for every model.
- An accurate but unstable model fails the gate, whatever its MAE.

## Done when

- [ ] Every finalist has survival fractions and onset statistics for both checkpoints and 3 seeds.
- [ ] NVE drift on Ac-Ala3-NHMe is reported for every finalist.
