---
name: goal-mlip-invariant-tests
description: Writes and runs the physical invariant test battery for any GOAL interatomic potential, in float64 on the CPU. Covers far-copy extensivity, isolated-atom and separated-pair energies, rotation, translation, reflection and permutation symmetry, finite-difference forces, net force, smooth cutoff crossing, bond-stretch smoothness, zero-initialised optional branches, disabled-flag identity, the static DDP graph and NVE energy drift. Use when adding or changing a model or a config flag, before trusting benchmark numbers, or when MD is unstable.
license: MIT
---

# Invariant tests for interatomic potentials

Run every test in float64 on the CPU (the model's `dtype: float64`, and `torch.set_default_dtype(torch.float64)` inside a fixture that restores the previous default afterwards; a global change leaks float64 into every later test), in eval mode, for every flag combination that an ablation can switch on. Reusable, model-agnostic helpers live in `tests/ml/simurgh_dev/utils.py`; call them for any registered model.

| # | Test | Pass condition (float64) | Catches |
|---|---|---|---|
| 1 | Far-copy extensivity: E(A ∪ B, with B 100 Å away) − E(A) − E(B), for B = A and B ≠ A | abs < 1e-5 eV | pooling, normalisation, virtual nodes or attention spanning molecules |
| 2 | Isolated atom | E = E0(Z) exactly, F = 0 | wrong element references, biases on empty neighbourhoods |
| 3 | Two atoms farther apart than r_c | E = E0(Z1) + E0(Z2) exactly | leakage beyond the cutoff |
| 4 | Random rotation and translation | E unchanged, F rotated with the structure (max abs difference < 1e-8) | non-equivariant layers, layout bugs |
| 5 | Reflection | E unchanged | parity errors |
| 6 | Permutation of the atom order | E unchanged, F permuted | scatter and gather index bugs |
| 7 | Finite-difference forces (central differences, h = 1e-4 Å) | abs difference < 1e-6 eV/Å | detached geometry, non-conservative terms |
| 8 | Net force on an isolated structure | abs < 1e-10 eV/Å | direct-force paths, broken autograd |
| 9 | Cutoff crossing: move one atom radially through r_c, 1000 points across it | energy step at r_c < 1e-9 eV, forces continuous | envelopes applied in the wrong place, hard cutoffs |
| 10 | Bond-stretch scans (C–H, C–C, C–O, O–H, N–H; 0.6–4 Å, 200 points) | no jumps in E or dE/dr; report force-sign flips, extra minima and inflections | rough radial bases, unphysical short range |
| 11 | A zero-initialised optional branch switched on | a fresh model's energy is unchanged | initialisations that move the base model |
| 12 | An optional feature with `enabled: false` | outputs bit-identical to the model without it | flags that leak into the base path |
| 13 | Static DDP graph: 2 CPU processes (gloo), `find_unused_parameters=False` | one optimiser step succeeds, including a batch where a branch has nothing to gather | parameters that miss the loss |

Where each test lives for SIMURGH (`tests/ml/simurgh_dev/`; the helpers are in `utils.py`):

| # | Helper or test |
|---|---|
| 1 | `far_copy_extensivity` |
| 2 | `isolated_atom` |
| 3 | `test_monolithic.py::test_separated_pair_is_sum_of_e0` |
| 4, 6 | `symmetry_errors` (rotation, translation, permutation) |
| 5 | no test yet (planned) |
| 7 | `finite_difference_force_error` (central differences, step 1e-4 Å) |
| 8 | no test yet (planned); the `newton_violation` metric reports it during training |
| 9 | `test_monolithic.py::test_cutoff_crossing_is_smooth` |
| 10 | `bond_scan`, `scan_jumps` |
| 11, 12 | `test_monolithic.py::test_zero_init_equivalence` with `load_shared`; `test_d3_disabled_is_bit_identical` |
| 13 | `test_mfc.py::test_ddp_static_graph_two_processes` |

```python
import torch
from ase.build import molecule

torch.set_default_dtype(torch.float64)

def test_far_copy_extensivity(energy):          # energy(atoms) -> float, wraps the model once
    a, b = molecule("CH3CH2OH"), molecule("H2O")
    b.translate([100.0, 0.0, 0.0])
    assert abs(energy(a + b) - energy(a) - energy(b)) < 1e-5
```

## Calibration

The battery must separate good models from broken ones. The legacy NF SIMURGH fails far-copy extensivity by 1.1 eV or more for a C5 copy, because its fragment virtual nodes couple separate molecules (as shipped; the opt-in `virtual_node.envelope_in_softmax: true` restores it). Legacy SIMURGH Simplified passes extensivity, but its energy jumps where a bond crosses its 1.8 Å fragment-perception cutoff (−0.64 eV for acetic acid's O–H in the trained Duplication model, up to 1.4 eV for diatomics), so bond scans must run through 1.8 Å and flag it. Normalisation layers are a common source of steps: during development a plain RMSNorm gave a 0.029 eV energy step at the cutoff (a node's features vanish as its last neighbour leaves), and a smooth variant fixed it. If a test cannot detect these cases, the test is wrong.

## Scripts rather than unit tests

- NVE drift: 10 ps for small molecules and 100 ps for an MD22 system at 0.5 fs, from an equilibrated frame; report the slope in meV/atom/ps (`goal-md-stability-benchmark`).
- Smoothness regression: compare scan metrics (test 10) with the previous accepted model; no metric may worsen by more than 10%.
- Tail force errors (Q95, Q99, max) next to every force MAE.

## Advanced checks

- Expressivity: pairs of structures that share all correlations up to body order n−1 but differ at order n must get different energies from a model that claims body order n.
- Product-basis multiplicities against exact counts (`goal-equivariant-kernels`).

## Done when

- [ ] All 13 tests pass for every flag combination.
- [ ] A failure is fixed, or recorded as a known limitation with an issue; never skipped silently.
