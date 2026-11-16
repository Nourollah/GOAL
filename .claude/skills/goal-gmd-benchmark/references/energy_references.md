# GMD energy references: evidence

Why the GMD energy columns need care, and how to tell a referencing error from a learning failure.

## Identifiability of OOD total energies from training energies

Rank tests on the training and OOD compositions (and on atom types of increasing radius):

| Task | OOD total energies identifiable from training energies? |
|---|---|
| Chain Extension | yes, at every additive level |
| Combination | yes, except radius-2 atom types in a few cases (the composition matrix has rank 3 of 4, but the OOD compositions lie in its row space) |
| Fusion | only at the bond (2-body) level, and only when the complex multifunctional molecules are in training |
| Duplication | at no level, not even element references |

Duplication training molecules are monoacids with composition (n_C, n_H, n_O) = (n, 2n, 2). The composition matrix has rank 2, with null vector C:H = 1:−0.5, so a least-squares fit pins only E0(C) + 2 E0(H) = −1055.67 eV. The minimum-norm solution is about C −211.1, H −422.3 and O −2041.1 eV.

## What the published energy errors are consistent with

| Element-reference handling | Predicted Duplication OOD energy error | Published values |
|---|---|---|
| minimum-norm least squares (what least-squares E0 fits and linear-reference fits for fine-tuning do) | about 822 eV per diacid | fine-tuned UMA-Small: 821.99 and 822.51 eV |
| no reference, per-atom mean scaling | about 3895 eV | SchNet, PaiNN, eSCN: 3929–3963 eV |
| isolated-atom references | small | MACE from scratch: 2.64 eV |

EquiformerV2's Duplication ID energy MAE is 6780.8 eV while its forces are good, another sign of reference handling rather than learning.

Conclusion: use isolated-atom references on GMD, run the probe on every checkpoint, and report centred and offset energy errors next to every total-energy MAE.
