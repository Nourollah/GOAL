# SIMURGH (development form)

SIMURGH is GOAL's native interatomic potential. While it is under development it is registered as `simurgh_dev` and lives in one monolithic model file, [`models/simurgh/dev/monolithic.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/simurgh/dev/monolithic.py), organised in one section per work package. The older SIMURGH models are under [`models/simurgh/legacy`](simurgh.md) with `simurgh_legacy*` keys. At release the `_dev` suffix goes and the model takes the `simurgh` key.

## Energy

With `core: ace` (default):

$$E = \sum_i E_0(Z_i) + s\Big[\sum_i e_i + \sum_{i<j}\phi_{Z_iZ_j}(r_{ij}) + \sum_{i\to j}\varepsilon_{ij}\Big] + W_{\text{rep}} + E_{\text{elec}} + E_{\text{D3}}$$

| Term | Section | Default |
| --- | --- | --- |
| $e_i$: body-ordered ACE atom line (channel-wise A-basis, symmetric product basis up to $\nu = 3$, linear update, per-layer readout) | WP2 | on |
| smooth local attention (`gate`, exactly uniform when fresh) and the PGS readout (arXiv:2606.15892, Eq. S10): $y_i = x_i + D(x_i) + \sum_m \alpha_m K((t_{im} - t_m)/w)$ lifts the final invariants that the energy MLP reads; 16 compact exponential-of-semicircle kernels on $[-3, 3]$, $\beta = 18.4$ | WP8 A, C1 | on |
| PGS edge (`readout.pgs_edge`, arXiv:2606.15892, Eq. S9): $w'_{ij} = w_{ij} + g(e_{ij}) \odot \sum_m \alpha_m B_m(r_{ij})$ on the per-edge tensor-product weights, 16 enveloped Fourier–Bessel functions | C2 | off |
| PAN gate (`attention.pan_gate`, arXiv:2606.15892, Eqs. S14-S15): $a_{ij} = 2\sigma(f([e_{ij}, Z_i, Z_j, \tilde\rho_i, \tilde\eta_i]))$ on the 0e slots of each message, from the coordination $\rho$ and angular variance $\eta$ of the receiving atom, standardised with training-split statistics that `goal-train` computes; `pan_gate_descriptors: generic` drops $\rho, \eta$ | C3 | off |
| $\phi_{ab}(r)$: pair line per unordered element pair, plus a fixed ZBL wall below `r_min` | WP3 | on |
| $\varepsilon_{ij}$: ECE bond line (`pair_line.mode: ece \| phi+ece`) | WP11 | off |
| $E_{\text{elec}}$: latent charges (and dipoles), damped Coulomb; `electrostatics.charge_constraint: uniform \| pair_transfer` conserves each molecule's total charge (`total_charge` on the graph, else 0) | WP7, C4 | off |
| $E_{\text{D3}}$: closed-form D3(BJ) dispersion (no DFT); `backend: torch` (tad-dftd3, in the `cuda` environments) or `precomputed` (dftd3-python) | WP1 | off |

`core: ece` replaces $\sum_i e_i$ with the edge cluster expansion: bond states in each bond's own frame (O(2) features), updated by line-graph message passing, with energy $\sum_{i\to j}\varepsilon_{ij}$. The ECE modules are generic GOAL code: `nn/primitives/edge_frame.py`, `nn/primitives/o2.py`, `nn/blocks/ece.py` and `nn/heads/edge_energy.py`.

Forces are conservative only, $F = -\partial E/\partial R$. Every map is bias-free and every optional section starts at zero. An isolated atom therefore gives exactly $E_0$, and switching a section on leaves a fresh model unchanged. $E_0$ is kept in float64 whatever the parameter dtype. It comes from `atomic_energies: {mode: provided}`: isolated-atom energies of the labels' level of theory (GMD's in the GMD configs, the atoms that ship with 3BPA in `simurgh_dev_3bpa`). rMD17 and MD22 ship no isolated atoms, so their configs use `mode: dataset` (least squares on the training set, fitted by `goal-train`): each model sees one composition, so this removes only a constant. Never use `mode: dataset` on GMD.

## Outputs

`energy`, `forces`, `num_atoms`, `energy_atom_line`, `energy_pair_line`, `energy_d3`, `energy_elec`, `atomic_energies`. Some outputs appear only when their part is active:

- `edge_energies`: with an ECE core or line;
- `charges` and `dipole`: with electrostatics;
- `fragment_energies` and `fragment_pair_energies`: when the graph carries `fragment_index`; this is an exact, parameter-free split;
- `aux_loss` and `aux_terms`: in training.

## Multi-scale fragment consistency (MFC)

The same network is evaluated on windows of a structure: half-space cuts or spheres, nested for three levels. On the window's core atoms (at least `mfc.buffer` from every removed atom) it must reproduce what it predicts inside the larger context:

- the atom-line energies;
- the forces those energies generate;
- the charges, when electrostatics are on.

The larger context is the stop-gradient teacher. With `buffer >= r_c` the objectives vanish for a strictly local model, so they penalise only dependence carried by message passing. MFC is on in the GMD configs and off for rMD17, MD22 and 3BPA. Its parts are logged under `train_mfc/`.

## Configuration

`src/goal/configs/ml/simurgh_dev/_base.yaml` lists every key with its default; unknown keys raise. Task configs:

- GMD-26: `simurgh_dev_gmd26_{FD,FF,FCE,FCB}` and `simurgh_dev_gmd26_FD_plus_alkane`;
- `simurgh_dev_rmd17_<molecule>`, `simurgh_dev_md22_<system>` and `simurgh_dev_3bpa`.

```bash
pixi run -e cuda goal-train --config-name simurgh_dev_gmd26_FD
```

`goal-train` derives `elements`, `avg_num_neighbors` and `scale: force_rms` (the RMS of the training force components) from the training set.

The training recipe:

- Huber loss on per-atom energy and on the per-atom force-error norm;
- five energy-only epochs, then E:F = 1:100;
- AMSGrad with EMA.

Checkpoints are selected on `val_forces/forces_mae_atomw`, the exact atom-weighted force MAE on the validation set (TestID on GMD).

`training.optimizer.name` selects `amsgrad`, `muon` (per irrep block) or `block_rms_adam`. The model builds its own optimiser through `build_optimizer`.

The ablation programme is in `src/goal/configs/ml/ablations/simurgh_dev/`, one key per arm, with stage numbers in the file names.

## Scripts (`scripts/simurgh_dev/`)

| Script | Purpose |
| --- | --- |
| `score_predictions.py` | dataset-level metrics of `goal-eval predictions_file=` outputs or any file in the same format |
| `probe_reference.py` | isolated-atom energies and a composition regression of the residuals |
| `context_sensitivity.py` | mean $\lVert\partial e_i/\partial r_j\rVert$ binned by distance |
| `mfc_locality.py` | locality map of any GOAL model, locality length, per-atom scores |
| `md_stability.py` | survival, instability onset, temperature and NVE drift |
| `export_pair_potentials.py` | $\phi_{ab}(r)$ and derivatives to CSV and PNG |
| `benchmark.py` | time and memory per training step |
| `migrate_legacy_checkpoints.py` | rewrite pre-legacy registry keys in saved checkpoints |

## Physical long-range ablation

`simurgh_dev_abl_physical_long_range` adds learned dispersion and erf-damped electrostatics, each behind an amplitude that starts at zero (DOI 10.1021/jacs.5c09558; C8 and C10 from the oscillator relations of DOI 10.1021/acs.jctc.3c00797). Volume ratios $h_i = \mathrm{softplus}(b_Z + \mathrm{MLP}(s_i))$ start at 0.8 and scale free-atom $\alpha_0$ and $C_6$ (H, C, N and O only; provisional values from the source's released code). The `long_range:` section sets `electrostatics_kernel` (`erf_fixed`, σ = 4 Å, or `gaussian_pair` for the WP7 module), `charge_constraint`, `exact_prefactor`, `disp_damping_gamma` (1.2) and an optional `cutoff`. Never stack it with the D3 term; `tests/ml/simurgh_dev/fixtures/longrange_dispersion_reference.csv` holds the reference pair energies the tests reproduce.
