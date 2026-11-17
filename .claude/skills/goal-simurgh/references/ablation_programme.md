# SIMURGH ablation programme

Configs in `src/goal/configs/ml/ablations/simurgh_dev/`, seeds 42, 43 and 44. General rules: `goal-ablation-study`.

## Base configuration B0

- Core: ACE, correlation 3 (`symmetric`), L = 2, lmax_sh 3, 2 layers, 5 Å, 128 channels, 8 Bessel functions.
- Lines: φ_ab pair line with ZBL wall, pair-conditioned radial weights, attention gate, PGS branch.
- Training: Huber loss (MSE since stage 11) after a 5-epoch warm-up at E:F = 1:1 (force weight 100 in every stage), AMSGrad, EMA, trainer safeguards, float32.
- MFC on in the GMD configs (`[energy, core_force]`, half-space, 2 levels, b = 5 Å).

References: B-REF (trained outside the repository) and L0, the earlier SIMURGH model as shipped in `v0.1.0` (its code lives in the git history).

Two arms still wait for inputs from the maintainer: the calibrated MFC tolerance (an offline calibration file) and Hessian distillation (teacher labels); see `goal-simurgh`.

## Stages (each starts from the best configuration of the previous one)

| Stage | Arms | Benchmarks | Purpose |
|---|---|---|---|
| 0 | B0 sanity (all invariant tests); L0 re-scored with the new metrics; MFC diagnostic on B0 without MFC and on B-REF; λ_MFC ∈ {0.1, 1, 10}; D3 on and off | GMD Duplication and Fusion | fix B0 on GMD |
| 1 | AMSGrad vs block-RMS Adam (betas 0.98/0.99) vs Muon per irrep block; node_norm none/rms inside the Muon arm | rMD17 (aspirin, ethanol), 3BPA including 1200 K, GMD Duplication and Fusion, full budget | optimiser first |
| 2 | correlation 4 (`ladder`; `symmetric` if feasible) and 5 (`ladder_reduced`), each with and without the regularity prior (σ 0.25 and 0.5 Å), at matched parameters | rMD17, 3BPA, GMD | body order |
| 3 | L = 3 messages, with and without the prior | rMD17, 3BPA, GMD | angular resolution |
| 4 | small-molecule settings: cutoff 4.0 or 4.5 Å with correlation 4; `radial.per_layer: [mlp, linear]` and `[linear, mlp]` (the source's order is ambiguous); small width | rMD17 (N = 1000 and 50), 3BPA | small-molecule recipe |
| 5 | num_rbf {8, 16, 32} × prior on/off; low-rank radial; environment-conditioned radial | rMD17, MD22 (2 systems), GMD | radial |
| 6 | Huber (B0) vs MSE vs L1; late energy-weight switch as a separate arm; weight-decay sweep | all screening sets | recipe |
| 7 | MD22 depth: 3 layers (matched) and 2 layers at 6 Å; then the electrostatics ladder (charges, then dipoles, then quadrupoles or MFC charges), each step beating the 3-layer model under the MD22 gate | MD22, long-molecule set | depth before long range |
| 8 | `core: ece` vs `core: ace` at matched parameters; ACE core plus ECE line (`pair_line.mode: phi+ece` and `ece`) | MD22, GMD, rMD17 | bond-centred representation |
| 9 | the literature ablation files; PAN gate; bond-order pair line; auxiliary denoising; Hessian distillation (reported separately); routed per-pair experts (negative control); MFC variants; strictly local branch (1 layer, correlation 4, with and without MFC) | as relevant | additions and controls |
| 10 | removals: attention (also radial-only, matched); PGS (and an equal-parameter MLP control); MFC; pair line; pair-conditioned radial; D3 if adopted | all four GMD tasks, rMD17, MD22 | leave-one-out |
| 11 | the combined base (`combined_base.yaml`): MSE (now the default loss), MFC at λ 0.1 with violation-weighted windows, PGS β 9.2, environment-conditioned radial, correlation 5 (`ladder_reduced`), weight decay 1e-3; leave-one-out of each (and MFC off); denoising ×0.25; the stage-9 long-range and quadrangle arms on it | GMD Duplication (seed 42); the base also on Fusion, seeds 42 and 43 | new reference |
| 12 | an ECE branch next to the unchanged ACE core: `s12_ace_ece_sum` (one energy, `pair_line.mode: phi+ece`), `s12_ace_ece_fusion` (ACE layers and ECE rounds exchange messages, zero-initialised links), `s12_ace_ece_heads` (two full energy heads averaged, agreement loss 0.1; their disagreement is an out-of-distribution score), `s12_core_ece` (ECE alone); MFC covers the ECE energy in all; a winner runs on Fusion with an ACE widened to the same parameter count as control | GMD Duplication (seed 42) | bond-centred branch |
| 13 | float64 training; float64 L-BFGS polish (rMD17, 3BPA; L2 on the product-basis weights, arm on all non-embedding weights); A100, H100 or H200 only | rMD17, 3BPA | final numbers in both precisions |

Arm configs exist for every row except these (planned): the stage-4 cutoff (4.0, 4.5 Å) and small-width arms, the stage-7 quadrupoles, and the stage-9 routed per-pair experts. Stage 13 runs as overrides (`model.backbone.dtype=float64 trainer.precision=64-true`) plus `scripts/simurgh_dev/lbfgs_polish.py`. File names carry the stage: `sNN_<arm>.yaml`.

## Protocol

1. Screen each arm with seed 42: GMD arms on Duplication and Fusion at 400 epochs; standard-benchmark arms on rMD17 aspirin and ethanol, 3BPA, and MD22 AT-AT and stachyose.
2. Seeds 43 and 44 only for arms whose seed-42 change exceeds 2× the base s.d. (L0's OOD force s.d. on Duplication is 7.5 meV/Å; re-estimate it for B0 in stage 0).
3. The final model and the top three alternatives run on all four GMD tasks plus `gmd26_FD_plus_alkane`, rMD17 (N = 1000 and 50), MD22 and 3BPA.
4. Gates: cutoff crossing, smoothness, MD stability and tail errors before accuracy counts; the MD22 per-system gate for depth, attention, long-range, ECE and literature long-range arms; the 3BPA 1200 K and dihedral gate (3 seeds) for stages 1–4, 8 and attention.

From stage 11 on, resources are limited: seed 42 for every run; seed 43 only for a run that beats the combined base by more than 40 meV/Å OOD force on Duplication or 8 meV/Å on Fusion, or that is planned for adoption; smaller single-seed differences are ties.  Tables report ID and OOD force, energy per atom and per molecule (MAE, MSE, RMSE), the checkpoint chosen on S = F_MAE (meV/Å) + E_MAE (meV/atom) on TestID plus the last epoch, stability notes, time per step and memory.
