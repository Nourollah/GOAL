# SIMURGH config surface

Generated from `src/goal/configs/ml/simurgh_dev_gmd26_FD.yaml` as composed by Hydra: the `model` and `training` sections of `src/goal/configs/ml/simurgh_dev/_base.yaml` with their comments, plus the task's `mfc.enabled: true`. Regenerate this file whenever those configs change; `tests/test_agent_skills.py` fails when the key sets drift apart.

The rMD17, MD22 and 3BPA configs differ in task data, `mfc.enabled: false` (the base default) and `atomic_energies`: `mode: dataset` for rMD17 and MD22, the isolated atoms shipped with 3BPA for `simurgh_dev_3bpa`.

```yaml
model:
  backbone:
    name: simurgh_dev
    core: ace # ace | ece
    cutoff: ${data.cutoff}
    elements: null # derived from the training set by goal-train
    embedding_dim: 32
    num_channels: 128
    message_irreps: "128x0e + 128x1o + 128x2e"
    lmax_sh: 3
    num_interactions: 2
    correlation: 3
    product_basis: symmetric # symmetric | ladder | ladder_reduced
    num_rbf: 8 # stage 5 sweeps 8 | 16 | 32
    envelope_p: 5
    radial_mlp: [64, 64, 64]
    pair_conditioned_radial: true
    element_weights: index # index | embedding
    avg_num_neighbors: null # derived from the training set by goal-train
    node_norm: none # none | rms (Muon arm only)
    kernels: auto # auto | cuequivariance | e3nn
    dtype: float32 # float64 only in stage 13, on A100/H100/H200
    radial:
      low_rank: null # null | 1-4
      per_layer: null # null | [mlp, linear] | [linear, mlp] (stage 4; the source order is ambiguous)
      env_conditioned: false
    a_basis:
      regularity_prior_sigma: null # null | 0.25 | 0.5 (Å)
      prior_weight_decay: null # required when the prior is on
    attention:
      mode: gate # gate | softmax_null | radial (scores from distance only) | none
      pan_gate: false
    readout:
      pgs: true
      pgs_edge: false # spectral correction of the per-edge radial weights (arXiv:2606.15892, Eq. S9)
      pgs_edge_alpha: scalar # scalar | per_channel
      pgs_components: 16 # kernel centres on [-pgs_t_max, pgs_t_max] (arXiv:2606.15892, App. F)
      pgs_t_max: 3.0
      pgs_beta: 18.4 # kernel shape; the arm is 9.2
      mlp_hidden: 16 # last-layer readout width (raise it for the equal-parameter no-PGS control)
    pair_line:
      enabled: true
      mode: phi # phi | ece | phi+ece
      coefficients: table # table | lowrank
      basis: bessel # bessel | bspline
      num_basis: 8
      pair_cutoff: null # null = atom cutoff
      r_min: 0.6 # the wall acts only below this distance (Å)
      repulsive_wall: zbl # zbl | exp | none
      curvature_penalty: 1.0e-4
      many_body_penalty: 0.0
      pair_first_epochs: 0
      bond_order: false # saturating bond-order pair term (arXiv:2608.22933)
      bond_order_body: 3 # 3 | 4 (four-body: planned)
      bond_order_lmax: 4
      bond_order_nmax: 6
      bond_order_cmax: 8
      bond_order_q3: 36
      bond_order_pmax: 32
      bond_order_cutoff_fn: tanh3 # tanh3 (source) | envelope (the model's u)
      bond_order_eta_map: power # power (source) | smooth (C-infinity)
    ece: # used when core: ece, or pair_line.mode contains ece
      lmax: 2
      mmax: 2
      channels: 32
      bottleneck: 128
      activation: fourier # fourier | o2_product
      grid_points: 16
      num_mp_layers: 2
      message_width: 64
      heads: 8
      energy_mlp: [256, 256]
      film: { element_dim: 4, mlp: [64, 128] }
    electrostatics:
      enabled: false
      dipoles: false
      sigma: learned # learned | 1.0
    dispersion:
      enabled: false # fixed analytic D3(BJ) term; no DFT; decided in stage 0
      backend: torch # torch (tad-dftd3) | precomputed (dftd3-python)
      functional: pbe
      damping: bj
      atm: false # GMD labels: ORCA D3BJ without ABC
    diagnostics:
      calm: false
    # Isolated-atom energies (eV) of the GMD labels (PBE D3BJ def2-TZVP).
    # provided | dataset (least squares on the training set, fitted by
    # goal-train; for single-composition benchmarks only, never on GMD).
    atomic_energies:
      mode: provided
      values:
        1: -13.595298 # H
        6: -1028.481145 # C
        7: -1483.895483 # N
        8: -2041.117735 # O
    scale: force_rms # set to the training force RMS by goal-train
    mfc:
      enabled: true # GMD and long-molecule configs; false in rMD17, 3BPA and MD22
      objective: [energy, core_force] # energy | core_force | force
      window_type: halfspace # halfspace (GMD) | sphere (long molecules) | chemical
      levels: 2
      radii: [8.0] # sphere only; L - 1 radii; [8.0, 14.0] for levels: 3
      buffer: 5.0 # = atom-line cutoff r_c; values below r_c are ablation only
      windows_per_structure: 1
      every_n_steps: 1 # MFC on every n-th training step
      fraction: 1.0 # MFC on this fraction of each batch's structures
      weight: 1.0 # lambda_MFC; stage 0 sweeps 0.1 | 1 | 10
      weight_energy: 1.0
      weight_force: 1.0
      start_epoch: 20
      warmup_epochs: 10
      stop_grad: parent # parent | none | child
      parameters: shared # shared | separate
      window_weighting: uniform # uniform | violation
      violation_every: 10 # epochs between violation scores (violation weighting)
      chemical_scheme: rotatable # level-1 groups of chemical windows (goal.ml.data.fragments)
      tolerance: none # none | calibrated
      tolerance_file: null # calibrated: CSV d_min_A, d_max_A, tau_energy_eV, tau_force_eV_per_A
      include_charges: true
      diagnostic: true
  head: null


training:
  gradient_clip: 10.0 # clips the gradient, not the AMSGrad step
  safeguards: # trainer-side divergence guards; none of them changes the model
    lr_warmup_steps: 1000 # lr rises linearly over the first 1000 optimiser steps
    weight_ramp_steps: 1000 # a stage that raises a loss weight ramps it in over 1000 steps
    skip_nonfinite: true # a step with a NaN or inf gradient is skipped, weights untouched
  checkpoint_dir: ${paths.output_dir}/checkpoints
  compile_model: false
  compile:
    mode: "default"
    fullgraph: false
    dynamic: null
  slurm_mode: false
  performance:
    float32_matmul_precision: highest
    cudnn_benchmark: false
    cudnn_deterministic: false
  optimizer:
    name: amsgrad # amsgrad | muon | block_rms_adam (stage 1 decides)
    # 5e-3, not the brief's 0.01: at 0.01 a GMD FF run (seed 42) diverged in the
    # warm-up with no weight change: AMSGrad moves every element ~lr per step, coherent
    # steps doubled the weight norms in 120 steps and the product basis did the rest.
    lr: 0.005
    weight_decay: 5.0e-7 # decoupled; not on embeddings or pair-line coefficients
    muon_lr: 0.02 # muon only
    min_lr: 1.0e-6
    scheduler_type: reduce_on_plateau
    scheduler:
      patience: 50
      factor: 0.8
  ema:
    enabled: true
    decay: 0.99
  position_noise_std: 0.0
  aux: # training-only objectives; goal-train copies them to model.backbone.aux
    denoise: false # corrupted copies + noise head (rung D1; arXiv:2510.22123)
    hessian_distill: false # teacher Hessian rows (loss and label format only)
  # Warm-up: 5 epochs at E:F = 1:1 (energy 100, forces 100), then E:F = 1:100.
  # The force weight is 100 in every stage; at the switch only the energy weight
  # changes, and it goes down.  Two schedules diverged to NaN on GMD: an
  # energy-only warm-up (forces 0: the unchecked energy surface ran away to
  # ~1e14 eV/atom within 2-4 epochs) and a warm-up at forces 1 (the jump to 100
  # outran AMSGrad's second moment, the first updates were 2-3 lr, and the
  # unnormalised product basis turned that into NaN within 4 steps).  Without a
  # warm-up the energy gradient (with Huber: 1000x weaker than forces at 1:100) never
  # learns the binding energy over the isolated-atom references.  A stage that
  # raises a weight ramps it in (safeguards.weight_ramp_steps).  The separate arm
  # warmup_huber_switch adds a last stage (final 20% of epochs) with the
  # energy weight raised 400x and lr 2.5e-4, e.g.
  #   - {name: switch, epochs: null, lr: 2.5e-4, loss_weights: {energy: 400.0, forces: 100.0}}
  # after a finite "main" stage.
  stages:
    - name: "warmup"
      epochs: 5
      loss_weights:
        energy: 100.0
        forces: 100.0
    - name: "main"
      epochs: null
      loss_weights:
        energy: 1.0
        forces: 100.0
  # MSE on per-atom energy and on the per-atom force-error norm (the default
  # since 2026-11-16: on GMD Duplication, seed 42, it cut the TestID energy MAE
  # from 19.7 to 1.1 meV/atom against Huber at delta 1 meV/atom, forces equal).
  # Arms: fn huber (with delta) | l1.
  losses:
    - name: energy
      fn: mse
      weight: 1.0
    - name: forces
      fn: mse
      error: norm
      normalize_by_n_atoms: false # mean over atoms; true = per structure first
      weight: 100.0
  strict_stage_labels: true
```
