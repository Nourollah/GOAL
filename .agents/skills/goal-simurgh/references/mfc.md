# Multi-scale fragment consistency (MFC)

Fragments are processed at several overlapping scales by the same network. A dedicated loss requires the fine-grained outputs of an atom computed inside a small window to equal those computed in the larger window or the full structure that contains it, for atoms deep enough inside the small window. The global energy and force loss trains everything; the MFC loss aligns the scales. The effect is a learned nearsightedness prior: it selects a local, transferable partition of the energy, which the training data cannot determine on GMD Duplication or Fusion, and it suppresses spurious dependence on distant groups, without imposing hard one-layer locality.

## Windows (`mfc.window_type`)

- `halfspace` (GMD default): a direction u (the principal axis with probability 0.5, otherwise uniform on the sphere) and a threshold t; the window keeps atoms with (r − r_mean) · u ≤ t. (u, t) is resampled until the window removes at least one heavy atom and its core keeps at least one; a structure is skipped after 8 failed tries. On chains this cuts off one chain end.
- `sphere` (long-molecule default): all atoms within R_ℓ of a seed atom, nested radii, R = (8.0,) Å for 2 levels and (8.0, 14.0) Å for 3; R_1 ≥ buffer + 3 Å; windows that contain the whole structure are rejected.
- `chemical`: level 1 = chemical groups from `src/goal/ml/data/fragments.py` (`mfc.chemical_scheme`, default `rotatable`), higher levels merge bonded neighbour groups (bonds: covalent radii × 1.2). H must be among the model's elements. Cut bonds are capped with H atoms on the cut-bond line (1.09, 1.01 and 0.96 Å from C, N and O), which count as changed atoms; cap positions are held fixed and caps are left out of the force comparison. Bond perception happens only here, in the training loss, never in the energy.

## Core mask

Core atoms are window atoms at least `mfc.buffer` = b from every removed atom and every cap atom. With b = r_c (5 Å, the default) every core atom keeps its exact first-shell environment, so the loss penalises only dependence carried by message passing. b < r_c would penalise real first-shell chemistry and is an ablation only.

## Objectives (`mfc.objective`, default `[energy, core_force]`)

- `energy`: Σ over core atoms of (e_i(window) − sg(e_i(parent)))² / σ_e², with e_i the atom-line atomic energy (excluding E0 and the pair-line, electrostatic and D3 terms).
- `core_force`: Σ over window atoms of ‖F_i^core(window) − sg(F_i^core(parent))‖² / σ_F², with F^core = −∂(Σ_core s·e_k)/∂r. It makes the core energies agree as functions of the positions; it needs a double backward pass on the window graph.
- `force` (optional): full forces on core atoms. It also penalises real cut effects unless b ≥ 2 r_c or the calibrated tolerance is used, so it is not the GMD default.
- Charges, when electrostatics are on: (q_i(window) − sg(q_i(parent)))² / σ_q².
- Pair-line, D3 and electrostatic terms are excluded by design.
- `stop_grad: parent` (default): the larger context is the teacher.

## Schedule, levels and coverage

- λ_MFC warms up linearly from 0 after `mfc.start_epoch` (default 20); stage 0 sweeps λ_MFC ∈ {0.1, 1, 10}.
- 2 levels on GMD (window against full structure); 3 levels need molecules longer than about 20–30 Å.
- Measured on GMD training sets (half-space, b = 5 Å): conformers with a valid cut are 20% for Chain Extension, 92% for Fusion, 100% for Duplication and 73% for Combination. MFC therefore cannot help Chain Extension. The trainer logs the per-epoch fraction of structures with a valid window.

## Selection rule

Keep MFC in the GMD configs only if ID force MAE is at most 2% worse than the same model without MFC, OOD force MAE improves by more than 2 s.d. on at least one of Duplication, Fusion and Combination, and no task worsens by more than 1 s.d.

## Diagnostic (`scripts/simurgh_dev/mfc_locality.py`)

For any GOAL checkpoint: sample windows, then for each core atom compute |Δe_i|, ‖ΔF_i^core‖ and ‖ΔF_i‖ against the distance to the nearest removed atom. Outputs: decay curves, a locality length (the distance beyond which the median ‖ΔF_i^core‖ stays below 1 meV/Å), per-atom locality scores and their Spearman correlation with per-atom OOD force errors. Run it before MFC training on the base model and on B-REF (window sets are exported under neutral file names for B-REF's workspace), and again on every MFC arm.

## Options

- `mfc.parameters: shared | separate`. `separate` gives every level above the first a residual energy section fed by a pooled window embedding; the MFC loss compares only the shared atomic energies. Ablation only: the pooled mean breaks far-copy extensivity for unlike fragments.
- `mfc.window_weighting: uniform | violation`. `violation` scores each training structure every `mfc.violation_every` epochs (default 10), from its own training windows in that epoch: v_s = mean ‖ΔF_i^core‖ over core atoms. Each structure's MFC loss is weighted by clip(v_s / mean v, 0.25, 4), renormalised to mean 1 (unscored structures: 1); with `mfc.fraction` < 1 the structures that get windows are drawn in proportion to the weights. The minimum and maximum weights are logged.
- `mfc.every_n_steps` and `mfc.fraction` reduce the cost: MFC on every n-th step, or on a fraction of each batch.
- `mfc.tolerance: none | calibrated`. `calibrated` will replace each squared error by max(0, ‖Δ‖ − τ(d))² (planned). It raises, naming `mfc.tolerance_file`, which it needs: a CSV with columns `d_min_A`, `d_max_A`, `tau_energy_eV` and `tau_force_eV_per_A`, one row per bin of the distance to the nearest removed or cap atom, computed offline by the maintainer from ORCA single points on capped `chemical` windows. The model never runs DFT.
