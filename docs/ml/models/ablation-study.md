# SIMURGH ablation study

The ablation study measures what each SIMURGH component contributes on GMD-26. It starts from one stripped **base model**, adds exactly one component per run, and never stacks components in phase 1. A second phase then combines the strongest component with each of the others. The study has its own isolated model (`simurgh_ablation`) and its own numbered configs, so it can run for months while the production architecture keeps changing.

The full study design, run order, results and confound list are in the study's README. This page summarises it and shows how to run it:

**[src/goal/configs/ml/ablations/README.md](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/ablations/README.md)**

## Where everything lives

| | |
|---|---|
| Model | [`src/goal/ml/nn/models/ablations/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/ml/nn/models/ablations): `blocks.py` (`AblationFeatures` switches and the frozen blocks), `backbone.py` (`SimurghAblationBackbone`, registry name **`simurgh_ablation`**) |
| Configs | [`src/goal/configs/ml/ablations/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/configs/ml/ablations): `_ablation_base.yaml`, numbered rungs, `make_combo_configs.py` |
| Tests | [`tests/ml/ablations/`](https://github.com/Nourollah/GOAL/tree/main/tests/ml/ablations) |

Nothing in the production models imports from the ablation package, and the ablation package does not modify them.

## Why one switchable model

An ablation is only interpretable if "base + module X" differs from "base" by *exactly* the code that X adds. A dozen near-duplicate model files cannot guarantee that. So `SimurghAblationBackbone` writes its forward pass once, and every architectural choice is a field of `AblationFeatures` whose **default is the base setting**. Each numbered config flips one switch, so the experiment *is* the config diff.

- **Frozen in the study package:** the base skeleton (pair interaction, edge → node aggregation, round wiring, energy readout).
- **Imported unchanged:** the leaf primitives and the modules under test (`EquivariantFragmentInteraction`, `AdaptiveDepthGate`, `SymmetricContraction`). Forking those would mean measuring a fork rather than the real module.

An unknown switch name raises, so a typo cannot silently train the base under an experiment's name.

## The base model

The base is a minimal equivariant MPNN in the ARACE rhythm: `hidden_irreps: "32x0e + 16x1o"`, 2 rounds, cutoff 8 Å. It has **one** pair interaction shared by all element pairs, a plain equivariant update with residual and RMS norm, and one energy readout on the final node features. That comes to **163,316 trainable parameters**.

`simurgh_ablation` is a **modular** backbone. It fills `node_energies`, and the configs pair it with an `energy_forces` head. It is deliberately absent from `goal-train`'s injection lists, so `elements: [1, 6, 8]` and `avg_num_neighbors: 20.861964` are pinned in `_ablation_base.yaml` and cannot drift between rungs. These are fixed for the whole study:

- the data (GMD-26 FragmentDuplication) and its splits
- the provided atomic energies
- the optimiser
- a **400-epoch budget with no early stopping**

Switches in `model.backbone.features` (base values shown):

```yaml
features:
  pair_specialisation: universal   # universal | element_pair  (element_pair = the artisan bank)
  element_conditioning: false
  pair_gate: false                 # requires readout: pair_per_round
  pair_layers: 1
  body_order: 1                    # 1 | 2 | 3  (SymmetricContraction)
  per_element_update: false        # control for body_order
  readout: node_final              # node_final | pair_per_round
  share_round_weights: false
  residual_norm: rms               # rms | none | layer
```

The optional add-ons `fragment_interaction:` and `adaptive_gate:` are enabled by the presence of their section, exactly as in production ARACE.

## The ladder

Parameter counts are measured, from the README.

| # | Config | Change vs base | Compare against | Params |
|---|---|---|---|---|
| 01 | `01_gmd26_base` | none (reference) | | 163,316 |
| 02 | `02_gmd26_artisan_bank` | `pair_specialisation: element_pair` | 01, **92** | 949,876 |
| 03 | `03_gmd26_element_conditioning` | `element_conditioning: true` | 01 | 204,788 |
| 04 | `04_gmd26_deep_pair_block` | `pair_layers: 2` | 01 | 165,880 |
| 05 | `05_gmd26_body_order` | `body_order: 2` | 01, **06** | 1,215,988 |
| 06 | `06_gmd26_per_element_update` | `per_element_update: true` | 01 | 183,796 |
| 07 | `07_gmd26_per_round_energy` | `readout: pair_per_round` | 01 | 164,100 |
| 08 | `08_gmd26_pair_gate` | `pair_gate: true` (+ its prerequisite, rung 07's readout) | **07** | 164,102 |
| 09 | `09_gmd26_lmax2` | `hidden_irreps: … + 8x2e` | 01 | 239,606 |
| 10 | `10_gmd26_fragment_interaction` | `fragment_interaction:` section (12 Å fragment cutoff) | 01, **93** | 323,188 |
| 11 | `11_gmd26_adaptive_gate` | `adaptive_gate:` section | 01 | 165,016 |
| 12 | `12_gmd26_shared_round_weights` | `share_round_weights: true` | 01 | 84,660 |
| 13 | `13_gmd26_no_residual_norm` | `residual_norm: none` | 01 | 163,312 |
| 14 | `14_gmd26_equivariant_layer_norm` | `residual_norm: layer` | 01 | 163,440 |

**Controls.** These are not optional. Each one rules out a specific confound:

| # | Config | Rules out |
|---|---|---|
| 90, 91 | `9x_gmd26_base_seed43/44` | the seed-to-seed noise floor |
| 92 | `92_gmd26_base_width_matched` | rung 02's gain being pure capacity (`78x0e + 39x1o`, 943,752 params) |
| 93 | `93_gmd26_base_rounds3` | rung 10's gain being pure receptive field (3 rounds) |

**Phase 2 combinations (201–208)** take rung 05 (body order 2) as the anchor and add one other module each: ℓmax 2, element conditioning, deep pair block, shared rounds, fragment interaction, per-round energy, pair gate, artisan bank. They are generated by `make_combo_configs.py`. That script skips combinations that cannot be measured (two values of the same switch, or a module with no effect next to the anchor) and states the reason, so no GPU time is spent on a guaranteed null result.

## Running the ladder

```bash
# Phase 0: the noise floor. Run these FIRST; no delta means anything without them.
pixi run -e cuda goal-train-ml --config-dir src/goal/configs/ml --config-name ablations/01_gmd26_base
pixi run -e cuda goal-train-ml --config-dir src/goal/configs/ml --config-name ablations/90_gmd26_base_seed43
pixi run -e cuda goal-train-ml --config-dir src/goal/configs/ml --config-name ablations/91_gmd26_base_seed44

# Phase 1: one module at a time, in any order
pixi run -e cuda goal-train-ml --config-dir src/goal/configs/ml --config-name ablations/02_gmd26_artisan_bank
# ... through 14_gmd26_equivariant_layer_norm

# Controls, before drawing conclusions about rungs 02, 05 and 10
pixi run -e cuda goal-train-ml --config-dir src/goal/configs/ml --config-name ablations/92_gmd26_base_width_matched
pixi run -e cuda goal-train-ml --config-dir src/goal/configs/ml --config-name ablations/93_gmd26_base_rounds3

# Phase 2: (re)generate the combination configs, then run 201-208
pixi run -e cuda python src/goal/configs/ml/ablations/make_combo_configs.py --list
pixi run -e cuda python src/goal/configs/ml/ablations/make_combo_configs.py --anchor 05_body_order --start 201 \
  --only 09_lmax2 03_element_conditioning 04_deep_pair_block 12_shared_round_weights \
         10_fragment_interaction 07_per_round_energy 08_pair_gate 02_artisan_bank
pixi run -e cuda goal-train-ml --config-dir src/goal/configs/ml --config-name "ablations/201_gmd26_body_order+lmax2"

# The study's own guard-rail tests
pixi run -e cuda pytest tests/ml/ablations -q
```

All rungs log to the W&B group `simurgh_ablation_gmd26`, so the ladder reads as one overlaid plot. The primary metric is **`val_forces/forces_mae`** (GMD TestID), which is also what checkpoint `top_k` selects on. Also record:

- `test_forces/forces_mae`, the OOD number. **Read it; never select on it.**
- `val_energy/energy_mae_per_atom`
- `val_forces/forces_cosine_similarity`
- the parameter count and s/epoch
- the gate and per-round energy diagnostics for rungs 07, 08 and 11

**Cost**, measured on one RTX 6000 Ada at batch 48: about 11 s/epoch for the base (about 1.2 h per run), about 14 s/epoch for rung 05, and about 56 s/epoch for rung 02 (about 6 h), because the zero-mask schedule runs all six pair interactions on every step. Budget about 30 h for the full ladder plus controls.

> [!WARNING]
> Every file in `src/goal/configs/ml/ablations/` must start with `# @package _global_`. Without it Hydra merges the file under an `ablations:` key, and the rung silently trains the shared base. Keep this line when you copy a rung.

## Phase 1 results (summary)

From the README (completed 2026-10-02). Force MAE in meV/Å, ID = GMD TestID, OOD = GMD TestOOD.

- **Noise floor:** base 23.76 ± 0.34 ID, 74.67 ± 7.52 OOD. A single-run difference is resolvable only above about 1.0 ID / 21 OOD.
- **Capacity alone does not transfer.** The width-matched control (92) improves ID by 20% but OOD by nothing measurable. **Receptive field helps both:** 3 rounds (93) gives 16.96 / 62.80.

| Rung | ID | OOD | Verdict in the README |
|---|---|---|---|
| 05 body order 2 | **12.64** | **44.74** | best on both: the phase-2 anchor |
| 09 ℓmax = 2 | 13.73 | 54.13 | second on both, at a fifth of 05's parameters |
| 02 artisan bank | 14.06 | 159.0 | third ID, **last OOD**: disqualified as an anchor |
| 03 element conditioning | 18.44 | 61.18 | best OOD per parameter |
| 06 per-element update | 22.07 | 60.66 | small ID gain, real OOD gain |
| 12 shared round weights | 23.65 | 70.25 | free: half the parameters, no loss |
| 04 deep pair block | 21.55 | 76.68 | ID only |
| 10 fragment interaction | 22.46 | 75.28 | within noise on both |
| 07 / 08 per-round energy / pair gate | 41.41 / 38.50 | 99.26 / 97.61 | the per-round readout costs a round of depth |
| 13 / 14 no norm / layer norm | 36.56 / 29.76 | 114.8 / 130.6 | the RMS norm is load-bearing |
| 11 adaptive gate | 1531 | 1103 | not a result (see README §6) |

The headline: per-pair specialisation (the artisan bank) raises capacity per pair type but shrinks the training data each parameter sees. Against its width-matched control it is 26% better ID and **2.3× worse OOD**. Element conditioning reaches element awareness through a *shared*, Z-modulated map and transfers better than the base. Selecting on ID alone would have picked the worst-transferring model in the study (Spearman ID-vs-OOD rank 0.60).

## Known confounds

The README lists these in full (§7). The ones that most limit interpretation:

1. **Parameters are not held constant.** Only rung 92 is width-matched, and only for rung 02.
2. **One learning rate (0.01) is used for the whole ladder.**
3. **Rung 05 moves two things,** body order and per-element update weights. Body order proper is 05 − 06.
4. **The pair gate requires the per-round readout,** so the model refuses `pair_gate` without `readout: pair_per_round`, and rung 08's baseline is 07.
5. **One dataset.** GMD-26 has H/C/O carboxylic acids and only 6 element pairs.
6. **The fragment channel is intramolecular here.** Every GMD frame is one connected molecule, so all rungs use the `custom` SMARTS cut into CH3/CH2/COOH.
7. **Dead parameters in `pair_per_round` mode.** The last round's update block gets no gradient. This reproduces production `simurgh_arace`. Do not run rungs 07/08 under DDP with `find_unused_parameters: false`.

## Adding a rung

1. Add a switch to `AblationFeatures` in `blocks.py`, defaulting to the **base** behaviour.
2. Add the same key, at the same default, to `features:` in `_ablation_base.yaml`. The tests fail if the two drift apart.
3. Copy a rung config (keep `# @package _global_`), bump the number, override one switch, and fill in the header.
4. Add the rung to `RUNG_SWITCHES` in `tests/ml/ablations/test_ablation_model.py`.

See also: [SIMURGH](simurgh.md) for the production architecture being measured, and [Hyperparameter tuning](../training/hyperparameter-tuning.md).
