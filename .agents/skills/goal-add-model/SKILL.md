---
name: goal-add-model
description: Adds a new interatomic-potential architecture, layer or ablation variant to GOAL. Registers it in MODEL_REGISTRY and BACKBONE_REGISTRY, chooses the modular (backbone plus head) or monolithic (head null) form, rebuilds differentiable geometry from graph.pos, computes conservative forces by autograd, keeps every parameter in the DDP graph, adds one strict config per benchmark, and adds tests and a docs page. Use when the user asks to implement, port or register a model in GOAL, or a new model gives zero forces or DDP errors.
license: MIT
---

# Adding a model to GOAL

Read first: `docs/ml/models/custom-model.md` (worked examples) and `docs/ml/models/overview.md` (how `GOALModule` composes models, the registry, dataset-derived arguments).

## 1. Choose the form

- **Modular.** The backbone returns node features and a head from `HEAD_REGISTRY` produces energy and forces. Expose `irreps_out` and `num_interactions`, no `output_keys`; the head's `irreps_in` must equal `irreps_out`.
- **Monolithic.** The model returns the property dict itself. Expose `output_keys`, return `energy`, `forces` and `num_atoms`, compute forces under `torch.enable_grad()`, and configure `model.head: null`. `goal-train` raises when the form and the head setting disagree.

```python
import torch
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

@MODEL_REGISTRY.register("my_model")
@BACKBONE_REGISTRY.register("my_model")
class MyModel(torch.nn.Module):
    ...
```

Put the module under `src/goal/ml/nn/models/` (subpackages included) so that auto-discovery imports it: the package `__init__` calls `goal.ml.registry.auto_discover`, which imports every submodule. A module that raises `ImportError` is skipped with only a `UserWarning`, so check `BACKBONE_REGISTRY.list_available()` and `MODEL_REGISTRY.list_available()` after `import goal.ml.nn.models`.

## 2. Physics that is not negotiable

- **Geometry from `graph.pos`.** Make `graph.pos` require grad and recompute edge vectors from it, including periodic shifts: `goal.ml.nn.models.simurgh.geometry.differentiable_edges(graph, positions)` returns `(edge_vectors, edge_lengths)`. Otherwise forces are zero, silently.
- **Conservative forces.** F = −∂E/∂R by autograd, with `create_graph=True` during training so that the force loss can be backpropagated. No direct-force heads in native models.
- **Smooth cutoff.** Every radial weight goes to zero with continuous derivatives at r_c: multiply the radial network output by a polynomial envelope.
- **Extensivity.** No pooling, normalisation or attention over all atoms of a structure, and no routing on global composition. Two far-apart molecules must give the sum of their energies.
- **Smooth normalisation.** Normalisation layers (RMSNorm, LayerNorm) must stay smooth when a node's features approach zero; a plain RMSNorm produced an energy step at the cutoff.
- **Element references.** Accept `atomic_energies` (`mode: learned | dataset | provided`, plus `values`) and `scale`.

## 3. Infrastructure rules

- **Static DDP graph.** Every learnable parameter reaches the loss on every step, including batches in which a branch has nothing to gather (add a zero-weighted contribution instead of skipping the branch). Never rely on `find_unused_parameters`.
- **Outputs.** `energy` (per structure), `forces`, `num_atoms`; optional decompositions (`atomic_energies`, per-term energies) and `aux_loss` for model-side regularisers. `GOALModule` adds a returned `aux_loss` (or a backbone attribute `last_aux_loss`) to the total loss and logs it as `aux_loss`; an optional `aux_terms` dict of its parts is logged, not added again.
- **Dataset-derived arguments.** `src/goal/ml/cli/train.py` injects `elements` only for `simurgh_legacy`, `simurgh_legacy_ace_first`, `simurgh_legacy_arace`, `simurgh_legacy_monolithic`, `simurgh_legacy_monolithic_arace` and `simurgh_dev`, and a null top-level `avg_num_neighbors` only for `simurgh_legacy_arace` and `simurgh_dev` (a null `dressing_kwargs.avg_num_neighbors` is filled for any backbone). For a new backbone name, put them in the config or extend that logic, and raise when `avg_num_neighbors` is missing instead of defaulting.
- **Generic hooks.** A backbone class may define `atom_statistics(config)` to request the training-split mean and s.d. of per-atom descriptors; `goal-train` passes them back as the constructor argument `atom_statistics`. A monolithic backbone (`model.head: null`) may define `build_optimizer(opt_cfg)`; `GOALModule` then uses its optimiser, receiving the whole `training.optimizer` block, and still builds the scheduler itself.
- **Precision.** Accept `dtype: float32 | float64`.
- **Autograd in eval mode.** Diagnostics differentiate atomic energies after training, so do not detach outputs or switch off gradients inside the model in eval mode.
- **Kernels.** e3nn is the only implemented kernel path; optional cuEquivariance kernels behind an import guard and a parity test (`goal-equivariant-kernels`) (planned).

## 4. Configs

- One self-contained YAML per benchmark task in `src/goal/configs/ml/`, in the style of the existing files (a header comment that explains the experiment).
- Unknown keys raise `ValueError`. Every optional capability defaults to off, so that each ablation is a one-key diff (`goal-ablation-study`).

## 5. Tests and docs

- The invariant battery in float64 (`goal-mlip-invariant-tests`), a registry test, a config-composition test (`--cfg job`) and a 2-process DDP test on the CPU (gloo).
- The smoke test in `docs/ml/models/custom-model.md`: forces non-zero, net force about zero for an isolated molecule.
- A page in `docs/ml/models/` and a `mkdocs.yml` navigation entry.
- `pixi run -e cuda test`, `lint`, `format` and `typecheck` before committing.

## 6. House rules

- Follow `goal-naming-policy` for every file, key, docstring and commit message.
- Do not change shared modules (`src/goal/ml/nn/blocks/`, `src/goal/ml/nn/primitives/`, `src/goal/ml/data/`) or the behaviour of existing models without the maintainer's approval. While a model is in development, copy small primitives into its own file.
- Ablation variants are subclasses that override one section only: one file per mechanism, named after the mechanism rather than a paper.

## Done when

- [ ] `+debug=fdr` and `+debug=overfit` pass with the new config.
- [ ] The invariant battery passes in float64.
- [ ] The DDP test passes with `find_unused_parameters=False`.
- [ ] The docs page and navigation entry exist.
