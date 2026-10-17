# Other backbones

Besides the SIMURGH family, GOAL ships two small modular baselines, HyperSpec (equivariant) and InvariantGNN (invariant), plus a minimal monolithic reference model and trainable wrappers around foundation potentials.

## HyperSpec (`hyperspec`)

[`src/goal/ml/nn/models/hyperspec.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/hyperspec.py), class `HyperSpecModel`, registered in `MODEL_REGISTRY` and `BACKBONE_REGISTRY`.

HyperSpec is a plain E(3)-equivariant message-passing network:

1. atomic-number embedding
2. `EquivariantLinear` to the hidden irreps $\bigoplus_{l=0}^{l_\max}$ `hidden_channels`$\times l^{(-1)^l}$
3. `num_interactions` × `EquivariantInteractionBlock` (tensor product of node features with edge spherical harmonics, radially weighted)
4. `NodeFeatures` returned, with no `node_energies`

The module docstring calls it "a placeholder implementation that establishes the interface". Treat it as a baseline and an example of the `EquivariantBackbone` protocol, not as a tuned model.

| Key | Default | Shipped in `hyperspec_md17.yaml` | Meaning |
|---|---|---|---|
| `num_elements` | 120 | 120 | embedding table size |
| `embedding_dim` | 64 | 64 | initial scalar embedding width |
| `hidden_channels` | 128 | 128 | multiplicity of every $l$ block |
| `num_interactions` | 3 | 3 | message-passing layers |
| `lmax` | 2 | 2 | maximum SH / feature order |
| `cutoff` | 5.0 | 5.0 | must match `data.cutoff` |
| `num_radial_basis` | 8 | 8 | Bessel basis size |

HyperSpec has no `atomic_energies`, `scale` or `elements` keys. The head's `irreps_in` must match the backbone's output: for `lmax: 2` and 128 channels that is `"128x0e+128x1o+128x2e"`.

```bash
pixi run -e cuda goal-train --config-name hyperspec_md17
```

## InvariantGNN (`invariant_gnn`)

[`src/goal/ml/nn/models/invariant.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/invariant.py), class `InvariantGNN`. It is a SchNet-style network: continuous-filter convolutions whose filters come from a Bessel basis of the edge lengths, with residual node MLPs. It satisfies the `InvariantBackbone` protocol and outputs `"{hidden_channels}x0e"`. Keys: `num_elements` (120), `embedding_dim` (64), `hidden_channels` (128), `num_interactions` (3), `cutoff` (5.0), `num_radial_basis` (20). No config ships for it.

Use it as a fast invariant baseline, or as a template for other invariant architectures (DimeNet-, GemNet-style). Because it has no $l=1$ channels, it cannot be used with `direct_forces` or with the direct/hybrid modes of `dual_forces`.

> [!NOTE]
> **Forces work since October 2026.** Both baselines used to read the neighbour list's stored `graph.edge_attr` / `graph.edge_weight`, which carry no autograd link to `graph.pos` (and are `None` on graphs built by `AtomicGraph.from_ase`). The `energy_forces` head therefore returned **all-zero forces** and the force loss never trained anything. They now re-derive edge vectors and lengths from `graph.pos` + `unit_shifts` with `differentiable_edges`, like SIMURGH, so `energy_forces` gives conservative forces; the tests check them against finite differences of the energy and check that they sum to zero.

## Monolithic example (`monolithic_example`)

[`src/goal/ml/nn/models/monolithic_example.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/monolithic_example.py) is a SchNet-without-message-passing reference for the `MonolithicModel` protocol. It sums a per-element MLP energy and a pairwise Bessel-MLP term with a cosine cutoff, and computes forces by autograd. It recomputes edge lengths from `graph.pos` inside `torch.enable_grad()`, which is the pattern to copy for monolithic models. Keys: `num_elements`, `embedding_dim`, `hidden_dim`, `cutoff`, `num_radial_basis`. It needs `head: null`. See [Writing a new model](custom-model.md#a-new-monolithic-model).

## Foundation-model backbones

[`src/goal/ml/nn/models/foundation/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/ml/nn/models/foundation) wraps pre-trained potentials as real `nn.Module` **monolithic** backbones. That means they train inside the standard `GOALModule` loop with `head: null`, reusing the same loss, stage curriculum, checkpoint manager, DDP/FSDP and HPO.

| Registry name | Class | Config |
|---|---|---|
| `mace_finetune` | `MACEFinetune` ([`foundation/mace.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/foundation/mace.py)) | [`src/goal/configs/ml/finetune/mace.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/mace.yaml) |
| `uma_finetune` | `UMAFinetune` ([`foundation/uma.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/foundation/uma.py)) | [`src/goal/configs/ml/finetune/uma.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/uma.yaml) |

The shared base class `FoundationModelBackbone` ([`foundation/base.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/foundation/base.py)) implements three strategies:

- `head_only`: freeze everything except the readouts; recommended below about 5,000 structures
- `full`: train everything with a small LR
- `lora`: inject LoRA adapters through `peft`

It also handles optional re-estimation of the per-element reference energies. Upstream packages (`mace-torch`, `fairchem-core`, `peft`) are imported lazily, so the modules register even when those packages are missing.

```bash
.venv-mace/bin/goal-train --config-name finetune/mace          # MACE needs its own venv
```

UPET/PET is fine-tuned through the separate `goal-finetune-upet` entry point (metatrain), not through a backbone. The older `mace-large` / `mace-medium` / `mace-small` / `uma-small` adapter names are plain classes for feature extraction and cannot be fine-tuned.

Full details: [Fine-tuning overview](../finetuning/overview.md), [MACE](../finetuning/mace.md), [UMA / FairChem](../finetuning/uma.md), [UPET / PET](../finetuning/upet.md).
