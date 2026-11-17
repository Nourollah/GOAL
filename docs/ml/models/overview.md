# Models: overview and model zoo

Every model in `goal.ml` is one of two kinds. A **modular** backbone produces per-atom features and a separate **head** turns them into properties. A **monolithic** model goes straight from the graph to the property dictionary. This page covers how the two kinds plug into the training loop, how the registry finds them, and lists every model that ships with GOAL.

## Modular vs monolithic

| | Modular | Monolithic |
|---|---|---|
| `forward(graph)` returns | `NodeFeatures` (per-atom irreps features, optionally `node_energies` / `node_forces`) | `dict[str, Tensor]` (`energy`, `forces`, `num_atoms`, …) |
| Protocol in [`nn/models/base.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/base.py) | `EquivariantBackbone`, `InvariantBackbone`, `Backbone`, `FeatureExtractorBackbone` | `MonolithicModel` |
| Config | `model.backbone` **and** a `model.head` block | `model.backbone` and `head: null` |
| Forces | computed by the head (autograd, direct, or hybrid; see [Heads](heads.md)) | computed inside the model |
| Typical use | swap heads (energy+forces, dipole, multi-property) on one feature extractor | self-contained research architectures, foundation-model fine-tuning |

The protocols are structural (`typing.Protocol` with `@runtime_checkable`), so you do not inherit from anything. A class is monolithic if it has a `forward` and an `output_keys` property. A class is an `EquivariantBackbone` if it has `forward`, `irreps_out` and `num_interactions`.

> [!IMPORTANT]
> The training CLI checks that the head setting matches the kind of backbone (`_build_head` in [`cli/train.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/cli/train.py)). A backbone that satisfies `MonolithicModel` **must** have `head: null`. Any other backbone **must** have a `head:` block. A mismatch raises `ValueError` at startup rather than failing silently during training.

## How `GOALModule` composes the model

[`GOALModule`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/training/module.py) is the single Lightning module for every model. Its forward pass is:

```python
def forward(self, graph):
    if self.head is None:                 # monolithic
        return self.backbone(graph)
    features = self.backbone(graph)       # NodeFeatures
    return self.head(features, graph)
```

The trainer only sees the resulting dictionary. Loss components in `training.losses` are matched to keys in that dictionary by name, and the metric logger reads the standard keys too: `energy`, `forces` and `num_atoms`.

A model may also return its own regularisation term as `aux_loss` (already weighted), with an optional `aux_terms` dictionary breaking it down. `GOALModule` adds `aux_loss` to the total training loss and logs it under `*_gate/`; the parts in `aux_terms` are logged, not added again. SIMURGH uses this for multi-scale fragment consistency, denoising and its other training-only terms.

Validation and test steps wrap the forward pass in `torch.enable_grad()`, so autograd forces still work there.

### What `goal-train` builds, step by step

1. Build the datamodule and run `setup("fit")`.
2. `BACKBONE_REGISTRY.get(cfg.model.backbone.name)` and gather every other key under `model.backbone` as constructor kwargs.
3. Fill in statistics from the training set (see the next section).
4. `backbone = backbone_cls(**kwargs)`.
5. `_build_head`: `None` for a monolithic backbone, otherwise `HEAD_REGISTRY.get(model.head.name)(**head_kwargs)`.
6. `GOALModule(backbone, head, loss, cfg)`.

### Dataset-derived constructor arguments

Some constructor arguments are computed from the training split in step 3. Which ones depends on the backbone name, so read this table before writing a config:

| Injected kwarg | When | Backbones |
|---|---|---|
| `elements` (overwrites the config value) | always | `simurgh_dev` and its `simurgh_dev_abl_*` arms **only** |
| `atomic_energies.values` (least-squares $E_0$) | `atomic_energies.mode: dataset` (or `compute_from_dataset: true`) with `values: null` | any backbone with an `atomic_energies` kwarg |
| `scale` | `atomic_energies.mode` is `dataset` or `provided`, and `scale` is null or missing; or `scale: force_rms` (RMS of the training forces) | any backbone with an `atomic_energies` kwarg; `force_rms` for `simurgh_dev` |
| `avg_num_neighbors` | it is null | `simurgh_dev` **only** |
| `atom_statistics` | the backbone requests descriptor statistics | `simurgh_dev` with `attention.pan_gate` |

> [!WARNING]
> Any other backbone gets `elements` and `avg_num_neighbors` from the config only. Set them yourself, and make a new model raise when `avg_num_neighbors` is missing rather than silently skipping the normalisation.

## The registry

[`src/goal/ml/registry.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/registry.py) defines the global registries `MODEL_REGISTRY`, `BACKBONE_REGISTRY`, `HEAD_REGISTRY`, `LOSS_REGISTRY`, `DATASET_REGISTRY`, `TRANSFORM_REGISTRY` and `STRATEGY_REGISTRY`. The training CLI resolves `model.backbone.name` in **`BACKBONE_REGISTRY`** and `model.head.name` in `HEAD_REGISTRY`. Every built-in model is registered in both `MODEL_REGISTRY` and `BACKBONE_REGISTRY`.

Ways to register a class:

| Mechanism | Used for |
|---|---|
| `@BACKBONE_REGISTRY.register("name")` decorator | every built-in model and head |
| `BACKBONE_REGISTRY.register_lazy("name", "pkg.module:Class")` | registration without importing anything (e.g. the MACE/UMA adapters) |
| `auto_discover(package)` | `goal.ml.nn.models`, `goal.ml.nn.heads` and `goal.ml.data.datasets` import every submodule, recursively, so the decorators run. Put a new file in the folder and it is registered. |
| `register_instance(name, cls)` | programmatic registration |
| `discover_plugins(group)` | entry-point plugins; the method exists, but **no CLI entry point calls it today** |

Registering a name twice raises `KeyError`. A lookup miss also raises `KeyError`, and the message lists every available name. That is the quickest way to see what is registered:

```python
import goal.ml.nn.models, goal.ml.nn.heads   # triggers auto-discovery
from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY

print(BACKBONE_REGISTRY.list_available())
print(HEAD_REGISTRY.list_available())
```

## Model zoo

Status key: **default** = the recommended model; **ablation** = one mechanism added to the default model for the ablation programme; **baseline / reference** = educational or comparison models; **foundation** = fine-tuning of a pre-trained potential.

All files are relative to `src/goal/ml/nn/models/`. All configs are under `src/goal/configs/ml/`.

| Registry name | Kind | Status | Forces | File | Example config |
|---|---|---|---|---|---|
| `simurgh_dev` | monolithic | **default** (in development) | autograd | [`simurgh/dev/monolithic.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/simurgh/dev/monolithic.py) | [`simurgh_dev_gmd26_FD.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_dev_gmd26_FD.yaml) and the other `simurgh_dev_*.yaml` |
| `simurgh_dev_abl_<mechanism>` | monolithic | ablation | autograd | [`simurgh/dev/ablations/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/ml/nn/models/simurgh/dev/ablations) | [`ablations/simurgh_dev/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/configs/ml/ablations/simurgh_dev) |
| `hyperspec` | modular | baseline (placeholder) | autograd (head; fixed Oct 2026) or direct | [`hyperspec.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/hyperspec.py) | [`hyperspec_md17.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/hyperspec_md17.yaml) |
| `invariant_gnn` | modular (invariant) | baseline | autograd (head; fixed Oct 2026) | [`invariant.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/invariant.py) | none shipped |
| `monolithic_example` | monolithic | reference | autograd | [`monolithic_example.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/monolithic_example.py) | none shipped |
| `mace_finetune` | monolithic | foundation | autograd | [`foundation/mace.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/foundation/mace.py) | [`finetune/mace.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/mace.yaml) |
| `uma_finetune` | monolithic | foundation | autograd | [`foundation/uma.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/foundation/uma.py) | [`finetune/uma.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/uma.yaml) |

`mace_finetune` and `uma_finetune`, and the lazy entries below, are registered **only** in `BACKBONE_REGISTRY`. "Autograd" means conservative forces $\mathbf F=-\partial E/\partial\mathbf r$; every model in the table predicts forces.

<details>
<summary>Lazy backbone entries for feature extraction (not trainable models)</summary>

These names are registered with `register_lazy` and are meant for feature extraction and analysis. They are not targets for `goal-train`.

| Name | Target |
|---|---|
| `mace-large`, `mace-medium`, `mace-small` | `goal.ml.adapters.mace:MACEAdapter` |
| `uma-small` | `goal.ml.adapters.fairchem:UMAAdapter` |
| `mace-large-final`, `mace-large-multiscale`, `mace-large-frozen` | builders in `goal.ml.utils.extraction` (registered when `goal.ml.utils` is imported) |

The adapters are plain classes, not `nn.Module`s, so their weights cannot be fine-tuned. Use `mace_finetune` / `uma_finetune` for training. See [Feature extraction](../features/feature-extraction.md).

</details>

### Which model should I use?

- **Native force field:** `simurgh_dev` (start from `goal-train`, which runs `simurgh_dev_gmd26_FD.yaml` by default). See [SIMURGH (development)](simurgh-dev.md).
- **Measuring what one SIMURGH mechanism contributes:** the `ablations/simurgh_dev/` configs, one key or one `simurgh_dev_abl_*` arm each.
- **Starting from a pre-trained foundation potential:** `mace_finetune` / `uma_finetune`. See [Fine-tuning](../finetuning/overview.md).
- **Writing your own architecture:** [Writing a new model](custom-model.md).

```bash
pixi run -e cuda goal-train                                                     # simurgh_dev on GMD-26 FragmentDuplication (default)
pixi run -e cuda goal-train --config-name simurgh_dev_rmd17_aspirin             # another task
pixi run -e cuda goal-train --config-name ablations/simurgh_dev/s05_num_rbf_16  # one ablation arm
```

## Further reading

- [SIMURGH (development)](simurgh-dev.md)
- [Other backbones](other-backbones.md): HyperSpec, InvariantGNN, foundation models
- [Heads](heads.md) and [Writing a new model](custom-model.md)
