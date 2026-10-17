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
    predictions = self.head(features, graph)
    # ...plus duck-typed side channels (see below)
    return predictions
```

The trainer only sees the resulting dictionary. Loss components in `training.losses` are matched to keys in that dictionary by name, and the metric logger reads the standard keys too: `energy`, `forces`, `num_atoms`, and optional diagnostics such as `layer_energies`.

`GOALModule` also reads some optional attributes from the backbone, purely by duck typing. Backbones that do not define them are skipped:

| Backbone attribute | Effect |
|---|---|
| `last_aux_loss` (grad-enabled scalar) | copied to `predictions["aux_loss"]` and added to the total training loss (e.g. the ARACE adaptive depth gate) |
| `last_gate_scores` | stacked into `predictions["gate_scores"]` and logged per round under `*_gate/` |
| `last_round_energies` / `predictions["layer_energies"]` | logged as `energy_round_{L}` (ARACE per-round energy decomposition) |

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
| `elements` (overwrites the config value) | always | `simurgh`, `simurgh_ace_first`, `simurgh_arace`, `simurgh_monolithic`, `monolithic_arace` **only** |
| `atomic_energies.values` (least-squares $E_0$) | `atomic_energies.mode: dataset` (or `compute_from_dataset: true`) with `values: null` | any backbone with an `atomic_energies` kwarg |
| `scale` | `atomic_energies.mode` is `dataset` or `provided`, and `scale` is null or missing | any backbone with an `atomic_energies` kwarg |
| `dressing_kwargs.avg_num_neighbors` | it is null | backbones with a `dressing_kwargs` section (legacy SIMURGH) |
| `avg_num_neighbors` (top level) | it is null | `simurgh_arace` **only** |
| `artisan.pair_counts` / `artisan_config.pair_counts` | `rare_artisan.enabled: true` and counts are null | any backbone with that section |

> [!WARNING]
> `nf_simurgh`, `simurgh_simplified_*` and `simurgh_ablation` are **not** in the `elements` list, and `monolithic_arace` is not in the `avg_num_neighbors` list. For these models, set the values in the config yourself. The shipped configs already do this, with comments explaining why. If `avg_num_neighbors` is left `null` on them, the aggregation is simply not normalised. You get no error.

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

Status key: **default** = the recommended production model; **legacy** = kept working, no longer the default; **research** = self-contained variant meant for A/B comparison; **experimental** = new research model, under active development; **ablation** = frozen model for the ablation study only; **baseline / reference** = educational or comparison models.

All files are relative to `src/goal/ml/nn/models/`. All configs are under `src/goal/configs/ml/`.

| Registry name | Kind | Status | Forces | File | Example config |
|---|---|---|---|---|---|
| `simurgh_arace` | modular | **default** | autograd (head) | [`simurgh/backbone_arace.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/simurgh/backbone_arace.py) | [`simurgh_gmd26.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_gmd26.yaml), [`simurgh_gmd26_hpo.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_gmd26_hpo.yaml) |
| `simurgh` (alias `simurgh_ace_first`) | modular | legacy | autograd / pairwise / hybrid (head) | [`simurgh/backbone.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/simurgh/backbone.py) | [`simurgh_ace_first_gmd26.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_ace_first_gmd26.yaml), [`simurgh_md17.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/simurgh_md17.yaml) |
| `simurgh_monolithic` | monolithic | legacy | autograd or pairwise (`forces_mode`) | [`simurgh/monolithic.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/simurgh/monolithic.py) | none shipped (same keys as legacy `simurgh`, plus `forces_mode`, `correction_weight`; `head: null`) |
| `monolithic_arace` | monolithic | research | autograd | [`simurgh/arace.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/simurgh/arace.py) | [`monolithic_arace_gmd26.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/monolithic_arace_gmd26.yaml) |
| `nf_simurgh` | monolithic | experimental | autograd | [`simurgh/nf_simurgh.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/simurgh/nf_simurgh.py) | `nf_simurgh_{gmd26,gmd26_typecond,gmd26_FF,md17,md17_benzene,md22,qm9,omol25}.yaml` |
| `simurgh_simplified_monolithic` | monolithic | experimental | autograd | [`simurgh/simurgh_simplified_monolithic.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/simurgh/simurgh_simplified_monolithic.py) | `simurgh_simplified_monolithic_{gmd26,md17}.yaml` |
| `simurgh_simplified_routed` | monolithic | experimental | autograd | [`simurgh/simurgh_simplified_routed.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/simurgh/simurgh_simplified_routed.py) | `simurgh_simplified_routed_{gmd26,md17}.yaml` |
| `simurgh_ablation` | modular | ablation | autograd (head) | [`ablations/backbone.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/ablations/backbone.py) | [`ablations/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/configs/ml/ablations) (numbered rungs) |
| `hyperspec` | modular | baseline (placeholder) | autograd (head; fixed Oct 2026) or direct | [`hyperspec.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/hyperspec.py) | [`hyperspec_md17.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/hyperspec_md17.yaml) |
| `invariant_gnn` | modular (invariant) | baseline | autograd (head; fixed Oct 2026) | [`invariant.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/invariant.py) | none shipped |
| `monolithic_example` | monolithic | reference | autograd | [`monolithic_example.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/monolithic_example.py) | none shipped |
| `mace_finetune` | monolithic | foundation | autograd | [`foundation/mace.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/foundation/mace.py) | [`finetune/mace.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/mace.yaml) |
| `uma_finetune` | monolithic | foundation | autograd | [`foundation/uma.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/foundation/uma.py) | [`finetune/uma.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/uma.yaml) |
| `simurgh_lora` | monolithic | fine-tuning | autograd (base model's) | [`goal/ml/adapters/simurgh_lora.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/adapters/simurgh_lora.py) | [`finetune/simurgh_lora.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/simurgh_lora.yaml) ([LoRA fine-tuning](../../integrations/lora_finetune.md)) |

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

- **Production force field on small- to medium-element-count chemistry:** `simurgh_arace` (start from `goal-train`, which runs `simurgh_gmd26.yaml` by default).
- **Same physics in a self-contained model, e.g. for an A/B comparison:** `monolithic_arace`.
- **Long-range coupling between molecular fragments, or large element sets with rare-pair routing (OMol25):** `nf_simurgh` (experimental). It needs fragment labels.
- **A routed vs plain-stack comparison on an identical skeleton:** the two `simurgh_simplified_*` models.
- **Measuring what each SIMURGH component contributes:** `simurgh_ablation` with the numbered configs.
- **Starting from a pre-trained foundation potential:** `mace_finetune` / `uma_finetune`. See [Fine-tuning](../finetuning/overview.md).
- **Writing your own architecture:** [Writing a new model](custom-model.md).

```bash
pixi run -e cuda goal-train                                                         # simurgh_arace on GMD-26 (default config)
pixi run -e cuda goal-train --config-name simurgh_ace_first_gmd26                   # legacy ACE-first
pixi run -e cuda goal-train --config-name monolithic_arace_gmd26                    # monolithic ARACE
pixi run -e cuda goal-train --config-name nf_simurgh_gmd26                          # NF_SIMURGH
pixi run -e cuda goal-train --config-name simurgh_simplified_routed_gmd26           # SIMURGH Simplified (routed)
pixi run -e cuda goal-train-ml --config-dir src/goal/configs/ml --config-name ablations/01_gmd26_base   # ablation base rung
```

## Further reading

- [SIMURGH](simurgh.md): ARACE and the legacy ACE-first architecture
- [NF_SIMURGH](nf-simurgh.md), [SIMURGH Simplified](simurgh-simplified.md), [Ablation study](ablation-study.md)
- [Other backbones](other-backbones.md): HyperSpec, InvariantGNN, foundation models
- [Heads](heads.md) and [Writing a new model](custom-model.md)
