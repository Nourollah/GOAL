# Heads

A head turns a modular backbone's `NodeFeatures` into the property dictionary that the loss and metrics read. Heads live in [`src/goal/ml/nn/heads/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/ml/nn/heads) and are registered in `HEAD_REGISTRY`. Any new file in that folder with an `@HEAD_REGISTRY.register("name")` decorator is discovered automatically. Monolithic models set `head: null` and skip this layer entirely (see [Overview](overview.md)).

## Summary

| `head.name` | Class | Predicts (output keys) | Forces | Honours `node_energies`? |
|---|---|---|---|---|
| `energy` | `EnergyHead` | `energy`, `num_atoms` | none | yes |
| `energy_forces` | `EnergyForcesHead` | `energy`, `forces`, `num_atoms` | conservative, $-\partial E/\partial\mathbf r$ | yes |
| `dual_forces` | `DualForcesHead` | `energy`, `forces`, `num_atoms` | `autograd` / `direct` / `hybrid` / `pairwise` | yes |
| `direct_forces` | `DirectForcesHead` | `energy`, `forces`, `num_atoms` | direct `1x1o` projection (non-conservative) | no |
| `stress` | `StressHead` | `energy`, `stress`, `num_atoms` | (uses $\partial E/\partial\mathbf r$ internally) | no |
| `dipole` | `DipoleHead` | `dipole`, `charges`, `total_charge`, `num_atoms` | none | n/a |
| `scalar` | `ScalarHead` | `<property_name>` | none | n/a |
| `multi` | `MultiHead` | union of its sub-heads' keys | per sub-head | per sub-head |

Every head except `multi` takes `irreps_in` (the backbone's output irreps, as a string) and `hidden_dim` (the width of its `ScalarReadout` MLP, default 64). `ScalarReadout` keeps only the `0e` channels of `irreps_in` (via an `EquivariantLinear`) and then applies `Linear → SiLU → Linear` to give one scalar per atom.

**`node_energies`.** Backbones that already compute per-atom energies (all SIMURGH backbones and `simurgh_ablation`) put them in `NodeFeatures.node_energies`. `energy`, `energy_forces` and `dual_forces` then **sum those directly**, and their own `ScalarReadout` is unused. `direct_forces` and `stress` always use their own readout, so paired with a SIMURGH backbone they ignore the artisan energies.

Keep `irreps_in` consistent with the backbone by interpolating it in the YAML. For example, the default config uses:

```yaml
head:
  name: energy_forces
  irreps_in: ${model.backbone.artisan.hidden_irreps}
  hidden_dim: 128
```

## Conservative vs direct forces

**Conservative** heads (`energy_forces`, and `dual_forces` in `autograd` mode) compute $\mathbf F_i = -\partial E/\partial \mathbf r_i$ with `torch.autograd.grad` (using `create_graph=True` during training, so the force loss can be backpropagated). The forces are then an exact gradient field: energy is conserved in MD, $\sum_i \mathbf F_i = 0$ for a translation-invariant energy, and rotating the structure rotates the forces. The costs are a second backward pass and the extra memory needed to differentiate through the force computation.

For this to work, the backbone must build its edge geometry from `graph.pos` and make sure `graph.pos.requires_grad` is set. The SIMURGH backbones do this through `differentiable_edges`. **If `graph.pos.requires_grad` is false, these heads silently return zero forces.** See the warning on [Other backbones](other-backbones.md) and the template in [Writing a new model](custom-model.md).

**Direct** heads project the backbone's `l = 1` features straight onto a `1x1o` vector per atom. This is cheaper, because no derivative pass is needed, and it is still rotation-equivariant. But the result is not a gradient field: energy is not conserved in NVE MD and $\sum_i \mathbf F_i$ is not exactly zero. It can still fit clean DFT forces well. It requires `1o` channels in `irreps_in`.

**Hybrid** (`dual_forces`, `mode: hybrid`) keeps the conservative term as the leading contribution and adds a small learnable correction:

$$\mathbf F = -\nabla_{\mathbf r} E + w\,\mathbf F_\text{direct}$$

With a small $w$ (e.g. 0.05) the model stays close to energy-conserving while the direct term can absorb systematic errors.

Use `energy_forces` for anything that will run MD. Use `direct`/`hybrid` when force-matching accuracy or speed matters more than strict conservation. Monitor `newton_violation` (logged automatically) to see how far a direct head is from momentum conservation.

## Head reference

### `energy`

The total energy is the per-graph sum of per-atom contributions. Use it for energy-only datasets, or as a sub-head of `multi`. Keys: `irreps_in`, `hidden_dim`.

### `energy_forces`

Energy as above, plus conservative autograd forces. Keys: `irreps_in`, `hidden_dim`, `compute_stress` (default `false`). This is the default head for SIMURGH ARACE.

> [!WARNING]
> `compute_stress: true` adds `"stress"` to `output_keys`, but the current `forward` **never computes stress**: the returned dict only holds `energy`, `forces` and `num_atoms`. A `stress` loss configured on top would fail to find its prediction. For stress training, use the `stress` head (inside `multi` if you also need forces).

### `dual_forces`

| Key | Default | Meaning |
|---|---|---|
| `irreps_in` | | must contain `1o` channels for `direct`/`hybrid` |
| `hidden_dim` | 64 | readout width |
| `mode` | `autograd` | `autograd`, `direct`, `hybrid`, `pairwise` |
| `correction_weight` | 0.1 | $w$ in `hybrid` mode |
| `pairwise_correction_weight` | 0.0 | in `pairwise` mode: `0` gives pure pairwise forces; `>0` gives $\mathbf F_\text{autograd} + w\,\mathbf F_\text{pairwise}$ |

`pairwise` takes `features.node_forces` from the backbone. Newton's third law is then exact by construction. Only the legacy `simurgh` backbone with `compute_pairwise_forces: true` provides `node_forces`. `simurgh_arace` does not, so `pairwise` raises there. The shipped legacy config `simurgh_ace_first_gmd26.yaml` uses `mode: hybrid`.

```yaml
head:
  name: dual_forces
  irreps_in: ${model.backbone.artisan.hidden_irreps}
  hidden_dim: 128
  mode: hybrid
  correction_weight: 0.05
```

### `direct_forces`

Energy comes from the head's own scalar readout, and forces from `EquivariantLinear(irreps_in → 1x1o)`. Keys: `irreps_in` (must contain `1o`), `hidden_dim`. This is equivalent to `dual_forces` with `mode: direct`, except that `dual_forces` honours `node_energies`.

### `stress`

This head predicts energy and a $3\times3$ stress per structure. Stress is a virial built from edge displacement vectors and the energy gradient at each edge's receiving atom, summed per graph and divided by $|\det(\text{cell})|$. Edge vectors are taken from `graph.edge_attr` when stored, otherwise recomputed from positions. Keys: `irreps_in`, `hidden_dim`. The head needs `graph.pos.requires_grad`; without it, no `stress` key is produced. It uses its own readout for the energy. Use it for periodic systems with virial labels, paired with the `stress` loss.

### `dipole`

A scalar readout predicts per-atom partial charges $q_i$, and the dipole is $\boldsymbol\mu = \sum_i q_i \mathbf r_i$, a vector built from invariant charges. The head also returns `charges` and `total_charge`, so you can add a charge-neutrality term (the `charge` loss compares `total_charge` to the target, or to 0 if no target is given). Keys: `irreps_in`, `hidden_dim`.

```yaml
training:
  losses:
    - name: dipole
      weight: 1.0
    - name: charge          # pushes total_charge toward the label (or 0)
      weight: 0.1
```

> [!NOTE]
> $\sum_i q_i \mathbf r_i$ is only origin-independent when $\sum_i q_i = 0$. That is why the `charge` loss is worth including for neutral molecules.

### `scalar`: any per-structure scalar

Use this head for HOMO, LUMO, band gap, polarisability and similar targets without writing a new class.

| Key | Default | Meaning |
|---|---|---|
| `irreps_in`, `hidden_dim` | `—`, 64 | as above |
| `property_name` | `"scalar"` | output key; it must match the target key on the graph and the loss's `property_name` |
| `reduction` | `"mean"` | `mean` for intensive properties, `sum` for extensive ones; anything else raises |

```yaml
head:
  name: scalar
  irreps_in: "128x0e"
  hidden_dim: 64
  property_name: band_gap
  reduction: mean

training:
  losses:
    - name: scalar_property
      property_name: band_gap
      weight: 1.0
      # per_atom: false   # true divides by the atom count before the loss
```

### `multi`: several properties from one backbone

`MultiHead` takes a list of head configs under `heads:` and builds each one from `HEAD_REGISTRY`. Every sub-head receives the same `NodeFeatures` and has its own readout MLP. Their output dicts are merged. Two sub-heads producing the same key raise `ValueError` at construction (only `num_atoms` may be shared). So you cannot, for example, combine `energy_forces` with `stress`, because both output `energy`.

Example: energy, forces, HOMO and LUMO on top of the default ARACE backbone. The `homo`/`lumo` targets must be present on your graphs.

```yaml
model:
  backbone:
    name: simurgh_arace
    # ... as in src/goal/configs/ml/simurgh_gmd26.yaml
  head:
    name: multi
    heads:
      - name: energy_forces            # sums the backbone's node_energies
        irreps_in: ${model.backbone.artisan.hidden_irreps}
        hidden_dim: 128
      - name: scalar
        irreps_in: ${model.backbone.artisan.hidden_irreps}
        hidden_dim: 64
        property_name: homo
        reduction: mean
      - name: scalar
        irreps_in: ${model.backbone.artisan.hidden_irreps}
        hidden_dim: 64
        property_name: lumo
        reduction: mean

training:
  losses:
    - name: energy
      weight: 1.0
    - name: forces
      weight: 10.0
    - name: scalar_property
      property_name: homo
      weight: 1.0
    - name: scalar_property
      property_name: lumo
      weight: 1.0
```

Energy, forces and a dipole from one backbone with `l = 1` features:

```yaml
head:
  name: multi
  heads:
    - name: energy_forces
      irreps_in: ${model.backbone.artisan.hidden_irreps}
    - name: dipole
      irreps_in: ${model.backbone.artisan.hidden_irreps}
```

> [!TIP]
> Loss components are labelled by their `name`. Two `scalar_property` entries therefore share the label `scalar_property`, and a stage curriculum (`training.stages[*].loss_weights`) cannot weight them separately. If you need per-property curricula, give each property its own run or a custom loss class. Loss options are covered in [Loss functions](../training/losses.md).

## Overriding the head from the CLI

```bash
pixi run -e cuda goal-train model.head.name=dual_forces +model.head.mode=hybrid +model.head.correction_weight=0.05
```

Use `+key=value` for keys that are not already in the config, and `~key` to remove one.

## Writing a head

A head is any `nn.Module` with `forward(features: NodeFeatures, graph: AtomicGraph) -> dict[str, Tensor]` and an `output_keys` property (the `TaskHead` protocol in `nn/models/base.py`). Put the file in `src/goal/ml/nn/heads/`, decorate the class with `@HEAD_REGISTRY.register("my_head")`, and it is available as `head.name: my_head`. Return `num_atoms` alongside the energy if you want per-atom energy metrics. Follow the existing heads: prefer `features.node_energies` when it is present, and fall back to a `ScalarReadout`.
