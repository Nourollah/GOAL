# Writing a new model

This page shows how to add your own architecture to `goal.ml`, either as a **modular backbone** (paired with any head) or as a **monolithic model** (returns the property dict itself). Both examples are complete. They were run against the current code: the backbone through the `energy_forces` head, the monolithic model through the CLI's head-wiring check.

The interfaces are the protocols in [`src/goal/ml/nn/models/base.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/base.py). They are structural `typing.Protocol`s, so you implement the methods and properties and do not inherit from anything:

| Protocol | Must provide |
|---|---|
| `EquivariantBackbone` | `forward(graph) -> NodeFeatures`, `irreps_out` (property, `e3nn.o3.Irreps`), `num_interactions` (property) |
| `InvariantBackbone` | `forward(graph) -> NodeFeatures` with `"Nx0e"` irreps, `hidden_dim`, `num_interactions` |
| `FeatureExtractorBackbone` | the above plus `extract_features(graph, layer)`, `num_layers`, `irreps_at_layer(layer)` |
| `MonolithicModel` | `forward(graph) -> dict[str, Tensor]`, `output_keys` (property) |

> [!IMPORTANT]
> The CLI decides between modular and monolithic with `isinstance(backbone, MonolithicModel)`, which is a check for a `forward` method plus an `output_keys` attribute. **Do not give a modular backbone an `output_keys` property.** If you do, it is treated as monolithic and `goal-train` demands `head: null`.

## The inputs you get

`forward` receives an `AtomicGraph` ([`data/graph.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/data/graph.py)), a PyG `Data`/`Batch`:

| Field | Shape | Notes |
|---|---|---|
| `graph.pos` | `(N, 3)` | positions; usually float64 from the dataset |
| `graph.atomic_numbers` (alias `graph.z`) | `(N,)` int64 | |
| `graph.edge_index` | `(2, E)` | directed neighbour pairs within `data.cutoff` |
| `graph.unit_shifts`, `graph.cell`, `graph.pbc` | | periodic images; handled for you by `differentiable_edges` |
| `graph.batch` | `(N,)` or `None` | structure index per atom (`None` for a single graph) |
| `graph.fragment_index` | `(N,)` or `None` | only when `data.compute_fragment_index: true` |
| `graph.edge_vectors` / `graph.edge_lengths` | `(E, 3)` / `(E,)` or `None` | **`None` on the standard path, and never connected to `graph.pos`. Do not use them.** |

`NodeFeatures` has four fields:

- `node_feats` `(N, irreps.dim)`
- `irreps` (string)
- `node_energies` `(N,)`, optional: if set, the energy heads sum it directly instead of using their own readout
- `node_forces` `(N, 3)`, optional: read by `dual_forces` in `mode: pairwise`

## Rule 1: build the geometry from `graph.pos`

Conservative force heads compute $-\partial E/\partial(\texttt{graph.pos})$. They return **zero forces, silently**, if `graph.pos.requires_grad` is false. Every model must therefore:

1. make sure `graph.pos` requires grad (detach, call `requires_grad_(True)`, and write it back to `graph.pos`), and
2. recompute edge vectors from those positions, including periodic shifts. `goal.ml.nn.models.simurgh.geometry.differentiable_edges(graph, positions)` does this and returns `(edge_vectors, edge_lengths)`.

Monolithic models also compute forces themselves. They must wrap the work in `torch.enable_grad()`, because validation can run with gradients disabled. `GOALModule` enables grad around its own validation forward, but standalone inference may not.

## A new modular backbone

Save this as `src/goal/ml/nn/models/tiny_conv.py`. It is a small equivariant convolution network:

```python
"""TinyConv — a minimal equivariant message-passing backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, spherical_harmonics
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.models.simurgh.geometry import differentiable_edges
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.nn.primitives.radial import BesselBasis, PolynomialEnvelope, RadialMLP
from goal.ml.nn.primitives.tp import WeightedTensorProduct
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


@MODEL_REGISTRY.register("tiny_conv")
@BACKBONE_REGISTRY.register("tiny_conv")
class TinyConv(nn.Module):
    """Embedding -> N x (equivariant convolution + residual) -> NodeFeatures."""

    def __init__(
        self,
        num_elements: int = 120,
        embedding_dim: int = 32,
        hidden_irreps: str = "32x0e + 16x1o + 8x2e",
        num_interactions: int = 2,
        cutoff: float = 5.0,
        num_rbf: int = 8,
        radial_hidden: int = 32,
        avg_num_neighbors: float = 10.0,
    ) -> None:
        super().__init__()
        self._irreps: Irreps = Irreps(hidden_irreps)
        self._num_interactions: int = num_interactions
        self._inv_avg_nn: float = 1.0 / avg_num_neighbors
        lmax: int = max(ir.l for _, ir in self._irreps)
        self._irreps_sh: Irreps = Irreps.spherical_harmonics(lmax)

        self.embedding = AtomicNumberEmbedding(num_elements, embedding_dim)
        self.input_linear = EquivariantLinear(f"{embedding_dim}x0e", self._irreps)
        self.rbf = BesselBasis(num_basis=num_rbf, cutoff=cutoff)
        self.envelope = PolynomialEnvelope(cutoff=cutoff)
        self.tps = nn.ModuleList()
        self.radials = nn.ModuleList()
        self.updates = nn.ModuleList()
        for _ in range(num_interactions):
            tp = WeightedTensorProduct(self._irreps, self._irreps_sh, self._irreps)
            self.tps.append(tp)
            self.radials.append(RadialMLP(num_rbf, radial_hidden, tp.weight_numel))
            self.updates.append(EquivariantLinear(self._irreps, self._irreps))

    # --- protocol properties -------------------------------------------
    @property
    def irreps_out(self) -> Irreps:
        return self._irreps

    @property
    def num_interactions(self) -> int:
        return self._num_interactions

    # --- forward ---------------------------------------------------------
    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        # Forces come from the HEAD as -dE/d(graph.pos), so the edge
        # geometry must be rebuilt from a grad-tracking graph.pos here.
        pos: torch.Tensor = graph.pos
        if not pos.requires_grad:
            pos = pos.detach().requires_grad_(True)
            graph.pos = pos
        edge_vec, edge_len = differentiable_edges(graph, pos)  # (E, 3), (E,)
        row, col = graph.edge_index

        h = self.input_linear(self.embedding(graph.atomic_numbers))  # (N, irreps.dim)
        sh = spherical_harmonics(
            self._irreps_sh, edge_vec.to(h.dtype), normalize=True, normalization="component"
        )
        radial_in = self.rbf(edge_len).to(h.dtype)  # (E, num_rbf)
        env = self.envelope(edge_len).to(h.dtype).unsqueeze(-1)  # (E, 1)

        for tp, radial, update in zip(self.tps, self.radials, self.updates):
            w = radial(radial_in) * env  # (E, weight_numel)
            msg = tp(h[row], sh, w)  # (E, irreps.dim)
            agg = scatter(msg, col, dim=0, dim_size=h.shape[0], reduce="sum")
            h = h + update(agg * self._inv_avg_nn)

        return NodeFeatures(node_feats=h, irreps=str(self._irreps))
```

Some notes on the choices in this code:

- **Primitives.** `WeightedTensorProduct(irreps_in1, irreps_in2, irreps_out)` exposes `weight_numel`, and its `forward(x1, x2, weight)` takes per-edge weights. `RadialMLP(num_basis, hidden_dim, num_out)` produces those weights. `BesselBasis` outputs float64, which is why it is cast to the model dtype. `EquivariantLinear(irreps_in, irreps_out, biases=True)` is the only map allowed to mix channels of the same $l$.
- **No `node_energies`.** The head's `ScalarReadout` makes per-atom energies from the `0e` channels. If your model computes per-atom energies itself (as SIMURGH does), put them in `NodeFeatures(node_energies=...)`, and `energy`, `energy_forces` and `dual_forces` will use them.
- **Dtype.** Datasets usually ship float64 positions while parameters are float32. Cast geometry-derived tensors to the dtype of the features, as shown.

Config (copy a self-contained file such as `src/goal/configs/ml/simurgh_gmd26.yaml` and replace the `model` section):

```yaml
model:
  backbone:
    name: tiny_conv
    num_elements: 9            # > max Z in your data
    embedding_dim: 32
    hidden_irreps: "32x0e + 16x1o + 8x2e"
    num_interactions: 2
    cutoff: ${data.cutoff}     # keep the graph and the model in sync
    num_rbf: 8
    radial_hidden: 32
    avg_num_neighbors: 12.0    # NOT auto-injected for new names; compute it once for your data
  head:
    name: energy_forces
    irreps_in: ${model.backbone.hidden_irreps}
    hidden_dim: 64
```

```bash
pixi run -e cuda goal-train --config-name my_tiny_conv            # src/goal/configs/ml/my_tiny_conv.yaml
pixi run -e cuda goal-train --config-name my_tiny_conv model.head.name=dual_forces +model.head.mode=hybrid
```

## A new monolithic model

A monolithic model owns everything, including forces. Here `TinyConv` is reused as the feature extractor, with a readout and a learnable per-element reference energy added. Save it as `src/goal/ml/nn/models/tiny_potential.py`:

```python
"""TinyPotential — a minimal monolithic model reusing TinyConv."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.readout import ScalarReadout
from goal.ml.nn.models.tiny_conv import TinyConv
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


@MODEL_REGISTRY.register("tiny_potential")
@BACKBONE_REGISTRY.register("tiny_potential")
class TinyPotential(nn.Module):
    """Graph -> {energy, forces, num_atoms}; configure with ``head: null``."""

    def __init__(self, hidden_dim: int = 64, **conv_kwargs) -> None:
        super().__init__()
        self.conv = TinyConv(**conv_kwargs)
        self.readout = ScalarReadout(self.conv.irreps_out, hidden_dim=hidden_dim)
        # Per-element reference energy, learned from zero.
        self.e0 = nn.Embedding(conv_kwargs.get("num_elements", 120), 1)
        nn.init.zeros_(self.e0.weight)

    @property
    def output_keys(self) -> list[str]:
        return ["energy", "forces"]

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        # Validation/test may run without grad; forces need a local grad tape.
        with torch.enable_grad():
            pos = graph.pos.detach().clone().requires_grad_(True)
            graph.pos = pos  # TinyConv builds its edges from graph.pos
            feats = self.conv(graph)
            z = graph.atomic_numbers
            node_e = self.readout(feats.node_feats).squeeze(-1) + self.e0(z).squeeze(-1)
            batch = graph.batch if graph.batch is not None else torch.zeros_like(z)
            energy = scatter(node_e, batch, dim=0, reduce="sum")  # (B,)
            (grad,) = torch.autograd.grad(
                energy.sum(), pos, create_graph=self.training, retain_graph=True
            )
        num_atoms = scatter(torch.ones_like(node_e), batch, dim=0, reduce="sum")
        return {"energy": energy, "forces": -grad, "num_atoms": num_atoms}
```

The parts that matter:

- **`create_graph=self.training`.** In training, the force loss has to backpropagate through the force computation. In evaluation, building that second-order graph only wastes memory.
- **`num_atoms`.** Return it so the per-atom energy loss and metrics work. Extra diagnostic keys are allowed (e.g. `layer_energies` `(L, B)`, which is logged as `energy_round_{L}`), and so is `aux_loss` (a 0-d tensor added to the total loss).
- **Reference.** [`monolithic_example.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/monolithic_example.py) is the shipped reference for this pattern.

```yaml
model:
  backbone:
    name: tiny_potential
    hidden_dim: 64
    num_elements: 9
    cutoff: ${data.cutoff}
    avg_num_neighbors: 12.0
  head: null                # required for monolithic models
```

## Registering the model

- **Inside the repository (recommended).** Put the file anywhere under `src/goal/ml/nn/models/`, including subpackages such as `simurgh/` (a new subpackage needs an `__init__.py`). The package calls `auto_discover`, which imports every submodule, so the decorators run as soon as any CLI imports `goal.ml.nn.models`. No `__init__.py` needs editing. If a module fails to import, it is skipped with a `UserWarning` saying why.
- **Register in both registries.** `goal-train` resolves names in `BACKBONE_REGISTRY`. `MODEL_REGISTRY` is the catalogue the built-in models also join. A duplicate name raises `KeyError`.
- **Outside the repository.** The registry supports `register_lazy("name", "pkg.module:Class")` and entry-point discovery (`[project.entry-points.'goal.ml.models']`). However, the shipped CLIs do not call `discover_plugins`, so an external module must be imported before `goal-train` builds the model. The simplest way is to add it under `src/goal/ml/nn/models/`.

## What `goal-train` does and does not fill in for you

Dataset-derived arguments are injected only for specific backbone names or config shapes (full table in [Overview](overview.md#dataset-derived-constructor-arguments)). For a new model:

- `elements` and a top-level `avg_num_neighbors` are **not** injected. Put them in the config.
- If your constructor accepts `atomic_energies` (a dict with `mode: learned | dataset | provided` and `values`), `mode: dataset` gets least-squares values from the training set. For `dataset` and `provided`, a `scale` kwarg is also passed, so accept `scale` too.
- A `dressing_kwargs` dict gets `avg_num_neighbors` filled in, and an `artisan`/`artisan_config` dict with `rare_artisan.enabled: true` gets `pair_counts` filled in.

## Checklist

- [ ] `forward` rebuilds edges from a grad-tracking `graph.pos` (or forces will be zero).
- [ ] Modular: `irreps_out` and `num_interactions` exist, there is no `output_keys`, and the head's `irreps_in` matches `irreps_out`.
- [ ] Monolithic: `output_keys` exists, forces are computed under `torch.enable_grad()`, `num_atoms` is returned, and the config has `head: null`.
- [ ] Every learnable parameter reaches the loss on every step. Unused parameters break DDP with `find_unused_parameters: false`.
- [ ] Equivariance test: rotate the input and check that $E$ is unchanged and $\mathbf F$ rotates with it.
- [ ] Conservation test: $\sum_i \mathbf F_i \approx 0$ for an isolated molecule, and autograd forces match finite differences.

A quick smoke test before a full run:

```python
import torch
from ase.build import molecule
from torch_geometric.data import Batch

import goal.ml.nn.models, goal.ml.nn.heads
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY

graphs = [AtomicGraph.from_ase(molecule(m), cutoff=5.0) for m in ("CH3CH2OH", "H2O")]
batch = Batch.from_data_list(graphs)

backbone = BACKBONE_REGISTRY.get("tiny_conv")()
head = HEAD_REGISTRY.get("energy_forces")(irreps_in=str(backbone.irreps_out))
out = head(backbone(batch), batch)
print(out["energy"].shape, out["forces"].shape)          # (2,), (12, 3)
assert out["forces"].abs().sum() > 0, "zero forces: geometry not built from graph.pos"
print(out["forces"][:9].sum(0))                          # ≈ 0 for ethanol
```
