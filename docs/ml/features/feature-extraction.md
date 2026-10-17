# Feature extraction

A trained interatomic potential learns a rich per-atom description of its chemical environment. `goal.ml.utils.extraction` lets you read those intermediate features out of a backbone, layer by layer, slice them by angular momentum, and pool them per structure. You can then use them as learned descriptors for downstream regression, clustering and visualisation, or to see what each message-passing layer has learned.

## What is in the toolbox

| Object | Kind | What it does |
| --- | --- | --- |
| `HookBasedExtractor` | class | attaches forward hooks to every block of an `nn.ModuleList` and records each block's output |
| `extract_scalars(features)` | function | keeps only the $\ell = 0$ (rotation-invariant) channels of a `NodeFeatures` |
| `extract_irrep_channels(features, l)` | function | keeps all channels of one angular momentum $\ell$ |
| `pool_nodes(features, batch, num_graphs, mode)` | function | per-atom → per-structure pooling (`"sum"`, `"mean"`, `"max"`) |
| `describe_irreps(irreps_str)` | function | prints a table of an irreps string (uses `rich` if installed) |
| `LayerBackbone`, `MultiScaleBackbone`, `FrozenBackbone` | wrappers | expose one layer, all layers concatenated, or a frozen backbone through the backbone interface |
| `mace-large-final`, `mace-large-multiscale`, `mace-large-frozen` | registry names | pre-built MACE-MP-large wrappers (registered when `goal.ml.utils` is imported) |

Everything is importable from `goal.ml.utils.extraction`. The helpers work on `goal.ml.data.graph.NodeFeatures`, the container every GOAL backbone returns:

```python
@dataclass
class NodeFeatures:
    node_feats: torch.Tensor            # (N, irreps.dim)
    irreps: str                         # e3nn irreps string, e.g. "16x0e+16x1o+16x2e"
    node_energies: torch.Tensor | None = None
    node_forces: torch.Tensor | None = None
```

> [!WARNING]
> **State of the wrapper path (October 2026).** Two problems stop the wrapper classes from working end to end. Use the `HookBasedExtractor` recipes below until they are fixed.
>
> - `MACEAdapter.irreps_out` reads `mace_model.irreps_out`, an attribute MACE models (mace-torch 0.3.16) do not have. `MACEAdapter.forward`, `irreps_at_layer`, `extract_features`, `LayerBackbone.irreps_out`, `MultiScaleBackbone.irreps_out` and the `mace-large-*` registry entries all fail with `AttributeError`. `MACEAdapter.extract_features` also hooks `interactions[i]`, whose first output is MACE's pre-product `(N, channels, (l_max+1)²)` tensor rather than the layer's node features.
> - `LayerBackbone`, `MultiScaleBackbone` and `FrozenBackbone` are plain Python classes, not `nn.Module`s, and define no `__call__`. `GOALModule` calls `self.backbone(graph)`, so they cannot be used as a `GOALModule` backbone. Their parameters are also not registered for `.to(device)` or `state_dict()`. Call `.forward(graph)` on them explicitly.

## `HookBasedExtractor`

```python
HookBasedExtractor(model, blocks_attr="interactions", output_index=None, detach=True)
```

| Argument | Meaning |
| --- | --- |
| `model` | the module that owns the block list |
| `blocks_attr` | name of the `nn.ModuleList` attribute whose blocks get hooked: `"rounds"` for SIMURGH ARACE, `"products"` / `"interactions"` for MACE, `"layers"` for NequIP-style models |
| `output_index` | if each block returns a tuple, which element holds the node features (`None` means the block returns a tensor) |
| `detach` | `True` (default) stores detached tensors. `False` keeps them in the autograd graph, for example to back-propagate a downstream loss into the backbone |

After any forward pass of `model` (or of a parent module that calls it), `ext.captured` is a dict `{"layer_0": tensor, "layer_1": tensor, …}` in block order. `ext.extract(*args, **kwargs)` clears `captured`, runs `model(*args, **kwargs)` and returns a copy. Hooks are removed by `ext.remove()`, by leaving a `with` block, or when the object is garbage-collected. Captured tensors remain available after removal.

To find the right `blocks_attr`, list the module's children:

```python
print([name for name, child in backbone.named_children()])
# SimurghAraceBackbone -> ['embedding', 'input_linear', 'rounds']
```

## Recipe: per-layer features of a trained SIMURGH model

Each SIMURGH ARACE round (`backbone.rounds[i]`, an `AraceRound`) returns `(h, e_round, gate_scores, aux_loss)`, so the node features are `output_index=0`. Their irreps equal the backbone's `irreps_out`.

```python
import torch
from ase.build import molecule
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.utils.calculator import GOALCalculator
from goal.ml.utils.extraction import (
    HookBasedExtractor, describe_irreps, extract_irrep_channels, extract_scalars, pool_nodes,
)

# Load from the run's frozen source (archive or checkpoint directory).
calc = GOALCalculator(checkpoint_path="models/simurgh_gmd26.simurgh")
module = calc.module
backbone = module.backbone
describe_irreps(str(backbone.irreps_out))          # e.g. 16x0e+16x1o+16x2e

# Build graphs exactly as training did (cutoff, dtype, fragment labels if the model uses them).
data = module.config.data
frag_keys = ("compute_fragment_index", "fragment_covalent_cutoff", "fragment_scheme",
             "fragment_charge", "fragment_on_failure", "fragment_smarts", "fragment_keep_groups")
frag_kwargs = {k: data[k] for k in frag_keys if data.get(k) is not None}
graphs = [
    AtomicGraph.from_ase(molecule(name), cutoff=calc._cutoff, dtype=calc.dtype, **frag_kwargs)
    for name in ["CH3COOH", "CH3CH2OH", "CH3OH"]
]
batch = Batch.from_data_list(graphs).to(calc.device)

with HookBasedExtractor(backbone, blocks_attr="rounds", output_index=0) as ext, torch.no_grad():
    backbone(batch)
    per_round = dict(ext.captured)                 # {"layer_0": (N, 144), "layer_1": (N, 144)}

irreps = str(backbone.irreps_out)
for name, x in per_round.items():
    feats = NodeFeatures(node_feats=x, irreps=irreps)
    scalars = extract_scalars(feats)               # (N, 16)  invariant
    vectors = extract_irrep_channels(feats, l=1)   # (N, 48)  16 vectors x 3 components
    per_structure = pool_nodes(
        NodeFeatures(node_feats=scalars, irreps=f"{scalars.shape[-1]}x0e"),
        batch.batch, batch.num_graphs, mode="mean",
    )                                              # (3, 16)
    print(name, scalars.shape, vectors.shape, per_structure.shape)
```

What to do with them:

- **Invariant descriptors.** `extract_scalars` gives features that do not change under rotation. You can feed them to any standard regressor (ridge, kernel ridge, gradient boosting, an MLP) or to t-SNE/UMAP.
- **Equivariant channels.** $\ell \ge 1$ channels rotate with the molecule. Keep them only if the downstream model is equivariant (e3nn layers), or turn them into invariants yourself, for example per-channel norms. `extract_irrep_channels(feats, l=1)` returns them flattened as `mul × (2l+1)` columns, multiplicity-major.
- **Early versus late layers.** Earlier rounds have seen a smaller neighbourhood. Comparing descriptors across layers shows where in the network a chemical distinction (functional group, conformer, protonation state) first becomes separable.
- **Pooling.** `"sum"` is extensive (it grows with system size) and suits extensive targets. `"mean"` is intensive and better for properties like band gaps or per-atom quantities. For equivariant features, `"sum"` and `"mean"` preserve equivariance and `"max"` does not.

## Recipe: MACE-MP descriptors

MACE's per-layer node features are the outputs of its `products` blocks (one per interaction). Hook those on the raw MACE model. `MACEAdapter` is still useful for loading MACE-MP and converting `AtomicGraph` batches.

```python
import torch

torch.serialization.add_safe_globals([slice])    # e3nn 0.4.4 + PyTorch >= 2.6, before e3nn is imported

from ase.build import molecule
from torch_geometric.data import Batch

from goal.ml.adapters.mace import MACEAdapter
from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.utils.extraction import HookBasedExtractor, describe_irreps, extract_scalars, pool_nodes

adapter = MACEAdapter.from_pretrained("small").to("cpu").double()   # mace_mp loads to CUDA if available
mace_model = adapter._model

names = ["H2O", "CH3OH", "C6H6", "CH3CH2OH", "CH4"]
batch = Batch.from_data_list([AtomicGraph.from_ase(molecule(n), cutoff=5.0) for n in names])
mace_batch = adapter._to_mace_batch(batch)

with torch.no_grad(), HookBasedExtractor(mace_model, blocks_attr="products") as ext:
    mace_model(mace_batch, compute_force=False)
    per_layer = dict(ext.captured)                 # {"layer_0": (35, 128), "layer_1": (35, 128)}

irreps = "+".join(str(p.linear.irreps_out) for p in mace_model.products)
describe_irreps(irreps)                            # 128x0e+128x0e for MACE-MP small
x = torch.cat([per_layer[k] for k in sorted(per_layer)], dim=-1)
scalars = extract_scalars(NodeFeatures(node_feats=x, irreps=irreps))
descriptors = pool_nodes(
    NodeFeatures(node_feats=scalars, irreps=f"{scalars.shape[-1]}x0e"),
    batch.batch, batch.num_graphs, mode="mean",
)                                                  # (5, 256)
```

Notes:

- MACE needs its own environment (`.venv-mace`; see [Installation](../../getting-started/installation.md)). The `add_safe_globals([slice])` line is needed there with e3nn 0.4.4.
- The model's dtype and device must match the batch. `mace_mp` defaults to float32 on CUDA, while `MACEAdapter` builds float64 inputs, hence `.to("cpu").double()`. On a GPU, use `.to("cuda").double()` and `mace_batch` built from `batch.to("cuda")`.
- MACE-MP small and medium carry only scalar hidden features (`128x0e`). Larger variants and custom models also carry $\ell \ge 1$ channels, which `extract_scalars` drops.

## Use case: a learned-descriptor regression

With descriptors in hand, fitting a property you did not train the potential on takes a few lines. The closed-form ridge regression below runs on any `(n_structures, d)` descriptor matrix:

```python
import torch

def ridge_fit(X: torch.Tensor, y: torch.Tensor, alpha: float = 1e-3) -> torch.Tensor:
    X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=1)   # bias column
    A = X1.T @ X1 + alpha * torch.eye(X1.shape[1], dtype=X.dtype)
    return torch.linalg.solve(A, X1.T @ y)

def ridge_predict(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=1) @ w

# descriptors: (M, d) from either recipe above, y: (M,) target property
perm = torch.randperm(len(descriptors))
train, test = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)) :]
mu, sigma = descriptors[train].mean(0), descriptors[train].std(0).clamp_min(1e-8)
Xs = (descriptors - mu) / sigma
w = ridge_fit(Xs[train], y[train])
mae = (ridge_predict(Xs[test], w) - y[test]).abs().mean()
```

For a learned non-linear head trained on cached descriptors, with early stopping and loss curves, use the [Mini trainer](mini-trainer.md).

## The backbone wrappers

The wrappers are meant to let a downstream head consume a specific depth of a feature-extractor backbone (one that implements `extract_features(graph, layer)`, `irreps_at_layer(layer)` and `num_layers`, the `FeatureExtractorBackbone` protocol in `goal.ml.nn.models.base`). Their interface:

| Wrapper | `forward(graph)` returns | `irreps_out` | Other members |
| --- | --- | --- | --- |
| `LayerBackbone(adapter, layer)` | `adapter.extract_features(graph, layer=layer)` | `adapter.irreps_at_layer(layer)` | `num_interactions`, `parameters()`, `named_parameters()` |
| `MultiScaleBackbone(adapter)` | `adapter.extract_features(graph, layer="all")`, all layers concatenated | `"+"`-join of every layer's irreps | `num_layers`, `num_interactions` |
| `FrozenBackbone(backbone)` | `backbone.forward(graph)` under `torch.no_grad()` | `backbone.irreps_out` | sets `requires_grad=False` on every parameter at construction |

`FrozenBackbone` freezes the wrapped object's parameters **in place**, so the same underlying module stays frozen when used elsewhere. The registry names build the MACE-MP-large versions: `mace-large-final` is `LayerBackbone(adapter, layer=-1)`, `mace-large-multiscale` is `MultiScaleBackbone(adapter)`, and `mace-large-frozen` is `FrozenBackbone(MultiScaleBackbone(adapter))`. All three are subject to the warning at the top of this page.

## Notebook

[`notebooks/feature_extraction_demo.ipynb`](https://github.com/Nourollah/GOAL/blob/main/notebooks/feature_extraction_demo.ipynb) walks through building `AtomicGraph`s from inline structures, layer-wise and multi-scale extraction from MACE, scalar slicing and freezing. It uses the `MACEAdapter`/wrapper API, so until the issues above are fixed, the cells that rely on `irreps_out` / `extract_features` fail. The `HookBasedExtractor`, `extract_scalars` and `pool_nodes` cells carry over directly to the recipes on this page.

## See also

- [Mini trainer](mini-trainer.md): train a head on cached descriptors.
- [MACE fine-tuning](../finetuning/mace.md): fine-tune MACE rather than freeze it.
- [ASE calculator](../deployment/ase-calculator.md): loading trained GOAL models.
