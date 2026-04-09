"""Feature extraction utilities for intermediate layer access.

Provides tools to extract, inspect, and transform intermediate node
features from any backbone model. Includes:

- ``HookBasedExtractor`` — generic hook-based feature capture
- ``LayerBackbone`` / ``MultiScaleBackbone`` / ``FrozenBackbone`` — composable wrappers
- ``extract_scalars`` / ``extract_irrep_channels`` — irrep slicing helpers
- ``pool_nodes`` — per-graph pooling
- ``describe_irreps`` — pretty-print irreps breakdown
"""

from __future__ import annotations

import typing
from contextlib import contextmanager

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from torch_geometric.utils import scatter

from gmd.data.graph import AtomicGraph, NodeFeatures


# ---------------------------------------------------------------------------
# HookBasedExtractor
# ---------------------------------------------------------------------------


class HookBasedExtractor:
    """Attach forward hooks to a model's interaction blocks to capture
    intermediate node feature tensors during the forward pass.

    Generic — works with any model whose interaction blocks are accessible
    as a ``nn.ModuleList`` attribute (e.g. ``model.interactions``,
    ``model.layers``).

    Parameters
    ----------
    model : nn.Module
        The model to attach hooks to.
    blocks_attr : str
        Attribute name of the ``nn.ModuleList`` containing interaction
        blocks.  For MACE: ``'interactions'``, for NequIP: ``'layers'``.
    output_index : int or None
        If interaction blocks return a tuple, which positional element
        contains the node features.  ``None`` if they return a tensor
        directly.  MACE blocks return ``(node_feats, sc)`` so use ``0``.
    detach : bool
        If ``True`` (default), captured tensors are detached from the
        computation graph — suitable for inference.  Set ``False`` when
        you need gradients to flow through the extracted features
        (e.g. for fine-tuning with a downstream head).

    Example::

        with HookBasedExtractor(mace_model, 'interactions', output_index=0) as ext:
            output = mace_model(input_dict)
            features = ext.captured   # {'layer_0': ..., 'layer_1': ..., ...}
    """

    def __init__(
        self,
        model: nn.Module,
        blocks_attr: str = "interactions",
        output_index: typing.Optional[int] = None,
        detach: bool = True,
    ) -> None:
        self._model: nn.Module = model
        self._blocks_attr: str = blocks_attr
        self._output_index: typing.Optional[int] = output_index
        self._detach: bool = detach
        self._hooks: typing.List[torch.utils.hooks.RemovableHook] = []
        self.captured: typing.Dict[str, torch.Tensor] = {}
        self._attach()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _attach(self) -> None:
        """Register forward hooks on each block in the ``ModuleList``."""
        blocks: nn.ModuleList = getattr(self._model, self._blocks_attr)
        for idx, block in enumerate(blocks):
            key: str = f"layer_{idx}"
            hook = block.register_forward_hook(self._make_hook(key))
            self._hooks.append(hook)

    def _make_hook(
        self, key: str,
    ) -> typing.Callable:
        """Return a hook closure that stores the block's output tensor."""

        def hook_fn(
            _module: nn.Module,
            _input: typing.Any,
            output: typing.Any,
        ) -> None:
            tensor: torch.Tensor
            if self._output_index is not None:
                tensor = output[self._output_index]
            else:
                tensor = output
            self.captured[key] = tensor.detach() if self._detach else tensor

        return hook_fn

    def remove(self) -> None:
        """Remove all registered hooks and clear captured data."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.captured.clear()

    def extract(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Dict[str, torch.Tensor]:
        """Run the model forward pass and return captured features.

        All positional and keyword arguments are forwarded to
        ``model.forward()``.

        Returns:
            Dict mapping ``'layer_0'``, ``'layer_1'``, … to feature tensors.
        """
        self.captured.clear()
        self._model(*args, **kwargs)
        return dict(self.captured)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> HookBasedExtractor:
        return self

    def __exit__(
        self,
        exc_type: typing.Optional[type],
        exc_val: typing.Optional[BaseException],
        exc_tb: typing.Any,
    ) -> None:
        self.remove()

    def __del__(self) -> None:
        self.remove()


# ---------------------------------------------------------------------------
# Backbone wrappers
# ---------------------------------------------------------------------------


class LayerBackbone:
    """Wrap a ``FeatureExtractorBackbone`` to expose a single specific
    layer as the backbone output.

    Satisfies the ``EquivariantBackbone`` protocol.

    Use when your downstream head should receive features from a specific
    interaction depth — early layers encode local geometry, late layers
    encode global context.

    Parameters
    ----------
    adapter : FeatureExtractorBackbone
        The backbone implementing ``extract_features``.
    layer : int
        0-indexed interaction layer to extract.

    Example::

        backbone = LayerBackbone(mace_adapter, layer=0)
        head = DeepSetHead(irreps_in=str(backbone.irreps_out), ...)
        module = GMDModule(backbone=backbone, head=head, ...)
    """

    def __init__(
        self,
        adapter: typing.Any,  # FeatureExtractorBackbone
        layer: int,
    ) -> None:
        self._adapter: typing.Any = adapter
        self._layer: int = layer

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Extract features at the configured layer.

        Args:
            graph: Input ``AtomicGraph``.

        Returns:
            ``NodeFeatures`` from the specified interaction layer.
        """
        return self._adapter.extract_features(graph, layer=self._layer)

    @property
    def irreps_out(self) -> Irreps:
        """Output irreps of the selected layer."""
        return self._adapter.irreps_at_layer(self._layer)

    @property
    def num_interactions(self) -> int:
        """Number of interaction layers in the underlying adapter."""
        return self._adapter.num_layers

    def parameters(self) -> typing.Iterator[nn.Parameter]:
        """Proxy to underlying adapter parameters."""
        return self._adapter.parameters()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True,
    ) -> typing.Iterator[typing.Tuple[str, nn.Parameter]]:
        """Proxy to underlying adapter named_parameters."""
        if hasattr(self._adapter, "named_parameters"):
            return self._adapter.named_parameters(prefix=prefix, recurse=recurse)
        return iter([])


class MultiScaleBackbone:
    """Wrap a ``FeatureExtractorBackbone`` to concatenate features from
    ALL interaction layers along the feature dimension.

    Satisfies the ``EquivariantBackbone`` protocol.

    Gives downstream models simultaneous access to:
    - Local structural information (early layers, small receptive field)
    - Global structural information (late layers, large receptive field)

    The output irreps is the concatenation of irreps at each layer.
    Use ``extract_scalars()`` downstream if your head is not equivariant.

    Parameters
    ----------
    adapter : FeatureExtractorBackbone
        The backbone implementing ``extract_features``.

    Example::

        backbone = MultiScaleBackbone(mace_adapter)
        # backbone.irreps_out is e.g. '256x0e+256x1o+256x0e+256x1o'
    """

    def __init__(self, adapter: typing.Any) -> None:  # FeatureExtractorBackbone
        self._adapter: typing.Any = adapter

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Extract and concatenate features from all layers.

        Args:
            graph: Input ``AtomicGraph``.

        Returns:
            ``NodeFeatures`` with concatenated multi-scale features.
        """
        return self._adapter.extract_features(graph, layer="all")

    @property
    def irreps_out(self) -> Irreps:
        """Concatenated irreps across all layers."""
        parts: typing.List[Irreps] = [
            self._adapter.irreps_at_layer(i)
            for i in range(self._adapter.num_layers)
        ]
        combined: str = "+".join(str(p) for p in parts)
        return Irreps(combined)

    @property
    def num_layers(self) -> int:
        """Number of interaction layers."""
        return self._adapter.num_layers

    @property
    def num_interactions(self) -> int:
        """Alias for ``num_layers``."""
        return self._adapter.num_layers

    def parameters(self) -> typing.Iterator[nn.Parameter]:
        """Proxy to underlying adapter parameters."""
        return self._adapter.parameters()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True,
    ) -> typing.Iterator[typing.Tuple[str, nn.Parameter]]:
        """Proxy to underlying adapter named_parameters."""
        if hasattr(self._adapter, "named_parameters"):
            return self._adapter.named_parameters(prefix=prefix, recurse=recurse)
        return iter([])


class FrozenBackbone:
    """Wrap any backbone and freeze all its parameters.

    Useful for using a foundation model purely as a feature extractor
    without fine-tuning — saves memory and compute significantly.

    Frozen parameters are excluded from optimiser parameter groups
    automatically when used inside ``GMDModule`` (parameters with
    ``requires_grad=False`` are skipped by default).

    Parameters
    ----------
    backbone : EquivariantBackbone or any backbone
        The backbone to freeze.  All parameters will have
        ``requires_grad`` set to ``False``.

    Example::

        frozen = FrozenBackbone(MultiScaleBackbone(mace_adapter))
        head = DeepSetHead(...)
        module = GMDModule(backbone=frozen, head=head, ...)
        # Only head parameters appear in the optimiser
    """

    def __init__(self, backbone: typing.Any) -> None:
        self._backbone: typing.Any = backbone
        # Freeze all parameters
        for param in self._backbone.parameters():
            param.requires_grad = False

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Run the backbone with no gradient tracking.

        Args:
            graph: Input ``AtomicGraph``.

        Returns:
            ``NodeFeatures`` with detached tensors.
        """
        with torch.no_grad():
            return self._backbone.forward(graph)

    @property
    def irreps_out(self) -> Irreps:
        """Output irreps delegated to the wrapped backbone."""
        return self._backbone.irreps_out

    @property
    def num_interactions(self) -> int:
        """Number of interaction layers in the wrapped backbone."""
        if hasattr(self._backbone, "num_interactions"):
            return self._backbone.num_interactions
        if hasattr(self._backbone, "num_layers"):
            return self._backbone.num_layers
        return 0

    def parameters(self) -> typing.Iterator[nn.Parameter]:
        """Return underlying parameters (all frozen)."""
        return self._backbone.parameters()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True,
    ) -> typing.Iterator[typing.Tuple[str, nn.Parameter]]:
        """Return underlying named parameters (all frozen)."""
        if hasattr(self._backbone, "named_parameters"):
            return self._backbone.named_parameters(prefix=prefix, recurse=recurse)
        return iter([])


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def extract_scalars(features: NodeFeatures) -> torch.Tensor:
    """Extract only the L=0 (invariant / scalar) channels from equivariant
    node features.

    Safe to pass to any standard MLP — these channels are rotationally
    invariant.  Higher-order channels (L=1 vectors, L=2 tensors) are
    discarded.

    Args:
        features: ``NodeFeatures`` with irreps string, e.g.
            ``'256x0e+256x1o+256x2e'``.

    Returns:
        Tensor of shape ``(N, scalar_dim)`` containing only L=0 channels.
        ``scalar_dim = sum of multiplicities for all L=0 irreps``.

    Example::

        # irreps '256x0e+256x1o' → scalar_dim = 256
        # (256 scalars kept, 256×3 vector channels discarded)
        scalars = extract_scalars(features)
        assert scalars.shape[-1] == 256
    """
    irreps: Irreps = Irreps(features.irreps)
    node_feats: torch.Tensor = features.node_feats
    scalar_slices: typing.List[torch.Tensor] = []
    offset: int = 0

    for mul, ir in irreps:
        dim: int = mul * ir.dim
        if ir.l == 0:
            scalar_slices.append(node_feats[:, offset : offset + dim])
        offset += dim

    if not scalar_slices:
        return torch.zeros(
            node_feats.shape[0], 0,
            dtype=node_feats.dtype, device=node_feats.device,
        )
    return torch.cat(scalar_slices, dim=-1)


def extract_irrep_channels(
    features: NodeFeatures,
    l: int,
) -> torch.Tensor:
    """Extract all channels corresponding to a specific angular momentum ``l``.

    Args:
        features: ``NodeFeatures`` with irreps string.
        l: Angular momentum order.
            ``0`` = scalar, ``1`` = vector, ``2`` = matrix, ``3`` = octupole.

    Returns:
        Tensor of shape ``(N, total_dim)`` where
        ``total_dim = sum(mul * (2l+1))`` for all irreps matching ``l``.

    Example::

        # irreps '128x0e+64x1o+32x2e'
        vectors = extract_irrep_channels(features, l=1)
        assert vectors.shape[-1] == 64 * 3  # 192
    """
    irreps: Irreps = Irreps(features.irreps)
    node_feats: torch.Tensor = features.node_feats
    slices: typing.List[torch.Tensor] = []
    offset: int = 0

    for mul, ir in irreps:
        dim: int = mul * ir.dim
        if ir.l == l:
            slices.append(node_feats[:, offset : offset + dim])
        offset += dim

    if not slices:
        return torch.zeros(
            node_feats.shape[0], 0,
            dtype=node_feats.dtype, device=node_feats.device,
        )
    return torch.cat(slices, dim=-1)


def pool_nodes(
    features: NodeFeatures,
    batch: torch.Tensor,
    num_graphs: int,
    mode: str = "sum",
) -> torch.Tensor:
    """Pool per-node features to per-graph features.

    Thin wrapper around ``torch_geometric`` global pool functions.

    Args:
        features: ``NodeFeatures`` with ``node_feats`` of shape ``(N, D)``.
        batch: ``(N,)`` tensor mapping each node to its graph index.
        num_graphs: Total number of graphs in the batch.
        mode: Pooling mode — ``'sum'``, ``'mean'``, or ``'max'``.

    Returns:
        Tensor of shape ``(num_graphs, D)``.

    Note:
        For equivariant features use ``mode='sum'`` — it preserves
        equivariance.  ``'mean'`` also preserves it.  ``'max'`` does not.

    Example::

        graph_repr = pool_nodes(features, graph.batch, graph.num_graphs, mode='sum')
        assert graph_repr.shape == (graph.num_graphs, features.node_feats.shape[-1])
    """
    node_feats: torch.Tensor = features.node_feats
    reduce: str = mode
    return scatter(node_feats, batch, dim=0, reduce=reduce, dim_size=num_graphs)


def describe_irreps(irreps_str: str) -> None:
    """Pretty-print a breakdown of an irreps string.

    Useful in notebooks to understand what features a backbone produces.
    Uses ``rich`` for pretty printing if available, falls back to plain text.

    Args:
        irreps_str: e3nn irreps string, e.g. ``'256x0e+256x1o+256x2e'``.

    Example::

        describe_irreps('256x0e+256x1o+256x2e')
        # Prints a table with columns: Irrep, Mul, L, Parity, Dim
    """
    irreps: Irreps = Irreps(irreps_str)
    total_dim: int = irreps.dim

    _L_NAMES: typing.Dict[int, str] = {
        0: "scalar",
        1: "vector",
        2: "matrix",
        3: "octupole",
    }
    _PARITY: typing.Dict[int, str] = {1: "even", -1: "odd"}

    rows: typing.List[typing.Tuple[str, int, str, str, int]] = []
    for mul, ir in irreps:
        l_name: str = _L_NAMES.get(ir.l, f"L={ir.l}")
        parity: str = _PARITY.get(ir.p, str(ir.p))
        dim: int = mul * ir.dim
        rows.append((str(ir), mul, l_name, parity, dim))

    try:
        from rich.console import Console
        from rich.table import Table

        console: typing.Any = Console()
        table: typing.Any = Table(title=f"Irreps: {irreps_str}  (total dim: {total_dim})")
        table.add_column("Irrep", style="cyan")
        table.add_column("Mul", justify="right", style="green")
        table.add_column("L", style="yellow")
        table.add_column("Parity", style="magenta")
        table.add_column("Dim (mul×(2l+1))", justify="right", style="bold")

        for irrep_str, mul, l_name, parity, dim in rows:
            table.add_row(irrep_str, str(mul), l_name, parity, str(dim))

        console.print(table)

    except ImportError:
        print(f"Irreps: {irreps_str}")
        print(f"Total dim: {total_dim}")
        print(f"{'Irrep':<10} {'Mul':>5} {'L':<10} {'Parity':<8} {'Dim':>8}")
        print("-" * 45)
        for irrep_str, mul, l_name, parity, dim in rows:
            print(f"{irrep_str:<10} {mul:>5} {l_name:<10} {parity:<8} {dim:>8}")


# ---------------------------------------------------------------------------
# Factory functions for registry
# ---------------------------------------------------------------------------


def _build_mace_large_final(**kwargs: typing.Any) -> LayerBackbone:
    """Build a MACE-large adapter exposing only the final layer.

    Registered as ``'mace-large-final'`` in the backbone registry.
    """
    from gmd.adapters.mace import MACEAdapter

    adapter: MACEAdapter = MACEAdapter.from_pretrained("large")
    return LayerBackbone(adapter, layer=-1)


def _build_mace_large_multiscale(**kwargs: typing.Any) -> MultiScaleBackbone:
    """Build a MACE-large adapter exposing all layers concatenated.

    Registered as ``'mace-large-multiscale'`` in the backbone registry.
    """
    from gmd.adapters.mace import MACEAdapter

    adapter: MACEAdapter = MACEAdapter.from_pretrained("large")
    return MultiScaleBackbone(adapter)


def _build_mace_large_frozen(**kwargs: typing.Any) -> FrozenBackbone:
    """Build a frozen MACE-large multi-scale backbone.

    Registered as ``'mace-large-frozen'`` in the backbone registry.
    All parameters are frozen — only downstream heads are trainable.
    """
    from gmd.adapters.mace import MACEAdapter

    adapter: MACEAdapter = MACEAdapter.from_pretrained("large")
    ms: MultiScaleBackbone = MultiScaleBackbone(adapter)
    return FrozenBackbone(ms)
