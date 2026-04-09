"""Protocol definitions for model backbones and task heads.

Every major component is defined by a ``Protocol`` — a structural
interface that any class satisfying it automatically implements,
without inheritance. This enables duck typing with static type checking.

Four backbone types are supported:
- ``EquivariantBackbone`` — uses e3nn irreps (MACE, NequIP, etc.)
- ``InvariantBackbone`` — outputs scalar features only (SchNet, DimeNet, etc.)
- ``FeatureExtractorBackbone`` — equivariant backbone with intermediate layer extraction
- ``Backbone`` — universal protocol accepting any output format
"""

from __future__ import annotations

import typing

import torch
from e3nn.o3 import Irreps

from goal.ml.data.graph import AtomicGraph, NodeFeatures


@typing.runtime_checkable
class EquivariantBackbone(typing.Protocol):
    """Protocol for equivariant backbones using e3nn irreps.

    Implement this for models that produce equivariant node features:
    NequIP, MACE, Allegro, BOTNet, etc.
    """

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Map atomic graph to equivariant node features."""
        ...

    @property
    def irreps_out(self) -> Irreps:
        """Output irreducible representations."""
        ...

    @property
    def num_interactions(self) -> int:
        """Number of message-passing layers."""
        ...


@typing.runtime_checkable
class InvariantBackbone(typing.Protocol):
    """Protocol for invariant backbones that produce only scalar features.

    Implement this for models like SchNet, DimeNet, DimeNet++, PaiNN
    (scalar mode), GemNet, etc. that output per-node invariant embeddings
    rather than equivariant irrep features.
    """

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Map atomic graph to invariant node features.

        The returned ``NodeFeatures.irreps`` should be ``"Nx0e"``
        (N scalar channels, even parity).
        """
        ...

    @property
    def hidden_dim(self) -> int:
        """Dimension of the invariant node embedding."""
        ...

    @property
    def num_interactions(self) -> int:
        """Number of message-passing layers."""
        ...


@typing.runtime_checkable
class Backbone(typing.Protocol):
    """Universal backbone protocol — any model that takes AtomicGraph
    and returns NodeFeatures.

    Use this when you don't need to distinguish between equivariant
    and invariant models.
    """

    def forward(self, graph: AtomicGraph) -> NodeFeatures: ...

    @property
    def num_interactions(self) -> int: ...


@typing.runtime_checkable
class FeatureExtractorBackbone(typing.Protocol):
    """Extended backbone protocol for models that support intermediate
    feature extraction.

    Any backbone implementing this can serve as a feature extractor for
    downstream models like DeepSet or HyperSet. Extends the
    ``EquivariantBackbone`` contract — any ``FeatureExtractorBackbone``
    is also a valid ``EquivariantBackbone``.

    The ``extract_features`` method provides access to intermediate
    interaction-layer outputs, enabling multi-scale feature analysis
    and layer-specific downstream heads.
    """

    def forward(self, graph: AtomicGraph) -> NodeFeatures: ...

    def extract_features(
        self,
        graph: AtomicGraph,
        layer: typing.Union[int, str] = -1,
    ) -> NodeFeatures:
        """Extract node features at a specific interaction layer.

        Args:
            graph: Input ``AtomicGraph``.
            layer: Which layer to extract from:
                   ``-1`` or ``'final'`` → last interaction block (default).
                   ``0, 1, ..., N`` → specific interaction block (0-indexed).
                   ``'all'`` → concatenation across all layers (multi-scale).

        Returns:
            ``NodeFeatures`` with the extracted features and corresponding irreps.
        """
        ...

    @property
    def num_layers(self) -> int:
        """Number of message-passing / interaction layers."""
        ...

    def irreps_at_layer(self, layer: int) -> Irreps:
        """Irreducible representations of features at a given layer.

        Args:
            layer: 0-indexed interaction layer.

        Returns:
            ``e3nn.o3.Irreps`` for that layer's output.
        """
        ...

    @property
    def irreps_out(self) -> Irreps: ...


@typing.runtime_checkable
class TaskHead(typing.Protocol):
    """Protocol for task-specific output heads."""

    def forward(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> typing.Dict[str, torch.Tensor]:
        """Map node features to task outputs (energy, forces, etc.)."""
        ...

    @property
    def output_keys(self) -> typing.List[str]:
        """Keys this head produces: ``['energy', 'forces', 'stress']``."""
        ...


@typing.runtime_checkable
class LossFunction(typing.Protocol):
    """Protocol for composable loss functions."""

    def forward(
        self,
        predictions: typing.Dict[str, torch.Tensor],
        targets: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor: ...

    @property
    def weight(self) -> float: ...
