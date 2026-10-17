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

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Map an atomic graph to per-node features.

        Implementations read whatever ``AtomicGraph`` fields they need
        (typically ``pos`` ``(N, 3)``, ``atomic_numbers`` ``(N,)``, ``edge_index``
        ``(2, E)`` and ``batch`` ``(N,)``) and return a ``NodeFeatures``
        whose ``node_feats`` has one row per atom and whose ``irreps``
        string describes its layout (``"Nx0e"`` for invariant models).
        ``node_energies`` / ``node_forces`` may be filled when the model
        computes them, and heads will honour them.
        """
        ...

    @property
    def num_interactions(self) -> int: ...


@typing.runtime_checkable
class FeatureExtractorBackbone(typing.Protocol):
    """Extended backbone protocol for models that support intermediate
    feature extraction.

    Any backbone implementing this can serve as a feature extractor for
    downstream heads or wrappers (e.g. ``LayerBackbone``,
    ``MultiScaleBackbone``).  Extends the ``EquivariantBackbone``
    contract — any ``FeatureExtractorBackbone`` is also a valid
    ``EquivariantBackbone``.

    The ``extract_features`` method provides access to intermediate
    interaction-layer outputs, enabling multi-scale feature analysis
    and layer-specific downstream heads.
    """

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Map an atomic graph to final-layer equivariant node features.

        The returned ``NodeFeatures.node_feats`` is ``(N, irreps_out.dim)``
        and ``NodeFeatures.irreps`` should equal ``str(irreps_out)``, i.e.
        the same features as ``extract_features(graph, layer=-1)``.
        """
        ...

    def extract_features(
        self,
        graph: AtomicGraph,
        layer: int | str = -1,
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
    ) -> dict[str, torch.Tensor]:
        """Map node features to task outputs (energy, forces, etc.)."""
        ...

    @property
    def output_keys(self) -> list[str]:
        """Keys this head produces: ``['energy', 'forces', 'stress']``."""
        ...


@typing.runtime_checkable
class MonolithicModel(typing.Protocol):
    """Protocol for monolithic models that bypass the backbone→head split.

    A monolithic model takes an ``AtomicGraph`` and directly returns a
    dictionary of predicted properties (energy, forces, etc.) — the same
    format that ``TaskHead.forward`` produces.  This allows self-contained
    architectures that handle everything internally whilst still
    integrating with the GOAL training loop.

    Monolithic models **can** use any available module from the project
    (blocks, primitives, heads) — they simply skip the separate head
    wiring.  The ``output_keys`` property declares which dictionary
    keys consumers can expect.
    """

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Map an atomic graph directly to predicted properties."""
        ...

    @property
    def output_keys(self) -> list[str]:
        """Keys this model produces, e.g. ``['energy', 'forces']``."""
        ...


@typing.runtime_checkable
class LossFunction(typing.Protocol):
    """Protocol for composable loss functions."""

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compare predictions with targets and return a scalar loss.

        Args:
            predictions: Model/head output dict (e.g. ``'energy'``,
                ``'forces'``, ``'num_atoms'``).
            targets: Reference labels keyed like ``predictions``
                (typically the PyG batch).

        Returns:
            A 0-d tensor, differentiable w.r.t. the predictions.
        """
        ...

    @property
    def weight(self) -> float: ...
