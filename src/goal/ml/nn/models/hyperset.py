"""HyperSet backbone — pairwise-expert invariant backbone (NOT IMPLEMENTED).

Originally conceived as a variant of DeepSet with atom-type–specific
expert sub-networks for each unique pair of chemical elements.  In
practice the SCAI reference implementation was identical to DeepSet —
the pair-specific routing was declared but never used.

The model is retained as a named backbone stub for compatibility
with experiment tracking and configuration files.  Attempting to
instantiate it will raise ``NotImplementedError``.

Would satisfy the ``InvariantBackbone`` protocol if implemented.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.registry import MODEL_REGISTRY

_NOT_IMPLEMENTED_REASON: str = (
    "HyperSet is not implemented in GOAL. "
    "The SCAI reference was functionally identical to DeepSet — "
    "the pair-specific expert routing was declared but never wired. "
    "Use 'deepset' instead."
)


@MODEL_REGISTRY.register("hyperset")
class HyperSet(nn.Module):
    """Pairwise-expert invariant backbone (NOT IMPLEMENTED).

    The original SCAI HyperSet was intended to route edge features
    through atom-type–specific expert MLPs, but the implementation
    never diverged from DeepSet.  This stub preserves the name for
    experiment tracking; instantiation raises ``NotImplementedError``.

    Would satisfy ``InvariantBackbone`` — produces scalar per-node
    features ``"Hx0e"`` — if implemented.

    Parameters
    ----------
    num_elements : int
        Maximum atomic number supported.
    embedding_dim : int
        Dimension of initial atomic embeddings.
    hidden_channels : int
        Width of hidden layers.
    num_filters : int
        Size of the projected feature space.
    num_radial_basis : int
        Number of radial basis functions.
    transform_depth : int
        Number of layers in projection MLPs.
    cutoff : float
        Cutoff radius (Ångström).

    Raises
    ------
    NotImplementedError
        Always — this model is a placeholder.
    """

    def __init__(
        self,
        num_elements: int = 120,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_radial_basis: int = 20,
        transform_depth: int = 2,
        cutoff: float = 5.0,
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_REASON)

    # ------------------------------------------------------------------
    # InvariantBackbone protocol stubs (never reached)
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        """Dimension of the invariant node embedding."""
        return 128

    @property
    def num_interactions(self) -> int:
        """Number of message-passing layers."""
        return 1

    @property
    def irreps_out(self) -> typing.Any:
        """Invariant output: all scalar channels."""
        from e3nn.o3 import Irreps

        return Irreps("128x0e")

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Not implemented — raises unconditionally at ``__init__``."""
        raise NotImplementedError(_NOT_IMPLEMENTED_REASON)
