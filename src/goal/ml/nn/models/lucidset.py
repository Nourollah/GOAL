"""LucidSet backbone — pairwise distance-binned MoE backbone (NOT IMPLEMENTED).

Originally conceived as a model that routes edge features through
different expert sub-networks based on *both* the chemical identity
of the atom pair and the interatomic distance bin.  Each unique
element pair (e.g. H-C, C-O) gets its own expert MLP, and edges
are routed to different experts depending on which distance bin
they fall into.

The approach does not scale naturally within GOAL's paradigm:
- It requires an external dictionary mapping element symbols to
  atomic numbers (``atom_references``), whereas GOAL operates
  directly on ``atomic_numbers`` from the graph.
- The combinatorial number of pair modules grows as
  ``O(Z² × num_distance_bins)``, making it impractical for
  diverse chemical compositions.

The model is retained as a named backbone stub for compatibility
with experiment tracking and configuration files.  Attempting to
instantiate it will raise ``NotImplementedError``.

Would satisfy the ``InvariantBackbone`` protocol if implemented.
"""

from __future__ import annotations

import typing

import torch.nn as nn

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.registry import MODEL_REGISTRY

_NOT_IMPLEMENTED_REASON: str = (
    "LucidSet is not implemented in GOAL. "
    "The pairwise distance-binned MoE approach requires an external "
    "atom_references dictionary and creates O(Z² × bins) expert "
    "modules, which does not scale within GOAL's paradigm. "
    "Use 'deepset' for a working edge-based invariant backbone."
)


@MODEL_REGISTRY.register("lucidset")
class LucidSet(nn.Module):
    """Pairwise distance-binned mixture-of-experts backbone (NOT IMPLEMENTED).

    Routes edge features through atom-pair–specific expert MLPs,
    each handling a particular distance bin.  The combinatorial
    explosion of pair×bin modules and the requirement for an
    external ``atom_references`` mapping make this impractical
    in GOAL.

    Would satisfy ``InvariantBackbone`` — produces scalar per-node
    features ``"Hx0e"`` — if implemented.

    Parameters
    ----------
    num_elements : int
        Maximum atomic number supported.
    hidden_channels : int
        Width of hidden layers.
    num_radial_basis : int
        Number of radial basis functions.
    cutoff : float
        Cutoff radius (Ångström).
    distance_boundaries : list[float] | None
        Distance thresholds defining expert bins.

    Raises
    ------
    NotImplementedError
        Always — this model is a placeholder.
    """

    def __init__(
        self,
        num_elements: int = 120,
        hidden_channels: int = 128,
        num_radial_basis: int = 20,
        cutoff: float = 5.0,
        distance_boundaries: list[float] | None = None,
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
