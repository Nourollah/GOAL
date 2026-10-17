"""Protocol for all foundation model adapters.

Adapters translate between upstream model conventions and GOAL's
``AtomicGraph`` / ``NodeFeatures`` contract. Never copy upstream
source code — import their package.
"""

from __future__ import annotations

import typing
from pathlib import Path


@typing.runtime_checkable
class FoundationAdapter(typing.Protocol):
    """Protocol that all foundation model adapters must satisfy."""

    def forward(self, graph: AtomicGraph) -> NodeFeatures:  # noqa: F821
        """Run the upstream model on a GOAL graph and return node features.

        Implementations convert the ``AtomicGraph`` (``pos`` ``(N, 3)``
        in Å, ``z``, ``edge_index``, ``batch``, optional ``cell``) to the
        upstream input format and wrap the upstream per-atom descriptors
        as ``NodeFeatures`` (``node_feats`` ``(N, irreps.dim)`` plus the
        matching e3nn ``irreps`` string).
        """
        ...

    @classmethod
    def from_pretrained(cls, variant: str) -> FoundationAdapter:
        """Download / load a published pre-trained model and wrap it.

        ``variant`` selects the upstream model size or name (e.g.
        ``"small"`` / ``"medium"`` / ``"large"`` for MACE-MP).  Should
        raise ``ImportError`` with an install hint when the upstream
        package is missing.
        """
        ...

    @classmethod
    def from_local(
        cls,
        checkpoint_path: str | Path,
        **kwargs: typing.Any,
    ) -> FoundationAdapter:
        """Load from a local fine-tuned checkpoint.

        If not implemented, raises ``NotImplementedError`` with a clear
        message indicating the feature is planned for a future version.
        """
        ...

    def parameters(self) -> typing.Iterator:
        """Access underlying model parameters (for freezing etc.)."""
        ...

    def requires(self) -> list[str]:
        """Return list of required pip packages."""
        ...
