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

    def forward(self, graph: AtomicGraph) -> NodeFeatures: ...  # noqa: F821

    @classmethod
    def from_pretrained(cls, variant: str) -> FoundationAdapter: ...

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
