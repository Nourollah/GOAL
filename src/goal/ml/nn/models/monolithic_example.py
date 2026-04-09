"""Monolithic example — a minimal self-contained model for demonstration.

Shows how to build a model satisfying the ``MonolithicModel`` protocol,
which bypasses the backbone→head split entirely.  The model takes an
``AtomicGraph`` and returns a dictionary of predicted properties
directly — the same format that ``TaskHead.forward`` produces.

This is intentionally simplistic (single-layer readout from atomic
embeddings) and is **not** intended as a production model.  Its purpose
is to demonstrate how external users can bring their own self-contained
architecture and plug it into the GOAL training loop by setting
``head: null`` in the config.

Satisfies the ``MonolithicModel`` protocol.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("monolithic_example")
class MonolithicExample(nn.Module):
    """Minimal monolithic model for demonstration purposes.

    Embeds atomic numbers, applies a small MLP to obtain per-atom
    energy contributions, sums to total energy, and derives forces
    via autograd.

    This exists purely to illustrate the ``MonolithicModel`` protocol.
    For real tasks, use a modular backbone + head combination instead.

    Parameters
    ----------
    num_elements : int
        Maximum atomic number supported.
    embedding_dim : int
        Dimension of atomic embeddings.
    hidden_dim : int
        Width of the hidden layer.

    Example
    -------
    >>> model = MonolithicExample()
    >>> preds = model(graph)  # {"energy": (B,), "forces": (N, 3)}
    """

    def __init__(
        self,
        num_elements: int = 120,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding: AtomicNumberEmbedding = AtomicNumberEmbedding(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
        )
        self.readout: nn.Sequential = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # MonolithicModel protocol
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[str]:
        """Keys this model produces."""
        return ["energy", "forces"]

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Predict energy and forces from an atomic graph.

        Args:
            graph: Input ``AtomicGraph`` with topology and features.

        Returns:
            Dictionary with ``"energy"`` (B,), ``"forces"`` (N, 3),
            and ``"num_atoms"`` (B,).
        """
        # Enable gradient tracking on positions for force computation
        positions: torch.Tensor = graph.pos
        positions.requires_grad_(True)

        # Atomic embeddings → per-atom energy contributions
        h: torch.Tensor = self.embedding(graph.z)  # (N, embedding_dim)
        atom_energies: torch.Tensor = self.readout(h).squeeze(-1)  # (N,)

        # Sum per-atom energies to get per-graph total energy
        energy: torch.Tensor = scatter(
            atom_energies,
            graph.batch,
            dim=0,
            reduce="sum",
        )  # (B,)

        # Forces via autograd: F = −∂E/∂R  (energy conserving)
        grad: tuple[torch.Tensor, ...] | None = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=positions,
            create_graph=self.training,
            retain_graph=True,
        )
        forces: torch.Tensor = -grad[0]  # (N, 3)

        num_atoms: torch.Tensor = scatter(
            torch.ones(graph.z.shape[0], device=graph.z.device),
            graph.batch,
            dim=0,
            reduce="sum",
        )  # (B,)

        return {
            "energy": energy,
            "forces": forces,
            "num_atoms": num_atoms,
        }
