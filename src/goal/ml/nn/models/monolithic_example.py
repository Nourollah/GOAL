"""Monolithic example — a minimal self-contained model.

Shows how to build a model satisfying the :class:`MonolithicModel`
protocol, which bypasses the backbone→head split entirely.  The model
takes an ``AtomicGraph`` and returns a dictionary of predicted
properties directly — the same format that ``TaskHead.forward``
produces.

The model has two energy terms:

1. **Per-element baseline** — atomic-number embedding ⊕ MLP scattered
   to atoms.  Captures the (large, constant) per-Z energy offset that
   dominates raw DFT labels.
2. **Pairwise distance term** — edge lengths recomputed *from*
   ``graph.pos`` inside the forward, expanded in a Bessel basis,
   passed through a small MLP, tapered by a cosine cutoff and
   scattered to atoms.  Because the distances are differentiable
   functions of positions, ``torch.autograd`` can derive forces from
   the total energy.

It is intentionally simple — a SchNet-without-message-passing — and
exists as a reference for new monolithic models.  Set ``head: null``
in the model config to use it; the training loop calls the model
directly and consumes its dict output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.artisans import cosine_cutoff
from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.primitives.radial import BesselBasis
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


@MODEL_REGISTRY.register("monolithic_example")
@BACKBONE_REGISTRY.register("monolithic_example")
class MonolithicExample(nn.Module):
    """Minimal monolithic model.

    Parameters
    ----------
    num_elements : int
        Maximum atomic number supported.
    embedding_dim : int
        Dimension of atomic embeddings.
    hidden_dim : int
        Width of the hidden layer in both readout MLPs.
    cutoff : float
        Cosine-cutoff radius (Angstrom) for the pairwise distance term.
    num_radial_basis : int
        Number of Bessel basis functions used to expand edge lengths.

    Example
    -------
    >>> model = MonolithicExample()
    >>> preds = model(graph)  # {"energy": (B,), "forces": (N, 3), ...}
    """

    def __init__(
        self,
        num_elements: int = 120,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        cutoff: float = 5.0,
        num_radial_basis: int = 8,
    ) -> None:
        super().__init__()
        self.cutoff: float = cutoff

        # Per-element baseline pathway
        self.embedding: AtomicNumberEmbedding = AtomicNumberEmbedding(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
        )
        self.atom_readout: nn.Sequential = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Pairwise distance pathway
        self.radial_basis: BesselBasis = BesselBasis(
            num_basis=num_radial_basis,
            cutoff=cutoff,
        )
        self.pair_readout: nn.Sequential = nn.Sequential(
            nn.Linear(num_radial_basis, hidden_dim),
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

        Parameters
        ----------
        graph : AtomicGraph
            Input graph with positions, atomic numbers, edge index,
            and ``batch`` membership.

        Returns
        -------
        dict
            ``{"energy": (B,), "forces": (N, 3), "num_atoms": (B,)}``.
        """
        # Lightning runs ``validation_step`` / ``test_step`` inside
        # ``torch.inference_mode`` by default, which (a) blocks
        # ``requires_grad_(True)`` and (b) prevents ``autograd.grad``
        # from finding a graph.  Force-prediction models must explicitly
        # re-enable grad locally; ``.clone()`` lifts the position tensor
        # out of inference mode so it can carry gradients.
        with torch.enable_grad():
            positions: torch.Tensor = graph.pos.clone().requires_grad_(True)

            # ----- Per-element baseline -----
            h: torch.Tensor = self.embedding(graph.z)  # (N, embedding_dim)
            model_dtype: torch.dtype = h.dtype
            atom_energies: torch.Tensor = self.atom_readout(h).squeeze(-1)  # (N,)

            # ----- Pairwise distance term -----
            # Recompute edge lengths from positions so the autograd graph
            # links energy back to ``graph.pos`` (the precomputed
            # ``graph.edge_weight`` is detached from positions).
            row: torch.Tensor
            col: torch.Tensor
            row, col = graph.edge_index  # (E,), (E,)
            edge_vec: torch.Tensor = positions[col] - positions[row]  # (E, 3)
            # ``+ eps`` inside the sqrt prevents 0/0 NaN gradients when two
            # atoms happen to coincide (rare, but real for synthetic data).
            edge_len: torch.Tensor = (edge_vec.pow(2).sum(-1) + 1e-12).sqrt()  # (E,)

            # Bessel expansion (returns float64 by convention) → cast to model dtype.
            basis: torch.Tensor = self.radial_basis(edge_len).to(model_dtype)  # (E, K)
            cut_env: torch.Tensor = cosine_cutoff(edge_len, self.cutoff).to(model_dtype)  # (E,)
            pair_e: torch.Tensor = self.pair_readout(basis).squeeze(-1) * cut_env  # (E,)

            # Half-energy per directed edge (bidirectional convention).
            atom_energies = atom_energies + 0.5 * scatter(
                pair_e,
                row,
                dim=0,
                dim_size=positions.shape[0],
                reduce="sum",
            )

            # ----- Sum per graph -----
            energy: torch.Tensor = scatter(
                atom_energies,
                graph.batch,
                dim=0,
                reduce="sum",
            )  # (B,)

            # Forces via autograd — F = −∂E/∂R.  Energy now genuinely
            # depends on positions through the pair-distance term.
            grad: tuple[torch.Tensor, ...] = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=positions,
                create_graph=self.training,
                retain_graph=True,
            )
            forces: torch.Tensor = -grad[0]  # (N, 3)

        num_atoms: torch.Tensor = scatter(
            torch.ones(graph.z.shape[0], device=graph.z.device, dtype=energy.dtype),
            graph.batch,
            dim=0,
            reduce="sum",
        )  # (B,)

        return {
            "energy": energy,
            "forces": forces,
            "num_atoms": num_atoms,
        }
