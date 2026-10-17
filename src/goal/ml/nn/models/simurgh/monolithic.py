"""Self-contained SIMURGH model.

Satisfies the ``MonolithicModel`` protocol — takes an ``AtomicGraph``
and returns a dictionary of predicted properties directly.  Uses the
same :class:`EnvironmentDressing` and :class:`SimurghArtisanBank` building
blocks as the modular backbone, plus a self-contained force pathway
with the same three modes as :class:`DualForcesHead`
(``autograd`` / ``direct`` / ``hybrid``).

Configure with ``head: null`` in the Hydra config — the training loop
then calls this model directly.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.artisans import ArtisanConfig, SimurghArtisanBank
from goal.ml.nn.blocks.env_dressing import EnvironmentDressing
from goal.ml.nn.models.simurgh.geometry import differentiable_edges
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


@MODEL_REGISTRY.register("simurgh_monolithic")
@BACKBONE_REGISTRY.register("simurgh_monolithic")
class SimurghMonolithic(nn.Module):
    """Self-contained SIMURGH model.

    Returns a property dictionary with ``"energy"``, ``"forces"`` (and
    ``"num_atoms"``) directly.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.
    dressing_kwargs : dict
        Forwarded to :class:`EnvironmentDressing`.  Includes
        ``body_order`` for the ACE body-order expansion.
    artisan_config : dict
        Forwarded to :class:`ArtisanConfig`.
    cutoff : float, optional
        Cosine-cutoff radius for the experts.  Defaults to the dressing
        cutoff if unset.
    forces_mode : str
        ``"autograd"`` (default), ``"direct"`` or ``"hybrid"``.
    correction_weight : float
        Weight on the direct correction term when ``forces_mode ==
        "hybrid"``.  Ignored otherwise.
    """

    def __init__(
        self,
        elements: typing.Sequence[int] = (1, 6, 7, 8),
        dressing_kwargs: dict[str, typing.Any] | None = None,
        artisan_config: dict[str, typing.Any] | None = None,
        cutoff: float | None = None,
        forces_mode: str = "autograd",
        correction_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if forces_mode not in {"autograd", "direct", "hybrid"}:
            raise ValueError(
                f"forces_mode must be 'autograd', 'direct' or 'hybrid', got '{forces_mode}'."
            )

        dressing_cfg: dict[str, typing.Any] = dict(dressing_kwargs or {})
        artisan_cfg: dict[str, typing.Any] = dict(artisan_config or {})
        if "hidden_dims" in artisan_cfg:
            artisan_cfg["hidden_dims"] = tuple(int(x) for x in artisan_cfg["hidden_dims"])

        # Normalise the nested equivariant sub-config (may arrive as an
        # OmegaConf DictConfig) to a plain dict for ArtisanConfig.
        if artisan_cfg.get("equivariant") is not None:
            artisan_cfg["equivariant"] = {
                str(k): v for k, v in dict(artisan_cfg["equivariant"]).items()
            }

        self.dressing: EnvironmentDressing = EnvironmentDressing(**dressing_cfg)
        bank_cutoff: float = cutoff if cutoff is not None else self.dressing.cutoff
        self.artisan_bank: SimurghArtisanBank = SimurghArtisanBank(
            elements=elements,
            irreps_in=self.dressing.irreps_out,
            artisan_config=ArtisanConfig(**artisan_cfg),
            cutoff=bank_cutoff,
        )

        self.forces_mode: str = forces_mode
        self.correction_weight: float = correction_weight
        if forces_mode in {"direct", "hybrid"}:
            self.force_proj: EquivariantLinear = EquivariantLinear(
                self.dressing.irreps_out, Irreps("1x1o")
            )
        else:
            self.force_proj = None  # type: ignore[assignment]

        self._irreps_out: Irreps = self.dressing.irreps_out

    # ------------------------------------------------------------------
    # MonolithicModel protocol
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[str]:
        """Return list of output keys produced by the model."""
        return ["energy", "forces"]

    @property
    def elements(self) -> tuple[int, ...]:
        """Return the atomic numbers covered by this model."""
        return self.artisan_bank.elements

    @property
    def num_artisans(self) -> int:
        """Return the total number of pair-based experts."""
        return self.artisan_bank.num_artisans

    @property
    def irreps_out(self) -> Irreps:
        """Return output irreps from the environment dressing block."""
        return self._irreps_out

    @property
    def body_order(self) -> int:
        """ACE body-order parameter inherited from the dressing block."""
        return self.dressing.body_order

    def gates(self) -> dict[str, torch.Tensor]:
        """Return expert gating weights indexed by atomic pair type."""
        return self.artisan_bank.gates()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Forward pass: returns dict with 'energy', 'forces', and 'num_atoms'.

        Parameters
        ----------
        graph : AtomicGraph
            Input atomic graph with positions, atomic numbers, and edges.

        Returns
        -------
        dict[str, torch.Tensor]
            Output dict containing 'energy', 'forces', and 'num_atoms'.
        """
        positions: torch.Tensor = graph.pos
        if self.forces_mode != "direct":
            positions.requires_grad_(True)

        edge_vectors: torch.Tensor
        edge_lengths: torch.Tensor
        edge_vectors, edge_lengths = differentiable_edges(graph, positions)

        dressed: torch.Tensor
        dressed, _layer_e = self.dressing(
            atomic_numbers=graph.atomic_numbers,
            edge_index=graph.edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
        )  # (N, irreps_out.dim), layer_energies ignored in monolithic path

        node_energies: torch.Tensor = self.artisan_bank(
            atom_features=dressed,
            atomic_numbers=graph.atomic_numbers,
            edge_index=graph.edge_index,
            edge_lengths=edge_lengths,
            edge_vectors=edge_vectors,
        )  # (N,)

        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(graph.num_atoms, dtype=torch.long, device=node_energies.device)
        )
        energy: torch.Tensor = scatter(node_energies, batch, dim=0, reduce="sum")  # (B,)
        num_atoms: torch.Tensor = scatter(
            torch.ones_like(node_energies), batch, dim=0, reduce="sum"
        )  # (B,)

        outputs: dict[str, torch.Tensor] = {
            "energy": energy,
            "num_atoms": num_atoms,
        }

        # ----- forces -----
        if self.forces_mode == "direct":
            forces_direct: torch.Tensor = self.force_proj(dressed)  # (N, 3)
            outputs["forces"] = forces_direct
            return outputs

        # autograd-derived
        grad_outputs: tuple[torch.Tensor, ...] = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=positions,
            create_graph=self.training,
            retain_graph=True,
        )
        forces_auto: torch.Tensor = -grad_outputs[0]  # (N, 3)

        if self.forces_mode == "autograd":
            outputs["forces"] = forces_auto
            return outputs

        # hybrid
        forces_direct = self.force_proj(dressed)
        outputs["forces"] = forces_auto + self.correction_weight * forces_direct
        return outputs
