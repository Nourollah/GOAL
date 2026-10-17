"""Dual-forces head — autograd-derived forces with optional corrections.

Four force-prediction modes are supported, all selectable from config:

* ``"autograd"`` (default)
    Forces come solely from ``-grad(E, positions)``.  Energy-conserving.
* ``"direct"``
    Forces are predicted by a small equivariant head from the backbone
    features (``EquivariantLinear → 1x1o``).  Non-conservative but often
    quicker and sometimes more accurate when force labels are clean.
* ``"hybrid"``
    Forces are the autograd-derived gradient *plus* a learnable
    correction predicted by the direct head, with a configurable
    ``correction_weight`` controlling the magnitude::

        F = -grad(E, r)  +  correction_weight * F_direct

    The autograd term is the **leading** contribution; the direct head
    plays a residual / correction role.  Set ``correction_weight`` low
    (e.g. ``0.05``) to keep the conservative term dominant whilst still
    allowing the head to absorb systematic errors.
* ``"pairwise"``
    Forces come from the backbone's per-pair autograd
    (``features.node_forces``).  Newton's third law is exact by
    construction.  When ``pairwise_correction_weight > 0`` the
    output is combined: ``F = F_autograd + w * F_pairwise``;
    otherwise it's pure pairwise.  Requires a backbone with
    ``compute_pairwise_forces=True`` (SIMURGH).

Set ``mode`` in the Hydra config — defaults to ``"autograd"`` so the
behaviour matches ``EnergyForcesHead`` out of the box.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.readout import ScalarReadout
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("dual_forces")
class DualForcesHead(nn.Module):
    """Head producing energy and forces, with selectable force-prediction mode.

    Parameters
    ----------
    irreps_in : str or Irreps
        Input irreps from the backbone.  Must contain ``1o`` channels
        when ``mode`` uses the direct head.
    hidden_dim : int
        Width of the scalar readout MLP.
    mode : str
        ``"autograd"``, ``"direct"``, ``"hybrid"`` or ``"pairwise"``.
    correction_weight : float
        Multiplier on the direct correction term in ``"hybrid"`` mode.
        Ignored otherwise.
    pairwise_correction_weight : float
        When ``mode == "pairwise"`` and this weight is ``> 0``, output
        forces are ``F_autograd + w * F_pairwise`` (combined mode);
        when ``0`` (default), output is pure pairwise.  Ignored in
        the other three modes.
    """

    _SUPPORTED_MODES: typing.ClassVar[set[str]] = {
        "autograd",
        "direct",
        "hybrid",
        "pairwise",
    }

    def __init__(
        self,
        irreps_in: str | Irreps,
        hidden_dim: int = 64,
        mode: str = "autograd",
        correction_weight: float = 0.1,
        pairwise_correction_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.irreps_in: Irreps = Irreps(irreps_in)
        if mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"DualForcesHead.mode must be one of "
                f"{sorted(self._SUPPORTED_MODES)}, got '{mode}'."
            )
        self.mode: str = mode
        self.correction_weight: float = correction_weight
        self.pairwise_correction_weight: float = pairwise_correction_weight

        self.readout: ScalarReadout = ScalarReadout(
            irreps_in=self.irreps_in, hidden_dim=hidden_dim
        )

        if mode in {"direct", "hybrid"}:
            self.force_proj: EquivariantLinear = EquivariantLinear(self.irreps_in, Irreps("1x1o"))
        else:
            self.force_proj = None  # type: ignore[assignment]

        self._output_keys: list[str] = ["energy", "forces"]

    @property
    def output_keys(self) -> list[str]:
        return self._output_keys

    def _energy(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Energy + helper tensors (batch index, num_atoms per graph)."""
        # Backbones that already produce per-atom energies (e.g. SIMURGH)
        # populate ``features.node_energies``; honour it when present.
        if features.node_energies is not None:
            node_energies: torch.Tensor = features.node_energies  # (N,)
        else:
            node_energies = self.readout(features.node_feats).squeeze(-1)  # (N,)
        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(graph.num_atoms, dtype=torch.long, device=node_energies.device)
        )
        energy: torch.Tensor = scatter(node_energies, batch, dim=0, reduce="sum")  # (B,)
        num_atoms: torch.Tensor = scatter(
            torch.ones_like(node_energies), batch, dim=0, reduce="sum"
        )  # (B,)
        return energy, batch, num_atoms

    def forward(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> dict[str, torch.Tensor]:
        """Predict total energy and forces according to ``self.mode``.

        The energy is the per-graph sum of ``features.node_energies``
        (when the backbone supplies them) or of the scalar readout.
        Forces depend on the mode:

        * ``"direct"``: ``F = force_proj(node_feats)``; no autograd.
        * ``"pairwise"``: ``F = features.node_forces``; if
          ``pairwise_correction_weight > 0`` and ``graph.pos`` tracks
          gradients, ``F = -dE/dr + w * node_forces`` instead.
        * ``"autograd"``: ``F = -dE/dr``.
        * ``"hybrid"``: ``F = -dE/dr + correction_weight * F_direct``.

        In ``"autograd"`` and ``"hybrid"`` modes the forces are all
        zeros when ``graph.pos.requires_grad`` is ``False``.  The
        autograd graph is kept (``create_graph``) only in training mode.

        Parameters
        ----------
        features : NodeFeatures
            Backbone output; reads ``node_energies`` ``(N,)`` or
            ``node_feats`` ``(N, irreps_in.dim)``, and ``node_forces``
            ``(N, 3)`` in ``"pairwise"`` mode.
        graph : AtomicGraph
            Input graph providing ``pos`` ``(N, 3)`` in Å and ``batch``
            ``(N,)``.

        Returns
        -------
        dict[str, Tensor]
            ``'energy'`` ``(B,)`` in eV, ``'num_atoms'`` ``(B,)`` (float
            atom counts) and ``'forces'`` ``(N, 3)`` in eV/Å.

        Raises
        ------
        ValueError
            In ``"pairwise"`` mode when ``features.node_forces`` is
            ``None``.
        """
        energy: torch.Tensor
        num_atoms: torch.Tensor
        energy, _batch, num_atoms = self._energy(features, graph)

        outputs: dict[str, torch.Tensor] = {
            "energy": energy,
            "num_atoms": num_atoms,
        }

        # ------------------------------------------------------------------
        # Force prediction
        # ------------------------------------------------------------------

        if self.mode == "direct":
            forces_direct: torch.Tensor = self.force_proj(features.node_feats)  # (N, 3)
            outputs["forces"] = forces_direct
            return outputs

        if self.mode == "pairwise":
            if features.node_forces is None:
                raise ValueError(
                    "DualForcesHead.mode='pairwise' requires the backbone to "
                    "populate features.node_forces (e.g. SIMURGH with "
                    "compute_pairwise_forces=True)."
                )
            forces_pairwise: torch.Tensor = features.node_forces  # (N, 3)
            if self.pairwise_correction_weight > 0.0 and graph.pos.requires_grad:
                # Combined mode: F = F_autograd + w * F_pairwise
                grad_outputs_c: tuple[torch.Tensor, ...] = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=graph.pos,
                    create_graph=self.training,
                    retain_graph=True,
                )
                forces_auto_c: torch.Tensor = -grad_outputs_c[0]
                outputs["forces"] = (
                    forces_auto_c + self.pairwise_correction_weight * forces_pairwise
                )
            else:
                outputs["forces"] = forces_pairwise
            return outputs

        # autograd-based: F = -∂E/∂r
        if not graph.pos.requires_grad:
            # No grad tracking → zero forces (matches EnergyForcesHead)
            outputs["forces"] = torch.zeros_like(graph.pos)
            return outputs

        grad_outputs: tuple[torch.Tensor, ...] = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=graph.pos,
            create_graph=self.training,
            retain_graph=True,
        )
        forces_auto: torch.Tensor = -grad_outputs[0]  # (N, 3)

        if self.mode == "autograd":
            outputs["forces"] = forces_auto
            return outputs

        # hybrid: leading autograd + weighted direct correction
        forces_direct = self.force_proj(features.node_feats)  # (N, 3)
        outputs["forces"] = forces_auto + self.correction_weight * forces_direct
        return outputs
