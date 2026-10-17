"""Adaptive depth gate — optional add-on for ARACE rounds.

Every atom currently receives the same number of message-passing rounds.
Physically that is wasteful: an interior atom in a saturated chain is
converged after one round, while an atom at a reactive centre or a
fragment boundary keeps changing.  This module learns a per-atom gate
that decides how much of each round's update an atom actually keeps.

Two regimes, deliberately different:

**Training (soft)**
    ``h_next = g_i · h_new + (1 - g_i) · h_prev`` — fully differentiable,
    every atom still updates, but the gate magnitude shapes the gradient
    each atom receives.  Atoms the gate calls converged get smaller
    updates, pushing capacity towards the hard ones.

**Inference (hard)**
    ``g_i > threshold`` keeps ``h_new``, otherwise the atom keeps
    ``h_prev`` and effectively exits early.  This is where the compute
    saving lives.  Never used during training — a step function has zero
    gradient, so training through it would silently freeze the gate.

**Why the gate is invariant, and why that is not a compromise.**  For
``h_next = g·h_new + (1-g)·h_prev`` to be equivariant, ``g`` *must* be
invariant: a gate that rotated with the frame would scale different
irrep blocks by rotation-dependent factors and destroy the symmetry.
This is the same construction as :class:`e3nn.nn.Gate` — invariant
scalars gating equivariant blocks — not a scalars-only shortcut.

What *is* a real choice is how finely the gate resolves the feature
vector, and ``per_irrep`` controls it:

* ``per_irrep=False`` — one gate for the whole vector.
* ``per_irrep=True`` (recommended) — one gate per irrep block, each still
  invariant, each applied to its own block.  The model can then keep the
  ``l = 1`` update of an atom while damping a noisy ``l = 2`` one, which a
  single scalar cannot express.  The convergence signal is likewise split
  per block, so the gate sees *which* angular channel is still moving
  instead of one blended norm.

Enabled per config section (``model.backbone.adaptive_gate``); absent
section → the ARACE round never builds the module and the forward path is
bit-identical to plain ARACE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn.o3 import Irreps


class AdaptiveDepthGate(nn.Module):
    """Per-atom gate over a round's node-feature update.

    Parameters
    ----------
    irreps_node : Irreps or str
        Node feature irreps.  Used to locate the ``l = 0`` channels,
        which must be the leading block (standard ARACE ordering).
    n_scalar : int
        Width of the gate's scalar input projection.
    hard_threshold : float
        Gate value above which an atom updates at inference (default
        ``0.5``).
    aux_loss_weight : float
        Weight ``λ`` on the auxiliary sparsity loss ``λ · mean(g_i)``,
        which pushes the gate to be selective rather than permanently
        open.  ``0.0`` disables it (the gate then only ever feels the
        energy/force gradients).  Default ``0.01``.
    init_bias : float
        Initial bias on the gate logits (default ``1.0`` →
        ``sigmoid(1) ≈ 0.73``, i.e. open).  Starting open means the model
        begins as (nearly) standard ARACE and must *learn* to close
        specific atoms, which is far more stable than starting closed and
        learning to open.
    per_irrep : bool
        ``True`` (recommended) → one invariant gate per irrep block, and
        one convergence ratio per block as extra gate input.  ``False`` →
        a single gate and a single global convergence ratio for the whole
        feature vector.  Both are exactly equivariant; the per-block form
        is strictly more expressive.
    """

    def __init__(
        self,
        irreps_node: Irreps | str,
        n_scalar: int = 16,
        hard_threshold: float = 0.5,
        aux_loss_weight: float = 0.01,
        init_bias: float = 1.0,
        per_irrep: bool = True,
    ) -> None:
        super().__init__()
        self.irreps_node: Irreps = Irreps(irreps_node)
        self.n_l0: int = sum(mul for mul, ir in self.irreps_node if ir.l == 0)
        if self.n_l0 == 0:
            raise ValueError(
                f"AdaptiveDepthGate needs scalar channels, but irreps_node "
                f"{self.irreps_node} has none."
            )
        # ``h[:, :n_l0]`` is only the scalars if every l=0 block leads.
        leading: int = 0
        for mul, ir in self.irreps_node:
            if ir.l != 0:
                break
            leading += mul
        if leading != self.n_l0:
            raise ValueError(
                f"AdaptiveDepthGate requires the l=0 channels of irreps_node "
                f"to be the leading block; got {self.irreps_node} ({leading} "
                f"leading scalars vs {self.n_l0} in total).  Reorder the irreps "
                "so all scalars come first."
            )
        if int(n_scalar) < 1:
            raise ValueError(f"n_scalar must be >= 1, got {n_scalar}.")

        self.n_scalar: int = int(n_scalar)
        self.hard_threshold: float = float(hard_threshold)
        self.aux_loss_weight: float = float(aux_loss_weight)
        self.per_irrep: bool = bool(per_irrep)

        # Flat column span of each irrep block, so a per-block gate can be
        # expanded back to the full feature width.
        self._block_dims: list[int] = [mul * ir.dim for mul, ir in self.irreps_node]
        self.n_gates: int = len(self._block_dims) if self.per_irrep else 1

        # Gate network: projected scalars + one convergence ratio per gate.
        self.atom_to_scalar: nn.Linear = nn.Linear(self.n_l0, self.n_scalar, bias=False)
        self.gate_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self.n_scalar + self.n_gates, self.n_scalar),
            nn.SiLU(),
            nn.Linear(self.n_scalar, self.n_gates),
        )

        # Start open: the model begins as plain ARACE and learns to close.
        nn.init.constant_(self.gate_mlp[-1].bias, float(init_bias))

    def _convergence_ratios(
        self,
        h_new: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Invariant ``‖Δh‖ / ‖h_prev‖``, per irrep block or globally.

        Every ratio is divided by the norm of the **whole** previous
        feature vector, never by the block's own norm.  A symmetry-
        suppressed block (the ``l = 1`` features of a tetrahedral centre
        vanish identically) would otherwise put numerical cancellation
        residue in the denominator and hand the MLP an O(1/ε) input with
        enormous position gradients — the same trap documented on
        :class:`~goal.ml.nn.blocks.artisans._PairRMSNorm`.  The global
        norm is bounded away from zero by the scalar channels.
        """
        denominator: torch.Tensor = h_prev.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        delta: torch.Tensor = h_new - h_prev
        if not self.per_irrep:
            return delta.norm(dim=-1, keepdim=True) / denominator  # (N, 1)

        ratios: list[torch.Tensor] = []
        start: int = 0
        for dim in self._block_dims:
            block: torch.Tensor = delta[:, start : start + dim]
            ratios.append(block.norm(dim=-1, keepdim=True) / denominator)
            start += dim
        return torch.cat(ratios, dim=-1)  # (N, n_blocks)

    def _expand_gates(self, gates: torch.Tensor) -> torch.Tensor:
        """``(N, n_gates)`` → ``(N, irreps.dim)``, one gate per irrep block."""
        if not self.per_irrep:
            return gates  # (N, 1) broadcasts over the whole vector
        return torch.repeat_interleave(
            gates,
            torch.tensor(self._block_dims, device=gates.device),
            dim=-1,
        )

    def forward(
        self,
        h_new: torch.Tensor,
        h_prev: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate this round's update.

        Parameters
        ----------
        h_new : Tensor ``(N, irreps_node.dim)``
            Node features produced by this round.
        h_prev : Tensor ``(N, irreps_node.dim)``
            Node features entering this round.
        training : bool
            ``True`` → soft (differentiable) gating; ``False`` → hard
            threshold gating.  Callers pass ``self.training`` so the
            regime follows ``model.train()`` / ``model.eval()``.

        Returns
        -------
        h_next : Tensor ``(N, irreps_node.dim)``
            Gated node features.
        aux_loss : Tensor, 0-d
            Sparsity penalty ``λ · mean(g)``; exactly zero in hard mode
            (nothing to regularise when no gradient flows) and when
            ``aux_loss_weight == 0``.
        gate_scores : Tensor ``(N,)``
            Per-atom gate values in ``(0, 1)`` — for logging and
            diagnostics.  With ``per_irrep=True`` this is the mean over
            the per-block gates, so the logged curve keeps one meaning
            regardless of the setting.
        """
        if h_new.shape != h_prev.shape:
            raise ValueError(
                f"h_new {tuple(h_new.shape)} and h_prev {tuple(h_prev.shape)} "
                "must have the same shape."
            )

        # Invariant inputs: scalars of the updated features …
        l0: torch.Tensor = h_new[:, : self.n_l0]  # (N, n_l0)
        scalars: torch.Tensor = self.atom_to_scalar(l0)  # (N, n_scalar)

        # … and the relative change, one ratio per gate (both invariant).
        ratios: torch.Tensor = self._convergence_ratios(h_new, h_prev)  # (N, n_gates)

        gate_input: torch.Tensor = torch.cat([scalars, ratios], dim=-1)
        gate_logits: torch.Tensor = self.gate_mlp(gate_input)  # (N, n_gates)
        gates: torch.Tensor = torch.sigmoid(gate_logits)  # (N, n_gates)

        if training:
            expanded: torch.Tensor = self._expand_gates(gates)
            h_next: torch.Tensor = expanded * h_new + (1.0 - expanded) * h_prev
            aux_loss: torch.Tensor = self.aux_loss_weight * gates.mean()
        else:
            hard: torch.Tensor = (gates > self.hard_threshold).to(h_new.dtype)
            expanded = self._expand_gates(hard)
            h_next = expanded * h_new + (1.0 - expanded) * h_prev
            aux_loss = torch.zeros((), device=h_new.device, dtype=h_new.dtype)

        # One number per atom for logging, whatever the gate resolution.
        gate_scores: torch.Tensor = gates.mean(dim=-1)  # (N,)
        return h_next, aux_loss, gate_scores

    def extra_repr(self) -> str:
        """Summarise the gate sizes, threshold and aux-loss weight for ``repr``."""
        return (
            f"n_l0={self.n_l0}, n_scalar={self.n_scalar}, "
            f"n_gates={self.n_gates}, per_irrep={self.per_irrep}, "
            f"hard_threshold={self.hard_threshold}, "
            f"aux_loss_weight={self.aux_loss_weight}"
        )
