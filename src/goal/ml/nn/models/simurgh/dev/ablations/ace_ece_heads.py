"""SIMURGH with two full energy heads, one ACE and one ECE, averaged; plus an agreement loss.

Each head is a complete energy model: E_k = sum_i E0 + scale (line_k + pair) + wall
(+ D3/electrostatics if on), with line_ACE the ACE core and line_ECE an ECE core of
its own; the heads share only the element references, the pair line with its wall,
and the energy scale.  The prediction is E = (E_ACE + E_ECE)/2 and
F = (F_ACE + F_ECE)/2 = -dE/dR (each head's forces by its own autograd pass).

Training: each head on the labels with the main loss.  For the MSE loss (the default)
1/2 [L(ACE) + L(ECE)] = L(mean) + 1/4 D, with D = w_E <(dE/N)^2> + w_F <|dF|^2> the
stage-weighted squared disagreement, so the main loss on the mean plus 1/4 D trains
each head on the labels; the agreement loss adds agreement_weight * D, ramped in like
MFC (mfc.start_epoch, mfc.warmup_epochs).  GOALModule passes the stage weights through
:meth:`set_loss_weights`.  MFC runs on each head; the shared pair line's curvature
penalty is counted once.

Outputs ``energy_ace``, ``energy_ece``, ``forces_ace`` and ``forces_ece`` for the
out-of-distribution score (per-atom energy and force disagreement).
Config: ``heads: {agreement_weight: 0.1}``.
"""

from __future__ import annotations

import torch

from goal.ml.nn.models.simurgh.dev.monolithic import SimurghDev, _ECECore
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

HEADS_DEFAULTS = {"agreement_weight": 0.1}


@MODEL_REGISTRY.register("simurgh_dev_abl_ace_ece_heads")
@BACKBONE_REGISTRY.register("simurgh_dev_abl_ace_ece_heads")
class SimurghAceEceHeads(SimurghDev):
    """SIMURGH with an ACE and an ECE energy head, averaged, and an agreement loss."""

    def __init__(self, **config) -> None:
        heads = {**HEADS_DEFAULTS, **dict(config.pop("heads", None) or {})}
        unknown = set(heads) - set(HEADS_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown heads key(s): {sorted(unknown)}")
        super().__init__(**config)
        if self.config["core"] != "ace" or self.ece_line is not None:
            raise ValueError("ace_ece_heads needs core: ace and no ECE bond line (pair_line phi)")
        self.ece_core = _ECECore(self.config, len(self.elements)).to(self.scale.dtype)
        self.agreement_weight = float(heads["agreement_weight"])
        self.head = "ace"
        self.loss_weights = {"energy": 1.0, "forces": 1.0}  # GOALModule sets the stage's

    def set_loss_weights(self, weights: dict[str, float]) -> None:
        """Loss weights in force (GOALModule calls this every step)."""
        self.loss_weights = dict(weights)

    def ramp(self) -> float:
        """0 before mfc.start_epoch, then linear over mfc.warmup_epochs (as MFC's weight)."""
        start = int(self.config["mfc"]["start_epoch"])
        warm = max(int(self.config["mfc"]["warmup_epochs"]), 1)
        return 0.0 if self.epoch < start else min(1.0, (self.epoch - start + 1) / warm)

    def line_energies(self, elem, edge_index, vectors, lengths):
        if self.head == "ece":
            e, h = self.ece_core(elem, edge_index, vectors, lengths, self.avg_num_neighbors)
            return e, h, self.ece_core.last_bond_energies
        return super().line_energies(elem, edge_index, vectors, lengths)

    def _regularisers(self, elem, edge_index, e_atom, emb) -> torch.Tensor:
        if self.head == "ace":
            return super()._regularisers(elem, edge_index, e_atom, emb)
        aux = e_atom.new_zeros(())  # the shared pair line's curvature: once, in the ACE pass
        penalty = self.config["pair_line"]["many_body_penalty"]
        return aux + penalty * e_atom.square().sum() if penalty else aux

    def forward(self, graph) -> dict[str, torch.Tensor]:
        try:
            self.head = "ace"
            ace = super().forward(graph)
            self.head = "ece"
            ece = super().forward(graph)
        finally:
            self.head = "ace"
        out = dict(ace)
        for key in ("energy", "forces", "atomic_energies", "energy_atom_line"):
            out[key] = 0.5 * (ace[key] + ece[key])
        out |= {"energy_ace": ace["energy"], "energy_ece": ece["energy"]}
        out |= {"forces_ace": ace["forces"], "forces_ece": ece["forces"]}
        if self.training:
            n = ace["num_atoms"].to(ace["energy"].dtype).clamp(min=1.0)
            d_e = ((ace["energy"] - ece["energy"]) / n).square().mean()
            d_f = (ace["forces"] - ece["forces"]).square().sum(-1).mean().to(d_e.dtype)
            w = self.loss_weights
            d = w.get("energy", 0.0) * d_e + w.get("forces", 0.0) * d_f
            lam = self.agreement_weight * self.ramp()
            out["aux_loss"] = ace["aux_loss"] + ece["aux_loss"] + (0.25 + lam) * d
            terms = {f"{k}_ace": v for k, v in ace.get("aux_terms", {}).items()}
            terms |= {f"{k}_ece": v for k, v in ece.get("aux_terms", {}).items()}
            terms |= {"agreement_energy": d_e.detach(), "agreement_forces": d_f.detach()}
            out["aux_terms"] = terms
        return out
