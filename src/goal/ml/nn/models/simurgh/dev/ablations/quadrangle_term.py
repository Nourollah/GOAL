"""Explicit dihedral (quadrangle) term in the layer-2 messages.

Node vector ``u_i = Σ_j u(r_ij) r̂_ij`` (enveloped, so smooth at the
cutoff), dihedral edge scalar ``d_ij = (u_i × r̂_ij) · (r̂_ij × u_j)``, fed
into the layer-2 messages through ``m_ij <- m_ij (1 + g_ij)`` with
``g_ij = a · SiLU(W [h_i⁰ + E(d_ij) + h_j⁰]) · u(r_ij)`` and no softmax.  ``a``
starts at zero.  Two ACE layers at ν = 3 already contain dihedral
information; this tests whether an explicit term still helps.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from goal.ml.nn.models.simurgh.dev.monolithic import SimurghDev
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


class _QuadrangleGate(nn.Module):
    """Per-edge factor ``1 + g_ij`` from the dihedral scalar ``d_ij``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.embed = nn.Linear(1, channels, bias=False)
        self.mix = nn.Linear(channels, 1, bias=False)
        self.amplitude = nn.Parameter(torch.zeros(()))
        self.channels = channels

    def forward(self, h, edge_index, sh, env) -> torch.Tensor:
        """``(E,)`` message factors."""
        sender, receiver = edge_index
        rhat = sh[:, 1:4] / math.sqrt(3.0)  # unit vector (component-normalised l = 1)
        u = torch.zeros(h.shape[0], 3, dtype=sh.dtype, device=sh.device)
        u = u.index_add(0, receiver, env.unsqueeze(-1) * rhat)
        d = (torch.cross(u[receiver], rhat, dim=-1) * torch.cross(rhat, u[sender], dim=-1)).sum(-1)
        scalars = h[:, : self.channels]
        x = scalars[receiver] + self.embed(d.unsqueeze(-1)) + scalars[sender]
        return 1.0 + self.amplitude * nn.functional.silu(self.mix(x)).squeeze(-1) * env


@MODEL_REGISTRY.register("simurgh_dev_abl_quadrangle_term")
@BACKBONE_REGISTRY.register("simurgh_dev_abl_quadrangle_term")
class SimurghQuadrangleTerm(SimurghDev):
    """SIMURGH with the quadrangle gate on the second interaction layer."""

    def __init__(self, **config) -> None:
        super().__init__(**config)
        if len(self.atom_line.layers) < 2:
            raise ValueError("the quadrangle term acts on layer 2: num_interactions >= 2")
        gate = _QuadrangleGate(self.config["num_channels"])
        self.atom_line.layers[1].message_gate = gate.to(self.scale.dtype)
