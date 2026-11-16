"""SIMURGH with ACE atom states and ECE bond states exchanging messages in every layer.

The ECE bond line (``pair_line.mode: phi+ece``) reads the ACE features after the
first interaction instead of the last, and its message-passing rounds are
interleaved with the ACE interactions:

    h_1 = ACE_1(h_0);  a_0 = ECE_init(h_1);  (a_1, m_1) = ECE_round_1(a_0)
    for t = 2..T:
        h_t = ACE_t(h_{t-1} + B2A_{t-1}(m_{t-1}))         bond -> atom
        a_{t-1} <- a_{t-1} + A2B_t(h_t at both ends)        atom -> bond
        (a_t, m_t) = ECE_round_t(a_{t-1})
    E = sum_i [ACE readouts]_i + sum_(i->j) eps_ij(a_T)    (+ pair line, wall, E0)

``m_t`` is the round's own bond -> atom aggregate (global frame, one block per
degree); B2A maps it to the ACE irreps and A2B maps both ends' projected ACE
features, rotated into the bond frame, onto the bond state.  B2A, A2B and the
ECE energy head start at zero, so a fresh model equals the ACE-only model.
Forces are -dE/dR (one autograd pass); the MFC windows run the same fused
forward through :meth:`line_energies`.  Needs ``ece.num_mp_layers`` equal to
``num_interactions``.
"""

from __future__ import annotations

import torch
from e3nn import o3
from torch import nn

from goal.ml.nn.blocks.ece import to_edge_frame
from goal.ml.nn.models.simurgh.dev.monolithic import (
    SimurghDev,
    _blocks,
    polynomial_envelope,
)
from goal.ml.nn.primitives.o2 import O2Linear
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


@MODEL_REGISTRY.register("simurgh_dev_abl_ace_ece_fusion")
@BACKBONE_REGISTRY.register("simurgh_dev_abl_ace_ece_fusion")
class SimurghAceEceFusion(SimurghDev):
    """SIMURGH whose ACE layers and ECE rounds exchange messages (deep fusion)."""

    def __init__(self, **config) -> None:
        super().__init__(**config)
        if self.ece_line is None:
            raise ValueError(
                "ace_ece_fusion needs an ECE bond line: pair_line.mode phi+ece or ece"
            )
        layers, ece = self.atom_line.layers, self.ece_line
        n_layers = len(layers)
        if len(ece.block.msg) != n_layers:
            raise ValueError("ace_ece_fusion needs ece.num_mp_layers equal to num_interactions")
        if any(layer.irreps_out != layers[-1].irreps_out for layer in layers):
            raise ValueError("ace_ece_fusion needs every ACE layer to output the same irreps")
        block = ece.block
        irreps_m = o3.Irreps([(block.width, (ell, (-1) ** ell)) for ell in range(block.lmax + 1)])
        self.b2a = nn.ModuleList(
            [o3.Linear(irreps_m, layers[t].irreps_out) for t in range(n_layers - 1)]
        )
        self.a2b_project = nn.ModuleList(
            [o3.Linear(layers[t].irreps_out, ece.irreps_dens) for t in range(1, n_layers)]
        )
        self.a2b = nn.ModuleList(
            [
                O2Linear(block.state, block.state, block.mmax, bias=False)
                for _ in range(1, n_layers)
            ]
        )
        with torch.no_grad():  # every cross-link starts at zero
            for link in self.b2a:
                link.weight.zero_()
            for link in self.a2b:
                link.weight.zero_()
        self.to(self.scale.dtype)

    def line_energies(self, elem, edge_index, vectors, lengths):
        """ACE atomic energies plus the ECE atomic split, with the layers exchanging messages."""
        line, ece = self.atom_line, self.ece_line
        block, avg, n = ece.block, self.avg_num_neighbors, elem.shape[0]
        i, j = edge_index
        ctx = line.prepare(elem, edge_index, vectors, lengths)
        h, energy = line.step(0, ctx["h"], ctx, elem, edge_index, lengths, avg)
        env = polynomial_envelope(lengths / ece.cutoff, ece.p)
        frames = ece.frames(vectors)
        a = block.init_state(
            _blocks(ece.project(h), ece.irreps_dens),
            edge_index,
            frames,
            self.ece_codes.weight[elem],
            ece.rbf(lengths),
        )
        a, m = block.mp_round(0, a, edge_index, frames, env, avg, n)
        for t in range(1, len(line.layers)):
            h = h + self.b2a[t - 1](torch.cat([x.reshape(n, -1) for x in m], dim=-1))
            h, e = line.step(t, h, ctx, elem, edge_index, lengths, avg)
            energy = energy + e
            dens = _blocks(self.a2b_project[t - 1](h), ece.irreps_dens)
            ends = torch.cat(
                [
                    to_edge_frame([x[i] for x in dens], frames, block.mmax),
                    to_edge_frame([x[j] for x in dens], frames, block.mmax),
                ],
                dim=1,
            )
            a = a + self.a2b[t - 1](ends)
            a, m = block.mp_round(t, a, edge_index, frames, env, avg, n)
        out = ece.head(a[..., block.mmax], env, edge_index, n)
        return energy + out["atomic_energies"], h, out["bond_energies"]
