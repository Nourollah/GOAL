"""Edge cluster expansion (ECE): persistent bond states updated on the line graph.

Every directed bond ``i -> j`` carries a state built from the atomic
densities of both end atoms, expressed in the bond's own frame, where only
O(2) remains (:mod:`goal.ml.nn.primitives.o2`).  Bond states are updated
by bond -> atom -> bond message passing.  Design from the paper equations
of the ECE network (LaCour and Head-Gordon, arXiv:2609.23134); no code
adapted.

* edge construction ``A^{ij} = [D(r̂_ij) A^i, D(r̂_ij) A^j]``, ``|m| <= mmax``;
* FiLM: per-channel scale (shared by ``±m``) and ``m = 0`` shift from an MLP
  of ``[e_Zi, e_Zj, RBF(r_ij)]``;
* residual equivariant blocks ``f(A) = A + W2 σ̃(W1 A)``;
* message passing, repeated ``num_mp_layers`` times: ``M^{ij} = f_msg(A^{ij})``
  with per-head, per-degree gates ``s^{ij}_{h,l}`` from extra ``m = 0``
  channels; the l-labelled message is rotated back to the global frame,
  aggregated at ``j`` with the envelope ``u(r_ij)``, rotated into each
  outgoing bond ``j -> k`` and added to ``A^{jk}`` through ``f_rcv``.  No atomic
  features persist between layers.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import scatter

from goal.ml.nn.primitives.o2 import FourierActivation, O2Linear, O2Product

__all__ = ["ECEBlock", "from_edge_frame", "to_edge_frame"]


def to_edge_frame(
    blocks: list[torch.Tensor], frames: list[torch.Tensor], mmax: int
) -> torch.Tensor:
    """Global per-degree features ``[(E, C, 2l+1)]`` -> edge-frame O(2) layout ``(E, C(L+1), 2M+1)``."""
    out = []
    for ell, (x, d) in enumerate(zip(blocks, frames)):
        y = torch.einsum("enm,ecm->ecn", d, x)  # D^l x, components m = -l..l
        padded = y.new_zeros(y.shape[0], y.shape[1], 2 * mmax + 1)
        keep = min(ell, mmax)
        padded[..., mmax - keep : mmax + keep + 1] = y[..., ell - keep : ell + keep + 1]
        out.append(padded)
    return torch.cat(out, dim=1)


def from_edge_frame(
    x: torch.Tensor, frames: list[torch.Tensor], channels: int
) -> list[torch.Tensor]:
    """Edge-frame O(2) layout -> global per-degree features (components ``|m| > mmax`` are 0)."""
    mmax = (x.shape[-1] - 1) // 2
    out = []
    for ell, d in enumerate(frames):
        block = x[:, ell * channels : (ell + 1) * channels]
        y = block.new_zeros(block.shape[0], channels, 2 * ell + 1)
        keep = min(ell, mmax)
        y[..., ell - keep : ell + keep + 1] = block[..., mmax - keep : mmax + keep + 1]
        out.append(torch.einsum("enm,ecn->ecm", d, y))  # D^lᵀ y
    return out


class _Activation(nn.Module):
    """``fourier`` (pointwise SiLU on the circle) or exact ``o2_product``."""

    def __init__(self, kind: str, channels: int, mmax: int, grid_points: int) -> None:
        super().__init__()
        if kind == "fourier":
            self.fn = FourierActivation(mmax, grid_points)
        elif kind == "o2_product":
            self.fn = O2Product(channels, channels, mmax)
        else:
            raise ValueError(f"ece.activation must be fourier or o2_product, got {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Equivariant nonlinearity, same shape."""
        return self.fn(x)


class _Residual(nn.Module):
    """``f(A) = A + W2 σ̃(W1 A)``."""

    def __init__(self, channels: int, bottleneck: int, mmax: int, kind: str, grid: int) -> None:
        super().__init__()
        self.w1 = O2Linear(channels, bottleneck, mmax)
        self.act = _Activation(kind, bottleneck, mmax, grid)
        self.w2 = O2Linear(bottleneck, channels, mmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual update of bond states."""
        return x + self.w2(self.act(self.w1(x)))


class ECEBlock(nn.Module):
    """Bond states from atomic densities, refined by line-graph message passing.

    Parameters
    ----------
    channels : int
        Channels per degree ``l`` of the atomic densities.
    lmax, mmax : int
        Largest degree of the densities and largest azimuthal frequency kept.
    bottleneck, message_width, heads : int
        Residual-block width, message channels per degree, gate heads.
    num_mp_layers : int
        Bond -> atom -> bond rounds (0: strictly local bond states).
    element_dim, film_mlp, num_rbf : int, list, int
        FiLM conditioning on the end elements and the bond length.
    activation, grid_points :
        ``fourier`` (with ``grid_points`` angles) or ``o2_product``.
    """

    def __init__(
        self,
        channels: int,
        lmax: int,
        mmax: int,
        bottleneck: int,
        message_width: int,
        heads: int,
        num_mp_layers: int,
        element_dim: int,
        film_mlp: list[int],
        num_rbf: int,
        activation: str = "fourier",
        grid_points: int = 16,
    ) -> None:
        super().__init__()
        if message_width % heads:
            raise ValueError("ece.message_width must be divisible by ece.heads")
        self.channels, self.lmax, self.mmax = channels, lmax, mmax
        self.width, self.heads = message_width, heads
        self.state = 2 * channels * (lmax + 1)  # [A^i, A^j]
        film = [2 * element_dim + num_rbf, *film_mlp, 2 * self.state]
        layers: list[nn.Module] = []
        for k, (a, b) in enumerate(zip(film[:-1], film[1:])):
            layers.append(nn.Linear(a, b, bias=False))
            if k < len(film) - 2:
                layers.append(nn.SiLU())
        self.film = nn.Sequential(*layers)
        nn.init.zeros_(self.film[-1].weight)
        self.embed = _Residual(self.state, bottleneck, mmax, activation, grid_points)
        msg_out = (lmax + 1) * message_width
        self.msg = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.rcv = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.msg.append(
                nn.Sequential(
                    _Residual(self.state, bottleneck, mmax, activation, grid_points),
                    O2Linear(self.state, msg_out, mmax, bias=False),
                )
            )
            self.gate.append(nn.Linear(self.state, heads * (lmax + 1), bias=False))
            self.rcv.append(
                nn.Sequential(
                    O2Linear(msg_out, bottleneck, mmax),
                    _Activation(activation, bottleneck, mmax, grid_points),
                    O2Linear(bottleneck, self.state, mmax),
                )
            )

    def init_state(
        self,
        densities: list[torch.Tensor],
        edge_index: torch.Tensor,
        frames: list[torch.Tensor],
        codes: torch.Tensor,
        rbf: torch.Tensor,
    ) -> torch.Tensor:
        """Bond states before message passing: both ends' densities in the bond frame, FiLM."""
        i, j = edge_index
        a = torch.cat(
            [
                to_edge_frame([x[i] for x in densities], frames, self.mmax),
                to_edge_frame([x[j] for x in densities], frames, self.mmax),
            ],
            dim=1,
        )
        film = self.film(torch.cat([codes[i], codes[j], rbf], dim=-1))
        scale, shift = film[:, : self.state], film[:, self.state :]
        a = a * (1.0 + scale).unsqueeze(-1)
        a = torch.cat(
            [
                a[..., : self.mmax],
                a[..., self.mmax : self.mmax + 1] + shift.unsqueeze(-1),
                a[..., self.mmax + 1 :],
            ],
            dim=-1,
        )
        return self.embed(a)

    def mp_round(
        self,
        k: int,
        a: torch.Tensor,
        edge_index: torch.Tensor,
        frames: list[torch.Tensor],
        env: torch.Tensor,
        avg_num_neighbors: float,
        num_nodes: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Round ``k``: bond -> atom -> bond.  New bond states, and the per-atom aggregate
        of the round's bond messages (global frame, one ``(N, width, 2l+1)`` block per degree)."""
        i, j = edge_index
        m = self.msg[k](a)  # (E, (L+1) W, 2M+1)
        s = torch.sigmoid(self.gate[k](a[..., self.mmax]))  # (E, H (L+1))
        s = s.reshape(-1, self.lmax + 1, self.heads, 1).expand(
            -1, -1, -1, self.width // self.heads
        )
        m = m * s.reshape(m.shape[0], m.shape[1], 1)
        blocks = from_edge_frame(m, frames, self.width)  # global, per degree
        agg = [
            scatter(b * env[:, None, None], j, dim=0, dim_size=num_nodes, reduce="sum")
            / avg_num_neighbors
            for b in blocks
        ]
        a = a + self.rcv[k](to_edge_frame([x[i] for x in agg], frames, self.mmax))  # i's bonds
        return a, agg

    def forward(
        self,
        densities: list[torch.Tensor],
        edge_index: torch.Tensor,
        frames: list[torch.Tensor],
        codes: torch.Tensor,
        rbf: torch.Tensor,
        env: torch.Tensor,
        avg_num_neighbors: float,
    ) -> torch.Tensor:
        """Bond states ``(E, state, 2M+1)`` of every directed edge ``i -> j``.

        ``densities``: per-degree atomic densities ``[(N, C, 2l+1)]`` (global
        frame); ``frames``: ``D^l`` of each edge; ``codes``: element codes
        ``(N, element_dim)``; ``rbf``, ``env``: radial basis and envelope.
        """
        a = self.init_state(densities, edge_index, frames, codes, rbf)
        n = densities[0].shape[0]
        for k in range(len(self.msg)):
            a, _ = self.mp_round(k, a, edge_index, frames, env, avg_num_neighbors, n)
        return a
