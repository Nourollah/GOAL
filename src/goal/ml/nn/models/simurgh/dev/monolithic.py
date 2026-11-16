"""SIMURGH, development form: one monolithic model file.

Registered as ``simurgh_dev`` in ``MODEL_REGISTRY`` and
``BACKBONE_REGISTRY``; configure it with ``head: null``.  It takes an
``AtomicGraph`` and returns the property dict itself, with conservative
forces ``F = -∂E/∂R``.

Energy (``core: ace``)::

    E = Σ_i E0(Z_i) + s · [Σ_i e_i + Σ_{i<j} φ_{Z_i Z_j}(r_ij)] + W_rep + E_elec + E_D3

* ``e_i``: the atom line, a body-ordered ACE core (channel-wise A-basis,
  symmetric product basis, linear update, per-layer readout);
* ``φ``: the pair line, a low-capacity 2-body potential per unordered
  element pair (zero-initialised) plus a fixed short-range wall;
* ``E_D3``: an optional closed-form D3(BJ) dispersion term (no DFT).

Every map is bias-free and every optional section is zero-initialised,
so an isolated atom gives exactly ``E0`` and switching a section on
leaves a fresh model unchanged.  The file is organised in one section
per work package; it becomes a modular GOAL model only at release.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import typing

import numpy as np
import torch
from e3nn import o3
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.models.simurgh.geometry import differentiable_edges
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

__all__ = ["SimurghDev"]


# =============================================================================
# Configuration
# =============================================================================

#: Every supported backbone key with its default.  Unknown keys raise.
#: Nested dicts are checked recursively; other values are leaves.
DEFAULTS: dict[str, typing.Any] = {
    "core": "ace",
    "cutoff": 5.0,
    "elements": None,
    "embedding_dim": 32,
    "num_channels": 128,
    "message_irreps": "128x0e + 128x1o + 128x2e",
    "lmax_sh": 3,
    "num_interactions": 2,
    "correlation": 3,
    "product_basis": "symmetric",
    "num_rbf": 8,
    "envelope_p": 5,
    "radial_mlp": [64, 64, 64],
    "pair_conditioned_radial": True,
    "element_weights": "index",
    "avg_num_neighbors": None,
    "node_norm": "none",
    "kernels": "auto",
    "dtype": "float32",
    "radial": {"low_rank": None, "per_layer": None, "env_conditioned": False},
    "a_basis": {"regularity_prior_sigma": None, "prior_weight_decay": None},
    "attention": {"mode": "gate", "pan_gate": False, "pan_gate_descriptors": "full"},
    "readout": {
        "pgs": True,
        "pgs_edge": False,
        "pgs_components": 16,
        "pgs_t_max": 3.0,
        "pgs_beta": 18.4,
        "pgs_edge_alpha": "scalar",
        "mlp_hidden": 16,
    },
    "pair_line": {
        "enabled": True,
        "mode": "phi",
        "coefficients": "table",
        "basis": "bessel",
        "num_basis": 8,
        "pair_cutoff": None,
        "r_min": 0.6,
        "repulsive_wall": "zbl",
        "curvature_penalty": 1.0e-4,
        "many_body_penalty": 0.0,
        "pair_first_epochs": 0,
        "bond_order": False,
        "bond_order_body": 3,  # 3 | 4 (four-body: planned)
        "bond_order_lmax": 4,
        "bond_order_nmax": 6,
        "bond_order_cmax": 8,
        "bond_order_q3": 36,
        "bond_order_pmax": 32,
        "bond_order_cutoff_fn": "tanh3",  # tanh3 | envelope
        "bond_order_eta_map": "power",  # power | smooth
    },
    "ece": {
        "lmax": 2,
        "mmax": 2,
        "channels": 32,
        "bottleneck": 128,
        "activation": "fourier",
        "grid_points": 16,
        "num_mp_layers": 2,
        "message_width": 64,
        "heads": 8,
        "energy_mlp": [256, 256],
        "film": {"element_dim": 4, "mlp": [64, 128]},
    },
    "electrostatics": {
        "enabled": False,
        "dipoles": False,
        "sigma": "learned",
        "charge_constraint": "uniform",
    },
    "dispersion": {
        "enabled": False,
        "backend": "torch",
        "functional": "pbe",
        "damping": "bj",
        "atm": False,
    },
    "diagnostics": {"calm": False},
    # training.aux (goal-train copies it here): training-only auxiliary objectives
    "aux": {
        "denoise": False,
        "denoise_rung": "D1",  # D1 (fixed isotropic noise); D2, D1-R: planned
        "denoise_sigma": 0.05,  # Å
        "denoise_p": 0.25,  # probability that a structure gets a corrupted copy
        "denoise_r": 0.25,  # probability that an atom of a copy is corrupted
        "force_weight": 100.0,  # λ_F of the force loss
        "denoise_scale": 1.0,  # λ_dn = scale · 0.0625 λ_F σ_F²; sweep 0.25 | 1 | 4
        "denoise_epochs": None,  # linear decay of λ_dn to 0 (goal-train: max_epochs)
        "hessian_distill": False,
        "hessian_weight": None,  # λ_KD; None = 4 λ_F (force_weight)
        "hessian_mode": "autograd",  # autograd (double backward) | finite_difference
        "hessian_teacher_val_loss": None,  # halve λ_KD once below it (not wired yet)
    },
    "atomic_energies": {"mode": "provided", "values": None},
    "scale": None,
    # training-split mean and s.d. of per-atom descriptors (goal-train sets it)
    "atom_statistics": None,
    "mfc": {
        "enabled": False,
        "objective": ["energy", "core_force"],
        "window_type": "halfspace",
        "levels": 2,
        "radii": [8.0],
        "buffer": 5.0,
        "windows_per_structure": 1,
        "every_n_steps": 1,
        "fraction": 1.0,
        "weight": 1.0,
        "weight_energy": 1.0,
        "weight_force": 1.0,
        "start_epoch": 20,
        "warmup_epochs": 10,
        "stop_grad": "parent",
        "parameters": "shared",
        "window_weighting": "uniform",
        "violation_every": 10,  # epochs between violation scores (window_weighting: violation)
        "chemical_scheme": "rotatable",  # level-1 groups of window_type: chemical
        "tolerance": "none",
        "tolerance_file": None,
        "include_charges": True,
        "diagnostic": True,
    },
}


def _plain(value: typing.Any) -> typing.Any:
    """OmegaConf containers -> plain Python (configs arrive as DictConfig)."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _merge(defaults: dict[str, typing.Any], given: typing.Mapping, path: str = "") -> dict:
    """Defaults overlaid with ``given``, recursively; unknown keys raise."""
    unknown = sorted(f"{path}{k}" for k in set(given) - set(defaults))
    if unknown:
        raise ValueError(f"unknown SIMURGH config key(s): {unknown}")
    out: dict[str, typing.Any] = {}
    for key, default in defaults.items():
        value = _plain(given[key]) if key in given else copy.deepcopy(default)
        if isinstance(default, dict) and key in given:
            if not isinstance(value, typing.Mapping):
                raise ValueError(f"SIMURGH config key {path}{key} must be a mapping")
            value = _merge(default, value, f"{path}{key}.")
        out[key] = value
    return out


def _choice(cfg: dict, key: str, allowed: tuple[str, ...], path: str = "") -> str:
    """``cfg[key]``, which must be one of ``allowed``."""
    value = cfg[key]
    if value not in allowed:
        raise ValueError(f"SIMURGH {path}{key} must be one of {allowed}, got {value!r}")
    return value


def _not_yet(flag: str) -> None:
    """Refuse an option whose work package has not landed yet."""
    raise NotImplementedError(f"SIMURGH option {flag} is not implemented yet")


# =============================================================================
# Primitives (copied into the model file during development)
# =============================================================================


def polynomial_envelope(x: torch.Tensor, p: int) -> torch.Tensor:
    """Smooth cutoff u(x): 1 at x = 0, 0 with p continuous derivatives at x >= 1."""
    u = (
        1.0
        - (p + 1) * (p + 2) / 2 * x**p
        + p * (p + 2) * x ** (p + 1)
        - p * (p + 1) / 2 * x ** (p + 2)
    )
    return torch.where(x < 1.0, u, torch.zeros_like(u))


class _BesselBasis(nn.Module):
    """sqrt(2/r_c) sin(nπ r / r_c) / r, n = 1..num, times fixed per-n factors."""

    def __init__(self, cutoff: float, num: int, factors: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.register_buffer("freq", torch.arange(1, num + 1) * math.pi / cutoff)
        self.register_buffer("factor", torch.ones(num) if factors is None else factors)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """``(E,)`` distances -> ``(E, num)`` basis values."""
        r = r.unsqueeze(-1)
        b = math.sqrt(2.0 / self.cutoff) * torch.sin(self.freq * r) / r
        return b * self.factor


class _MLP(nn.Sequential):
    """Bias-free SiLU MLP; ``zero_last`` zero-initialises the output layer."""

    def __init__(self, sizes: typing.Sequence[int], zero_last: bool = False) -> None:
        layers: list[nn.Module] = []
        for k, (a, b) in enumerate(itertools.pairwise(sizes)):
            layers.append(nn.Linear(a, b, bias=False))
            if k < len(sizes) - 2:
                layers.append(nn.SiLU())
        super().__init__(*layers)
        if zero_last:
            nn.init.zeros_(self[-1].weight)


def _irrep_blocks(irreps: o3.Irreps) -> tuple[int, list[tuple[int, int]]]:
    """``C x (l, p)`` blocks of a uniform-multiplicity irreps -> (C, [(l, p), ...])."""
    muls = {mul for mul, _ in irreps}
    if len(muls) != 1:
        raise ValueError(f"SIMURGH needs one multiplicity for every irrep, got {irreps}")
    return muls.pop(), [(ir.l, ir.p) for _, ir in irreps]


def _to_channels(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """e3nn layout ``(N, dim)`` -> channel layout ``(N, C, D)``."""
    parts = [
        x[:, sl].reshape(x.shape[0], mul, ir.dim) for (mul, ir), sl in zip(irreps, irreps.slices())
    ]
    return torch.cat(parts, dim=-1)


def _from_channels(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """Channel layout ``(N, C, D)`` -> e3nn layout ``(N, dim)``."""
    parts, start = [], 0
    for mul, ir in irreps:
        parts.append(x[:, :, start : start + ir.dim].reshape(x.shape[0], mul * ir.dim))
        start += ir.dim
    return torch.cat(parts, dim=-1)


class _ElementLinear(nn.Module):
    """Equivariant linear map with one weight matrix per element and irrep.

    Applied only between irreps present in both ``irreps_in`` and
    ``irreps_out``; loops over the elements present (few), so memory is O(N C).
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, num_elements: int) -> None:
        super().__init__()
        self.irreps_in, self.irreps_out = irreps_in, irreps_out
        self.paths: list[tuple[int, int]] = []
        weights = []
        for i, (mul_in, ir_in) in enumerate(irreps_in):
            for j, (mul_out, ir_out) in enumerate(irreps_out):
                if ir_in == ir_out:
                    self.paths.append((i, j))
                    weights.append(
                        nn.Parameter(
                            torch.randn(num_elements, mul_in, mul_out) / math.sqrt(mul_in)
                        )
                    )
        self.weights = nn.ParameterList(weights)

    def forward(self, x: torch.Tensor, elem: torch.Tensor) -> torch.Tensor:
        """Apply each atom's element-specific weights."""
        out = x.new_zeros(x.shape[0], self.irreps_out.dim)
        in_slices, out_slices = self.irreps_in.slices(), self.irreps_out.slices()
        for (i, j), w in zip(self.paths, self.weights):
            mul_in, ir = self.irreps_in[i]
            mul_out = self.irreps_out[j].mul
            xi = x[:, in_slices[i]].reshape(-1, mul_in, ir.dim)
            yj = torch.zeros(x.shape[0], mul_out, ir.dim, dtype=x.dtype, device=x.device)
            for e in torch.unique(elem).tolist():
                mask = elem == e
                yj[mask] = torch.einsum("nim,io->nom", xi[mask], w[e])
            out[:, out_slices[j]] = yj.reshape(-1, mul_out * ir.dim)
        return out


# =============================================================================
# WP2: atom line, body-ordered ACE core
# =============================================================================


def symmetric_coupling_basis(
    blocks: list[tuple[int, int]], L: int, parity: int, nu: int, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Orthonormal basis of symmetric, equivariant maps ``Sym^ν(V) -> (L, parity)``.

    ``V`` is one channel of the A-basis, the direct sum of ``blocks``
    (``(l, p)`` pairs, dimension ``D``).  Built from every sequential
    Clebsch-Gordan coupling tree, symmetrised over the ν input slots and
    reduced to an orthonormal set, so the number of basis elements is the
    exact multiplicity of ``L`` in ``Sym^ν(V)``.

    Returns ``U`` of shape ``(2L+1, D, ..., D, K)`` with ν ``D`` axes.
    """
    starts = [0]
    for ell, _ in blocks:
        starts.append(starts[-1] + 2 * ell + 1)
    dim = starts[-1]
    vectors = []
    for combo in itertools.combinations_with_replacement(range(len(blocks)), nu):
        if math.prod(blocks[b][1] for b in combo) != parity:
            continue
        ls = [blocks[b][0] for b in combo]
        trees = [[ls[0]]]
        for ell in ls[1:]:
            trees = [t + [lam] for t in trees for lam in range(abs(t[-1] - ell), t[-1] + ell + 1)]
        for tree in trees:
            if tree[-1] != L:
                continue
            t = torch.eye(2 * ls[0] + 1, dtype=dtype)
            for k in range(1, nu):
                w3j = o3.wigner_3j(tree[k - 1], ls[k], tree[k], dtype=dtype)
                t = torch.einsum("p...,pmq->q...m", t, w3j)
            full = torch.zeros((2 * L + 1,) + (dim,) * nu, dtype=dtype)
            full[(slice(None),) + tuple(slice(starts[b], starts[b + 1]) for b in combo)] = t
            sym = sum(
                full.permute(0, *[1 + i for i in perm])
                for perm in itertools.permutations(range(nu))
            )
            vectors.append(sym.reshape(-1))
    if not vectors:
        return torch.zeros((2 * L + 1,) + (dim,) * nu + (0,), dtype=dtype)
    _, s, vh = torch.linalg.svd(torch.stack(vectors), full_matrices=False)
    k = int((s > 1e-8 * s.max()).sum())
    return vh[:k].T.reshape((2 * L + 1,) + (dim,) * nu + (k,))


_LETTERS = "defghijlno"  # not b, c, k, M: used by the contraction


class _SymmetricProductBasis(nn.Module):
    """Channel-wise symmetric contraction ``B = Σ_ν W_ν[Z] · U_ν(A^{⊗ν})``.

    ``A`` is in channel layout ``(N, C, D)``.  Weights are indexed by the
    receiving element (``element_weights: index``) or produced from its
    embedding (``embedding``).  Evaluated Horner-style from the highest
    correlation down, so no ``D^ν`` tensor is ever formed per atom.
    """

    def __init__(
        self,
        irreps_a: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: int,
        element_weights: str,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.channels, blocks = _irrep_blocks(irreps_a)
        self.irreps_out = irreps_out
        self.correlation = correlation
        self.element_weights = element_weights
        self.outputs: list[tuple[int, int]] = []  # (L, parity) per output irrep
        self.num_basis: list[list[int]] = []
        weights = nn.ParameterList()
        projections = nn.ModuleList()
        for _, ir in irreps_out:
            sizes = []
            for nu in range(1, correlation + 1):
                u = symmetric_coupling_basis(blocks, ir.l, ir.p, nu)
                self.register_buffer(f"u_{ir.l}_{ir.p}_{nu}", u.to(torch.get_default_dtype()))
                k = u.shape[-1]
                sizes.append(k)
                if element_weights == "index":
                    weights.append(
                        nn.Parameter(
                            torch.randn(num_elements, k, self.channels) / math.sqrt(max(k, 1))
                        )
                    )
                else:
                    projections.append(nn.Linear(embedding_dim, k * self.channels, bias=False))
            self.outputs.append((ir.l, ir.p))
            self.num_basis.append(sizes)
        self.weights, self.projections = weights, projections

    def _weights(self, idx: int, elem: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Per-atom weights ``(N, K, C)`` of output irrep / order ``idx``."""
        if self.element_weights == "index":
            return self.weights[idx][elem]  # (N, K, C)
        w = self.projections[idx](emb[elem])
        return w.reshape(elem.shape[0], -1, self.channels)

    def forward(self, a: torch.Tensor, elem: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Channel-layout ``A`` ``(N, C, D)`` -> ``B`` in channel layout over ``irreps_out``."""
        outs, idx = [], 0
        for (l, p), sizes in zip(self.outputs, self.num_basis):
            x: torch.Tensor | None = None
            for nu in range(self.correlation, 0, -1):
                u = getattr(self, f"u_{l}_{p}_{nu}")
                w = self._weights(idx + nu - 1, elem, emb)
                axes = _LETTERS[:nu]
                if sizes[nu - 1] > 0:
                    term = torch.einsum(f"M{axes}k,bkc->bcM{axes}", u, w)
                    x = term if x is None else x + term
                if x is not None:
                    x = torch.einsum(f"bcM{axes},bc{axes[-1]}->bcM{axes[:-1]}", x, a)
            idx += self.correlation
            if x is None:
                x = a.new_zeros(a.shape[0], self.channels, 2 * l + 1)
            outs.append(x)
        return torch.cat(outs, dim=-1)  # channel layout over irreps_out

    def scalar_basis(self, a: torch.Tensor) -> torch.Tensor:
        """Unweighted 0e product basis ``(N, C · Σ_ν K_ν)``: ``U_ν(A^{⊗ν})`` per channel.

        The same contraction as :meth:`forward` before the element weights,
        for the scalar output only (the extrapolation grade's feature).
        """
        parts = []
        for nu in range(1, self.correlation + 1):
            if not hasattr(self, f"u_0_1_{nu}"):
                continue
            u = getattr(self, f"u_0_1_{nu}")
            if u.shape[-1] == 0:
                continue
            axes = _LETTERS[:nu]
            x = torch.einsum(f"M{axes}k,bc{axes[-1]}->bcM{axes[:-1]}k", u, a)
            for k in range(nu - 1, 0, -1):
                x = torch.einsum(f"bcM{axes[:k]}K,bc{axes[k - 1]}->bcM{axes[: k - 1]}K", x, a)
            parts.append(x.reshape(a.shape[0], -1))
        return torch.cat(parts, dim=-1)


class _AttentionGate(nn.Module):
    """Smooth local attention on the messages of one layer (WP8 A).

    ``gate``: ``A_i = (1/⟨N⟩) Σ_j 2σ(s_ij) m_ij`` with
    ``s_ij = b(r_ij) + τ/√D ⟨q_i, m_ij⟩``; ``b`` (enveloped) and ``q`` are
    zero-initialised, so a fresh layer is exactly the uniform layer.
    ``softmax_null``: ``w_ij = u² e^{s_ij} / (softplus(ζ) + Σ_k u_k² e^{s_ik})``,
    a weighted average with the envelope inside the normalisation.
    """

    def __init__(
        self, mode: str, irreps_h: o3.Irreps, irreps_msg: o3.Irreps, num_rbf: int
    ) -> None:
        super().__init__()
        self.mode = mode
        self.query = None  # radial: scores from the distance alone (removal control)
        if mode != "radial":
            self.query = o3.Linear(irreps_h, irreps_msg)
            with torch.no_grad():
                self.query.weight.zero_()
        self.bias = _MLP([num_rbf, 16, 1], zero_last=True)
        self.temperature = nn.Parameter(torch.zeros(())) if self.query is not None else None
        # only the softmax arm reads it; a parameter outside the loss would break DDP
        self.null = nn.Parameter(torch.zeros(())) if mode == "softmax_null" else None
        self.dim = irreps_msg.dim

    def forward(
        self,
        messages: torch.Tensor,
        h: torch.Tensor,
        receiver: torch.Tensor,
        rbf: torch.Tensor,
        env: torch.Tensor,
        num_nodes: int,
        avg_num_neighbors: float,
    ) -> torch.Tensor:
        """Aggregate ``messages`` at their receivers with the attention weights."""
        tau = 0.25 + 3.75 * torch.sigmoid(self.temperature) if self.query is not None else 0.0
        score = env * self.bias(rbf).squeeze(-1)
        if self.query is not None:
            score = score + tau / math.sqrt(self.dim) * (self.query(h)[receiver] * messages).sum(
                -1
            )
        if self.mode in ("gate", "radial"):
            weighted = 2.0 * torch.sigmoid(score).unsqueeze(-1) * messages
            return (
                scatter(weighted, receiver, dim=0, dim_size=num_nodes, reduce="sum")
                / avg_num_neighbors
            )
        shift = scatter(score.detach(), receiver, dim=0, dim_size=num_nodes, reduce="max")[
            receiver
        ]
        num = env.square() * torch.exp(score - shift)
        den = scatter(num, receiver, dim=0, dim_size=num_nodes, reduce="sum")[receiver]
        den = den + nn.functional.softplus(self.null) * torch.exp(-shift)
        weighted = (num / den).unsqueeze(-1) * messages
        return scatter(weighted, receiver, dim=0, dim_size=num_nodes, reduce="sum")


class _LadderProductBasis(nn.Module):
    """Fixed-width correlation ladder (``product_basis: ladder | ladder_reduced``).

    ``φ1 = W1 A``; ``φq = Wq (φ_{q-1} ⊗ A)`` for ``q = 2..ν``, with ``⊗`` the
    channel-wise Clebsch-Gordan product over every admissible path,
    pseudo-tensor intermediates included (all parities up to the largest
    degree of ``A``); ``B = P Σ_q φq``.  ``ladder``: ``Wq`` mixes channels and
    paths per output degree (``element_indexed``: one map per element, the
    coupled-product ablation).  ``ladder_reduced``: one weight per path and
    channel, ``a = 1 + δ`` with ``δ = 0`` at start, and the channel mixing only
    in ``P``.  The cost grows linearly with ν.  ``A`` and ``B`` are in e3nn
    layout; ``A`` stays linear in the neighbours (no norm, no gate).
    """

    def __init__(
        self,
        irreps_a: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        reduced: bool,
        num_elements: int = 0,
    ) -> None:
        super().__init__()
        channels, blocks = _irrep_blocks(irreps_a)
        lmax = max(ell for ell, _ in blocks)
        self.irreps_phi = o3.Irreps(
            [(channels, (ell, p)) for ell in range(lmax + 1) for p in (1, -1)]
        )
        self.reduced = reduced
        self.first = None if reduced else o3.Linear(irreps_a, self.irreps_phi)
        tps, mixes = nn.ModuleList(), nn.ModuleList()
        irreps_prev = irreps_a if reduced else self.irreps_phi
        for _ in range(2, correlation + 1):
            if reduced:
                instructions = [
                    (i, j, self._slot(ir_out), "uuu", True)
                    for i, (_, ir1) in enumerate(irreps_prev)
                    for j, (_, ir2) in enumerate(irreps_a)
                    for ir_out in ir1 * ir2
                    if ir_out.l <= lmax
                ]
                tp = o3.TensorProduct(irreps_prev, irreps_a, self.irreps_phi, instructions)
                with torch.no_grad():
                    tp.weight.fill_(1.0)
                tps.append(tp)
                mixes.append(nn.Identity())
            else:
                mid, instructions = [], []
                for i, (_, ir1) in enumerate(irreps_prev):
                    for j, (_, ir2) in enumerate(irreps_a):
                        for ir_out in ir1 * ir2:
                            if ir_out.l <= lmax:
                                instructions.append((i, j, len(mid), "uuu", False))
                                mid.append((channels, ir_out))
                tp = o3.TensorProduct(irreps_prev, irreps_a, o3.Irreps(mid), instructions)
                tps.append(tp)
                if num_elements:
                    mixes.append(_ElementLinear(o3.Irreps(mid), self.irreps_phi, num_elements))
                else:
                    mixes.append(o3.Linear(o3.Irreps(mid), self.irreps_phi))
            irreps_prev = self.irreps_phi
        self.tps, self.mixes = tps, mixes
        self.out = o3.Linear(self.irreps_phi, irreps_out)
        self.lift = o3.Linear(irreps_a, self.irreps_phi) if reduced else None
        self.element_indexed = bool(num_elements)

    def _slot(self, ir: o3.Irrep) -> int:
        """Index of ``ir`` in ``irreps_phi``."""
        return [irp for _, irp in self.irreps_phi].index(ir)

    def forward(self, a: torch.Tensor, elem: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """``A`` (e3nn layout) -> ``B`` over ``irreps_out`` (e3nn layout)."""
        phi = a if self.reduced else self.first(a)
        total = self.lift(a) if self.reduced else phi
        for tp, mix in zip(self.tps, self.mixes):
            x = tp(phi, a)
            phi = mix(x, elem) if self.element_indexed else mix(x)
            total = total + phi
        return self.out(total)


class _RMSNorm(nn.Module):
    """Smooth equivariant RMS norm per irrep block with a per-channel scale (Muon arm only).

    ``x / sqrt(mean(x²) + 1)``: an exact RMS norm sends the features of an
    atom whose last neighbour leaves the cutoff from ~0 to unit size and
    back to 0, a jump in the energy (0.029 eV measured on the cutoff-crossing
    test); the ``+ 1`` keeps small features small and the energy smooth.
    """

    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = irreps
        self.scales = nn.ParameterList(nn.Parameter(torch.ones(mul)) for mul, _ in irreps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise every irrep block of ``x`` (e3nn layout)."""
        parts = []
        for (mul, ir), sl, scale in zip(self.irreps, self.irreps.slices(), self.scales):
            block = x[:, sl].reshape(-1, mul, ir.dim)
            rms = block.square().mean(dim=(1, 2), keepdim=True).add(1.0).sqrt()
            parts.append((block / rms * scale[:, None]).reshape(-1, mul * ir.dim))
        return torch.cat(parts, dim=-1)


def pan_descriptors(
    vectors: torch.Tensor,
    lengths: torch.Tensor,
    receiver: torch.Tensor,
    num_atoms: int,
    cutoff: float,
    p: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Coordination ``ρ`` and angular variance ``η`` of every atom, ``(N, 2)``.

    arXiv:2606.15892, Eqs. S4-S7: ``ρ_i = Σ_j u(r_ij)``; ``η_i`` is the
    ``u``-weighted variance of ``c_jk = r̂_ij · r̂_ik`` over ordered pairs
    ``j ≠ k``, evaluated in O(N_nb) from the moments ``m1 = Σ_j u r̂`` and
    ``M2 = Σ_j u r̂ r̂ᵀ``: with ``W2 = ρ² - Σu²``, ``S1 = |m1|² - Σu²``,
    ``S2 = ‖M2‖² - Σu²`` and ``c̄ = S1 / (ε + W2)``,
    ``η = (S2 - 2 c̄ S1 + c̄² W2) / (ε + W2)``.  ``η = 0`` with fewer than
    three neighbours; both are C2 as a neighbour crosses ``r_c``.
    """
    w = polynomial_envelope(lengths / cutoff, p)
    unit = vectors / lengths.unsqueeze(-1)
    outer = (unit.unsqueeze(-1) * unit.unsqueeze(-2)).reshape(-1, 9)
    sums = scatter(
        torch.cat(
            [
                w.unsqueeze(-1),
                (w * w).unsqueeze(-1),
                w.unsqueeze(-1) * unit,
                w.unsqueeze(-1) * outer,
            ],
            dim=-1,
        ),
        receiver,
        dim=0,
        dim_size=num_atoms,
        reduce="sum",
    )
    rho, w2, m1, m2 = sums[:, 0], sums[:, 1], sums[:, 2:5], sums[:, 5:]
    pairs = rho * rho - w2
    s1 = m1.square().sum(-1) - w2
    s2 = m2.square().sum(-1) - w2
    mean = s1 / (eps + pairs)
    eta = (s2 - 2.0 * mean * s1 + mean * mean * pairs) / (eps + pairs)
    return torch.stack([rho, eta], dim=-1)


#: Per-atom descriptors the PAN gate standardises with training-split statistics.
PAN_STATISTICS: tuple[str, ...] = ("pan_rho", "pan_eta")


class _PANGate(nn.Module):
    """PAN gate on the scalar messages (arXiv:2606.15892, Eqs. S14-S15, Algorithm S1).

    ``a_ij = 2σ(f([e_ij, onehot(Z_i), onehot(Z_j), ρ̃_i, η̃_i]))`` with ``i`` the
    receiving atom, ``e_ij`` the radial network's input and ``ρ̃``, ``η̃``
    standardised with training-split statistics.  ``a_ij`` multiplies the 0e
    slots of each per-edge message before the neighbour sum; the l > 0 slots
    are untouched.  The last layer of ``f`` starts at zero, so ``a = 1``.
    ``descriptors=False`` is the paper's generic-gate control (no ``ρ, η``).
    """

    def __init__(
        self, irreps_mid: o3.Irreps, in_dim: int, num_elements: int, descriptors: bool
    ) -> None:
        super().__init__()
        self.num_elements = num_elements
        width = in_dim + 2 * num_elements + (2 if descriptors else 0)
        self.f = _MLP([width, 64, 1], zero_last=True)
        scalar = torch.cat(
            [
                torch.full((mul * ir.dim,), ir == o3.Irrep("0e"), dtype=torch.bool)
                for mul, ir in irreps_mid
            ]
        )
        self.register_buffer("scalar", scalar, persistent=False)

    def forward(
        self,
        messages: torch.Tensor,
        e: torch.Tensor,
        elem: torch.Tensor,
        edge_index: torch.Tensor,
        descriptors: torch.Tensor | None,
    ) -> torch.Tensor:
        """Messages ``(E, dim)`` with their 0e slots scaled by ``a_ij``."""
        sender, receiver = edge_index
        onehot = nn.functional.one_hot(elem, self.num_elements).to(e.dtype)
        psi = [e, onehot[receiver], onehot[sender]]
        if descriptors is not None:
            psi.append(descriptors[receiver])
        a = 2.0 * torch.sigmoid(self.f(torch.cat(psi, dim=-1)))
        return torch.where(self.scalar, messages * a, messages)


class _PGSEdge(nn.Module):
    """PGS edge correction (arXiv:2606.15892, Eq. S9, App. F): ``w' = w + g(e) ⊙ s(r)``.

    ``s(r) = Σ_m α_m B_m(r)`` with ``B_m = √(2/r_c) sin(mπr/r_c)/r · u(r)``, the
    model's own envelope ``u``; ``g = 2σ(MLP(e))`` reads the radial network's
    input ``e``.  ``α`` and the gate's last layer start at zero (``w' = w``,
    ``g = 1``).  ``s`` carries ``u``, so the correction and its first two
    derivatives vanish at ``r_c``.
    """

    def __init__(
        self, in_dim: int, out_dim: int, cutoff: float, p: int, per_channel: bool
    ) -> None:
        super().__init__()
        self.cutoff, self.p = cutoff, p
        self.basis = _BesselBasis(cutoff, 16)
        self.alpha = nn.Parameter(torch.zeros(16, out_dim) if per_channel else torch.zeros(16, 1))
        self.gate = _MLP([in_dim, 64, out_dim], zero_last=True)

    def forward(self, e: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Additive correction ``(E, P)`` to the tensor-product weights."""
        env = polynomial_envelope(lengths / self.cutoff, self.p).unsqueeze(-1)
        s = (self.basis(lengths) * env) @ self.alpha
        return 2.0 * torch.sigmoid(self.gate(e)) * s


def _triplets(center: torch.Tensor, num_atoms: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairs of distinct directed edges ``(e1, e2)`` that share their centre atom."""
    degree = torch.bincount(center, minlength=num_atoms)
    order = torch.argsort(center, stable=True)
    start = torch.cumsum(degree, 0) - degree
    counts = degree[center]
    e1 = torch.repeat_interleave(torch.arange(center.numel(), device=center.device), counts)
    offset = torch.arange(e1.numel(), device=center.device) - torch.repeat_interleave(
        torch.cumsum(counts, 0) - counts, counts
    )
    e2 = order[start[center[e1]] + offset]
    keep = e1 != e2
    return e1[keep], e2[keep]


class _BondOrder(nn.Module):
    """Saturating bond-order pair term (arXiv:2608.22933, Eqs. 11-35), three-body features.

    ``E_bo = α_bo Σ_{i<j} f(r) [V_R(r) - ½(b_ij + b_ji) V_A(r)]`` with
    ``V_R = (1 + Q/r) D_R e^(-a r)`` and ``V_A = D_A e^(-b r)`` per unordered
    element pair; only the attraction is scaled by ``b``, so compressed
    configurations stay repulsive.  ``b_ij`` saturates the bond's three-body
    descriptor ``ζ_ij`` (Eqs. 33-35): ``b = (1+n)^(-σ) + Σ_p v_p ẑ_p (1+n)^(-σ-1)``,
    ``n`` the smoothed norm of ``ẑ``, so ``b = 1`` when ``i`` has no other
    neighbour.  ``Bt3[ij,n,l] = Σ_k χ[n,Z_i,Z_k] f(r_ik) e^(λ_n (r_ij - r_ik)) S_l
    P_l(cos θ_ijk)`` (Legendre form of Eq. 23, ``S_l = 2l + 1``) uses its
    own exponential basis.  ``α_bo`` starts at zero.
    """

    def __init__(self, cfg: dict, num_elements: int, cutoff: float, p: int) -> None:
        super().__init__()
        pl = cfg["pair_line"]
        self.cutoff, self.p = cutoff, p
        self.lmax, self.nmax = pl["bond_order_lmax"], pl["bond_order_nmax"]
        self.cutoff_fn, self.eta_map = pl["bond_order_cutoff_fn"], pl["bond_order_eta_map"]
        self.num_elements = num_elements
        a, b = torch.triu_indices(num_elements, num_elements)
        pair_id = torch.zeros(num_elements, num_elements, dtype=torch.long)
        pair_id[a, b] = pair_id[b, a] = torch.arange(a.numel())
        self.register_buffer("pair_id", pair_id, persistent=False)
        self.pair_raw = nn.Parameter(torch.rand(a.numel(), 5))  # D_R, D_A, a, b, Q
        lam0 = torch.linspace(10.0 / cutoff / self.nmax, 10.0 / cutoff, self.nmax)
        self.lambda_raw = nn.Parameter(torch.log(torch.expm1(lam0)))
        self.chi = nn.Parameter(torch.rand(num_elements, num_elements, self.nmax) - 0.5)
        cmax, q3, pmax = pl["bond_order_cmax"], pl["bond_order_q3"], pl["bond_order_pmax"]
        self.pair_code = nn.Embedding(num_elements * num_elements, cmax, max_norm=1.0)
        self.w = nn.Parameter(torch.rand(cmax, self.nmax, q3) - 0.5)
        self.u = nn.Parameter(torch.rand(q3, self.lmax + 1, pmax) - 0.5)
        self.eta_raw = nn.Parameter(torch.randn(pmax))
        self.sigma_raw = nn.Parameter(torch.zeros(()))
        self.v = nn.Linear(pmax, 1, bias=False)
        nn.init.xavier_uniform_(self.v.weight, gain=0.01)
        self.amplitude = nn.Parameter(torch.zeros(()))  # α_bo

    def envelope(self, r: torch.Tensor) -> torch.Tensor:
        """``f(r)``: ``[tanh(1 - r/R)/tanh(1)]³`` (B_cut = 1, frozen) or the model's ``u``."""
        if self.cutoff_fn == "envelope":
            return polynomial_envelope(r / self.cutoff, self.p)
        x = (1.0 - r / self.cutoff).clamp_min(0.0)
        return (torch.tanh(x) / math.tanh(1.0)) ** 3

    def bond_orders(self, vectors, lengths, edge_index, elem) -> torch.Tensor:
        """``b`` of every directed edge ``i -> j`` (``i = edge_index[0]``), ``(E,)``."""
        center, other = edge_index
        f = self.envelope(lengths)
        lam = nn.functional.softplus(self.lambda_raw).clamp_max(20.0 / self.cutoff)
        e1, e2 = _triplets(center, elem.shape[0])
        unit = vectors / lengths.unsqueeze(-1)
        cos = (unit[e1] * unit[e2]).sum(-1)
        legendre = [torch.ones_like(cos), cos]
        for ell in range(1, self.lmax):
            legendre.append(
                ((2 * ell + 1) * cos * legendre[ell] - ell * legendre[ell - 1]) / (ell + 1)
            )
        scale = torch.arange(self.lmax + 1, dtype=cos.dtype, device=cos.device) * 2 + 1
        angular = torch.stack(legendre[: self.lmax + 1], -1) * scale  # (T, L+1)
        radial = (
            self.chi[elem[center[e1]], elem[other[e2]]]
            * f[e2].unsqueeze(-1)
            * torch.exp(lam * (lengths[e1] - lengths[e2]).unsqueeze(-1))
        )  # (T, n)
        bt3 = scatter(
            radial.unsqueeze(-1) * angular.unsqueeze(1),
            e1,
            dim=0,
            dim_size=lengths.shape[0],
            reduce="sum",
        )  # (E, n, L+1)
        code = self.pair_code(elem[center] * self.num_elements + elem[other])  # (E, c)
        zeta = torch.einsum(
            "eql,qlp->ep", torch.einsum("enl,cnq,ec->eql", bt3, self.w, code), self.u
        )
        eta = 1.0 + 4.0 * torch.sigmoid(self.eta_raw)  # η_max = 5
        eps = 1e-3
        if self.eta_map == "smooth":
            zh = zeta * (zeta * zeta + eps * eps) ** ((eta - 1.0) / 2.0)
        else:
            zh = torch.sign(zeta) * ((zeta.abs() + eps) ** eta - eps**eta)
        n = torch.sqrt(zh.square().sum(-1) + 1e-4) - 1e-2
        sigma = 1.0 + nn.functional.softplus(self.sigma_raw)
        return (1.0 + n) ** (-sigma) + self.v(zh).squeeze(-1) * (1.0 + n) ** (-sigma - 1.0)

    def forward(self, vectors, lengths, edge_index, elem) -> torch.Tensor:
        """Per-atom energy ``(N,)`` (eV): half of each pair term to each atom."""
        a, b = elem[edge_index[0]], elem[edge_index[1]]
        d_r, d_a, k_r, k_a, q = nn.functional.softplus(
            self.pair_raw[self.pair_id[a, b]], beta=math.log(2.0)
        ).unbind(-1)
        f = self.envelope(lengths)
        repulsion = (1.0 + q / lengths) * d_r * torch.exp(-k_r * lengths)
        attraction = d_a * torch.exp(-k_a * lengths)
        b_ij = self.bond_orders(vectors, lengths, edge_index, elem)
        directed = 0.5 * self.amplitude * f * (repulsion - b_ij * attraction)  # Σ = E_bo
        n = elem.shape[0]
        return 0.5 * (
            scatter(directed, edge_index[0], dim=0, dim_size=n, reduce="sum")
            + scatter(directed, edge_index[1], dim=0, dim_size=n, reduce="sum")
        )


class _Interaction(nn.Module):
    """One ACE layer: channel-wise A-basis, product basis and linear update."""

    def __init__(
        self,
        layer: int,
        irreps_h: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_a: o3.Irreps,
        irreps_out: o3.Irreps,
        cfg: dict,
        num_elements: int,
        pair_dim: int,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.irreps_h, self.irreps_a, self.irreps_out = irreps_h, irreps_a, irreps_out
        self.linear_up = o3.Linear(irreps_h, irreps_h)
        # uvu paths from (h ⊗ Y) into every irrep of A, one copy per path
        mid, instructions = [], []
        for i, (mul, ir_in) in enumerate(irreps_h):
            for j, (_, ir_sh) in enumerate(irreps_sh):
                for ir_out in ir_in * ir_sh:
                    if ir_out in irreps_a:
                        instructions.append((i, j, len(mid), "uvu", True))
                        mid.append((mul, ir_out))
        irreps_mid = o3.Irreps(mid)
        irreps_mid_sorted, perm, _ = irreps_mid.sort()
        instructions = [(i, j, perm[k], mode, w) for i, j, k, mode, w in instructions]
        self.irreps_mid = irreps_mid_sorted
        self.tp = o3.TensorProduct(
            irreps_h,
            irreps_sh,
            irreps_mid_sorted,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        per_layer = cfg["radial"]["per_layer"]
        self.linear_radial = (
            bool(per_layer) and per_layer[min(layer, len(per_layer) - 1)] == "linear"
        )
        in_dim = cfg["num_rbf"] + (0 if self.linear_radial else pair_dim)
        if cfg["radial"]["env_conditioned"] and layer > 0:
            self.env_proj = nn.Linear(irreps_h.count("0e"), 8, bias=False)
            nn.init.zeros_(self.env_proj.weight)
            in_dim += 16
        else:
            self.env_proj = None
        rank = cfg["radial"]["low_rank"]
        self.num_paths = len(instructions)
        radial_out = self.tp.weight_numel
        self.low_rank = None
        if rank:
            if not 1 <= int(rank) <= 4:
                raise ValueError("radial.low_rank must be between 1 and 4")
            radial_out = self.num_paths * int(rank)
            mul = self.tp.weight_numel // self.num_paths
            self.low_rank = nn.Parameter(
                torch.randn(self.num_paths, int(rank), mul) / math.sqrt(rank)
            )
        if self.linear_radial:
            self.radial = nn.Linear(in_dim, radial_out, bias=False)
        else:
            self.radial = _MLP([in_dim, *cfg["radial_mlp"], radial_out])
        rd = cfg["readout"]
        self.pgs_edge = (
            _PGSEdge(
                in_dim,
                self.tp.weight_numel,
                cfg["cutoff"],
                cfg["envelope_p"],
                rd["pgs_edge_alpha"] == "per_channel",
            )
            if rd["pgs_edge"]
            else None
        )
        mode = cfg["attention"]["mode"]
        self.attention = (
            _AttentionGate(mode, irreps_h, irreps_mid_sorted, cfg["num_rbf"])
            if mode != "none"
            else None
        )
        self.pan_gate = (
            _PANGate(
                irreps_mid_sorted,
                in_dim,
                num_elements,
                cfg["attention"]["pan_gate_descriptors"] == "full",
            )
            if cfg["attention"]["pan_gate"]
            else None
        )
        self.linear_a = o3.Linear(irreps_mid_sorted, irreps_a)
        self.symmetric = cfg["product_basis"] == "symmetric"
        if self.symmetric:
            self.product = _SymmetricProductBasis(
                irreps_a,
                irreps_out,
                cfg["correlation"],
                num_elements,
                cfg["element_weights"],
                cfg["embedding_dim"],
            )
        else:
            self.product = _LadderProductBasis(
                irreps_a, irreps_out, cfg["correlation"], cfg["product_basis"] == "ladder_reduced"
            )
        self.norm = _RMSNorm(irreps_out) if cfg["node_norm"] == "rms" else None
        #: ``diagnostics.calm``: keep the unweighted 0e product basis of the last pass.
        self.record_basis = bool(cfg["diagnostics"]["calm"])
        self.last_basis: torch.Tensor | None = None
        #: Optional per-edge factor on the messages (literature ablations set it).
        self.message_gate: nn.Module | None = None
        self.linear_b = o3.Linear(irreps_out, irreps_out)
        self.self_connection = (
            _ElementLinear(irreps_h, irreps_out, num_elements) if layer > 0 else None
        )

    def forward(
        self,
        h: torch.Tensor,
        elem: torch.Tensor,
        emb: torch.Tensor,
        edge_index: torch.Tensor,
        sh: torch.Tensor,
        rbf: torch.Tensor,
        pair_code: torch.Tensor | None,
        env: torch.Tensor,
        avg_num_neighbors: float,
        lengths: torch.Tensor | None = None,
        pan: torch.Tensor | None = None,
        a_shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One layer: messages, A-basis, product basis and update."""
        sender, receiver = edge_index
        radial_in = [rbf]
        if not self.linear_radial and pair_code is not None:
            radial_in.append(pair_code)
        if self.env_proj is not None:
            scalars = h[:, : self.irreps_h.count("0e")]
            radial_in += [self.env_proj(scalars)[sender], self.env_proj(scalars)[receiver]]
        radial_in = torch.cat(radial_in, dim=-1)
        weights = self.radial(radial_in)
        if self.low_rank is not None:
            factors = weights.reshape(weights.shape[0], self.num_paths, self.low_rank.shape[1])
            weights = torch.einsum("epr,prc->epc", factors, self.low_rank).reshape(
                weights.shape[0], self.tp.weight_numel
            )
        weights = weights * env.unsqueeze(-1)
        if self.pgs_edge is not None:
            weights = weights + self.pgs_edge(radial_in, lengths)
        messages = self.tp(self.linear_up(h)[sender], sh, weights)
        if self.message_gate is not None:
            messages = messages * self.message_gate(h, edge_index, sh, env).unsqueeze(-1)
        if self.pan_gate is not None:
            messages = self.pan_gate(messages, radial_in, elem, edge_index, pan)
        n = h.shape[0]
        if self.attention is None:
            a = scatter(messages, receiver, dim=0, dim_size=n, reduce="sum") / avg_num_neighbors
        else:
            a = self.attention(messages, h, receiver, rbf, env, n, avg_num_neighbors)
        a = self.linear_a(a)
        if a_shift is not None:
            a = a + a_shift
        if self.symmetric:
            channels = _to_channels(a, self.irreps_a)
            if self.record_basis:
                self.last_basis = self.product.scalar_basis(channels.detach())
            b = _from_channels(self.product(channels, elem, emb), self.irreps_out)
        else:
            b = self.product(a, elem, emb)
        out = self.linear_b(b)
        if self.self_connection is not None:
            out = out + self.self_connection(h, elem)
        return out if self.norm is None else self.norm(out)


def _es_kernel(z: torch.Tensor, beta: float) -> torch.Tensor:
    """Compact exponential-of-semicircle kernel, C2 on the real line.

    ``τ(|z|) [exp(β(√(1-z²) - 1)) - e^{-β}]`` for ``|z| < 1``, else 0, with the
    quintic taper ``τ(z) = 1 - 10z³ + 15z⁴ - 6z⁵`` (arXiv:2606.15892, App. F;
    the exponential-of-semicircle form of arXiv:1808.06736, Eq. 1.8).  The
    double ``where`` keeps autograd finite at and beyond ``|z| = 1``.
    """
    inside = (1.0 - z * z) > 0
    zs = torch.where(inside, z, torch.zeros_like(z))
    a = zs.abs()
    taper = 1.0 - a**3 * (10.0 - a * (15.0 - 6.0 * a))
    root = torch.sqrt((1.0 - zs * zs).clamp_min(torch.finfo(z.dtype).tiny))
    value = taper * (torch.exp(beta * (root - 1.0)) - math.exp(-beta))
    return torch.where(inside, value, torch.zeros_like(value))


class _PGSReadout(nn.Module):
    """PGS spectral readout (arXiv:2606.15892, Eq. S10): lifts the final invariants.

    ``y = x + D(x) + Σ_m α_m [K((t_m(x) - c_m)/w) - K(-c_m/w)]`` with
    ``t_m = u_mᵀ x̂``, ``x̂ = x / sqrt(mean(x²) + 1e-6)``, ``u_m = v_m/‖v_m‖``,
    ``M`` centres ``c_m`` uniform on ``[-t_max, t_max]`` and ``w`` four
    spacings.  ``α ∈ R^(M×C)`` and the last layer of ``D`` start at zero, so
    ``y = x``.  Subtracting the ``x = 0`` value keeps ``y(0) = 0`` (the
    energy MLP is bias-free, so an isolated atom still gives exactly E0).
    """

    def __init__(self, channels: int, components: int, t_max: float, beta: float) -> None:
        super().__init__()
        self.beta = beta
        v = torch.empty(components, channels)
        nn.init.orthogonal_(v)  # QR of a Gaussian draw from the run's seed
        self.directions = nn.Parameter(v)
        self.alpha = nn.Parameter(torch.zeros(components, channels))
        self.low = _MLP([channels, channels, channels], zero_last=True)
        centres = torch.linspace(-t_max, t_max, components, dtype=torch.float64)
        self.register_buffer("centres", centres)
        self.width = 4.0 * 2.0 * t_max / (components - 1)
        self.reach = t_max + self.width  # beyond it every kernel is zero
        self.last_in_support: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lifted invariants ``y`` ``(N, C)`` from the final invariants ``x``."""
        unit = self.directions / self.directions.norm(dim=-1, keepdim=True)
        xhat = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
        t = xhat @ unit.T  # (N, M)
        centres = self.centres.to(x.dtype)
        k = _es_kernel((t - centres) / self.width, self.beta)
        k0 = _es_kernel(-centres / self.width, self.beta)
        self.last_in_support = (t.detach().abs() < self.reach).any(-1).to(x.dtype).mean()
        return x + self.low(x) + (k - k0) @ self.alpha


def _pgs(readout: dict, channels: int) -> _PGSReadout | None:
    """The PGS readout configured from the ``readout:`` section (``None`` when off)."""
    if not readout["pgs"]:
        return None
    return _PGSReadout(
        channels, readout["pgs_components"], readout["pgs_t_max"], readout["pgs_beta"]
    )


class _AtomLine(nn.Module):
    """Embedding, radial and angular bases, ACE layers and per-layer readouts."""

    def __init__(self, cfg: dict, num_elements: int) -> None:
        super().__init__()
        c = cfg["num_channels"]
        self.cutoff, self.p = cfg["cutoff"], cfg["envelope_p"]
        self.irreps_a = o3.Irreps(cfg["message_irreps"])
        if _irrep_blocks(self.irreps_a)[0] != c:
            raise ValueError("message_irreps multiplicity must equal num_channels")
        self.irreps_sh = o3.Irreps.spherical_harmonics(cfg["lmax_sh"])
        sigma = cfg["a_basis"]["regularity_prior_sigma"]
        rc2 = cfg["cutoff"] ** 2
        n = torch.arange(1, cfg["num_rbf"] + 1, dtype=torch.float64)
        rbf_factor = torch.exp(-5 * sigma**2 * n**2 / (2 * rc2)) if sigma else None
        self.rbf = _BesselBasis(
            cfg["cutoff"], cfg["num_rbf"], None if rbf_factor is None else rbf_factor.float()
        )
        sh_factor = torch.cat(
            [
                torch.full((2 * ell + 1,), math.exp(-((sigma or 0.0) ** 2) * ell**2 / (2 * rc2)))
                for ell in range(cfg["lmax_sh"] + 1)
            ]
        )
        self.register_buffer("sh_factor", sh_factor)
        self.embedding = nn.Embedding(num_elements, cfg["embedding_dim"])
        self.embed_to_h = nn.Linear(cfg["embedding_dim"], c, bias=False)
        self.pair_code = (
            nn.Linear(2 * cfg["embedding_dim"], cfg["embedding_dim"], bias=False)
            if cfg["pair_conditioned_radial"]
            else None
        )
        hidden = self.irreps_a
        scalars = o3.Irreps(f"{c}x0e")
        layers, readouts = [], []
        irreps_h = scalars
        num_layers = cfg["num_interactions"]
        for t in range(num_layers):
            last = t == num_layers - 1
            irreps_out = hidden
            if last:
                irreps_out = scalars
                if "ece" in cfg["pair_line"]["mode"]:
                    irreps_out = hidden  # the ECE bond line reads equivariant features
                elif cfg["electrostatics"]["enabled"] and cfg["electrostatics"]["dipoles"]:
                    irreps_out = o3.Irreps(f"{c}x0e + {c}x1o")  # vectors for the dipole head
            layers.append(
                _Interaction(
                    t,
                    irreps_h,
                    self.irreps_sh,
                    self.irreps_a,
                    irreps_out,
                    cfg,
                    num_elements,
                    cfg["embedding_dim"] if self.pair_code is not None else 0,
                )
            )
            hidden_width = cfg["readout"]["mlp_hidden"]
            readouts.append(_MLP([c, hidden_width, 1]) if last else nn.Linear(c, 1, bias=False))
            irreps_h = irreps_out
        self.layers, self.readouts = nn.ModuleList(layers), nn.ModuleList(readouts)
        self.pgs = _pgs(cfg["readout"], c)
        self.channels = c
        att = cfg["attention"]
        stats = None
        if att["pan_gate"] and att["pan_gate_descriptors"] == "full":
            given = cfg["atom_statistics"] or {}
            missing = [k for k in PAN_STATISTICS if k not in given]
            if missing:
                raise ValueError(
                    f"attention.pan_gate needs atom_statistics {missing}: the training-split "
                    "mean and s.d. of the PAN descriptors (goal-train sets them)"
                )
            stats = torch.tensor(
                [[given[k]["mean"], max(given[k]["std"], 1e-6)] for k in PAN_STATISTICS],
                dtype=torch.float64,
            )
        self.register_buffer("pan_stats", stats)  # (2, [mean, s.d.]), frozen

    def prepare(self, elem, edge_index, vectors, lengths) -> dict[str, typing.Any]:
        """What every layer shares: ``h`` (layer input), bases, envelope, pair code, PAN."""
        emb = self.embedding.weight
        h = self.embed_to_h(emb[elem])
        sh = o3.spherical_harmonics(
            self.irreps_sh, vectors, normalize=True, normalization="component"
        )
        sh = sh * self.sh_factor
        rbf = self.rbf(lengths)
        env = polynomial_envelope(lengths / self.cutoff, self.p)
        pair_code = None
        if self.pair_code is not None:
            ei, ej = emb[elem[edge_index[0]]], emb[elem[edge_index[1]]]
            pair_code = self.pair_code(torch.cat([ei + ej, ei * ej], dim=-1))
        pan = None
        if self.pan_stats is not None:
            raw = pan_descriptors(vectors, lengths, edge_index[1], h.shape[0], self.cutoff, self.p)
            pan = (raw - self.pan_stats[:, 0].to(raw.dtype)) / self.pan_stats[:, 1].to(raw.dtype)
        return {
            "h": h,
            "emb": emb,
            "sh": sh,
            "rbf": rbf,
            "env": env,
            "pair_code": pair_code,
            "pan": pan,
        }

    def step(
        self,
        t: int,
        h: torch.Tensor,
        ctx: dict[str, typing.Any],
        elem: torch.Tensor,
        edge_index: torch.Tensor,
        lengths: torch.Tensor,
        avg_num_neighbors: float,
        a_shift: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer ``t`` and its readout: the layer's features and atomic energies (unscaled)."""
        h = self.layers[t](
            h,
            elem,
            ctx["emb"],
            edge_index,
            ctx["sh"],
            ctx["rbf"],
            ctx["pair_code"],
            ctx["env"],
            avg_num_neighbors,
            lengths,
            ctx["pan"],
            a_shift if t == 0 else None,
        )
        scalars = h[:, : self.channels]
        if t == len(self.layers) - 1 and self.pgs is not None:
            scalars = self.pgs(scalars)  # the energy MLP reads the lifted invariants
        return h, self.readouts[t](scalars).squeeze(-1)

    def forward(
        self,
        elem: torch.Tensor,
        edge_index: torch.Tensor,
        vectors: torch.Tensor,
        lengths: torch.Tensor,
        avg_num_neighbors: float,
        a_shift: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Atomic energies ``e_i`` (unscaled) and the final invariant features.

        ``a_shift`` (denoising copies only) is added to the first layer's A-basis.
        """
        ctx = self.prepare(elem, edge_index, vectors, lengths)
        h = ctx["h"]
        energy = h.new_zeros(h.shape[0])
        for t in range(len(self.layers)):
            h, e = self.step(t, h, ctx, elem, edge_index, lengths, avg_num_neighbors, a_shift)
            energy = energy + e
        return energy, h


# =============================================================================
# WP11: edge cluster expansion (core: ece, and the ECE bond line)
# =============================================================================


def _density_irreps(channels: int, lmax: int) -> o3.Irreps:
    """``C x (0e + 1o + 2e + ...)`` up to ``lmax`` (natural parity)."""
    return o3.Irreps([(channels, (ell, (-1) ** ell)) for ell in range(lmax + 1)])


def _blocks(x: torch.Tensor, irreps: o3.Irreps) -> list[torch.Tensor]:
    """e3nn layout ``(N, dim)`` -> per-degree ``[(N, C, 2l+1)]``."""
    return [
        x[:, sl].reshape(x.shape[0], mul, ir.dim) for (mul, ir), sl in zip(irreps, irreps.slices())
    ]


def _ece_block(ece: dict, num_rbf: int, channels: int) -> nn.Module:
    """The GOAL ECE block configured from the ``ece:`` section."""
    from goal.ml.nn.blocks.ece import ECEBlock

    return ECEBlock(
        channels=channels,
        lmax=ece["lmax"],
        mmax=ece["mmax"],
        bottleneck=ece["bottleneck"],
        message_width=ece["message_width"],
        heads=ece["heads"],
        num_mp_layers=ece["num_mp_layers"],
        element_dim=ece["film"]["element_dim"],
        film_mlp=list(ece["film"]["mlp"]),
        num_rbf=num_rbf,
        activation=ece["activation"],
        grid_points=ece["grid_points"],
    )


class _ECECore(nn.Module):
    """``core: ece``: energy as a sum over directed bonds.

    Same interface as :class:`_AtomLine`: returns the atomic split of the
    bond energies (unscaled) and per-atom invariants (bond invariants
    summed over the atom's bonds, for the charge head).  Atomic densities
    are the layer-1 uvu A-basis from the element embeddings, with SIMURGH's
    radial basis, envelope and pair code.  The ECE gates take the place of
    the attention; PGS acts on the bond invariants.
    """

    def __init__(self, cfg: dict, num_elements: int) -> None:
        super().__init__()
        from goal.ml.nn.heads.edge_energy import EdgeEnergyHead
        from goal.ml.nn.primitives.edge_frame import EdgeFrames

        ece = cfg["ece"]
        c, lmax = ece["channels"], ece["lmax"]
        self.cutoff, self.p, self.mmax = cfg["cutoff"], cfg["envelope_p"], ece["mmax"]
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.irreps_dens = _density_irreps(c, lmax)
        self.rbf = _BesselBasis(cfg["cutoff"], cfg["num_rbf"])
        self.embedding = nn.Embedding(num_elements, cfg["embedding_dim"])
        self.codes = nn.Embedding(num_elements, ece["film"]["element_dim"])
        self.embed_to_h = nn.Linear(cfg["embedding_dim"], c, bias=False)
        self.pair_code = (
            nn.Linear(2 * cfg["embedding_dim"], cfg["embedding_dim"], bias=False)
            if cfg["pair_conditioned_radial"]
            else None
        )
        instructions = [(0, ell, ell, "uvu", True) for ell in range(lmax + 1)]
        self.tp = o3.TensorProduct(
            f"{c}x0e",
            self.irreps_sh,
            self.irreps_dens,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        in_dim = cfg["num_rbf"] + (cfg["embedding_dim"] if self.pair_code is not None else 0)
        self.radial = _MLP([in_dim, *cfg["radial_mlp"], self.tp.weight_numel])
        self.frames = EdgeFrames(lmax)
        self.block = _ece_block(ece, cfg["num_rbf"], c)
        self.head = EdgeEnergyHead(self.block.state, list(ece["energy_mlp"]))
        self.pgs = _pgs(cfg["readout"], self.block.state)
        self.channels = self.block.state  # width of the per-atom invariants
        self.last_bond_energies: torch.Tensor | None = None

    def forward(self, elem, edge_index, vectors, lengths, avg_num_neighbors):
        """Atomic energies (unscaled) and per-atom invariants ``(N, state)``."""
        n = elem.shape[0]
        sender, receiver = edge_index
        emb = self.embedding.weight
        sh = o3.spherical_harmonics(
            self.irreps_sh, vectors, normalize=True, normalization="component"
        )
        rbf = self.rbf(lengths)
        env = polynomial_envelope(lengths / self.cutoff, self.p)
        radial_in = [rbf]
        if self.pair_code is not None:
            ei, ej = emb[elem[sender]], emb[elem[receiver]]
            radial_in.append(self.pair_code(torch.cat([ei + ej, ei * ej], dim=-1)))
        weights = self.radial(torch.cat(radial_in, dim=-1)) * env.unsqueeze(-1)
        messages = self.tp(self.embed_to_h(emb[elem])[sender], sh, weights)
        dens = scatter(messages, receiver, dim=0, dim_size=n, reduce="sum") / avg_num_neighbors
        frames = self.frames(vectors)
        state = self.block(
            _blocks(dens, self.irreps_dens),
            edge_index,
            frames,
            self.codes.weight[elem],
            rbf,
            env,
            avg_num_neighbors,
        )
        inv = state[..., self.mmax]
        out = self.head(inv if self.pgs is None else self.pgs(inv), env, edge_index, n)
        atomic = out["atomic_energies"]
        bond = out["bond_energies"]
        self.last_bond_energies = bond
        h = scatter(inv * env.unsqueeze(-1), receiver, dim=0, dim_size=n, reduce="sum")
        return atomic, h / avg_num_neighbors


class _ECELine(nn.Module):
    """ECE bond line on the ACE core (``pair_line.mode: ece | phi+ece``).

    Atomic densities are the ACE core's final equivariant features,
    projected to ``ece.channels`` per degree; the bond-energy head starts at
    zero, so a fresh model equals the ACE-only model.
    """

    def __init__(self, cfg: dict, irreps_h: o3.Irreps) -> None:
        super().__init__()
        from goal.ml.nn.heads.edge_energy import EdgeEnergyHead
        from goal.ml.nn.primitives.edge_frame import EdgeFrames

        ece = cfg["ece"]
        lmax = min(ece["lmax"], max(ir.l for _, ir in irreps_h))
        self.cutoff, self.p = cfg["cutoff"], cfg["envelope_p"]
        self.irreps_dens = _density_irreps(ece["channels"], lmax)
        self.project = o3.Linear(irreps_h, self.irreps_dens)
        self.rbf = _BesselBasis(cfg["cutoff"], cfg["num_rbf"])
        self.frames = EdgeFrames(lmax)
        self.block = _ece_block({**ece, "lmax": lmax}, cfg["num_rbf"], ece["channels"])
        self.head = EdgeEnergyHead(self.block.state, list(ece["energy_mlp"]), zero_init=True)

    def forward(self, h, codes, edge_index, vectors, lengths, avg_num_neighbors) -> dict:
        """Bond energies (unscaled) of every directed edge and their atomic split."""
        env = polynomial_envelope(lengths / self.cutoff, self.p)
        dens = _blocks(self.project(h), self.irreps_dens)
        state = self.block(
            dens,
            edge_index,
            self.frames(vectors),
            codes,
            self.rbf(lengths),
            env,
            avg_num_neighbors,
        )
        return self.head(state[..., self.block.mmax], env, edge_index, h.shape[0])


# =============================================================================
# WP3: pair line
# =============================================================================


def _cubic_bspline_basis(r: torch.Tensor, r_min: float, r_max: float, num: int) -> torch.Tensor:
    """Uniform cubic B-splines on ``[r_min, r_max]``; ``num`` >= 4 functions."""
    h = (r_max - r_min) / (num - 3)
    knots = r_min + (torch.arange(num, dtype=r.dtype, device=r.device) - 3) * h
    u = (r.unsqueeze(-1) - knots) / h
    out = torch.zeros_like(u)
    out = torch.where((u >= 0) & (u < 1), u**3 / 6, out)
    out = torch.where((u >= 1) & (u < 2), (-3 * u**3 + 12 * u**2 - 12 * u + 4) / 6, out)
    out = torch.where((u >= 2) & (u < 3), (3 * u**3 - 24 * u**2 + 60 * u - 44) / 6, out)
    return torch.where((u >= 3) & (u < 4), (4 - u) ** 3 / 6, out)


_COULOMB = 14.399645  # e² / (4π ε0) in eV Å


class _PairLine(nn.Module):
    """``φ_ab(r) = u(r) Σ_n c_ab,n B_n(r)`` per unordered element pair, plus a fixed wall.

    Coefficients come from a ``(n_pairs, K)`` table gathered per edge (or a
    low-rank map of the symmetric pair code) and start at zero.  The wall
    (ZBL or a single exponential) is physical (eV, not scaled by ``s``)
    and acts only below ``r_min``.
    """

    def __init__(self, cfg: dict, elements: list[int], embedding_dim: int, cutoff: float) -> None:
        super().__init__()
        pc = cfg["pair_cutoff"] or cutoff
        if pc > cutoff:
            raise ValueError(
                f"pair_line.pair_cutoff ({pc}) cannot exceed the atom cutoff ({cutoff})"
            )
        self.cutoff, self.r_min, self.num = float(pc), float(cfg["r_min"]), int(cfg["num_basis"])
        self.basis = _choice(cfg, "basis", ("bessel", "bspline"), "pair_line.")
        self.wall = _choice(cfg, "repulsive_wall", ("zbl", "exp", "none"), "pair_line.")
        self.coefficients = _choice(cfg, "coefficients", ("table", "lowrank"), "pair_line.")
        if self.basis == "bessel":
            self.bessel = _BesselBasis(pc, self.num)
        n = len(elements)
        lookup = torch.full((n, n), -1, dtype=torch.long)
        for k, (a, b) in enumerate(itertools.combinations_with_replacement(range(n), 2)):
            lookup[a, b] = lookup[b, a] = k
        self.register_buffer("pair_id", lookup)
        self.register_buffer("z", torch.tensor(elements, dtype=torch.get_default_dtype()))
        if self.coefficients == "table":
            self.table = nn.Parameter(torch.zeros(n * (n + 1) // 2, self.num))
        else:
            self.lowrank = nn.Linear(2 * embedding_dim, self.num, bias=False)
            nn.init.zeros_(self.lowrank.weight)

    def _basis(self, r: torch.Tensor) -> torch.Tensor:
        """Radial basis ``(E, K)`` of the pair potential."""
        if self.basis == "bessel":
            return self.bessel(r)
        return _cubic_bspline_basis(r, self.r_min, self.cutoff, self.num)

    def learned(
        self, r: torch.Tensor, a: torch.Tensor, b: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Learned ``φ_ab(r)`` (unscaled) for element indices ``a``, ``b``."""
        if self.coefficients == "table":
            c = self.table[self.pair_id[a, b]]
        else:
            c = self.lowrank(torch.cat([emb[a] + emb[b], emb[a] * emb[b]], dim=-1))
        return polynomial_envelope(r / self.cutoff, 5) * (c * self._basis(r)).sum(-1)

    def wall_energy(self, r: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fixed short-range repulsion (eV), zero with zero slope at ``r >= r_min``."""
        if self.wall == "none":
            return torch.zeros_like(r)
        za, zb = self.z[a].to(r.dtype), self.z[b].to(r.dtype)
        switch = polynomial_envelope(r / self.r_min, 5)
        rs = torch.where(r < self.r_min, r, torch.full_like(r, self.r_min))  # finite beyond r_min
        if self.wall == "zbl":
            x = rs * (za**0.23 + zb**0.23) / 0.46850
            phi = (
                0.18175 * torch.exp(-3.19980 * x)
                + 0.50986 * torch.exp(-0.94229 * x)
                + 0.28022 * torch.exp(-0.40290 * x)
                + 0.02817 * torch.exp(-0.20162 * x)
            )
            return _COULOMB * za * zb / rs * phi * switch
        return (
            _COULOMB
            * za
            * zb
            / self.r_min
            * torch.exp(-3.0 * (rs - self.r_min) / self.r_min)
            * switch
        )

    def curvature_penalty(self, pairs: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """``∫ (φ'')² dr`` on a 256-point grid, summed over the given ``(a, b)`` pairs."""
        grid = torch.linspace(
            0.5 * self.r_min, self.cutoff, 256, dtype=emb.dtype, device=emb.device
        )
        step = float(grid[1] - grid[0])
        a = pairs[:, 0:1].expand(-1, 256)
        b = pairs[:, 1:2].expand(-1, 256)
        phi = self.learned(grid.expand_as(a.to(grid.dtype)), a, b, emb)
        second = (phi[:, 2:] - 2 * phi[:, 1:-1] + phi[:, :-2]) / step**2
        return (second.square().sum(-1) * step).sum()


# =============================================================================
# WP1: analytic D3(BJ) dispersion (no DFT, no trainable parameters)
# =============================================================================

_BOHR = 0.52917721067  # Å
_HARTREE = 27.211386245988  # eV

#: PBE D3(BJ), Grimme, Ehrlich and Goerigk 2011 (doi:10.1002/jcc.21759).
D3BJ_PARAMETERS: dict[str, dict[str, float]] = {
    "pbe": {"s6": 1.0, "a1": 0.4289, "s8": 0.7875, "a2": 4.4407},
}


class _D3Term(nn.Module):
    """Fixed D3(BJ) energy inside the energy sum.

    ``torch`` evaluates ``tad-dftd3`` in the forward pass (forces by
    autograd).  ``precomputed`` adds ``energy_d3`` / ``forces_d3`` stored
    on the graph, or evaluates ``dftd3-python`` (simple-dftd3) on the fly
    when they are absent: the value is exact and the gradient is the D3
    gradient, through a term linear in the positions (its Hessian is not
    represented, which nothing in training or MD uses).  Two-body only
    unless ``atm`` is set.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.backend = _choice(cfg, "backend", ("torch", "precomputed"), "dispersion.")
        _choice(cfg, "damping", ("bj",), "dispersion.")
        functional = _choice(cfg, "functional", tuple(D3BJ_PARAMETERS), "dispersion.")
        self.params = {**D3BJ_PARAMETERS[functional], "s9": 1.0 if cfg["atm"] else 0.0}
        if self.backend == "torch":
            try:
                import tad_dftd3  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "dispersion.backend: torch needs tad-dftd3 (pip install tad-dftd3); "
                    "use dispersion.backend: precomputed for dftd3-python"
                ) from exc

    def forward(
        self, graph: AtomicGraph, pos: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        """Per-graph D3 energy ``(G,)`` whose gradient is the D3 gradient."""
        if self.backend == "torch":
            return self._tad(graph.atomic_numbers, pos, batch, num_graphs)
        energy, forces = graph.get("energy_d3", None), graph.get("forces_d3", None)
        if energy is None or forces is None:
            energy, forces = self._dftd3(graph.atomic_numbers, pos.detach(), batch, num_graphs)
        energy = energy.to(pos.dtype).reshape(-1)
        linear = -(forces.to(pos.dtype) * (pos - pos.detach())).sum(-1)
        return energy + scatter(linear, batch, dim=0, dim_size=num_graphs, reduce="sum")

    def _dftd3(self, numbers, pos, batch, num_graphs) -> tuple[torch.Tensor, torch.Tensor]:
        """Energies ``(G,)`` and forces ``(N, 3)`` from dftd3-python (eV, eV/Å)."""
        from dftd3.interface import DispersionModel, RationalDampingParam

        param = RationalDampingParam(**self.params)
        energy = torch.zeros(num_graphs, dtype=torch.float64)
        forces = torch.zeros(pos.shape[0], 3, dtype=torch.float64)
        z, x, b = numbers.cpu().numpy(), pos.cpu().double().numpy(), batch.cpu().numpy()
        for g in range(num_graphs):
            idx = np.flatnonzero(b == g)
            res = DispersionModel(z[idx], x[idx] / _BOHR).get_dispersion(param, grad=True)
            energy[g] = res["energy"] * _HARTREE
            forces[idx] = torch.from_numpy(-res["gradient"] * _HARTREE / _BOHR)
        return energy.to(pos.device), forces.to(pos.device)

    def _tad(
        self, numbers, pos, batch, num_graphs
    ) -> torch.Tensor:  # pragma: no cover - optional dep
        """Per-graph energies from tad-dftd3, differentiable in ``pos``."""
        import tad_dftd3 as d3

        param = {
            k: torch.tensor(v, dtype=pos.dtype, device=pos.device) for k, v in self.params.items()
        }
        out = []
        for g in range(num_graphs):
            mask = batch == g
            e = d3.dftd3(numbers[mask], pos[mask] / _BOHR, param)
            out.append(e.sum() * _HARTREE)
        return torch.stack(out)


# =============================================================================
# WP7: factorised electrostatics (default off)
# =============================================================================


def damped_coulomb(r: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``T0 = erf(r / (√2 σ)) / r`` and the radial factors ``B``, ``C`` of its gradients.

    Charge-dipole terms use ``B = -T0'/r``; dipole-dipole terms ``B`` and
    ``C = (T0'' - T0'/r) / r²``.  For ``σ -> 0`` they become ``1/r``, ``1/r³``
    and ``3/r⁵``.
    """
    a = 1.0 / (math.sqrt(2.0) * sigma)
    erf = torch.erf(a * r)
    g = 2.0 * a / math.sqrt(math.pi) * torch.exp(-((a * r) ** 2))
    t0 = erf / r
    t1 = g / r - erf / r**2
    t2 = -2.0 * a**2 * g - 2.0 * g / r**2 + 2.0 * erf / r**3
    return t0, -t1 / r, (t2 - t1 / r) / r**2


class _ChargeHead(nn.Module):
    """Latent charges from final 0e features that sum exactly to each molecule's charge Q.

    ``uniform`` (DOI 10.1021/jacs.5c09558, SI Eq. S12): ``q_i = q̂_i + (Q - Σ_j q̂_j)/N``
    with ``q̂_i = q_Z + MLP(s_i)``, a per-molecule mean.  ``pair_transfer``
    (local): ``q_i = Q/N + Σ_j t_ij`` over pairs within ``cutoff``, with
    ``t_ij = [φ(s_i, s_j, r) - φ(s_j, s_i, r)] u(r) = -t_ji``.  ``q_Z`` and the
    last layers start at zero, so a fresh head gives ``q_i = Q/N``.
    """

    def __init__(
        self,
        channels: int,
        num_elements: int,
        constraint: str,
        cutoff: float,
        p: int,
        num_rbf: int,
    ) -> None:
        super().__init__()
        self.constraint, self.channels, self.cutoff, self.p = constraint, channels, cutoff, p
        if constraint == "uniform":
            self.q_z = nn.Parameter(torch.zeros(num_elements))
            self.mlp = _MLP([channels, 64, 1], zero_last=True)
        else:
            self.rbf = _BesselBasis(cutoff, num_rbf)
            self.phi = _MLP([2 * channels + num_rbf, 64, 1], zero_last=True)

    def forward(
        self,
        h: torch.Tensor,
        elem: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        pairs: torch.Tensor,
        lengths: torch.Tensor,
        total: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Charges ``(N,)``; ``pairs`` ``(2, P)`` with ``lengths`` feed ``pair_transfer``."""
        s = h[:, : self.channels]
        count = scatter(torch.ones_like(s[:, 0]), batch, dim=0, dim_size=num_graphs, reduce="sum")
        total = s.new_zeros(num_graphs) if total is None else total.to(s.dtype).reshape(-1)
        if self.constraint == "uniform":
            raw = self.q_z[elem] + self.mlp(s).squeeze(-1)
            excess = total - scatter(raw, batch, dim=0, dim_size=num_graphs, reduce="sum")
            return raw + (excess / count)[batch]
        i, j = pairs
        keep = lengths < self.cutoff
        i, j, r = i[keep], j[keep], lengths[keep]
        rbf = self.rbf(r)
        forward = self.phi(torch.cat([s[i], s[j], rbf], dim=-1))
        backward = self.phi(torch.cat([s[j], s[i], rbf], dim=-1))
        t = (forward - backward).squeeze(-1) * polynomial_envelope(r / self.cutoff, self.p)
        n = s.shape[0]
        moved = scatter(t, i, dim=0, dim_size=n, reduce="sum") - scatter(
            t, j, dim=0, dim_size=n, reduce="sum"
        )
        return (total / count)[batch] + moved


def _intramolecular_pairs(batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Every pair ``i < j`` inside each molecule, ``(2, P)``."""
    pairs, start = [], 0
    for n in torch.bincount(batch, minlength=num_graphs).tolist():
        pairs.append(torch.triu_indices(n, n, offset=1, device=batch.device) + start)
        start += n
    return torch.cat(pairs, dim=1)


class _Electrostatics(nn.Module):
    """Latent charges (and optional dipoles) from local features; open-boundary energy.

    ``q_i = Linear(h_i[0e])`` shifted so every molecule is neutral,
    ``μ_i = Linear(h_i[1o])``; both heads start at zero, so a fresh model is
    unchanged.  Gaussian widths ``σ_ij = √(σ_i² + σ_j²)`` with ``σ`` per
    element (softplus, starting at 1 Å) or fixed.  The energy is in eV and
    not scaled by ``s``.  Population charges are never supervised.
    """

    def __init__(self, cfg: dict, channels: int, num_elements: int, model_cfg: dict) -> None:
        super().__init__()
        self.charge = _ChargeHead(
            channels,
            num_elements,
            cfg["charge_constraint"],
            model_cfg["cutoff"],
            model_cfg["envelope_p"],
            model_cfg["num_rbf"],
        )
        self.dipole_head = None
        if cfg["dipoles"]:
            self.dipole_head = o3.Linear(f"{channels}x1o", "1x1o")
            with torch.no_grad():
                self.dipole_head.weight.zero_()
        self.sigma_raw = None
        self.fixed_sigma = None
        if cfg["sigma"] == "learned":
            init = math.log(math.e - 1.0)  # softplus(init) = 1 Å
            self.sigma_raw = nn.Parameter(torch.full((num_elements,), init))
        else:
            self.fixed_sigma = float(cfg["sigma"])
        self.channels = channels
        _choice(cfg, "charge_constraint", ("uniform", "pair_transfer"), "electrostatics.")

    def charges(
        self,
        h: torch.Tensor,
        elem: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        total: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Latent charges ``(N,)`` summing to each molecule's ``total`` (0 when absent)."""
        i, j = _intramolecular_pairs(batch, num_graphs)
        r = (pos[j] - pos[i]).norm(dim=-1)
        return self.charge(h, elem, batch, num_graphs, torch.stack([i, j]), r, total)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        elem: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        total: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Energy per molecule, charges and the molecular dipole."""
        q = self.charges(h, elem, pos, batch, num_graphs, total)
        mu = None if self.dipole_head is None else self.dipole_head(h[:, self.channels :])
        i, j = _intramolecular_pairs(batch, num_graphs)
        rij = pos[j] - pos[i]
        r = rij.norm(dim=-1)
        if self.sigma_raw is not None:
            sig = nn.functional.softplus(self.sigma_raw)
            sigma = torch.sqrt(sig[elem[i]] ** 2 + sig[elem[j]] ** 2)
        else:
            sigma = torch.full_like(r, math.sqrt(2.0) * self.fixed_sigma)
        t0, b, c = damped_coulomb(r, sigma)
        e = q[i] * q[j] * t0
        dipole = scatter(q.unsqueeze(-1) * pos, batch, dim=0, dim_size=num_graphs, reduce="sum")
        if mu is not None:
            mi, mj = (mu[i] * rij).sum(-1), (mu[j] * rij).sum(-1)
            e = e + (q[j] * mi - q[i] * mj) * b + (mu[i] * mu[j]).sum(-1) * b - mi * mj * c
            dipole = dipole + scatter(mu, batch, dim=0, dim_size=num_graphs, reduce="sum")
        energy = _COULOMB * scatter(e, batch[i], dim=0, dim_size=num_graphs, reduce="sum")
        return {"energy": energy, "charges": q, "dipole": dipole}


# =============================================================================
# WP6: multi-scale fragment consistency (MFC)
# =============================================================================
#
# One engine for training and diagnosis.  A *window* W is a subset of a
# structure's atoms; its *core* is the atoms of W at least ``buffer`` from
# every atom of the parent that W removed.  The same network evaluated on
# W must reproduce, on the core atoms, the atom-line energies (and the
# forces those energies generate) it predicts inside the parent: the full
# structure, or the next larger window.  With ``buffer >= r_c`` the
# objectives vanish exactly for a strictly local (one-layer) model, so
# they penalise only dependence carried by message passing.


@dataclasses.dataclass
class Window:
    """One window of one structure: local atom indices of ``W`` and its core.

    ``parent`` is the index (in the same list) of the enclosing window, or
    -1 for the full structure.
    """

    nodes: np.ndarray
    core: np.ndarray
    parent: int
    #: positions of H caps on the window's cut bonds (``chemical`` windows only)
    caps: np.ndarray | None = None


def core_mask(
    positions: np.ndarray, inside: np.ndarray, parent: np.ndarray, buffer: float
) -> np.ndarray:
    """Atoms of ``inside`` at least ``buffer`` from every atom of ``parent & ~inside``."""
    removed = parent & ~inside
    if not removed.any():
        return np.zeros_like(inside)
    d = np.linalg.norm(positions[:, None] - positions[None, removed], axis=-1).min(axis=1)
    return inside & (d >= buffer)


def principal_axis(positions: np.ndarray) -> np.ndarray:
    """Largest-variance direction, signed so the projections have positive skew."""
    x = positions - positions.mean(axis=0)
    axis = np.linalg.eigh(x.T @ x)[1][:, -1]
    skew = float(((x @ axis) ** 3).sum())
    return axis if skew >= 0 else -axis


def halfspace_windows(
    positions: np.ndarray,
    heavy: np.ndarray,
    buffer: float,
    levels: int,
    rng: np.random.Generator,
    principal_prob: float = 0.5,
    tries: int = 8,
) -> list[Window]:
    """Nested half-space cuts ``{(r - r̄)·u <= t_1 < ... }`` along one direction.

    ``u`` is the principal axis with probability ``principal_prob``, else
    uniform on the sphere; thresholds are uniform over the projected range.
    A level is valid when it removes at least one heavy atom and its core
    holds one.  Returns ``[]`` after ``tries`` failures.
    """
    full = np.ones(len(positions), dtype=bool)
    for _ in range(tries):
        if rng.random() < principal_prob:
            u = principal_axis(positions)
        else:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
        s = (positions - positions.mean(axis=0)) @ u
        cuts = np.sort(rng.uniform(s.min(), s.max(), size=levels - 1))
        windows: list[Window] = []
        parent, parent_index = full, -1
        for t in cuts[::-1]:  # largest window first
            inside = s <= t
            core = core_mask(positions, inside, parent, buffer)
            if not (heavy & parent & ~inside).any() or not (heavy & core).any():
                break
            windows.append(Window(np.flatnonzero(inside), np.flatnonzero(core), parent_index))
            parent, parent_index = inside, len(windows) - 1
        if len(windows) == levels - 1:
            return windows
    return []


def sphere_windows(
    positions: np.ndarray,
    heavy: np.ndarray,
    buffer: float,
    radii: typing.Sequence[float],
    rng: np.random.Generator,
    tries: int = 8,
) -> list[Window]:
    """Nested spheres of ``radii`` around a random heavy seed atom."""
    full = np.ones(len(positions), dtype=bool)
    seeds = np.flatnonzero(heavy) if heavy.any() else np.arange(len(positions))
    for _ in range(tries):
        d = np.linalg.norm(positions - positions[rng.choice(seeds)], axis=1)
        windows: list[Window] = []
        parent, parent_index = full, -1
        for radius in sorted(radii, reverse=True):
            inside = d <= radius
            core = core_mask(positions, inside, parent, buffer)
            if inside.all() or not (parent & ~inside).any() or not (heavy & core).any():
                break
            windows.append(Window(np.flatnonzero(inside), np.flatnonzero(core), parent_index))
            parent, parent_index = inside, len(windows) - 1
        if len(windows) == len(radii):
            return windows
    return []


#: Cap bond length (Å) from the kept atom X of a cut bond X-Y.
CAP_LENGTH: dict[int, float] = {6: 1.09, 7: 1.01, 8: 0.96}


def chemical_windows(
    positions: np.ndarray,
    numbers: np.ndarray,
    buffer: float,
    levels: int,
    rng: np.random.Generator,
    scheme: str = "rotatable",
    tries: int = 8,
) -> list[Window]:
    """Nested windows of chemical groups (``goal.ml.data.fragments``) with H caps.

    Level 1 is one randomly chosen group; each further level adds the groups
    bonded to it.  Every cut bond X-Y (X kept) is capped with an H on the
    X→Y line at :data:`CAP_LENGTH` from X (C, N or O; other X: no window).
    The core keeps ``buffer`` from every removed atom and every cap.
    """
    from ase.data import covalent_radii

    from goal.ml.data.fragments import compute_fragment_index

    groups = compute_fragment_index(
        torch.as_tensor(positions), torch.as_tensor(numbers), scheme=scheme, on_failure="fallback"
    ).numpy()
    radius = covalent_radii[numbers]
    d = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    bonded = (d < 1.2 * (radius[:, None] + radius[None])) & ~np.eye(len(numbers), dtype=bool)
    heavy = numbers > 1
    full = np.ones(len(numbers), dtype=bool)
    for _ in range(tries):
        inside = groups == rng.choice(np.unique(groups))
        levels_in = [inside]
        for _ in range(levels - 2):
            grown = levels_in[-1] | np.isin(groups, groups[bonded[levels_in[-1]].any(axis=0)])
            levels_in.append(grown)
        windows: list[Window] = []
        parent, parent_index = full, -1
        for inside in reversed(levels_in):  # largest first, as the other generators
            x, y = np.nonzero(bonded & inside[:, None] & ~inside[None, :])
            if (
                inside.all()
                or not (parent & ~inside).any()
                or not np.isin(numbers[x], list(CAP_LENGTH)).all()
            ):
                break
            unit = positions[y] - positions[x]
            unit /= np.linalg.norm(unit, axis=1, keepdims=True)
            caps = positions[x] + unit * np.array([CAP_LENGTH[z] for z in numbers[x]])[:, None]
            core = core_mask(positions, inside, parent, buffer)
            if len(caps):
                far = (
                    np.linalg.norm(positions[:, None] - caps[None], axis=-1).min(axis=1) >= buffer
                )
                core &= far
            if not (heavy & core).any():
                break
            windows.append(
                Window(np.flatnonzero(inside), np.flatnonzero(core), parent_index, caps)
            )
            parent, parent_index = inside, len(windows) - 1
        if len(windows) == levels - 1:
            return windows
    return []


class _MFC(nn.Module):
    """Training objective of multi-scale fragment consistency.

    Builds windows per structure, evaluates the atom line on copies of the
    window atoms (batched), and returns the weighted consistency loss and
    its parts.  Parent (teacher) values are detached with
    ``stop_grad: parent``; the window values with ``child``; neither with
    ``none``.  Running scales normalise the energy and force terms.
    """

    def __init__(
        self, cfg: dict, cutoff: float, channels: int = 0, cap_element: int | None = None
    ) -> None:
        super().__init__()
        self.cfg, self.cutoff = cfg, cutoff
        self.window_type = _choice(cfg, "window_type", ("halfspace", "sphere", "chemical"), "mfc.")
        self.stop_grad = _choice(cfg, "stop_grad", ("parent", "none", "child"), "mfc.")
        self.objectives = set(cfg["objective"])
        if not self.objectives or self.objectives - {"energy", "core_force", "force"}:
            raise ValueError(
                f"mfc.objective must be a subset of energy, core_force, force: {cfg['objective']}"
            )
        if cfg["levels"] not in (2, 3):
            raise ValueError("mfc.levels must be 2 or 3")
        if not 0.0 < float(cfg["fraction"]) <= 1.0:
            raise ValueError(f"mfc.fraction must be in (0, 1], got {cfg['fraction']}")
        if self.window_type == "sphere" and len(cfg["radii"]) != cfg["levels"] - 1:
            raise ValueError("mfc.radii needs levels - 1 radii")
        _choice(cfg, "parameters", ("shared", "separate"), "mfc.")
        _choice(cfg, "window_weighting", ("uniform", "violation"), "mfc.")
        if _choice(cfg, "tolerance", ("none", "calibrated"), "mfc.") == "calibrated":
            raise NotImplementedError(
                "mfc.tolerance: calibrated needs the maintainer's offline calibration file, "
                f"mfc.tolerance_file (now {cfg['tolerance_file']!r}): a CSV with columns "
                "d_min_A, d_max_A, tau_energy_eV and tau_force_eV_per_A, one row per bin of the "
                "distance to the nearest removed or cap atom, from ORCA single points on capped "
                "chemical windows.  Not wired yet."
            )
        #: ``parameters: separate``: residual sections of levels 2..L (the full structure is L)
        self.residual = (
            nn.ModuleList(
                _MLP([2 * channels, 64, 1], zero_last=True) for _ in range(cfg["levels"] - 1)
            )
            if cfg["parameters"] == "separate"
            else None
        )
        self.cap_element = cap_element
        if self.window_type == "chemical" and cap_element is None:
            raise ValueError(
                "mfc.window_type: chemical caps cut bonds with H: H must be in elements"
            )
        #: ``window_weighting: violation``: mean core-force violation per structure (by geometry)
        self.violation: dict[bytes, float] = {}
        self.register_buffer("sigma2", torch.zeros(3))  # running scales: energy, force, charge
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.rng = np.random.default_rng(0)
        self.principal_prob = 0.5  # half-space direction: principal axis vs uniform

    def weight(self, epoch: int) -> float:
        """λ_MFC: 0 before ``start_epoch``, then a linear ramp over ``warmup_epochs``."""
        start, ramp = self.cfg["start_epoch"], max(int(self.cfg["warmup_epochs"]), 1)
        if epoch < start:
            return 0.0
        return float(self.cfg["weight"]) * min(1.0, (epoch - start + 1) / ramp)

    def structures(self, batch: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """The structures that get windows: all, or a ``mfc.fraction`` of them (∝ ``weights``)."""
        graphs = np.unique(batch)
        fraction = float(self.cfg["fraction"])
        if fraction < 1.0:
            k = max(1, round(fraction * len(graphs)))
            p = None if weights is None else weights[graphs] / weights[graphs].sum()
            graphs = np.sort(self.rng.choice(graphs, size=k, replace=False, p=p))
        return graphs

    @staticmethod
    def keys(pos: np.ndarray, numbers: np.ndarray, batch: np.ndarray, num_graphs: int) -> list:
        """A geometry key per structure (identifies it across epochs)."""
        return [
            hash((numbers[batch == g].tobytes(), np.round(pos[batch == g], 4).tobytes()))
            for g in range(num_graphs)
        ]

    def structure_weights(self, keys: list) -> np.ndarray:
        """``w_s = clip(v_s / mean v, 0.25, 4)``, mean 1 over scored structures; 1 if unscored."""
        if self.cfg["window_weighting"] != "violation" or not self.violation:
            return np.ones(len(keys))
        mean = float(np.mean(list(self.violation.values())))
        scored = np.array([k in self.violation for k in keys])
        w = np.ones(len(keys))
        if scored.any() and mean > 0:
            v = np.array([self.violation.get(k, mean) for k in keys])
            raw = np.clip(v / mean, 0.25, 4.0)
            w[scored] = raw[scored] / raw[scored].mean()
        return w

    def residual_energy(self, h: torch.Tensor, batch: torch.Tensor, num_graphs: int):
        """``parameters: separate``: ``Δe^(L)`` per atom of the full structure, and a zero
        holding the other levels' sections on the graph."""
        s = h[:, : self.residual[0][0].in_features // 2]
        pooled = scatter(s, batch, dim=0, dim_size=num_graphs, reduce="mean")[batch]
        x = torch.cat([s, pooled], dim=-1)
        held = sum(p.sum() for m in self.residual[:-1] for p in m.parameters()) * 0.0
        return self.residual[-1](x).squeeze(-1), held

    def windows(
        self, pos: np.ndarray, numbers: np.ndarray, batch: np.ndarray, graphs: np.ndarray
    ) -> list[tuple[int, Window, int]]:
        """``(structure, window, slot)`` for every window of the given structures."""
        out = []
        for g in graphs:
            idx = np.flatnonzero(batch == g)
            heavy = numbers[idx] > 1
            for k in range(int(self.cfg["windows_per_structure"])):
                if self.window_type == "halfspace":
                    ws = halfspace_windows(
                        pos[idx],
                        heavy,
                        self.cfg["buffer"],
                        self.cfg["levels"],
                        self.rng,
                        self.principal_prob,
                    )
                elif self.window_type == "sphere":
                    ws = sphere_windows(
                        pos[idx], heavy, self.cfg["buffer"], self.cfg["radii"], self.rng
                    )
                else:
                    ws = chemical_windows(
                        pos[idx],
                        numbers[idx],
                        self.cfg["buffer"],
                        self.cfg["levels"],
                        self.rng,
                        self.cfg["chemical_scheme"],
                    )
                base = len(out)
                for level, w in enumerate(ws):
                    parent = -1 if w.parent < 0 else base + w.parent
                    window = Window(idx[w.nodes], idx[w.core], parent, w.caps)
                    out.append((g, window, k * 3 + level))
        return out

    def _scale(self, i: int, value: torch.Tensor) -> torch.Tensor:
        """Running mean of ``value`` (squared targets), initialised on first use."""
        with torch.no_grad():
            v = value.detach().to(self.sigma2.dtype)
            if self.sigma2[i] == 0:
                self.sigma2[i] = v
            else:
                self.sigma2[i] = 0.99 * self.sigma2[i] + 0.01 * v
        return self.sigma2[i].clamp_min(1e-12)

    def forward(
        self,
        model: SimurghDev,
        graph: AtomicGraph,
        pos: torch.Tensor,
        pos_d: torch.Tensor,
        elem: torch.Tensor,
        batch: torch.Tensor,
        e_full: torch.Tensor,
        q_full: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """MFC loss (``mfc``) and logged parts for one training batch."""
        zero = e_full.new_zeros(())
        lam = self.weight(model.epoch)
        self.step += 1
        if lam == 0.0 or int(self.step) % int(self.cfg["every_n_steps"]):
            return {"mfc": zero}
        shifts = graph.get("unit_shifts", None)
        if shifts is not None and bool(shifts.abs().sum() > 0):
            raise ValueError("MFC windows are defined for molecules (non-periodic graphs)")
        batch_np = batch.cpu().numpy()
        pos_np, numbers_np = pos_d.detach().cpu().numpy(), graph.atomic_numbers.cpu().numpy()
        num_graphs = int(batch_np.max()) + 1
        violation = self.cfg["window_weighting"] == "violation"
        keys = self.keys(pos_np, numbers_np, batch_np, num_graphs) if violation else None
        weights = self.structure_weights(keys) if violation else np.ones(num_graphs)
        graphs = self.structures(batch_np, weights if violation else None)
        windows = self.windows(pos_np, numbers_np, batch_np, graphs)
        valid = len({g for g, _, _ in windows}) / len(graphs)
        if not windows:
            return {"mfc": zero, "mfc_valid_fraction": zero + valid}

        # copies of the window atoms, one block per window
        device = pos.device
        copies = [torch.as_tensor(w.nodes, device=device) for _, w, _ in windows]
        caps = [
            pos_d.new_zeros(0, 3) if w.caps is None else torch.as_tensor(w.caps).to(pos_d)
            for _, w, _ in windows
        ]  # constant H caps (chemical windows), after each window's own atoms
        offsets = np.cumsum([0] + [len(c) + len(p) for c, p in zip(copies, caps)])
        lookup = torch.full((pos.shape[0],), -1, dtype=torch.long, device=device)
        row, col = graph.edge_index
        edges, pos_blocks, elem_blocks = [], [], []
        for k, (c, p) in enumerate(zip(copies, caps)):
            pos_blocks += [pos_d[c], p]
            elem_blocks += [elem[c], elem.new_full((len(p),), self.cap_element or 0)]
            if len(p):  # caps: the window's own neighbour list
                x = torch.cat([pos_d[c], p]).detach()
                pair = (torch.cdist(x, x) < self.cutoff) & ~torch.eye(
                    len(x), dtype=torch.bool, device=device
                )
                edges.append(torch.nonzero(pair).T + int(offsets[k]))
                continue
            lookup.fill_(-1)
            lookup[c] = torch.arange(len(c), device=device) + int(offsets[k])
            keep = (lookup[row] >= 0) & (lookup[col] >= 0)
            edges.append(torch.stack([lookup[row[keep]], lookup[col[keep]]]))
        edge_index = torch.cat(edges, dim=1)
        elem_w = torch.cat(elem_blocks)
        child_grad = self.stop_grad != "child"
        parent_grad = self.stop_grad != "parent"
        pos_w = torch.cat(pos_blocks)
        if not child_grad:
            pos_w = pos_w.detach().requires_grad_(True)
        vec = pos_w[edge_index[1]] - pos_w[edge_index[0]]
        # the same line energies as the full structure's e_atom (the ECE line's atomic split
        # e_i = 1/2 sum_j (eps_ij + eps_ji) included), or MFC compares unlike quantities
        e_w, h_w, _ = model.line_energies(elem_w, edge_index, vec, vec.norm(dim=-1))
        window_weight = [float(weights[g]) for g, _, _ in windows]
        s = model.scale

        def block(k: int) -> slice:
            """The window's own atoms (caps excluded)."""
            return slice(int(offsets[k]), int(offsets[k]) + len(windows[k][1].nodes))

        def rows(k: int, atoms: np.ndarray) -> torch.Tensor:
            """Rows of window ``k``'s copy holding the given (global) atoms."""
            return torch.as_tensor(
                int(offsets[k]) + np.searchsorted(windows[k][1].nodes, atoms), device=device
            )

        def parent_rows(k: int, atoms: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
            """``(energies, rows)`` of the given atoms inside window ``k``'s parent."""
            p = windows[k][1].parent
            if p < 0:
                return e_full, torch.as_tensor(atoms, device=device)
            return e_w, rows(p, atoms)

        def maybe_detach(x: torch.Tensor, keep: bool) -> torch.Tensor:
            return x if keep else x.detach()

        n_windows = len(windows)
        out: dict[str, torch.Tensor] = {"mfc_valid_fraction": zero + valid}
        score_now = model.epoch % int(self.cfg["violation_every"]) == 0
        scores: dict = {}
        loss = zero
        if "energy" in self.objectives:
            child, parent, wt = [], [], []
            for k, (_, w, _) in enumerate(windows):
                child.append(s * e_w[rows(k, w.core)])
                e_p, r_p = parent_rows(k, w.core)
                parent.append(s * e_p[r_p])
                wt.append(torch.full((len(w.core),), window_weight[k], device=device))
            child_t = maybe_detach(torch.cat(child), child_grad)
            parent_t = maybe_detach(torch.cat(parent), parent_grad)
            sigma = self._scale(0, parent_t.detach().square().mean())
            wt_t = torch.cat(wt).to(child_t.dtype)
            term = ((child_t - parent_t).square() * wt_t).sum() / sigma / n_windows
            out["mfc_energy"] = term.detach()
            loss = loss + self.cfg["weight_energy"] * term
        if q_full is not None and self.cfg["include_charges"]:
            owner = torch.repeat_interleave(
                torch.arange(n_windows, device=device),
                torch.as_tensor(np.diff(offsets), device=device),
            )
            q_w = model.electrostatics.charges(h_w, elem_w, pos_w, owner, n_windows)
            child, parent = [], []
            for k, (_, w, _) in enumerate(windows):
                child.append(q_w[rows(k, w.core)])
                p = w.parent
                parent.append(
                    q_full[torch.as_tensor(w.core, device=device)]
                    if p < 0
                    else q_w[rows(p, w.core)]
                )
            child_t = maybe_detach(torch.cat(child), child_grad)
            parent_t = maybe_detach(torch.cat(parent), parent_grad)
            sigma = self._scale(2, parent_t.detach().square().mean())
            term = (child_t - parent_t).square().sum() / sigma / n_windows
            out["mfc_charge"] = term.detach()
            loss = loss + self.cfg["weight_energy"] * term
        for name in sorted(self.objectives & {"core_force", "force"}):
            term = zero
            for slot in sorted({sl for _, _, sl in windows}):
                ks = [k for k, (_, _, sl) in enumerate(windows) if sl == slot]
                full_parent = windows[ks[0]][1].parent < 0  # one level per slot
                q_child, q_parent = zero, zero
                for k in ks:
                    g, w, _ = windows[k]
                    if name == "core_force":
                        q_child = q_child + (s * e_w[rows(k, w.core)]).sum()
                        e_p, r_p = parent_rows(k, w.core)
                        q_parent = q_parent + (s * e_p[r_p]).sum()
                    else:  # full atom-line forces: every atom's energy counts
                        q_child = q_child + (s * e_w[block(k)]).sum()
                        if full_parent:
                            q_parent = q_parent + (s * e_full[batch == g]).sum()
                        else:
                            q_parent = q_parent + (s * e_w[block(w.parent)]).sum()
                f_child = -torch.autograd.grad(
                    q_child, pos_w, create_graph=child_grad, retain_graph=True
                )[0]
                leaf = pos if full_parent else pos_w
                f_parent = -torch.autograd.grad(
                    q_parent, leaf, create_graph=parent_grad, retain_graph=True
                )[0]
                f_child = maybe_detach(f_child, child_grad)
                f_parent = maybe_detach(f_parent, parent_grad).to(f_child.dtype)
                for k in ks:
                    w = windows[k][1]
                    atoms = w.core if name == "force" else w.nodes  # core_force: every atom of W
                    if full_parent:
                        ref = f_parent[torch.as_tensor(atoms, device=device)]
                    else:
                        ref = f_parent[rows(w.parent, atoms)]
                    sigma = self._scale(1, ref.detach().square().sum(-1).mean())
                    diff = f_child[rows(k, atoms)] - ref
                    term = term + window_weight[k] * diff.square().sum() / sigma / n_windows
                    if violation and name == "core_force" and score_now:
                        core = np.isin(atoms, w.core)
                        v = float(
                            diff.detach()[torch.as_tensor(core, device=device)].norm(dim=-1).mean()
                        )
                        scores.setdefault(keys[windows[k][0]], []).append(v)
            out[f"mfc_{name}"] = term.detach()
            loss = loss + self.cfg["weight_force"] * term
        for key, values in scores.items():
            self.violation[key] = float(np.mean(values))
        if violation:
            w = torch.tensor(window_weight, device=device)
            out["mfc_weight_min"], out["mfc_weight_max"] = w.min(), w.max()
        out["mfc"] = lam * loss
        return out


# =============================================================================
# WP4: optimiser (built by the model; GOALModule calls ``build_optimizer``)
# =============================================================================

_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def _newton_schulz(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate orthogonalisation ``U Vᵀ`` of ``g`` (quintic Newton-Schulz, bf16)."""
    a, b, c = _NS_COEFFICIENTS
    x = g.to(torch.bfloat16)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        xx = x @ x.T
        x = a * x + (b * xx + c * xx @ xx) @ x
    return (x.T if transpose else x).to(g.dtype)


def _blocks_of(
    module: nn.Module, name: str
) -> typing.Callable[[torch.Tensor], list[torch.Tensor]] | None:
    """2-D channel-mixing blocks of a matrix-like parameter, or ``None``.

    ``o3.Linear``: one ``(mul_in, mul_out)`` block per irrep path;
    ``_ElementLinear``: one block per element; ``nn.Linear`` with more than
    one output: the weight itself.
    """
    if isinstance(module, o3.Linear) and name == "weight":
        return lambda t, m=module: [w for w in m.weight_views(weight=t)]
    if isinstance(module, _ElementLinear):
        return lambda t: list(t)
    if type(module).__name__ == "O2Linear" and name == "weight":
        return lambda t: list(t)  # one matrix per |m|
    if isinstance(module, nn.Linear) and name == "weight" and module.out_features > 1:
        return lambda t: [t]
    return None


class _SimurghOptimizer(torch.optim.Optimizer):
    """AdamW, Muon or block-RMS Adam per parameter group (``kind``).

    ``adam``: AdamW (AMSGrad optional).  ``muon``: Muon on every 2-D block
    (Nesterov momentum, Newton-Schulz orthogonalised update scaled by
    ``sqrt(max(1, rows / cols))``).  ``block_rms``: Adam whose update is
    rescaled to unit RMS per block.  Weight decay is decoupled throughout.
    """

    def __init__(self, groups: list[dict], splits: dict[int, typing.Callable], **defaults) -> None:
        super().__init__(groups, defaults)
        self._splits = splits

    @torch.no_grad()
    def step(self, closure: typing.Callable | None = None) -> torch.Tensor | None:
        """One update of every parameter group."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if wd:
                    p.mul_(1.0 - lr * wd)
                if group["kind"] == "muon":
                    buf = state.setdefault("momentum", torch.zeros_like(p))
                    buf.mul_(group["momentum"]).add_(p.grad)
                    g = p.grad.add(buf, alpha=group["momentum"])
                    split = self._splits[id(p)]
                    for gv, pv in zip(split(g), split(p)):
                        scale = max(1.0, gv.shape[0] / gv.shape[1]) ** 0.5
                        pv.add_(_newton_schulz(gv, group["ns_steps"]), alpha=-lr * scale)
                    continue
                beta1, beta2 = group["betas"]
                if not state:
                    state["step"] = 0
                    state["m"], state["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["v_max"] = torch.zeros_like(p)
                state["step"] += 1
                state["m"].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                v = state["v"]
                if group["amsgrad"]:
                    torch.maximum(state["v_max"], v, out=state["v_max"])
                    v = state["v_max"]
                bc1, bc2 = 1 - beta1 ** state["step"], 1 - beta2 ** state["step"]
                update = (state["m"] / bc1) / ((v / bc2).sqrt() + group["eps"])
                if group["kind"] == "block_rms":
                    split = self._splits[id(p)]
                    for uv in split(update):
                        uv.div_(uv.square().mean().sqrt().clamp_min(1e-12))
                p.add_(update, alpha=-lr)
        return loss


def build_simurgh_optimizer(model: nn.Module, opt_cfg: typing.Any) -> _SimurghOptimizer:
    """Optimiser of the training recipe for ``model`` (a :class:`SimurghDev`).

    ``name: amsgrad`` (default) | ``muon`` | ``block_rms_adam``.  Decoupled
    weight decay ``weight_decay`` applies to every parameter except the
    embedding, the pair-line coefficients and scalar gates; with the
    regularity prior on, the first radial layers, the linear radial maps
    and the product-basis weights take ``a_basis.prior_weight_decay``.
    ``muon`` / ``block_rms_adam`` act on the channel-mixing matrices
    (``o3.Linear`` paths, element-wise self-connections) and the radial
    MLP matrices; every other parameter stays on AdamW.
    """
    cfg = _plain(opt_cfg)
    name = cfg.get("name", "amsgrad")
    if name not in ("amsgrad", "adamw", "muon", "block_rms_adam"):
        raise ValueError(f"optimizer.name must be amsgrad, muon or block_rms_adam, got {name!r}")
    lr, wd = float(cfg["lr"]), float(cfg.get("weight_decay", 0.0))
    prior_wd = model.config["a_basis"]["prior_weight_decay"]
    prior_on = bool(model.config["a_basis"]["regularity_prior_sigma"])
    no_decay: set[int] = {id(model.atom_line.embedding.weight)}
    if model.pair_line is not None:
        no_decay |= {id(p) for p in model.pair_line.parameters()}
    prior: set[int] = set()
    if prior_on:
        for layer in model.atom_line.layers:
            first = layer.radial if isinstance(layer.radial, nn.Linear) else layer.radial[0]
            prior.add(id(first.weight))
            prior |= {id(p) for p in layer.product.parameters()}
    splits: dict[int, typing.Callable] = {}
    if name in ("muon", "block_rms_adam"):
        for module in model.modules():
            for pname, p in module.named_parameters(recurse=False):
                fn = _blocks_of(module, pname)
                if fn is not None and id(p) not in prior and id(p) not in no_decay:
                    splits[id(p)] = fn
    matrix_kind = {"muon": "muon", "block_rms_adam": "block_rms"}.get(name, "adam")
    buckets: dict[tuple[str, float], list[nn.Parameter]] = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in prior:
            key = ("adam", float(prior_wd))
        elif id(p) in no_decay or p.dim() == 0:
            key = ("adam", 0.0)
        else:
            key = (matrix_kind if id(p) in splits else "adam", wd)
        buckets.setdefault(key, []).append(p)
    betas = (0.98, 0.99) if name == "block_rms_adam" else tuple(cfg.get("betas", (0.9, 0.999)))
    groups = []
    for (kind, decay), params in buckets.items():
        group = {"params": params, "kind": kind, "weight_decay": decay, "lr": lr}
        if kind == "muon":
            group["lr"] = float(cfg.get("muon_lr", 0.02))
        groups.append(group)
    return _SimurghOptimizer(
        groups,
        splits,
        lr=lr,
        betas=betas,
        eps=float(cfg.get("eps", 1e-8)),
        amsgrad=name in ("amsgrad",) or bool(cfg.get("amsgrad", False)),
        momentum=float(cfg.get("muon_momentum", 0.99)),
        ns_steps=int(cfg.get("ns_steps", 5)),
        weight_decay=0.0,
        kind="adam",
    )


# =============================================================================
# WP5: model assembly
# =============================================================================


# =============================================================================
# C7: auxiliary denoising (training only)
# =============================================================================


class _Denoise(nn.Module):
    """Denoising of partially corrupted copies (arXiv:2510.22123; arXiv:2403.09549), rung D1.

    Each training structure gets, with probability ``p``, a copy whose atoms
    are each displaced with probability ``r`` by ``σ ε``, ``ε ~ N(0, I)``; the
    neighbour list is rebuilt on the copy.  The clean forces of the
    corrupted atoms, over the training force scale ``σ_F``, are encoded as
    ``|f| Y_l(f̂)`` (l ≤ 2) and added through a zero-initialised linear map
    to the first layer's A-basis (SIMURGH's initial features are scalars).
    A head (one extra interaction step, then a zero-initialised map to 1o)
    predicts ``ε`` of the corrupted atoms: ``L = (1/N) Σ_i m_i |o_i - ε_i|²``
    per copy.  The copy carries only this loss: no energy or force loss, no
    autograd forces.  The structure keeps its normal losses.
    """

    def __init__(self, cfg: dict, atom_line: nn.Module, num_elements: int) -> None:
        super().__init__()
        aux = cfg["aux"]
        self.sigma, self.p, self.r = aux["denoise_sigma"], aux["denoise_p"], aux["denoise_r"]
        self.cutoff, self.envelope_p = cfg["cutoff"], cfg["envelope_p"]
        self.factor = aux["denoise_scale"] * 0.0625 * aux["force_weight"]
        self.epochs = aux["denoise_epochs"]
        self.irreps_code = o3.Irreps.spherical_harmonics(2)
        irreps_a = atom_line.layers[0].irreps_a
        self.encode = o3.Linear(self.irreps_code, irreps_a)
        last = atom_line.layers[-1]
        pair_dim = cfg["embedding_dim"] if atom_line.pair_code is not None else 0
        self.head = _Interaction(
            1,
            last.irreps_out,
            atom_line.irreps_sh,
            irreps_a,
            irreps_a,
            cfg,
            num_elements,
            pair_dim,
        )
        self.out = o3.Linear(irreps_a, "1x1o")
        with torch.no_grad():
            self.encode.weight.zero_()
            self.out.weight.zero_()
        self.last_copies = 0

    def weight(self, epoch: int, force_scale: float) -> float:
        """``λ_dn = scale · 0.0625 λ_F σ_F²``, decaying linearly to 0 over ``denoise_epochs``."""
        decay = 1.0 if not self.epochs else max(0.0, 1.0 - epoch / float(self.epochs))
        return self.factor * force_scale**2 * decay

    def corrupt(self, pos, batch, num_graphs):
        """``(atoms, positions x̃, mask, ε, structure index, edges of x̃)`` of the copies."""
        device = pos.device
        pick = torch.rand(num_graphs, device=device) < self.p
        atoms = torch.nonzero(pick[batch]).squeeze(-1)
        x = pos[atoms].detach()
        mask = torch.rand(atoms.numel(), device=device) < self.r
        eps = torch.randn_like(x)
        x_t = x + self.sigma * eps * mask.unsqueeze(-1)
        owner = torch.unique(batch[atoms], return_inverse=True)[1]
        d = torch.cdist(x_t, x_t)
        adj = (owner[:, None] == owner[None, :]) & (d < self.cutoff)
        adj.fill_diagonal_(False)
        return atoms, x_t, mask, eps, owner, adj.nonzero().T

    def predict(self, model, elem, x_t, edges, f) -> torch.Tensor:
        """Head output ``o`` (1o, e3nn basis) on positions ``x_t`` with encoded forces ``f``."""
        line = model.atom_line
        size = f.norm(dim=-1, keepdim=True)
        code = size * o3.spherical_harmonics(
            self.irreps_code, f / size.clamp_min(1e-12), normalize=False, normalization="component"
        )
        vec = x_t[edges[1]] - x_t[edges[0]]
        lengths = vec.norm(dim=-1)
        _, h = line(elem, edges, vec, lengths, model.avg_num_neighbors, self.encode(code))
        emb = line.embedding.weight
        sh = o3.spherical_harmonics(line.irreps_sh, vec, normalize=True, normalization="component")
        pair_code = None
        if line.pair_code is not None:
            ei, ej = emb[elem[edges[0]]], emb[elem[edges[1]]]
            pair_code = line.pair_code(torch.cat([ei + ej, ei * ej], dim=-1))
        env = polynomial_envelope(lengths / self.cutoff, self.envelope_p)
        h = self.head(
            h,
            elem,
            emb,
            edges,
            sh * line.sh_factor,
            line.rbf(lengths),
            pair_code,
            env,
            model.avg_num_neighbors,
            lengths,
        )
        return self.out(h)

    def forward(self, model, graph, pos, elem, batch, num_graphs) -> torch.Tensor:
        """Mean denoising loss over the batch's copies (0, still on the graph, without any)."""
        forces = graph.get("forces", None)
        zero = sum(p.sum() for p in self.parameters()) * 0.0
        if forces is None:
            return zero
        atoms, x_t, mask, eps, owner, edges = self.corrupt(pos, batch, num_graphs)
        self.last_copies = int(owner.max()) + 1 if owner.numel() else 0
        if not self.last_copies or edges.numel() == 0:
            return zero
        f = forces[atoms].to(x_t.dtype) * mask.unsqueeze(-1) / float(model.scale)
        o = self.predict(model, elem[atoms], x_t, edges, f)
        target = o3.spherical_harmonics(1, eps, normalize=False, normalization="component")
        err = (o - target / math.sqrt(3.0)).square().sum(-1) * mask
        per_copy = scatter(err, owner, dim=0, dim_size=self.last_copies, reduce="mean")
        return per_copy.mean() + zero


@MODEL_REGISTRY.register("simurgh_dev")
@BACKBONE_REGISTRY.register("simurgh_dev")
class SimurghDev(nn.Module):
    """SIMURGH (development form).  See the module docstring.

    Construct it from the backbone config keys of
    ``src/goal/configs/ml/simurgh_dev_<task>.yaml``; every key has a
    default (:data:`DEFAULTS`) and unknown keys raise ``ValueError``.
    ``elements`` defaults to the keys of ``atomic_energies.values``;
    ``avg_num_neighbors`` and a numeric ``scale`` are required.
    """

    def __init__(self, **config: typing.Any) -> None:
        super().__init__()
        cfg = _merge(DEFAULTS, config)
        self.config = cfg
        _choice(cfg, "core", ("ace", "ece"))
        _choice(cfg, "product_basis", ("symmetric", "ladder", "ladder_reduced"))
        _choice(cfg, "element_weights", ("index", "embedding"))
        _choice(cfg, "node_norm", ("none", "rms"))
        _choice(cfg, "kernels", ("auto", "cuequivariance", "e3nn"))
        _choice(cfg, "dtype", ("float32", "float64"))
        _choice(cfg["attention"], "mode", ("gate", "softmax_null", "radial", "none"), "attention.")
        _choice(cfg["pair_line"], "mode", ("phi", "ece", "phi+ece"), "pair_line.")
        if cfg["kernels"] == "cuequivariance":  # auto resolves to e3nn
            raise NotImplementedError(
                "kernels: cuequivariance is not implemented: SIMURGH has only the e3nn path "
                "(cuequivariance-torch is not a GOAL dependency yet)"
            )
        _choice(cfg["readout"], "pgs_edge_alpha", ("scalar", "per_channel"), "readout.")
        _choice(cfg["attention"], "pan_gate_descriptors", ("full", "generic"), "attention.")
        per_layer = cfg["radial"]["per_layer"]
        if per_layer is not None and any(kind not in ("linear", "mlp") for kind in per_layer):
            raise ValueError(f"radial.per_layer entries must be linear or mlp, got {per_layer}")
        if cfg["diagnostics"]["calm"] and (
            cfg["core"] != "ace" or cfg["product_basis"] != "symmetric"
        ):
            raise ValueError("diagnostics.calm reads the symmetric product basis of core: ace")
        if cfg["core"] == "ece" and cfg["attention"]["pan_gate"]:
            raise ValueError("attention.pan_gate gates the ACE messages: needs core: ace")
        if cfg["core"] == "ece" and cfg["readout"]["pgs_edge"]:
            raise ValueError("readout.pgs_edge acts on the ACE radial weights: needs core: ace")
        for flag, on in (
            ("pair_line.bond_order_body: 4", cfg["pair_line"]["bond_order_body"] == 4),
        ):
            if on:
                _not_yet(flag)

        ae = cfg["atomic_energies"]
        _choice(ae, "mode", ("provided", "dataset"), "atomic_energies.")
        if not ae["values"]:
            raise ValueError(
                "SIMURGH needs atomic_energies.values {Z: E0}: given (mode: provided), or "
                "fitted on the training set by goal-train (mode: dataset)"
            )
        e0 = {int(z): float(e) for z, e in ae["values"].items()}
        elements = sorted(int(z) for z in (cfg["elements"] or e0))
        missing = [z for z in elements if z not in e0]
        if missing:
            raise ValueError(f"atomic_energies.values has no E0 for elements {missing}")
        if cfg["avg_num_neighbors"] is None:
            raise ValueError(
                "avg_num_neighbors is required (goal-train sets it from the training set)"
            )
        if not isinstance(cfg["scale"], (int, float)):
            raise ValueError(
                "scale must be a number (goal-train sets it to the training force RMS)"
            )
        prior = cfg["a_basis"]
        if prior["regularity_prior_sigma"] and not prior["prior_weight_decay"]:
            raise ValueError(
                "a_basis.regularity_prior_sigma needs a non-zero a_basis.prior_weight_decay"
            )
        self.elements = elements
        self.avg_num_neighbors = float(cfg["avg_num_neighbors"])
        index = torch.full((119,), -1, dtype=torch.long)
        index[torch.tensor(elements)] = torch.arange(len(elements))
        self.register_buffer("z_to_index", index)
        # E0 stays float64 whatever the parameter dtype: totals of ~1e4-1e5 eV
        # would lose meV in float32.  Not a buffer, so .to(dtype) leaves it alone.
        self._e0 = torch.tensor([e0[z] for z in elements], dtype=torch.float64)
        self.register_buffer("scale", torch.tensor(float(cfg["scale"])))

        if cfg["core"] == "ece":
            if cfg["pair_line"]["mode"] != "phi":
                raise ValueError(
                    "with core: ece the pair line stays phi (the core is edge-centred)"
                )
            if cfg["electrostatics"]["dipoles"]:
                _not_yet("electrostatics.dipoles with core: ece")
            if cfg["a_basis"]["regularity_prior_sigma"]:
                _not_yet("a_basis.regularity_prior_sigma with core: ece")
            self.atom_line = _ECECore(cfg, len(elements))
        else:
            self.atom_line = _AtomLine(cfg, len(elements))
        self.ece_line = None
        if cfg["core"] == "ace" and "ece" in cfg["pair_line"]["mode"]:
            self.ece_line = _ECELine(cfg, self.atom_line.layers[-1].irreps_out)
            self.ece_codes = nn.Embedding(len(elements), cfg["ece"]["film"]["element_dim"])
        pl = cfg["pair_line"]
        if pl["mode"] == "ece":
            pl = {**pl, "enabled": False}  # ECE bond line only
        self.pair_line = (
            _PairLine(pl, elements, cfg["embedding_dim"], cfg["cutoff"]) if pl["enabled"] else None
        )
        self.dispersion = _D3Term(cfg["dispersion"]) if cfg["dispersion"]["enabled"] else None
        es = cfg["electrostatics"]
        if es["sigma"] != "learned" and not isinstance(es["sigma"], (int, float)):
            raise ValueError("electrostatics.sigma must be 'learned' or a width in Å")
        self.electrostatics = (
            _Electrostatics(es, self.atom_line.channels, len(elements), cfg)
            if es["enabled"]
            else None
        )
        _choice(pl, "bond_order_cutoff_fn", ("tanh3", "envelope"), "pair_line.")
        _choice(pl, "bond_order_eta_map", ("power", "smooth"), "pair_line.")
        if pl["bond_order_body"] not in (3, 4):
            raise ValueError("pair_line.bond_order_body must be 3 or 4")
        aux = cfg["aux"]
        _choice(aux, "denoise_rung", ("D1", "D2", "D1-R"), "aux.")
        _choice(aux, "hessian_mode", ("autograd", "finite_difference"), "aux.")
        if aux["denoise"] and aux["denoise_rung"] != "D1":
            _not_yet(f"aux.denoise_rung: {aux['denoise_rung']}")
        if aux["denoise"] and cfg["core"] != "ace":
            raise ValueError("aux.denoise encodes forces into the ACE A-basis: needs core: ace")
        self.denoise = _Denoise(cfg, self.atom_line, len(elements)) if aux["denoise"] else None
        self.bond_order = (
            _BondOrder(
                cfg, len(elements), min(pl["pair_cutoff"] or cfg["cutoff"], 5.5), cfg["envelope_p"]
            )
            if pl["bond_order"]
            else None
        )
        if pl["pair_first_epochs"] and not pl["enabled"]:
            raise ValueError("pair_line.pair_first_epochs needs pair_line.enabled")
        self.mfc = (
            _MFC(
                cfg["mfc"],
                cfg["cutoff"],
                self.atom_line.channels,
                elements.index(1) if 1 in elements else None,
            )
            if cfg["mfc"]["enabled"]
            else None
        )
        self.epoch = 0
        self.to(getattr(torch, cfg["dtype"]))

    def line_energies(
        self,
        elem: torch.Tensor,
        edge_index: torch.Tensor,
        vectors: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Line energies ``e_i`` (unscaled), final atom features and bond energies (or None).

        The core's atomic energies plus, with an ECE bond line, its atomic split
        ``e_i = 1/2 sum_j (eps_ij + eps_ji)``.  The full structure and the MFC windows
        both use this, so the consistency loss always compares the same quantity.
        """
        e_atom, h = self.atom_line(elem, edge_index, vectors, lengths, self.avg_num_neighbors)
        bond = getattr(self.atom_line, "last_bond_energies", None)
        if self.ece_line is not None:
            line = self.ece_line(
                h,
                self.ece_codes.weight[elem],
                edge_index,
                vectors,
                lengths,
                self.avg_num_neighbors,
            )
            e_atom = e_atom + line["atomic_energies"]
            bond = line["bond_energies"]
        return e_atom, h, bond

    def set_epoch(self, epoch: int) -> None:
        """Current training epoch (GOALModule calls this every step)."""
        self.epoch = int(epoch)

    def extra_energy(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        elem: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor | None:
        """Additional energy per molecule (eV); ``None`` here, set by ablation models."""
        return None

    def hessian_rows(
        self,
        graph: AtomicGraph,
        pos: torch.Tensor,
        forces: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Student Hessian rows ``H_θ^(j)`` ``(N, s, 3)`` at the sampled ``hessian_row_number``.

        ``autograd``: per slot one vector-Jacobian product of the forces with a
        vector holding one 1 per structure (block-diagonal: all structures at
        once), ``H^(j) = -∂F_j/∂x``; needs a double backward of the force graph.
        ``finite_difference``: ``-[F(x + h e_j) - F(x)]/h`` with h = 1e-3 Å
        (float64) or 5e-3 Å (float32), all structures shifted at once.
        """
        index = graph.hessian_row_number.reshape(num_graphs, -1)
        start = torch.cumsum(torch.bincount(batch, minlength=num_graphs), 0)
        start = start - torch.bincount(batch, minlength=num_graphs)
        atom = start[:, None] + torch.div(index, 3, rounding_mode="floor")
        axis = index % 3
        rows = []
        for k in range(index.shape[1]):
            v = torch.zeros_like(forces)
            v[atom[:, k], axis[:, k]] = 1.0
            if self.config["aux"]["hessian_mode"] == "autograd":
                (g,) = torch.autograd.grad(forces, pos, grad_outputs=v, create_graph=True)
                rows.append(-g.to(forces.dtype))
            else:
                h = 1e-3 if forces.dtype == torch.float64 else 5e-3
                shifted = graph.clone()
                shifted.pos = pos + h * v.to(pos.dtype)
                shifted.hessian_row = None
                inner = self.config["aux"]
                self.config["aux"] = {**inner, "hessian_distill": False}
                try:
                    plus = self.forward(shifted)["forces"]
                finally:
                    self.config["aux"] = inner
                rows.append(-(plus - forces) / h)
        return torch.stack(rows, dim=1)

    def hessian_loss(self, graph, pos, forces, batch, num_graphs) -> torch.Tensor:
        """``(1/s) Σ_j ‖H_T^(j) - H_θ^(j)‖² / N`` per structure, averaged (arXiv:2501.09009, Eq. 3)."""
        rows = self.hessian_rows(graph, pos, forces, batch, num_graphs)
        teacher = graph.hessian_row.to(rows.dtype)
        err = (teacher - rows).square().sum(dim=(1, 2)) / rows.shape[1]  # (N,)
        count = torch.bincount(batch, minlength=num_graphs).to(err.dtype)
        per_structure = scatter(err, batch, dim=0, dim_size=num_graphs, reduce="sum") / count
        return per_structure.mean()

    def build_optimizer(self, opt_cfg: typing.Any) -> torch.optim.Optimizer:
        """The recipe's optimiser; see :func:`build_simurgh_optimizer`."""
        return build_simurgh_optimizer(self, opt_cfg)

    @classmethod
    def atom_statistics(
        cls, config: typing.Mapping[str, typing.Any]
    ) -> dict[str, typing.Callable[[AtomicGraph], torch.Tensor]]:
        """Per-atom descriptors whose training-split mean and s.d. the model needs.

        ``goal-train`` evaluates each function on every training graph and
        passes ``{name: {mean, std}}`` back as ``atom_statistics``.  Empty
        unless the PAN gate uses its descriptors.
        """
        # an arm's own sections (long_range, ...) are checked by its constructor
        cfg = _merge(DEFAULTS, {k: v for k, v in dict(config).items() if k in DEFAULTS})
        att = cfg["attention"]
        if not att["pan_gate"] or att["pan_gate_descriptors"] != "full":
            return {}
        cutoff, p = float(cfg["cutoff"]), int(cfg["envelope_p"])

        def column(k: int) -> typing.Callable[[AtomicGraph], torch.Tensor]:
            def descriptor(graph: AtomicGraph) -> torch.Tensor:
                pos = graph.pos.detach().double()
                vectors, lengths = differentiable_edges(graph, pos)
                inside = lengths < cutoff
                receiver = graph.edge_index[1][inside]
                n = pos.shape[0]
                return pan_descriptors(vectors[inside], lengths[inside], receiver, n, cutoff, p)[
                    :, k
                ]

            return descriptor

        return {name: column(k) for k, name in enumerate(PAN_STATISTICS)}

    @property
    def output_keys(self) -> list[str]:
        return [
            "energy",
            "forces",
            "num_atoms",
            "energy_atom_line",
            "energy_pair_line",
            "energy_d3",
            "energy_elec",
            "atomic_energies",
        ]

    def _e0_on(self, device: torch.device) -> torch.Tensor:
        """The float64 E0 table on ``device`` (moved once, then cached)."""
        if self._e0.device != device:
            self._e0 = self._e0.to(device)
        return self._e0

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Energy, forces and the decomposition outputs of ``graph``."""
        dtype = self.scale.dtype
        pos = graph.pos
        caller_grad = pos.requires_grad  # the caller will differentiate the outputs further
        if not caller_grad:
            pos = pos.detach().requires_grad_(True)
        pos_d = pos.to(dtype)
        vectors, lengths = differentiable_edges(graph, pos_d)
        n = pos.shape[0]
        batch = graph.get("batch", None)
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=pos.device)
        num_graphs = int(graph.num_graphs) if hasattr(graph, "num_graphs") else 1
        elem = self.z_to_index[graph.atomic_numbers]
        if bool((elem < 0).any()):
            unknown = sorted(set(graph.atomic_numbers[elem < 0].tolist()))
            raise ValueError(f"SIMURGH was built for elements {self.elements}, got {unknown}")
        edge_index = graph.edge_index
        emb = self.atom_line.embedding.weight

        e_atom, h_last, bond = self.line_energies(elem, edge_index, vectors, lengths)
        pgs = getattr(self.atom_line, "pgs", None)
        in_support = None if pgs is None else pgs.last_in_support  # before MFC windows reuse it
        calm = None
        if self.config["diagnostics"]["calm"]:  # output only: the grade's feature blocks
            calm = [layer.last_basis for layer in self.atom_line.layers]
        if self.training and self.epoch < self.config["pair_line"]["pair_first_epochs"]:
            e_atom = 0.0 * e_atom  # pair-first stage: the atom line gets zero gradients
        e_pair = torch.zeros_like(e_atom)
        wall = torch.zeros_like(e_atom)
        if self.pair_line is not None:
            a, b = elem[edge_index[0]], elem[edge_index[1]]
            half = 0.5 * self.pair_line.learned(lengths, a, b, emb)
            e_pair = scatter(half, edge_index[1], dim=0, dim_size=n, reduce="sum")
            half_wall = 0.5 * self.pair_line.wall_energy(lengths, a, b)
            wall = scatter(half_wall, edge_index[1], dim=0, dim_size=n, reduce="sum")
        if self.bond_order is not None:  # eV, half of each pair term on each atom
            wall = wall + self.bond_order(vectors, lengths, edge_index, elem)
        line_energy = e_atom + e_pair
        held = None
        if self.mfc is not None and self.mfc.residual is not None:
            # mfc.parameters: separate (ablation only: the pooled mean breaks far-copy
            # extensivity for unlike fragments); MFC compares e_atom only
            delta, held = self.mfc.residual_energy(h_last, batch, num_graphs)
            line_energy = line_energy + delta
        atomic = self._e0_on(pos.device)[elem] + (self.scale * line_energy + wall).double()

        def per_graph(x: torch.Tensor) -> torch.Tensor:
            return scatter(x, batch, dim=0, dim_size=num_graphs, reduce="sum")

        energy = per_graph(atomic)
        e_d3 = (
            self.dispersion(graph, pos_d, batch, num_graphs)
            if self.dispersion is not None
            else None
        )
        if e_d3 is not None:
            energy = energy + e_d3
        elec = None
        if self.electrostatics is not None:
            total = graph.get("total_charge", None)
            elec = self.electrostatics(h_last, pos_d, elem, batch, num_graphs, total)
            energy = energy + elec["energy"]
        extra = self.extra_energy(h_last, pos_d, elem, batch, num_graphs)
        if extra is not None:
            energy = energy + extra
        forces = -torch.autograd.grad(
            energy.sum(),
            pos,
            create_graph=self.training,
            retain_graph=self.training or caller_grad,
        )[0]
        zeros = energy.new_zeros(num_graphs)
        forces = forces.to(dtype)
        out: dict[str, torch.Tensor] = {
            "energy": energy,
            "forces": forces,
            "num_atoms": torch.bincount(batch, minlength=num_graphs),
            "energy_atom_line": per_graph(self.scale * e_atom),
            "energy_pair_line": per_graph(self.scale * e_pair + wall),
            "energy_d3": zeros if e_d3 is None else e_d3,
            "energy_elec": zeros if elec is None else elec["energy"],
            "atomic_energies": atomic,
        }
        if elec is not None:
            out["charges"], out["dipole"] = elec["charges"], elec["dipole"]
        if bond is not None:
            out["edge_energies"] = self.scale * bond
        if calm is not None:
            out["calm_basis"] = torch.cat(calm, dim=-1)
            out["calm_layer_norms"] = torch.stack([b.norm(dim=-1) for b in calm], dim=-1)
        fragments = graph.get("fragment_index", None)
        if fragments is not None:
            out.update(
                self._fragment_decomposition(fragments, atomic, edge_index, lengths, elem, emb)
            )
        if self.training:
            aux = self._regularisers(elem, edge_index, e_atom, emb)
            if held is not None:
                aux = aux + held  # the other levels' residual sections stay on the graph
            terms: dict[str, torch.Tensor] = {}
            if in_support is not None:
                terms["pgs_in_support"] = in_support  # atoms inside some kernel's support
            if self.mfc is not None:
                q_full = None if elec is None else elec["charges"]
                terms.update(self.mfc(self, graph, pos, pos_d, elem, batch, e_atom, q_full))
                aux = aux + terms.pop("mfc")
            if self.denoise is not None:
                dn = self.denoise(self, graph, pos_d, elem, batch, num_graphs)
                aux = aux + self.denoise.weight(self.epoch, float(self.scale)) * dn
                terms["denoise"] = dn.detach()
            cfg_aux = self.config["aux"]
            if cfg_aux["hessian_distill"] and graph.get("hessian_row", None) is not None:
                kd = self.hessian_loss(graph, pos, forces, batch, num_graphs)
                weight = cfg_aux["hessian_weight"]
                weight = 4.0 * cfg_aux["force_weight"] if weight is None else weight
                aux = aux + weight * kd
                terms["hessian"] = kd.detach()
            if terms:
                out["aux_terms"] = terms
            out["aux_loss"] = aux
        return out

    def _fragment_decomposition(self, fragments, atomic, edge_index, lengths, elem, emb):
        """Exact, parameter-free split: fragment energies and inter-fragment pair energies."""
        k = int(fragments.max()) + 1 if fragments.numel() else 0
        out = {"fragment_energies": scatter(atomic, fragments, dim=0, dim_size=k, reduce="sum")}
        pair = atomic.new_zeros(k, k)
        if self.pair_line is not None:
            a, b = elem[edge_index[0]], elem[edge_index[1]]
            phi = self.scale * self.pair_line.learned(
                lengths, a, b, emb
            ) + self.pair_line.wall_energy(lengths, a, b)
            fi, fj = fragments[edge_index[0]], fragments[edge_index[1]]
            cross = fi != fj
            pair = pair.index_put(
                (fi[cross], fj[cross]), phi[cross].to(pair.dtype), accumulate=True
            )
        out["fragment_pair_energies"] = pair
        return out

    def _regularisers(self, elem, edge_index, e_atom, emb) -> torch.Tensor:
        """Training-only penalties returned through ``aux_loss``."""
        aux = e_atom.new_zeros(())
        pl = self.config["pair_line"]
        if self.pair_line is not None and pl["curvature_penalty"]:
            ids = self.pair_line.pair_id[elem[edge_index[0]], elem[edge_index[1]]]
            present = torch.unique(ids)
            n = len(self.elements)
            pairs = torch.tensor(
                list(itertools.combinations_with_replacement(range(n), 2)), device=ids.device
            )[present]
            aux = aux + pl["curvature_penalty"] * self.pair_line.curvature_penalty(pairs, emb)
        if pl["many_body_penalty"]:
            aux = aux + pl["many_body_penalty"] * e_atom.square().sum()
        return aux
