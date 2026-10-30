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
    "attention": {"mode": "gate", "pan_gate": False},
    "readout": {"pgs": True, "pgs_edge": False, "pgs_beta": 4.0, "pgs_directions": 16},
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
    "electrostatics": {"enabled": False, "dipoles": False, "sigma": "learned"},
    "dispersion": {
        "enabled": False,
        "backend": "torch",
        "functional": "pbe",
        "damping": "bj",
        "atm": False,
    },
    "diagnostics": {"calm": False},
    "atomic_energies": {"mode": "provided", "values": None},
    "scale": None,
    "mfc": {
        "enabled": False,
        "objective": ["energy", "core_force"],
        "window_type": "halfspace",
        "levels": 2,
        "radii": [8.0],
        "buffer": 5.0,
        "windows_per_structure": 1,
        "every_n_steps": 1,
        "weight": 1.0,
        "weight_energy": 1.0,
        "weight_force": 1.0,
        "start_epoch": 20,
        "warmup_epochs": 10,
        "stop_grad": "parent",
        "parameters": "shared",
        "window_weighting": "uniform",
        "tolerance": "none",
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
        self.query = o3.Linear(irreps_h, irreps_msg)
        with torch.no_grad():
            self.query.weight.zero_()
        self.bias = _MLP([num_rbf, 16, 1], zero_last=True)
        self.temperature = nn.Parameter(torch.zeros(()))
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
        tau = 0.25 + 3.75 * torch.sigmoid(self.temperature)
        score = env * self.bias(rbf).squeeze(-1)
        score = score + tau / math.sqrt(self.dim) * (self.query(h)[receiver] * messages).sum(-1)
        if self.mode == "gate":
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
        if self.linear_radial:
            self.radial = nn.Linear(in_dim, self.tp.weight_numel, bias=False)
        else:
            self.radial = _MLP([in_dim, *cfg["radial_mlp"], self.tp.weight_numel])
        mode = cfg["attention"]["mode"]
        self.attention = (
            _AttentionGate(mode, irreps_h, irreps_mid_sorted, cfg["num_rbf"])
            if mode != "none"
            else None
        )
        self.linear_a = o3.Linear(irreps_mid_sorted, irreps_a)
        self.product = _SymmetricProductBasis(
            irreps_a,
            irreps_out,
            cfg["correlation"],
            num_elements,
            cfg["element_weights"],
            cfg["embedding_dim"],
        )
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
    ) -> torch.Tensor:
        """One layer: messages, A-basis, product basis and update."""
        sender, receiver = edge_index
        radial_in = [rbf]
        if not self.linear_radial and pair_code is not None:
            radial_in.append(pair_code)
        if self.env_proj is not None:
            scalars = h[:, : self.irreps_h.count("0e")]
            radial_in += [self.env_proj(scalars)[sender], self.env_proj(scalars)[receiver]]
        weights = self.radial(torch.cat(radial_in, dim=-1)) * env.unsqueeze(-1)
        messages = self.tp(self.linear_up(h)[sender], sh, weights)
        n = h.shape[0]
        if self.attention is None:
            a = scatter(messages, receiver, dim=0, dim_size=n, reduce="sum") / avg_num_neighbors
        else:
            a = self.attention(messages, h, receiver, rbf, env, n, avg_num_neighbors)
        a = _to_channels(self.linear_a(a), self.irreps_a)
        b = _from_channels(self.product(a, elem, emb), self.irreps_out)
        out = self.linear_b(b)
        if self.self_connection is not None:
            out = out + self.self_connection(h, elem)
        return out


def _es_kernel(z: torch.Tensor, beta: float) -> torch.Tensor:
    """Compact exponential-of-semicircle bump with a (1 - z²)² taper.

    ``(1-z²)² [exp(β(√(1-z²) - 1)) - e^{-β}]`` for ``|z| < 1``, else 0.  The
    taper makes it vanish with two continuous derivatives at ``|z| = 1``;
    the double ``where`` keeps autograd finite at the edge.
    """
    s = 1.0 - z * z
    inside = s > 0
    safe = torch.where(inside, s, torch.ones_like(s))
    value = safe.square() * (torch.exp(beta * (torch.sqrt(safe) - 1.0)) - math.exp(-beta))
    return torch.where(inside, value, torch.zeros_like(value))


class _PGSReadout(nn.Module):
    """PGS spectral readout branch (WP8 A): ``Σ_m α_m [K(u_mᵀx / σ_m) - K(0)]``.

    ``u_m`` start orthonormal; ``σ_m`` (a buffer, set on the first training
    batch) gives ``u_mᵀx`` unit RMS; ``α`` is zero-initialised, so the
    branch starts as the identity.  Subtracting ``K(0)`` keeps the readout
    bias-free (an isolated atom, ``x = 0``, contributes nothing).
    """

    def __init__(self, channels: int, directions: int, beta: float) -> None:
        super().__init__()
        self.beta = beta
        u = torch.empty(directions, channels)
        nn.init.orthogonal_(u)
        self.directions = nn.Parameter(u)
        self.alpha = nn.Parameter(torch.zeros(directions))
        self.register_buffer("width", torch.ones(directions))
        self.register_buffer("initialised", torch.zeros((), dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-atom readout contribution ``(N,)`` from invariant features ``x``."""
        z = x @ self.directions.T
        if self.training and not bool(self.initialised):
            with torch.no_grad():
                self.width.copy_(z.square().mean(dim=0).sqrt().clamp_min(1e-6))
                self.initialised.fill_(True)
        k0 = _es_kernel(z.new_zeros(()), self.beta)
        return ((_es_kernel(z / self.width, self.beta) - k0) * self.alpha).sum(-1)


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
            irreps_out = scalars if last else hidden
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
            readouts.append(_MLP([c, 16, 1]) if last else nn.Linear(c, 1, bias=False))
            irreps_h = irreps_out
        self.layers, self.readouts = nn.ModuleList(layers), nn.ModuleList(readouts)
        rd = cfg["readout"]
        self.pgs = _PGSReadout(c, rd["pgs_directions"], rd["pgs_beta"]) if rd["pgs"] else None
        self.channels = c

    def forward(
        self,
        elem: torch.Tensor,
        edge_index: torch.Tensor,
        vectors: torch.Tensor,
        lengths: torch.Tensor,
        avg_num_neighbors: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Atomic energies ``e_i`` (unscaled) and the final invariant features."""
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
        energy = h.new_zeros(h.shape[0])
        for t, (layer, readout) in enumerate(zip(self.layers, self.readouts)):
            h = layer(h, elem, emb, edge_index, sh, rbf, pair_code, env, avg_num_neighbors)
            scalars = h[:, : self.channels]
            energy = energy + readout(scalars).squeeze(-1)
            if t == len(self.layers) - 1 and self.pgs is not None:
                energy = energy + self.pgs(scalars)
        return energy, h


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
# WP5: model assembly
# =============================================================================


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
        _choice(cfg["attention"], "mode", ("gate", "softmax_null", "none"), "attention.")
        _choice(cfg["pair_line"], "mode", ("phi", "ece", "phi+ece"), "pair_line.")
        if cfg["kernels"] == "cuequivariance":
            raise ImportError(
                "kernels: cuequivariance requested but cuequivariance-torch is not installed"
            )
        for flag, on in (
            ("core: ece", cfg["core"] == "ece"),
            ("product_basis: ladder", cfg["product_basis"] != "symmetric"),
            ("node_norm: rms", cfg["node_norm"] == "rms"),
            ("radial.low_rank", cfg["radial"]["low_rank"] is not None),
            ("attention.pan_gate", cfg["attention"]["pan_gate"]),
            ("readout.pgs_edge", cfg["readout"]["pgs_edge"]),
            ("pair_line.mode: ece", cfg["pair_line"]["mode"] != "phi"),
            ("pair_line.bond_order", cfg["pair_line"]["bond_order"]),
            ("electrostatics", cfg["electrostatics"]["enabled"]),
            ("diagnostics.calm", cfg["diagnostics"]["calm"]),
            ("mfc", cfg["mfc"]["enabled"]),
        ):
            if on:
                _not_yet(flag)

        ae = cfg["atomic_energies"]
        if ae["mode"] != "provided" or not ae["values"]:
            raise ValueError("SIMURGH needs atomic_energies: {mode: provided, values: {Z: E0}}")
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

        self.atom_line = _AtomLine(cfg, len(elements))
        pl = cfg["pair_line"]
        self.pair_line = (
            _PairLine(pl, elements, cfg["embedding_dim"], cfg["cutoff"]) if pl["enabled"] else None
        )
        self.dispersion = _D3Term(cfg["dispersion"]) if cfg["dispersion"]["enabled"] else None
        self.to(getattr(torch, cfg["dtype"]))

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
        if not pos.requires_grad:
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

        e_atom, _ = self.atom_line(elem, edge_index, vectors, lengths, self.avg_num_neighbors)
        e_pair = torch.zeros_like(e_atom)
        wall = torch.zeros_like(e_atom)
        if self.pair_line is not None:
            a, b = elem[edge_index[0]], elem[edge_index[1]]
            half = 0.5 * self.pair_line.learned(lengths, a, b, emb)
            e_pair = scatter(half, edge_index[1], dim=0, dim_size=n, reduce="sum")
            half_wall = 0.5 * self.pair_line.wall_energy(lengths, a, b)
            wall = scatter(half_wall, edge_index[1], dim=0, dim_size=n, reduce="sum")
        atomic = self._e0_on(pos.device)[elem] + (self.scale * (e_atom + e_pair) + wall).double()

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
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training, allow_unused=False
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
            "energy_elec": zeros,
            "atomic_energies": atomic,
        }
        fragments = graph.get("fragment_index", None)
        if fragments is not None:
            out.update(
                self._fragment_decomposition(fragments, atomic, edge_index, lengths, elem, emb)
            )
        if self.training:
            out["aux_loss"] = self._regularisers(elem, edge_index, e_atom, emb)
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
            pair = pair.index_put((fi[cross], fj[cross]), phi[cross], accumulate=True)
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
