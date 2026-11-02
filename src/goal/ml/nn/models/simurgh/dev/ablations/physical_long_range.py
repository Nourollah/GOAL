"""Physical long-range terms: learned dispersion and damped electrostatics.

``E = E_base + a_disp E_disp + a_elec E_elec`` with amplitudes ``a`` that
start at zero, so a fresh model equals :class:`SimurghDev` exactly.

Dispersion (DOI 10.1021/jacs.5c09558, Eqs. 4-5; oscillator relations of
DOI 10.1021/acs.jctc.3c00797)::

    E_disp = -Σ_{i<j} Σ_{n=3..5} C_{2n,ij} / (r^{2n} + R_d,ij^{2n})   (atomic units)

* volume ratios ``h_i = softplus(b_Z + MLP(s_i))`` on the final 0e features,
  ``b_Z = softplus⁻¹(0.8)`` and the MLP's last layer zero, so ``h = 0.8`` at
  initialisation; ``α_i = h_i α0``, ``C6_i = h_i² C6_0``;
* ``C6_ij = 2 α_i α_j C6_i C6_j / (α_j² C6_i + α_i² C6_j)`` (Eq. 19 of the
  oscillator paper) and ``α_ij = (α_i + α_j) / 2``;
* ``C8 = 5 C6 / m`` and ``C10 = 245 C6 / (8 m²)`` with the oscillator
  product ``m = 1/(2 s²)``, ``s`` a cubic in ``R = α_fsc^(-4/21) α_ij^(1/7)``
  (closure from the authors' released code; it reproduces the damped
  products of the oscillator paper's SI Table S3);
* ``R_d = γ · 2 · 2.54 · α_ij^(1/7)`` bohr with ``γ = 1.2`` fixed (the
  released code's literal 2.54, with which ``γ`` was tuned);
  ``exact_prefactor`` uses ``2 α_fsc^(-4/21)`` = 2 × 2.552796.

Electrostatics (Eq. 6): ``E_elec = Σ_{i<j} k_e q_i q_j erf(r/σ)/r``, σ = 4 Å
fixed, with charges from a local head on the final 0e features under
``charge_constraint: uniform | pair_transfer``.
``electrostatics_kernel: gaussian_pair`` uses the WP7 module instead.

Pairs: every intramolecular pair, bonded ones included, with no cutoff.
A finite ``cutoff`` R_c switches dispersion off over (R_c - 2 Å, R_c) with a
C∞ multiplicative step, and shifts electrostatics with the curvature-shifted
``u_FS2`` blended over (0.45 R_c, R_c) (C2 at R_c; the source is C1).

On GMD (labels with D3(BJ)) compare this ablation against the D3 term and
never stack the two; on MD22 (PBE+MBD) the learned dispersion replaces D3.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import scatter

from goal.ml.nn.models.simurgh.dev.monolithic import (
    _MLP,
    SimurghDev,
    _ChargeHead,
    _intramolecular_pairs,
)
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

#: CODATA 2014, as used by the source (C-2.8).
BOHR = 0.5291772105638411  # Å
HARTREE = 27.211386024367243  # eV
FINE_STRUCTURE = 7.297352566206496e-3
COULOMB = HARTREE * BOHR  # k_e = 14.399645351950548 eV Å / e²
#: α_fsc^(-4/21): the pair van der Waals radius is this times α^(1/7) bohr.
VDW_PREFACTOR = FINE_STRUCTURE ** (-4.0 / 21.0)
#: Cubic closure s(R) = b3 R³ + b2 R² + b1 R + b0 of the oscillator width.
CLOSURE = (-0.00433008, 0.24428889, 0.04125273, -0.00078893)  # b0, b1, b2, b3

#: Free-atom polarisability α0 (bohr³) and C6 (E_h bohr⁶).  Provisional: from the
#: source authors' released code (equal to an independent compilation of the
#: same reference set); the primary table was not read.
FREE_ATOM: dict[int, tuple[float, float]] = {
    1: (4.5, 6.5),
    6: (12.0, 46.6),
    7: (7.4, 24.2),
    8: (5.4, 15.6),
}
DEFAULTS: dict[str, object] = {
    "electrostatics_kernel": "erf_fixed",  # erf_fixed | gaussian_pair
    "charge_constraint": "uniform",  # uniform | pair_transfer
    "exact_prefactor": False,
    "disp_damping_gamma": 1.2,
    "sigma": 4.0,  # Å
    "cutoff": None,  # Å; None = every intramolecular pair
}


def oscillator_product(alpha_ij: torch.Tensor) -> torch.Tensor:
    """``osc_mw`` = m = μω of the pair oscillator (bohr⁻²) from the mixed ``α_ij`` (bohr³)."""
    r = VDW_PREFACTOR * alpha_ij ** (1.0 / 7.0)
    b0, b1, b2, b3 = CLOSURE
    s = ((b3 * r + b2) * r + b1) * r + b0
    return 1.0 / (2.0 * s * s)


def dispersion_pair_terms(
    alpha_i: torch.Tensor,
    alpha_j: torch.Tensor,
    c6_i: torch.Tensor,
    c6_j: torch.Tensor,
    gamma: float = 1.2,
    exact_prefactor: bool = False,
) -> dict[str, torch.Tensor]:
    """``alpha_ij``, ``C6``, ``osc_mw``, ``C8``, ``C10`` and ``R_d`` (bohr) of each pair (a.u.)."""
    c6 = 2.0 * alpha_i * alpha_j * c6_i * c6_j / (alpha_j**2 * c6_i + alpha_i**2 * c6_j)
    alpha_ij = 0.5 * (alpha_i + alpha_j)
    osc_mw = oscillator_product(alpha_ij)
    prefactor = VDW_PREFACTOR if exact_prefactor else 2.54
    return {
        "alpha_ij": alpha_ij,
        "C6": c6,
        "osc_mw": osc_mw,
        "C8": 5.0 * c6 / osc_mw,
        "C10": 245.0 * c6 / (8.0 * osc_mw**2),
        "Rd": gamma * 2.0 * prefactor * alpha_ij ** (1.0 / 7.0),
    }


def dispersion_pair_energy(r: torch.Tensor, terms: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pair dispersion energy (eV) at distance ``r`` (Å)."""
    rb, rd = r / BOHR, terms["Rd"]
    total = sum(terms[f"C{2 * n}"] / (rb ** (2 * n) + rd ** (2 * n)) for n in (3, 4, 5))
    return -HARTREE * total


def smooth_step(x: torch.Tensor, x_on: float, x_off: float) -> torch.Tensor:
    """C∞ switch: 1 below ``x_on``, 0 above ``x_off``; ``S(1-c)/(S(1-c)+S(c))``, ``S = e^(-1/t)``."""
    c = ((x - x_on) / (x_off - x_on)).clamp(0.0, 1.0)

    def s(t: torch.Tensor) -> torch.Tensor:
        positive = t > 0
        safe = torch.where(positive, t, torch.ones_like(t))
        return torch.where(positive, torch.exp(-1.0 / safe), torch.zeros_like(t))

    up, down = s(1.0 - c), s(c)
    return up / (up + down)


def erf_coulomb(r: torch.Tensor, sigma: float) -> tuple[torch.Tensor, ...]:
    """``u = erf(r/σ)/r`` and its first two derivatives (Å⁻¹, Å⁻², Å⁻³)."""
    u = torch.erf(r / sigma) / r
    g = torch.exp(-((r / sigma) ** 2)) / (math.sqrt(math.pi) * sigma)
    du = (2.0 * r * g - torch.erf(r / sigma)) / r**2
    d2u = 2.0 * torch.erf(r / sigma) / r**3 - 4.0 * g / r**2 - 4.0 * g / sigma**2
    return u, du, d2u


def electrostatic_pair_energy(
    qi: torch.Tensor, qj: torch.Tensor, r: torch.Tensor, sigma: float, cutoff: float | None
) -> torch.Tensor:
    """``k_e q_i q_j u(r)`` (eV); with a cutoff, the switched curvature-shifted form."""
    u = erf_coulomb(r, sigma)[0]
    if cutoff is not None:
        rc = torch.as_tensor(cutoff, dtype=r.dtype, device=r.device)
        u_c, du_c, d2u_c = erf_coulomb(rc, sigma)
        shifted = u - u_c
        fs2 = shifted - (r - rc) * du_c - 0.5 * (r - rc) ** 2 * d2u_c
        f = smooth_step(r, 0.45 * cutoff, cutoff)
        u = torch.where(r < rc, f * shifted + (1.0 - f) * fs2, torch.zeros_like(r))
    return COULOMB * qi * qj * u


@MODEL_REGISTRY.register("simurgh_dev_abl_physical_long_range")
@BACKBONE_REGISTRY.register("simurgh_dev_abl_physical_long_range")
class SimurghPhysicalLongRange(SimurghDev):
    """SIMURGH plus learned dispersion and erf-damped electrostatics (amplitude-gated).

    Configured by the ``long_range`` section (keys in :data:`DEFAULTS`).
    """

    def __init__(self, **config) -> None:
        lr = {**DEFAULTS, **dict(config.pop("long_range", None) or {})}
        unknown = sorted(set(lr) - set(DEFAULTS))
        if unknown:
            raise ValueError(f"unknown long_range key(s): {unknown}")
        if lr["electrostatics_kernel"] not in ("erf_fixed", "gaussian_pair"):
            raise ValueError("long_range.electrostatics_kernel must be erf_fixed or gaussian_pair")
        if lr["charge_constraint"] not in ("uniform", "pair_transfer"):
            raise ValueError("long_range.charge_constraint must be uniform or pair_transfer")
        if lr["electrostatics_kernel"] == "gaussian_pair":
            config["electrostatics"] = {
                **config.get("electrostatics", {}),
                "enabled": True,
                "charge_constraint": lr["charge_constraint"],
            }
        super().__init__(**config)
        if self.dispersion is not None:
            raise ValueError(
                "physical_long_range replaces the D3 term: set dispersion.enabled false"
            )
        missing = [z for z in self.elements if z not in FREE_ATOM]
        if missing:
            raise ValueError(f"no free-atom dispersion data for elements {missing} (H, C, N, O)")
        self.long_range = lr
        c = self.atom_line.channels
        self.ratio_bias = nn.Parameter(
            torch.full((len(self.elements),), math.log(math.expm1(0.8)))
        )
        self.ratio_mlp = _MLP([c, 64, 1], zero_last=True)
        self.charge_head = None
        if lr["electrostatics_kernel"] == "erf_fixed":
            self.charge_head = _ChargeHead(
                c,
                len(self.elements),
                lr["charge_constraint"],
                self.config["cutoff"],
                self.config["envelope_p"],
                self.config["num_rbf"],
            )
        self.a_disp = nn.Parameter(torch.zeros(()))
        self.a_elec = nn.Parameter(torch.zeros(()))
        self.last_terms: dict[str, torch.Tensor] = {}
        self._total: torch.Tensor | None = None
        self.to(self.scale.dtype)
        # float64 master copies, cast to the model dtype at use
        alpha0, c6 = zip(*(FREE_ATOM[z] for z in self.elements))
        self.register_buffer("alpha0_bohr3", torch.tensor(alpha0, dtype=torch.float64))
        self.register_buffer("c6_hartree_bohr6", torch.tensor(c6, dtype=torch.float64))

    def ratios(self, h: torch.Tensor, elem: torch.Tensor) -> torch.Tensor:
        """Volume ratios ``h_i > 0`` ``(N,)``; 0.8 at initialisation."""
        s = h[:, : self.atom_line.channels]
        return nn.functional.softplus(self.ratio_bias[elem] + self.ratio_mlp(s).squeeze(-1))

    def extra_energy(self, h, pos, elem, batch, num_graphs) -> torch.Tensor:
        """``a_disp E_disp + a_elec E_elec`` per molecule (eV)."""
        lr, total = self.long_range, self._total
        i, j = _intramolecular_pairs(batch, num_graphs)
        r = (pos[j] - pos[i]).norm(dim=-1)
        ratio = self.ratios(h, elem)
        alpha = ratio * self.alpha0_bohr3.to(ratio.dtype)[elem]
        c6 = ratio**2 * self.c6_hartree_bohr6.to(ratio.dtype)[elem]
        terms = dispersion_pair_terms(
            alpha[i], alpha[j], c6[i], c6[j], lr["disp_damping_gamma"], lr["exact_prefactor"]
        )
        e_pair = dispersion_pair_energy(r, terms)
        cutoff = lr["cutoff"]
        if cutoff is not None:
            e_pair = e_pair * smooth_step(r, cutoff - 2.0, cutoff)
        e_disp = scatter(e_pair, batch[i], dim=0, dim_size=num_graphs, reduce="sum")
        energy = self.a_disp * e_disp
        terms_log = {"lr_a_disp": self.a_disp.detach(), "lr_a_elec": self.a_elec.detach()}
        if self.charge_head is not None:
            q = self.charge_head(h, elem, batch, num_graphs, torch.stack([i, j]), r, total)
            e_q = electrostatic_pair_energy(q[i], q[j], r, lr["sigma"], cutoff)
            e_elec = scatter(e_q, batch[i], dim=0, dim_size=num_graphs, reduce="sum")
            energy = energy + self.a_elec * e_elec
        for k, z in enumerate(self.elements):
            mask = elem == k
            if bool(mask.any()):
                terms_log[f"lr_ratio_mean_Z{z}"] = ratio[mask].detach().mean()
        self.last_terms = terms_log
        return energy

    def forward(self, graph):
        """SimurghDev's outputs, with the amplitudes and ratio statistics logged in training."""
        self._total = graph.get("total_charge", None)
        out = super().forward(graph)
        if self.training and "aux_loss" in out:
            out.setdefault("aux_terms", {}).update(self.last_terms)
        return out
