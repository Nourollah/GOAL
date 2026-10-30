"""Additive physical long-range kernels with parameters from local heads.

Electrostatics: latent charges (and optional dipoles) from the WP7 section.
Dispersion: ``E = -Σ_{i<j} Σ_{n=3,4,5} C_{2n,ij} / (r^{2n} + R_{d,ij}^{2n})``:

* ``C6`` from Tkatchenko-Scheffler combination rules on predicted positive
  volume ratios ``v_i`` (softplus, starting at 1): ``C6_i = C6_free v_i²``,
  ``α_i = α_free v_i``, ``C6_ij = 2 C6_i C6_j / (α_j/α_i C6_i + α_i/α_j C6_j)``;
* ``C8``, ``C10`` from quantum Drude oscillator scaling,
  ``C8 = 5 (αω) C6`` and ``C10 = (245/8) (αω)² C6`` with ``ω = 4 C6 / (3 α²)``
  (atomic units), combined between elements by geometric means;
* damping radius ``R_d,ij = κ (R_i + R_j)`` with ``R_i = 2.54 α_i^{1/7}`` bohr
  and one global ``κ``.

shortcut: reconstructed from the brief, not from the reference code; the
free-atom data (C6, α) are the Tkatchenko-Scheffler references for H, C,
N, O; check against the published model before reporting the rung.

The dispersion term is multiplied by a zero-initialised amplitude, so a
fresh model equals SimurghDev.  Every parameter learns from energies and
forces only (no dipole or Hirshfeld labels).  On GMD (D3(BJ) labels) it
replaces, never stacks with, the D3 term; on MD22 it replaces D3.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import scatter

from goal.ml.nn.models.simurgh.dev.monolithic import SimurghDev
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

_BOHR = 0.52917721067  # Å
_HARTREE = 27.211386245988  # eV
#: Free-atom C6 (Ha bohr⁶) and polarisability (bohr³), Tkatchenko-Scheffler references.
FREE_ATOM: dict[int, tuple[float, float]] = {
    1: (6.50, 4.50),
    6: (46.6, 12.0),
    7: (24.2, 7.4),
    8: (15.6, 5.4),
}


@MODEL_REGISTRY.register("simurgh_dev_abl_physical_long_range")
@BACKBONE_REGISTRY.register("simurgh_dev_abl_physical_long_range")
class SimurghPhysicalLongRange(SimurghDev):
    """SIMURGH plus learned electrostatics and many-body-free learned dispersion."""

    def __init__(self, **config) -> None:
        electrostatics = {**config.get("electrostatics", {}), "enabled": True}
        super().__init__(**{**config, "electrostatics": electrostatics})
        if self.dispersion is not None:
            raise ValueError(
                "physical_long_range replaces the D3 term: set dispersion.enabled false"
            )
        missing = [z for z in self.elements if z not in FREE_ATOM]
        if missing:
            raise ValueError(f"no free-atom dispersion data for elements {missing}")
        channels = self.config["num_channels"]
        self.volume_head = nn.Linear(channels, 1, bias=False)
        nn.init.zeros_(self.volume_head.weight)
        c6, alpha = zip(*(FREE_ATOM[z] for z in self.elements))
        self.register_buffer("c6_free", torch.tensor(c6))
        self.register_buffer("alpha_free", torch.tensor(alpha))
        self.log_kappa = nn.Parameter(torch.zeros(()))
        self.dispersion_amplitude = nn.Parameter(torch.zeros(()))
        self.to(self.scale.dtype)

    def extra_energy(self, h, pos, elem, batch, num_graphs) -> torch.Tensor:
        """Learned dispersion energy per molecule (eV)."""
        v = nn.functional.softplus(
            self.volume_head(h[:, : self.config["num_channels"]]).squeeze(-1)
            + math.log(math.e - 1.0)
        )
        c6 = self.c6_free[elem] * v**2
        alpha = self.alpha_free[elem] * v
        omega = 4.0 * c6 / (3.0 * alpha**2)
        c8 = 5.0 * alpha * omega * c6
        c10 = 245.0 / 8.0 * (alpha * omega) ** 2 * c6
        radius = 2.54 * alpha ** (1.0 / 7.0)
        pairs, start = [], 0
        for n in torch.bincount(batch, minlength=num_graphs).tolist():
            pairs.append(torch.triu_indices(n, n, offset=1, device=pos.device) + start)
            start += n
        i, j = torch.cat(pairs, dim=1)
        r = (pos[j] - pos[i]).norm(dim=-1) / _BOHR
        c6_ij = 2 * c6[i] * c6[j] / (alpha[j] / alpha[i] * c6[i] + alpha[i] / alpha[j] * c6[j])
        rd = torch.exp(self.log_kappa) * (radius[i] + radius[j])
        e = c6_ij / (r**6 + rd**6)
        e = e + torch.sqrt(c8[i] * c8[j]) / (r**8 + rd**8)
        e = e + torch.sqrt(c10[i] * c10[j]) / (r**10 + rd**10)
        energy = -_HARTREE * scatter(e, batch[i], dim=0, dim_size=num_graphs, reduce="sum")
        return self.dispersion_amplitude * energy
