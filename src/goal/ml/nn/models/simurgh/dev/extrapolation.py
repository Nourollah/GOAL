"""Atom-resolved extrapolation grade γ (``diagnostics.calm``; arXiv:2609.40060, Eqs. 1-11).

γ_i is the Mahalanobis distance of atom ``i``'s feature to the nearest
training cluster of its element, divided by that cluster's robust training
threshold: γ ≤ 1 inside the training support, γ > 1 outside.  It is
output only and never enters energies or forces.

Feature (Eqs. 4-7): ``B`` = the unweighted 0e product basis of every layer
(model output ``calm_basis``); ``z = (B/‖B‖) R, ln‖B‖, ln‖B_1‖, ...`` with a
Gaussian projection ``R`` (``D_rp`` = 128 columns, seed 42).  Build (Methods
4.1): three additive passes over the training atoms: k-means per element
with an elbow over K ∈ {1, 2, 4, 8, 16} and small clusters merged; counts,
sums and scatter matrices per cluster; then thresholds
``θ = median + 3 · 1.4826 · MAD`` of the training σ.  Quadratic forms in
float64.  The grade is a file next to the checkpoint, not model state.
"""

from __future__ import annotations

import math

import torch
from torch_geometric.utils import scatter

EPS0 = 1e-12
KS = (1, 2, 4, 8, 16)
EDGES = torch.linspace(0.0, 100.0, 251, dtype=torch.float64)  # σ histogram bins


def projection(d_basis: int, d_rp: int = 128, seed: int = 42) -> torch.Tensor:
    """``R`` ``(d_basis, d_rp)``, entries ``N(0, 1)/√d_rp`` from ``seed`` (float32)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(d_basis, d_rp, generator=g, dtype=torch.float64) / math.sqrt(d_rp)).float()


def features(basis: torch.Tensor, layer_norms: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """``z = (B̂ R, ln‖B‖, ln‖B_m‖)`` ``(N, D_rp + 1 + L)``, float64."""
    basis = basis.double()
    norm = basis.norm(dim=-1, keepdim=True)
    p = (basis / (norm + EPS0)) @ r.to(basis)
    logs = torch.log(torch.cat([norm, layer_norms.double()], dim=-1) + EPS0)
    return torch.cat([p, logs], dim=-1)


def kmeans(x: torch.Tensor, k: int, seed: int = 42, iters: int = 100) -> torch.Tensor:
    """Centroids ``(k, D)`` by Lloyd's algorithm from a seeded k-means++ start."""
    g = torch.Generator().manual_seed(seed)
    k = min(k, x.shape[0])
    centres = [x[torch.randint(x.shape[0], (1,), generator=g)].squeeze(0)]
    for _ in range(1, k):
        d2 = torch.cdist(x, torch.stack(centres)).square().min(dim=1).values
        prob = d2 if float(d2.sum()) > 0 else torch.ones_like(d2)
        centres.append(x[torch.multinomial(prob, 1, generator=g)].squeeze(0))
    mu = torch.stack(centres)
    for _ in range(iters):
        label = torch.cdist(x, mu).argmin(dim=1)
        new = torch.stack(
            [x[label == c].mean(0) if (label == c).any() else mu[c] for c in range(k)]
        )
        if torch.allclose(new, mu):
            break
        mu = new
    return mu


def wcss(x: torch.Tensor, mu: torch.Tensor) -> float:
    """Within-cluster sum of squares."""
    return float(torch.cdist(x, mu).square().min(dim=1).values.sum())


def elbow(ks: list[int], values: list[float]) -> int:
    """The K farthest from the line through the first and last (K, WCSS) points."""
    if len(ks) < 3:
        return ks[0]
    x = torch.tensor(ks, dtype=torch.float64)
    y = torch.tensor(values, dtype=torch.float64)
    x, y = (x - x[0]) / (x[-1] - x[0]), (y - y[-1]) / max(float(y[0] - y[-1]), 1e-300)
    distance = (y - (1 - x)).abs() / math.sqrt(2.0)  # line from (0, 1) to (1, 0)
    return ks[int(distance.argmax())]


def merge_small(x: torch.Tensor, mu: torch.Tensor, minimum: int) -> torch.Tensor:
    """Merge clusters with fewer than ``minimum`` atoms into their nearest centroid."""
    while mu.shape[0] > 1:
        label = torch.cdist(x, mu).argmin(dim=1)
        counts = torch.bincount(label, minlength=mu.shape[0])
        small = int(counts.argmin())
        if counts[small] >= minimum:
            break
        keep = [c for c in range(mu.shape[0]) if c != small]
        mu = mu[keep]
    return mu


class GradeBuilder:
    """The three additive build passes; feed shards in any split (passes 2 and 3 are sums)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.centroids: dict[int, torch.Tensor] = {}
        self.stats: dict[int, dict[str, torch.Tensor]] = {}
        self.sigmas: dict[int, list[torch.Tensor]] = {}
        self.hist: dict[int, torch.Tensor] = {}

    def pass1(self, z: torch.Tensor, numbers: torch.Tensor, seed: int = 42) -> None:
        """Clusters per element (on all of ``z``: the training atoms, or a subset)."""
        for zz in torch.unique(numbers).tolist():
            x = z[numbers == zz]
            ks = [k for k in KS if k <= max(1, x.shape[0] // (self.dim + 1))]
            fits = [kmeans(x, k, seed) for k in ks]
            mu = fits[ks.index(elbow(ks, [wcss(x, m) for m in fits]))]
            self.centroids[zz] = merge_small(x, mu, self.dim + 1)

    def assign(self, z: torch.Tensor, zz: int) -> torch.Tensor:
        return torch.cdist(z, self.centroids[zz]).argmin(dim=1)

    def pass2(self, z: torch.Tensor, numbers: torch.Tensor) -> None:
        """Accumulate counts, sums and scatter matrices per (element, cluster)."""
        for zz in torch.unique(numbers).tolist():
            x = z[numbers == zz]
            k = self.centroids[zz].shape[0]
            label = self.assign(x, zz)
            s = self.stats.setdefault(
                zz,
                {
                    "count": torch.zeros(k, dtype=torch.float64),
                    "sum": torch.zeros(k, self.dim, dtype=torch.float64),
                    "scatter": torch.zeros(k, self.dim, self.dim, dtype=torch.float64),
                },
            )
            d = x - self.centroids[zz][label]  # centred on pass 1: no cancellation
            s["count"] += torch.bincount(label, minlength=k).double()
            s["sum"] += scatter(d, label, dim=0, dim_size=k, reduce="sum")
            s["scatter"] += scatter(
                d.unsqueeze(-1) * d.unsqueeze(-2), label, dim=0, dim_size=k, reduce="sum"
            )

    def finish_moments(self) -> None:
        """Means and precisions ``pinv(Σ + 1e-6 I)`` (population covariance)."""
        for zz, s in self.stats.items():
            n = s["count"].clamp_min(1.0)
            shift = s["sum"] / n[:, None]
            cov = s["scatter"] / n[:, None, None] - shift.unsqueeze(-1) * shift.unsqueeze(-2)
            eye = torch.eye(self.dim, dtype=torch.float64)
            s["mean"] = self.centroids[zz] + shift
            s["covariance"] = cov
            s["precision"] = torch.linalg.pinv(cov + 1e-6 * eye, hermitian=True)

    def pass3(self, z: torch.Tensor, numbers: torch.Tensor) -> None:
        """Histogram σ per (element, cluster) and keep σ for the median and MAD."""
        for zz in torch.unique(numbers).tolist():
            x = z[numbers == zz]
            s = self.stats[zz]
            label = self.assign(x, zz)
            sigma = mahalanobis(x, s["mean"][label], s["precision"][label])
            k = s["mean"].shape[0]
            h = self.hist.setdefault(zz, torch.zeros(k, EDGES.numel() - 1, dtype=torch.float64))
            for c in range(k):
                h[c] += torch.histogram(sigma[label == c].clamp(0, 100), EDGES).hist
            self.sigmas.setdefault(zz, []).append(torch.stack([label.double(), sigma]))

    def grade(self, r: torch.Tensor, seed: int) -> dict:
        """The finished grade (thresholds from the median and MAD of σ)."""
        out: dict = {"seed": seed, "projection": r, "elements": {}}
        floor = max(50, self.dim)
        for zz, s in self.stats.items():
            label, sigma = torch.cat(self.sigmas[zz], dim=1)
            theta = torch.ones(s["mean"].shape[0], dtype=torch.float64)
            reliable = []
            for c in range(theta.numel()):
                v = sigma[label == c]
                if v.numel():
                    med = v.median()
                    theta[c] = med + 3.0 * 1.4826 * (v - med).abs().median()
                reliable.append(v.numel() >= floor)
            ok = torch.tensor(reliable)
            if ok.any():
                theta[~ok] = theta[ok].max()
            out["elements"][zz] = {
                "mean": s["mean"],
                "covariance": s["covariance"],
                "precision": s["precision"],
                "theta": theta,
                "count": s["count"],
                "histogram": self.hist[zz],
            }
        return out


def mahalanobis(z: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    """``sqrt((z - μ)ᵀ P (z - μ) + 1e-8)`` per row, float64."""
    d = (z - mu).double()
    return torch.sqrt(torch.einsum("ni,nij,nj->n", d, precision.double(), d) + 1e-8)


def score(grade: dict, z: torch.Tensor, numbers: torch.Tensor) -> torch.Tensor:
    """γ ``(N,)``: σ to the nearest cluster of each atom's element over its θ (inf if unseen)."""
    gamma = torch.full((z.shape[0],), float("inf"), dtype=torch.float64)
    for zz in torch.unique(numbers).tolist():
        g = grade["elements"].get(int(zz))
        if g is None:
            continue
        mask = numbers == zz
        x = z[mask].double()
        label = torch.cdist(x, g["mean"]).argmin(dim=1)
        gamma[mask] = mahalanobis(x, g["mean"][label], g["precision"][label]) / g["theta"][label]
    return gamma


def neighbourhood(
    gamma: torch.Tensor, edge_index: torch.Tensor, lengths: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """``(γ_i + Σ_j w_ij γ_j) / (1 + Σ_j w_ij)`` with ``w = ½(1 + cos(π r / r_c))``."""
    w = torch.where(lengths < cutoff, 0.5 * (1 + torch.cos(math.pi * lengths / cutoff)), 0.0)
    i, j = edge_index
    w = w.double()
    weighted = torch.where(w > 0, w * gamma[j], torch.zeros_like(w))  # no 0 · inf
    num = gamma + scatter(weighted, i, dim=0, dim_size=gamma.numel(), reduce="sum")
    den = 1.0 + scatter(w, i, dim=0, dim_size=gamma.numel(), reduce="sum")
    return num / den
