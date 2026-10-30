#!/usr/bin/env python
"""Export SIMURGH's pair potentials φ_ab(r) and their derivatives.

For every unordered element pair writes, in eV and Å, the scaled learned
potential ``s·φ_ab(r)``, the fixed short-range wall, their sum, and the
derivative of the sum, to ``pair_potentials.csv`` (one row per distance),
plus ``pair_potentials.png``.

Example::

    python scripts/simurgh_dev/export_pair_potentials.py logs/<task>/runs/<run>/checkpoints \\
        --out exports/
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import torch
from ase.data import chemical_symbols


def pair_potentials(model: torch.nn.Module, r_max: float | None = None, points: int = 500) -> dict:
    """``{"r": grid, "<A>-<B>": {"phi", "wall", "total", "dtotal_dr"}}`` for every pair."""
    pl = model.pair_line
    if pl is None:
        raise ValueError("this model has no pair line (pair_line.enabled: false)")
    dtype = model.scale.dtype
    grid = torch.linspace(0.3, r_max or pl.cutoff, points, dtype=dtype, requires_grad=True)
    emb = model.atom_line.embedding.weight
    out: dict = {"r": grid.detach().numpy()}
    for a, b in itertools.combinations_with_replacement(range(len(model.elements)), 2):
        ia = torch.full((points,), a, dtype=torch.long)
        ib = torch.full((points,), b, dtype=torch.long)
        phi = model.scale * pl.learned(grid, ia, ib, emb)
        wall = pl.wall_energy(grid, ia, ib)
        total = phi + wall
        (slope,) = torch.autograd.grad(total.sum(), grid)
        name = f"{chemical_symbols[model.elements[a]]}-{chemical_symbols[model.elements[b]]}"
        out[name] = {
            "phi": phi.detach().numpy(),
            "wall": wall.detach().numpy(),
            "total": total.detach().numpy(),
            "dtotal_dr": slope.numpy(),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    from goal.ml.utils.calculator import load_goal_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help=".simurgh archive, checkpoint directory or .ckpt")
    parser.add_argument("--select", default="best", help="best | last | epoch=N (directories)")
    parser.add_argument("--out", default=".", help="output directory")
    args = parser.parse_args(argv)
    model = load_goal_checkpoint(args.checkpoint, checkpoint=args.select).module.backbone
    data = pair_potentials(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [k for k in data if k != "r"]
    with open(out / "pair_potentials.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["r"] + [f"{p}:{q}" for p in pairs for q in data[p]])
        for i, r in enumerate(data["r"]):
            writer.writerow([r] + [data[p][q][i] for p in pairs for q in data[p]])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_e, ax_f) = plt.subplots(1, 2, figsize=(10, 4))
    for p in pairs:
        ax_e.plot(data["r"], data[p]["total"], label=p)
        ax_f.plot(data["r"], data[p]["dtotal_dr"], label=p)
    ax_e.set(xlabel="r (Å)", ylabel="φ(r) (eV)", ylim=(-2, 2))
    ax_f.set(xlabel="r (Å)", ylabel="dφ/dr (eV/Å)", ylim=(-5, 5))
    ax_e.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pair_potentials.png", dpi=150)
    print(f"wrote {out / 'pair_potentials.csv'} and {out / 'pair_potentials.png'}")


if __name__ == "__main__":
    main()
