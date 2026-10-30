#!/usr/bin/env python
"""Context sensitivity of atomic energies: mean ‖∂e_i/∂r_j‖ binned by d_ij.

A strictly local model has ``∂e_i/∂r_j = 0`` beyond its cutoff; message
passing extends the dependence to the receptive field.  The curve shows
how much each atom's energy depends on atoms at a given distance, and is
the yardstick for multi-scale fragment consistency (it should fall faster
with it).  Run it on OOD molecules.

The model must return ``atomic_energies`` (one per atom) computed from
``graph.pos`` when that tensor already requires grad, as SIMURGH does.

Example::

    python scripts/simurgh_dev/context_sensitivity.py <checkpoint> \\
        data/GMD/FragmentDuplication/TestOOD/*.traj --max-structures 50 --out ctx.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def context_sensitivity(
    module: torch.nn.Module, graph: object, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin sums of ``‖∂e_i/∂r_j‖`` (i ≠ j) and pair counts for one structure."""
    pos = graph.pos.detach().requires_grad_(True)
    graph.pos = pos
    energies = module(graph)["atomic_energies"].reshape(-1)
    n = energies.shape[0]
    jac = torch.stack(
        [torch.autograd.grad(energies[i], pos, retain_graph=i < n - 1)[0] for i in range(n)]
    )  # (i, j, 3)
    sens = jac.norm(dim=-1).detach().cpu().numpy()
    dist = torch.cdist(pos.detach(), pos.detach()).cpu().numpy()
    off = ~np.eye(n, dtype=bool)
    sums, _ = np.histogram(dist[off], bins=bin_edges, weights=sens[off])
    counts, _ = np.histogram(dist[off], bins=bin_edges)
    return sums, counts


def main(argv: list[str] | None = None) -> None:
    from goal.md.core.batched import load_structures
    from goal.ml.utils.calculator import GOALCalculator

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help=".simurgh archive, checkpoint directory or .ckpt")
    parser.add_argument("structures", nargs="+", help="structure files (every frame is used)")
    parser.add_argument("--select", default="best", help="best | last | epoch=N (directories)")
    parser.add_argument("--max-structures", type=int, default=50)
    parser.add_argument("--bin-width", type=float, default=0.5, help="Å")
    parser.add_argument("--rmax", type=float, default=20.0, help="Å")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write bin_centres, mean, counts to this .npz")
    args = parser.parse_args(argv)

    calc = GOALCalculator(checkpoint_path=args.checkpoint, checkpoint=args.select)
    frames = load_structures(args.structures)
    rng = np.random.default_rng(args.seed)
    picked = rng.permutation(len(frames))[: args.max_structures]
    edges = np.arange(0.0, args.rmax + args.bin_width, args.bin_width)
    sums, counts = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    with torch.enable_grad():
        for k in picked:
            s, c = context_sensitivity(
                calc.module, calc.build_graph(frames[k]).to(calc.device), edges
            )
            sums, counts = sums + s, counts + c
    mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    centres = 0.5 * (edges[1:] + edges[:-1])
    print("d_ij [Å]   mean |de_i/dr_j| [meV/Å]   pairs")
    for d, m, c in zip(centres, mean, counts):
        if c:
            print(f"{d:7.2f}   {1e3 * m:24.4f}   {int(c)}")
    if args.out:
        np.savez(args.out, bin_centres=centres, mean=mean, counts=counts)


if __name__ == "__main__":
    main()
