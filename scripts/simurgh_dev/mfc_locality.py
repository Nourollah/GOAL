#!/usr/bin/env python
"""MFC locality map: how much each atom's predictions change when far atoms are cut away.

For every structure, the window set is every distinct half-space cut along
the principal axis plus ``--random-cuts`` cuts along random directions
(fixed seed, so the map is deterministic and invariant to rigid motions of
the principal-axis cuts).  For every atom ``i`` of a window, at distance
``d`` from the nearest removed atom, it records

* ``|Δe_i|``: change of its atomic energy (models returning
  ``atomic_energies``, e.g. SIMURGH);
* ``‖ΔF_i^core‖``: change of ``-∂(Σ_{k∈core} e_k)/∂r_i``, core = window atoms
  at least ``--buffer`` from every removed atom (same models);
* ``‖ΔF_i‖``: change of the full force (any GOAL model).

Outputs: median decay curves binned by ``d``, the locality length (beyond
which the median core-force change stays below 1 meV/Å), per-atom
locality scores (mean full-force change over the windows that keep the
atom at least ``--buffer`` from the cut) and, with ``--errors``, their
Spearman correlation with per-atom OOD force errors (an evaluation-format
``.npz`` of the same structures in the same order).  ``--export`` writes
the windows (with ``source``, ``window`` and ``distance`` arrays) as extxyz
for models evaluated outside GOAL.

Example::

    python scripts/simurgh_dev/mfc_locality.py <checkpoint> \\
        data/GMD/FragmentDuplication/TestOOD/*.traj --max-structures 20 --out locality.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from ase import Atoms

from goal.ml.nn.models.simurgh.dev.monolithic import core_mask, principal_axis


def window_masks(atoms: Atoms, random_cuts: int, seed: int) -> list[np.ndarray]:
    """Every distinct principal-axis half-space cut plus random cuts (boolean masks)."""
    rng = np.random.default_rng(seed)
    x = atoms.positions - atoms.positions.mean(axis=0)
    directions = [principal_axis(atoms.positions)]
    for _ in range(random_cuts):
        u = rng.normal(size=3)
        directions.append(u / np.linalg.norm(u))
    masks: list[np.ndarray] = []
    for k, u in enumerate(directions):
        s = np.sort(x @ u)
        cuts = 0.5 * (s[1:] + s[:-1]) if k == 0 else [rng.uniform(s[0], s[-1])]
        for t in cuts:
            inside = (x @ u) <= t
            if 0 < inside.sum() < len(atoms) and not any((inside == m).all() for m in masks):
                masks.append(inside)
    return masks


def _predict(calc, atoms: Atoms, core: np.ndarray | None) -> dict[str, np.ndarray]:
    """Forces, and atomic energies / core forces when the model returns them."""
    graph = calc.build_graph(atoms).to(calc.device)
    graph.pos = graph.pos.detach().requires_grad_(True)
    with torch.enable_grad():
        out = calc.module(graph)
        result = {"forces": out["forces"].detach().double().cpu().numpy()}
        e = out.get("atomic_energies")
        if e is not None:
            result["atomic"] = e.detach().double().cpu().numpy()
            if core is not None and core.any():
                q = e[torch.as_tensor(np.flatnonzero(core), device=e.device)].sum()
                (g,) = torch.autograd.grad(q, graph.pos)
                result["core_forces"] = -g.double().cpu().numpy()
    return result


def locality(calc, frames: list[Atoms], buffer: float, random_cuts: int, seed: int) -> dict:
    """Per-atom records ``(structure, atom, d, |Δe|, ‖ΔF_core‖, ‖ΔF‖)`` over all windows."""
    rows = []
    exported: list[Atoms] = []
    for s, atoms in enumerate(frames):
        full_mask = np.ones(len(atoms), dtype=bool)
        for w, inside in enumerate(window_masks(atoms, random_cuts, seed + s)):
            removed = atoms.positions[~inside]
            d = np.linalg.norm(atoms.positions[inside][:, None] - removed[None], axis=-1).min(1)
            core = core_mask(atoms.positions, inside, full_mask, buffer)
            full = _predict(calc, atoms, core)
            sub = atoms[inside]
            part = _predict(calc, sub, core[inside])
            idx = np.flatnonzero(inside)
            df = np.linalg.norm(part["forces"] - full["forces"][idx], axis=1)
            de = np.abs(part["atomic"] - full["atomic"][idx]) if "atomic" in part else np.nan * df
            if "core_forces" in part:
                dfc = np.linalg.norm(part["core_forces"] - full["core_forces"][idx], axis=1)
            else:
                dfc = np.nan * df
            for k, i in enumerate(idx):
                rows.append((s, i, d[k], de[k], dfc[k], df[k]))
            sub.arrays["source"] = idx
            sub.arrays["distance"] = d
            sub.info.update(source=s, window=w)
            exported.append(sub)
    table = np.array(rows, dtype=float).reshape(-1, 6)
    return {"table": table, "windows": exported}


def summarise(table: np.ndarray, bin_width: float = 0.5, threshold: float = 1e-3) -> dict:
    """Median decay curves by distance and the locality length (Å)."""
    d = table[:, 2]
    edges = np.arange(0.0, d.max() + bin_width, bin_width)
    which = np.digitize(d, edges) - 1
    curves = {name: [] for name in ("energy", "core_force", "force")}
    for b in range(len(edges) - 1):
        sel = which == b
        for col, name in ((3, "energy"), (4, "core_force"), (5, "force")):
            curves[name].append(np.nanmedian(table[sel, col]) if sel.any() else np.nan)
    centres = 0.5 * (edges[1:] + edges[:-1])
    core = np.array(
        curves["core_force"] if np.isfinite(curves["core_force"]).any() else curves["force"]
    )
    above = np.flatnonzero(np.nan_to_num(core, nan=0.0) >= threshold)
    length = float(centres[above[-1]] + bin_width / 2) if above.size else 0.0
    return {
        "bin_centres": centres,
        **{k: np.array(v) for k, v in curves.items()},
        "locality_length": length,
    }


def atom_scores(table: np.ndarray, buffer: float) -> dict[tuple[int, int], float]:
    """Mean full-force change per atom over windows keeping it ``buffer`` from the cut."""
    scores: dict[tuple[int, int], list[float]] = {}
    for s, i, d, _, _, df in table:
        if d >= buffer:
            scores.setdefault((int(s), int(i)), []).append(df)
    return {k: float(np.mean(v)) for k, v in scores.items()}


def main(argv: list[str] | None = None) -> None:
    from ase.io import write
    from scipy.stats import spearmanr

    from goal.md.core.batched import load_structures
    from goal.ml.utils.calculator import GOALCalculator

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("structures", nargs="+")
    parser.add_argument("--select", default="best")
    parser.add_argument("--buffer", type=float, default=5.0, help="core buffer b (Å)")
    parser.add_argument("--random-cuts", type=int, default=4)
    parser.add_argument("--max-structures", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--errors", help="evaluation-format .npz of the same structures")
    parser.add_argument("--out", help="write the per-atom table and curves (.npz)")
    parser.add_argument("--export", help="write the windows as extxyz")
    args = parser.parse_args(argv)

    calc = GOALCalculator(checkpoint_path=args.checkpoint, checkpoint=args.select)
    frames = load_structures(args.structures)[: args.max_structures]
    result = locality(calc, frames, args.buffer, args.random_cuts, args.seed)
    summary = summarise(result["table"])
    print(f"locality length: {summary['locality_length']:.2f} Å")
    print("d [Å]   median |Δe| [meV]   median ‖ΔF_core‖ [meV/Å]   median ‖ΔF‖ [meV/Å]")
    for d, e, fc, f in zip(
        summary["bin_centres"], summary["energy"], summary["core_force"], summary["force"]
    ):
        if np.isfinite(f):
            print(f"{d:5.2f}   {1e3 * e:17.4f}   {1e3 * fc:24.4f}   {1e3 * f:19.4f}")
    if args.errors:
        data = np.load(args.errors)
        offsets = np.concatenate([[0], np.cumsum(data["num_atoms"])])
        err = np.linalg.norm(data["forces"] - data["forces_ref"], axis=1)
        scores = atom_scores(result["table"], args.buffer)
        keys = sorted(scores)
        rho = spearmanr([scores[k] for k in keys], [err[offsets[s] + i] for s, i in keys])
        print(f"Spearman(locality score, OOD force error) = {rho.statistic:.3f} (n = {len(keys)})")
    if args.out:
        np.savez(args.out, table=result["table"], **summary)
    if args.export:
        write(args.export, result["windows"], format="extxyz")


if __name__ == "__main__":
    main()
