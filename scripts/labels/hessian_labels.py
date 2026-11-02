#!/usr/bin/env python
"""Offline Hessian labels for Hessian distillation, from any GOAL model (the teacher).

    python scripts/labels/hessian_labels.py frames.xyz labelled.xyz \\
        --checkpoint logs/<task>/runs/<run>/checkpoints \\
        --teacher-id <name> --level-of-theory "PBE D3BJ def2-TZVP"

For every frame: the full energy Hessian d2E/dx2 (eV/A^2) by automatic
differentiation of the teacher's forces, 3N backward passes, written as the
per-atom array ``hessian`` (N, 9N): atom i's three rows, float32.  Frame
metadata: ``hessian_teacher``, ``hessian_level_of_theory``,
``hessian_dtype``, ``hessian_units``.  The teacher must reproduce the
dataset's level of theory.  Outside SIMURGH on purpose: a teacher may be any
registered model, foundation models included.
"""

from __future__ import annotations

import argparse
import typing

import numpy as np
import torch


def hessian(model: typing.Callable, atoms, cutoff: float) -> np.ndarray:
    """Full Hessian ``(3N, 3N)`` (eV/A^2) of ``model``'s energy at ``atoms``."""
    from goal.ml.data.graph import AtomicGraph

    graph = AtomicGraph.from_ase(atoms, cutoff=cutoff, dtype=torch.float64)
    graph.batch = torch.zeros(len(atoms), dtype=torch.long)
    graph.pos.requires_grad_(True)
    out = model(graph)
    forces = (
        out["forces"]
        if "forces" in out
        else -torch.autograd.grad(out["energy"].sum(), graph.pos, create_graph=True)[0]
    )
    flat = forces.reshape(-1)
    rows = [
        -torch.autograd.grad(flat[k], graph.pos, retain_graph=True)[0].reshape(-1)
        for k in range(flat.numel())
    ]
    return torch.stack(rows).detach().double().numpy()


def label(frames: list, model: typing.Callable, cutoff: float, teacher: str, theory: str) -> list:
    """The frames with ``hessian`` arrays and provenance metadata attached."""
    out = []
    for atoms in frames:
        atoms = atoms.copy()
        n = len(atoms)
        h = hessian(model, atoms, cutoff)
        atoms.arrays["hessian"] = h.astype(np.float32).reshape(n, 9 * n)
        atoms.info.update(
            hessian_teacher=teacher,
            hessian_level_of_theory=theory,
            hessian_dtype="float32",
            hessian_units="eV/A^2",
        )
        out.append(atoms)
    return out


def main() -> None:
    from ase.io import read, write

    from goal.ml.utils.calculator import load_goal_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input")
    parser.add_argument("output", help="extxyz file with the hessian arrays")
    parser.add_argument("--checkpoint", required=True, help="teacher: any GOAL checkpoint")
    parser.add_argument("--teacher-id", required=True)
    parser.add_argument("--level-of-theory", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    loaded = load_goal_checkpoint(args.checkpoint, device=args.device)
    model = loaded.module.double().eval()
    frames = read(args.input, index=":")
    labelled = label(frames, model, loaded.cutoff, args.teacher_id, args.level_of_theory)
    write(args.output, labelled, format="extxyz")
    print(f"{len(labelled)} frames with Hessians -> {args.output}")


if __name__ == "__main__":
    main()
