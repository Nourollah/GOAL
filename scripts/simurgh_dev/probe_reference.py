#!/usr/bin/env python
"""Check how a checkpoint handles isolated-atom reference energies.

Run it on every baseline before comparing total energies.

1. **Isolated atoms.**  Predicts single H, C, N and O atoms (the elements
   the model knows).  With isolated-atom E0 applied the model must return
   the E0 values of its config.  Two failure signatures:

   * a minimum-norm least-squares element reference fitted on
     Duplication pins only E0(C) + 2 E0(H), so it returns roughly
     C -211.1, H -422.3, O -2041.1 eV, with H *below* C;
   * no reference plus per-atom mean scaling returns about the same value
     for every element.

2. **Composition regression** (``--predictions``, a file in GOAL's
   evaluation format): regresses the per-structure energy residual on the
   element counts.  A referencing error shows up as large per-element
   coefficients, i.e. an error linear in composition.

Example::

    python scripts/simurgh_dev/probe_reference.py logs/<task>/runs/<run>/checkpoints \\
        --predictions ood_predictions.npz
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from ase import Atoms

SYMBOLS: dict[int, str] = {1: "H", 6: "C", 7: "N", 8: "O"}


def isolated_atom_energies(checkpoint: str, select: str = "best") -> dict[int, dict[str, float]]:
    """``{Z: {"predicted": E, "e0": config E0 or nan}}`` for H, C, N, O."""
    from goal.ml.utils.calculator import GOALCalculator

    calc = GOALCalculator(checkpoint_path=checkpoint, checkpoint=select)
    backbone = calc.config.model.backbone
    values = (backbone.get("atomic_energies") or {}).get("values") or {}
    e0 = {int(z): float(e) for z, e in values.items()}
    known = backbone.get("elements") or list(e0) or list(SYMBOLS)
    out: dict[int, dict[str, float]] = {}
    for z, symbol in SYMBOLS.items():
        if z not in {int(k) for k in known}:
            continue
        atoms = Atoms(numbers=[z], positions=[[0.0, 0.0, 0.0]])
        atoms.calc = calc
        try:
            energy = float(atoms.get_potential_energy())
        except Exception as exc:  # noqa: BLE001  (report, keep probing)
            print(f"  {symbol}: prediction failed: {exc!r}")
            energy = float("nan")
        out[z] = {"predicted": energy, "e0": e0.get(z, float("nan"))}
    return out


def composition_regression(path: str) -> tuple[dict[int, float], float]:
    """Least-squares ``residual ≈ Σ_Z c_Z n_Z``; returns ``({Z: c_Z}, R²)``."""
    data = np.load(path, allow_pickle=False)
    residual = data["energy"] - data["energy_ref"]
    offsets = np.concatenate([[0], np.cumsum(data["num_atoms"])])
    elements = sorted(set(data["atomic_numbers"].tolist()))
    counts = np.array(
        [
            [np.count_nonzero(data["atomic_numbers"][lo:hi] == z) for z in elements]
            for lo, hi in itertools.pairwise(offsets)
        ],
        dtype=float,
    )
    coef, *_ = np.linalg.lstsq(counts, residual, rcond=None)
    ss_res = float(np.sum((residual - counts @ coef) ** 2))
    ss_tot = float(np.sum((residual - residual.mean()) ** 2)) or float("nan")
    return dict(zip(elements, coef.tolist())), 1.0 - ss_res / ss_tot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help=".simurgh archive, checkpoint directory or .ckpt")
    parser.add_argument("--select", default="best", help="best | last | epoch=N (directories)")
    parser.add_argument("--predictions", help="evaluation-format .npz for the regression")
    args = parser.parse_args(argv)

    print("isolated atoms (eV):")
    rows = isolated_atom_energies(args.checkpoint, args.select)
    for z, row in rows.items():
        diff = row["predicted"] - row["e0"]
        print(
            f"  {SYMBOLS[z]}: predicted {row['predicted']:14.6f}  "
            f"E0 {row['e0']:14.6f}  diff {diff:+.3e}"
        )
    if 1 in rows and 6 in rows and rows[1]["predicted"] < rows[6]["predicted"]:
        print("  WARNING: H below C - the signature of a minimum-norm least-squares reference")
    if args.predictions:
        coef, r2 = composition_regression(args.predictions)
        print(f"residual ~ sum_Z c_Z n_Z  (R^2 = {r2:.3f}):")
        for z, c in coef.items():
            print(f"  c_{SYMBOLS.get(z, z)} = {1000 * c:+.2f} meV per atom")


if __name__ == "__main__":
    main()
