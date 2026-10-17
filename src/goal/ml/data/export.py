"""Export GOAL datasets to extended XYZ (ExtXYZ).

GOAL reads many formats but, until now, had no dataset→xyz writer (the only
writer in the codebase was the MD trajectory observer).  A file-based export
is what the UPET / metatrain subprocess fine-tuning path needs: metatrain
consumes ``.xyz`` files, not in-memory graphs.

The energy label is written to ``atoms.info["energy"]`` and forces to
``atoms.arrays["forces"]`` via an ASE ``SinglePointCalculator`` so the ExtXYZ
writer emits them in the standard places that metatrain (and ASE) read back.
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch


def atomicgraph_to_atoms(graph: typing.Any) -> typing.Any:
    """Convert a single (unbatched) ``AtomicGraph`` to an ASE ``Atoms``.

    Energy/forces, when present, are attached via a ``SinglePointCalculator``
    so they survive the ExtXYZ round-trip.
    """
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    z = graph.z.detach().cpu().numpy()
    positions = graph.pos.detach().cpu().numpy()

    cell = getattr(graph, "cell", None)
    pbc = getattr(graph, "pbc", None)
    if cell is not None:
        cell_np = cell.detach().cpu().numpy().reshape(3, 3)
    else:
        cell_np = np.zeros((3, 3))
    pbc_np = (
        pbc.detach().cpu().numpy().reshape(-1)[:3] if pbc is not None else np.zeros(3, dtype=bool)
    )

    atoms = Atoms(numbers=z, positions=positions, cell=cell_np, pbc=pbc_np)

    energy = getattr(graph, "energy", None)
    forces = getattr(graph, "forces", None)
    results: dict[str, typing.Any] = {}
    if energy is not None:
        results["energy"] = float(torch.as_tensor(energy).reshape(-1)[0].item())
    if forces is not None:
        results["forces"] = forces.detach().cpu().numpy().reshape(-1, 3)
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def write_extxyz(
    dataset: typing.Iterable[typing.Any],
    path: str | Path,
) -> int:
    """Write an iterable of ``AtomicGraph`` to an ExtXYZ file.

    Parameters
    ----------
    dataset : iterable of AtomicGraph
        Unbatched graphs (e.g. a GOAL dataset or a list of graphs).
    path : str or Path
        Output ``.xyz`` / ``.extxyz`` file.

    Returns
    -------
    int
        Number of structures written.
    """
    from ase.io import write

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    images = [atomicgraph_to_atoms(g) for g in dataset]
    write(str(out), images, format="extxyz")
    return len(images)
