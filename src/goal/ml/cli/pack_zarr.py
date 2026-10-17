"""``goal-pack-zarr`` — convert structure files to a GOAL Zarr store.

Examples::

    goal-pack-zarr --input data/train.xyz --output data/gmd.zarr --split train
    goal-pack-zarr --input data/val.xyz --output data/gmd.zarr --split val
    # also store neighbour lists for a 5 Å model (reused when cutoffs match)
    goal-pack-zarr --input data/train.xyz --output data/gmd.zarr --compute-edges --cutoff 5.0

Train on the store with ``data.dataset_type: zarr`` and
``data.root: data/gmd.zarr`` (splits are the group names).  Every format
``ase.io.read`` understands works as input; labels are taken from
``atoms.info`` / ``atoms.arrays`` or an attached calculator, like the
xyz and trajectory datasets do.
"""

from __future__ import annotations

import argparse
import sys
import typing


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="goal-pack-zarr", description="Convert structure files to a GOAL Zarr store."
    )
    p.add_argument(
        "--input", nargs="+", required=True, help="structure file(s), any ase.io format"
    )
    p.add_argument("--output", required=True, help="Zarr store directory (created if missing)")
    p.add_argument("--split", default="train", help="group name inside the store (default: train)")
    p.add_argument("--compute-edges", action="store_true", help="also store neighbour lists")
    p.add_argument("--cutoff", type=float, help="cutoff in Å for --compute-edges")
    p.add_argument(
        "--neighbor-list-backend",
        default="ase",
        help="ase | matscipy | nvalchemiops | torch | auto",
    )
    p.add_argument("--overwrite", action="store_true", help="replace an existing split group")
    return p


def main(argv: typing.Sequence[str] | None = None) -> int:
    """Entry point of the ``goal-pack-zarr`` console script.

    Parses ``argv`` (``sys.argv[1:]`` when ``None``), streams every
    frame of every ``--input`` file (``ase.io.iread``) into the
    ``--split`` group of the Zarr store at ``--output`` with
    :func:`~goal.ml.data.zarr_dataset.write_zarr`, storing neighbour
    lists too with ``--compute-edges``, then prints the structure count
    and the stored arrays.

    Returns
    -------
    int
        ``0`` on success; ``1`` if ``--compute-edges`` is given without
        ``--cutoff``, or writing raises ``ImportError``,
        ``FileExistsError`` or ``FileNotFoundError`` (message on stderr).
    """
    args = _parser().parse_args(argv)
    if args.compute_edges and args.cutoff is None:
        print("goal-pack-zarr: --compute-edges needs --cutoff", file=sys.stderr)
        return 1

    from ase.io import iread

    from goal.ml.data.zarr_dataset import write_zarr

    def frames() -> typing.Iterator[typing.Any]:
        for path in args.input:
            yield from iread(path, index=":")

    try:
        store = write_zarr(
            frames(),
            args.output,
            split=args.split,
            cutoff=args.cutoff if args.compute_edges else None,
            neighbor_list_backend=args.neighbor_list_backend,
            overwrite=args.overwrite,
        )
    except (ImportError, FileExistsError, FileNotFoundError) as exc:
        print(f"goal-pack-zarr: {exc}", file=sys.stderr)
        return 1

    import zarr

    group = zarr.open_group(str(store), mode="r")[args.split]
    stored = sorted(k for k in group.array_keys() if k not in ("n_atoms", "n_edges"))
    print(
        f"Packed {group.attrs['num_structures']} structures into {store}/{args.split}: {', '.join(stored)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
