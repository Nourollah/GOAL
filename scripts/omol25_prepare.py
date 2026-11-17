#!/usr/bin/env python
"""Index an OMol25 (or any FairChem ``.aselmdb``) split and report training stats.

Two jobs, both one-off:

1. **Build the row catalogues.**  ``ASELMDBDataset`` needs a per-row
   summary (elements, atom count, energy, charge, spin, formula,
   sub-corpus) before it can filter a shard, and scanning one shard takes
   ~8 s.  A 4M-row split is 80 shards, so doing that inside the first
   training run costs ten minutes of a GPU sitting idle.  This script
   does every shard in parallel and caches the result beside it as
   ``<shard>.goal-catalogue.npz``; training then starts immediately.

2. **Print the numbers a config can pin.**  ``elements``,
   ``atomic_energies.values``, ``scale`` and ``avg_num_neighbors`` all
   depend on the *filtered* subset the run will actually train on.
   Pinned in the config, they are not derived again at start-up, so they
   are computed here under exactly the filters the config declares, and
   printed as a paste-ready YAML block.

Reference energies and the energy scale come from the catalogue's
formulas (no geometry touched — seconds, not hours);
``avg_num_neighbors`` needs real neighbour lists, so it is estimated from
a random sample.

Examples
--------
Index the whole split with 8 workers and report stats for the default
CHNOFPSCl / neutral / singlet subset at a 6 Å cutoff::

    python scripts/omol25_prepare.py --root data/OMol25/train_4M --workers 8

Stats only, from the shards already indexed::

    python scripts/omol25_prepare.py --root data/OMol25/train_4M --no-index
"""

from __future__ import annotations

import argparse
import time
import typing
from pathlib import Path

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from ase.data import chemical_symbols  # noqa: E402

from goal.ml.data.datasets.aselmdb import (  # noqa: E402
    ASELMDBCatalogue,
    ASELMDBReader,
    linear_reference_energies,
    residual_energy_scale,
    resolve_aselmdb_files,
)

#: Default chemistry: the organic elements H, C, N, O, F, P, S and Cl.
DEFAULT_ELEMENTS: tuple[int, ...] = (1, 6, 7, 8, 9, 15, 16, 17)


def _index_shard(shard: str) -> tuple[str, int, float]:
    """Build (or load) one shard's catalogue.  Runs in a worker process."""
    started: float = time.perf_counter()
    reader: ASELMDBReader = ASELMDBReader(shard)
    catalogue: ASELMDBCatalogue = reader.catalogue(cache=True)
    reader.close()
    return shard, len(catalogue), time.perf_counter() - started


def build_catalogues(root: str, workers: int) -> None:
    """Scan every shard under *root*, in parallel, caching each summary."""
    import concurrent.futures as futures

    shards: list[Path] = resolve_aselmdb_files(root)
    print(f"[prepare] indexing {len(shards)} shards with {workers} worker(s)")
    started: float = time.perf_counter()
    done: int = 0
    total_rows: int = 0
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for shard, rows, elapsed in pool.map(_index_shard, [str(s) for s in shards]):
            done += 1
            total_rows += rows
            print(
                f"[prepare] {done:>3}/{len(shards)}  {Path(shard).name}  "
                f"{rows:>7,} rows  {elapsed:5.1f}s"
            )
    print(f"[prepare] {total_rows:,} rows indexed in {time.perf_counter() - started:.1f}s wall")


def load_catalogue(root: str, shard_limit: int | None) -> ASELMDBCatalogue:
    """Merge the cached catalogues of (the first *shard_limit*) shards."""
    shards: list[Path] = resolve_aselmdb_files(root)
    if shard_limit is not None:
        shards = shards[:shard_limit]
    reader: ASELMDBReader = ASELMDBReader(shards)
    return reader.catalogue(cache=True, progress=print)


def apply_filters(
    catalogue: ASELMDBCatalogue,
    elements: typing.Sequence[int],
    charges: typing.Sequence[int] | None,
    spins: typing.Sequence[int] | None,
    max_atoms: int | None,
) -> ASELMDBCatalogue:
    """The subset a config with these filters would train on."""
    keep: np.ndarray = catalogue.element_mask(allowed=elements)
    keep &= np.isfinite(catalogue.energy)
    if charges is not None:
        keep &= np.isin(catalogue.charge, np.asarray(charges, dtype=np.int16))
    if spins is not None:
        keep &= np.isin(catalogue.spin, np.asarray(spins, dtype=np.int16))
    if max_atoms is not None:
        keep &= catalogue.num_atoms <= max_atoms
    return catalogue.select(keep)


def average_num_neighbors(
    root: str,
    subset: ASELMDBCatalogue,
    cutoff: float,
    sample: int,
    seed: int,
) -> float:
    """Mean degree at *cutoff* over a random sample of the subset.

    Uses the same definition as
    :func:`goal.ml.data.statistics.compute_avg_num_neighbors` — mean of
    ``num_edges / num_atoms`` per structure — but over a sample, since
    building neighbour lists for a whole OMol25 split to average one
    scalar is not a good trade.
    """
    from ase.neighborlist import neighbor_list

    reader: ASELMDBReader = ASELMDBReader(root)
    # The subset holds (shard, row) addresses; the reader turns those back
    # into the flat positions ``read_many`` walks in shard order.
    chosen: np.ndarray = reader.positions_of(subset)
    if chosen.shape[0] > sample:
        chosen = np.sort(np.random.default_rng(seed).choice(chosen, size=sample, replace=False))

    degrees: list[float] = []
    for _, row in reader.read_many(chosen):
        numbers: np.ndarray = np.asarray(row["numbers"], dtype=int)
        positions: np.ndarray = np.asarray(row["positions"], dtype=float)
        first: np.ndarray = neighbor_list("i", _bare_atoms(numbers, positions), cutoff)
        degrees.append(first.shape[0] / max(1, numbers.shape[0]))
    reader.close()
    return float(np.mean(degrees)) if degrees else 1.0


def _bare_atoms(numbers: np.ndarray, positions: np.ndarray) -> typing.Any:
    """A non-periodic ``ase.Atoms`` — enough for a neighbour count."""
    from ase import Atoms

    return Atoms(numbers=numbers, positions=positions, pbc=False)


def report(
    subset: ASELMDBCatalogue,
    full_size: int,
    elements: typing.Sequence[int],
    cutoff: float,
    avg_neighbors: float | None,
) -> None:
    """Print the subset's composition and the config block it implies."""
    print()
    print("=" * 72)
    print(
        f"filtered subset: {len(subset):,} / {full_size:,} rows "
        f"({len(subset) / max(1, full_size):.1%})"
    )
    if len(subset) == 0:
        return

    counts: dict[str, int] = {}
    for tag in subset.data_id:
        counts[str(tag)] = counts.get(str(tag), 0) + 1
    print(
        "  by sub-corpus  : "
        + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    )
    sizes: np.ndarray = subset.num_atoms
    print(
        f"  atoms/structure: min {sizes.min()}  median {int(np.median(sizes))}  "
        f"p95 {int(np.percentile(sizes, 95))}  max {sizes.max()}"
    )
    print(f"  distinct formulas: {len(np.unique(subset.formula_codes)):,}")
    present: list[int] = subset.unique_elements()
    print("  elements present : " + " ".join(f"{chemical_symbols[z]}({z})" for z in present))
    missing: list[int] = [z for z in elements if z not in present]
    if missing:
        print(
            "  WARNING: declared but absent: "
            + " ".join(f"{chemical_symbols[z]}({z})" for z in missing)
        )

    references: dict[int, float] = linear_reference_energies(subset)
    scale: float = residual_energy_scale(subset, references)

    print()
    print("--- paste into the model.backbone section -------------------------")
    print(f"    elements: [{', '.join(str(z) for z in present)}]")
    print("    atomic_energies:")
    print("      mode: provided")
    print("      values:")
    for z in present:
        print(f"        {z}: {references[z]:.6f} # {chemical_symbols[z]}")
    print("      compute_from_dataset: false")
    print(f"    scale: {scale:.8f}")
    if avg_neighbors is not None:
        print(f"    avg_num_neighbors: {avg_neighbors:.4f} # at cutoff {cutoff} A")
    print("-" * 68)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default="data/OMol25/train_4M", help="split directory")
    parser.add_argument("--workers", type=int, default=8, help="parallel indexing workers")
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="skip the scan; use the catalogues already cached",
    )
    parser.add_argument(
        "--stats-shards",
        type=int,
        default=None,
        help="use only the first N shards for the statistics (default: all)",
    )
    parser.add_argument(
        "--elements",
        type=int,
        nargs="+",
        default=list(DEFAULT_ELEMENTS),
        help="allowed atomic numbers",
    )
    parser.add_argument("--charges", type=int, nargs="*", default=[0], help="allowed charges")
    parser.add_argument("--spins", type=int, nargs="*", default=[1], help="allowed spins")
    parser.add_argument("--max-atoms", type=int, default=128, help="atom-count ceiling")
    parser.add_argument("--cutoff", type=float, default=6.0, help="neighbour cutoff, Angstrom")
    parser.add_argument(
        "--neighbor-sample",
        type=int,
        default=4000,
        help="structures used to estimate avg_num_neighbors (0 to skip)",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    args: argparse.Namespace = parser.parse_args()

    if not args.no_index:
        build_catalogues(args.root, args.workers)

    catalogue: ASELMDBCatalogue = load_catalogue(args.root, args.stats_shards)
    subset: ASELMDBCatalogue = apply_filters(
        catalogue,
        args.elements,
        args.charges or None,
        args.spins or None,
        args.max_atoms,
    )
    neighbors: float | None = None
    if args.neighbor_sample > 0 and len(subset) > 0:
        started: float = time.perf_counter()
        neighbors = average_num_neighbors(
            args.root, subset, args.cutoff, args.neighbor_sample, args.seed
        )
        print(
            f"[prepare] avg_num_neighbors at {args.cutoff} A = {neighbors:.4f} "
            f"({min(args.neighbor_sample, len(subset)):,} structures, "
            f"{time.perf_counter() - started:.1f}s)"
        )
    report(subset, len(catalogue), args.elements, args.cutoff, neighbors)


if __name__ == "__main__":
    main()
