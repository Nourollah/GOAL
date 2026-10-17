"""ASE-LMDB dataset — reads FairChem ``.aselmdb`` shards (OMol25, OMat24, ODAC23).

FairChem distributes its large datasets as *many* LMDB files written
through ASE's own database layer (``ase.db`` with the ``aselmdb`` backend).
The layout is **not** the ``lmdb`` dataset's layout (pickled GOAL dicts
keyed by ``"0"``, ``"1"``, …) — hence a separate loader rather than a flag
on :class:`~goal.ml.data.datasets.lmdb.LMDBDataset`:

* each key is an ASE database row id (``b"1"``, ``b"2"``, …, 1-based),
  plus the bookkeeping keys ``nextid`` / ``metadata`` / ``deleted_ids``;
* each value is ``zlib.compress(json.dumps(<ase.db row dict>))``, where
  arrays are ASE's ``{"__ndarray__": [shape, dtype, flat]}`` encoding;
* ``row["data"]`` carries the dataset-specific extras.  For OMol25 that
  is ``charge``, ``spin``, ``composition``, ``data_id`` (which sub-corpus
  the structure came from), ``source``, plus per-atom population
  analyses and SCF diagnostics.

Nothing here needs ``fairchem-core`` installed: decoding goes through
``ase.io.jsonio`` + ``ase.db.row.AtomsRow``, which are the same classes
that wrote the files.

Two layers live in this module:

:class:`ASELMDBReader`
    Low-level, framework-free random access over one or many shards.
    Also builds an :class:`ASELMDBCatalogue` — a compact per-row summary
    (element bitmask, atom count, energy, charge, spin, formula, corpus)
    used both for **filtering at dataset construction** and for
    **searching the dataset from a notebook**.

:class:`ASELMDBDataset`
    ``dataset_type: aselmdb`` — a :class:`BaseAtomicDataset` that filters
    a shard down to a trainable subset (elements, charge, spin, size,
    sub-corpus, sample cap) and yields ``AtomicGraph`` objects.

Because a whole OMol25 split is far larger than any single run needs, the
filter arguments are the primary interface; see
``src/goal/configs/ml/nf_simurgh_omol25.yaml`` for a worked example.
"""

from __future__ import annotations

import dataclasses
import hashlib
import typing
import zlib
from pathlib import Path

import numpy as np
import torch

from goal.ml.data.datasets.base import BaseAtomicDataset
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import DATASET_REGISTRY

#: LMDB keys that are database bookkeeping rather than structures.
RESERVED_KEYS: frozenset[bytes] = frozenset({b"nextid", b"metadata", b"deleted_ids"})

#: File suffix used by every FairChem ASE-LMDB shard.
ASELMDB_SUFFIX: str = ".aselmdb"

#: Bytes per row in the element bitmask — 16 bytes = atomic numbers 0…127,
#: which covers the whole periodic table as distributed by FairChem.
Z_MASK_BYTES: int = 16

#: Highest atomic number representable in :data:`Z_MASK_BYTES`.
Z_MASK_MAX: int = Z_MASK_BYTES * 8 - 1


# ---------------------------------------------------------------------------
# Shard discovery and row decoding
# ---------------------------------------------------------------------------


def resolve_aselmdb_files(root: str | Path) -> list[Path]:
    """Return the ``.aselmdb`` shards named by *root*, sorted.

    *root* may be a single shard, a directory of shards, or an iterable of
    either.  Sorting is by file name so the flat index a
    :class:`ASELMDBReader` builds is reproducible across machines.
    """
    if isinstance(root, (list, tuple)):
        files: list[Path] = []
        for item in root:
            files.extend(resolve_aselmdb_files(item))
        return files

    path: Path = Path(root)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ASELMDB_SUFFIX)
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"No ASE-LMDB shard at {path}")


def decode_row(raw: bytes) -> dict[str, typing.Any]:
    """Decode one stored value into an ``ase.db`` row dict.

    The inverse of what ASE's LMDB backend wrote: zlib, then JSON with
    ASE's ``__ndarray__`` object hook (so ``numbers``, ``positions``,
    ``forces``, ``cell`` and ``pbc`` come back as ``numpy`` arrays).
    ``always_array=False`` keeps plain JSON lists as lists, which leaves
    the string-valued extras (``warnings``) readable.
    """
    from ase.io.jsonio import decode

    return decode(zlib.decompress(raw).decode(), always_array=False)


def row_to_atoms(row: dict[str, typing.Any]) -> typing.Any:
    """Build an ``ase.Atoms`` from a decoded row.

    Energy and forces arrive attached as a ``SinglePointCalculator`` (ASE
    does that in :meth:`ase.db.row.AtomsRow.toatoms`), so the standard
    :func:`goal.ml.data.keys.extract_ase_labels` calculator fallback finds
    them.  On top of that, every *scalar* entry of ``row["data"]`` is
    copied into ``atoms.info`` — that is what makes
    ``data.energy_key: nl_energy`` (or any other scalar the corpus
    records) work through the usual ``key_mapping`` machinery — and the
    untouched dict is kept at ``atoms.info["data"]`` for callers that want
    the arrays too (Löwdin/Mulliken/NBO charges, HOMO/LUMO, …).
    """
    from ase.db.row import AtomsRow

    atoms: typing.Any = AtomsRow(row).toatoms()
    extras: dict[str, typing.Any] = dict(row.get("data") or {})
    if extras:
        atoms.info["data"] = extras
        for key, value in extras.items():
            if isinstance(value, (bool, int, float, str)):
                atoms.info.setdefault(key, value)
    return atoms


def _hill_formula(numbers: np.ndarray) -> str:
    """Hill-system chemical formula for an array of atomic numbers.

    Used instead of OMol25's own ``data["composition"]`` so that formulas
    are comparable across FairChem corpora *and* against RDKit's
    ``CalcMolFormula`` (OMol25 writes ``C19H18N4O5S1``, Hill and RDKit
    both write ``C19H18N4O5S``).
    """
    from ase.symbols import Symbols

    return Symbols(np.asarray(numbers, dtype=int)).get_chemical_formula(mode="hill")


def parse_formula(text: str) -> dict[int, int]:
    """Parse a chemical formula string into ``{atomic_number: count}``.

    Handles the Hill strings this module produces and the ``C19H18N4O5S1``
    style OMol25 stores, i.e. element symbols each followed by an optional
    count.  Anything else (charge suffixes, brackets, hydrates) raises.
    """
    import re

    from ase.data import atomic_numbers

    counts: dict[int, int] = {}
    position: int = 0
    pattern: typing.Any = re.compile(r"([A-Z][a-z]?)(\d*)")
    while position < len(text):
        match: typing.Any = pattern.match(text, position)
        if match is None:
            raise ValueError(f"Cannot parse formula {text!r} at position {position}.")
        symbol: str = match.group(1)
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown element {symbol!r} in formula {text!r}.")
        number: int = int(atomic_numbers[symbol])
        counts[number] = counts.get(number, 0) + int(match.group(2) or 1)
        position = match.end()
    return counts


# ---------------------------------------------------------------------------
# Catalogue — the compact per-row summary
# ---------------------------------------------------------------------------


def _encode_categorical(values: typing.Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Factorise a list of strings into ``(codes, categories)``."""
    categories: np.ndarray
    codes: np.ndarray
    categories, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    return codes.astype(np.int32), categories


@dataclasses.dataclass(frozen=True)
class ASELMDBCatalogue:
    """Per-row summary of one or more ``.aselmdb`` shards.

    Everything needed to *choose* structures without decoding them again.
    Roughly 40 bytes per row once the repeated strings are factorised, so
    a whole OMol25 4M split summarises to well under a gigabyte and each
    shard's summary caches to a few megabytes on disk.

    Attributes
    ----------
    files : ndarray of str
        Shard paths, indexed by :attr:`file_index`.
    file_index, row_id : ndarray
        Where each entry lives — shard, then LMDB key (an ``ase.db`` row
        id, 1-based).  Together they are the address
        :meth:`ASELMDBReader.read` resolves.
    num_atoms, energy, charge, spin : ndarray
        Per-row scalars.  ``energy`` is ``nan`` when the row carries no
        energy; ``charge`` / ``spin`` are ``-999`` when the corpus does
        not record them (they are OMol25 ``data`` fields, not ASE ones).
    z_mask : ndarray of uint8, shape (M, 16)
        Bitmask of the elements present in each structure — bit ``z & 7``
        of byte ``z >> 3``.  Element filtering is then a handful of
        vectorised ``&`` operations over the whole catalogue instead of a
        per-row Python set.
    formula_codes, formula_categories : ndarray
        Factorised Hill formula.  Use :attr:`formula`.
    data_id_codes, data_id_categories : ndarray
        Factorised sub-corpus tag (OMol25: ``biomolecules``, ``elytes``,
        ``metal_complexes``, ``reactivity``, ``spice``, …).  Use
        :attr:`data_id`.
    source : ndarray of str or None
        The provenance string, when the catalogue was built with
        ``include_source=True``.  Long and rarely needed, so it is off by
        default.
    """

    files: np.ndarray
    file_index: np.ndarray
    row_id: np.ndarray
    num_atoms: np.ndarray
    energy: np.ndarray
    charge: np.ndarray
    spin: np.ndarray
    z_mask: np.ndarray
    formula_codes: np.ndarray
    formula_categories: np.ndarray
    data_id_codes: np.ndarray
    data_id_categories: np.ndarray
    source: np.ndarray | None = None

    #: Sentinel written into :attr:`charge` / :attr:`spin` when a corpus
    #: does not record them, chosen to be outside any physical range.
    MISSING: typing.ClassVar[int] = -999

    def __len__(self) -> int:
        return int(self.row_id.shape[0])

    @property
    def formula(self) -> np.ndarray:
        """Hill formula per row, expanded from the factorised codes."""
        return self.formula_categories[self.formula_codes]

    @property
    def data_id(self) -> np.ndarray:
        """Sub-corpus tag per row, expanded from the factorised codes."""
        return self.data_id_categories[self.data_id_codes]

    # -- selection ----------------------------------------------------

    def select(self, index: np.ndarray) -> ASELMDBCatalogue:
        """A catalogue holding only the rows named by *index*.

        *index* may be a boolean mask or an integer index array; the
        category tables are shared rather than rebuilt, so slicing is
        cheap and repeatable.
        """

        def take(array: np.ndarray) -> np.ndarray:
            return array[index]

        return dataclasses.replace(
            self,
            file_index=take(self.file_index),
            row_id=take(self.row_id),
            num_atoms=take(self.num_atoms),
            energy=take(self.energy),
            charge=take(self.charge),
            spin=take(self.spin),
            z_mask=self.z_mask[index],
            formula_codes=take(self.formula_codes),
            data_id_codes=take(self.data_id_codes),
            source=None if self.source is None else take(self.source),
        )

    def elements(self, position: int) -> list[int]:
        """Atomic numbers present in row *position*, from its bitmask."""
        bits: np.ndarray = np.unpackbits(self.z_mask[position], bitorder="little")
        return [int(z) for z in np.flatnonzero(bits)]

    def element_mask(
        self,
        allowed: typing.Iterable[int] | None = None,
        excluded: typing.Iterable[int] | None = None,
    ) -> np.ndarray:
        """Boolean mask over rows selecting structures by chemistry.

        A row passes when every element it contains is in *allowed* (when
        given) and none is in *excluded* (when given).  Both are evaluated
        on :attr:`z_mask`, so cost is independent of structure size.
        """
        keep: np.ndarray = np.ones(len(self), dtype=bool)
        if allowed is not None:
            forbidden: np.ndarray = _z_mask_bytes(
                z for z in range(Z_MASK_MAX + 1) if z not in {int(a) for a in allowed}
            )
            keep &= ~(self.z_mask & forbidden).any(axis=1)
        if excluded is not None:
            keep &= ~(self.z_mask & _z_mask_bytes(excluded)).any(axis=1)
        return keep

    def unique_elements(self) -> list[int]:
        """Sorted atomic numbers appearing anywhere in the catalogue."""
        if len(self) == 0:
            return []
        union: np.ndarray = np.bitwise_or.reduce(self.z_mask, axis=0)
        return [int(z) for z in np.flatnonzero(np.unpackbits(union, bitorder="little"))]

    def concat(self, other: ASELMDBCatalogue) -> ASELMDBCatalogue:
        """Merge two catalogues, re-basing *other*'s shard indices.

        Merging a whole split shard by shard this way is quadratic in the
        number of shards — every step re-maps every code merged so far.
        Use :func:`concat_catalogues` for more than a handful.
        """
        return concat_catalogues([self, other])

    # -- persistence --------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the catalogue to a compressed ``.npz``."""
        arrays: dict[str, np.ndarray] = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if getattr(self, field.name) is not None
        }
        target: Path = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target then rename: a catalogue interrupted
        # half-way through (a killed job, a full disk) must not be picked
        # up as valid on the next run.
        staging: Path = target.with_suffix(target.suffix + ".partial")
        with staging.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        staging.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> ASELMDBCatalogue:
        """Read a catalogue written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            fields: dict[str, np.ndarray] = {key: data[key] for key in data.files}
        return cls(**fields)


def _z_mask_bytes(elements: typing.Iterable[int]) -> np.ndarray:
    """Pack atomic numbers into the 16-byte bitmask layout."""
    mask: np.ndarray = np.zeros(Z_MASK_BYTES, dtype=np.uint8)
    for z in elements:
        z_int: int = int(z)
        if not 0 <= z_int <= Z_MASK_MAX:
            raise ValueError(f"Atomic number {z_int} outside the supported range 0…{Z_MASK_MAX}.")
        mask[z_int >> 3] |= np.uint8(1 << (z_int & 7))
    return mask


def _merge_categorical(
    columns: typing.Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate factorised string columns onto one shared table.

    Each ``(codes, categories)`` pair is re-based by looking its own
    category table up in the merged one — a single ``searchsorted``
    per input rather than a dict lookup per row, which is what keeps
    merging 80 shards of 50k rows off the critical path.
    """
    categories: np.ndarray = np.unique(np.concatenate([pair[1] for pair in columns]))
    codes: list[np.ndarray] = []
    for pair in columns:
        # ``categories`` is sorted and a superset of ``pair[1]``, so
        # searchsorted is an exact old-code → new-code table.
        table: np.ndarray = np.searchsorted(categories, pair[1]).astype(np.int32)
        codes.append(table[pair[0]] if pair[0].shape[0] else np.zeros(0, dtype=np.int32))
    return np.concatenate(codes), categories


def concat_catalogues(
    catalogues: typing.Sequence[ASELMDBCatalogue],
) -> ASELMDBCatalogue:
    """Merge many catalogues in a single pass, re-basing shard indices.

    ``source`` survives only if every input carries it — a merged
    catalogue must not claim provenance it only has for some rows.
    """
    if not catalogues:
        raise ValueError("concat_catalogues() needs at least one catalogue.")
    if len(catalogues) == 1:
        return catalogues[0]

    offsets: np.ndarray = np.cumsum([0] + [c.files.shape[0] for c in catalogues[:-1]])

    def stack(name: str) -> np.ndarray:
        return np.concatenate([getattr(c, name) for c in catalogues])

    formula_codes, formula_categories = _merge_categorical(
        [(c.formula_codes, c.formula_categories) for c in catalogues]
    )
    data_id_codes, data_id_categories = _merge_categorical(
        [(c.data_id_codes, c.data_id_categories) for c in catalogues]
    )
    return ASELMDBCatalogue(
        files=stack("files"),
        file_index=np.concatenate(
            [c.file_index + offset for c, offset in zip(catalogues, offsets)]
        ),
        row_id=stack("row_id"),
        num_atoms=stack("num_atoms"),
        energy=stack("energy"),
        charge=stack("charge"),
        spin=stack("spin"),
        z_mask=stack("z_mask"),
        formula_codes=formula_codes,
        formula_categories=formula_categories,
        data_id_codes=data_id_codes,
        data_id_categories=data_id_categories,
        source=stack("source") if all(c.source is not None for c in catalogues) else None,
    )


# ---------------------------------------------------------------------------
# Shared LMDB environments
# ---------------------------------------------------------------------------
#
# LMDB refuses to open one file twice in a single process ("The
# environment ... is already open in this process"), and two readers over
# the same shard is the normal case, not an edge case: a notebook
# browsing a shard while a dataset trains on it, a DataModule whose
# val_dir equals its train_dir, a stats pass alongside a loader.  So
# environments are shared per (process, file) and reference-counted:
# whoever asks second gets the same handle, and the handle closes when
# the last holder lets go.
#
# The pid is part of the key because a ``fork``ed child must not touch
# handles created by its parent — it opens its own instead.

_ENVIRONMENTS: dict[tuple[int, str], typing.Any] = {}
_ENVIRONMENT_REFS: dict[tuple[int, str], int] = {}


def _environment_key(path: Path) -> tuple[int, str]:
    """Cache key for a shard — resolved, so aliased paths share a handle."""
    import os

    return os.getpid(), str(path.resolve())


def acquire_environment(path: str | Path) -> typing.Any:
    """Open (or join) the shared read-only LMDB environment for *path*.

    Every call must be paired with :func:`release_environment`.
    """
    key: tuple[int, str] = _environment_key(Path(path))
    env: typing.Any = _ENVIRONMENTS.get(key)
    if env is None:
        import lmdb

        env = lmdb.open(
            key[1],
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        _ENVIRONMENTS[key] = env
        _ENVIRONMENT_REFS[key] = 0
    _ENVIRONMENT_REFS[key] += 1
    return env


def release_environment(path: str | Path) -> None:
    """Drop one reference, closing the environment at zero."""
    key: tuple[int, str] = _environment_key(Path(path))
    if key not in _ENVIRONMENTS:
        return
    _ENVIRONMENT_REFS[key] -= 1
    if _ENVIRONMENT_REFS[key] <= 0:
        _ENVIRONMENT_REFS.pop(key, None)
        _ENVIRONMENTS.pop(key).close()


def _forget_inherited_environments() -> None:
    """Forget handles inherited from another process after a fork.

    They belong to the parent — closing them here would pull the mapping
    out from under it — so the child drops the bookkeeping and opens its
    own.
    """
    import os

    pid: int = os.getpid()
    for key in [k for k in _ENVIRONMENTS if k[0] != pid]:
        _ENVIRONMENTS.pop(key, None)
        _ENVIRONMENT_REFS.pop(key, None)


# ---------------------------------------------------------------------------
# Reader — low-level random access
# ---------------------------------------------------------------------------


class ASELMDBReader:
    """Random access over one or many ``.aselmdb`` shards.

    Independent of the training stack: nothing here touches
    ``AtomicGraph``, so a notebook can use it to browse a corpus, and
    :class:`ASELMDBDataset` uses it to feed the model.

    Rows are addressed by a flat position ``0 … len(reader) - 1`` running
    over the shards in sorted file-name order.  That address is stable for
    a given set of files, which is what lets a catalogue be cached.

    Parameters
    ----------
    root : path or sequence of paths
        A shard, a directory of shards, or a list of either.
    cache_dir : path, optional
        Where :meth:`catalogue` writes its per-shard summaries.  Default
        is beside each shard.  Point this at a scratch directory when the
        dataset lives on a read-only mount.

    Notes
    -----
    LMDB environments are opened lazily and re-opened after a ``fork``
    (the handle is tagged with the pid that created it), so a reader
    built in the parent process works unchanged inside DataLoader
    workers.
    """

    def __init__(
        self,
        root: str | Path | typing.Sequence[str | Path],
        cache_dir: str | Path | None = None,
    ) -> None:
        # Absolute from here on: a catalogue records the path it was scanned
        # under, and a relative one ("data/OMol25/...") is meaningless to
        # anything running from a different working directory — a notebook in
        # notebooks/, say.  Absolute paths make a cached catalogue portable
        # across the processes that read it.
        self.files: list[Path] = [f.resolve() for f in resolve_aselmdb_files(root)]
        if not self.files:
            raise FileNotFoundError(f"No {ASELMDB_SUFFIX} shards found under {root!r}")
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self._envs: list[typing.Any] = [None] * len(self.files)
        self._acquired: set[int] = set()
        self._owner_pid: int = -1
        self._address_lookup: dict[tuple[int, int], int] | None = None
        self._file_index: np.ndarray
        self._row_id: np.ndarray
        self._file_index, self._row_id = self._build_index()

    # -- environment handling -----------------------------------------

    def _env(self, file_position: int) -> typing.Any:
        """The LMDB environment for one shard, opened on first use."""
        import os

        pid: int = os.getpid()
        if pid != self._owner_pid:
            # Inherited across a fork: the parent's handles must not be
            # touched (LMDB's docs are explicit about this), so drop them
            # and reopen in this process.
            _forget_inherited_environments()
            self._envs = [None] * len(self.files)
            self._acquired = set()
            self._owner_pid = pid

        env: typing.Any = self._envs[file_position]
        if env is None:
            env = acquire_environment(self.files[file_position])
            self._envs[file_position] = env
            self._acquired.add(file_position)
        return env

    def close(self) -> None:
        """Release this reader's share of every environment it opened.

        The handle itself closes only when no other reader is using it —
        see the shared-environment note above.
        """
        import os

        if os.getpid() != self._owner_pid:
            return
        for position in sorted(self._acquired):
            release_environment(self.files[position])
            self._envs[position] = None
        self._acquired.clear()

    def __del__(self) -> None:  # pragma: no cover — interpreter teardown
        try:
            self.close()
        except Exception:  # noqa: BLE001 — nothing useful to do while dying
            pass

    # -- index --------------------------------------------------------

    def _shard_row_ids(self, file_position: int) -> np.ndarray:
        """Row ids stored in one shard.

        ASE's LMDB backend keeps a ``nextid`` counter and a
        ``deleted_ids`` list, so the id set is normally just
        ``1 … nextid-1`` minus the deletions — an O(1) read rather than a
        scan of every key.  The derived count is checked against the
        environment's own entry count and a full key scan is used if they
        disagree (a hand-edited or otherwise unusual shard).
        """
        env: typing.Any = self._env(file_position)
        with env.begin(write=False) as txn:
            next_id_raw: bytes | None = txn.get(b"nextid")
            deleted_raw: bytes | None = txn.get(b"deleted_ids")
            n_entries: int = int(txn.stat()["entries"])

            if next_id_raw is not None:
                next_id: int = int(decode_row(next_id_raw))
                deleted: list[int] = (
                    [int(i) for i in decode_row(deleted_raw)] if deleted_raw is not None else []
                )
                candidate: np.ndarray = np.setdiff1d(
                    np.arange(1, next_id, dtype=np.int64),
                    np.asarray(deleted, dtype=np.int64),
                )
                # Bookkeeping keys present in this shard, counted so the
                # comparison below is against structures only.
                overhead: int = sum(1 for key in RESERVED_KEYS if txn.get(key) is not None)
                if candidate.shape[0] == n_entries - overhead:
                    return candidate

            ids: list[int] = [
                int(key)
                for key in txn.cursor().iternext(keys=True, values=False)
                if key not in RESERVED_KEYS
            ]
        return np.sort(np.asarray(ids, dtype=np.int64))

    def _build_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Flat ``(file_index, row_id)`` address arrays over every shard."""
        file_index: list[np.ndarray] = []
        row_id: list[np.ndarray] = []
        for position in range(len(self.files)):
            ids: np.ndarray = self._shard_row_ids(position)
            row_id.append(ids)
            file_index.append(np.full(ids.shape[0], position, dtype=np.int32))
        if not row_id:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
        return np.concatenate(file_index), np.concatenate(row_id)

    def __len__(self) -> int:
        return int(self._row_id.shape[0])

    def address(self, position: int) -> tuple[Path, int]:
        """``(shard path, row id)`` for a flat position."""
        return self.files[int(self._file_index[position])], int(self._row_id[position])

    def position_of(self, path: str | Path, row_id: int) -> int:
        """The flat position of a ``(shard, row id)`` address.

        The inverse of :meth:`address`.  A catalogue stores addresses, not
        positions — so that it survives being cached, merged and sliced —
        and this is how those addresses are turned back into something
        :meth:`read` accepts.  The lookup table is built on first use.
        """
        if self._address_lookup is None:
            self._address_lookup = {
                (int(file_position), int(row)): index
                for index, (file_position, row) in enumerate(zip(self._file_index, self._row_id))
            }
        file_position: int = self._file_position(path)
        key: tuple[int, int] = (file_position, int(row_id))
        if key not in self._address_lookup:
            raise KeyError(f"Row {row_id} is not in {self.files[file_position]}.")
        return self._address_lookup[key]

    def _file_position(self, path: str | Path) -> int:
        """Index of *path* among this reader's shards.

        Matched on the resolved path, then — if that fails — on the file
        name.  The fallback is what makes an *older* cached catalogue still
        addressable: one written before paths were stored absolute records
        e.g. ``data/OMol25/train_4M/data0005.aselmdb``, which resolves
        against whatever directory the reading process happens to sit in and
        so points nowhere.  Shard names are unique inside a split, so
        matching on the name is unambiguous there; when it is not, the
        caller is told rather than guessed at.
        """
        target: Path = Path(path)
        resolved: Path = target.resolve()
        for position, shard in enumerate(self.files):
            if shard == resolved:
                return position
        by_name: list[int] = [i for i, shard in enumerate(self.files) if shard.name == target.name]
        if len(by_name) == 1:
            return by_name[0]
        if by_name:
            raise KeyError(
                f"{path} matches {len(by_name)} of this reader's shards by name "
                f"and none by path; open a reader on the split it came from."
            )
        raise KeyError(f"{path} is not one of this reader's shards.")

    def positions_of(self, catalogue: ASELMDBCatalogue) -> np.ndarray:
        """Flat positions for every row of *catalogue*, in its order.

        Lets a catalogue that was filtered, merged or loaded from cache be
        turned straight into reader positions — the bridge between
        "which rows do I want" and "how do I read them".
        """
        return np.asarray(
            [
                self.position_of(str(catalogue.files[int(file_position)]), int(row))
                for file_position, row in zip(catalogue.file_index, catalogue.row_id)
            ],
            dtype=np.int64,
        )

    # -- reading ------------------------------------------------------

    def read(self, position: int) -> dict[str, typing.Any]:
        """The decoded ``ase.db`` row dict at a flat position."""
        if not -len(self) <= position < len(self):
            raise IndexError(f"Position {position} out of range for {len(self)} rows.")
        index: int = int(position) % len(self)
        env: typing.Any = self._env(int(self._file_index[index]))
        key: bytes = str(int(self._row_id[index])).encode()
        with env.begin(write=False) as txn:
            raw: bytes | None = txn.get(key)
        if raw is None:  # pragma: no cover — corrupt shard
            path, row = self.address(index)
            raise KeyError(f"Row {row} missing from {path}")
        return decode_row(raw)

    def read_row(self, path: str | Path, row_id: int) -> dict[str, typing.Any]:
        """Read by explicit ``(shard, row id)`` address."""
        file_position: int = self._file_position(path)
        env: typing.Any = self._env(file_position)
        with env.begin(write=False) as txn:
            raw: bytes | None = txn.get(str(int(row_id)).encode())
        if raw is None:
            raise KeyError(f"Row {row_id} missing from {self.files[file_position]}")
        return decode_row(raw)

    def read_many(
        self,
        positions: typing.Sequence[int] | np.ndarray,
    ) -> typing.Iterator[tuple[int, dict[str, typing.Any]]]:
        """Yield ``(position, row)`` for many positions, shard by shard.

        One LMDB transaction per shard rather than per row, which is
        roughly a 1.5x speed-up over calling :meth:`read` in a loop and
        is what makes a full-split :meth:`scan` bearable.  Positions are
        visited in shard order, hence the position is yielded alongside
        the row.
        """
        index: np.ndarray = np.asarray(positions, dtype=np.int64)
        files: np.ndarray = self._file_index[index]
        for file_position in np.unique(files):
            env: typing.Any = self._env(int(file_position))
            with env.begin(write=False) as txn:
                for position in index[files == file_position]:
                    key: bytes = str(int(self._row_id[position])).encode()
                    raw: bytes | None = txn.get(key)
                    if raw is None:  # pragma: no cover — corrupt shard
                        path, row_id = self.address(int(position))
                        raise KeyError(f"Row {row_id} missing from {path}")
                    yield int(position), decode_row(raw)

    def atoms(self, position: int) -> typing.Any:
        """The structure at a flat position, as ``ase.Atoms``.

        Energy and forces come attached as a ``SinglePointCalculator``;
        the corpus extras live in ``atoms.info`` (see :func:`row_to_atoms`).
        """
        return row_to_atoms(self.read(position))

    # -- catalogue ----------------------------------------------------

    def scan(
        self,
        include_source: bool = False,
        positions: typing.Sequence[int] | np.ndarray | None = None,
        progress: typing.Callable[[str], None] | None = None,
    ) -> ASELMDBCatalogue:
        """Decode every row once and summarise it into a catalogue.

        This is the expensive operation in this module — decoding the
        JSON dominates, at roughly 6k rows/second, so ~8 seconds per
        50k-row OMol25 shard and ~10 minutes for a whole 4M split.
        :meth:`catalogue` caches the result per shard so it happens once,
        and ``scripts/omol25_prepare.py`` does all the shards in parallel
        up front.

        Parameters
        ----------
        include_source : bool
            Also keep ``data["source"]`` (the upstream provenance path).
            Useful for searching, costly in memory — a few hundred bytes
            per row.
        positions : sequence of int, optional
            Summarise only these flat positions (e.g. a subsample).
            Default is every row.
        progress : callable, optional
            Called with a short status line after each shard.
        """
        index: np.ndarray = (
            np.arange(len(self), dtype=np.int64)
            if positions is None
            else np.asarray(positions, dtype=np.int64)
        )
        total: int = int(index.shape[0])

        num_atoms: np.ndarray = np.zeros(total, dtype=np.int32)
        energy: np.ndarray = np.full(total, np.nan, dtype=np.float64)
        charge: np.ndarray = np.full(total, ASELMDBCatalogue.MISSING, dtype=np.int16)
        spin: np.ndarray = np.full(total, ASELMDBCatalogue.MISSING, dtype=np.int16)
        z_mask: np.ndarray = np.zeros((total, Z_MASK_BYTES), dtype=np.uint8)
        # Written by slot rather than appended: ``read_many`` walks shards
        # in file order, which need not be the order of *positions*.
        formulas: list[str] = [""] * total
        data_ids: list[str] = [""] * total
        sources: list[str] = [""] * total

        slot_of: dict[int, int] = {int(position): slot for slot, position in enumerate(index)}
        last_file: int = -1
        for position, row in self.read_many(index):
            slot: int = slot_of[position]
            file_position: int = int(self._file_index[position])
            if progress is not None and file_position != last_file:
                progress(
                    f"[aselmdb] indexing {self.files[file_position].name} "
                    f"({file_position + 1}/{len(self.files)})"
                )
                last_file = file_position

            numbers: np.ndarray = np.asarray(row["numbers"], dtype=np.int64)
            num_atoms[slot] = numbers.shape[0]
            if row.get("energy") is not None:
                energy[slot] = float(row["energy"])
            for z in np.unique(numbers):
                z_int: int = int(z)
                if z_int > Z_MASK_MAX:  # pragma: no cover — no such element ships
                    raise ValueError(f"Atomic number {z_int} exceeds {Z_MASK_MAX}.")
                z_mask[slot, z_int >> 3] |= np.uint8(1 << (z_int & 7))

            extras: dict[str, typing.Any] = row.get("data") or {}
            if extras.get("charge") is not None:
                charge[slot] = int(extras["charge"])
            if extras.get("spin") is not None:
                spin[slot] = int(extras["spin"])
            formulas[slot] = _hill_formula(numbers)
            data_ids[slot] = str(extras.get("data_id", ""))
            if include_source:
                sources[slot] = str(extras.get("source", ""))

        formula_codes, formula_categories = _encode_categorical(formulas)
        data_id_codes, data_id_categories = _encode_categorical(data_ids)
        return ASELMDBCatalogue(
            files=np.asarray([str(f) for f in self.files], dtype=str),
            file_index=self._file_index[index].astype(np.int32),
            row_id=self._row_id[index].astype(np.int64),
            num_atoms=num_atoms,
            energy=energy,
            charge=charge,
            spin=spin,
            z_mask=z_mask,
            formula_codes=formula_codes,
            formula_categories=formula_categories,
            data_id_codes=data_id_codes,
            data_id_categories=data_id_categories,
            source=np.asarray(sources, dtype=str) if include_source else None,
        )

    def catalogue_path(self, file_position: int, include_source: bool = False) -> Path:
        """Where the cached summary for one shard is written."""
        shard: Path = self.files[file_position]
        suffix: str = ".goal-catalogue-src.npz" if include_source else ".goal-catalogue.npz"
        if self.cache_dir is None:
            return shard.with_name(shard.name + suffix)
        # A user-specified cache directory is shared by shards from
        # anywhere, so the parent path goes into the name via a digest to
        # keep identically-named shards from different splits apart.
        digest: str = hashlib.sha1(str(shard.parent).encode(), usedforsecurity=False).hexdigest()[
            :8
        ]
        return self.cache_dir / f"{shard.name}.{digest}{suffix}"

    def catalogue(
        self,
        cache: bool = True,
        include_source: bool = False,
        progress: typing.Callable[[str], None] | None = None,
    ) -> ASELMDBCatalogue:
        """Catalogue for every shard, cached per shard on disk.

        Each shard is summarised independently and the summaries are
        merged, so adding a shard to a directory costs one shard's scan
        rather than a full re-index.

        Parameters
        ----------
        cache : bool
            Read and write ``.goal-catalogue.npz`` beside each shard (or
            in :attr:`cache_dir`).  ``False`` scans every time and writes
            nothing — the right choice for a read-only mount without a
            scratch directory.
        include_source, progress
            As in :meth:`scan`.
        """
        shards: list[ASELMDBCatalogue] = []
        for file_position in range(len(self.files)):
            path: Path = self.catalogue_path(file_position, include_source)
            shard: ASELMDBCatalogue | None = None
            if cache and path.is_file():
                try:
                    shard = ASELMDBCatalogue.load(path)
                except Exception:  # noqa: BLE001 — a stale/corrupt cache is not fatal
                    shard = None
            if shard is None or len(shard) != int((self._file_index == file_position).sum()):
                positions: np.ndarray = np.flatnonzero(self._file_index == file_position)
                # ``scan`` addresses rows through the whole-reader index,
                # so a single-shard catalogue still records every shard in
                # ``files``; rebase it onto just this one before caching.
                shard = _restrict_to_single_file(
                    self.scan(
                        include_source=include_source,
                        positions=positions,
                        progress=progress,
                    ),
                    file_position,
                )
                if cache:
                    try:
                        shard.save(path)
                    except OSError as error:  # pragma: no cover — read-only mount
                        if progress is not None:
                            progress(f"[aselmdb] could not cache {path.name}: {error}")
            shards.append(shard)
        # One merge over every shard, not 80 pairwise merges: each pairwise
        # step would re-map every code accumulated so far.
        return concat_catalogues(shards)


def _restrict_to_single_file(
    catalogue: ASELMDBCatalogue,
    file_position: int,
) -> ASELMDBCatalogue:
    """Rewrite a catalogue so it names exactly one shard."""
    return dataclasses.replace(
        catalogue,
        files=catalogue.files[file_position : file_position + 1],
        file_index=np.zeros(len(catalogue), dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Dataset-level statistics straight from a catalogue
# ---------------------------------------------------------------------------


def _counts_matrix(catalogue: ASELMDBCatalogue) -> tuple[np.ndarray, list[int]]:
    """Species-count matrix ``C[i, k]`` and its column → element map.

    Built from the *factorised* formulas, so each distinct formula is
    parsed once no matter how many conformers share it.
    """
    parsed: list[dict[int, int]] = [
        parse_formula(str(name)) for name in catalogue.formula_categories
    ]
    columns: list[int] = sorted({z for counts in parsed for z in counts})
    lookup: dict[int, int] = {z: k for k, z in enumerate(columns)}
    per_formula: np.ndarray = np.zeros((len(parsed), len(columns)), dtype=np.float64)
    for row, counts in enumerate(parsed):
        for z, n in counts.items():
            per_formula[row, lookup[z]] = float(n)
    design: np.ndarray = per_formula[catalogue.formula_codes]
    # ``select`` shares the category table with its parent, so the parsed
    # columns can include elements no *selected* row contains.  Those
    # columns are all-zero and would come back from the regression as a
    # meaningless 0.0 reference energy, so drop them.
    present: np.ndarray = np.flatnonzero(design.any(axis=0))
    return design[:, present], [columns[k] for k in present]


def _finite_energy_subset(
    catalogue: ASELMDBCatalogue,
    max_rows: int,
    seed: int,
) -> ASELMDBCatalogue:
    """Rows with a usable energy, capped at *max_rows* (deterministically)."""
    usable: np.ndarray = np.flatnonzero(np.isfinite(catalogue.energy))
    if usable.shape[0] > max_rows:
        usable = np.sort(np.random.default_rng(seed).choice(usable, size=max_rows, replace=False))
    return catalogue.select(usable)


def linear_reference_energies(
    catalogue: ASELMDBCatalogue,
    max_rows: int = 200_000,
    seed: int = 0,
) -> dict[int, float]:
    """Per-element energy baseline ``{Z: e_Z}`` solving ``E ≈ Σ_Z n_Z e_Z``.

    Same regression as :func:`goal.ml.data.statistics.compute_atomic_references`,
    but read off the catalogue's formulas instead of iterating the
    dataset — no geometry decoded, no neighbour lists built.  On an OMol25
    split that turns a multi-hour pass into a few seconds, which is what
    makes ``atomic_energies.mode: provided`` practical there.

    OMol25 energies are raw ωB97M-V total energies (order 10⁴ eV), so
    this baseline is not cosmetic: without it the interaction network is
    asked to fit numbers five orders of magnitude larger than the
    chemistry it is meant to learn.

    Parameters
    ----------
    catalogue : ASELMDBCatalogue
        Catalogue of the rows the model will actually train on — filter it
        first, since the baseline must match the training distribution.
    max_rows : int
        Cap on the number of equations.  Beyond ~10⁵ rows the solution
        stops moving; the cap keeps the design matrix in memory and the
        result is a deterministic random subsample.
    seed : int
        Seed for that subsample.

    Returns
    -------
    dict
        ``{Z: e_Z}`` in the catalogue's energy unit (eV for OMol25), for
        every element present.
    """
    subset: ASELMDBCatalogue = _finite_energy_subset(catalogue, max_rows, seed)
    if len(subset) == 0:
        return {}
    design: np.ndarray
    columns: list[int]
    design, columns = _counts_matrix(subset)
    solution: np.ndarray = np.linalg.lstsq(design, subset.energy, rcond=None)[0]
    return {int(z): float(solution[k]) for k, z in enumerate(columns)}


def residual_energy_scale(
    catalogue: ASELMDBCatalogue,
    references: typing.Mapping[int, float],
    max_rows: int = 200_000,
    seed: int = 0,
) -> float:
    """``std(E − Σ_Z n_Z e_Z) / mean(n_atoms)`` over the catalogue.

    The counterpart to :func:`goal.ml.data.statistics.compute_energy_scale`
    — identical definition, computed from formulas — giving the
    multiplicative gain that puts the model's dimensionless interaction
    output on the physical scale of the energy residual.
    """
    subset: ASELMDBCatalogue = _finite_energy_subset(catalogue, max_rows, seed)
    if len(subset) == 0:
        return 1.0
    design: np.ndarray
    columns: list[int]
    design, columns = _counts_matrix(subset)
    baseline: np.ndarray = design @ np.asarray(
        [float(references.get(int(z), 0.0)) for z in columns], dtype=np.float64
    )
    residual: np.ndarray = subset.energy - baseline
    mean_atoms: float = float(subset.num_atoms.mean())
    deviation: float = float(residual.std(ddof=1)) if residual.shape[0] > 1 else 0.0
    if deviation == 0.0 or mean_atoms == 0.0:
        return 1.0
    return deviation / mean_atoms


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@DATASET_REGISTRY.register("aselmdb")
class ASELMDBDataset(BaseAtomicDataset):
    """``dataset_type: aselmdb`` — FairChem ASE-LMDB shards as ``AtomicGraph``.

    One instance covers **one shard**.  A whole split is loaded the way
    every other multi-file corpus is: point ``data.train_dir`` /
    ``data.val_dir`` at the directories (Mode 4), or ``data.root`` at a
    directory and let ``GOALDataModule`` discover the shards and split
    them numerically (Mode 1) — ``.aselmdb`` is a recognised extension.
    A directory root with no ``{split}.aselmdb`` inside raises
    ``FileNotFoundError`` so that fallback fires instead of every split
    silently loading the same rows.

    Rows are decoded lazily on ``__getitem__``: an OMol25 split does not
    fit in memory, and LMDB's memory-mapped random access is what the
    format exists for.  What *is* eager is the **catalogue** — one pass
    that summarises every row (elements, atom count, energy, charge,
    spin, formula, sub-corpus) so the filters below can be applied
    without decoding geometry.  It is cached beside each shard as
    ``<shard>.goal-catalogue.npz``, so the ~8 s/shard cost is paid once
    (``scripts/omol25_prepare.py`` pays it for every shard in parallel).

    Filtering is the point of this class, not a convenience.  A 4M-row
    OMol25 split spans 83 elements, charges −8…+9 and spins 1…11; a
    SIMURGH-family backbone builds one artisan per element *pair*, and
    charge and spin are not inputs to it at all.  Handing it the raw
    split would mean thousands of pair experts and a target that is not a
    function of the input.  So: restrict the chemistry with
    ``allowed_elements``, pin ``charges`` / ``spins``, and let the model
    see a well-posed problem.

    Parameters
    ----------
    root : str or Path
        A single ``.aselmdb`` shard, or a directory containing
        ``{split}.aselmdb``.
    cutoff : float
        Neighbour-list cutoff in Angstrom; must match the backbone's.
    split : str
        Used only to look for ``{split}.aselmdb`` inside a directory root.
    energy_key, forces_key, stress_key, key_mapping
        Label key resolution, exactly as for the other ASE-backed
        datasets (see :mod:`goal.ml.data.keys`).  The defaults read the
        row's own ``energy`` / ``forces``.  Every *scalar* field of the
        corpus extras (``row["data"]``) is visible to this mapping too,
        so e.g. ``energy_key: nl_energy`` reads OMol25's non-local
        dispersion energy instead of the total.
    dtype : torch.dtype
        Precision of the emitted tensors.
    neighbor_list_backend : str
        ``"ase"``, ``"matscipy"`` or ``"radius_graph"``.
    compute_fragment_index, fragment_covalent_cutoff, fragment_scheme, \
    fragment_charge, fragment_on_failure, fragment_smarts, \
    fragment_keep_groups
        Fragment labelling, as documented on :class:`BaseAtomicDataset`.
        Note that this dataset decodes lazily, so the decomposition runs
        per ``__getitem__`` rather than once at load; the rdkit schemes
        cache on the perceived bond graph, which absorbs most of the cost
        for corpora full of conformers of the same molecule.
    allowed_elements : sequence of int, optional
        Keep a structure only if **every** atom is one of these.  Set it
        to the backbone's ``elements`` list: an edge whose element pair
        has no artisan is zero-masked, so an unfiltered element
        contributes exactly zero energy and nothing warns.
    excluded_elements : sequence of int, optional
        Drop a structure if it contains any of these.
    charges, spins : sequence of int, optional
        Keep only these total charges / spin multiplicities (OMol25's
        ``data["charge"]`` and ``data["spin"]``).  Leaving both open mixes
        charge states of the same geometry, whose energies differ by
        eV — unlearnable for a backbone with no charge input.
    data_ids : sequence of str, optional
        Keep only these sub-corpora (OMol25: ``biomolecules``,
        ``elytes``, ``metal_complexes``, ``reactivity``, ``spice``,
        ``ani2x``, ``trans1x``, ``rgd``, ``orbnet_denali``,
        ``geom_orca6``).
    min_atoms, max_atoms : int, optional
        Structure-size window.
    require_energy : bool
        Drop rows with no energy label (default ``True``).
    max_samples : int, optional
        Cap on the number of rows *per shard* after filtering.
    sample_stride : int
        Keep every n-th surviving row.  Applied before ``max_samples``;
        the cheap way to thin a corpus of dense MD-like conformers.
    sample_seed : int, optional
        When set, ``max_samples`` draws a deterministic random subset
        rather than the first rows — worth it here because rows within a
        shard are correlated (consecutive conformers of one molecule).
    attach_charge_spin : bool
        Attach ``total_charge`` and ``spin_multiplicity`` (shape ``(1,)``)
        to every graph.  Off by default: no SIMURGH-family backbone reads
        them, and ``total_charge`` is a *prediction* key of the dipole
        head, so attaching it turns the row's charge into that head's
        regression target.
    fragment_charge_from_data : bool
        Use each structure's own ``data["charge"]`` for rdkit bond
        perception instead of the fixed ``fragment_charge``.  Only
        matters when ``charges`` admits more than one value.
    catalogue_cache : bool
        Write/read the per-shard catalogue on disk.  ``False`` re-scans on
        every construction.
    catalogue_cache_dir : str or Path, optional
        Where to keep those caches; default is beside each shard.  Set it
        when the dataset sits on a read-only mount.
    """

    def __init__(
        self,
        root: str | Path,
        cutoff: float,
        split: str = "train",
        energy_key: str | None = None,
        forces_key: str | None = None,
        stress_key: str | None = None,
        key_mapping: typing.Mapping[str, str] | None = None,
        dtype: torch.dtype = torch.float64,
        neighbor_list_backend: str = "ase",
        compute_fragment_index: bool = False,
        fragment_covalent_cutoff: float = 1.8,
        fragment_scheme: str = "connected",
        fragment_charge: int = 0,
        fragment_on_failure: str = "fallback",
        fragment_smarts: typing.Sequence[str] | None = None,
        fragment_keep_groups: typing.Sequence[str] | None = None,
        allowed_elements: typing.Sequence[int] | None = None,
        excluded_elements: typing.Sequence[int] | None = None,
        charges: typing.Sequence[int] | None = None,
        spins: typing.Sequence[int] | None = None,
        data_ids: typing.Sequence[str] | None = None,
        min_atoms: int | None = None,
        max_atoms: int | None = None,
        require_energy: bool = True,
        max_samples: int | None = None,
        sample_stride: int = 1,
        sample_seed: int | None = None,
        attach_charge_spin: bool = False,
        fragment_charge_from_data: bool = False,
        catalogue_cache: bool = True,
        catalogue_cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            root=root,
            cutoff=cutoff,
            split=split,
            dtype=dtype,
            neighbor_list_backend=neighbor_list_backend,
            compute_fragment_index=compute_fragment_index,
            fragment_covalent_cutoff=fragment_covalent_cutoff,
            fragment_scheme=fragment_scheme,
            fragment_charge=fragment_charge,
            fragment_on_failure=fragment_on_failure,
            fragment_smarts=fragment_smarts,
            fragment_keep_groups=fragment_keep_groups,
        )
        from goal.ml.data.keys import resolve_label_keys

        self._key_map: dict[str, str]
        self._explicit_keys: set[str]
        self._key_map, self._explicit_keys = resolve_label_keys(
            energy_key, forces_key, stress_key, key_mapping
        )

        self.allowed_elements: list[int] | None = (
            [int(z) for z in allowed_elements] if allowed_elements is not None else None
        )
        self.excluded_elements: list[int] | None = (
            [int(z) for z in excluded_elements] if excluded_elements is not None else None
        )
        self.charges: list[int] | None = [int(c) for c in charges] if charges is not None else None
        self.spins: list[int] | None = [int(s) for s in spins] if spins is not None else None
        self.data_ids: list[str] | None = (
            [str(d) for d in data_ids] if data_ids is not None else None
        )
        self.min_atoms: int | None = int(min_atoms) if min_atoms is not None else None
        self.max_atoms: int | None = int(max_atoms) if max_atoms is not None else None
        self.require_energy: bool = bool(require_energy)
        self.max_samples: int | None = int(max_samples) if max_samples is not None else None
        self.sample_stride: int = max(1, int(sample_stride))
        self.sample_seed: int | None = int(sample_seed) if sample_seed is not None else None
        self.attach_charge_spin: bool = bool(attach_charge_spin)
        self.fragment_charge_from_data: bool = bool(fragment_charge_from_data)

        self._reader: ASELMDBReader = ASELMDBReader(
            self._resolve_shard(), cache_dir=catalogue_cache_dir
        )
        catalogue: ASELMDBCatalogue = self._reader.catalogue(
            cache=bool(catalogue_cache), progress=_log
        )
        # Flat reader positions of the surviving rows.  The reader's index
        # is (file, row) sorted and this dataset owns a single shard, so a
        # catalogue position *is* a reader position and the two stay in
        # step: ``self._catalogue`` row i describes ``self[i]``.
        self._positions: np.ndarray = self._survivor_positions(catalogue)
        self._catalogue: ASELMDBCatalogue = catalogue.select(self._positions)
        self._validate_label_keys()
        _log(
            f"[aselmdb] {Path(self._resolve_shard()).name}: "
            f"{len(self._catalogue):,} / {len(catalogue):,} rows kept"
        )

    # -- construction helpers -----------------------------------------

    def _resolve_shard(self) -> Path:
        """The single shard this dataset reads.

        Mirrors :class:`~goal.ml.data.datasets.trajectory.TrajectoryDataset`:
        a directory root must contain ``{split}.aselmdb``, otherwise the
        error is a ``FileNotFoundError`` so ``GOALDataModule`` falls back
        to discovering the directory's shards and splitting them
        numerically.  Loading a whole directory here instead would make
        train, val and test the same rows.
        """
        named: Path = self.root / f"{self.split}.aselmdb"
        if named.is_file():
            return named
        if self.root.is_file():
            return self.root
        if self.root.is_dir():
            raise FileNotFoundError(
                f"No '{self.split}{ASELMDB_SUFFIX}' in directory {self.root}, and a "
                f"directory of shards cannot be one dataset (train/val/test would "
                f"share every row). Point 'root' at the directory and let the "
                f"DataModule discover and split its shards, use "
                f"train_dir/val_dir/test_dir, or name per-split files "
                f"'{self.split}{ASELMDB_SUFFIX}'."
            )
        raise FileNotFoundError(f"Dataset file not found: {self.root}")

    def _validate_label_keys(self) -> None:
        """Fail now if a configured label key is not in the data.

        :func:`goal.ml.data.keys.extract_ase_labels` resolves a key by
        looking in ``atoms.info`` / ``atoms.arrays`` and then falling back
        to the attached calculator.  For file formats where the calculator
        is often absent that fallback is what makes a *missing* label an
        error; here it is a trap, because an ``ase.db`` row **always**
        carries a ``SinglePointCalculator`` with the row's energy and
        forces.  A typo in ``data.energy_key`` would therefore resolve
        silently to the total energy and the run would train on the wrong
        target while looking healthy.

        So an explicitly configured key is checked against what the first
        row actually offers, at construction — before the model is built,
        not thousands of steps in.
        """
        if not self._explicit_keys or len(self) == 0:
            return

        row: dict[str, typing.Any] = self._reader.read(int(self._positions[0]))
        extras: dict[str, typing.Any] = row.get("data") or {}
        available: set[str] = set(row) | set(extras)
        for prop in sorted(self._explicit_keys):
            key: str = self._key_map[prop]
            if key in available:
                continue
            raise ValueError(
                f"Label key {key!r} (configured as data.{prop}_key / "
                f"data.key_mapping.{prop}) is not found in this corpus.\n"
                f"  row fields  : {sorted(k for k in row if k != 'data')}\n"
                f"  data fields : {sorted(extras)}\n"
                f"Note the row's own energy and forces are reachable as "
                f"'energy' and 'forces' (the defaults); a wrong key here "
                f"would otherwise fall back to them silently."
            )

    def _filter(self, catalogue: ASELMDBCatalogue) -> np.ndarray:
        """Boolean mask of the rows this dataset keeps."""
        keep: np.ndarray = catalogue.element_mask(self.allowed_elements, self.excluded_elements)
        if self.require_energy:
            keep &= np.isfinite(catalogue.energy)
        if self.charges is not None:
            keep &= np.isin(catalogue.charge, np.asarray(self.charges, dtype=np.int16))
        if self.spins is not None:
            keep &= np.isin(catalogue.spin, np.asarray(self.spins, dtype=np.int16))
        if self.data_ids is not None:
            keep &= np.isin(catalogue.data_id, np.asarray(self.data_ids, dtype=str))
        if self.min_atoms is not None:
            keep &= catalogue.num_atoms >= self.min_atoms
        if self.max_atoms is not None:
            keep &= catalogue.num_atoms <= self.max_atoms
        return keep

    def _survivor_positions(self, catalogue: ASELMDBCatalogue) -> np.ndarray:
        """Reader positions of the kept rows, thinned and capped."""
        positions: np.ndarray = np.flatnonzero(self._filter(catalogue))
        if self.sample_stride > 1:
            positions = positions[:: self.sample_stride]
        if self.max_samples is not None and positions.shape[0] > self.max_samples:
            if self.sample_seed is None:
                positions = positions[: self.max_samples]
            else:
                positions = np.sort(
                    np.random.default_rng(self.sample_seed).choice(
                        positions, size=self.max_samples, replace=False
                    )
                )
        return positions

    # -- introspection ------------------------------------------------

    @property
    def catalogue(self) -> ASELMDBCatalogue:
        """Summary of the rows this dataset serves, row *i* ↔ ``self[i]``."""
        return self._catalogue

    @property
    def reader(self) -> ASELMDBReader:
        """The underlying shard reader."""
        return self._reader

    def atoms(self, idx: int) -> typing.Any:
        """The raw ``ase.Atoms`` behind ``self[idx]`` — handy in notebooks."""
        return self._reader.atoms(int(self._positions[idx]))

    # -- Dataset protocol ---------------------------------------------

    def __len__(self) -> int:
        return int(self._positions.shape[0])

    def __getitem__(self, idx: int) -> AtomicGraph:
        from goal.ml.data.keys import extract_ase_labels

        row: dict[str, typing.Any] = self._reader.read(int(self._positions[idx]))
        atoms: typing.Any = row_to_atoms(row)
        labels: dict[str, typing.Any] = extract_ase_labels(
            atoms, self._key_map, self._explicit_keys
        )
        extras: dict[str, typing.Any] = row.get("data") or {}
        charge: int = int(extras.get("charge", self.fragment_charge))

        graph: AtomicGraph = AtomicGraph.from_ase(
            atoms,
            cutoff=self.cutoff,
            energy=labels["energy"],
            forces=labels["forces"],
            stress=labels["stress"],
            dtype=self.dtype,
            neighbor_list_backend=self.neighbor_list_backend,
            compute_fragment_index=self.compute_fragment_index,
            fragment_covalent_cutoff=self.fragment_covalent_cutoff,
            fragment_scheme=self.fragment_scheme,
            fragment_charge=charge if self.fragment_charge_from_data else self.fragment_charge,
            fragment_on_failure=self.fragment_on_failure,
            fragment_smarts=self.fragment_smarts,
            fragment_keep_groups=self.fragment_keep_groups,
        )
        if self.attach_charge_spin:
            graph.total_charge = torch.tensor([charge], dtype=self.dtype)
            graph.spin_multiplicity = torch.tensor([int(extras.get("spin", 1))], dtype=torch.long)
        return graph

    def close(self) -> None:
        """Release the shard's LMDB environment."""
        self._reader.close()


def _log(message: str) -> None:
    """Rank-zero status line, so DDP does not print it once per rank."""
    from lightning.pytorch.utilities.rank_zero import rank_zero_info

    rank_zero_info(message)
