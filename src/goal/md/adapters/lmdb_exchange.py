"""LMDB exchange — share molecules with external programs (e.g. the Rust MD driver Belero).

Both processes open the same LMDB environment; LMDB does the cross-process
locking, so one side can write while the other reads.  The layout is
language-neutral (no pickle):

* **environment** — a plain LMDB directory (``data.mdb`` + ``lock.mdb``),
  unnamed main database.  Open both sides with the same ``map_size``.
* **key** — record id as a big-endian ``u64`` (8 bytes), so LMDB's byte
  order is numeric order.
* **value** — one `safetensors <https://github.com/huggingface/safetensors>`_
  blob: ``u64 LE header length | JSON header | little-endian array bytes``.

Arrays in a record (any other numeric array is stored the same way):

==================  =====  ======  =======  =========
name                dtype  shape   unit     required
==================  =====  ======  =======  =========
``atomic_numbers``  I64    [N]              yes
``positions``       F64    [N, 3]  Å        yes
``cell``            F64    [3, 3]  Å        no
``pbc``             BOOL   [3]              no
``energy``          F64    []      eV       no
``forces``          F64    [N, 3]  eV/Å     no
==================  =====  ======  =======  =========

The record's ``"metadata"`` dict goes in the safetensors ``__metadata__``
(``string → string``) with every value JSON-encoded, e.g. ``charge: 0`` →
``"0"`` and ``method: "PBE"`` → ``"\\"PBE\\""``.

Reading from Rust (``heed`` + ``safetensors`` crates)::

    let db: Database<U64<BE>, Bytes> = env.open_database(&rtxn, None)?.unwrap();
    let bytes = db.get(&rtxn, &id)?.unwrap();
    let st = SafeTensors::deserialize(bytes)?;          // st.tensor("forces")?.data()
    let (_, meta) = SafeTensors::read_metadata(bytes)?; // meta.metadata()

Request/response use: Belero writes a record without ``energy``/``forces``,
GOAL reads it, computes, and writes the result back under the same key.
"""

from __future__ import annotations

import json
import struct
import typing
from pathlib import Path

import lmdb
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

#: numpy dtype (without byte order) ↔ safetensors dtype.
_DTYPES: dict[str, str] = {
    "f8": "F64", "f4": "F32", "f2": "F16",
    "i8": "I64", "i4": "I32", "i2": "I16", "i1": "I8",
    "u1": "U8", "b1": "BOOL",
}  # fmt: skip
_NP_DTYPES: dict[str, str] = {v: k for k, v in _DTYPES.items()}

_RESULT_KEYS: tuple[str, ...] = ("energy", "forces", "stress")


def _json_default(obj: typing.Any) -> typing.Any:
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()
    raise TypeError(f"metadata value {obj!r} is not JSON-serialisable")


def _json_or_str(text: str) -> typing.Any:
    # Writers that skip JSON-encoding (plain strings) still decode.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def encode_record(record: typing.Mapping[str, typing.Any]) -> bytes:
    """Serialise a record dict to a safetensors blob."""
    header: dict[str, typing.Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, value in record.items():
        if name == "metadata":
            continue
        arr = np.asarray(value)
        code = _DTYPES.get(arr.dtype.str[1:])
        if code is None:
            raise TypeError(f"{name!r}: dtype {arr.dtype} has no safetensors equivalent")
        data = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("<")).tobytes()
        header[name] = {
            "dtype": code,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    if record.get("metadata"):
        header["__metadata__"] = {
            str(k): json.dumps(v, default=_json_default) for k, v in record["metadata"].items()
        }
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)  # safetensors pads the header to 8 bytes
    return struct.pack("<Q", len(head)) + head + b"".join(blobs)


def decode_record(buf: bytes) -> dict[str, typing.Any]:
    """Parse a safetensors blob into a record dict (arrays are read-only views)."""
    (n,) = struct.unpack_from("<Q", buf)
    header: dict[str, typing.Any] = json.loads(buf[8 : 8 + n])
    metadata = header.pop("__metadata__", None) or {}
    body = memoryview(buf)[8 + n :]
    record: dict[str, typing.Any] = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        dtype = np.dtype("<" + _NP_DTYPES[info["dtype"]])
        record[name] = np.frombuffer(body[start:end], dtype=dtype).reshape(info["shape"])
    record["metadata"] = {k: _json_or_str(v) for k, v in metadata.items()}
    return record


def atoms_to_record(atoms: Atoms) -> dict[str, typing.Any]:
    """ASE Atoms → record; energy/forces/stress come from cached calculator results only."""
    record: dict[str, typing.Any] = {
        "atomic_numbers": atoms.numbers,
        "positions": atoms.positions,
        "cell": atoms.cell.array,
        "pbc": atoms.pbc,
        "metadata": dict(atoms.info),
    }
    results = atoms.calc.results if atoms.calc is not None else {}
    record.update({k: results[k] for k in _RESULT_KEYS if k in results})
    return record


def record_to_atoms(record: typing.Mapping[str, typing.Any]) -> Atoms:
    """Record → ASE Atoms; energy/forces/stress are attached as a SinglePointCalculator."""
    atoms = Atoms(
        numbers=record["atomic_numbers"],
        positions=record["positions"],
        cell=record.get("cell"),
        pbc=record.get("pbc", False),
        info=dict(record.get("metadata", {})),
    )
    results = {k: record[k] for k in _RESULT_KEYS if k in record}
    if "energy" in results:
        results["energy"] = float(results["energy"])
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def _key(idx: int) -> bytes:
    return struct.pack(">Q", idx)


class LMDBExchange:
    """Read and write molecule records in an LMDB database shared with another process.

    Parameters
    ----------
    path : str or Path
        LMDB environment directory (created if missing unless *readonly*).
    map_size : int
        Maximum database size in bytes (sparse on Linux, so a large value is free).
    readonly : bool
        Open without write access.

    Examples
    --------
    >>> with LMDBExchange("exchange.lmdb") as db:
    ...     idx = db.put(atoms)          # append, returns the new id
    ...     atoms = db.get_atoms(idx)
    ...     db.put(atoms, key=idx)       # overwrite in place
    """

    def __init__(self, path: str | Path, map_size: int = 2**34, readonly: bool = False) -> None:
        self.env = lmdb.open(str(path), map_size=map_size, readonly=readonly)

    def put(self, record: Atoms | typing.Mapping[str, typing.Any], key: int | None = None) -> int:
        """Write *record* under *key* (overwriting), or append at the next free id."""
        if isinstance(record, Atoms):
            record = atoms_to_record(record)
        value = encode_record(record)
        with self.env.begin(write=True) as txn:
            if key is None:
                cursor = txn.cursor()
                key = int.from_bytes(cursor.key(), "big") + 1 if cursor.last() else 0
            txn.put(_key(key), value)
        return key

    def get(self, key: int) -> dict[str, typing.Any]:
        """Return the record dict stored under *key*."""
        with self.env.begin() as txn:
            raw = txn.get(_key(key))
        if raw is None:
            raise KeyError(key)
        return decode_record(raw)

    def get_atoms(self, key: int) -> Atoms:
        """Return the record under *key* as ASE Atoms."""
        return record_to_atoms(self.get(key))

    def keys(self, start: int = 0) -> list[int]:
        """Return the stored ids ``>= start`` in ascending order."""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            if not cursor.set_range(_key(start)):
                return []
            return [int.from_bytes(k, "big") for k in cursor.iternext(values=False)]

    def __len__(self) -> int:
        return self.env.stat()["entries"]

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
