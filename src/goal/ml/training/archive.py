"""Self-contained checkpoint archives for GOAL / SIMURGH training runs.

A checkpoint saved during training only contains *weights*.  Loading it
later requires the exact model source code that produced those weights —
any signature change, renamed class, or deleted module between *save*
and *load* breaks the checkpoint silently.

This module makes checkpoints self-contained:

1. :func:`freeze_model_source` copies every ``goal``-package ``.py`` file
   the model depends on (discovered by walking its import graph) into
   ``{dirpath}/frozen_source/`` at training start, before any weights
   are written.
2. :func:`pack_simurgh_archive` zips weights + resolved config +
   metadata + frozen source into a single ``.simurgh`` file that can be
   shipped anywhere.
3. :func:`frozen_source_imports` is an import context that makes
   ``goal.*`` imports resolve against a frozen source tree first (and
   fall back to the live package for anything that was not frozen), so
   a checkpoint directory or archive always loads with the code it was
   trained with.

Directory layout produced by ``GOALCheckpointManager``::

    {dirpath}/
      frozen_source/goal/...        ← model source, frozen at t=0
      config.yaml                   ← fully resolved Hydra config
      metadata.json                 ← version, git commit, elements, ...
      best_val_forces_forces_mae=...ckpt ← weights (+ .json sidecar each)
      last.ckpt
      checkpoint_state.json

Archive layout (``.simurgh`` — a plain zip)::

    weights.pt
    config.yaml
    metadata.json
    checkpoint.json                 ← per-checkpoint sidecar (if present)
    frozen_source/goal/...
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import types
import typing
import zipfile
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

#: File extension of the self-contained model archive (a plain zip).
ARCHIVE_SUFFIX: str = ".simurgh"

#: Name of the frozen-source subdirectory inside a checkpoint directory.
FROZEN_SOURCE_DIRNAME: str = "frozen_source"

#: Modules the *loader* needs even when the model itself never touches
#: them at module level (registries, the LightningModule wrapper, the
#: graph container).  Seeded into the import-graph walk so a frozen tree
#: can always rebuild a ``GOALModule`` from a checkpoint.
_LOADER_MODULES: tuple[str, ...] = (
    "goal",
    "goal.ml",
    "goal.ml.registry",
    "goal.ml.data.graph",
    "goal.ml.training.module",
    "goal.ml.training.loss",
    "goal.ml.training.ema",
    "goal.ml.nn.models",
    "goal.ml.nn.heads",
)


def _goal_src_root() -> Path:
    """Directory that *contains* the ``goal`` package (i.e. ``src/``)."""
    import goal

    return Path(goal.__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def get_git_commit() -> str | None:
    """Return the current git commit hash of the goal source tree, or None."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=str(_goal_src_root()),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def get_goal_version() -> str:
    """Installed ``goal`` package version, or ``"unknown"``."""
    try:
        return importlib.metadata.version("goal")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def gather_model_metadata(model: typing.Any) -> dict[str, typing.Any]:
    """Collect JSON-serialisable provenance metadata from a model.

    Every field is gathered defensively with ``getattr`` so this works
    for any backbone (SIMURGH or otherwise) — absent attributes are
    simply omitted.
    """
    import torch

    meta: dict[str, typing.Any] = {
        "simurgh_version": get_goal_version(),
        "git_commit": get_git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_class": type(model).__name__,
        "source_frozen": True,
    }

    elements: typing.Any = getattr(model, "elements", None)
    if elements is not None:
        meta["elements"] = [int(z) for z in elements]

    # Per-element reference energies.  Prefer a dedicated accessor when
    # the model provides one; otherwise read the SIMURGH-style
    # ``atomic_energies`` table indexed by atomic number.
    refs_fn = getattr(model, "atomic_references_dict", None)
    if callable(refs_fn):
        try:
            meta["atomic_references"] = {str(k): float(v) for k, v in refs_fn().items()}
        except Exception:  # noqa: BLE001 — metadata must never break training
            pass
    elif elements is not None:
        table: typing.Any = getattr(model, "atomic_energies", None)
        if isinstance(table, torch.Tensor) and table.dim() == 1:
            meta["atomic_references"] = {
                str(int(z)): float(table[int(z)]) for z in elements if int(z) < table.numel()
            }

    pair_freq: typing.Any = getattr(model, "pair_frequencies", None)
    if pair_freq is not None:
        with contextlib.suppress(Exception):
            meta["pair_frequencies"] = {str(k): float(v) for k, v in dict(pair_freq).items()}

    bank: typing.Any = getattr(model, "artisan_bank", None)
    if bank is not None:
        dedicated: typing.Any = getattr(bank, "_dedicated_pairs", None)
        if dedicated is not None:
            meta["dedicated_pairs"] = [[int(a), int(b)] for a, b in dedicated]
        meta["rare_artisan"] = getattr(bank, "rare_artisan_module", None) is not None

    dressing: typing.Any = getattr(model, "dressing", None)
    avg_nn: typing.Any = getattr(dressing, "_avg_num_neighbors", None)
    if avg_nn is not None:
        meta["avg_num_neighbors"] = float(avg_nn)

    scale: typing.Any = getattr(model, "scale", None)
    if isinstance(scale, torch.Tensor) and scale.numel() == 1:
        meta["scale"] = float(scale)
    elif isinstance(scale, (int, float)):
        meta["scale"] = float(scale)

    return meta


# ---------------------------------------------------------------------------
# Import-graph walk + source freezing
# ---------------------------------------------------------------------------


def _enqueue_goal_module(queue: list[types.ModuleType], obj: typing.Any) -> None:
    """If ``obj`` is (or belongs to) a ``goal.*`` module, enqueue that module."""
    if isinstance(obj, types.ModuleType):
        name = getattr(obj, "__name__", "")
        if name == "goal" or name.startswith("goal."):
            queue.append(obj)
        return
    owner = getattr(obj, "__module__", None)
    if isinstance(owner, str) and (owner == "goal" or owner.startswith("goal.")):
        mod = sys.modules.get(owner)
        if mod is not None:
            queue.append(mod)


def get_model_source_files(model: typing.Any) -> dict[str, Path]:
    """Return every ``goal``-package ``.py`` file the model depends on.

    Walks the import graph breadth-first starting from the defining
    module of every ``nn.Module`` inside ``model`` (plus the loader
    modules in :data:`_LOADER_MODULES`): each module's globals are
    scanned for imported ``goal.*`` modules and for objects whose
    ``__module__`` lives under ``goal``.  Files outside the ``goal``
    package (site-packages, stdlib) are never included.

    Returns
    -------
    dict
        Mapping of package-relative path (``"goal/ml/...py"``) to the
        absolute source file.
    """
    src_root: Path = _goal_src_root()

    queue: list[types.ModuleType] = []
    for name in _LOADER_MODULES:
        try:
            queue.append(importlib.import_module(name))
        except ImportError:
            log.warning("archive: loader module %s could not be imported", name)
    _enqueue_goal_module(queue, type(model))
    modules_fn = getattr(model, "modules", None)
    if callable(modules_fn):
        for sub in modules_fn():
            _enqueue_goal_module(queue, type(sub))

    seen: set[str] = set()
    files: dict[str, Path] = {}

    def _add_file(mod: types.ModuleType) -> None:
        try:
            file = Path(inspect.getfile(mod)).resolve()
        except (TypeError, OSError):
            return
        if file.suffix != ".py":
            return
        try:
            rel = file.relative_to(src_root)
        except ValueError:
            return  # not under the goal source tree
        files[str(rel)] = file

    while queue:
        mod = queue.pop()
        name: str = getattr(mod, "__name__", "")
        if not name or name in seen:
            continue
        if name != "goal" and not name.startswith("goal."):
            continue
        seen.add(name)
        _add_file(mod)

        # Parent packages (their __init__.py must exist for imports to work)
        parts = name.split(".")
        for i in range(1, len(parts)):
            parent = sys.modules.get(".".join(parts[:i]))
            if parent is not None:
                queue.append(parent)

        for obj in list(vars(mod).values()):
            _enqueue_goal_module(queue, obj)

    # Safety net: include every sibling .py of the model's own package so
    # function-local imports within the model directory are never missed.
    model_mod = sys.modules.get(type(model).__module__)
    if model_mod is not None and hasattr(model_mod, "__file__") and model_mod.__file__:
        pkg_dir = Path(model_mod.__file__).resolve().parent
        with contextlib.suppress(ValueError):
            pkg_dir.relative_to(src_root)  # raises if outside goal
            for sibling in pkg_dir.glob("*.py"):
                files[str(sibling.relative_to(src_root))] = sibling

    return files


def freeze_model_source(model: typing.Any, dest_dir: str | Path) -> list[str]:
    """Copy the model's source closure into ``dest_dir`` (preserving layout).

    Returns the sorted list of package-relative paths that were frozen.
    """
    dest = Path(dest_dir)
    files = get_model_source_files(model)
    for rel, src in sorted(files.items()):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return sorted(files)


# ---------------------------------------------------------------------------
# Frozen-source import context
# ---------------------------------------------------------------------------


class _FrozenSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve ``goal.*`` imports against an ordered list of source roots.

    The first root is the frozen snapshot; later roots (the live package)
    act as a fallback for modules that were not part of the frozen
    closure (e.g. utilities the model never touched).
    """

    def __init__(self, roots: typing.Sequence[Path]) -> None:
        self._roots: list[Path] = [Path(r) for r in roots]

    def find_spec(
        self,
        fullname: str,
        path: typing.Any = None,
        target: typing.Any = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != "goal" and not fullname.startswith("goal."):
            return None
        rel = Path(*fullname.split("."))
        for root in self._roots:
            pkg_init = root / rel / "__init__.py"
            if pkg_init.is_file():
                return importlib.util.spec_from_file_location(
                    fullname,
                    pkg_init,
                    submodule_search_locations=[str(pkg_init.parent)],
                )
            mod_file = root / rel.with_suffix(".py")
            if mod_file.is_file():
                return importlib.util.spec_from_file_location(fullname, mod_file)
        return None


@contextlib.contextmanager
def frozen_source_imports(frozen_root: str | Path) -> typing.Iterator[None]:
    """Context manager: ``goal.*`` imports resolve from ``frozen_root`` first.

    Inside the context every (re-)import of a ``goal`` module executes
    the frozen copy under ``frozen_root`` (falling back to the live
    package for modules that were not frozen).  On exit the original
    ``sys.modules`` entries are restored, so the rest of the process
    keeps using the live code.  Objects created inside the context keep
    working afterwards — they hold references to the frozen modules.

    Not thread-safe: it swaps global interpreter state (``sys.modules``).
    """
    frozen = Path(frozen_root).resolve()
    live_root = _goal_src_root()

    saved: dict[str, types.ModuleType] = {}
    for name in list(sys.modules):
        if name == "goal" or name.startswith("goal."):
            saved[name] = sys.modules.pop(name)

    finder = _FrozenSourceFinder([frozen, live_root])
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name == "goal" or name.startswith("goal."):
                del sys.modules[name]
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def is_managed_checkpoint_dir(dirpath: str | Path) -> bool:
    """True if ``dirpath`` has the GOALCheckpointManager self-contained layout."""
    d = Path(dirpath)
    return (
        (d / "config.yaml").is_file()
        and (d / "metadata.json").is_file()
        and (d / FROZEN_SOURCE_DIRNAME).is_dir()
    )


def resolve_checkpoint(dirpath: str | Path, checkpoint: str = "best") -> Path:
    """Resolve a checkpoint selector inside a managed checkpoint directory.

    Selectors
    ---------
    ``"best"``
        Best top-k checkpoint according to ``checkpoint_state.json``
        (falls back to parsing ``best_*.ckpt`` filenames, lower=better).
    ``"last"``
        ``last.ckpt`` (or the filename recorded in the pool state).
    ``"epoch=N"``
        The interval checkpoint for epoch ``N``.
    anything ending in ``.ckpt``
        Used directly (relative paths are resolved against ``dirpath``).
    """
    d = Path(dirpath)

    if checkpoint.endswith(".ckpt"):
        cand = Path(checkpoint)
        if not cand.is_absolute():
            cand = d / cand
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"Checkpoint file not found: {cand}")

    state: dict[str, typing.Any] = {}
    state_path = d / "checkpoint_state.json"
    if state_path.is_file():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            state = json.loads(state_path.read_text())

    if checkpoint == "last":
        filename = str(state.get("config", {}).get("last_filename", "last.ckpt"))
        cand = d / filename
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"No last checkpoint in {d} (looked for {filename})")

    if checkpoint == "best":
        pool: list[typing.Any] = state.get("top_k_pool", [])
        mode: str = str(state.get("config", {}).get("top_k_mode", "min"))
        candidates: list[tuple[float, Path]] = [
            (float(v), Path(p)) for v, p in pool if Path(p).is_file()
        ]
        if not candidates:
            # Fall back to parsing best_*=<value>_epoch=*.ckpt filenames
            for f in d.glob("best_*.ckpt"):
                with contextlib.suppress(ValueError, IndexError):
                    value = float(f.stem.split("=")[1].split("_epoch")[0])
                    candidates.append((value, f))
        if not candidates:
            raise FileNotFoundError(f"No top-k ('best_*') checkpoint found in {d}")
        pick = max if mode == "max" else min
        return pick(candidates, key=lambda t: t[0])[1]

    if checkpoint.startswith("epoch="):
        try:
            epoch = int(checkpoint.split("=", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid epoch selector: {checkpoint!r}") from exc
        cand = d / f"interval_epoch={epoch:04d}.ckpt"
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"No interval checkpoint for epoch {epoch} in {d}")

    raise ValueError(
        f"Unknown checkpoint selector {checkpoint!r}. "
        f"Use 'best', 'last', 'epoch=N', or a .ckpt path."
    )


# ---------------------------------------------------------------------------
# Archive packing / unpacking
# ---------------------------------------------------------------------------


def _missing_layout_error(dirpath: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"Directory {dirpath} is missing config.yaml, metadata.json, or "
        f"{FROZEN_SOURCE_DIRNAME}/. This is not a self-contained SIMURGH "
        f"checkpoint directory. Pass a {ARCHIVE_SUFFIX} archive or a "
        f"directory produced by GOALCheckpointManager."
    )


def pack_simurgh_archive(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Pack one checkpoint plus its directory context into a ``.simurgh`` zip.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.ckpt`` file inside a managed checkpoint directory
        (one containing ``config.yaml``, ``metadata.json`` and
        ``frozen_source/``).
    output_path:
        Destination archive.  Defaults to ``<ckpt stem>.simurgh`` next
        to the checkpoint.

    Returns the path of the written archive.  The write is atomic
    (``.tmp`` then ``os.replace``).
    """
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    dirpath = ckpt.parent
    if not is_managed_checkpoint_dir(dirpath):
        raise _missing_layout_error(dirpath)

    out = Path(output_path) if output_path is not None else dirpath / (ckpt.stem + ARCHIVE_SUFFIX)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")

    frozen_dir = dirpath / FROZEN_SOURCE_DIRNAME
    sidecar = ckpt.with_suffix(".json")

    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(ckpt, "weights.pt")
        zf.write(dirpath / "config.yaml", "config.yaml")
        zf.write(dirpath / "metadata.json", "metadata.json")
        if sidecar.is_file():
            zf.write(sidecar, "checkpoint.json")
        for file in sorted(frozen_dir.rglob("*")):
            if file.is_file():
                zf.write(file, f"{FROZEN_SOURCE_DIRNAME}/{file.relative_to(frozen_dir)}")
    os.replace(tmp, out)
    log.info("[archive] Packed %s", out)
    return out


def unpack_simurgh_archive(archive_path: str | Path, dest_dir: str | Path) -> Path:
    """Extract a ``.simurgh`` archive into ``dest_dir`` and return that path.

    Rejects archive members that would escape the destination directory.
    """
    archive = Path(archive_path)
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"Unsafe archive member path: {member!r}")
        zf.extractall(dest)
    return dest


# ---------------------------------------------------------------------------
# Inspection (no model loading)
# ---------------------------------------------------------------------------


def inspect_checkpoint(path: str | Path) -> dict[str, typing.Any]:
    """Describe an archive / checkpoint directory / bare ckpt without loading it.

    Returns a dict with: ``format`` (``"archive"`` | ``"directory"`` |
    ``"checkpoint"``), ``source_frozen``, ``metadata`` (metadata.json
    contents or None), ``sidecar`` (per-checkpoint sidecar or None) and
    ``available_checkpoints`` (names, directories only).
    """
    p = Path(path)
    info: dict[str, typing.Any] = {
        "path": str(p),
        "format": None,
        "source_frozen": False,
        "metadata": None,
        "sidecar": None,
        "available_checkpoints": [],
    }

    if p.is_file() and p.suffix == ARCHIVE_SUFFIX:
        info["format"] = "archive"
        with zipfile.ZipFile(p) as zf:
            names = set(zf.namelist())
            info["source_frozen"] = any(n.startswith(f"{FROZEN_SOURCE_DIRNAME}/") for n in names)
            if "metadata.json" in names:
                info["metadata"] = json.loads(zf.read("metadata.json"))
            if "checkpoint.json" in names:
                info["sidecar"] = json.loads(zf.read("checkpoint.json"))
        return info

    if p.is_dir():
        info["format"] = "directory"
        info["source_frozen"] = (p / FROZEN_SOURCE_DIRNAME).is_dir()
        meta_path = p / "metadata.json"
        if meta_path.is_file():
            info["metadata"] = json.loads(meta_path.read_text())
        ckpts = sorted(f.name for f in p.glob("*.ckpt"))
        info["available_checkpoints"] = ckpts
        # Best checkpoint's sidecar gives epoch / metric detail
        with contextlib.suppress(FileNotFoundError, ValueError):
            best = resolve_checkpoint(p, "best")
            sidecar_path = best.with_suffix(".json")
            if sidecar_path.is_file():
                info["sidecar"] = json.loads(sidecar_path.read_text())
        return info

    if p.is_file() and p.suffix == ".ckpt":
        info["format"] = "checkpoint"
        sidecar_path = p.with_suffix(".json")
        if sidecar_path.is_file():
            info["sidecar"] = json.loads(sidecar_path.read_text())
        # A bare ckpt may still sit inside a managed directory
        if is_managed_checkpoint_dir(p.parent):
            info["source_frozen"] = True
            info["metadata"] = json.loads((p.parent / "metadata.json").read_text())
        return info

    raise FileNotFoundError(f"No archive, checkpoint directory, or .ckpt at: {p}")
