#!/usr/bin/env python
"""Rewrite pre-legacy SIMURGH registry keys in saved checkpoints.

The older SIMURGH models now live under ``simurgh_legacy*`` registry keys.
Their old keys keep loading through deprecated aliases until release,
when ``simurgh`` is reassigned to the new model; this script updates saved
runs so that they keep loading after that.

* bare Lightning ``.ckpt`` files: ``hyper_parameters.config.model.backbone.name``;
* ``GOALCheckpointManager`` directories: every ``.ckpt``, ``config.yaml``
  and the registration decorators in ``frozen_source/`` (a directory
  loads with its frozen registry, which must know the key its config
  now names);
* ``.simurgh`` archives: the same, inside the zip.

Checkpoints store registry keys, not import paths, so module paths need
no rewrite.  Dry run by default.  ``--apply`` writes a ``.migrated`` copy
next to each input (``run.ckpt`` -> ``run.migrated.ckpt``); add
``--in-place`` to overwrite instead.  Every change is logged.

Examples
--------
::

    python scripts/simurgh_dev/migrate_legacy_checkpoints.py logs/<task>/runs/<run>/checkpoints
    python scripts/simurgh_dev/migrate_legacy_checkpoints.py model.simurgh --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import tempfile
import typing
import zipfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from goal.ml.nn.models.simurgh.legacy import DEPRECATED_KEYS
from goal.ml.training.archive import ARCHIVE_SUFFIX, FROZEN_SOURCE_DIRNAME

log = logging.getLogger("migrate_legacy_checkpoints")

_DECORATOR = re.compile(r"""(REGISTRY\.register\(\s*["'])([A-Za-z0-9_]+)(["']\s*\))""")


def _rename_backbone(cfg: typing.Any, where: str, apply: bool) -> int:
    """Rename ``cfg.model.backbone.name`` if it is a deprecated key."""
    try:
        backbone = cfg["model"]["backbone"]
        old = str(backbone["name"])
    except (KeyError, TypeError):
        log.warning("%s: no model.backbone.name, skipped", where)
        return 0
    new = DEPRECATED_KEYS.get(old)
    if new is None:
        return 0
    log.info("%s: model.backbone.name %s -> %s", where, old, new)
    if apply:
        backbone["name"] = new
    return 1


def migrate_ckpt(path: Path, apply: bool) -> int:
    """Rename the backbone key inside one Lightning checkpoint."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    cfg = raw.get("hyper_parameters", {}).get("config")
    n = _rename_backbone(cfg, str(path), apply) if cfg is not None else 0
    if n and apply:
        torch.save(raw, path)
    return n


def migrate_directory(path: Path, apply: bool) -> int:
    """Migrate a checkpoint directory (or an extracted archive)."""
    n = 0
    for ckpt in sorted([*path.glob("*.ckpt"), *path.glob("weights.pt")]):
        n += migrate_ckpt(ckpt, apply)
    cfg_path = path / "config.yaml"
    if cfg_path.is_file():
        cfg = OmegaConf.load(cfg_path)
        if _rename_backbone(cfg, str(cfg_path), apply):
            n += 1
            if apply:
                OmegaConf.save(cfg, cfg_path)
    for src in sorted((path / FROZEN_SOURCE_DIRNAME).rglob("*.py")):
        text = src.read_text()

        def _sub(m: re.Match[str], src: Path = src) -> str:
            new = DEPRECATED_KEYS.get(m.group(2))
            if new is None:
                return m.group(0)
            log.info("%s: register(%r) -> register(%r)", src, m.group(2), new)
            return m.group(1) + new + m.group(3)

        new_text = _DECORATOR.sub(_sub, text)
        if new_text != text:
            n += 1
            if apply:
                src.write_text(new_text)
    return n


def migrate_archive(path: Path, apply: bool) -> int:
    """Migrate a ``.simurgh`` archive by extracting, migrating and re-zipping it."""
    with tempfile.TemporaryDirectory(prefix="goal_migrate_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if not (root / member).resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"Unsafe archive member path: {member!r}")
            zf.extractall(root)
        n = migrate_directory(root, apply)
        if n and apply:
            out = path.with_name(path.name + ".tmp")
            with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file in sorted(root.rglob("*")):
                    if file.is_file():
                        zf.write(file, str(file.relative_to(root)))
            os.replace(out, path)
    return n


def migrated_path(path: Path) -> Path:
    """``run.ckpt`` -> ``run.migrated.ckpt``; ``checkpoints`` -> ``checkpoints.migrated``."""
    if path.is_dir():
        return path.with_name(path.name + ".migrated")
    return path.with_name(f"{path.stem}.migrated{path.suffix}")


def migrate(path: Path, apply: bool = False, in_place: bool = False) -> tuple[Path, int]:
    """Migrate one input; returns ``(path written or scanned, number of changes)``."""
    target = path
    if apply and not in_place:
        target = migrated_path(path)
        if target.exists():
            raise FileExistsError(f"{target} exists; remove it or pass --in-place")
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    if target.is_dir():
        n = migrate_directory(target, apply)
    elif target.suffix == ARCHIVE_SUFFIX:
        n = migrate_archive(target, apply)
    elif target.suffix in (".ckpt", ".pt"):
        n = migrate_ckpt(target, apply)
    else:
        raise ValueError(f"Not a .ckpt, {ARCHIVE_SUFFIX} archive or checkpoint directory: {path}")
    if apply and not in_place and n == 0:
        # nothing to change: do not leave an identical copy behind
        shutil.rmtree(target) if target.is_dir() else target.unlink()
        target = path
    return target, n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help=".ckpt, .simurgh or directory")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    parser.add_argument("--in-place", action="store_true", help="with --apply: overwrite inputs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in args.paths:
        target, n = migrate(path, apply=args.apply, in_place=args.in_place)
        verb = "changed" if args.apply else "would change"
        log.info("%s: %d item(s) %s%s", path, n, verb, f" -> {target}" if target != path else "")


if __name__ == "__main__":
    main()
