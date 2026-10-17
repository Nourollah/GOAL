"""Dataset-derived backbone statistics: record them once, reuse them on resume.

``goal-train`` derives several backbone arguments from the training set
before it builds the model:

===================== ==============================================================
statistic             backbone argument(s)
===================== ==============================================================
``elements``          ``elements``
``atomic_energies``   ``atomic_energies.values`` (and ``mode`` → ``"dataset"``)
``scale``             ``scale``
``avg_num_neighbors`` ``avg_num_neighbors`` or ``dressing_kwargs.avg_num_neighbors``
``pair_counts``       ``artisan.pair_counts`` or ``artisan_config.pair_counts``
===================== ==============================================================

They shape the model (element tables, artisan routing) and fix its
normalisation, so a checkpoint is only meaningful together with the values
it was trained with.  This module

* records them, with the dataset they came from, in the run config's
  ``dataset_stats`` block, which the checkpoint manager saves as
  ``checkpoints/config.yaml`` (and Lightning stores in the ``.ckpt``
  hyper-parameters);
* reads them back when a run is resumed, extended or continued on new data,
  so they are not recomputed and the model is rebuilt exactly as trained;
* compares the stored dataset fingerprint with the current one, so a changed
  training set is reported instead of silently producing a different model.

Which values are reused is controlled by the experiment config::

    dataset_stats:
      reuse: true        # take stored values on resume / extension
      recompute: []      # statistics to derive again anyway, or "all"

A statistic listed in ``recompute`` is resolved exactly as on a fresh run:
an explicit value in ``model.backbone`` is kept, otherwise it is computed
from the current training set.
"""

from __future__ import annotations

import logging
import typing
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

#: User-facing statistic name → the backbone-argument paths that hold it.
STAT_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "elements": (("elements",),),
    "atomic_energies": (("atomic_energies", "values"), ("atomic_energies", "mode")),
    "scale": (("scale",),),
    "avg_num_neighbors": (("avg_num_neighbors",), ("dressing_kwargs", "avg_num_neighbors")),
    "pair_counts": (("artisan", "pair_counts"), ("artisan_config", "pair_counts")),
}

#: ``data:`` keys that only affect how batches are loaded, not which
#: structures are trained on.  Everything else identifies the dataset.
_LOADER_ONLY_DATA_KEYS: frozenset[str] = frozenset(
    {
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "monitor_test_during_training",
        "val_dir",
        "test_dir",
        "val_path",
        "test_path",
        "val_paths",
        "test_paths",
        "drop_last",
        "shuffle",
    }
)


# ----------------------------------------------------------------------
# Small helpers for nested backbone mappings
# ----------------------------------------------------------------------


def _plain(value: typing.Any) -> typing.Any:
    """Convert OmegaConf containers to plain Python containers."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _get_path(mapping: Mapping[str, typing.Any], path: tuple[str, ...]) -> typing.Any:
    """Return the value at ``path`` in nested mappings, or ``None`` if absent."""
    node: typing.Any = mapping
    for key in path:
        if not isinstance(node, (Mapping, DictConfig)) or key not in node:
            return None
        node = node[key]
    return _plain(node)


def _set_path(mapping: dict[str, typing.Any], path: tuple[str, ...], value: typing.Any) -> bool:
    """Set ``path`` in nested mappings; parents must already exist.

    Returns ``False`` without changing anything when an intermediate
    mapping is missing, so a statistic is never applied to a backbone that
    does not have the corresponding sub-config.
    """
    node: dict[str, typing.Any] = mapping
    for key in path[:-1]:
        child: typing.Any = node.get(key)
        if not isinstance(child, (Mapping, DictConfig)):
            return False
        child = dict(_plain(child))
        node[key] = child
        node = child
    node[path[-1]] = value
    return True


def normalise_recompute(raw: typing.Any) -> set[str]:
    """Turn ``dataset_stats.recompute`` into a set of statistic names.

    Accepts a list of names, a single name, ``"all"`` / ``True`` (every
    statistic) or ``None`` / ``False`` / ``[]`` (none).

    Raises
    ------
    ValueError
        If a name is not one of :data:`STAT_PATHS`.
    """
    if raw is None or raw is False:
        return set()
    if raw is True or raw == "all":
        return set(STAT_PATHS)
    names: list[str] = [raw] if isinstance(raw, str) else [str(n) for n in _plain(raw)]
    if "all" in names:
        return set(STAT_PATHS)
    unknown: list[str] = sorted(set(names) - set(STAT_PATHS))
    if unknown:
        raise ValueError(
            f"dataset_stats.recompute: unknown statistic(s) {unknown}. "
            f"Valid names: {sorted(STAT_PATHS)} or 'all'."
        )
    return set(names)


# ----------------------------------------------------------------------
# Collect / apply
# ----------------------------------------------------------------------


def collect(backbone: Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Return the dataset statistics present in a backbone config.

    Keys are dotted argument paths (``"dressing_kwargs.avg_num_neighbors"``);
    only non-null values are returned.  ``atomic_energies.mode`` is included
    only together with its ``values``, since a mode alone is a user setting,
    not a derived statistic.
    """
    found: dict[str, typing.Any] = {}
    for paths in STAT_PATHS.values():
        for path in paths:
            value: typing.Any = _get_path(backbone, path)
            if value is not None:
                found[".".join(path)] = value
    if "atomic_energies.values" not in found:
        found.pop("atomic_energies.mode", None)
    return found


def apply(
    backbone_kwargs: dict[str, typing.Any],
    values: Mapping[str, typing.Any],
    skip: set[str],
) -> list[str]:
    """Write stored statistics into ``backbone_kwargs``, in place.

    Parameters
    ----------
    backbone_kwargs:
        Keyword arguments the backbone will be built with.
    values:
        Dotted-path → value map, as returned by :func:`collect`.
    skip:
        Statistic names (see :data:`STAT_PATHS`) not to apply.

    Returns
    -------
    list of str
        The statistic names that were applied.
    """
    applied: list[str] = []
    for name, paths in STAT_PATHS.items():
        if name in skip:
            continue
        hit: bool = False
        for path in paths:
            dotted: str = ".".join(path)
            if dotted in values and _set_path(backbone_kwargs, path, values[dotted]):
                hit = True
        if hit:
            applied.append(name)
    return applied


# ----------------------------------------------------------------------
# Dataset fingerprint
# ----------------------------------------------------------------------


def dataset_fingerprint(cfg: DictConfig, n_train: int | None) -> dict[str, typing.Any]:
    """Describe which training data the statistics are computed from.

    The resolved ``data:`` block minus loader-only settings (batch size,
    workers, validation/test locations), plus the number of training
    structures.
    """
    data_cfg: typing.Any = cfg.get("data") or {}
    try:
        data: dict[str, typing.Any] = dict(OmegaConf.to_container(data_cfg, resolve=True))
    except Exception:  # noqa: BLE001 — unresolvable interpolations
        data = dict(OmegaConf.to_container(data_cfg, resolve=False))
    fingerprint: dict[str, typing.Any] = {
        k: v for k, v in sorted(data.items()) if k not in _LOADER_ONLY_DATA_KEYS
    }
    if n_train is not None:
        fingerprint["n_train"] = int(n_train)
    return fingerprint


def fingerprint_changes(
    old: Mapping[str, typing.Any], new: Mapping[str, typing.Any]
) -> dict[str, tuple[typing.Any, typing.Any]]:
    """Return ``{key: (old, new)}`` for every fingerprint entry that differs."""
    keys: set[str] = set(old) | set(new)
    return {k: (old.get(k), new.get(k)) for k in sorted(keys) if old.get(k) != new.get(k)}


# ----------------------------------------------------------------------
# Record / load
# ----------------------------------------------------------------------


def record(
    cfg: DictConfig,
    backbone_kwargs: Mapping[str, typing.Any],
    fingerprint: Mapping[str, typing.Any],
    origin: str,
    trained_on: Mapping[str, typing.Any] | None = None,
) -> None:
    """Store the statistics and their provenance in ``cfg.dataset_stats``.

    The checkpoint manager saves the config as ``checkpoints/config.yaml``,
    and Lightning stores it in every ``.ckpt``, so the values travel with the
    weights.  The user's ``reuse`` / ``recompute`` settings are preserved.

    Parameters
    ----------
    cfg:
        The run config (modified in place).
    backbone_kwargs:
        The arguments the backbone is built with.
    fingerprint:
        :func:`dataset_fingerprint` of the data the values came from.
    origin:
        Human-readable note, e.g. ``"computed"`` or ``"reused from <dir>"``.
    trained_on:
        Fingerprint of the data this run trains on, when it differs from
        ``fingerprint`` (reused values on new data).  The next resume
        compares against this, so continuing on the same new data again is
        not reported as a change.
    """
    from omegaconf import open_dict

    block: dict[str, typing.Any] = dict(_plain(cfg.get("dataset_stats") or {}))
    block.setdefault("reuse", True)
    block.setdefault("recompute", [])
    block["values"] = collect(backbone_kwargs)
    block["provenance"] = {
        "origin": origin,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dict(fingerprint),
        "trained_on": dict(trained_on if trained_on is not None else fingerprint),
    }
    with open_dict(cfg):
        cfg.dataset_stats = block


def load_stored(source_dir: Path | None, ckpt_path: str | None) -> DictConfig | None:
    """Read the run config a checkpoint was trained with.

    Looks for ``config.yaml`` next to the checkpoint first, then falls back
    to the config stored in the checkpoint's hyper-parameters.

    Returns
    -------
    DictConfig or None
        The saved config, or ``None`` if neither source has one.
    """
    if source_dir is not None:
        path: Path = Path(source_dir) / "config.yaml"
        if path.is_file():
            return OmegaConf.load(path)  # type: ignore[return-value]
    if ckpt_path is not None and Path(ckpt_path).is_file():
        import torch

        try:
            ckpt: dict[str, typing.Any] = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:  # noqa: BLE001 — unreadable checkpoint
            log.warning("[stats] could not read %s for stored statistics: %s", ckpt_path, exc)
            return None
        saved: typing.Any = (ckpt.get("hyper_parameters") or {}).get("config")
        if saved is not None:
            return saved if OmegaConf.is_config(saved) else OmegaConf.create(saved)
    return None


def stored_values(saved_cfg: DictConfig) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    """Return ``(values, fingerprint)`` from a saved run config.

    The fingerprint is that of the data the saved run last trained on
    (``provenance.trained_on``, falling back to ``provenance.dataset``).

    Configs written since dataset statistics were recorded carry a
    ``dataset_stats`` block.  Older ones only have the values in
    ``model.backbone``; those are recovered too, without a fingerprint.
    """
    block: typing.Any = saved_cfg.get("dataset_stats")
    if block is not None and block.get("values") is not None:
        values: dict[str, typing.Any] = dict(_plain(block["values"]))
        provenance: typing.Any = block.get("provenance") or {}
        fingerprint: dict[str, typing.Any] = dict(
            _plain(provenance.get("trained_on") or provenance.get("dataset") or {})
        )
        return values, fingerprint
    backbone: typing.Any = (saved_cfg.get("model") or {}).get("backbone") or {}
    return collect(backbone), {}


# ----------------------------------------------------------------------
# What a changed statistic means for a resumed model
# ----------------------------------------------------------------------

#: Statistics held in backbone *buffers*.  Lightning restores buffers from
#: the checkpoint, so a recomputed value only takes effect if the restore is
#: told to keep the freshly built one (``GOALModule.keep_fresh_buffers``).
BUFFER_STATS: frozenset[str] = frozenset({"scale", "atomic_energies"})

#: Statistics that decide parameter shapes or which modules exist.  A model
#: rebuilt with different values cannot load the checkpoint's weights.
STRUCTURAL_STATS: frozenset[str] = frozenset({"elements", "pair_counts"})


def uncovered_elements(model_elements: typing.Any, dataset: typing.Any) -> list[int]:
    """Return atomic numbers in ``dataset`` that the model has no slot for.

    Parameters
    ----------
    model_elements:
        The ``elements`` list the model is built with.
    dataset:
        Training dataset, scanned with
        :func:`goal.ml.data.statistics.compute_unique_elements`.
    """
    from goal.ml.data.statistics import compute_unique_elements

    known: set[int] = {int(z) for z in (_plain(model_elements) or [])}
    return sorted(set(compute_unique_elements(dataset)) - known)


def structural_mismatches(module: typing.Any, ckpt_path: str) -> list[str]:
    """Compare a freshly built module's tensors with a checkpoint's.

    Returns
    -------
    list of str
        One line per state-dict entry that is missing, unexpected, or has a
        different shape.  Empty when the checkpoint's weights will load.
    """
    import torch

    ckpt: dict[str, typing.Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved: dict[str, typing.Any] = ckpt.get("state_dict") or {}
    own: dict[str, typing.Any] = module.state_dict()
    problems: list[str] = []
    for key in sorted(set(own) | set(saved)):
        if key not in saved:
            problems.append(f"{key}: not in the checkpoint")
        elif key not in own:
            problems.append(f"{key}: in the checkpoint but not in the rebuilt model")
        elif tuple(own[key].shape) != tuple(saved[key].shape):
            problems.append(
                f"{key}: shape {tuple(saved[key].shape)} in the checkpoint, "
                f"{tuple(own[key].shape)} now"
            )
    return problems
