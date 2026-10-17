"""Label key-mapping for datasets read from ASE files.

Different datasets store energy/forces/stress under different keys (``energy``
vs ``REF_energy`` vs ``DFT_energy``; ``forces`` vs ``REF_forces``; …). Like
MACE and FairChem, GOAL lets you **map** the keys used in *your* file onto the
canonical property names GOAL trains on, via a single ``key_mapping`` dict::

    data:
      key_mapping:
        energy: REF_energy     # atoms.info key holding total energy
        forces: REF_forces     # atoms.arrays key holding per-atom forces
        stress: virial         # atoms.info key holding the stress tensor

The older per-key form (``energy_key``/``forces_key``/``stress_key``) still
works and is equivalent; ``key_mapping`` overrides it.

Resolution order for each property:
1. the mapped key in ``atoms.info`` (scalars/stress) or ``atoms.arrays`` (forces);
2. the ASE calculator (``get_potential_energy`` / ``get_forces`` / ``get_stress``);
3. otherwise ``None`` — unless the key was *explicitly configured*, in which
   case a clear error is raised naming the keys actually present in the file
   (so a wrong key fails loudly instead of silently dropping the label).
"""

from __future__ import annotations

import typing

# Canonical GOAL property → default key in the file.
DEFAULT_LABEL_KEYS: dict[str, str] = {
    "energy": "energy",
    "forces": "forces",
    "stress": "stress",
}


def resolve_label_keys(
    energy_key: str | None = None,
    forces_key: str | None = None,
    stress_key: str | None = None,
    key_mapping: typing.Mapping[str, str] | None = None,
) -> tuple[dict[str, str], set[str]]:
    """Merge per-key args and a ``key_mapping`` dict into one mapping.

    Returns ``(mapping, explicit)`` where ``mapping`` maps every canonical
    property to the file key to read, and ``explicit`` is the set of
    properties the user *configured* (used to decide whether a missing key is
    an error or just an absent-and-optional label).
    """
    mapping: dict[str, str] = dict(DEFAULT_LABEL_KEYS)
    explicit: set[str] = set()

    for prop, key in (("energy", energy_key), ("forces", forces_key), ("stress", stress_key)):
        if key is not None:
            mapping[prop] = str(key)
            explicit.add(prop)

    if key_mapping:
        # Accept plain dict or OmegaConf DictConfig.
        for prop, key in dict(key_mapping).items():
            if key is None:
                continue
            mapping[str(prop)] = str(key)
            explicit.add(str(prop))

    return mapping, explicit


def _from_calc(atoms: typing.Any, getter: typing.Callable[[typing.Any], typing.Any]) -> typing.Any:
    """Best-effort read from the attached ASE calculator; ``None`` on failure."""
    if getattr(atoms, "calc", None) is None:
        return None
    try:
        return getter(atoms)
    except Exception:  # noqa: BLE001 — any calculator failure means "not available"
        return None


def _resolve_property(
    atoms: typing.Any,
    prop: str,
    key: str,
    explicit: set[str],
    store: typing.Mapping[str, typing.Any],
    calc_getter: typing.Callable[[typing.Any], typing.Any],
) -> typing.Any:
    value = store.get(key)
    if value is not None:
        return value
    value = _from_calc(atoms, calc_getter)
    if value is not None:
        return value
    if prop in explicit:
        raise ValueError(
            f"Label '{prop}' not found: the configured key '{key}' is not in "
            f"atoms.info/atoms.arrays and no ASE calculator provides it.\n"
            f"  available atoms.info keys : {sorted(atoms.info.keys())}\n"
            f"  available atoms.arrays keys: {sorted(atoms.arrays.keys())}\n"
            f"Set data.key_mapping.{prop} (or data.{prop}_key) to the correct key."
        )
    return None


def extract_ase_labels(
    atoms: typing.Any,
    mapping: dict[str, str],
    explicit: set[str],
) -> dict[str, typing.Any]:
    """Extract ``{energy, forces, stress}`` from an ASE ``Atoms`` per ``mapping``.

    ``forces`` are read from ``atoms.arrays`` (per-atom); ``energy``/``stress``
    from ``atoms.info``.  Each falls back to the ASE calculator, then errors if
    the key was explicitly configured but nothing was found.
    """
    return {
        "energy": _resolve_property(
            atoms,
            "energy",
            mapping["energy"],
            explicit,
            atoms.info,
            lambda a: a.get_potential_energy(),
        ),
        "forces": _resolve_property(
            atoms,
            "forces",
            mapping["forces"],
            explicit,
            atoms.arrays,
            lambda a: a.get_forces(),
        ),
        "stress": _resolve_property(
            atoms,
            "stress",
            mapping["stress"],
            explicit,
            atoms.info,
            lambda a: a.get_stress(voigt=False),
        ),
    }
