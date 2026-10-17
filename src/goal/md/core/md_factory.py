"""Molecular dynamics factory for creating dynamics/integrators."""

from __future__ import annotations

import abc
import typing

import torch
from ase import Atoms, units
from ase.md.langevin import Langevin

# Any failure, not just ImportError: an installed-but-broken flashmd stack
# must not take ``import goal.md`` down with it.  (Seen with torch 2.14 /
# Python 3.14: once torch_geometric is imported, metatensor-operations'
# TorchScript compilation raises RuntimeError — so importing goal.ml before
# goal.md broke the whole MD module.)
try:
    from flashmd import get_pretrained
    from flashmd.ase.langevin import Langevin as FlashMDLangevin

    HAS_FLASHMD = True
except Exception as _exc:  # noqa: BLE001
    HAS_FLASHMD = False
    _FLASHMD_IMPORT_ERROR: Exception | None = _exc
else:
    _FLASHMD_IMPORT_ERROR = None


class DynamicsFactory:
    """Factory for creating molecular dynamics integrators.

    Supports registration of custom dynamics builders.

    Examples
    --------
    >>> dyn = DynamicsFactory.create(
    ...     "langevin_ase",
    ...     atoms=atoms,
    ...     temperature_K=300.0,
    ...     timestep_fs=0.5,
    ... )
    """

    _builders: dict[str, DynamicsBuilder] = {}

    @classmethod
    def register(cls, key: str, builder: DynamicsBuilder) -> None:
        """Register a dynamics builder."""
        cls._builders[key] = builder

    @classmethod
    def create(cls, key: str, *args: typing.Any, **kwargs: typing.Any) -> object:
        """Create dynamics integrator using registered builder.

        Parameters
        ----------
        key : str
            Builder identifier (e.g., "langevin_ase", "langevin_flashmd")
        *args, **kwargs
            Arguments passed to builder

        Returns
        -------
        object
            Dynamics integrator object

        Raises
        ------
        ValueError
            If builder not found
        """
        builder = cls._builders.get(key)
        if builder is None:
            available = ", ".join(cls._builders.keys())
            raise ValueError(
                f"Dynamics builder '{key}' not registered. " f"Available: {available}"
            )
        return builder.build(*args, **kwargs)


class DynamicsBuilder(abc.ABC):
    """Abstract base class for dynamics builders."""

    @abc.abstractmethod
    def build(self, *args: typing.Any, **kwargs: typing.Any) -> object:
        """Build dynamics integrator from parameters."""
        raise NotImplementedError


def register_dynamics(
    key: str,
) -> typing.Callable[[type[DynamicsBuilder]], type[DynamicsBuilder]]:
    """Decorator to register dynamics builder.

    Parameters
    ----------
    key : str
        Identifier for the builder
    """

    def decorator(
        builder_cls: type[DynamicsBuilder],
    ) -> type[DynamicsBuilder]:
        instance = builder_cls()
        DynamicsFactory.register(key, instance)
        return builder_cls

    return decorator


@register_dynamics("langevin_ase")
class LangevinASEBuilder(DynamicsBuilder):
    """Create standard ASE Langevin thermostat."""

    def build(
        self,
        atoms: Atoms,
        timestep: float | None = None,
        temperature_K: float = 300.0,
        friction: float | None = None,
        fixcm: bool = True,
        timestep_fs: float | None = None,
        friction_per_fs: float | None = None,
        **kwargs: typing.Any,
    ) -> Langevin:
        """Create ASE Langevin dynamics.

        Time-like arguments come in two spellings.  The ``_fs`` ones are in
        femtoseconds and are what configs should use; the bare ones are
        passed to ASE untouched, i.e. in **ASE units** (1 ASE time unit ≈
        10.18 fs), exactly like ``ase.md.langevin.Langevin`` itself.

        Parameters
        ----------
        atoms : ase.Atoms
            System to simulate
        timestep : float, optional
            Timestep in ASE units, e.g. ``0.5 * ase.units.fs``.
        temperature_K : float
            Temperature in Kelvin (default: 300)
        friction : float, optional
            Friction in inverse ASE time units, e.g. ``0.01 / ase.units.fs``.
        fixcm : bool
            Fix center of mass (default: True)
        timestep_fs : float, optional
            Timestep in fs.  Default when neither timestep is given: 1.0 fs.
        friction_per_fs : float, optional
            Friction in 1/fs.  When neither friction is given the historic
            default of 0.5 ASE units (≈ 0.049 fs⁻¹) is used.
        **kwargs
            Additional arguments for Langevin

        Returns
        -------
        ase.md.langevin.Langevin
            Dynamics integrator
        """
        if timestep is not None and timestep_fs is not None:
            raise ValueError("Pass either 'timestep' (ASE units) or 'timestep_fs', not both.")
        if friction is not None and friction_per_fs is not None:
            raise ValueError("Pass either 'friction' (ASE units) or 'friction_per_fs', not both.")
        if timestep is None:
            timestep = (1.0 if timestep_fs is None else float(timestep_fs)) * units.fs
        if friction is None:
            friction = 0.5 if friction_per_fs is None else float(friction_per_fs) / units.fs

        dyn = Langevin(
            atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            friction=friction,
            fixcm=fixcm,
            **kwargs,
        )
        return dyn


@register_dynamics("langevin_flashmd")
class LangevinFlashMDBuilder(DynamicsBuilder):
    """Create FlashMD-accelerated Langevin thermostat."""

    def build(
        self,
        atoms: Atoms,
        timestep: float = 16,
        temperature_K: float = 300.0,
        time_constant: float = 100.0,
        model_name: str = "pet-omatpes-v2",
        device: str | None = None,
        **kwargs: typing.Any,
    ) -> object:
        """Create FlashMD Langevin dynamics.

        Parameters
        ----------
        atoms : ase.Atoms
            System to simulate
        timestep : float
            Timestep in fs (default: 16, optimized for FlashMD)
        temperature_K : float
            Temperature in Kelvin (default: 300)
        time_constant : float
            Thermostat time constant in fs (default: 100)
        model_name : str
            FlashMD model name (default: "pet-omatpes-v2")
        device : str
            Torch device ("cuda" or "cpu"). Auto-detected if None.
        **kwargs
            Additional arguments for FlashMD Langevin

        Returns
        -------
        flashmd.ase.langevin.Langevin
            Accelerated dynamics integrator

        Raises
        ------
        ImportError
            If FlashMD not installed
        """
        if not HAS_FLASHMD:
            if isinstance(_FLASHMD_IMPORT_ERROR, ImportError) or _FLASHMD_IMPORT_ERROR is None:
                raise ImportError("FlashMD not installed. Install with: pip install flashmd")
            raise ImportError(
                f"FlashMD is installed but failed to import: {_FLASHMD_IMPORT_ERROR!r}"
            ) from _FLASHMD_IMPORT_ERROR

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        energy_model, flashmd_model = get_pretrained(model_name, timestep)

        dyn = FlashMDLangevin(
            atoms=atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature_K,
            time_constant=time_constant * units.fs,
            model=flashmd_model,
            device=device,
            **kwargs,
        )
        return dyn
