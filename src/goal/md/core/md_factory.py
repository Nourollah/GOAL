"""Molecular dynamics factory for creating dynamics/integrators."""

from __future__ import annotations

import abc
import typing

import torch
from ase import Atoms, units
from ase.md.langevin import Langevin

try:
    from flashmd import get_pretrained
    from flashmd.ase.langevin import Langevin as FlashMDLangevin

    HAS_FLASHMD = True
except ImportError:
    HAS_FLASHMD = False


class DynamicsFactory:
    """Factory for creating molecular dynamics integrators.

    Supports registration of custom dynamics builders.

    Examples
    --------
    >>> dyn = DynamicsFactory.create(
    ...     "langevin_ase",
    ...     molecule=atoms,
    ...     temperature_K=300.0,
    ...     timestep=1.0
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
        timestep: float = 1.0,
        temperature_K: float = 300.0,
        friction: float = 0.5,
        fixcm: bool = True,
        **kwargs: typing.Any,
    ) -> Langevin:
        """Create ASE Langevin dynamics.

        Parameters
        ----------
        atoms : ase.Atoms
            System to simulate
        timestep : float
            Timestep in fs (default: 1.0)
        temperature_K : float
            Temperature in Kelvin (default: 300)
        friction : float
            Friction coefficient (default: 0.5)
        fixcm : bool
            Fix center of mass (default: True)
        **kwargs
            Additional arguments for Langevin

        Returns
        -------
        ase.md.langevin.Langevin
            Dynamics integrator
        """
        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature_K,
            friction=friction / units.fs,
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
            raise ImportError("FlashMD not installed. Install with: pip install flashmd")

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
