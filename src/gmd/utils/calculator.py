"""ASE Calculator interface for trained GMD models.

Wraps any trained ``GMDModule`` checkpoint into a standard ASE
``Calculator`` so it can be used for:

- Single-point energy / force / stress calculations
- Geometry optimisation (``ase.optimize``)
- Molecular dynamics (``ase.md``)
- Nudged elastic band (NEB) transition-state searches
- Phonon calculations (``ase.phonons``)

Usage::

    from gmd.utils.calculator import GMDCalculator

    # From a checkpoint file
    calc = GMDCalculator(checkpoint_path="logs/train/runs/.../last.ckpt")

    # From an already-loaded module
    calc = GMDCalculator(module=my_module, cutoff=5.0)

    # Attach to ASE Atoms and compute
    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.calc = calc
    print(atoms.get_potential_energy())
    print(atoms.get_forces())

    # Run MD
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.langevin import Langevin
    from ase import units
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01)
    dyn.run(100)
"""

from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress


class GMDCalculator(Calculator):
    """ASE Calculator backed by a trained GMD model.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to a Lightning checkpoint (``.ckpt``).  Mutually exclusive
        with ``module``.
    module : GMDModule, optional
        An already-instantiated ``GMDModule``.  Mutually exclusive with
        ``checkpoint_path``.
    cutoff : float, optional
        Neighbour-list cutoff in Ångström.  Required when ``module`` is
        provided directly.  When loading from checkpoint, read from the
        saved config automatically.
    device : str
        Torch device — ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
    dtype : torch.dtype
        Precision for atomic positions and cell.  Should match the model
        training dtype (usually ``torch.float64``).
    head : str or None
        Multi-head tag for models trained with multiple heads.
    **kwargs
        Forwarded to ``ase.calculators.calculator.Calculator.__init__``.
    """

    implemented_properties: typing.ClassVar[typing.List[str]] = [
        "energy",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        checkpoint_path: typing.Optional[typing.Union[str, Path]] = None,
        module: typing.Optional[typing.Any] = None,
        cutoff: typing.Optional[float] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        head: typing.Optional[str] = None,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)

        if checkpoint_path is not None and module is not None:
            raise ValueError(
                "Provide either 'checkpoint_path' or 'module', not both."
            )
        if checkpoint_path is None and module is None:
            raise ValueError(
                "Provide either 'checkpoint_path' or 'module'."
            )

        self.device: torch.device = torch.device(device)
        self.dtype: torch.dtype = dtype
        self.head: typing.Optional[str] = head

        if checkpoint_path is not None:
            self._module, self._cutoff = self._load_checkpoint(checkpoint_path)
        else:
            if cutoff is None:
                raise ValueError(
                    "'cutoff' is required when providing a module directly."
                )
            self._module = module
            self._cutoff = float(cutoff)

        self._module = self._module.to(self.device)
        self._module.eval()

    def _load_checkpoint(
        self,
        path: typing.Union[str, Path],
    ) -> typing.Tuple[typing.Any, float]:
        """Load a GMDModule from a Lightning checkpoint.

        Extracts the cutoff from the saved Hydra config stored in the
        checkpoint's ``hyper_parameters``.
        """
        from gmd.training.module import GMDModule

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        module = GMDModule.load_from_checkpoint(
            str(path),
            map_location=self.device,
        )

        # Extract cutoff from saved config
        cutoff: float = float(
            module.config.data.get(
                "cutoff",
                module.config.model.backbone.get("cutoff", 5.0),
            )
        )
        return module, cutoff

    @torch.no_grad()
    def calculate(
        self,
        atoms: typing.Optional[Atoms] = None,
        properties: typing.Optional[typing.List[str]] = None,
        system_changes: typing.List[str] = all_changes,
    ) -> None:
        """Calculate energy, forces, and/or stress for the given Atoms.

        This method is called automatically by ASE when you access
        ``atoms.get_potential_energy()``, ``atoms.get_forces()``, etc.
        """
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        if self.atoms is None:
            raise RuntimeError("No atoms object set on calculator.")

        from gmd.data.graph import AtomicGraph

        graph: AtomicGraph = AtomicGraph.from_ase(
            self.atoms,
            cutoff=self._cutoff,
            dtype=self.dtype,
            head=self.head,
        )
        graph = graph.to(self.device)

        predictions: typing.Dict[str, torch.Tensor] = self._module(graph)

        # Energy — scalar per structure
        if "energy" in predictions:
            energy: torch.Tensor = predictions["energy"]
            self.results["energy"] = energy.detach().cpu().item()

        # Forces — (N, 3) per atom
        if "forces" in predictions:
            forces: torch.Tensor = predictions["forces"]
            self.results["forces"] = forces.detach().cpu().numpy()

        # Stress — (3, 3) → Voigt (6,) in eV/Å³
        if "stress" in predictions:
            stress: torch.Tensor = predictions["stress"]
            stress_np: np.ndarray = stress.detach().cpu().numpy()
            # ASE expects Voigt notation with sign convention: positive = compressive
            if stress_np.shape == (3, 3):
                self.results["stress"] = full_3x3_to_voigt_6_stress(stress_np)
            elif stress_np.shape == (6,):
                self.results["stress"] = stress_np
            else:
                self.results["stress"] = stress_np.flatten()[:6]

    def __repr__(self) -> str:
        return (
            f"GMDCalculator(cutoff={self._cutoff}, "
            f"device={self.device}, dtype={self.dtype})"
        )
