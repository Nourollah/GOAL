"""ASE Calculator interface for trained GOAL models.

Wraps any trained ``GOALModule`` checkpoint into a standard ASE
``Calculator`` so it can be used for:

- Single-point energy / force / stress calculations
- Geometry optimisation (``ase.optimize``)
- Molecular dynamics (``ase.md``)
- Nudged elastic band (NEB) transition-state searches
- Phonon calculations (``ase.phonons``)

Usage::

    from goal.ml.utils.calculator import GOALCalculator

    # From a checkpoint file
    calc = GOALCalculator(checkpoint_path="logs/train/runs/.../last.ckpt")

    # From an already-loaded module
    calc = GOALCalculator(module=my_module, cutoff=5.0)

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


class GOALCalculator(Calculator):
    """ASE Calculator backed by a trained GOAL model.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to a Lightning checkpoint (``.ckpt``).  Mutually exclusive
        with ``module``.
    module : GOALModule, optional
        An already-instantiated ``GOALModule``.  Mutually exclusive with
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

    implemented_properties: typing.ClassVar[list[str]] = [
        "energy",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        module: typing.Any | None = None,
        cutoff: float | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        head: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)

        if checkpoint_path is not None and module is not None:
            raise ValueError("Provide either 'checkpoint_path' or 'module', not both.")
        if checkpoint_path is None and module is None:
            raise ValueError("Provide either 'checkpoint_path' or 'module'.")

        self.device: torch.device = torch.device(device)
        self.dtype: torch.dtype = dtype
        self.head: str | None = head

        if checkpoint_path is not None:
            self._module, self._cutoff = self._load_checkpoint(checkpoint_path)
        else:
            if cutoff is None:
                raise ValueError("'cutoff' is required when providing a module directly.")
            self._module = module
            self._cutoff = float(cutoff)

        self._module = self._module.to(self.device)
        self._module.eval()

    def _load_checkpoint(
        self,
        path: str | Path,
    ) -> tuple[typing.Any, float]:
        """Load a GOALModule from a Lightning checkpoint.

        Extracts the cutoff from the saved Hydra config stored in the
        checkpoint's ``hyper_parameters``.
        """
        from goal.ml.training.module import GOALModule

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        module = GOALModule.load_from_checkpoint(
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
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
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

        from goal.ml.data.graph import AtomicGraph

        graph: AtomicGraph = AtomicGraph.from_ase(
            self.atoms,
            cutoff=self._cutoff,
            dtype=self.dtype,
            head=self.head,
        )
        graph = graph.to(self.device)

        predictions: dict[str, torch.Tensor] = self._module(graph)

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
            f"GOALCalculator(cutoff={self._cutoff}, " f"device={self.device}, dtype={self.dtype})"
        )
