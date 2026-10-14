"""Calculator factory for creating ASE calculators from various backends.

Supports:
- ML models trained in goal.ml (via checkpoint loading)
- Quantum chemistry: ORCA, CP2K, Quantum ESPRESSO, VASP, PSI4
- Pretrained models: MACE, NequIP, FlashMD, UPET
- Semi-empirical: xTB, TBLite
"""

from __future__ import annotations

import abc
import logging
import os
import shutil
import typing
import warnings
from pathlib import Path

import ase.calculators.calculator
import torch

from goal.md.calculators.base import GOALCalculatorNotFound, UnsupportedOperationError

log = logging.getLogger(__name__)

try:
    from ase.calculators.orca import ORCA, OrcaProfile, OrcaTemplate

    HAS_ORCA = True
except ImportError:
    HAS_ORCA = False

try:
    from ase.calculators.cp2k import CP2K

    HAS_CP2K = True
except ImportError:
    HAS_CP2K = False

try:
    from flashmd import get_pretrained
    from flashmd.ase import EnergyCalculator

    HAS_FLASHMD = True
except ImportError:
    HAS_FLASHMD = False

try:
    from mace.calculators import MACECalculator, mace_mp

    HAS_MACE = True
except ImportError:
    HAS_MACE = False

try:
    from xtb.ase.calculator import XTB

    HAS_XTB = True
except ImportError:
    HAS_XTB = False

try:
    from tblite.ase import TBLite

    HAS_TBLITE = True
except ImportError:
    HAS_TBLITE = False

try:
    import nequip  # noqa: F401

    HAS_NEQUIP = True
except ImportError:
    HAS_NEQUIP = False

try:
    from ase.calculators.espresso import Espresso, EspressoProfile

    HAS_ESPRESSO = True
except ImportError:
    Espresso = None  # type: ignore[assignment,misc]
    EspressoProfile = None  # type: ignore[assignment,misc]
    HAS_ESPRESSO = False

try:
    from ase.calculators.vasp import Vasp

    HAS_VASP = True
except ImportError:
    Vasp = None  # type: ignore[assignment,misc]
    HAS_VASP = False

try:
    from upet.calculator import UPETCalculator

    HAS_UPET = True
except ImportError:
    UPETCalculator = None  # type: ignore[assignment,misc]
    HAS_UPET = False

# Recognised UPET model identifiers.  Read from upet itself when available —
# the released families are NOT a family x size cross-product (there is no
# pet-mad-xl, and pet-omad / pet-omatpes exist), so a hand-maintained list
# rejects valid models and waves through invalid ones.  The fallback below is
# only for environments without upet, where nothing can be built anyway.
try:
    from upet._version import UPET_AVAILABLE_MODELS as _UPET_MODEL_LIST
except ImportError:  # pragma: no cover — exercised only without upet installed
    _UPET_MODEL_LIST = [
        "pet-mad-xs",
        "pet-mad-s",
        "pet-omat-xs",
        "pet-omat-s",
        "pet-omat-m",
        "pet-omat-l",
        "pet-omat-xl",
        "pet-oam-l",
        "pet-oam-xl",
        "pet-omad-xs",
        "pet-omad-s",
        "pet-omad-l",
        "pet-omatpes-l",
        "pet-spice-s",
        "pet-spice-l",
    ]

_UPET_MODELS: frozenset[str] = frozenset(_UPET_MODEL_LIST)


class CalculatorFactory:
    """Factory for creating ASE calculators.

    Supports registration of custom builders for any calculator backend.

    Examples
    --------
    >>> calc = CalculatorFactory.create("goal_model", checkpoint="model.ckpt")
    >>> calc = CalculatorFactory.create("flashmd")
    >>> calc = CalculatorFactory.create("orca", orca_path="/path/to/orca")
    """

    _builders: dict[str, CalculatorBuilder] = {}

    @classmethod
    def register(cls, key: str, builder: CalculatorBuilder) -> None:
        """Register a calculator builder."""
        cls._builders[key] = builder

    @classmethod
    def create(
        cls, key: str, *args: typing.Any, **kwargs: typing.Any
    ) -> ase.calculators.calculator.Calculator:
        """Create calculator using registered builder.

        Parameters
        ----------
        key : str
            Builder identifier
        *args, **kwargs
            Arguments passed to builder

        Returns
        -------
        ase.calculators.calculator.Calculator
            Calculator instance

        Raises
        ------
        ValueError
            If builder not found
        """
        builder = cls._builders.get(key)
        if builder is None:
            available = ", ".join(cls._builders.keys())
            raise ValueError(
                f"Calculator builder '{key}' not registered. " f"Available: {available}"
            )
        return builder.build(*args, **kwargs)


class CalculatorBuilder(abc.ABC):
    """Abstract base class for calculator builders."""

    @abc.abstractmethod
    def build(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> ase.calculators.calculator.Calculator:
        """Build calculator from parameters."""
        raise NotImplementedError


def register_calculator(
    key: str,
) -> typing.Callable[[type[CalculatorBuilder]], type[CalculatorBuilder]]:
    """Decorator to register calculator builder."""

    def decorator(
        builder_cls: type[CalculatorBuilder],
    ) -> type[CalculatorBuilder]:
        instance = builder_cls()
        CalculatorFactory.register(key, instance)
        return builder_cls

    return decorator


# ============================================================================
# ML Models from goal.ml
# ============================================================================


@register_calculator("goal_model")
class GOALModelBuilder(CalculatorBuilder):
    """Load trained models from goal.ml checkpoints."""

    def build(
        self,
        checkpoint: str | Path,
        cutoff: float | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        head: str | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Load goal.ml trained model as ASE calculator.

        Parameters
        ----------
        checkpoint : str or Path
            Path to Lightning checkpoint (.ckpt)
        cutoff : float, optional
            Cutoff in Ångströms (read from config if None)
        device : str
            Torch device ("cpu" or "cuda")
        dtype : torch.dtype
            Precision (torch.float64 default)
        head : str, optional
            Multi-head identifier

        Returns
        -------
        GOALCalculator
            ASE-compatible calculator wrapping trained model
        """
        from goal.ml.utils.calculator import GOALCalculator

        calc = GOALCalculator(
            checkpoint_path=checkpoint,
            cutoff=cutoff,
            device=device,
            dtype=dtype,
            head=head,
        )
        return calc


# ============================================================================
# FlashMD
# ============================================================================


@register_calculator("flashmd")
class FlashMDBuilder(CalculatorBuilder):
    """Create FlashMD pretrained model calculator."""

    def build(
        self,
        model_name: str = "pet-omatpes-v2",
        device: str | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Create FlashMD calculator.

        Parameters
        ----------
        model_name : str
            Model identifier (default: "pet-omatpes-v2")
        device : str
            Torch device. Auto-detects CUDA if None.

        Returns
        -------
        flashmd.ase.EnergyCalculator
            FlashMD calculator

        Raises
        ------
        ImportError
            If FlashMD not installed
        """
        if not HAS_FLASHMD:
            raise ImportError("FlashMD not installed. Install with: pip install flashmd")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        energy_model, _ = get_pretrained(model_name, 16)
        calc = EnergyCalculator(energy_model, device=device)
        return calc


# ============================================================================
# MACE
# ============================================================================


@register_calculator("mace")
class MACEBuilder(CalculatorBuilder):
    """Create MACE model calculator."""

    def build(
        self,
        model_name_or_path: str = "small",
        device: str | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Create MACE calculator.

        Parameters
        ----------
        model_name_or_path : str
            "small"/"medium"/"large" (universal) or path to checkpoint
        device : str
            Torch device. Auto-detects CUDA if None.

        Returns
        -------
        mace.calculators.MACECalculator
            MACE calculator

        Raises
        ------
        ImportError
            If MACE not installed
        """
        if not HAS_MACE:
            raise ImportError("MACE not installed. Install with: pip install mace-torch")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_name_or_path in ["small", "medium", "large"]:
            calc = mace_mp(model_name_or_path, device=device)
        else:
            calc = MACECalculator(model_name_or_path, device=device)

        return calc


# ============================================================================
# UPET (metatensor PET foundation models)
# ============================================================================


@register_calculator("upet")
class UPETBuilder(CalculatorBuilder):
    """Create a UPET (metatensor PET) foundation-model calculator.

    Recognised model identifiers come from ``upet`` itself
    (``upet._version.UPET_AVAILABLE_MODELS``) — e.g. ``pet-mad-s``,
    ``pet-omat-l``, ``pet-spice-s``, ``pet-omad-l``.

    Also accepts a fine-tuned ``checkpoint_path`` (a metatrain ``.ckpt``),
    in which case ``variants={"energy": "finetune"}`` selects the fine-tuned
    head (see :mod:`goal.ml.cli.finetune_upet`).
    """

    def build(
        self,
        model: str = "pet-mad-s",
        checkpoint_path: str | None = None,
        variants: dict[str, str] | None = None,
        version: str | None = None,
        device: str = "cuda",
        non_conservative: bool = False,
    ) -> ase.calculators.calculator.Calculator:
        """Create a UPET calculator.

        Parameters
        ----------
        model : str
            One of the identifiers in ``upet._version.UPET_AVAILABLE_MODELS``.
        checkpoint_path : str, optional
            Path to a fine-tuned metatrain checkpoint.  When given it takes
            precedence over ``model``.
        variants : dict, optional
            Head selection *within* an output, e.g. ``{"energy": "r2scan"}``
            to pick the ``energy/r2scan`` head, or ``{"energy": "finetune"}``
            after ``goal-finetune-upet``.  ``None`` (the default) uses each
            output's default head — do NOT pass ``{"energy": "energy"}``,
            which asks for a head literally named ``energy/energy`` and
            raises ``ValueError`` in upet >= 0.2.
        version : str, optional
            Pin a released weight version (e.g. ``"1.5.0"``).  ``None``
            resolves to the latest, which makes a run non-reproducible once
            upstream publishes new weights — pin it for anything you intend
            to cite.
        device : str
            Torch device.
        non_conservative : bool
            If ``True``, use a direct (non-conservative) force head.  This
            **violates energy conservation** and is unsafe for long MD; a loud
            warning is emitted and it is never the default.  Unsupported by
            PET-SPICE and PET-MAD < 1.1.0.

        Raises
        ------
        GOALCalculatorNotFound
            If ``upet`` is not installed.
        ValueError
            If ``model`` is not a recognised identifier.
        """
        if not HAS_UPET:
            raise GOALCalculatorNotFound(
                "UPET is not installed. Install with: pip install upet. "
                "See docs/calculators/upet.md."
            )

        # Validate the model identifier unless a checkpoint overrides it.
        if checkpoint_path is None:
            self._validate_model_name(model)

        if non_conservative:
            msg = (
                "UPET non_conservative=True selects a DIRECT force head that "
                "VIOLATES ENERGY CONSERVATION. Do not use this for production "
                "MD (energy drift, broken thermostats). Use only for fast "
                "single-point screening where conservation is irrelevant."
            )
            log.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

        # Pass `variants` through untouched: upet treats None as "each output's
        # default head", and there is no sensible non-None default to invent.
        kwargs: dict[str, typing.Any] = {
            "device": device,
            "variants": dict(variants) if variants else None,
            "non_conservative": non_conservative,
        }
        if version is not None:
            kwargs["version"] = version
        if checkpoint_path is not None:
            kwargs["checkpoint_path"] = checkpoint_path
        else:
            kwargs["model"] = model
        return UPETCalculator(**kwargs)

    @staticmethod
    def _validate_model_name(model: str) -> None:
        """Raise ``ValueError`` listing the valid identifiers if ``model`` is unknown."""
        if model.lower() not in _UPET_MODELS:
            raise ValueError(
                f"Unknown UPET model {model!r}. Valid models: "
                f"{', '.join(sorted(_UPET_MODELS))}."
            )


# ============================================================================
# Quantum Chemistry
# ============================================================================


@register_calculator("orca")
class ORCABuilder(CalculatorBuilder):
    """Create ORCA quantum chemistry calculator."""

    def build(
        self,
        orca_path: str,
        keywords: str = "EnGrad wB97M-V def2-TZVPD RIJCOSX TightSCF DEFGRID3",
        charge: int = 0,
        multiplicity: int = 1,
        num_threads: int = 1,
        directory: str = "orca_files",
    ) -> ase.calculators.calculator.Calculator:
        """Create ORCA calculator.

        Parameters
        ----------
        orca_path : str
            Path to ORCA executable
        keywords : str
            ORCA input keywords
        charge : int
            Molecular charge
        multiplicity : int
            Spin multiplicity
        num_threads : int
            Number of threads
        directory : str
            Output directory

        Returns
        -------
        ase.calculators.orca.ORCA
            ORCA calculator

        Raises
        ------
        ImportError
            If ASE ORCA support not available
        """
        if not HAS_ORCA:
            raise ImportError("ORCA support not available. Check ASE installation.")

        profile = OrcaProfile(command=orca_path)
        calc = ORCA(
            template=OrcaTemplate(
                keywords=keywords,
                charge=charge,
                multiplicity=multiplicity,
            ),
            profile=profile,
            num_threads=num_threads,
            directory=directory,
        )
        return calc


#: Maps friendly functional names to the explicit libxc functional pairs that
#: CP2K's SIRIUS backend understands.  SIRIUS resolves functionals through
#: libxc, so the native CP2K shortcut ``&PBE`` (passed as ``XC_PBE``) is *not*
#: recognised — it must be split into exchange + correlation libxc names.
_SIRIUS_XC_MAP: dict[str, str] = {
    "PBE": "XC_GGA_X_PBE XC_GGA_C_PBE",
    "PBESOL": "XC_GGA_X_PBE_SOL XC_GGA_C_PBE_SOL",
    "BLYP": "XC_GGA_X_B88 XC_GGA_C_LYP",
    "BP": "XC_GGA_X_B88 XC_GGA_C_P86",
    "LDA": "XC_LDA_X XC_LDA_C_PZ",
    "PADE": "XC_LDA_X XC_LDA_C_PZ",
    "PZ": "XC_LDA_X XC_LDA_C_PZ",
}

#: Canonical DFT-D3(BJ) rational-damping parameters (2-body) per functional —
#: the same values Grimme/ORCA use.  Kept explicit so the dispersion correction
#: is self-documenting and provably matches the reference; ``simple-dftd3``'s
#: ``method=`` lookup yields identical numbers for these functionals.
_D3BJ_PARAMS: dict[str, dict[str, float]] = {
    # s6, s8, a1, a2 (see Grimme et al. J. Comput. Chem. 2011, 32, 1456).
    "PBE": {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407},
    "PBE0": {"s6": 1.0, "s8": 1.2177, "a1": 0.4145, "a2": 4.8593},
    "BLYP": {"s6": 1.0, "s8": 2.6996, "a1": 0.4298, "a2": 4.2359},
    "B3LYP": {"s6": 1.0, "s8": 1.9889, "a1": 0.3981, "a2": 4.4211},
    "TPSS": {"s6": 1.0, "s8": 1.9435, "a1": 0.4535, "a2": 4.4752},
}

#: Plane-wave (SIRIUS) ``&PW_DFT`` block, merged into ASE's generated input.
#: Only the ``&PW_DFT`` section is templated — ASE fills in ``METHOD SIRIUS``,
#: the ``&DFT/&XC`` functional, the ``&SUBSYS`` cell/coords, and one
#: ``&KIND`` per element (``POTENTIAL GTH-<xc>``, no Gaussian ``BASIS_SET``).
#: Values mirror CP2K's own ``tests/SIRIUS/regtest-1`` molecular examples.
#: VERBOSITY 0 keeps SIRIUS's (stdout) chatter minimal — it collides with ASE's
#: shell protocol and is filtered out by the cp2k_shell wrapper regardless.
_SIRIUS_PW_DFT_TEMPLATE = """\
&FORCE_EVAL
  &PW_DFT
    &CONTROL
      PROCESSING_UNIT {processing_unit}
      VERBOSITY 0
    &END CONTROL
    &PARAMETERS
      ELECTRONIC_STRUCTURE_METHOD pseudopotential
      SMEARING_WIDTH {smearing_width}
      USE_SYMMETRY FALSE
      NUM_MAG_DIMS {num_mag_dims}
      GK_CUTOFF {gk_cutoff}
      PW_CUTOFF {pw_cutoff}
      NGRIDK {kx} {ky} {kz}
      NUM_DFT_ITER {num_dft_iter}
      ENERGY_TOL {energy_tol}
      DENSITY_TOL {density_tol}
    &END PARAMETERS
    &ITERATIVE_SOLVER
      TYPE davidson
      CONVERGE_BY_ENERGY 1
      ENERGY_TOLERANCE {solver_energy_tol}
      NUM_STEPS 20
      SUBSPACE_SIZE 4
    &END ITERATIVE_SOLVER
    &MIXER
      TYPE broyden2
      BETA {mixer_beta}
      MAX_HISTORY 8
    &END MIXER
  &END PW_DFT
&END FORCE_EVAL
"""


@register_calculator("cp2k")
class CP2KBuilder(CalculatorBuilder):
    """Create a CP2K quantum-chemistry calculator via ASE's ``cp2k_shell`` mode.

    The ASE ``CP2K`` calculator drives CP2K through a persistent
    ``cp2k_shell`` subprocess.  With MPI builds, streaming large structures
    through the shell's stdin can stall; the fix is ``set_pos_file=True``
    (CP2K >= 2024.2), which sends atomic positions via a temporary file
    instead of stdin.  It is the default here.

    Two electronic-structure ``method``s are supported:

    * ``"quickstep"`` (default) — CP2K's native **GPW** (Gaussian and plane
      waves): Kohn–Sham orbitals in a Gaussian ``basis_set`` with a
      plane-wave auxiliary density grid (``cutoff_ry``).
    * ``"sirius"`` (aliases ``"pw"`` / ``"plane_wave"``) — **pure plane-wave**
      pseudopotential DFT through CP2K's SIRIUS backend (``METHOD SIRIUS``,
      ``&PW_DFT``).  No Gaussian basis at all — the wavefunctions are plane
      waves, exactly like Quantum ESPRESSO / VASP.  Requires a SIRIUS-enabled
      CP2K build (the official ``cp2k/cp2k`` container is one).

    Two presets tune periodicity-related defaults:

    * ``"molecular"`` (default) — isolated molecule (or molecule-in-a-box):
      ``stress_tensor=False``; GPW uses ``poisson_solver="MT"``
      (Martyna–Tuckerman); SIRIUS uses a Γ-point grid (``kpts=(1,1,1)``).
    * ``"bulk"`` — periodic solid: ``stress_tensor=True`` and the periodic
      Poisson solver (GPW) / a k-point grid (SIRIUS).

    Any explicit keyword overrides the preset value.
    """

    def build(
        self,
        command: str | None = None,
        preset: str = "molecular",
        method: str = "quickstep",
        n_mpi: int | None = None,
        omp_threads: int | None = None,
        set_pos_file: bool = True,
        cutoff_ry: float = 400.0,
        xc: str = "PBE",
        basis_set: str = "DZVP-MOLOPT-SR-GTH",
        pseudo_potential: str = "GTH-PBE",
        basis_set_file: str = "BASIS_MOLOPT",
        potential_file: str = "GTH_POTENTIALS",
        stress_tensor: bool | None = None,
        poisson_solver: str | None = None,
        charge: int = 0,
        uks: bool = False,
        max_scf: int = 50,
        inp: str | None = None,
        input_file: str | None = None,
        print_level: str = "LOW",
        directory: str | None = None,
        pw_cutoff: float | None = None,
        gk_cutoff: float | None = None,
        kpts: tuple[int, int, int] | list[int] = (1, 1, 1),
        smearing_width: float = 0.001,
        num_dft_iter: int = 100,
        scf_energy_tol: float = 1.0e-6,
        scf_density_tol: float = 1.0e-6,
        solver_energy_tol: float = 1.0e-4,
        mixer_beta: float = 0.5,
        processing_unit: str = "cpu",
        dispersion: str | None = None,
        dispersion_atm: bool = False,
        **overrides: typing.Any,
    ) -> ase.calculators.calculator.Calculator:
        """Create a CP2K calculator.

        Parameters
        ----------
        command : str, optional
            CP2K shell binary (e.g. ``"cp2k_shell.psmp"``) or a full launch
            string.  ``None`` falls back to ``$ASE_CP2K_COMMAND`` and then to
            ``"cp2k_shell.psmp"``.  When ``n_mpi`` is set, an ``mpirun -n
            {n_mpi}`` prefix is added unless ``command`` already contains one.
        preset : str
            ``"molecular"`` or ``"bulk"``.
        method : str
            ``"quickstep"`` (GPW, Gaussian basis) or ``"sirius"`` /
            ``"pw"`` / ``"plane_wave"`` (pure plane-wave via SIRIUS).
        n_mpi, omp_threads : int, optional
            MPI ranks and OpenMP threads per rank.  ``omp_threads`` sets
            ``OMP_NUM_THREADS`` in the environment.
        set_pos_file : bool
            Send positions via a temp file rather than stdin (default True;
            requires CP2K >= 2024.2).
        cutoff_ry : float
            Plane-wave density cutoff in Rydberg.  For GPW it is the ``&MGRID``
            cutoff (converted to eV for ASE).  For SIRIUS it seeds the default
            ``pw_cutoff`` (``|G|_max = sqrt(cutoff_ry)`` bohr⁻¹) when the
            latter is not given explicitly.
        basis_set, pseudo_potential, basis_set_file, potential_file : str
            Basis/pseudopotential names and the data files that define them.
            In SIRIUS mode ``basis_set`` / ``basis_set_file`` are forced to
            ``None`` (plane waves need no Gaussian basis); only the
            ``pseudo_potential`` (default ``GTH-PBE``) and ``potential_file``
            are used.
        pw_cutoff, gk_cutoff : float, optional
            SIRIUS plane-wave cutoffs in bohr⁻¹ — ``pw_cutoff`` for the density
            (``|G|_max``), ``gk_cutoff`` for the wavefunctions (``|G+k|_max``).
            Defaults: ``pw_cutoff = sqrt(cutoff_ry)`` and
            ``gk_cutoff = pw_cutoff / 2`` (i.e. E_wfc = E_rho / 4).
        kpts : tuple, optional
            SIRIUS Monkhorst–Pack grid (``NGRIDK``).  ``(1, 1, 1)`` = Γ-point,
            correct for a molecule in a box.
        smearing_width, num_dft_iter, scf_energy_tol, scf_density_tol,
        solver_energy_tol, mixer_beta, processing_unit :
            SIRIUS ``&PW_DFT`` knobs.  Ignored unless ``method`` is SIRIUS.
            ``solver_energy_tol`` is the Davidson eigensolver's per-step
            ``ENERGY_TOLERANCE`` (tighten for accurate forces).
        dispersion : str, optional
            Add an empirical dispersion correction as an independent additive
            term (ASE ``SumCalculator``): ``"d3bj"`` = DFT-D3 with
            Becke–Johnson damping, using the canonical 2-body parameters for
            ``xc`` (matches ORCA's ``<func> D3BJ``).  ``None`` disables it.
            **Required for SIRIUS** because CP2K's ``&VDW_POTENTIAL`` is
            silently ignored under ``METHOD SIRIUS``.  Needs ``dftd3-python``.
        dispersion_atm : bool
            Include the D3 Axilrod–Teller–Muto 3-body term (``ABC``).  Default
            ``False`` (2-body only, ORCA's default).
        inp, input_file : str, optional
            Raw CP2K input (string, or a path to read).  Merged by ASE with
            the structured keywords above.  In SIRIUS mode, supplying ``inp``
            (or ``input_file``) replaces the auto-generated ``&PW_DFT`` block.

        Raises
        ------
        GOALCalculatorNotFound
            If ASE's CP2K support is unavailable or the binary is not found.
        """
        if not HAS_CP2K:
            raise GOALCalculatorNotFound(
                "CP2K support is unavailable — ASE could not import "
                "ase.calculators.cp2k.  Install a recent ASE (>= 3.23) and a "
                "CP2K build (>= 2024.2 for set_pos_file).  See docs/calculators/cp2k.md."
            )
        if preset not in ("molecular", "bulk"):
            raise ValueError(f"CP2K preset must be 'molecular' or 'bulk', got {preset!r}.")

        method_norm = method.lower().replace("-", "_")
        if method_norm in ("quickstep", "gpw"):
            is_sirius = False
        elif method_norm in ("sirius", "pw", "plane_wave", "planewave"):
            is_sirius = True
        else:
            raise ValueError(
                f"CP2K method must be 'quickstep' (GPW) or 'sirius'/'pw'/"
                f"'plane_wave', got {method!r}."
            )

        if input_file:
            inp_path = Path(input_file)
            if not inp_path.exists():
                raise GOALCalculatorNotFound(f"CP2K input file not found: {inp_path}")
            inp = inp_path.read_text()

        # OpenMP threads per rank.
        if omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(int(omp_threads))

        # Resolve the shell command.  ASE talks to `cp2k_shell`, so the binary
        # must be a *_shell flavour; we keep the historical symlink shim for
        # plain `cp2k.*` binaries.
        base_command = command or os.environ.get("ASE_CP2K_COMMAND") or "cp2k_shell.psmp"
        resolved = self._resolve_shell_binary(base_command)
        final_command = resolved
        if n_mpi is not None and "mpirun" not in final_command and "srun" not in final_command:
            final_command = f"mpirun -n {int(n_mpi)} {resolved}"

        # Preset-dependent defaults (explicit kwargs still win).
        if preset == "molecular":
            stress = False if stress_tensor is None else stress_tensor
            poisson = "MT" if poisson_solver is None else poisson_solver
        else:  # bulk
            stress = True if stress_tensor is None else stress_tensor
            poisson = "PERIODIC" if poisson_solver is None else poisson_solver

        import ase.units

        if is_sirius:
            params = self._sirius_params(
                final_command=final_command,
                set_pos_file=set_pos_file,
                xc=xc,
                pseudo_potential=pseudo_potential,
                potential_file=potential_file,
                stress=stress,
                charge=charge,
                uks=uks,
                print_level=print_level,
                inp=inp,
                cutoff_ry=cutoff_ry,
                pw_cutoff=pw_cutoff,
                gk_cutoff=gk_cutoff,
                kpts=kpts,
                smearing_width=smearing_width,
                num_dft_iter=num_dft_iter,
                scf_energy_tol=scf_energy_tol,
                scf_density_tol=scf_density_tol,
                solver_energy_tol=solver_energy_tol,
                mixer_beta=mixer_beta,
                processing_unit=processing_unit,
            )
        else:
            params = dict(
                command=final_command,
                set_pos_file=bool(set_pos_file),
                cutoff=cutoff_ry * ase.units.Rydberg,  # ASE expects eV
                xc=xc,
                basis_set=basis_set,
                basis_set_file=basis_set_file,
                pseudo_potential=pseudo_potential,
                potential_file=potential_file,
                stress_tensor=stress,
                poisson_solver=poisson,
                charge=charge,
                uks=uks,
                max_scf=max_scf,
                print_level=print_level,
                inp=inp,
            )
        params.update(overrides)
        if directory is not None:
            params["directory"] = directory

        try:
            cp2k_calc = CP2K(**params)
        except FileNotFoundError as exc:
            raise GOALCalculatorNotFound(
                f"CP2K executable not found ({final_command!r}). Set `command` or "
                f"$ASE_CP2K_COMMAND to a cp2k_shell binary. Original error: {exc}"
            ) from exc

        if not dispersion:
            return cp2k_calc

        # CP2K's own &VDW_POTENTIAL is silently ignored under METHOD SIRIUS, so
        # add dispersion as an independent additive ASE calculator.  D3(BJ) is a
        # geometry-only pairwise term, so summing it is exact.
        from ase.calculators.mixing import SumCalculator

        disp_calc = self._build_dispersion(dispersion, xc, dispersion_atm)
        return SumCalculator([cp2k_calc, disp_calc])

    @staticmethod
    def _build_dispersion(
        dispersion: str, xc: str, atm: bool
    ) -> ase.calculators.calculator.Calculator:
        """Build the standalone dispersion calculator (currently ``d3bj``)."""
        key = dispersion.lower().replace("-", "").replace("_", "")
        if key not in ("d3bj", "d3"):
            raise ValueError(f"Unsupported dispersion {dispersion!r}; only 'd3bj' is implemented.")
        try:
            from dftd3.ase import DFTD3
        except ImportError as exc:
            raise GOALCalculatorNotFound(
                "DFT-D3 dispersion needs the 'dftd3-python' package (Grimme's "
                "simple-dftd3).  Install it, e.g. `pixi add dftd3-python` or "
                "`pip install dftd3`."
            ) from exc

        func = xc.upper()
        if func in _D3BJ_PARAMS:
            # Explicit canonical 2-body params (self-documenting, matches ORCA).
            tweaks: dict[str, float] = dict(_D3BJ_PARAMS[func])
            tweaks["s9"] = 1.0 if atm else 0.0
            return DFTD3(damping="d3bj", params_tweaks=tweaks)
        # Fall back to simple-dftd3's built-in method lookup for other functionals.
        return DFTD3(damping="d3bj", params_tweaks={"method": xc.lower(), "atm": atm})

    @staticmethod
    def _sirius_params(
        *,
        final_command: str,
        set_pos_file: bool,
        xc: str,
        pseudo_potential: str,
        potential_file: str,
        stress: bool,
        charge: int,
        uks: bool,
        print_level: str,
        inp: str | None,
        cutoff_ry: float,
        pw_cutoff: float | None,
        gk_cutoff: float | None,
        kpts: tuple[int, int, int] | list[int],
        smearing_width: float,
        num_dft_iter: int,
        scf_energy_tol: float,
        scf_density_tol: float,
        solver_energy_tol: float,
        mixer_beta: float,
        processing_unit: str,
    ) -> dict[str, typing.Any]:
        """Assemble ASE ``CP2K`` kwargs for a pure plane-wave (SIRIUS) run.

        The wavefunctions are plane waves (no Gaussian basis): ``basis_set`` and
        ``basis_set_file`` are ``None`` so ASE emits neither a ``BASIS_SET`` per
        ``&KIND`` nor a ``BASIS_SET_FILE_NAME``.  ``cutoff`` (the GPW ``&MGRID``)
        and ``max_scf`` (the GPW ``&SCF``) are ``None`` — the plane-wave cutoffs
        and SCF loop live in the ``&PW_DFT`` block instead.  ``poisson_solver`` is
        ``None`` because SIRIUS treats the cell as periodic itself.
        """
        import math

        pw = pw_cutoff if pw_cutoff is not None else math.sqrt(cutoff_ry)
        gk = gk_cutoff if gk_cutoff is not None else pw / 2.0
        kx, ky, kz = (int(k) for k in kpts)

        # SIRIUS resolves XC through libxc; translate friendly names (e.g. "PBE")
        # to explicit libxc functional pairs.  Names already in libxc form
        # (containing "XC_") or unknown names are passed through untouched.
        xc_sirius = xc if "XC_" in xc.upper() else _SIRIUS_XC_MAP.get(xc.upper(), xc)

        if inp is None:
            inp = _SIRIUS_PW_DFT_TEMPLATE.format(
                processing_unit=processing_unit,
                smearing_width=smearing_width,
                num_mag_dims=1 if uks else 0,
                gk_cutoff=f"{gk:.8f}",
                pw_cutoff=f"{pw:.8f}",
                kx=kx,
                ky=ky,
                kz=kz,
                num_dft_iter=int(num_dft_iter),
                energy_tol=f"{scf_energy_tol:.3e}",
                density_tol=f"{scf_density_tol:.3e}",
                solver_energy_tol=f"{solver_energy_tol:.3e}",
                mixer_beta=mixer_beta,
            )

        return dict(
            command=final_command,
            set_pos_file=bool(set_pos_file),
            force_eval_method="SIRIUS",
            xc=xc_sirius,
            basis_set=None,
            basis_set_file=None,
            pseudo_potential=pseudo_potential,
            potential_file=potential_file,
            cutoff=None,
            max_scf=None,
            stress_tensor=stress,
            poisson_solver=None,
            charge=charge,
            uks=False,  # spin handled by SIRIUS NUM_MAG_DIMS, not DFT/UKS
            print_level=print_level,
            inp=inp,
        )

    @staticmethod
    def _resolve_shell_binary(cp2k_path: str) -> str:
        """Return a usable ``cp2k_shell`` command.

        If ``cp2k_path`` is already a shell flavour (or a compound launch
        string) it is returned unchanged.  For a plain ``cp2k.*`` binary we
        symlink it to ``cp2k_shell.*`` in the CWD (ASE requires the shell
        entry point), preserving the historical behaviour.
        """
        # Compound command (mpirun …, or contains spaces) — trust the caller.
        if " " in cp2k_path or "cp2k_shell" in cp2k_path:
            return cp2k_path
        real_binary = shutil.which(cp2k_path)
        if not real_binary:
            # Defer to ASE/CP2K to raise; keep the raw string so the error
            # message shows what was attempted.
            return cp2k_path
        if "cp2k_shell" in str(real_binary):
            return real_binary
        symlink_path = Path.cwd() / "cp2k_shell.ssmp"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        os.symlink(real_binary, symlink_path)
        return str(symlink_path.absolute())


@register_calculator("psi4")
class PSI4Builder(CalculatorBuilder):
    """Create PSI4 quantum chemistry calculator."""

    def build(self, **kwargs: typing.Any) -> ase.calculators.calculator.Calculator:
        """Create PSI4 calculator.

        Raises
        ------
        UnsupportedOperationError
            PSI4 support not yet implemented.
        """
        raise UnsupportedOperationError("PSI4 support not yet implemented.")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@register_calculator("espresso")
class EspressoBuilder(CalculatorBuilder):
    """Create a Quantum ESPRESSO (``pw.x``) calculator.

    Uses ``ase.calculators.espresso.Espresso`` with ``EspressoProfile``
    (requires ASE >= 3.23).  Follows the same structure as :class:`ORCABuilder`
    — a profile carrying the launch command + data directory, and a template
    of input parameters.

    Presets:

    * ``"molecular"`` — adds ``system.assume_isolated = "mt"``
      (Martyna–Tuckerman), gamma-point only.
    * ``"bulk"`` — defaults ``kpts`` to ``[2, 2, 2]`` if not given.
    """

    def build(
        self,
        command: str | None = None,
        pseudo_dir: str | None = None,
        pseudopotentials: dict[str, str] | None = None,
        preset: str = "molecular",
        kpts: list[int] | tuple[int, int, int] | None = None,
        input_data: dict[str, typing.Any] | None = None,
        n_mpi: int | None = None,
        omp_threads: int | None = None,
        elements: list[str] | None = None,
        directory: str = "espresso_files",
    ) -> ase.calculators.calculator.Calculator:
        """Create an Espresso calculator.

        Parameters
        ----------
        command : str, optional
            ``pw.x`` launch string.  ``None`` uses ``$ASE_ESPRESSO_COMMAND``.
            When ``n_mpi`` is set, an ``mpirun -n {n_mpi} pw.x`` command is
            built unless ``command`` already specifies a launcher.
        pseudo_dir : str, optional
            Pseudopotential directory.  ``None`` uses ``$ESPRESSO_PSEUDO``.
        pseudopotentials : dict
            ``{element_symbol: UPF_filename}`` — **required**.  If ``elements``
            is given, every element must have an entry.
        preset : str
            ``"molecular"`` or ``"bulk"``.
        elements : list of str, optional
            Element symbols present in the target system.  When provided, the
            builder verifies each has a pseudopotential entry.

        Raises
        ------
        GOALCalculatorNotFound
            If ASE is too old, or pseudopotentials are missing / incomplete.
        """
        if not HAS_ESPRESSO:
            raise GOALCalculatorNotFound(
                "Quantum ESPRESSO support requires ASE >= 3.23 with "
                "ase.calculators.espresso.EspressoProfile.  Upgrade ASE. "
                "See docs/calculators/espresso.md."
            )
        if preset not in ("molecular", "bulk"):
            raise ValueError(f"Espresso preset must be 'molecular' or 'bulk', got {preset!r}.")

        if not pseudopotentials:
            raise GOALCalculatorNotFound(
                "Quantum ESPRESSO requires a `pseudopotentials` mapping "
                "{element: UPF_filename}; none were provided.  Point `pseudo_dir` "
                "at your SSSP library and list a UPF file per element. "
                "See docs/calculators/espresso.md."
            )
        if elements:
            missing = [el for el in elements if el not in pseudopotentials]
            if missing:
                raise GOALCalculatorNotFound(
                    "Quantum ESPRESSO is missing pseudopotentials for element(s): "
                    f"{', '.join(sorted(missing))}.  Provided: "
                    f"{sorted(pseudopotentials)}.  Add a UPF file for each. "
                    "See docs/calculators/espresso.md."
                )

        if omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(int(omp_threads))

        # Build launch command.
        pw_command = command
        if pw_command is None and n_mpi is not None:
            pw_command = f"mpirun -n {int(n_mpi)} pw.x"

        resolved_pseudo_dir = pseudo_dir or os.environ.get("ESPRESSO_PSEUDO")

        profile_kwargs: dict[str, typing.Any] = {}
        if pw_command is not None:
            profile_kwargs["command"] = pw_command
        if resolved_pseudo_dir is not None:
            profile_kwargs["pseudo_dir"] = resolved_pseudo_dir
        try:
            profile = EspressoProfile(**profile_kwargs)
        except TypeError as exc:
            raise GOALCalculatorNotFound(
                "Could not build EspressoProfile — set `command`/$ASE_ESPRESSO_COMMAND "
                f"and `pseudo_dir`/$ESPRESSO_PSEUDO.  Original error: {exc}"
            ) from exc

        # Merge preset into input_data.
        merged_input: dict[str, typing.Any] = _deep_merge(
            {
                "control": {"calculation": "scf", "verbosity": "low"},
                "system": {"ecutwfc": 80, "ecutrho": 640},
                "electrons": {"conv_thr": 1.0e-8},
            },
            input_data or {},
        )
        resolved_kpts = kpts
        if preset == "molecular":
            merged_input = _deep_merge(merged_input, {"system": {"assume_isolated": "mt"}})
        elif preset == "bulk" and resolved_kpts is None:
            resolved_kpts = [2, 2, 2]

        return Espresso(
            profile=profile,
            pseudopotentials=dict(pseudopotentials),
            input_data=merged_input,
            kpts=(tuple(resolved_kpts) if resolved_kpts is not None else None),
            directory=directory,
        )


@register_calculator("vasp")
class VaspBuilder(CalculatorBuilder):
    """Create a VASP calculator (commercial — degrades gracefully if absent).

    Uses ``ase.calculators.vasp.Vasp``.  Presets:

    * ``"molecular"`` — ``kpts=[1,1,1]``, ``ismear=0``, ``sigma=0.01``.
    * ``"bulk"`` — ``kpts=[4,4,4]``, ``ismear=1``, ``sigma=0.2``.
    """

    _DEFAULT_PARAMS: typing.ClassVar[dict[str, typing.Any]] = {
        "xc": "PBE",
        "encut": 520,
        "ediff": 1.0e-6,
        "nsw": 0,
        "ibrion": -1,
        "lwave": False,
        "lcharg": False,
    }

    def build(
        self,
        command: str | None = None,
        pp_path: str | None = None,
        preset: str = "molecular",
        n_mpi: int | None = None,
        omp_threads: int | None = None,
        parameters: dict[str, typing.Any] | None = None,
        directory: str = "vasp_files",
    ) -> ase.calculators.calculator.Calculator:
        """Create a VASP calculator.

        Parameters
        ----------
        command : str, optional
            VASP launch string.  ``None`` uses ``$ASE_VASP_COMMAND``.
        pp_path : str, optional
            POTCAR library root.  ``None`` uses ``$VASP_PP_PATH``.
        preset : str
            ``"molecular"`` or ``"bulk"``.

        Raises
        ------
        GOALCalculatorNotFound
            If ASE's VASP support is unavailable, or no POTCAR path is set.
        """
        if not HAS_VASP:
            raise GOALCalculatorNotFound(
                "VASP support is unavailable — ASE could not import "
                "ase.calculators.vasp.Vasp.  VASP is commercial: install a "
                "licensed build and a recent ASE.  See docs/calculators/vasp.md."
            )
        if preset not in ("molecular", "bulk"):
            raise ValueError(f"VASP preset must be 'molecular' or 'bulk', got {preset!r}.")

        resolved_pp = pp_path or os.environ.get("VASP_PP_PATH")
        if not resolved_pp:
            raise GOALCalculatorNotFound(
                "VASP needs a POTCAR library: set `pp_path` or the $VASP_PP_PATH "
                "environment variable to the directory containing potpaw_PBE/ etc. "
                "See docs/calculators/vasp.md."
            )
        os.environ["VASP_PP_PATH"] = resolved_pp

        if omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(int(omp_threads))

        vasp_command = command or os.environ.get("ASE_VASP_COMMAND")
        if vasp_command is None and n_mpi is not None:
            vasp_command = f"mpirun -n {int(n_mpi)} vasp_std"

        params: dict[str, typing.Any] = dict(self._DEFAULT_PARAMS)
        params.update(parameters or {})
        if preset == "molecular":
            params.setdefault("kpts", [1, 1, 1])
            params.setdefault("ismear", 0)
            params.setdefault("sigma", 0.01)
        else:  # bulk
            params.setdefault("kpts", [4, 4, 4])
            params.setdefault("ismear", 1)
            params.setdefault("sigma", 0.2)

        vasp_kwargs: dict[str, typing.Any] = dict(directory=directory, **params)
        if vasp_command is not None:
            vasp_kwargs["command"] = vasp_command
        return Vasp(**vasp_kwargs)


# ============================================================================
# Semi-empirical
# ============================================================================


@register_calculator("xtb")
class XTBBuilder(CalculatorBuilder):
    """Create xTB semi-empirical calculator (GFN2-xTB by default)."""

    def build(
        self,
        method: str = "GFN2-xTB",
        charge: int = 0,
        uhf: int = 0,
        accuracy: float = 1.0,
        electronic_temperature: float = 300.0,
        max_iterations: int = 250,
        solvent: str | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Create xTB semi-empirical calculator.

        Parameters
        ----------
        method : str
            xTB method: ``"GFN1-xTB"`` or ``"GFN2-xTB"`` (default).
        charge : int
            Total molecular charge.
        uhf : int
            Number of unpaired electrons.
        accuracy : float
            Numerical accuracy (1.0 = default).
        electronic_temperature : float
            Fermi smearing temperature in Kelvin.
        max_iterations : int
            Maximum SCF iterations.
        solvent : str, optional
            Implicit solvent name (e.g. ``"water"``).

        Returns
        -------
        xtb.ase.calculator.XTB
            xTB calculator instance.

        Raises
        ------
        ImportError
            If ``xtb-python`` is not installed.
        """
        if not HAS_XTB:
            raise ImportError(
                "xtb-python not installed. Install with: pip install xtb-python "
                "or: pip install 'goal[md]'"
            )

        kwargs: dict[str, typing.Any] = dict(
            method=method,
            charge=charge,
            uhf=uhf,
            accuracy=accuracy,
            electronic_temperature=electronic_temperature,
            max_iterations=max_iterations,
        )
        if solvent is not None:
            kwargs["solvent"] = solvent

        return XTB(**kwargs)


@register_calculator("tblite")
class TBLiteBuilder(CalculatorBuilder):
    """Create TBLite semi-empirical calculator (GFN2-xTB by default)."""

    def build(
        self,
        method: str = "GFN2-xTB",
        charge: int = 0,
        uhf: int = 0,
        accuracy: float = 1.0,
        electronic_temperature: float = 300.0,
        max_iterations: int = 250,
        solvent: str | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Create TBLite semi-empirical calculator.

        Parameters
        ----------
        method : str
            TBLite method: ``"GFN1-xTB"`` or ``"GFN2-xTB"`` (default).
        charge : int
            Total molecular charge.
        uhf : int
            Number of unpaired electrons.
        accuracy : float
            Numerical accuracy (1.0 = default).
        electronic_temperature : float
            Fermi smearing temperature in Kelvin.
        max_iterations : int
            Maximum SCF iterations.
        solvent : str, optional
            Implicit solvent name (e.g. ``"water"``).

        Returns
        -------
        tblite.ase.TBLite
            TBLite calculator instance.

        Raises
        ------
        ImportError
            If ``tblite`` is not installed.
        """
        if not HAS_TBLITE:
            raise ImportError(
                "tblite not installed. Install by following the instructions on https://tblite.readthedocs.io/en/latest/installation.html#python-build "
                # "or: pip install 'goal[md]'"
            )

        kwargs: dict[str, typing.Any] = dict(
            method=method,
            charge=charge,
            uhf=uhf,
            accuracy=accuracy,
            electronic_temperature=electronic_temperature,
            max_iterations=max_iterations,
        )
        if solvent is not None:
            kwargs["solvent"] = solvent

        return TBLite(**kwargs)


# ============================================================================
# NequIP
# ============================================================================


@register_calculator("nequip")
class NequIPBuilder(CalculatorBuilder):
    """Create a NequIP / Allegro trained-model calculator."""

    def build(
        self,
        model_path: str | Path,
        device: str | None = None,
        species_to_type_name: dict[str, str] | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Load a NequIP or Allegro model as an ASE calculator.

        Parameters
        ----------
        model_path : str or Path
            Path to a deployed NequIP model (``*.pth`` / ``*.pt``).
        device : str, optional
            Torch device. Auto-detected if ``None``.
        species_to_type_name : dict, optional
            Mapping of element symbol to NequIP type name when the model
            uses custom type names.

        Returns
        -------
        nequip.ase.NequIPCalculator
            Calculator wrapping the trained NequIP model.

        Raises
        ------
        ImportError
            If ``nequip`` is not installed.
        """
        if not HAS_NEQUIP:
            raise ImportError("NequIP not installed. Install with: pip install nequip")

        from nequip.ase import NequIPCalculator

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        kwargs: dict[str, typing.Any] = {"model_path": str(model_path), "device": device}
        if species_to_type_name is not None:
            kwargs["species_to_type_name"] = species_to_type_name

        return NequIPCalculator.from_deployed_model(**kwargs)
