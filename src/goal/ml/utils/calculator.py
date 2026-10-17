"""ASE Calculator interface for trained GOAL models.

Wraps any trained ``GOALModule`` checkpoint into a standard ASE
``Calculator`` so it can be used for:

- Single-point energy / force / stress calculations
- Geometry optimisation (``ase.optimize``)
- Molecular dynamics (``ase.md``)
- Nudged elastic band (NEB) transition-state searches
- Phonon calculations (``ase.phonons``)

Three checkpoint formats are accepted:

A. ``.simurgh`` archive — a single self-contained zip (weights +
   resolved config + metadata + frozen model source).  Recommended for
   sharing and deployment; loads from the frozen source, so it works
   even after the codebase has changed.
B. Checkpoint *directory* written by ``GOALCheckpointManager`` — must
   contain ``config.yaml``, ``metadata.json`` and ``frozen_source/``.
   Select which weights to use via ``checkpoint=`` (``"best"``,
   ``"last"``, ``"epoch=N"``, or a ``.ckpt`` filename).
C. Bare ``.ckpt`` file (legacy) — NOT self-contained: loading rebuilds
   the model from the *live* source code and may fail if it changed
   since the checkpoint was saved.

Whatever the format, two things are taken from the checkpoint itself
rather than from the caller:

* **Weights.**  When the run trained with ``training.ema.enabled`` the
  checkpoint carries the EMA shadow under ``goal_ema``.  Validation and
  best-checkpoint selection were done on those averaged weights, so they
  are the ones loaded (``use_ema=False`` opts out).
* **Graph construction.**  A model trained with fragment labels
  (``data.compute_fragment_index: true``) needs every inference graph to
  carry them too.  The ``data.fragment_*`` options are read from the
  saved config and applied in :meth:`GOALCalculator.build_graph`.

Usage::

    from goal.ml.utils.calculator import GOALCalculator

    # From a self-contained archive (FORMAT A — recommended)
    calc = GOALCalculator(checkpoint_path="my_model.simurgh")

    # From a checkpoint directory (FORMAT B)
    calc = GOALCalculator(checkpoint_path="logs/.../checkpoints", checkpoint="best")

    # From a bare checkpoint file (FORMAT C — legacy)
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

import dataclasses
import inspect
import logging
import tempfile
import typing
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from goal.ml.training.archive import (
    ARCHIVE_SUFFIX,
    FROZEN_SOURCE_DIRNAME,
    frozen_source_imports,
    is_managed_checkpoint_dir,
    resolve_checkpoint,
    unpack_simurgh_archive,
)

logger = logging.getLogger(__name__)

#: ``data.*`` keys forwarded to :meth:`AtomicGraph.from_ase` when the
#: checkpoint was trained with ``data.compute_fragment_index: true``.
#: Same names as the dataset keyword arguments (see
#: ``goal.ml.data.datamodule._FRAGMENT_KEYS``).
_FRAGMENT_OPTION_KEYS: tuple[str, ...] = (
    "fragment_covalent_cutoff",
    "fragment_scheme",
    "fragment_charge",
    "fragment_on_failure",
    "fragment_smarts",
    "fragment_keep_groups",
)


# ---------------------------------------------------------------------------
# Checkpoint loading (shared by GOALCalculator, goal-eval and the adapters)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LoadedCheckpoint:
    """A ``GOALModule`` rebuilt from a checkpoint, plus what came with it.

    Attributes
    ----------
    module : GOALModule
        The rebuilt module, on the requested device, in the checkpoint's
        floating-point dtype.
    cutoff : float
        Neighbour-list cutoff read from the saved config.
    config : DictConfig
        The resolved Hydra config the run was trained with.
    weights : str
        ``"ema"`` when the EMA shadow was loaded into the parameters,
        ``"raw"`` otherwise.
    tmpdir : TemporaryDirectory or None
        Keeps an unpacked ``.simurgh`` archive alive; hold a reference
        for as long as the module is used.
    """

    module: typing.Any
    cutoff: float
    config: typing.Any
    weights: str
    tmpdir: tempfile.TemporaryDirectory | None = None


def load_goal_checkpoint(
    path: str | Path,
    *,
    checkpoint: str = "best",
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> LoadedCheckpoint:
    """Rebuild a ``GOALModule`` from any of the three checkpoint formats.

    Parameters
    ----------
    path : str or Path
        A ``.simurgh`` archive, a ``GOALCheckpointManager`` directory, or
        a bare Lightning ``.ckpt`` file.
    checkpoint : str
        Selector inside a checkpoint directory (``"best"``, ``"last"``,
        ``"epoch=N"`` or a ``.ckpt`` name).  Ignored for the other formats.
    device : str or torch.device
        Device to load onto.
    use_ema : bool
        Load the EMA shadow weights when the checkpoint has them.

    Returns
    -------
    LoadedCheckpoint
    """
    device = torch.device(device)
    path_str = str(path)

    if path_str.endswith(ARCHIVE_SUFFIX):
        # FORMAT A — self-contained .simurgh archive
        archive = Path(path)
        if not archive.is_file():
            raise FileNotFoundError(f"Archive not found: {archive}")
        tmpdir = tempfile.TemporaryDirectory(prefix="goal_simurgh_")
        dest = unpack_simurgh_archive(archive, tmpdir.name)
        weights = dest / "weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Archive {archive} contains no weights.pt")
        logger.info("Loading %s from frozen source (self-contained archive).", archive.name)
        loaded = _load_from_directory(dest, weights, device, use_ema)
        loaded.tmpdir = tmpdir
        return loaded

    if Path(path).is_dir():
        # FORMAT B — checkpoint directory written by GOALCheckpointManager
        dirpath = Path(path)
        if not is_managed_checkpoint_dir(dirpath):
            raise FileNotFoundError(
                f"Directory {dirpath} is missing config.yaml, metadata.json, "
                f"or {FROZEN_SOURCE_DIRNAME}/. This is not a valid SIMURGH "
                f"checkpoint directory. Pass a {ARCHIVE_SUFFIX} archive or a "
                f"directory produced by GOALCheckpointManager."
            )
        ckpt_path = resolve_checkpoint(dirpath, checkpoint)
        return _load_from_directory(dirpath, ckpt_path, device, use_ema)

    if path_str.endswith(".ckpt"):
        # FORMAT C — legacy bare checkpoint
        logger.warning(
            "Loading from a bare .ckpt file. This format is not "
            "self-contained — if source files have changed since this "
            "checkpoint was saved, loading may fail. Use a %s archive "
            "or the checkpoint directory for reliable loading.",
            ARCHIVE_SUFFIX,
        )
        return _build_module_from_ckpt(path, device, use_ema)

    raise ValueError(
        f"Unrecognised checkpoint path: {path}. Expected a {ARCHIVE_SUFFIX} "
        f"archive, a checkpoint directory, or a .ckpt file."
    )


def _load_from_directory(
    dirpath: Path,
    ckpt_path: Path,
    device: torch.device,
    use_ema: bool,
) -> LoadedCheckpoint:
    """FORMAT B: rebuild the model with imports served from frozen source."""
    frozen = dirpath / FROZEN_SOURCE_DIRNAME
    logger.info(
        "Loading %s using frozen source at %s — immune to later code changes.",
        ckpt_path.name,
        frozen,
    )
    with frozen_source_imports(frozen):
        return _build_module_from_ckpt(ckpt_path, device, use_ema)


def _checkpoint_dtype(state_dict: typing.Mapping[str, typing.Any]) -> torch.dtype | None:
    """Floating-point dtype the weights were saved in (``None`` if unknown)."""
    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.dtype
    return None


def apply_ema_weights(module: torch.nn.Module, raw: typing.Mapping[str, typing.Any]) -> bool:
    """Copy the ``goal_ema`` shadow of a checkpoint into ``module``'s parameters.

    ``GOALModule.on_save_checkpoint`` stores the shadow as a list in
    ``module.parameters()`` order, so a module rebuilt from the same
    config lines up one-to-one.  Anything that does not line up (an older
    checkpoint, a different architecture) leaves the raw weights in
    place and returns ``False``.

    The module's own ``ema`` (built from freshly initialised parameters
    when the config enables EMA) is re-seeded from the loaded weights,
    so a later ``ema.average_parameters()`` swap — ``validation_step`` /
    ``test_step`` — evaluates these weights rather than random ones.
    """
    params: list[torch.nn.Parameter] = list(module.parameters())
    state: typing.Any = raw.get("goal_ema")
    shadow: typing.Any = state.get("shadow") if isinstance(state, typing.Mapping) else None
    applied = False
    if isinstance(shadow, list):
        if len(shadow) == len(params) and all(
            tuple(s.shape) == tuple(p.shape) for s, p in zip(shadow, params)
        ):
            with torch.no_grad():
                for param, saved in zip(params, shadow):
                    param.copy_(saved.to(dtype=param.dtype, device=param.device))
            applied = True
        else:
            logger.warning(
                "Checkpoint EMA shadow (%d tensors) does not match the rebuilt "
                "model (%d parameters); using the raw weights instead.",
                len(shadow),
                len(params),
            )

    ema: typing.Any = getattr(module, "ema", None)
    if ema is not None and hasattr(ema, "load_state_dict"):
        ema.load_state_dict({"shadow": [p.detach().clone() for p in params]})
    return applied


def _build_module_from_ckpt(
    path: str | Path,
    device: torch.device,
    use_ema: bool,
) -> LoadedCheckpoint:
    """Load a GOALModule from a Lightning checkpoint.

    Reconstructs backbone + head + loss from the saved Hydra config,
    loads the state dict, and extracts the neighbour-list cutoff.
    Lightning's ``load_from_checkpoint`` cannot be used directly because
    backbone/head/loss are excluded from ``save_hyperparameters``.

    All ``goal`` imports happen inside this function so that, when it
    runs under :func:`frozen_source_imports`, the model classes are
    executed from the frozen snapshot instead of the live codebase.
    """
    # Populate the registries (auto-discovery walks the package path,
    # which under a frozen import context is the frozen snapshot).
    import goal.ml.nn.heads  # noqa: F401
    import goal.ml.nn.models  # noqa: F401

    try:  # adapter backbones (e.g. simurgh_lora) register here
        import goal.ml.adapters  # noqa: F401
    except ImportError:  # an older frozen snapshot without them
        pass
    from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY
    from goal.ml.training.loss import CompositeLoss, WeightedLoss
    from goal.ml.training.module import GOALModule

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    raw = torch.load(str(path), map_location=device, weights_only=False)
    cfg = raw["hyper_parameters"]["config"]

    # Rebuild backbone from saved config
    bb_cls = BACKBONE_REGISTRY.get(cfg.model.backbone.name)
    bb_kw = {k: v for k, v in cfg.model.backbone.items() if k != "name"}
    backbone = bb_cls(**bb_kw)

    # Rebuild head (None for monolithic models)
    head_cfg = cfg.model.get("head", None)
    if head_cfg is not None:
        head_cls = HEAD_REGISTRY.get(head_cfg.name)
        head_kw = {k: v for k, v in head_cfg.items() if k != "name"}
        head = head_cls(**head_kw)
    else:
        head = None

    # Rebuild loss components (not used for inference; skip gracefully on
    # API mismatch between frozen source and config, e.g. when the frozen
    # source pre-dates a new loss parameter).  Like goal-train, only the
    # keyword arguments a loss class accepts are forwarded.
    try:
        losses: list[WeightedLoss] = []
        for lc in cfg.training.losses:
            loss_cls = LOSS_REGISTRY.get(lc.name)
            fn_spec = lc.get("fn", "mse")
            params = inspect.signature(loss_cls.__init__).parameters
            open_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            lkw = {
                k: v
                for k, v in lc.items()
                if k not in ("name", "weight", "fn") and (open_kw or k in params)
            }
            fn = fn_spec if isinstance(fn_spec, str) else "mse"
            losses.append(
                WeightedLoss(loss_cls(loss_fn=fn, **lkw), weight=lc.weight, label=lc.name)
            )
        loss = CompositeLoss(losses)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not reconstruct loss components from frozen source (%s). "
            "Using an empty CompositeLoss — safe for inference only.",
            exc,
        )
        loss = CompositeLoss([])

    module = GOALModule(backbone=backbone, head=head, loss=loss, config=cfg)

    # Build in the dtype the weights were saved in.  ``load_state_dict``
    # copies into the existing parameters and keeps *their* dtype, so a
    # float64 run would otherwise be silently evaluated in float32.
    saved_dtype = _checkpoint_dtype(raw["state_dict"])
    if saved_dtype is not None:
        module = module.to(saved_dtype)
    module = module.to(device)
    module.load_state_dict(raw["state_dict"], strict=False)

    weights = "raw"
    if use_ema and apply_ema_weights(module, raw):
        weights = "ema"
    elif not use_ema:
        apply_ema_weights(module, {})  # re-seed module.ema from the raw weights
    logger.info("Loaded %s weights from %s.", weights.upper(), path.name)

    # Extract cutoff from saved config
    cutoff: float = float(
        cfg.data.get(
            "cutoff",
            cfg.model.backbone.get("cutoff", 5.0),
        )
    )
    return LoadedCheckpoint(module=module, cutoff=cutoff, config=cfg, weights=weights)


def fragment_options_from_config(config: typing.Any) -> dict[str, typing.Any]:
    """``AtomicGraph.from_ase`` fragment kwargs implied by a training config.

    Returns an empty dict when the run did not compute fragment labels.
    """
    if config is None:
        return {}
    data: typing.Any = config.get("data", None) if hasattr(config, "get") else None
    if not data or not bool(data.get("compute_fragment_index", False)):
        return {}
    options: dict[str, typing.Any] = {"compute_fragment_index": True}
    for key in _FRAGMENT_OPTION_KEYS:
        value: typing.Any = data.get(key, None)
        if value is None:
            continue
        if not isinstance(value, (str, int, float, bool)):
            value = list(value)  # ListConfig → plain list
        options[key] = value
    return options


class GOALCalculator(Calculator):
    """ASE Calculator backed by a trained GOAL model.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        A ``.simurgh`` archive, a checkpoint directory written by
        ``GOALCheckpointManager``, or a bare Lightning ``.ckpt`` file
        (legacy).  Mutually exclusive with ``module``.
    module : GOALModule, optional
        An already-instantiated ``GOALModule``.  Mutually exclusive with
        ``checkpoint_path``.
    cutoff : float, optional
        Neighbour-list cutoff in Ångström.  Required when ``module`` is
        provided directly.  When loading from checkpoint, read from the
        saved config automatically.
    device : str
        Torch device — ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
    dtype : torch.dtype or None
        Precision for atomic positions and cell.  When ``None`` (the
        recommended default) the calculator sniffs the dtype of the
        loaded model's parameters and uses that, so the inputs always
        match the model.  Pass an explicit dtype only to override.
    head : str or None
        Multi-head tag for models trained with multiple heads.
    checkpoint : str
        Which weights to load when ``checkpoint_path`` is a checkpoint
        *directory*: ``"best"`` (default), ``"last"``, ``"epoch=N"``,
        or a ``.ckpt`` filename/path.  Ignored for the other formats.
    use_ema : bool
        Load the EMA weights stored in the checkpoint (``goal_ema``) when
        present — the weights validation and best-checkpoint selection
        used.  ``False`` loads the raw optimiser weights.
    fragment_options : mapping, optional
        Fragment-label keyword arguments for ``AtomicGraph.from_ase``.
        ``None`` (default) derives them from the model's training config
        (``data.compute_fragment_index`` and ``data.fragment_*``); pass
        ``{}`` to disable fragment labels explicitly.
    fragment_on_failure : str or None
        Overrides ``fragment_on_failure`` for the rdkit-based fragment
        schemes.  Defaults to ``"fallback"``: bond perception can fail on
        geometries an MD run or optimiser walks into, and aborting the
        run there is rarely what you want.  ``None`` keeps the value the
        model was trained with.
    neighbor_list_backend : str or None
        Neighbour-list backend for inference graphs (``"ase"``,
        ``"matscipy"``, ``"nvalchemiops"``, ``"auto"``...).  ``None``
        uses the training config's ``data.neighbor_list_backend``, or
        ``"ase"``.
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
        dtype: torch.dtype | None = None,
        head: str | None = None,
        checkpoint: str = "best",
        use_ema: bool = True,
        fragment_options: typing.Mapping[str, typing.Any] | None = None,
        fragment_on_failure: str | None = "fallback",
        neighbor_list_backend: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)

        if checkpoint_path is not None and module is not None:
            raise ValueError("Provide either 'checkpoint_path' or 'module', not both.")
        if checkpoint_path is None and module is None:
            raise ValueError("Provide either 'checkpoint_path' or 'module'.")

        self.device: torch.device = torch.device(device)
        self.head: str | None = head
        self._checkpoint_selector: str = checkpoint
        # Keeps an unpacked .simurgh archive alive for the calculator's lifetime
        self._archive_tmpdir: tempfile.TemporaryDirectory | None = None
        self._config: typing.Any = None
        #: ``"ema"`` / ``"raw"`` for checkpoints, ``"module"`` when a module was passed in.
        self.weights: str = "module"

        if checkpoint_path is not None:
            loaded = load_goal_checkpoint(
                checkpoint_path,
                checkpoint=checkpoint,
                device=self.device,
                use_ema=use_ema,
            )
            self._module, self._cutoff = loaded.module, loaded.cutoff
            self._config = loaded.config
            self._archive_tmpdir = loaded.tmpdir
            self.weights = loaded.weights
        else:
            if cutoff is None:
                raise ValueError("'cutoff' is required when providing a module directly.")
            self._module = module
            self._cutoff = float(cutoff)
            self._config = getattr(module, "config", None)

        self._module = self._module.to(self.device)
        self._module.eval()

        # Default the input dtype to the model's actual parameter dtype so
        # the AtomicGraph never disagrees with the network it feeds.  An
        # explicit ``dtype=...`` still wins.
        if dtype is None:
            try:
                dtype = next(self._module.parameters()).dtype
            except StopIteration:
                dtype = torch.float64
        self.dtype: torch.dtype = dtype

        # How inference graphs are built — must match training.
        if fragment_options is None:
            options = fragment_options_from_config(self._config)
        else:
            options = dict(fragment_options)
        if options.get("compute_fragment_index") and fragment_on_failure is not None:
            options["fragment_on_failure"] = fragment_on_failure
        self._fragment_options: dict[str, typing.Any] = options
        if options.get("compute_fragment_index"):
            logger.info(
                "Fragment labels enabled for inference (scheme=%s).",
                options.get("fragment_scheme", "connected"),
            )

        if neighbor_list_backend is None:
            data_cfg: typing.Any = (
                self._config.get("data", None) if hasattr(self._config, "get") else None
            )
            neighbor_list_backend = (
                str(data_cfg.get("neighbor_list_backend", "ase")) if data_cfg else "ase"
            )
        self.neighbor_list_backend: str = neighbor_list_backend

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def module(self) -> typing.Any:
        """The wrapped ``GOALModule`` (shared, not a copy)."""
        return self._module

    @property
    def cutoff(self) -> float:
        """Neighbour-list cutoff in Ångström."""
        return self._cutoff

    @property
    def config(self) -> typing.Any:
        """The training config the model was built from, if known."""
        return self._config

    @property
    def fragment_options(self) -> dict[str, typing.Any]:
        """Fragment-label kwargs applied to every inference graph."""
        return dict(self._fragment_options)

    def build_graph(
        self,
        atoms: Atoms,
        energy: float | None = None,
        forces: typing.Any = None,
        stress: typing.Any = None,
    ) -> typing.Any:
        """Build the ``AtomicGraph`` this model expects for ``atoms``.

        Same cutoff, dtype, head tag and fragment labels as inference,
        so callers that train on calculator-built graphs (the online
        fine-tuner) stay consistent with what the model sees in MD.
        """
        from goal.ml.data.graph import AtomicGraph

        return AtomicGraph.from_ase(
            atoms,
            cutoff=self._cutoff,
            energy=energy,
            forces=forces,
            stress=stress,
            dtype=self.dtype,
            head=self.head,
            neighbor_list_backend=self.neighbor_list_backend,
            **self._fragment_options,
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate energy, forces, and/or stress for the given Atoms.

        This method is called automatically by ASE when you access
        ``atoms.get_potential_energy()``, ``atoms.get_forces()``, etc.

        Runs under ``torch.enable_grad`` so that autograd-based force heads
        (``EnergyForcesHead``, ``DualForcesHead`` in autograd/hybrid mode)
        can compute ``-∂E/∂r``.  Gradients are not retained after this call.
        """
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        if self.atoms is None:
            raise RuntimeError("No atoms object set on calculator.")

        graph = self.build_graph(self.atoms).to(self.device)

        with torch.enable_grad():
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
            f"GOALCalculator(cutoff={self._cutoff}, device={self.device}, "
            f"dtype={self.dtype}, weights={self.weights})"
        )
