"""Model loading utilities for integrating goal.ml models in MD simulations."""

from __future__ import annotations

import typing
from pathlib import Path

from ase.calculators.calculator import Calculator

if typing.TYPE_CHECKING:
    from goal.ml.utils.calculator import GOALCalculator


def load_goal_calculator(
    checkpoint_path: str | Path,
    cutoff: float | None = None,
    device: str = "cpu",
    dtype: typing.Any = None,
    head: str | None = None,
    checkpoint: str = "best",
    use_ema: bool = True,
) -> GOALCalculator:
    """Load a goal.ml trained model as an ASE calculator.

    This is the primary method for using trained models from goal.ml in
    molecular dynamics simulations.

    Parameters
    ----------
    checkpoint_path : str or Path
        A ``.simurgh`` archive, a checkpoint directory, or a bare ``.ckpt``.
    cutoff : float, optional
        Ignored — the cutoff is always read from the checkpoint's config.
    device : str
        Torch device ("cpu", "cuda", "cuda:0", etc.). Default: "cpu"
    dtype : torch.dtype, optional
        Input precision.  ``None`` (default) matches the model's dtype.
    head : str, optional
        Multi-head identifier for multi-task models
    checkpoint : str
        Selector inside a checkpoint directory: ``"best"`` (default),
        ``"last"``, ``"epoch=N"`` or a ``.ckpt`` name.
    use_ema : bool
        Load the EMA weights when the checkpoint has them (default).

    Returns
    -------
    goal.ml.utils.calculator.GOALCalculator
        ASE-compatible calculator wrapping the trained model

    Raises
    ------
    FileNotFoundError
        If checkpoint not found
    RuntimeError
        If model loading fails

    Examples
    --------
    Load a trained model and run MD:

    >>> from goal.md.adapters.model_loader import load_goal_calculator
    >>> from ase.md.langevin import Langevin
    >>> from ase import units
    >>>
    >>> calc = load_goal_calculator("outputs/train/run_001/last.ckpt")
    >>> atoms.calc = calc
    >>> dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01)
    >>> dyn.run(1000)
    """
    from goal.ml.utils.calculator import GOALCalculator

    calc = GOALCalculator(
        checkpoint_path=str(checkpoint_path),
        device=device,
        dtype=dtype,
        head=head,
        checkpoint=checkpoint,
        use_ema=use_ema,
    )

    return calc


def get_model_cutoff(checkpoint_path: str | Path, checkpoint: str = "best") -> float:
    """Extract the neighbour-list cutoff a model was trained with.

    Reads the saved config only — the model is not built, so this is
    cheap and works for checkpoints the current code can no longer load.

    Parameters
    ----------
    checkpoint_path : str or Path
        A ``.simurgh`` archive, a checkpoint directory, or a bare ``.ckpt``.
    checkpoint : str
        Selector inside a checkpoint directory (see ``load_goal_calculator``).

    Returns
    -------
    float
        Cutoff in Ångströms (``data.cutoff``, else ``model.backbone.cutoff``,
        else 5.0).

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist.
    """
    import zipfile

    import torch
    from omegaconf import OmegaConf

    from goal.ml.training.archive import ARCHIVE_SUFFIX, resolve_checkpoint

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if str(path).endswith(ARCHIVE_SUFFIX):
        with zipfile.ZipFile(path) as zf:
            cfg = OmegaConf.create(zf.read("config.yaml").decode())
    elif path.is_dir():
        cfg_file = path / "config.yaml"
        if cfg_file.is_file():
            cfg = OmegaConf.load(cfg_file)
        else:
            raw = torch.load(
                str(resolve_checkpoint(path, checkpoint)), map_location="cpu", weights_only=False
            )
            cfg = raw["hyper_parameters"]["config"]
    else:
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = raw["hyper_parameters"]["config"]

    data = cfg.get("data") or {}
    backbone = (cfg.get("model") or {}).get("backbone") or {}
    return float(data.get("cutoff", backbone.get("cutoff", 5.0)))
