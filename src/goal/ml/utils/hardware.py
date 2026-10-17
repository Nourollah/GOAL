"""Hardware detection and backend selection for GOAL.

One place that answers "what accelerator is this, and may the NVIDIA-only
fast paths run here?", so every integration makes the same decision.

The rule for every NVIDIA-specific feature (nvalchemi-toolkit and its
kernel library nvalchemi-toolkit-ops):

* it is used only when the package imports **and** an NVIDIA CUDA device
  is present — ROCm builds of PyTorch also report ``torch.cuda`` as
  available, so the HIP check matters;
* otherwise GOAL falls back to a portable implementation (PyG / ASE /
  matscipy on CPU, AMD ROCm or Apple MPS) and logs which one, once;
* nothing ever raises because the NVIDIA packages are missing.

Import names differ from the distribution names:

==============================  ===============
distribution                    import
==============================  ===============
``nvalchemi-toolkit``           ``nvalchemi``
``nvalchemi-toolkit-ops``       ``nvalchemiops``
==============================  ===============
"""

from __future__ import annotations

import importlib.util
import logging
import typing
from functools import cache, lru_cache

log = logging.getLogger(__name__)

Backend = typing.Literal["nvidia", "amd", "mps", "cpu"]


@lru_cache(maxsize=1)
def detect_compute_backend() -> Backend:
    """Best available compute backend, in priority order.

    ``"nvidia"``
        CUDA device present on a CUDA build of PyTorch.
    ``"amd"``
        ``torch.cuda`` available on a ROCm/HIP build (``torch.version.hip``).
    ``"mps"``
        Apple Metal Performance Shaders (Apple Silicon).
    ``"cpu"``
        Fallback.

    The result is cached; :func:`clear_cache` resets it (tests).
    """
    import torch

    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            log.info("GOAL hardware backend: amd (ROCm/HIP)")
            return "amd"
        log.info("GOAL hardware backend: nvidia (CUDA)")
        return "nvidia"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        log.info("GOAL hardware backend: mps (Apple Silicon)")
        return "mps"
    log.info("GOAL hardware backend: cpu")
    return "cpu"


def _importable(module: str) -> bool:
    """True when ``module`` can be imported (without importing it twice)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def nvalchemi_available() -> bool:
    """``nvalchemi-toolkit`` is installed **and** an NVIDIA GPU is present.

    AMD ROCm and Apple MPS are not supported by nvalchemi.
    """
    if not _importable("nvalchemi"):
        return False
    try:
        import nvalchemi  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — a broken install is "not available"
        log.info("nvalchemi-toolkit is installed but failed to import: %r", exc)
        return False
    return detect_compute_backend() == "nvidia"


@lru_cache(maxsize=1)
def nvalchemiops_available() -> bool:
    """``nvalchemi-toolkit-ops`` (Warp kernels) is installed **and** an NVIDIA
    GPU is present."""
    if not _importable("nvalchemiops"):
        return False
    try:
        import nvalchemiops.torch.neighbors  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        log.info("nvalchemi-toolkit-ops is installed but failed to import: %r", exc)
        return False
    return detect_compute_backend() == "nvidia"


#: Alias with the spelling used in the integration plan.
nvalchemi_ops_available = nvalchemiops_available


def resolve_backend(
    requested: str,
    *,
    fast: str,
    fallback: str,
    available: typing.Callable[[], bool],
    feature: str,
) -> str:
    """Resolve an ``"auto"`` / explicit backend request for one feature.

    Parameters
    ----------
    requested : str
        ``"auto"``, the fast backend's name, or any other backend name
        (returned unchanged).
    fast, fallback : str
        Names of the NVIDIA fast path and of the portable fallback.
    available : callable
        Availability probe for the fast path (e.g. :func:`nvalchemi_available`).
    feature : str
        Human-readable feature name for the log line.

    Returns
    -------
    str
        The backend to use.  An explicit request for an unavailable fast
        path degrades to ``fallback`` with a warning instead of failing.
    """
    if requested == "auto":
        chosen = fast if available() else fallback
        _log_choice_once(feature, chosen)
        return chosen
    if requested == fast and not available():
        log.warning(
            "GOAL %s: backend %r requested but it needs %s and an NVIDIA CUDA GPU; "
            "falling back to %r.",
            feature,
            fast,
            "nvalchemi-toolkit" if fast == "nvalchemi" else "nvalchemi-toolkit-ops",
            fallback,
        )
        return fallback
    return requested


@cache
def _log_choice_once(feature: str, chosen: str) -> None:
    """One INFO line per (feature, backend) — resolution can run every MD step."""
    log.info("GOAL %s backend: %s", feature, chosen)


def clear_cache() -> None:
    """Forget every cached detection result (for tests)."""
    detect_compute_backend.cache_clear()
    nvalchemi_available.cache_clear()
    nvalchemiops_available.cache_clear()
    _log_choice_once.cache_clear()
