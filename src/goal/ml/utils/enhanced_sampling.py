"""Enhanced sampling with GOAL models — roadmap placeholders.

Well-tempered metadynamics and replica-exchange MD (REMD) exist upstream
in nvalchemi-toolkit's ``nvalchemi.enhanced_sampling``
(``WellTemperedMetaDynamicsBias``, ``ReplicaExchange``) on its ``main``
branch, not yet in a release.  GOAL models already run in that engine
through :class:`goal.ml.adapters.nvalchemi_wrapper.SIMURGHNvalchemiWrapper`;
these entry points will drive it once the upstream API is released.

Until then both functions raise ``NotImplementedError`` with a pointer to
the upstream work, so code paths that select a sampler (e.g.
``goal-simulate-batch --sampler metadynamics``) fail with a clear message.
"""

from __future__ import annotations

import typing

UPSTREAM_URL: str = "https://github.com/NVIDIA/nvalchemi-toolkit"

_MESSAGE: str = (
    "{what} is on the GOAL roadmap but not implemented yet.  It will be built on "
    "nvalchemi-toolkit's enhanced-sampling engine (nvalchemi.enhanced_sampling, "
    "unreleased on its main branch): " + UPSTREAM_URL + "\n"
    "Today you can already run a GOAL model there yourself through "
    "goal.ml.adapters.nvalchemi_wrapper.SIMURGHNvalchemiWrapper."
)


def run_metadynamics(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
    """Well-tempered metadynamics with a GOAL model (not implemented yet)."""
    raise NotImplementedError(_MESSAGE.format(what="Metadynamics"))


def run_remd(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
    """Replica-exchange MD with a GOAL model (not implemented yet)."""
    raise NotImplementedError(_MESSAGE.format(what="Replica-exchange MD (REMD)"))


SAMPLERS: dict[str, typing.Callable[..., typing.Any]] = {
    "metadynamics": run_metadynamics,
    "remd": run_remd,
}
