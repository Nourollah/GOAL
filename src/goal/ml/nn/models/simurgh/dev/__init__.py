"""SIMURGH under development (registry key ``simurgh_dev``).

One monolithic model file, :mod:`goal.ml.nn.models.simurgh.dev.monolithic`,
until the release criteria are met; literature ablations live in
:mod:`goal.ml.nn.models.simurgh.dev.ablations`.
"""

from goal.ml.nn.models.simurgh.dev.monolithic import SimurghDev

__all__ = ["SimurghDev"]
