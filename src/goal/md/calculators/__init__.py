"""Calculator support package.

The concrete ASE calculator builders live in
:mod:`goal.md.core.calculator_factory` (the ``CalculatorBuilder`` /
``@register_calculator`` pattern).  This package holds cross-cutting,
dependency-free pieces — currently the shared exception types.
"""

from __future__ import annotations

from goal.md.calculators.base import GOALCalculatorNotFound, UnsupportedOperationError

__all__ = ["GOALCalculatorNotFound", "UnsupportedOperationError"]
