"""Shared exception types for GOAL calculators and foundation-model wrappers.

Kept dependency-free on purpose: this module imports nothing from
``goal`` (nor ase/torch), so it can be imported from either the MD side
(``goal.md.core.calculator_factory``) or, lazily, from the ML side
(``goal.ml.nn.models.foundation``) without dragging in heavy packages or
creating an import cycle.
"""

from __future__ import annotations


class GOALCalculatorNotFound(Exception):
    """External QM code (or its Python bindings) is not installed or not configured.

    Raised by calculator builders when the underlying engine — ORCA, CP2K,
    Quantum ESPRESSO, VASP, UPET, … — cannot be located, or when a
    required piece of configuration (pseudopotential set, ``$VASP_PP_PATH``,
    …) is missing.  The message should say *exactly* what is missing and how
    to provide it.
    """


class UnsupportedOperationError(Exception):
    """The requested operation is recognised but not yet supported.

    Used for capabilities that are planned but not implemented (e.g. loading
    a UMA foundation model as a raw ``nn.Module`` when the upstream API does
    not expose that path), as opposed to a hard programming error.
    """
