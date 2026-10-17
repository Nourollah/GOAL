"""GOAL — General Open Atomistic Laboratory.

A modular Python framework for building, training, and deploying
machine-learning interatomic potentials.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Python 3.14 / Hydra 1.3.2 compatibility shim.
#
# Python 3.14 added an *eager* ``argparse._ActionsContainer._check_help`` that
# format-checks every action's help string at ``add_argument`` time.  Hydra
# 1.3.2's ``--shell-completion`` argument is registered with a
# ``LazyCompletionHelp()`` instance whose only purpose is to defer rendering
# of the help text — it doesn't implement ``__contains__``, so the new
# eager check raises ``TypeError`` and aborts every entry point that uses
# ``@hydra.main`` (``goal-train``, ``goal-eval``, ``goal-finetune``,
# ``goal-tune``, ``goal-simulate``, ``goal-simulate-mts``).
#
# We restore the pre-3.14 behaviour by making ``_check_help`` swallow the
# eager validation failure.  The help text is still validated when ``--help``
# is invoked for real, so users see exactly the same diagnostics.
import os as _os
import sys as _sys

#: Absolute path to the Hydra configs bundled with the package
#: (``goal/configs``: ``ml/`` experiments, ``md/`` simulations, presets).
CONFIGS_DIR: str = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "configs")

if _sys.version_info >= (3, 14):
    import argparse as _argparse

    _orig_check_help = _argparse._ActionsContainer._check_help

    def _patched_check_help(self, action):  # type: ignore[no-redef]
        try:
            _orig_check_help(self, action)
        except (TypeError, ValueError):
            # Eager check failed because ``action.help`` is a lazy / non-str
            # object (e.g. Hydra's ``LazyCompletionHelp``).  argparse will
            # re-validate at help-display time, so silently allow it here.
            pass

    _argparse._ActionsContainer._check_help = _patched_check_help
