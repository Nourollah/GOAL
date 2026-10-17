"""MD module command-line entry points (isolated to this module only).

* ``goal-simulate`` / ``goal-simulate-mts`` — Hydra apps in
  :mod:`goal.md.cli.simulate`.
* ``goal-simulate-batch`` — batched MD / relaxation of many structures,
  :mod:`goal.md.cli.simulate_batch`.
* ``python -m goal.md.cli.main`` — placeholder Typer app (needs ``typer``).

Nothing is imported here, so the Hydra commands work without the optional
``typer`` dependency.
"""
