"""``goal-distill`` — knowledge distillation from a foundation model (roadmap).

Planned: label structures with a large teacher (a foundation model or a
big SIMURGH) on the GPU — nvalchemi-toolkit's batched inference on NVIDIA
hardware — and train a small, fast GOAL student on those labels.  Not
implemented yet; the command exists so scripts and docs can refer to it.
"""

from __future__ import annotations

import argparse
import sys
import typing

ROADMAP_MESSAGE: str = (
    "goal-distill is on the GOAL roadmap but not implemented yet.\n"
    "Today: label your structures with the teacher yourself (e.g. "
    "goal-simulate-batch or a GOALCalculator / ASE loop writing extxyz), "
    "then train the student with goal-train on those files."
)


def run_distillation(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
    """Teacher → student distillation (not implemented yet)."""
    raise NotImplementedError(ROADMAP_MESSAGE)


def main(argv: typing.Sequence[str] | None = None) -> int:
    """Entry point of the ``goal-distill`` console script (roadmap stub).

    Parses ``argv`` (``sys.argv[1:]`` when ``None``) for the planned
    ``--teacher``, ``--student-config`` and ``--structures`` options,
    then calls :func:`run_distillation`, which is not implemented yet.

    Returns
    -------
    int
        ``1`` after printing :data:`ROADMAP_MESSAGE` to stderr
        (:func:`run_distillation` always raises ``NotImplementedError``).
    """
    parser = argparse.ArgumentParser(
        prog="goal-distill",
        description="Knowledge distillation from a foundation model (roadmap, not implemented yet).",
    )
    parser.add_argument("--teacher", help="teacher checkpoint or foundation-model name (planned)")
    parser.add_argument("--student-config", help="goal-train experiment for the student (planned)")
    parser.add_argument("--structures", nargs="*", help="structures to label (planned)")
    parser.parse_args(argv)
    try:
        run_distillation()
    except NotImplementedError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
