"""``goal-simulate-batch`` — MD or relaxation of many structures with one GOAL model.

Examples::

    # 2000 Langevin steps at 300 K for every frame in frames.xyz
    goal-simulate-batch --checkpoint logs/<task>/runs/<run>/checkpoints \\
        --structures frames.xyz --steps 2000 --temperature 300

    # relax everything to fmax 0.02 eV/Å, portable backend
    goal-simulate-batch --checkpoint model.simurgh --structures a.xyz b.xyz \\
        --mode relax --fmax 0.02 --backend ase

    # defaults from a YAML file (keys = BatchSimulationConfig fields), flags override
    goal-simulate-batch --config src/goal/configs/md/simulations/batch.yaml --checkpoint model.simurgh

``--backend auto`` (default) runs every structure in one GPU batch through
nvalchemi-toolkit when it is installed and an NVIDIA GPU is present, and
otherwise one after another with ASE + GOALCalculator (CPU, AMD, Apple).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import typing

from goal.md.core.batched import (
    BACKENDS,
    MODES,
    SAMPLERS,
    TRAJECTORY_FORMATS,
    BatchSimulationConfig,
    run_batched_simulation,
)

#: CLI flag → BatchSimulationConfig field, where the names differ.
_FLAG_TO_FIELD: dict[str, str] = {"temperature": "temperature_K", "timestep": "timestep_fs"}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="goal-simulate-batch",
        description="Batched MD / geometry relaxation of many structures with a GOAL model.",
    )
    p.add_argument(
        "--config", help="YAML file with BatchSimulationConfig fields (flags override it)"
    )
    p.add_argument("--checkpoint", help=".simurgh archive, checkpoint directory or .ckpt")
    p.add_argument(
        "--checkpoint-select", dest="checkpoint_select", help="best | last | epoch=N (directories)"
    )
    p.add_argument("--structures", nargs="+", help="structure file(s); every frame is simulated")
    p.add_argument("--steps", type=int, help="MD steps / max optimiser steps (default 1000)")
    p.add_argument("--temperature", type=float, help="temperature in K (default 300)")
    p.add_argument("--timestep", type=float, help="MD timestep in fs (default 0.5)")
    p.add_argument("--friction", type=float, help="Langevin friction in 1/fs (default 0.01)")
    p.add_argument("--fmax", type=float, help="relaxation force threshold in eV/Å (default 0.05)")
    p.add_argument("--mode", choices=MODES, help="md (default) or relax")
    p.add_argument("--backend", choices=BACKENDS, help="auto (default), nvalchemi or ase")
    p.add_argument(
        "--sampler", choices=SAMPLERS, help="langevin (default); others are roadmap stubs"
    )
    p.add_argument(
        "--output-dir", dest="output_dir", help="output directory (default outputs/md/batch)"
    )
    p.add_argument("--trajectory-format", dest="trajectory_format", choices=TRAJECTORY_FORMATS)
    p.add_argument("--no-trajectory", dest="write_trajectory", action="store_false", default=None)
    p.add_argument(
        "--log-every", dest="log_every", type=int, help="trajectory/log interval (default 100)"
    )
    p.add_argument("--device", help="torch device (default: cuda when available)")
    p.add_argument("--seed", type=int)
    return p


def build_config(argv: typing.Sequence[str] | None = None) -> BatchSimulationConfig:
    """Parse CLI flags (and the optional ``--config`` YAML) into a config."""
    args = _parser().parse_args(argv)
    values: dict[str, typing.Any] = {}
    if args.config:
        from omegaconf import OmegaConf

        loaded = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        fields = {f.name for f in dataclasses.fields(BatchSimulationConfig)}
        unknown = set(loaded) - fields
        if unknown:
            raise SystemExit(
                f"goal-simulate-batch: unknown keys in {args.config}: {sorted(unknown)}"
            )
        values.update(loaded)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        values[_FLAG_TO_FIELD.get(key, key)] = value
    return BatchSimulationConfig(**values)


def main(argv: typing.Sequence[str] | None = None) -> int:
    """Entry point of the ``goal-simulate-batch`` console script.

    Configures INFO logging, builds a :class:`BatchSimulationConfig`
    from ``argv`` (``sys.argv[1:]`` when ``None``) and the optional
    ``--config`` YAML with :func:`build_config`, runs
    :func:`run_batched_simulation`, and prints a summary and the files
    written.  Unknown ``--config`` keys raise ``SystemExit``.

    Returns
    -------
    int
        ``0`` on success; ``2`` for a roadmap sampler
        (``NotImplementedError``); ``1`` for an invalid config or a
        missing file (``ValueError`` / ``FileNotFoundError``).  Errors
        are printed to stderr.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = build_config(argv)
    try:
        result = run_batched_simulation(config)
    except NotImplementedError as exc:
        print(f"goal-simulate-batch: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(f"goal-simulate-batch: {exc}", file=sys.stderr)
        return 1
    print(
        f"Done: {len(result.final_structures)} structure(s), {result.steps_completed} steps, "
        f"backend={result.backend}, {result.wall_time_s:.1f} s"
    )
    for path in result.output_files:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
