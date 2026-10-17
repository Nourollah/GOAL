"""``goal-pack-archive`` — pack a self-contained ``.simurgh`` archive.

Builds the archive from a checkpoint directory written by
``GOALCheckpointManager`` at any time — during training (to ship the
current best checkpoint without waiting for the run to finish) or after::

    goal-pack-archive \\
        --checkpoint-dir logs/train/runs/.../checkpoints/ \\
        --checkpoint best \\
        --output my_model.simurgh
"""

from __future__ import annotations

import argparse
import sys
import typing

from goal.ml.training.archive import (
    ARCHIVE_SUFFIX,
    pack_simurgh_archive,
    resolve_checkpoint,
)


def main(argv: typing.Sequence[str] | None = None) -> int:
    """Entry point of the ``goal-pack-archive`` console script.

    Parses ``argv`` (``sys.argv[1:]`` when ``None``), resolves
    ``--checkpoint`` (``best``, ``last``, ``epoch=N`` or a ``.ckpt``
    filename/path) inside ``--checkpoint-dir`` with
    :func:`resolve_checkpoint`, and packs it with
    :func:`pack_simurgh_archive` to ``--output`` (default: the
    checkpoint stem plus ``.simurgh`` inside the checkpoint directory).

    Returns
    -------
    int
        ``0`` on success; ``1`` if resolving or packing raises
        ``FileNotFoundError`` or ``ValueError`` (message on stderr).
    """
    parser = argparse.ArgumentParser(
        prog="goal-pack-archive",
        description=f"Pack a checkpoint into a self-contained {ARCHIVE_SUFFIX} archive.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Checkpoint directory written by GOALCheckpointManager.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Which checkpoint to pack: 'best' (default), 'last', 'epoch=N', "
        "or a .ckpt filename/path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output archive path (default: <ckpt stem>{ARCHIVE_SUFFIX} "
        f"inside the checkpoint directory).",
    )
    args = parser.parse_args(argv)

    try:
        ckpt = resolve_checkpoint(args.checkpoint_dir, args.checkpoint)
        output = pack_simurgh_archive(ckpt, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Packed {ckpt.name} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
