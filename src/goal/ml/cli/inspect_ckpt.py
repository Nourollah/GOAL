"""``goal-inspect`` — describe a SIMURGH checkpoint without loading the model.

Accepts any of the three checkpoint formats::

    goal-inspect path/to/model.simurgh        # self-contained archive
    goal-inspect path/to/checkpoint_dir/      # GOALCheckpointManager directory
    goal-inspect path/to/specific.ckpt        # bare checkpoint file

Prints provenance (version, git commit, save time), model facts
(elements, pairs, scale), the best validation metrics, and — for
directories — the list of available checkpoints.
"""

from __future__ import annotations

import argparse
import sys
import typing

from goal.ml.training.archive import inspect_checkpoint

_LABEL_WIDTH = 18


def _row(label: str, value: typing.Any) -> str:
    return f"{label + ':':<{_LABEL_WIDTH}}{value}"


def _element_symbols(numbers: list[int]) -> str:
    try:
        from ase.data import chemical_symbols

        return " ".join(chemical_symbols[z] for z in numbers)
    except Exception:  # noqa: BLE001 — fall back to atomic numbers
        return " ".join(str(z) for z in numbers)


def format_report(info: dict[str, typing.Any]) -> str:
    """Render the dict from :func:`inspect_checkpoint` as a human report."""
    meta: dict[str, typing.Any] = info.get("metadata") or {}
    sidecar: dict[str, typing.Any] = info.get("sidecar") or {}

    lines: list[str] = [
        _row("Format", info["format"]),
        _row("Source frozen", "yes" if info["source_frozen"] else "NO — not self-contained"),
    ]

    if meta:
        lines.append(_row("Version", meta.get("simurgh_version", "?")))
        commit = meta.get("git_commit")
        lines.append(_row("Git commit", commit[:8] if commit else "unknown"))
        lines.append(_row("Saved", meta.get("timestamp", "?")))
        lines.append(_row("Model class", meta.get("model_class", "?")))
        if "elements" in meta:
            lines.append(_row("Elements", _element_symbols(meta["elements"])))
        if "dedicated_pairs" in meta:
            n_rare = 1 if meta.get("rare_artisan") else 0
            lines.append(_row("Pairs", f"{len(meta['dedicated_pairs'])} dedicated, {n_rare} rare"))
        if "avg_num_neighbors" in meta:
            lines.append(_row("Avg neighbours", f"{meta['avg_num_neighbors']:.2f}"))
        if "scale" in meta:
            lines.append(_row("Scale", f"{meta['scale']:.6g}"))

    if sidecar:
        if "epoch" in sidecar:
            lines.append(_row("Epoch", sidecar["epoch"]))
        if "pool" in sidecar:
            lines.append(_row("Pool", sidecar["pool"]))
        for name, value in sorted((sidecar.get("metrics") or {}).items()):
            if name.startswith("val/"):
                lines.append(_row(name, f"{value:.4f}"))

    ckpts: list[str] = info.get("available_checkpoints") or []
    if ckpts:
        lines.append(_row("Available ckpts", ckpts[0]))
        lines.extend(" " * _LABEL_WIDTH + name for name in ckpts[1:])

    return "\n".join(lines)


def main(argv: typing.Sequence[str] | None = None) -> int:
    """Entry point of the ``goal-inspect`` console script.

    Parses ``argv`` (``sys.argv[1:]`` when ``None``) for one path (a
    ``.simurgh`` archive, a checkpoint directory or a ``.ckpt`` file)
    and prints :func:`format_report` of :func:`inspect_checkpoint` to
    stdout.

    Returns
    -------
    int
        ``0`` on success; ``1`` if the path is not found (error on
        stderr).
    """
    parser = argparse.ArgumentParser(
        prog="goal-inspect",
        description="Inspect a .simurgh archive, checkpoint directory, or .ckpt file.",
    )
    parser.add_argument(
        "path",
        help="Path to a .simurgh archive, a checkpoint directory, or a .ckpt file.",
    )
    args = parser.parse_args(argv)

    try:
        info = inspect_checkpoint(args.path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_report(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
