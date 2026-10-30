"""Naming policy: the external reference baseline is never named in SIMURGH files.

The forbidden token is built at runtime from a reversed string so that it
never appears literally in the repository.  It matches case-insensitively,
and only where it is not preceded by a letter, so ordinary words that
contain it in the middle are not flagged.

Scope: the ``simurgh/dev`` package (with its ablations), the native ECE
modules, and the ``simurgh_dev`` configs, tests and scripts — both file
contents and file names.  False positives go in ``naming_allowlist.txt``
as ``path`` (whole file) or ``path:line``.

Run this file directly to list every occurrence in the whole repository::

    python tests/ml/simurgh_dev/test_naming_policy.py
"""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
TOKEN: re.Pattern[str] = re.compile(r"(?<![A-Za-z])" + "ecam"[::-1], re.IGNORECASE)
ALLOWLIST: Path = Path(__file__).with_name("naming_allowlist.txt")

#: Glob patterns (relative to the repository root) of the guarded files.
SCOPE: tuple[str, ...] = (
    "src/goal/ml/nn/models/simurgh/dev/**/*",
    "src/goal/ml/nn/primitives/edge_frame.py",
    "src/goal/ml/nn/primitives/o2.py",
    "src/goal/ml/nn/blocks/ece.py",
    "src/goal/ml/nn/heads/edge_energy.py",
    "src/goal/configs/ml/simurgh_dev_*.yaml",
    "src/goal/configs/ml/ablations/simurgh_dev/**/*",
    "tests/ml/simurgh_dev/**/*",
    "scripts/simurgh_dev/**/*",
)


def _allowed() -> set[str]:
    if not ALLOWLIST.is_file():
        return set()
    lines = (line.split("#", 1)[0].strip() for line in ALLOWLIST.read_text().splitlines())
    return {line for line in lines if line}


def scan(paths: list[Path]) -> list[str]:
    """``path`` (file name hits) and ``path:line`` (content hits) for every match."""
    allowed = _allowed()
    hits: list[str] = []
    for path in sorted(set(paths)):
        rel = path.relative_to(ROOT).as_posix()
        if rel in allowed:
            continue
        if TOKEN.search(rel):
            hits.append(rel)
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), start=1):
            if TOKEN.search(line) and f"{rel}:{n}" not in allowed:
                hits.append(f"{rel}:{n}")
    return hits


def scoped_files() -> list[Path]:
    return [p for pattern in SCOPE for p in ROOT.glob(pattern) if p.is_file()]


def test_simurgh_files_never_name_the_reference_baseline() -> None:
    hits = scan(scoped_files())
    assert not hits, "forbidden token in SIMURGH files:\n" + "\n".join(hits)


def test_token_matching_rule() -> None:
    token = "ecam"[::-1]
    assert TOKEN.search(token.upper())
    assert TOKEN.search(f"{token}_finetune")
    assert TOKEN.search(f"use_{token}")
    assert not TOKEN.search(f"ormo{token}")  # inside a word: not flagged


if __name__ == "__main__":
    git = shutil.which("git") or "git"
    tracked = subprocess.run(
        [git, "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    for hit in scan([ROOT / p for p in tracked]):
        print(hit)
