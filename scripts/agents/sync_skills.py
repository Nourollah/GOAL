#!/usr/bin/env python
"""Copy GOAL's agent skills from .agents/skills/ (the source) to .claude/skills/.

Claude Code reads project skills only from .claude/skills/; Codex, Cursor and other
Agent Skills tools read .agents/skills/. Only folders named goal-* are managed.

    python scripts/agents/sync_skills.py           # update .claude/skills/
    python scripts/agents/sync_skills.py --check   # exit 1 if .claude/skills/ is stale
    python scripts/agents/sync_skills.py --user    # also install into the user skill folders
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".agents" / "skills"
MIRROR = ROOT / ".claude" / "skills"
USER_TARGETS = (Path.home() / ".claude" / "skills", Path.home() / ".agents" / "skills")


def skills(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {p.name: p for p in sorted(root.glob("goal-*")) if (p / "SKILL.md").is_file()}


def same_tree(a: Path, b: Path) -> bool:
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    return not (mismatch or errors) and all(same_tree(a / d, b / d) for d in cmp.common_dirs)


def stale(target: Path, prune: bool) -> list[str]:
    src, dst = skills(SOURCE), skills(target)
    names = set(src) | (set(dst) if prune else set())
    return sorted(
        n for n in names if n not in src or n not in dst or not same_tree(src[n], dst[n])
    )


def copy_into(target: Path, prune: bool, force: bool) -> list[str]:
    src, done = skills(SOURCE), []
    target.mkdir(parents=True, exist_ok=True)
    for name in stale(target, prune):
        path = target / name
        if path.exists() and not (prune or force):
            print(f"skip {path}: differs from the source (use --force to replace)")
            continue
        if path.exists():
            shutil.rmtree(path)
        if name in src:
            shutil.copytree(src[name], path)
        done.append(name)
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report a stale copy only")
    parser.add_argument("--user", action="store_true", help="also install for the user")
    parser.add_argument("--force", action="store_true", help="with --user, replace copies")
    args = parser.parse_args()
    if not skills(SOURCE):
        print(f"no goal-* skills found in {SOURCE}")
        return 1
    if args.check:
        bad = stale(MIRROR, prune=True)
        for name in bad:
            print(f"stale: .claude/skills/{name}")
        return 1 if bad else 0
    for name in copy_into(MIRROR, prune=True, force=True):
        print(f"synced .claude/skills/{name}")
    if args.user:
        for target in USER_TARGETS:
            for name in copy_into(target, prune=False, force=args.force):
                print(f"installed {target / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
