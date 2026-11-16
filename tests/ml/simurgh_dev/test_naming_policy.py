"""Naming policy: SIMURGH is native and names no other model.

Inside the scope below, three things are checked:

* **names**: no entry of ``tests/naming/other_model_names.txt`` in any file's
  text or path.  Default entries match case-insensitively with no letter
  directly before or after; ``cs:`` entries are case-sensitive; ``prefix:``
  entries match only at the start of a word (letters may follow);
* **imports**: no Python file imports a package of
  ``tests/naming/denied_imports.txt`` (parsed with ``ast``);
* **file names**: covered by the name check on each path;
* **versions**: no version number in a SIMURGH name.

Both lists live outside the scope, so they may spell the names.  Genuine
false positives go in ``naming_allowlist.txt`` as ``path`` (whole file) or
``path:line``, each followed by ``# reason``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT: Path = Path(__file__).resolve().parents[3]
DENYLIST: Path = ROOT / "tests" / "naming" / "other_model_names.txt"
DENIED_IMPORTS: Path = ROOT / "tests" / "naming" / "denied_imports.txt"
ALLOWLIST: Path = Path(__file__).with_name("naming_allowlist.txt")

#: Glob patterns (relative to the repository root) of the files in SIMURGH's scope.
SCOPE: tuple[str, ...] = (
    "src/goal/ml/nn/models/simurgh/dev/**/*",
    "src/goal/ml/nn/primitives/edge_frame.py",
    "src/goal/ml/nn/primitives/o2.py",
    "src/goal/ml/nn/blocks/ece.py",
    "src/goal/ml/nn/heads/edge_energy.py",
    "src/goal/configs/ml/simurgh_dev_*.yaml",
    "src/goal/configs/ml/simurgh_dev/**/*",
    "src/goal/configs/ml/ablations/simurgh_dev/**/*",
    "tests/ml/simurgh_dev/**/*",
    "scripts/simurgh_dev/**/*",
    "docs/ml/models/simurgh-dev.md",
    ".agents/skills/goal-simurgh/**/*",
    ".claude/skills/goal-simurgh/**/*",
)


def _entries(path: Path) -> list[str]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


def name_patterns(entries: list[str]) -> list[re.Pattern[str]]:
    """One regex per denylist entry (format in the module docstring)."""
    patterns = []
    for entry in entries:
        flags, after = re.IGNORECASE, r"(?![A-Za-z])"
        if entry.startswith("cs:"):
            entry, flags = entry[3:], 0
        elif entry.startswith("prefix:"):
            entry, after = entry[7:], ""
        patterns.append(re.compile(r"(?<![A-Za-z])" + re.escape(entry) + after, flags))
    return patterns


def allowed() -> set[str]:
    """Allowlist entries; each must carry a ``# reason``."""
    if not ALLOWLIST.is_file():
        return set()
    out = set()
    for raw in ALLOWLIST.read_text().splitlines():
        entry, _, reason = raw.partition("#")
        if entry.strip():
            assert reason.strip(), f"allowlist entry without a reason: {raw!r}"
            out.add(entry.strip())
    return out


def scoped_files() -> list[Path]:
    files = {p for pattern in SCOPE for p in ROOT.glob(pattern) if p.is_file()}
    return sorted(p for p in files if "__pycache__" not in p.parts)


def name_hits(paths: list[Path], patterns: list[re.Pattern[str]]) -> list[str]:
    """``path`` (a name in the path) and ``path:line: name`` (in the text) for every match."""
    skip = allowed()
    hits: list[str] = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if rel in skip:
            continue
        hits += [f"{rel}: {m.group(0)}" for p in patterns if (m := p.search(rel))]
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), start=1):
            if f"{rel}:{n}" in skip:
                continue
            hits += [f"{rel}:{n}: {m.group(0)}" for p in patterns if (m := p.search(line))]
    return hits


def imported_modules(source: str) -> set[str]:
    """Every module named by an ``import`` or ``from ... import`` statement."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module)
            out |= {f"{node.module}.{alias.name}" for alias in node.names}
    return out


def denied(module: str, packages: list[str]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in packages)


def test_scope_is_not_empty() -> None:
    rels = {p.relative_to(ROOT).as_posix() for p in scoped_files()}
    assert "src/goal/ml/nn/models/simurgh/dev/monolithic.py" in rels
    assert "src/goal/ml/nn/blocks/ece.py" in rels
    assert "src/goal/configs/ml/simurgh_dev/_base.yaml" in rels


def test_simurgh_names_no_other_model() -> None:
    hits = name_hits(scoped_files(), name_patterns(_entries(DENYLIST)))
    assert not hits, "other model names in SIMURGH files:\n" + "\n".join(hits)


VERSIONED: re.Pattern[str] = re.compile(r"simurgh(?:[ _-]?v?\d)", re.IGNORECASE)


def test_simurgh_names_carry_no_version_number() -> None:
    hits = name_hits(scoped_files(), [VERSIONED])
    assert not hits, "versioned SIMURGH names:\n" + "\n".join(hits)
    name = "SIMURGH"  # the examples are built here so this file passes its own check
    versioned = (f"{name}-2", f"{name} v3", f"{name.lower()}2", f"{name}_V2")
    assert all(VERSIONED.search(s) for s in versioned)
    assert not any(VERSIONED.search(s) for s in ("simurgh_dev", "SIMURGH (dev)", "simurgh_legacy"))


def test_simurgh_imports_no_other_model_package() -> None:
    packages = _entries(DENIED_IMPORTS)
    hits = [
        f"{path.relative_to(ROOT).as_posix()}: {module}"
        for path in scoped_files()
        if path.suffix == ".py"
        for module in sorted(imported_modules(path.read_text()))
        if denied(module, packages)
    ]
    assert not hits, "denied imports in SIMURGH files:\n" + "\n".join(hits)


@pytest.mark.parametrize(
    ("entry", "text", "hit"),
    [
        ("Foonet", "the FOONET model", True),  # case-insensitive by default
        ("Foonet", "foonet_v2", True),  # digits and underscores may touch it
        ("Foonet", "Foonets", False),  # but not letters
        ("Foonet", "SuperFoonet", False),
        ("cs:FOO", "FOO block", True),  # case-sensitive
        ("cs:FOO", "foo block", False),
        ("prefix:Bar", "barfinetune", True),  # letters may follow a prefix entry
        ("prefix:Bar", "crowbar", False),  # but not precede it
    ],
)
def test_matching_rule(entry: str, text: str, hit: bool) -> None:
    assert bool(name_patterns([entry])[0].search(text)) is hit


def test_import_check_sees_every_form() -> None:
    source = "import pkg_a\nimport pkg_b.sub as s\nfrom pkg_c.x import y\nfrom . import local\n"
    modules = imported_modules(source)
    assert all(denied(m, ["pkg_b"]) is (m == "pkg_b.sub") for m in modules)
    assert {"pkg_a", "pkg_b.sub", "pkg_c.x", "pkg_c.x.y"} == modules
    assert denied("pkg_c.x.y", ["pkg_c"]) and not denied("pkg_cx", ["pkg_c"])
