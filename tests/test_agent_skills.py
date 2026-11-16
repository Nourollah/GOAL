"""Checks for the agent skills in .agents/skills/ and their Claude Code copy in .claude/skills/."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".agents" / "skills"
SKILL_DIRS = sorted(p for p in SKILLS.glob("goal-*") if p.is_dir())
AGENT_FILES = sorted(
    [p for d in SKILL_DIRS for p in d.rglob("*") if p.is_file()]
    + [ROOT / "AGENTS.md", ROOT / "CLAUDE.md", ROOT / "docs/getting-started/agent-skills.md"]
)
FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
NAME = re.compile(r"^goal-[a-z0-9]+(?:-[a-z0-9]+)*$")
DENYLIST = ROOT / "tests/naming/other_model_names.txt"
SIMURGH_SKILL = SKILLS / "goal-simurgh"
VERSIONED = re.compile(r"simurgh(?:[ _-]?v?\d)", re.IGNORECASE)
PRIVATE = {
    "IPv4 address": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "home path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_]"),
    "e-mail or user@host": re.compile(r"\b[\w.-]+@[\w-]+(?:\.[\w-]+)+\b"),
}
REPO_PATH = re.compile(r"`((?:src|docs|tests|scripts|notebooks)/[^`\s]+)`")
LINK = re.compile(r"\]\(([^)#\s]+)\)")
CONFIG = ROOT / "src/goal/configs/ml/simurgh_dev_gmd26_FD.yaml"
SURFACE = SKILLS / "goal-simurgh/references/config_surface.md"


def split(path: Path) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", path.read_text(encoding="utf-8"), re.DOTALL)
    assert match, f"{path}: no YAML frontmatter between --- lines"
    return yaml.safe_load(match.group(1)), match.group(2)


def name_patterns() -> list[re.Pattern]:
    """Model names that must not appear inside SIMURGH (format: see goal-naming-policy)."""
    patterns = []
    for raw in DENYLIST.read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if not entry:
            continue
        flags, after = re.IGNORECASE, r"(?![A-Za-z])"
        if entry.startswith("cs:"):
            entry, flags = entry[3:], 0
        elif entry.startswith("prefix:"):
            entry, after = entry[7:], ""
        patterns.append(re.compile(r"(?<![A-Za-z])" + re.escape(entry) + after, flags))
    return patterns


def keys(tree: dict, prefix: str = "") -> set[str]:
    out = set()
    for key, value in tree.items():
        out.add(prefix + str(key))
        if isinstance(value, dict):
            out |= keys(value, f"{prefix}{key}.")
    return out


def test_skills_exist():
    assert SKILL_DIRS, "no goal-* skills in .agents/skills/"


@pytest.mark.parametrize("skill", SKILL_DIRS, ids=lambda p: p.name)
def test_frontmatter(skill: Path):
    meta, body = split(skill / "SKILL.md")
    assert set(meta) <= FIELDS, set(meta) - FIELDS
    name, description = meta["name"], meta["description"]
    assert name == skill.name and len(name) <= 64 and NAME.match(name)
    assert "claude" not in name and "anthropic" not in name
    assert 1 <= len(description) <= 1024
    assert "<" not in description and ">" not in description
    assert len(body.splitlines()) <= 500


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_content_policy(path: Path):
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        assert not VERSIONED.search(line), f"{path}:{number}: version number in a SIMURGH name"
        for label, pattern in PRIVATE.items():
            assert not pattern.search(line), f"{path}:{number}: {label}"


def test_simurgh_skill_names_no_other_model():
    assert DENYLIST.exists(), f"missing {DENYLIST}"
    patterns = name_patterns()
    for path in sorted(p for p in SIMURGH_SKILL.rglob("*") if p.is_file()):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern in patterns:
                hit = pattern.search(line)
                assert not hit, f"{path}:{number}: other model's name {hit.group(0)!r}"


@pytest.mark.parametrize("skill", SKILL_DIRS, ids=lambda p: p.name)
def test_links_and_repo_paths_exist(skill: Path):
    for path in (p for p in skill.rglob("*.md")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "(planned" in line:
                continue
            for target in LINK.findall(line):
                if not target.startswith(("http://", "https://")):
                    broken = f"{path}:{number}: broken link {target}"
                    assert (path.parent / target).exists(), broken
            for target in REPO_PATH.findall(line):
                if not any(c in target for c in "<>*{}$"):
                    assert (ROOT / target).exists(), f"{path}:{number}: missing {target}"


def test_index_lists_every_skill():
    names = {p.name for p in SKILL_DIRS}
    for doc in ("AGENTS.md", "docs/getting-started/agent-skills.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        listed = set(re.findall(r"^\| `(goal-[a-z0-9-]+)` \|", text, re.MULTILINE))
        assert listed == names, f"{doc}: {sorted(listed ^ names)}"


def test_claude_copy_is_in_sync():
    script = ROOT / "scripts/agents/sync_skills.py"
    command = [sys.executable, str(script), "--check"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not CONFIG.exists(), reason="SIMURGH GMD config not present")
def test_simurgh_config_surface_matches_config():
    block = re.search(r"```yaml\n(.*?)```", SURFACE.read_text(encoding="utf-8"), re.DOTALL)
    assert block, f"{SURFACE}: no yaml block"
    documented = yaml.safe_load(block.group(1))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIG.parent), version_base=None):
        composed = compose(config_name=CONFIG.stem, overrides=["paths.root_dir=/root"])
    actual = OmegaConf.to_container(composed, resolve=False)  # the config as goal-train sees it
    for section in ("model", "training"):
        assert keys(documented[section]) == keys(actual[section]), section
