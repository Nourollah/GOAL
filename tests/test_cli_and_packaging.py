"""Every console script answers ``--help``; packaging metadata stays sane;
no doc or script uses the removed config-group command syntax.

* ``goal-eval`` / ``goal-tune`` / ``goal-finetune`` used to name primary
  configs that never existed (``eval.yaml``, ``train.yaml``) and died with
  ``MissingConfigException`` before printing anything.
* ``pip install -e ".[md]"`` was unresolvable: the extra pinned
  ``rdkit<2026`` against the core ``rdkit>=2026.3.3`` and required
  ``xtb-python``, which is not on PyPI.
* Scripts and READMEs used ``model=…`` / ``data=…`` config groups that the
  single-file experiment configs replaced.
"""

from __future__ import annotations

import concurrent.futures
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
SCRIPTS: dict[str, str] = PYPROJECT["project"]["scripts"]


def _run_help(name: str, target: str) -> subprocess.CompletedProcess:
    """Call the entry point exactly as the console script would, with --help."""
    module, func = target.split(":")
    code = (
        "import sys\n"
        f"sys.argv = [{name!r}, '--help']\n"
        f"from {module} import {func} as main\n"
        "rc = main()\n"
        "sys.exit(rc if isinstance(rc, int) else 0)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=ROOT,
        check=False,
    )


@pytest.fixture(scope="module")
def help_results() -> dict[str, subprocess.CompletedProcess]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {name: pool.submit(_run_help, name, target) for name, target in SCRIPTS.items()}
        return {name: f.result() for name, f in futures.items()}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_console_script_help_exits_zero(name: str, help_results) -> None:
    result = help_results[name]
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{name} --help failed:\n{output[-3000:]}"
    assert "MissingConfigException" not in output
    assert "Traceback" not in output


def test_new_commands_are_registered() -> None:
    for name in ("goal-simulate-batch", "goal-pack-zarr", "goal-distill"):
        assert name in SCRIPTS


@pytest.mark.parametrize(
    ("module", "config_name"),
    [
        ("goal.ml.cli.evaluate", "eval"),
        ("goal.ml.cli.tune", "simurgh_gmd26_hpo"),
        ("goal.ml.cli.finetune", "finetune/simurgh_lora"),
        ("goal.ml.cli.train", "simurgh_gmd26"),
    ],
)
def test_default_configs_exist(module: str, config_name: str) -> None:
    from goal import CONFIGS_DIR

    path = Path(CONFIGS_DIR) / "ml" / f"{config_name}.yaml"
    assert path.is_file(), f"{module} defaults to {config_name}, which does not exist"
    assert (
        f'config_name="{config_name}"'
        in (ROOT / "src" / Path(*module.split("."))).with_suffix(".py").read_text()
    )


class TestPyproject:
    def _deps(self, extra: str) -> list[str]:
        return PYPROJECT["project"]["optional-dependencies"][extra]

    def test_md_extra_is_resolvable(self) -> None:
        md = " ".join(self._deps("md"))
        assert "xtb-python" not in md, "xtb-python is not on PyPI"
        assert "rdkit" not in md, "rdkit is a core dependency; an md pin conflicts with it"
        core_rdkit = next(d for d in PYPROJECT["project"]["dependencies"] if d.startswith("rdkit"))
        assert ">=2026" in core_rdkit

    def test_self_references_use_the_distribution_name(self) -> None:
        name = PYPROJECT["project"]["name"]
        for extra, deps in PYPROJECT["project"]["optional-dependencies"].items():
            for dep in deps:
                if dep.startswith("goal"):
                    assert dep.startswith(f"{name}["), f"[{extra}] references {dep}"

    def test_new_extras(self) -> None:
        assert any(d.startswith("zarr>=3") for d in self._deps("zarr"))
        assert any(d.startswith("peft") for d in self._deps("lora"))
        assert "nvalchemi-toolkit-ops[torch-cu12]" in self._deps("nvalchemiops-cu12")

    def test_no_unresolvable_nvalchemi_toolkit_extra(self) -> None:
        """Every nvalchemi-toolkit release so far requires Python < 3.14 and
        GOAL requires >= 3.14, so an extra naming it could never resolve."""
        for deps in PYPROJECT["project"]["optional-dependencies"].values():
            assert not any(re.match(r"nvalchemi-toolkit(\[|$|[<>=])", d) for d in deps)

    def test_pixi_tasks_call_console_scripts(self) -> None:
        tasks = PYPROJECT["tool"]["pixi"]["tasks"]
        assert tasks["simulate"] == "goal-simulate"
        assert tasks["simulate-mts"] == "goal-simulate-mts"

    def test_gpu_marker_registered(self) -> None:
        markers = PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]
        assert any(m.startswith("gpu:") for m in markers)


# ---------------------------------------------------------------------------
# Stale command syntax (pixi task: lint-configs)
# ---------------------------------------------------------------------------

#: A goal command (console script, ``python -m goal…cli``, or a pixi task).
_COMMAND = re.compile(
    r"(goal-(train|eval|tune|finetune|simulate)\b|goal\.ml\.cli\.\w+|python src/train\.py"
    r"|pixi run\b[^\n]*\b(train|eval|tune|finetune|simulate)\b)"
)
#: ``group=option`` selection of a config group that no longer exists, i.e.
#: not ``group.key=…``, ``+group=…`` (appending a preset) or ``…@_global_=``.
_GROUP_SELECTION = re.compile(
    r"(?<![\w.+@~/-])(model|data|trainer|logger|callbacks|experiment|hparams_search)=[A-Za-z]"
)
_ALLOWED_CONTEXT = (
    "no longer work",
    "does not work",
    "older docs",
    "old multi-file",
    "removed config-group syntax",
)

_SCANNED = [
    *ROOT.glob("scripts/*.sh"),
    *ROOT.glob("scripts/*.py"),
    ROOT / "README.md",
    *ROOT.glob("docs/**/*.md"),
    *ROOT.glob("src/**/*.md"),
    *ROOT.glob("src/goal/configs/**/*.yaml"),
    *ROOT.glob("src/goal/**/*.py"),
]


def _stale_lines() -> list[str]:
    hits: list[str] = []
    for path in _SCANNED:
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if any(ctx in line for ctx in _ALLOWED_CONTEXT):
                continue
            if (
                "+experiment=" in line
                or "python src/train.py" in line
                or _COMMAND.search(line)
                and _GROUP_SELECTION.search(line)
            ):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    return hits


def test_no_stale_config_group_syntax() -> None:
    hits = _stale_lines()
    assert not hits, "Removed config-group syntax still in use:\n" + "\n".join(hits)
