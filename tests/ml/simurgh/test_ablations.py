"""Test 13 — the commented add-on blocks in the ARACE configs are valid.

The add-ons ship **commented out** in the ARACE experiment configs, and
uncommenting a block is the whole enabling mechanism.  That makes those
comments executable documentation, so they need the same protection as
code: a renamed hyperparameter, a width that no longer fits the artisan
irreps, or a key that no longer maps to a constructor parameter would
otherwise sit there rotting until someone uncommented it mid-experiment.

These tests therefore read the real blocks out of the real files —
commented or already uncommented, since enabling a module *is* editing
the file — merge them into the composed config the way Hydra would, and
run the resulting model.  Every rung is set up explicitly (sections it
wants are added, the rest removed), so the suite gives the same answer
whichever blocks happen to be live in the working copy.  All four rungs
of the ablation ladder are covered from the single base file:

    1. baseline  — both blocks commented (as shipped)
    2. fragment  — fragment_interaction uncommented
    3. gate only — adaptive_gate uncommented
    4. full      — both uncommented

The documented CLI recipe (Hydra ``+section={...}`` / ``~section``) is
checked too, since that is the other way the configs tell you to switch
the modules on and off.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

from goal.ml.data.fragments import compute_fragment_index
from goal.ml.nn.heads.energy_forces import EnergyForcesHead
from goal.ml.nn.models.simurgh.backbone_arace import SimurghAraceBackbone
from goal.ml.registry import BACKBONE_REGISTRY
from tests.ml.simurgh.conftest import _build_graph
from tests.ml.simurgh.test_fragment_interaction import _methane_positions

_CONFIGS_ML_DIR = Path(__file__).parents[3] / "src" / "goal" / "configs" / "ml"
_BASE_CONFIG = "simurgh_gmd26"

# Configs that document the add-ons inline.  Both must keep the blocks in
# sync — the monolithic variant takes the identical keys.
ARACE_CONFIG_FILES: list[str] = ["simurgh_gmd26.yaml", "monolithic_arace_gmd26.yaml"]

ADDON_KEYS: list[str] = ["fragment_interaction", "adaptive_gate"]

# (label, sections to uncomment)
LADDER: list[tuple[str, tuple[str, ...]]] = [
    ("baseline", ()),
    ("fragment_only", ("fragment_interaction",)),
    ("adaptive_gate_only", ("adaptive_gate",)),
    ("full", ("fragment_interaction", "adaptive_gate")),
]

# Legacy ACE-first backbone: the add-ons attach to an artisan+ACE round it
# does not have, so these files must NOT offer uncommentable blocks.
LEGACY_CONFIG_FILES: list[str] = ["simurgh_ace_first_gmd26.yaml", "simurgh_md17.yaml"]


# ----------------------------------------------------------------------
# Helpers — read the documentation back out of the YAML
# ----------------------------------------------------------------------


def _live_block(config_file: str, key: str) -> dict | None:
    """Return ``model.backbone.<key>`` if it is *active* YAML in the file."""
    parsed = yaml.safe_load((_CONFIGS_ML_DIR / config_file).read_text()) or {}
    backbone = (parsed.get("model") or {}).get("backbone") or {}
    block = backbone.get(key)
    return block if isinstance(block, dict) else None


def _commented_block(config_file: str, key: str) -> dict | None:
    """Return the commented-out ``key:`` block of *config_file*, as YAML.

    Finds ``# <key>:`` and takes the following comment lines that are
    indented deeper than it, strips one comment marker per line, and
    parses the result.  ``None`` when the file has no such block.
    """
    lines: list[str] = (_CONFIGS_ML_DIR / config_file).read_text().splitlines()
    collected: list[str] = []
    indent: int | None = None
    for line in lines:
        if indent is None:
            if re.match(rf"^\s*#\s*{key}:\s*$", line):
                indent = len(line) - len(line.lstrip())
                collected.append(re.sub(r"^(\s*)#\s?", r"\1", line))
            continue
        if not line.strip().startswith("#"):
            break  # blank line or real YAML ends the block
        decommented: str = re.sub(r"^(\s*)#\s?", r"\1", line)
        if decommented.strip() and (len(decommented) - len(decommented.lstrip())) <= indent:
            break  # dedented back out of the block
        collected.append(decommented)

    if not collected:
        return None
    parsed = yaml.safe_load(textwrap.dedent("\n".join(collected)))
    return parsed.get(key) if isinstance(parsed, dict) else None


def _documented_block(config_file: str, key: str) -> dict | None:
    """The add-on block, whether it is currently on or off in the file.

    Uncommenting a block is the documented way to enable a module, so the
    file legitimately exists in both states and these tests must pass in
    either.  A live section wins over the commented one (that is the
    value a run would actually use).
    """
    return _live_block(config_file, key) or _commented_block(config_file, key)


def _compose(config_name: str = _BASE_CONFIG, overrides: list[str] | None = None):
    from hydra import compose, initialize_config_dir

    os.environ.setdefault("PROJECT_ROOT", str(_CONFIGS_ML_DIR.parents[1]))
    with initialize_config_dir(config_dir=str(_CONFIGS_ML_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def _cfg_with_addons(sections: tuple[str, ...]) -> DictConfig:
    """Composed base config with exactly *sections* switched on.

    Sections named in *sections* are set from the block documented in the
    file; every other add-on section is removed.  Being explicit in both
    directions makes each rung of the ladder reproducible no matter which
    blocks happen to be uncommented in the working copy right now.

    ``set_struct(False)`` mirrors what Hydra's ``+key={...}`` override
    does: a composed config rejects new keys by default, which is exactly
    why the config documents the override with a leading ``+``.
    """
    cfg = _compose()
    OmegaConf.set_struct(cfg, False)
    for key in ADDON_KEYS:
        if key in sections:
            block = _documented_block("simurgh_gmd26.yaml", key)
            assert block is not None, f"simurgh_gmd26.yaml documents no '{key}' block"
            cfg.model.backbone[key] = OmegaConf.create(block)
        else:
            cfg.model.backbone.pop(key, None)
    return cfg


def _backbone_from_cfg(cfg) -> SimurghAraceBackbone:
    """Build the backbone the way ``train.py`` does.

    ``train.py`` forwards every ``model.backbone`` key except ``name`` as a
    constructor kwarg and overrides ``elements`` from the training set.
    Radial widths are shrunk to keep the test fast; the add-on sections are
    used verbatim, which is the point.
    """
    kwargs = OmegaConf.to_container(
        OmegaConf.create({k: v for k, v in cfg.model.backbone.items() if k != "name"}),
        resolve=True,
    )
    kwargs["elements"] = [1, 6, 7, 8]
    kwargs["embedding_dim"] = 16
    kwargs["num_elements"] = 9
    kwargs["cutoff"] = 5.0
    kwargs["artisan"] = {
        **dict(kwargs["artisan"]),
        "hidden_irreps": "16x0e + 16x1o + 16x2e",
        "num_rbf": 4,
        "radial_hidden": 16,
    }
    return SimurghAraceBackbone(**kwargs).double()


def _methane_batch(with_fragments: bool) -> Batch:
    positions, numbers = _methane_positions()
    graph = _build_graph(positions, numbers, cutoff=5.0)
    if with_fragments:
        graph.fragment_index = compute_fragment_index(positions, numbers)
    return Batch.from_data_list([graph])


# ----------------------------------------------------------------------
# The commented blocks are present, parseable, and in sync
# ----------------------------------------------------------------------


class TestDocumentedBlocks:
    @pytest.mark.parametrize("config_file", ARACE_CONFIG_FILES)
    @pytest.mark.parametrize("key", ADDON_KEYS)
    def test_block_exists_and_parses(self, config_file: str, key: str) -> None:
        block = _documented_block(config_file, key)
        assert block is not None, (
            f"{config_file} no longer carries a '{key}' block, commented or "
            f"otherwise — the config is the only documentation of this module"
        )
        assert isinstance(block, dict) and block, f"{config_file}: '{key}' block is empty"

    @pytest.mark.parametrize("key", ADDON_KEYS)
    def test_both_arace_configs_document_the_same_defaults(self, key: str) -> None:
        """The two ARACE configs must not drift apart *as documentation*.

        Only compared while both are still commented out: once a block is
        uncommented it is a live experiment knob, and tuning it in one file
        is a legitimate thing to do, not drift.
        """
        commented = {f: _commented_block(f, key) for f in ARACE_CONFIG_FILES}
        if any(block is None for block in commented.values()):
            pytest.skip(f"'{key}' is enabled in at least one config — nothing to compare")

        modular, monolithic = commented.values()
        assert modular == monolithic, (
            f"'{key}' documented differently in the modular and monolithic "
            f"ARACE configs: {commented}"
        )

    @pytest.mark.parametrize("key", ADDON_KEYS)
    def test_documented_keys_match_the_module_signature(self, key: str) -> None:
        """Every documented hyperparameter must be a real constructor
        argument — this is what catches a renamed or dropped parameter."""
        import inspect

        from goal.ml.nn.blocks.adaptive_gate import AdaptiveDepthGate
        from goal.ml.nn.blocks.fragment_interaction import (
            EquivariantFragmentInteraction,
        )

        module_cls = {
            "fragment_interaction": EquivariantFragmentInteraction,
            "adaptive_gate": AdaptiveDepthGate,
        }[key]
        accepted = set(inspect.signature(module_cls.__init__).parameters) - {
            "self",
            "irreps_node",
        }
        documented = set(_documented_block("simurgh_gmd26.yaml", key) or {})
        unknown = documented - accepted
        assert not unknown, (
            f"simurgh_gmd26.yaml documents {sorted(unknown)} under '{key}', "
            f"but {module_cls.__name__} accepts only {sorted(accepted)}"
        )

    @pytest.mark.parametrize("config_file", LEGACY_CONFIG_FILES)
    @pytest.mark.parametrize("key", ADDON_KEYS)
    def test_legacy_configs_offer_no_uncommentable_block(self, config_file: str, key: str) -> None:
        """The ACE-first backbone rejects these keys, so its configs must not
        tempt anyone with a block to uncomment — only a pointer to ARACE."""
        assert _documented_block(config_file, key) is None, (
            f"{config_file} carries a '{key}' block, but its backbone "
            f"(name: simurgh) does not accept that key"
        )

    @pytest.mark.parametrize("config_file", LEGACY_CONFIG_FILES)
    def test_legacy_configs_point_at_the_arace_configs(self, config_file: str) -> None:
        text = (_CONFIGS_ML_DIR / config_file).read_text()
        assert "ARACE-only" in text
        assert "simurgh_gmd26.yaml" in text


# ----------------------------------------------------------------------
# The four rungs of the ladder all build and run
# ----------------------------------------------------------------------


class TestAblationLadder:
    @pytest.mark.parametrize(("label", "sections"), LADDER)
    def test_config_composes_with_the_expected_addons(
        self, label: str, sections: tuple[str, ...]
    ) -> None:
        cfg = _cfg_with_addons(sections)

        assert cfg.model.backbone.name == "simurgh_arace"
        assert BACKBONE_REGISTRY.get(cfg.model.backbone.name) is SimurghAraceBackbone
        for key in ADDON_KEYS:
            assert (cfg.model.backbone.get(key, None) is not None) is (key in sections)

        # Fragment CA cannot work without the dataset labels, and the base
        # config keeps them on so enabling the module is a one-line edit.
        if "fragment_interaction" in sections:
            assert cfg.data.compute_fragment_index is True
            assert float(cfg.data.fragment_covalent_cutoff) > 0.0

    @pytest.mark.parametrize(("label", "sections"), LADDER)
    def test_backbone_instantiates_with_the_configured_addons(
        self, label: str, sections: tuple[str, ...]
    ) -> None:
        backbone = _backbone_from_cfg(_cfg_with_addons(sections))

        assert backbone.fragment_interaction_enabled is ("fragment_interaction" in sections)
        assert backbone.adaptive_gate_enabled is ("adaptive_gate" in sections)
        for rnd in backbone.rounds:
            assert (rnd.fragment_interaction is not None) is ("fragment_interaction" in sections)
            assert (rnd.adaptive_gate is not None) is ("adaptive_gate" in sections)

    @pytest.mark.parametrize(("label", "sections"), LADDER)
    def test_methane_forward_gives_finite_energy_and_real_forces(
        self, label: str, sections: tuple[str, ...]
    ) -> None:
        cfg = _cfg_with_addons(sections)
        backbone = _backbone_from_cfg(cfg)
        backbone.eval()
        head = EnergyForcesHead(irreps_in="16x0e + 16x1o + 16x2e", hidden_dim=16).double()
        head.eval()

        batch = _methane_batch(with_fragments=bool(cfg.data.compute_fragment_index))
        out = head(backbone(batch), batch)

        assert out["energy"].shape == (1,)
        assert torch.isfinite(out["energy"]).all(), f"{label}: energy not finite"
        assert torch.isfinite(out["forces"]).all(), f"{label}: forces not finite"
        assert (out["forces"].abs() > 0).any(), f"{label}: forces identically zero"

        net = out["forces"].sum(dim=0).norm().item()
        assert net < 1e-5, f"{label}: Newton violated, ‖Σ F‖ = {net:.3e}"

    @pytest.mark.parametrize(("label", "sections"), LADDER)
    def test_side_channels_match_the_enabled_addons(
        self, label: str, sections: tuple[str, ...]
    ) -> None:
        cfg = _cfg_with_addons(sections)
        backbone = _backbone_from_cfg(cfg)
        backbone.train()
        backbone(_methane_batch(with_fragments=bool(cfg.data.compute_fragment_index)))

        if "adaptive_gate" in sections:
            assert backbone.last_aux_loss is not None
            assert float(backbone.last_aux_loss) > 0.0
            assert all(g is not None for g in backbone.last_gate_scores)
        else:
            assert backbone.last_aux_loss is None
            assert all(g is None for g in backbone.last_gate_scores)

    def test_each_addon_adds_parameters_over_the_baseline(self) -> None:
        counts = {
            label: sum(p.numel() for p in _backbone_from_cfg(_cfg_with_addons(s)).parameters())
            for label, s in LADDER
        }
        assert counts["fragment_only"] > counts["baseline"]
        assert counts["adaptive_gate_only"] > counts["baseline"]
        assert counts["full"] > counts["fragment_only"]
        assert counts["full"] > counts["adaptive_gate_only"]

    def test_working_copy_builds_what_it_says(self) -> None:
        """Whatever is currently uncommented in the file is what you get.

        Deliberately not asserting that the working copy is the baseline —
        uncommenting a block is the documented way to enable a module, so
        the file is expected to change.  What must always hold is that the
        model agrees with the file: a live section builds the module, an
        absent one does not.
        """
        cfg = _compose()
        backbone = _backbone_from_cfg(cfg)

        assert backbone.fragment_interaction_enabled is (
            _live_block("simurgh_gmd26.yaml", "fragment_interaction") is not None
        )
        assert backbone.adaptive_gate_enabled is (
            _live_block("simurgh_gmd26.yaml", "adaptive_gate") is not None
        )


# ----------------------------------------------------------------------
# The documented CLI recipe works
# ----------------------------------------------------------------------


class TestDocumentedCliRecipe:
    def test_plus_override_adds_a_section(self) -> None:
        # `+` adds a section that the file does not have.  Start from a file
        # state without it so the override is doing the work.
        overrides = [
            "+model.backbone.adaptive_gate="
            "{n_scalar:16,hard_threshold:0.5,aux_loss_weight:0.01,init_bias:1.0}"
        ]
        if _live_block("simurgh_gmd26.yaml", "adaptive_gate") is not None:
            # Already enabled in the working copy — drop it first, then the
            # `+` recipe is what puts it back.
            overrides = ["~model.backbone.adaptive_gate", *overrides]
        cfg = _compose(overrides=overrides)

        assert cfg.model.backbone.adaptive_gate.n_scalar == 16
        assert _backbone_from_cfg(cfg).adaptive_gate_enabled

    def test_tilde_override_removes_a_section(self) -> None:
        # Ensure the section is present (from the file or from `+`), then
        # delete it the way the config's comment says to.
        block = _documented_block("simurgh_gmd26.yaml", "fragment_interaction")
        base = _compose()
        OmegaConf.set_struct(base, False)
        base.model.backbone.fragment_interaction = OmegaConf.create(block)
        assert _backbone_from_cfg(base).fragment_interaction_enabled

        # `~` only applies when the key is actually in the file; with the
        # block commented out there is nothing to delete and the plain
        # compose already is the stripped state.
        stripped = _compose(
            overrides=(
                ["~model.backbone.fragment_interaction"]
                if _live_block("simurgh_gmd26.yaml", "fragment_interaction") is not None
                else []
            )
        )
        assert stripped.model.backbone.get("fragment_interaction", None) is None
        assert not _backbone_from_cfg(stripped).fragment_interaction_enabled

    def test_empty_section_counts_as_disabled(self) -> None:
        """Documented behaviour: commenting out only the inner keys leaves an
        empty mapping, which must disable the module rather than build it
        with defaults."""
        cfg = _compose()
        OmegaConf.set_struct(cfg, False)
        cfg.model.backbone.fragment_interaction = OmegaConf.create({})
        cfg.model.backbone.adaptive_gate = OmegaConf.create({})
        backbone = _backbone_from_cfg(cfg)

        assert not backbone.fragment_interaction_enabled
        assert not backbone.adaptive_gate_enabled
