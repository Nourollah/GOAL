"""``artisan.n_elements`` sizing of the element-conditioning tables.

The tables are looked up by atomic number (``one_hot(Z)`` /
``embedding(Z)``), so they need ``max(Z) + 1`` rows — not one row per
distinct element, which would silently fold C and O onto the same row.
With the legacy default of 120 rows every row above ``max(Z)`` is a dead
parameter: on GMD-26 (H, C, O) that was ~70 % of the model.
"""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from goal import CONFIGS_DIR
from goal.ml.nn.blocks.artisans import LEGACY_N_ELEMENTS, resolve_n_elements
from goal.ml.nn.models.simurgh.arace import MonolithicArace
from goal.ml.nn.models.simurgh.backbone_arace import SimurghAraceBackbone

ELEMENTS = (1, 6, 8)
ARTISAN = {
    "architecture": "equivariant",
    "hidden_irreps": "4x0e + 4x1o",
    "num_rbf": 4,
    "radial_hidden": 8,
    "n_scalar_out": 4,
    "final_hidden": 4,
    "element_conditioned": True,
}


def _arace(n_elements=None, **extra) -> SimurghAraceBackbone:
    artisan = dict(ARTISAN)
    if n_elements is not None:
        artisan["n_elements"] = n_elements
    return SimurghAraceBackbone(
        elements=ELEMENTS,
        num_rounds=1,
        artisan=artisan,
        cutoff=4.0,
        embedding_dim=8,
        num_elements=9,
        avg_num_neighbors=4.0,
        **extra,
    )


def _n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class TestResolve:
    def test_auto_is_max_z_plus_one_not_element_count(self) -> None:
        assert resolve_n_elements("auto", [1, 6, 8]) == 9
        assert resolve_n_elements("AUTO", [8, 1]) == 9

    def test_absent_is_the_legacy_size(self) -> None:
        assert resolve_n_elements(None, [1, 6, 8]) == LEGACY_N_ELEMENTS == 120

    def test_explicit_int_must_index_every_element(self) -> None:
        assert resolve_n_elements(12, [1, 6, 8]) == 12
        with pytest.raises(ValueError, match="max\\(Z\\) \\+ 1 = 9"):
            resolve_n_elements(3, [1, 6, 8])  # would fold C and O together

    def test_unknown_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="'auto'"):
            resolve_n_elements("all", [1])


class TestBackboneSizing:
    def test_auto_sizes_every_conditioning_table(self) -> None:
        backbone = _arace("auto")
        tables = [m for name, m in backbone.named_modules() if name.endswith("element_linear")]
        assert tables and all(t.in_features == 9 for t in tables)

    def test_absent_key_keeps_legacy_120_rows(self) -> None:
        """Checkpoints whose config predates the key must rebuild unchanged."""
        backbone = _arace(None)
        tables = [m for name, m in backbone.named_modules() if name.endswith("element_linear")]
        assert all(t.in_features == 120 for t in tables)

    def test_auto_removes_only_dead_rows(self) -> None:
        legacy, auto = _arace(None), _arace("auto")
        tables = [m for n, m in auto.named_modules() if n.endswith("element_linear")]
        dead = sum((120 - 9) * t.out_features for t in tables)
        assert _n_params(legacy) - _n_params(auto) == dead

    def test_conditioning_still_distinguishes_elements(self) -> None:
        """One-hot rows for H, C and O stay distinct under 'auto'."""
        backbone = _arace("auto")
        table = next(m for n, m in backbone.named_modules() if n.endswith("element_linear"))
        rows = torch.nn.functional.one_hot(torch.tensor([1, 6, 8]), table.in_features).double()
        assert torch.linalg.matrix_rank(rows) == 3

    def test_rare_artisan_embedding_is_sized_too(self) -> None:
        artisan = {
            **ARTISAN,
            "n_elements": "auto",
            "rare_artisan": {"enabled": True, "min_pair_frequency": 0.2},
            "pair_counts": {"1_1": 50, "1_6": 40, "6_6": 5, "1_8": 3, "6_8": 1, "8_8": 1},
        }
        backbone = SimurghAraceBackbone(
            elements=ELEMENTS,
            num_rounds=1,
            artisan=artisan,
            cutoff=4.0,
            embedding_dim=8,
            num_elements=9,
            avg_num_neighbors=4.0,
        )
        embeds = [m for n, m in backbone.named_modules() if n.endswith("pair_embed")]
        assert embeds and all(e.num_embeddings == 9 for e in embeds)

    def test_monolithic_arace_accepts_auto(self) -> None:
        model = MonolithicArace(
            elements=ELEMENTS,
            num_rounds=1,
            artisan={**ARTISAN, "n_elements": "auto"},
            cutoff=4.0,
            embedding_dim=8,
            num_elements=9,
        )
        tables = [m for n, m in model.named_modules() if n.endswith("element_linear")]
        assert tables and all(t.in_features == 9 for t in tables)

    def test_too_small_explicit_value_fails_at_construction(self) -> None:
        with pytest.raises(ValueError, match="cannot index atomic number 8"):
            _arace(5)


@pytest.mark.parametrize(
    "config_name", ["simurgh_gmd26", "nf_simurgh_gmd26", "monolithic_arace_gmd26"]
)
def test_shipped_gmd26_configs_are_sized_by_auto(config_name: str) -> None:
    """The production GMD-26 configs set ``n_elements: auto``: with the H/C/O
    elements train.py discovers, the model drops below 2 M parameters
    (it was 5.2-5.4 M with 120-row tables)."""
    from goal.ml.registry import BACKBONE_REGISTRY

    with initialize_config_dir(config_dir=f"{CONFIGS_DIR}/ml", version_base=None):
        cfg = compose(config_name=config_name)
    assert cfg.model.backbone.artisan.n_elements == "auto"
    kwargs = OmegaConf.to_container(cfg.model.backbone, resolve=True)
    name = kwargs.pop("name")
    kwargs["elements"] = [1, 6, 8]
    if kwargs.get("avg_num_neighbors", 0) is None:
        kwargs["avg_num_neighbors"] = 20.0
    kwargs["artisan"]["rare_artisan"] = {"enabled": False}
    kwargs["artisan"].pop("pair_counts", None)
    model = BACKBONE_REGISTRY.get(name)(**kwargs)
    assert _n_params(model) < 2_000_000


class TestDerivedArgsPersisted:
    """train.py writes dataset-derived backbone arguments into the saved
    config, so a rebuild (and ``n_elements: auto``) sees the real elements."""

    def test_derived_values_written_back(self) -> None:
        from goal.ml.cli.train import _persist_derived_backbone_args

        cfg = OmegaConf.create(
            {
                "model": {
                    "backbone": {
                        "name": "simurgh_arace",
                        "scale": None,
                        "avg_num_neighbors": None,
                        "atomic_energies": {"mode": "dataset", "values": None},
                    }
                }
            }
        )
        OmegaConf.set_struct(cfg, True)
        kwargs = {
            "elements": [1, 6, 8],
            "scale": 0.17,
            "avg_num_neighbors": 19.8,
            "atomic_energies": {"mode": "dataset", "values": {1: -13.6, 6: -1028.5, 8: -2041.1}},
        }
        _persist_derived_backbone_args(cfg, kwargs)
        bb = cfg.model.backbone
        assert list(bb.elements) == [1, 6, 8]
        assert bb.scale == pytest.approx(0.17)
        assert bb.avg_num_neighbors == pytest.approx(19.8)
        assert bb.atomic_energies["values"][6] == pytest.approx(-1028.5)
        # round-trips through YAML (config.yaml / hparams)
        again = OmegaConf.create(OmegaConf.to_yaml(cfg))
        assert again.model.backbone.atomic_energies["values"][8] == pytest.approx(-2041.1)

    def test_unchanged_values_leave_config_untouched(self) -> None:
        from goal.ml.cli.train import _persist_derived_backbone_args

        cfg = OmegaConf.create({"model": {"backbone": {"elements": [1, 6], "scale": 1.0}}})
        before = OmegaConf.to_yaml(cfg)
        _persist_derived_backbone_args(cfg, {"elements": [1, 6], "scale": 1.0, "num_rounds": 2})
        assert OmegaConf.to_yaml(cfg) == before
