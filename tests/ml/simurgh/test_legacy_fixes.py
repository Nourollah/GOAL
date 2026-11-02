"""Legacy SIMURGH fixes: new errors, opt-in numerical fixes, and bit identity at the defaults."""

from __future__ import annotations

import io
import typing
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.cli.builders import build_loss
from goal.ml.registry import LOSS_REGISTRY
from tests.ml.simurgh.fixtures.make_legacy_reference import (
    MODELS,
    STRUCTURES,
    batch,
    build,
    graph,
    seeded,
)
from tests.ml.simurgh.test_backbone_arace import _build_backbone as arace_backbone
from tests.ml.simurgh.test_elements import _minimal_simurgh

REFERENCE = Path(__file__).parent / "fixtures" / "legacy_reference.pt"
SIMPLIFIED = ("simurgh_legacy_simplified_monolithic", "simurgh_legacy_simplified_routed")
E0 = {1: -13.6, 6: -1028.5, 8: -2041.1}


@pytest.fixture(autouse=True)
def _float64() -> typing.Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def energy(model: torch.nn.Module, b: Batch) -> torch.Tensor:
    return model(b)["energy"].detach()


# ---------------------------------------------------------------------------
# Bit identity: a saved checkpoint reproduces the pre-fix code exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(MODELS))
def test_saved_checkpoint_is_bit_identical_at_default_flags(key: str) -> None:
    reference = torch.load(REFERENCE, weights_only=True)[key]
    buffer = io.BytesIO()
    torch.save(seeded(key).state_dict(), buffer)
    buffer.seek(0)
    model = build(key)
    model.load_state_dict(torch.load(buffer, weights_only=True))
    assert list(model.state_dict()) == reference["keys"]
    out = model(batch())
    assert torch.equal(out["energy"].detach(), reference["energy"])
    assert torch.equal(out["forces"].detach(), reference["forces"])


# ---------------------------------------------------------------------------
# Silent failures that now raise
# ---------------------------------------------------------------------------

NITROGEN = Batch.from_data_list(
    [graph([7, 1, 1, 1], [0, 0, 0, 0], [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])]
)


@pytest.mark.parametrize("key", ["simurgh_legacy_nf", "simurgh_legacy_monolithic_arace"])
def test_undeclared_element_raises(key: str) -> None:
    with pytest.raises(ValueError, match=r"atomic number\(s\) \[7\] are not in elements"):
        build(key)(NITROGEN)


def test_undeclared_element_raises_in_the_other_legacy_backbones() -> None:
    with pytest.raises(ValueError, match="are not in elements"):
        arace_backbone()(Batch.from_data_list([graph([9, 1], [0, 0], [[0, 0, 0], [1.0, 0, 0]])]))
    with pytest.raises(ValueError, match="are not in elements"):
        _minimal_simurgh([1, 6], "learned", None)(NITROGEN)


@pytest.mark.parametrize(
    ("key", "flag"),
    [
        ("simurgh_legacy_simplified_monolithic", {"fragment_line": {"max_fragments": 3}}),
        ("simurgh_legacy_nf", {"virtual_node": {"vnode_max_fragments": 3}}),
    ],
)
def test_max_fragments_overflow_raises(key: str, flag: dict) -> None:
    section, value = next(iter(flag.items()))
    model = build(key, **{section: {**MODELS[key].get(section, {}), **value}})
    with pytest.raises(ValueError, match="exceeds max_fragments=3"):
        model(batch())  # 4 fragments in the batch


# ---------------------------------------------------------------------------
# Opt-in numerical fixes
# ---------------------------------------------------------------------------


def _far_copy(dx: float = 100.0) -> tuple[Batch, Batch]:
    numbers, fragments, positions = STRUCTURES[0]
    single = Batch.from_data_list([graph(numbers, fragments, positions)])
    far = [[x + dx, y, z] for x, y, z in positions]
    doubled = Batch.from_data_list(
        [graph(numbers * 2, fragments + [f + 2 for f in fragments], positions + far)]
    )
    return single, doubled


@pytest.mark.parametrize("fixed", [False, True])
def test_nf_envelope_in_softmax_restores_extensivity(fixed: bool) -> None:
    key = "simurgh_legacy_nf"
    vn = {**MODELS[key]["virtual_node"], "envelope_in_softmax": fixed}
    model = seeded(key, virtual_node=vn)
    single, doubled = _far_copy()
    error = float((energy(model, doubled) - 2 * energy(model, single)).abs())
    if fixed:
        assert error < 1e-10
    else:
        assert error > 1e-4  # the shipped defect: far fragments dilute the attention


def _three_waters(d: float) -> Batch:
    water = [[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.240, 0.927, 0.0]]
    near = [[x, y + 3.0, z] for x, y, z in water]
    crossing = [[x + d, y, z] for x, y, z in water]
    numbers = [8, 1, 1] * 3
    return Batch.from_data_list(
        [graph(numbers, [0, 0, 0, 1, 1, 1, 2, 2, 2], water + near + crossing)]
    )


@pytest.mark.parametrize("smooth", [False, True])
def test_fragment_interaction_smooth_normalisation(smooth: bool) -> None:
    key = "simurgh_legacy_monolithic_arace"
    frag = {**MODELS[key]["fragment_interaction"], "smooth_normalisation": smooth}
    model = seeded(key, fragment_interaction=frag)
    # The first water has two fragment neighbours; the third leaves it at d = cutoff.
    d = frag["cutoff"]
    jump = float(
        (energy(model, _three_waters(d - 1e-9)) - energy(model, _three_waters(d + 1e-9))).abs()
    )
    if smooth:
        assert jump < 1e-9
    else:
        assert jump > 1e-6  # the shipped defect: the neighbour count jumps at the cutoff


@pytest.mark.parametrize("key", SIMPLIFIED)
@pytest.mark.parametrize("fixed", [False, True])
def test_simplified_isolated_atom_reference(key: str, fixed: bool) -> None:
    model = seeded(
        key,
        isolated_atom_reference=fixed,
        atomic_energies={"mode": "provided", "values": E0},
    )
    for z, e0 in E0.items():
        lone = Batch.from_data_list([graph([z], [0], [[0.0, 0.0, 0.0]])])
        error = abs(float(energy(model, lone)) - e0)
        assert (error < 1e-12) if fixed else (error > 1e-3)
    if fixed:  # still trainable end to end
        model.train()
        out = model(batch())
        (out["energy"].sum() + out["forces"].square().sum()).backward()


def test_newton_loss_penalises_the_net_force() -> None:
    loss = LOSS_REGISTRY.get("newton")(loss_fn="mse")
    target = Batch.from_data_list(
        [graph([1, 1], [0, 0], [[0, 0, 0], [1.0, 0, 0]]), graph([8], [0], [[0, 0, 0]])]
    )
    forces = torch.tensor([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    # net forces (2, 2, 0) and (0, 0, 3): mse over the 6 components = (4 + 4 + 9) / 6
    assert float(loss({"forces": forces}, target)) == pytest.approx(17 / 6)


def test_newton_stage_weight_matches_a_declared_loss() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "training": {
                "losses": [{"name": "forces", "weight": 1.0}, {"name": "newton", "weight": 0.01}]
            }
        }
    )
    labels = [wl.label for wl in build_loss(cfg).losses]
    assert labels == ["forces", "newton"]
