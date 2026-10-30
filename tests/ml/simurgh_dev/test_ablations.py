"""Literature ablation models: zero-init equivalence and the physical invariants."""

from __future__ import annotations

import copy

import pytest
import torch
from ase.build import molecule

import goal.ml.nn.models  # noqa: F401  (registers the ablation models)
from goal.ml.registry import MODEL_REGISTRY
from tests.ml.simurgh_dev.test_monolithic import (
    SMALL,
    _graph,
    awake,
    build,
    energy,
    predict_fn,
)
from tests.ml.simurgh_dev.utils import (
    far_copy_extensivity,
    finite_difference_force_error,
    symmetry_errors,
)

ABLATIONS = [
    "simurgh_dev_abl_quadrangle_term",
    "simurgh_dev_abl_coupled_product",
    "simurgh_dev_abl_physical_long_range",
]


def ablation(name: str, seed: int = 0, **overrides):
    torch.manual_seed(seed)
    return MODEL_REGISTRY.get(name)(**{**copy.deepcopy(SMALL), **overrides}).eval()


@pytest.mark.parametrize("name", ABLATIONS)
def test_fresh_ablation_equals_simurgh(name: str) -> None:
    ethanol = molecule("CH3CH2OH")
    assert energy(ablation(name), ethanol) == energy(build(), ethanol)


@pytest.mark.parametrize("name", ABLATIONS)
def test_ablation_keeps_the_invariants(name: str) -> None:
    p = predict_fn(awake(ablation(name), gain=0.1))
    ethanol = molecule("CH3CH2OH")
    assert abs(far_copy_extensivity(p, ethanol)) < 1e-5
    assert max(symmetry_errors(p, ethanol).values()) < 1e-9
    assert finite_difference_force_error(p, ethanol) < 1e-6


@pytest.mark.parametrize("name", ABLATIONS)
def test_every_parameter_gets_a_gradient(name: str) -> None:
    model = awake(ablation(name), gain=0.1).train()
    out = model(_graph(molecule("CH3CH2OH")))
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_coupled_product_reports_its_parameter_cost() -> None:
    model = ablation("simurgh_dev_abl_coupled_product")
    assert model.extra_parameters > 0


def test_physical_long_range_refuses_stacking_with_d3() -> None:
    with pytest.raises(ValueError, match="replaces the D3 term"):
        ablation(
            "simurgh_dev_abl_physical_long_range",
            dispersion={"enabled": True, "backend": "precomputed"},
        )


def test_physical_long_range_dispersion_is_attractive_and_smooth() -> None:
    model = ablation("simurgh_dev_abl_physical_long_range")
    with torch.no_grad():
        model.dispersion_amplitude.fill_(1.0)
    p = predict_fn(model)
    from ase import Atoms

    def e(d: float) -> float:
        return p(Atoms("CC", positions=[[0, 0, 0], [d, 0, 0]]))[0]

    far = e(30.0)
    assert e(4.0) < e(8.0) < far  # attractive, decaying
    assert abs(e(5.0 - 1e-7) - e(5.0 + 1e-7)) < 1e-9  # no cutoff in the dispersion
