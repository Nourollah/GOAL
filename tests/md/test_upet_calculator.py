"""Unit tests for the UPET MD calculator builder (mocked — no upet package)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from goal.md.calculators.base import GOALCalculatorNotFound
from goal.md.core import calculator_factory as cf


def test_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "HAS_UPET", False)
    with pytest.raises(GOALCalculatorNotFound, match="pip install upet"):
        cf.CalculatorFactory.create("upet", model="pet-mad-s")


def test_invalid_model_lists_families(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "HAS_UPET", True)
    monkeypatch.setattr(cf, "UPETCalculator", MagicMock())
    with pytest.raises(ValueError) as exc:
        cf.CalculatorFactory.create("upet", model="totally-bogus")
    msg = str(exc.value)
    for fam in ("pet-mad", "pet-oam", "pet-omat", "pet-spice"):
        assert fam in msg


def test_valid_model_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    upet = MagicMock()
    monkeypatch.setattr(cf, "HAS_UPET", True)
    monkeypatch.setattr(cf, "UPETCalculator", upet)
    cf.CalculatorFactory.create("upet", model="pet-omat-l", device="cpu")
    assert upet.call_args.kwargs["model"] == "pet-omat-l"


@pytest.mark.parametrize("model", ["pet-omad-l", "pet-omatpes-l", "pet-spice-s"])
def test_released_families_accepted(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    """Families outside the old pet-{mad,oam,omat,spice} guess must not be rejected."""
    monkeypatch.setattr(cf, "HAS_UPET", True)
    monkeypatch.setattr(cf, "UPETCalculator", MagicMock())
    cf.CalculatorFactory.create("upet", model=model, device="cpu")


def test_unreleased_size_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The released set is not family x size — pet-mad-xl does not exist."""
    monkeypatch.setattr(cf, "HAS_UPET", True)
    monkeypatch.setattr(cf, "UPETCalculator", MagicMock())
    with pytest.raises(ValueError, match="Unknown UPET model"):
        cf.CalculatorFactory.create("upet", model="pet-mad-xl", device="cpu")


def test_variants_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: {"energy": "energy"} asks upet for a head named `energy/energy`.

    upet >= 0.2 raises "variant 'energy' for output 'energy' not found in outputs",
    which made every stock model unloadable through the factory.
    """
    upet = MagicMock()
    monkeypatch.setattr(cf, "HAS_UPET", True)
    monkeypatch.setattr(cf, "UPETCalculator", upet)
    cf.CalculatorFactory.create("upet", model="pet-mad-s", device="cpu")
    assert upet.call_args.kwargs["variants"] is None


def test_explicit_variant_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    upet = MagicMock()
    monkeypatch.setattr(cf, "HAS_UPET", True)
    monkeypatch.setattr(cf, "UPETCalculator", upet)
    cf.CalculatorFactory.create(
        "upet",
        checkpoint_path="runs/upet_finetuned/model.ckpt",
        variants={"energy": "finetune"},
        device="cpu",
    )
    assert upet.call_args.kwargs["variants"] == {"energy": "finetune"}


def test_version_only_forwarded_when_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting `version` must leave upet's own 'latest' default in charge."""
    upet = MagicMock()
    monkeypatch.setattr(cf, "HAS_UPET", True)
    monkeypatch.setattr(cf, "UPETCalculator", upet)
    cf.CalculatorFactory.create("upet", model="pet-mad-s", device="cpu")
    assert "version" not in upet.call_args.kwargs
    cf.CalculatorFactory.create("upet", model="pet-mad-s", version="1.5.0", device="cpu")
    assert upet.call_args.kwargs["version"] == "1.5.0"


def test_non_conservative_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "HAS_UPET", True)
    monkeypatch.setattr(cf, "UPETCalculator", MagicMock())
    with pytest.warns(UserWarning, match="ENERGY CONSERVATION"):
        cf.CalculatorFactory.create("upet", model="pet-mad-s", non_conservative=True)
