"""Hardware detection and the NVIDIA fast-path gates (goal.ml.utils.hardware)."""

from __future__ import annotations

import builtins
import importlib.util
import logging
import types

import pytest
import torch

from goal.ml.utils import hardware


@pytest.fixture(autouse=True)
def _fresh_cache():
    hardware.clear_cache()
    yield
    hardware.clear_cache()


def _fake_backend(monkeypatch, cuda: bool, hip: str | None = None, mps: bool = False) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(
        torch.backends, "mps", types.SimpleNamespace(is_available=lambda: mps), raising=False
    )


class TestDetectComputeBackend:
    def test_returns_a_valid_literal(self) -> None:
        assert hardware.detect_compute_backend() in ("nvidia", "amd", "mps", "cpu")

    def test_cpu(self, monkeypatch) -> None:
        _fake_backend(monkeypatch, cuda=False)
        assert hardware.detect_compute_backend() == "cpu"

    def test_nvidia(self, monkeypatch) -> None:
        _fake_backend(monkeypatch, cuda=True, hip=None)
        assert hardware.detect_compute_backend() == "nvidia"

    def test_amd_rocm_is_not_nvidia(self, monkeypatch) -> None:
        _fake_backend(monkeypatch, cuda=True, hip="6.2")
        assert hardware.detect_compute_backend() == "amd"

    def test_apple_mps(self, monkeypatch) -> None:
        _fake_backend(monkeypatch, cuda=False, mps=True)
        assert hardware.detect_compute_backend() == "mps"

    def test_clear_cache_redetects(self, monkeypatch) -> None:
        _fake_backend(monkeypatch, cuda=False)
        assert hardware.detect_compute_backend() == "cpu"
        _fake_backend(monkeypatch, cuda=True)
        assert hardware.detect_compute_backend() == "cpu"  # cached
        hardware.clear_cache()
        assert hardware.detect_compute_backend() == "nvidia"


def _block_import(monkeypatch, name: str) -> None:
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda mod, *a, **k: None if mod.split(".")[0] == name else real_find_spec(mod, *a, **k),
    )


class TestNvalchemiGates:
    def test_nvalchemi_unavailable_when_package_missing(self, monkeypatch) -> None:
        _block_import(monkeypatch, "nvalchemi")
        _fake_backend(monkeypatch, cuda=True)
        assert hardware.nvalchemi_available() is False

    def test_nvalchemiops_unavailable_when_package_missing(self, monkeypatch) -> None:
        _block_import(monkeypatch, "nvalchemiops")
        _fake_backend(monkeypatch, cuda=True)
        assert hardware.nvalchemiops_available() is False
        assert hardware.nvalchemi_ops_available is hardware.nvalchemiops_available

    def test_ops_unavailable_on_amd_even_if_installed(self, monkeypatch) -> None:
        if importlib.util.find_spec("nvalchemiops") is None:
            pytest.skip("nvalchemi-toolkit-ops not installed")
        _fake_backend(monkeypatch, cuda=True, hip="6.2")
        assert hardware.nvalchemiops_available() is False

    def test_broken_install_is_unavailable_not_an_error(self, monkeypatch) -> None:
        monkeypatch.setattr(importlib.util, "find_spec", lambda mod, *a, **k: object())
        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name.startswith("nvalchemi"):
                raise RuntimeError("broken CUDA build")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)
        assert hardware.nvalchemi_available() is False


class TestResolveBackend:
    def _resolve(self, requested: str, available: bool) -> str:
        return hardware.resolve_backend(
            requested,
            fast="nvalchemi",
            fallback="ase",
            available=lambda: available,
            feature="test",
        )

    def test_auto(self) -> None:
        assert self._resolve("auto", True) == "nvalchemi"
        assert self._resolve("auto", False) == "ase"

    def test_explicit_fast_path_degrades_with_warning(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="goal.ml.utils.hardware"):
            assert self._resolve("nvalchemi", False) == "ase"
        assert "falling back" in caplog.text

    def test_other_names_pass_through(self) -> None:
        assert self._resolve("ase", True) == "ase"


def test_importable_without_any_extras() -> None:
    from goal.ml.utils.hardware import (  # noqa: F401
        detect_compute_backend,
        nvalchemi_available,
    )
