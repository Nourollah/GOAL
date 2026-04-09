"""Tests for the distributed training strategy factory."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf


class TestBuildStrategy:
    """Verify build_strategy dispatches to the correct builder."""

    def test_no_strategy_config_returns_auto(self):
        """Missing cfg.strategy should default to 'auto'."""
        from gmd.training.strategies.factory import build_strategy

        cfg: DictConfig = OmegaConf.create({"model": {}})
        result: typing.Any = build_strategy(cfg)
        assert result == "auto"

    def test_auto_strategy(self):
        """Explicit name='auto' should return 'auto' string."""
        from gmd.training.strategies.factory import build_strategy

        cfg: DictConfig = OmegaConf.create({"strategy": {"name": "auto"}})
        assert build_strategy(cfg) == "auto"

    def test_single_strategy(self):
        """name='single' should return a SingleDeviceStrategy."""
        from gmd.training.strategies.factory import build_strategy

        cfg: DictConfig = OmegaConf.create({"strategy": {"name": "single"}})
        result: typing.Any = build_strategy(cfg)

        from lightning.pytorch.strategies import SingleDeviceStrategy
        assert isinstance(result, SingleDeviceStrategy)

    def test_ddp_strategy(self):
        """name='ddp' should return a DDPStrategy with configured options."""
        from gmd.training.strategies.factory import build_strategy

        cfg: DictConfig = OmegaConf.create({
            "strategy": {
                "name": "ddp",
                "find_unused_parameters": True,
                "static_graph": False,
                "gradient_as_bucket_view": True,
                "bucket_cap_mb": 50,
            }
        })
        result: typing.Any = build_strategy(cfg)

        from lightning.pytorch.strategies import DDPStrategy
        assert isinstance(result, DDPStrategy)

    def test_fsdp_strategy(self):
        """name='fsdp' should return an FSDPStrategy."""
        from gmd.training.strategies.factory import build_strategy

        cfg: DictConfig = OmegaConf.create({
            "strategy": {
                "name": "fsdp",
                "auto_wrap_policy": "size",
                "min_num_params": 100_000,
            }
        })
        result: typing.Any = build_strategy(cfg)

        from lightning.pytorch.strategies import FSDPStrategy
        assert isinstance(result, FSDPStrategy)

    def test_fsdp2_strategy(self):
        """name='fsdp2' should return a ModelParallelStrategy."""
        from gmd.training.strategies.factory import build_strategy

        cfg: DictConfig = OmegaConf.create({
            "strategy": {"name": "fsdp2"}
        })
        result: typing.Any = build_strategy(cfg)

        from lightning.pytorch.strategies import ModelParallelStrategy
        assert isinstance(result, ModelParallelStrategy)

    def test_unknown_strategy_raises(self):
        """Unknown strategy name should raise ValueError."""
        from gmd.training.strategies.factory import build_strategy

        cfg: DictConfig = OmegaConf.create({"strategy": {"name": "nonexistent"}})
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_strategy(cfg)

    def test_deepspeed_missing_raises(self):
        """DeepSpeed strategy without deepspeed package should raise ImportError."""
        from gmd.training.strategies.factory import build_strategy

        cfg: DictConfig = OmegaConf.create({
            "strategy": {"name": "deepspeed_zero2", "stage": 2}
        })

        with patch.dict("sys.modules", {"deepspeed": None}):
            with patch(
                "gmd.training.strategies.deepspeed._require_deepspeed",
                side_effect=ImportError("deepspeed is not installed"),
            ):
                with pytest.raises(ImportError, match="deepspeed"):
                    build_strategy(cfg)


class TestDDPBuilder:
    """Tests for ddp.build_ddp_strategy."""

    def test_defaults(self):
        """Default config should produce a valid DDPStrategy."""
        from gmd.training.strategies.ddp import build_ddp_strategy

        cfg: DictConfig = OmegaConf.create({"name": "ddp"})
        result: typing.Any = build_ddp_strategy(cfg)

        from lightning.pytorch.strategies import DDPStrategy
        assert isinstance(result, DDPStrategy)

    def test_find_unused_parameters(self):
        """find_unused_parameters should be passed to the strategy."""
        from gmd.training.strategies.ddp import build_ddp_strategy

        cfg: DictConfig = OmegaConf.create({
            "name": "ddp",
            "find_unused_parameters": True,
        })
        result: typing.Any = build_ddp_strategy(cfg)
        assert result._ddp_kwargs.get("find_unused_parameters") is True


class TestFSDPBuilder:
    """Tests for fsdp.build_fsdp_strategy."""

    def test_default_fsdp(self):
        """Default FSDP config should produce a valid FSDPStrategy."""
        from gmd.training.strategies.fsdp import build_fsdp_strategy

        cfg: DictConfig = OmegaConf.create({"name": "fsdp"})
        result: typing.Any = build_fsdp_strategy(cfg)

        from lightning.pytorch.strategies import FSDPStrategy
        assert isinstance(result, FSDPStrategy)

    def test_fsdp2_returns_model_parallel(self):
        """build_fsdp2_strategy should return ModelParallelStrategy."""
        from gmd.training.strategies.fsdp import build_fsdp2_strategy

        cfg: DictConfig = OmegaConf.create({"name": "fsdp2"})
        result: typing.Any = build_fsdp2_strategy(cfg)

        from lightning.pytorch.strategies import ModelParallelStrategy
        assert isinstance(result, ModelParallelStrategy)


class TestKnownStrategies:
    """Verify the _KNOWN_STRATEGIES list covers all expected names."""

    def test_known_strategies_list(self):
        """All strategy names should be in the known list."""
        from gmd.training.strategies.factory import _KNOWN_STRATEGIES

        expected: typing.Set[str] = {
            "auto", "single", "ddp", "fsdp", "fsdp2",
            "deepspeed_zero1", "deepspeed_zero2",
            "deepspeed_zero3", "deepspeed_zero3_offload",
        }
        assert set(_KNOWN_STRATEGIES) == expected
