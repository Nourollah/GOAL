"""Tests for the Registry system."""

from __future__ import annotations

import pytest

from goal.ml.registry import Registry


class TestRegistry:
    """Verify all three registration mechanisms and error handling."""

    def test_decorator_registration(self):
        """Decorator @register should store and retrieve a class."""
        reg = Registry("test")

        @reg.register("foo")
        class Foo:
            pass

        assert reg.get("foo") is Foo

    def test_lazy_registration(self):
        """register_lazy should resolve only on first get()."""
        reg = Registry("test")
        reg.register_lazy("os_path", "os.path:join")

        # Not yet imported
        assert "os_path" not in reg._registry
        assert "os_path" in reg._lazy

        # Now resolve
        cls = reg.get("os_path")
        import os.path

        assert cls is os.path.join

    def test_register_instance(self):
        """register_instance should store a class directly."""
        reg = Registry("test")

        class Bar:
            pass

        reg.register_instance("bar", Bar)
        assert reg.get("bar") is Bar

    def test_build(self):
        """build() should instantiate the class with kwargs."""
        reg = Registry("test")

        @reg.register("mydict")
        class MyDict(dict):
            pass

        obj = reg.build("mydict", a=1)
        assert isinstance(obj, MyDict)

    def test_duplicate_registration_raises(self):
        """Registering the same name twice should raise KeyError."""
        reg = Registry("test")

        @reg.register("dup")
        class A:
            pass

        with pytest.raises(KeyError, match="already registered"):

            @reg.register("dup")
            class B:
                pass

    def test_unknown_key_raises(self):
        """Getting an unregistered name should raise KeyError with suggestions."""
        reg = Registry("test")
        reg.register_lazy("known", "os.path:join")

        with pytest.raises(KeyError, match="not found"):
            reg.get("unknown")

    def test_list_available(self):
        """list_available should return sorted union of eager and lazy names."""
        reg = Registry("test")

        @reg.register("b")
        class B:
            pass

        reg.register_lazy("a", "os.path:join")

        assert reg.list_available() == ["a", "b"]

    def test_contains(self):
        """__contains__ should check both registries."""
        reg = Registry("test")
        reg.register_lazy("lazy_one", "os.path:join")

        @reg.register("eager_one")
        class X:
            pass

        assert "lazy_one" in reg
        assert "eager_one" in reg
        assert "missing" not in reg

    def test_bad_lazy_import_raises(self):
        """Lazy import of a non-existent module should raise ImportError."""
        reg = Registry("test")
        reg.register_lazy("bad", "nonexistent.module:Class")

        with pytest.raises(ImportError, match="Failed to import"):
            reg.get("bad")
