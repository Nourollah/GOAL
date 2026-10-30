"""Central registry system for extensible component discovery.

The registry is the backbone of GOAL's extensibility. It supports four
registration mechanisms:

1. **Decorator registration** — for built-in classes within GOAL itself.
2. **Lazy registration** — zero import cost; loads only when requested.
3. **Entry-point plugin discovery** — third-party packages declare
   entry points in their ``pyproject.toml``.
4. **Package auto-discovery** (see :func:`auto_discover`) — walk a
   package and import every submodule so the ``@register`` decorators
   inside them fire automatically.  Drop a new module in the folder and
   it is registered without touching any ``__init__.py``.
"""

from __future__ import annotations

import importlib
import pkgutil
import typing
import warnings
from importlib.metadata import entry_points

T = typing.TypeVar("T")


class Registry:
    """Lazy registry supporting decorator, explicit, and entry-point registration.

    Never imports modules until the registered class is actually requested.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._registry: dict[str, type] = {}
        self._lazy: dict[str, str] = {}  # name → 'module.path:ClassName'
        self._aliases: dict[str, str] = {}  # deprecated name → current name

    # ------------------------------------------------------------------
    # Registration mechanisms
    # ------------------------------------------------------------------

    def register(self, name: str) -> typing.Callable[[type[T]], type[T]]:
        """Decorator for explicit registration at import time.

        Use this for built-in classes within the GOAL package itself.
        Third parties use this in their own modules — triggered by their
        import, not GOAL's import. No hidden side effects in GOAL's own
        ``__init__`` files.

        Usage::

            @MODEL_REGISTRY.register('my-model')
            class MyModel(nn.Module): ...
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise KeyError(
                    f"'{name}' already registered in {self.name} registry. "
                    f"Existing: {type(self._registry[name])}"
                )
            self._registry[name] = cls
            return cls

        return decorator

    def register_lazy(self, name: str, import_path: str) -> None:
        """Register a dotted import path without importing it.

        Format: ``'goal.ml.nn.models.hyperspec:HyperSpecModel'``

        Use this for all built-in registrations in GOAL — zero import cost.

        Usage::

            MODEL_REGISTRY.register_lazy(
                'hyperspec', 'goal.ml.nn.models.hyperspec:HyperSpecModel'
            )
        """
        self._lazy[name] = import_path

    def register_instance(self, name: str, cls: type) -> None:
        """Explicit programmatic registration without decorator.

        Use when you have the class object directly.
        """
        self._registry[name] = cls

    def register_alias(self, old: str, new: str) -> None:
        """Keep a renamed entry reachable under its old name.

        ``get(old)`` returns the class registered as ``new`` and emits a
        ``DeprecationWarning``.
        """
        self._aliases[old] = new

    def canonical(self, name: str) -> str:
        """Current name for ``name``: the alias target, or ``name`` itself."""
        return self._aliases.get(name, name)

    # ------------------------------------------------------------------
    # Lookup and instantiation
    # ------------------------------------------------------------------

    def get(self, name: str) -> type:
        """Retrieve a registered class by name, resolving lazy imports as needed."""
        if name in self._aliases:
            warnings.warn(
                f"'{name}' is a deprecated {self.name} registry name; "
                f"use '{self._aliases[name]}'.",
                DeprecationWarning,
                stacklevel=2,
            )
            name = self._aliases[name]
        if name not in self._registry:
            if name in self._lazy:
                module_path: str
                cls_name: str
                module_path, cls_name = self._lazy[name].rsplit(":", 1)
                try:
                    module = importlib.import_module(module_path)
                except ImportError as e:
                    raise ImportError(
                        f"Failed to import '{name}' from {module_path}. "
                        f"You may need to install optional dependencies. "
                        f"Original error: {e}"
                    ) from e
                self._registry[name] = getattr(module, cls_name)
            else:
                available: list[str] = sorted(set(self._registry) | set(self._lazy))
                raise KeyError(
                    f"'{name}' not found in {self.name} registry.\n"
                    f"Available: {available}\n"
                    f"Register with: {self.name}.register_lazy("
                    f"'{name}', 'your.module:YourClass')"
                )
        return self._registry[name]

    def build(self, name: str, **kwargs: typing.Any) -> typing.Any:
        """Get class and instantiate in one call."""
        return self.get(name)(**kwargs)

    # ------------------------------------------------------------------
    # Plugin discovery
    # ------------------------------------------------------------------

    def discover_plugins(self, group: str) -> None:
        """Discover and load entry-point plugins.

        Call once at startup::

            MODEL_REGISTRY.discover_plugins('goal.ml.models')

        Third parties declare in their ``pyproject.toml``::

            [project.entry-points.'goal.ml.models']
            my_model = 'my_package.module:MyModel'
        """
        for ep in entry_points(group=group):
            try:
                cls: type = ep.load()
                if ep.name not in self._registry:
                    self._registry[ep.name] = cls
            except Exception as e:
                warnings.warn(
                    f"Failed to load plugin '{ep.name}': {e}",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._registry or name in self._lazy or name in self._aliases

    def list_available(self) -> list[str]:
        """Return sorted list of all registered and lazily-registered names."""
        return sorted(set(self._registry) | set(self._lazy))

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, available={self.list_available()})"


# ---------------------------------------------------------------------------
# Global registries — import these everywhere
# ---------------------------------------------------------------------------

MODEL_REGISTRY = Registry("models")
BACKBONE_REGISTRY = Registry("backbones")
LOSS_REGISTRY = Registry("losses")
DATASET_REGISTRY = Registry("datasets")
HEAD_REGISTRY = Registry("heads")
TRANSFORM_REGISTRY = Registry("transforms")
STRATEGY_REGISTRY = Registry("strategies")


# ---------------------------------------------------------------------------
# Package auto-discovery
# ---------------------------------------------------------------------------


def auto_discover(package_name: str) -> None:
    """Import every submodule of ``package_name`` so decorator-based
    ``@<REGISTRY>.register("name")`` calls inside them fire automatically.

    Use this in a package's ``__init__.py`` instead of listing every
    submodule by hand.  A new file dropped into the folder is picked up
    on the next import — no ``__init__.py`` change, no double source of
    truth, no "I forgot to register the BACKBONE alongside the MODEL"
    bug.  All registries declared in this module receive whichever
    decorators fire inside the discovered modules.

    Parameters
    ----------
    package_name : str
        Dotted name of a package, e.g. ``"goal.ml.nn.models"``.  Must
        actually be a package (have ``__path__``); a plain module
        raises ``ValueError``.

    Notes
    -----
    * Modules that fail to import emit a ``UserWarning`` and are
      skipped — the rest of discovery still runs.  This matches the
      behaviour of :meth:`Registry.discover_plugins` for entry points.
    * Safe to call from within the target package's own ``__init__.py``
      (``walk_packages`` only enumerates submodules, not the package
      itself, so no recursion).
    """
    package = importlib.import_module(package_name)
    if not hasattr(package, "__path__"):
        raise ValueError(
            f"auto_discover('{package_name}') requires a package, " f"not a plain module."
        )
    for _finder, modname, _ispkg in pkgutil.walk_packages(
        package.__path__, prefix=f"{package_name}."
    ):
        try:
            importlib.import_module(modname)
        except ImportError as e:
            warnings.warn(
                f"auto_discover skipped '{modname}': {e}",
                stacklevel=2,
            )
