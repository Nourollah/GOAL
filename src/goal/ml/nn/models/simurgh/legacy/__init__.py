"""Legacy SIMURGH models, moved here unchanged.

The ``simurgh`` name belongs to the new model (``simurgh_dev`` while it
is under development, in :mod:`goal.ml.nn.models.simurgh.dev`).  The
older variants keep their classes and behaviour under ``simurgh_legacy*``
registry keys; their previous keys resolve through deprecated aliases
(:data:`DEPRECATED_KEYS`) until release.

Original description:

**Symmetry-aware Interatomic Model with element-pair Routed Gated
artisans (an element-pair mixture of potential artisans).**

Variants exposed:

* :class:`backbone_arace.SimurghAraceBackbone` — **default**
    The ARACE architecture (ARtisan + Atomic Cluster Expansion):
    element-pair potential artisans are the primary computation at
    every layer, alternating with ACE message passing that aggregates
    the artisan edge features back onto nodes.  Implements the
    ``EquivariantBackbone`` protocol; combine with any standard
    ``TaskHead`` (e.g. ``energy_forces``).  Registered
    ``"simurgh_legacy_arace"``.

* :class:`backbone.SimurghBackbone` — legacy
    The original ACE-dressing-first pipeline (environment dressing →
    artisan bank as final readout).  Registered ``"simurgh_legacy"`` and
    ``"simurgh_legacy_ace_first"``.

* :class:`monolithic.SimurghMonolithic`
    Self-contained ``MonolithicModel`` counterpart of the legacy
    backbone (``"simurgh_legacy_monolithic"``).

* :class:`arace.MonolithicArace`
    Self-contained ``MonolithicModel`` counterpart of the ARACE
    backbone, for research/ablation (``"simurgh_legacy_monolithic_arace"``).

The modular and monolithic variants of each architecture share the same
building blocks so the physics is identical; the difference is only in
how they integrate with the GOAL Lightning training loop.
"""

from __future__ import annotations

from goal.ml.nn.models.simurgh.legacy.arace import MonolithicArace
from goal.ml.nn.models.simurgh.legacy.backbone import SimurghBackbone
from goal.ml.nn.models.simurgh.legacy.backbone_arace import SimurghAraceBackbone
from goal.ml.nn.models.simurgh.legacy.monolithic import SimurghMonolithic
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

#: Registry key before the legacy move -> current key.  Removed at release,
#: when ``simurgh`` is reassigned to the new model.
DEPRECATED_KEYS: dict[str, str] = {
    "simurgh": "simurgh_legacy",
    "simurgh_ace_first": "simurgh_legacy_ace_first",
    "simurgh_arace": "simurgh_legacy_arace",
    "monolithic_arace": "simurgh_legacy_monolithic_arace",
    "simurgh_monolithic": "simurgh_legacy_monolithic",
    "nf_simurgh": "simurgh_legacy_nf",
    "simurgh_simplified_monolithic": "simurgh_legacy_simplified_monolithic",
    "simurgh_simplified_routed": "simurgh_legacy_simplified_routed",
}

for _old, _new in DEPRECATED_KEYS.items():
    MODEL_REGISTRY.register_alias(_old, _new)
    BACKBONE_REGISTRY.register_alias(_old, _new)

__all__ = [
    "DEPRECATED_KEYS",
    "MonolithicArace",
    "SimurghAraceBackbone",
    "SimurghBackbone",
    "SimurghMonolithic",
]
