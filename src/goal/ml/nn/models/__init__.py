"""Full model architectures.

Drop a new ``.py`` file into this folder with the registration
decorators on its class (``@MODEL_REGISTRY.register("name")`` and, for
backbones, also ``@BACKBONE_REGISTRY.register("name")``) and it is
discovered automatically — no edit to this file required.

The project ships four reference models:

* ``invariant_gnn``     — SchNet-style **invariant** backbone
                          (educational baseline).
* ``hyperspec``         — Native equivariant **modular** backbone using
                          e3nn primitives (educational baseline).
* ``monolithic_example``— Minimal **monolithic** model returning the
                          property dict directly (educational baseline).
* ``simurgh_legacy*``   — The older **native** SIMURGH models
                          (equivariant force fields with an element-pair
                          bank of potential artisans).  See
                          :mod:`goal.ml.nn.models.simurgh.legacy`.
"""

from __future__ import annotations

from goal.ml.registry import auto_discover

auto_discover(__name__)
