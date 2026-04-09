"""GOAL benchmark dataset examples.

These are **not** part of the core ``goal.ml`` package.  Import this module
to register benchmark datasets into the :data:`DATASET_REGISTRY`::

    import examples.datasets  # triggers lazy registration

    from goal.ml.registry import DATASET_REGISTRY
    dataset = DATASET_REGISTRY.build('md17', root='data/md17', molecule='aspirin')

Or use via Hydra config::

    goal-train data=md17_aspirin
"""

from __future__ import annotations

from goal.ml.registry import DATASET_REGISTRY

# Lazy registration — no imports until actually requested
DATASET_REGISTRY.register_lazy("md17", "examples.datasets.md17:MD17Dataset")
DATASET_REGISTRY.register_lazy("rmd17", "examples.datasets.md17:MD17Dataset")
DATASET_REGISTRY.register_lazy("ani1", "examples.datasets.ani1:ANI1Dataset")
DATASET_REGISTRY.register_lazy("ani1x", "examples.datasets.ani1:ANI1Dataset")
DATASET_REGISTRY.register_lazy("qm9", "examples.datasets.qm9:QM9Dataset")
DATASET_REGISTRY.register_lazy("spice", "examples.datasets.spice:SPICEDataset")
