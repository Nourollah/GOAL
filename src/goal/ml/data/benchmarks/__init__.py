"""GOAL benchmark dataset adapters.

Importing ``goal.ml.data.datasets`` (which every CLI does) registers these
benchmark datasets lazily in the ``DATASET_REGISTRY``::

    import goal.ml.data.benchmarks  # triggers lazy registration

    from goal.ml.registry import DATASET_REGISTRY
    dataset = DATASET_REGISTRY.build('md17', root='data/md17', molecule='aspirin')

Or use via Hydra config::

    goal-train --config-name nf_simurgh_md17
"""

from __future__ import annotations

from goal.ml.registry import DATASET_REGISTRY

# Lazy registration — no imports until actually requested
DATASET_REGISTRY.register_lazy("md17", "goal.ml.data.benchmarks.md17:MD17Dataset")
DATASET_REGISTRY.register_lazy("rmd17", "goal.ml.data.benchmarks.md17:MD17Dataset")
DATASET_REGISTRY.register_lazy("md22", "goal.ml.data.benchmarks.md22:MD22Dataset")
DATASET_REGISTRY.register_lazy("ani1", "goal.ml.data.benchmarks.ani1:ANI1Dataset")
DATASET_REGISTRY.register_lazy("ani1x", "goal.ml.data.benchmarks.ani1:ANI1Dataset")
DATASET_REGISTRY.register_lazy("qm9", "goal.ml.data.benchmarks.qm9:QM9Dataset")
DATASET_REGISTRY.register_lazy("spice", "goal.ml.data.benchmarks.spice:SPICEDataset")
