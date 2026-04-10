"""Multi-head compositor — run multiple heads on the same backbone features.

Allows a single backbone to predict several properties simultaneously
(e.g. energy + forces + HOMO + LUMO + dipole) by composing multiple
task heads and merging their output dictionaries.

Each sub-head receives the same ``NodeFeatures`` from the backbone and
independently produces its output keys.  ``MultiHead`` merges them into
a single flat dict, which the ``CompositeLoss`` system consumes as usual.

Configuration example (Hydra YAML)::

    model:
      head:
        name: multi
        heads:
          - name: energy_forces
            irreps_in: "128x0e"
            hidden_dim: 64
          - name: scalar
            irreps_in: "128x0e"
            hidden_dim: 64
            property_name: homo
            reduction: mean
          - name: scalar
            irreps_in: "128x0e"
            hidden_dim: 64
            property_name: lumo
            reduction: mean
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("multi")
class MultiHead(nn.Module):
    """Compose multiple task heads into a single head.

    Each sub-head operates on the same backbone features and contributes
    its output keys to a merged dictionary.  Duplicate keys raise an
    error at construction time (except ``num_atoms``, which is shared).

    Parameters
    ----------
    heads : list[dict[str, Any]]
        List of head configurations.  Each dict must have a ``name`` key
        (matching a ``HEAD_REGISTRY`` entry) and any keyword arguments
        for that head's constructor.  Example::

            [
                {"name": "energy_forces", "irreps_in": "128x0e", "hidden_dim": 64},
                {"name": "scalar", "irreps_in": "128x0e", "property_name": "homo"},
            ]
    """

    def __init__(self, heads: list[dict[str, typing.Any]]) -> None:
        super().__init__()
        self._sub_heads: nn.ModuleList = nn.ModuleList()
        self._output_keys: list[str] = []
        seen_keys: set[str] = set()

        for head_cfg in heads:
            cfg: dict[str, typing.Any] = dict(head_cfg)  # shallow copy
            name: str = cfg.pop("name")
            head_cls: type[nn.Module] = HEAD_REGISTRY.get(name)
            head: nn.Module = head_cls(**cfg)
            self._sub_heads.append(head)

            # Collect and validate output keys
            for key in head.output_keys:
                if key in seen_keys and key != "num_atoms":
                    raise ValueError(
                        f"Duplicate output key '{key}' across sub-heads. "
                        f"Each property must be predicted by exactly one head. "
                        f"If two heads both output 'energy', use a composite "
                        f"head instead."
                    )
                seen_keys.add(key)
                if key not in self._output_keys:
                    self._output_keys.append(key)

    @property
    def output_keys(self) -> list[str]:
        """All output keys across all sub-heads."""
        return self._output_keys

    def forward(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> dict[str, torch.Tensor]:
        """Run all sub-heads and merge outputs.

        Parameters
        ----------
        features : NodeFeatures
            Output of the backbone.
        graph : AtomicGraph
            The input graph.

        Returns
        -------
        dict[str, Tensor]
            Merged dictionary from all sub-heads.
        """
        merged: dict[str, torch.Tensor] = {}
        for head in self._sub_heads:
            outputs: dict[str, torch.Tensor] = head(features, graph)
            merged.update(outputs)
        return merged
