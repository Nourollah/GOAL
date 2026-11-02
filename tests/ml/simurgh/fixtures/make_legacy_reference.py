"""Write ``legacy_reference.pt``: outputs of small, seeded legacy models.

Run once with the code from before the legacy fixes (commit 9c8d722):

    PYTHONPATH=<checkout at 9c8d722>/src python tests/ml/simurgh/fixtures/make_legacy_reference.py

The weights are not stored: :func:`seeded` rebuilds them exactly (seeded
construction, then seeded values for the zero-initialised readouts), and the
file keeps the state-dict keys to prove it.  ``test_legacy_fixes.py`` saves
and reloads that checkpoint with the current code and requires bit-identical
energies and forces with every new flag at its default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

import goal.ml.nn.models  # noqa: F401  (registers the models)
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import MODEL_REGISTRY

CUTOFF = 5.0
ARTISAN = {
    "architecture": "equivariant",
    "hidden_irreps": "4x0e + 4x1o + 4x2e",
    "num_layers": 1,
    "num_rbf": 4,
    "radial_hidden": 8,
    "n_scalar_out": 4,
    "final_hidden": 4,
    "element_conditioned": False,
}
ARACE = {
    "elements": (1, 6, 8),
    "num_rounds": 2,
    "cutoff": CUTOFF,
    "embedding_dim": 8,
    "num_elements": 9,
    "avg_num_neighbors": 4.0,
    "artisan": ARTISAN,
}
SIMPLIFIED = {
    "cutoff": CUTOFF,
    "embedding_dim": 4,
    "num_elements": 9,
    "num_rbf": 4,
    "radial_hidden": 8,
    "atom_line": {"hidden_irreps": "4x0e + 4x1o + 4x2e", "num_layers": 3},
    "pair_line": {"hidden_irreps": "4x0e + 4x1e + 4x2e", "num_layers": 2},
    "fragment_line": {"hidden_irreps": "4x0e + 4x1o + 4x2e", "num_layers": 2},
    "readout": {"n_scalar": 4, "hidden": 4},
}
MODELS = {
    "simurgh_legacy_nf": {
        **ARACE,
        "virtual_node": {"n_scalar": 4, "vnode_num_rbf": 4, "n_heads": 2, "final_hidden": 4},
    },
    "simurgh_legacy_monolithic_arace": {
        **ARACE,
        "fragment_interaction": {"num_rbf": 4, "radial_hidden": 8, "cutoff": 4.0},
    },
    "simurgh_legacy_simplified_monolithic": SIMPLIFIED,
    "simurgh_legacy_simplified_routed": {
        **SIMPLIFIED,
        "artisan": {
            "rare_artisan": {
                "enabled": True,
                "min_element_frequency": 0.15,
                "min_pair_frequency": 0.05,
                "type_embed_dim": 4,
            },
            "pair_counts": {"1_1": 400, "1_6": 300, "6_6": 100, "1_8": 150, "6_8": 40, "8_8": 10},
        },
        "fragment_experts": {"num_experts": 3, "top_k": 2, "balance_loss_weight": 0.01},
    },
}

# Methanol and water (two fragments), and a lone water 3.5 Å from an ethane.
STRUCTURES = [
    (
        [6, 8, 1, 1, 1, 1, 8, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [
            [0.000, 0.000, 0.000],
            [1.410, 0.050, -0.020],
            [-0.360, 1.020, 0.110],
            [-0.390, -0.520, 0.880],
            [-0.330, -0.480, -0.930],
            [1.720, -0.840, 0.150],
            [3.900, 0.400, 0.300],
            [4.300, 1.250, 0.500],
            [4.450, -0.200, 0.850],
        ],
    ),
    (
        [6, 6, 1, 1, 1, 1, 1, 1, 8, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [
            [0.000, 0.000, 0.000],
            [1.530, 0.000, 0.000],
            [-0.390, 1.020, 0.050],
            [-0.380, -0.520, 0.880],
            [-0.390, -0.480, -0.900],
            [1.920, -1.010, 0.060],
            [1.910, 0.500, 0.890],
            [1.920, 0.540, -0.880],
            [0.700, 3.600, 0.200],
            [1.400, 4.200, 0.000],
            [-0.050, 4.150, 0.400],
        ],
    ),
]


def graph(numbers: list[int], fragments: list[int], positions: list[list[float]]) -> AtomicGraph:
    pos = torch.tensor(positions, dtype=torch.float64)
    dist = torch.cdist(pos, pos)
    rows, cols = torch.nonzero((dist < CUTOFF) & (dist > 0), as_tuple=True)
    vectors = pos[cols] - pos[rows]
    return AtomicGraph(
        positions=pos,
        atomic_numbers=torch.tensor(numbers),
        cell=torch.zeros(3, 3),
        pbc=torch.zeros(3, dtype=torch.bool),
        edge_index=torch.stack([rows, cols]),
        edge_vectors=vectors,
        edge_lengths=vectors.norm(dim=-1),
        fragment_index=torch.tensor(fragments),
    )


def batch() -> Batch:
    return Batch.from_data_list([graph(*s) for s in STRUCTURES])


def build(key: str, **flags: object) -> torch.nn.Module:
    kwargs = dict(MODELS[key])
    if flags:
        kwargs.update(flags)
    return MODEL_REGISTRY.get(key)(**kwargs).double().eval()


def seeded(key: str, **flags: object) -> torch.nn.Module:
    """The reference model, with every branch contributing."""
    torch.manual_seed(0)
    model = build(key, **flags)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # wake the zero-initialised readouts
        for p in model.parameters():
            if p.numel() and not p.any():
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
    return model


def main(out: Path) -> None:
    torch.set_default_dtype(torch.float64)
    reference: dict[str, dict[str, object]] = {}
    for key in MODELS:
        model = seeded(key)
        result = model(batch())
        reference[key] = {
            "keys": list(model.state_dict()),
            "energy": result["energy"].detach(),
            "forces": result["forces"].detach(),
        }
    torch.save(reference, out)


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("legacy_reference.pt")
    )
