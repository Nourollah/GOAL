"""Every neighbour-list backend must return exactly ASE's edges and shifts.

Covers the pure-PyTorch backend (the portable fallback — PyG's
``radius_graph`` needs ``pyg-lib``/``torch-cluster`` and has no PBC), the
``"auto"`` resolution, the batched tensor-level builder, and the
``goal.ml.nn.primitives.neighbor_list`` front end.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule

from goal.ml.data import neighbor_list as nlmod
from goal.ml.data.neighbor_list import build_neighbor_list, build_neighbor_list_batch
from goal.ml.utils import hardware

CUTOFF = 4.0


def _slab() -> Atoms:
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)
    atoms.pbc = [True, True, False]
    atoms.center(vacuum=5.0, axis=2)
    return atoms


def _unwrapped() -> Atoms:
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions += [7.0, -3.0, 12.0]  # well outside the cell
    return atoms


STRUCTURES = {
    "molecule": lambda: molecule("CH3CH2OH"),
    "fcc_bulk": lambda: bulk("Cu", "fcc", a=3.6) * (2, 2, 2),
    "small_triclinic": lambda: Atoms(
        "NaCl",
        positions=[[0.2, 0, 0], [4.9, 0.1, 0.3]],
        cell=[[5, 0, 0], [1.2, 4.5, 0], [0.5, 0.7, 3.1]],
        pbc=True,
    ),
    "slab": _slab,
    "unwrapped": _unwrapped,
}


def _edges(nl) -> list[tuple[int, ...]]:
    ei = nl.edge_index.cpu().numpy()
    sh = nl.unit_shifts.cpu().numpy()
    return sorted(zip(ei[0].tolist(), ei[1].tolist(), *(sh.T.tolist())))


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_torch_backend_matches_ase(name: str) -> None:
    atoms = STRUCTURES[name]()
    ref = build_neighbor_list(atoms, CUTOFF, backend="ase")
    got = build_neighbor_list(atoms, CUTOFF, backend="torch")
    assert _edges(got) == _edges(ref)
    # vectors agree too (same convention r_j - r_i + S @ cell)
    order_ref = np.lexsort(np.vstack([ref.unit_shifts.T.numpy(), ref.edge_index.numpy()[::-1]]))
    order_got = np.lexsort(np.vstack([got.unit_shifts.T.numpy(), got.edge_index.numpy()[::-1]]))
    np.testing.assert_allclose(
        got.edge_vectors.numpy()[order_got], ref.edge_vectors.numpy()[order_ref], atol=1e-10
    )


def test_three_atom_triangle() -> None:
    atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.8, 0]])
    nl = build_neighbor_list(atoms, 1.5, backend="torch")
    assert _edges(nl) == sorted((i, j, 0, 0, 0) for i in range(3) for j in range(3) if i != j)


def test_radius_graph_backend_survives_without_pyg_lib() -> None:
    nl = build_neighbor_list(molecule("CH4"), CUTOFF, backend="radius_graph")
    assert nl.edge_index.shape[1] == 20


class TestAuto:
    @pytest.fixture(autouse=True)
    def _clear(self):
        nlmod._auto_backend_for.cache_clear()
        hardware.clear_cache()
        yield
        nlmod._auto_backend_for.cache_clear()
        hardware.clear_cache()

    def test_small_structures_never_go_to_the_gpu(self) -> None:
        assert nlmod._auto_backend(10) in ("matscipy", "ase")

    def test_large_structures_fall_back_without_nvalchemiops(self, monkeypatch) -> None:
        monkeypatch.setattr(hardware, "nvalchemiops_available", lambda: False)
        assert nlmod._auto_backend(10_000) in ("matscipy", "ase")

    def test_auto_gives_the_same_edges(self) -> None:
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        assert _edges(build_neighbor_list(atoms, CUTOFF, backend="auto")) == _edges(
            build_neighbor_list(atoms, CUTOFF, backend="ase")
        )


def _batch_inputs():
    atoms_list = [molecule("CH3CH2OH"), bulk("Cu", "fcc", a=3.6) * (2, 2, 2), _slab()]
    pos = torch.tensor(np.concatenate([a.positions for a in atoms_list]))
    batch = torch.cat([torch.full((len(a),), i) for i, a in enumerate(atoms_list)])
    cell = torch.tensor(
        np.stack([np.array(a.cell) if a.pbc.any() else np.eye(3) for a in atoms_list])
    )
    pbc = torch.tensor(np.stack([a.pbc for a in atoms_list]))
    offsets = np.cumsum([0] + [len(a) for a in atoms_list])
    expected = sorted(
        (i + offsets[g], j + offsets[g], *s)
        for g, a in enumerate(atoms_list)
        for i, j, *s in _edges(build_neighbor_list(a, CUTOFF, backend="ase"))
    )
    return pos, batch, cell, pbc, expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["torch", "auto"])
def test_batched_builder_matches_per_structure_ase(device: str, backend: str) -> None:
    pos, batch, cell, pbc, expected = _batch_inputs()
    nl = build_neighbor_list_batch(
        pos.to(device), CUTOFF, batch.to(device), cell.to(device), pbc.to(device), backend=backend
    )
    assert nl.edge_index.device.type == device
    assert _edges(nl) == expected
    assert bool((nl.edge_lengths <= CUTOFF + 1e-9).all())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
def test_batched_nvalchemiops_kernel_matches_ase() -> None:
    if not hardware.nvalchemiops_available():
        pytest.skip("nvalchemi-toolkit-ops not installed")
    pos, batch, cell, pbc, expected = _batch_inputs()
    nl = build_neighbor_list_batch(
        pos.cuda(), CUTOFF, batch.cuda(), cell.cuda(), pbc.cuda(), backend="nvalchemiops"
    )
    assert _edges(nl) == expected


def test_explicit_nvalchemiops_degrades_to_torch(monkeypatch) -> None:
    monkeypatch.setattr(hardware, "nvalchemiops_available", lambda: False)
    pos, batch, cell, pbc, expected = _batch_inputs()
    nl = build_neighbor_list_batch(pos, CUTOFF, batch, cell, pbc, backend="nvalchemiops")
    assert _edges(nl) == expected


def test_cell_requires_pbc() -> None:
    with pytest.raises(ValueError, match="pbc"):
        build_neighbor_list_batch(torch.zeros(2, 3), CUTOFF, cell=torch.eye(3))


class TestPrimitivesFrontEnd:
    def test_returns_edge_index_and_vectors(self) -> None:
        from goal.ml.nn.primitives.neighbor_list import build_neighbor_list as front

        pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.5, 0.8, 0]])
        edge_index, edge_vec = front(pos, 1.5, backend="pyg")
        assert edge_index.shape == (2, 6)
        torch.testing.assert_close(edge_vec, pos[edge_index[1]] - pos[edge_index[0]])

    def test_batch_splitting_never_links_structures(self) -> None:
        from goal.ml.nn.primitives.neighbor_list import build_neighbor_list as front

        pos = torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [0.6, 0, 0], [1.0, 0, 0]])
        batch = torch.tensor([0, 0, 1, 1])
        edge_index, _ = front(pos, 2.0, batch=batch)
        assert bool((batch[edge_index[0]] == batch[edge_index[1]]).all())
        assert edge_index.shape[1] == 4

    def test_unknown_backend(self) -> None:
        from goal.ml.nn.primitives.neighbor_list import build_neighbor_list as front

        with pytest.raises(ValueError, match="backend"):
            front(torch.zeros(1, 3), 1.0, backend="magic")


@pytest.mark.parametrize("device", DEVICES)
def test_repeated_image_pairs_come_out_in_one_order(device: str) -> None:
    """In a cell smaller than twice the cutoff one (i, j) pair appears once
    per periodic image; sorting by (i, j, shift) makes every backend and
    device return the same edge order, shifts included."""
    atoms = bulk("C", "diamond", a=3.57)  # 2 atoms, many images within 4 Å
    pos = torch.tensor(atoms.positions, device=device)
    cell = torch.tensor(np.array(atoms.cell), device=device)
    pbc = torch.tensor(atoms.pbc, device=device)
    reference = build_neighbor_list_batch(
        pos.cpu(), CUTOFF, cell=cell.cpu(), pbc=pbc.cpu(), backend="torch"
    )
    backends = ["torch"] + (
        ["nvalchemiops"] if device == "cuda" and hardware.nvalchemiops_available() else []
    )
    for backend in backends:
        nl = build_neighbor_list_batch(pos, CUTOFF, cell=cell, pbc=pbc, backend=backend)
        assert torch.equal(nl.edge_index.cpu(), reference.edge_index), backend
        assert torch.equal(nl.unit_shifts.cpu(), reference.unit_shifts), backend
