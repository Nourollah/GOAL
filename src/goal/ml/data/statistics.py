"""Dataset-derived statistics used to initialise model buffers.

Mirrors the conventions used by MACE
(``mace.modules.blocks.AtomicEnergiesBlock`` and ``ScaleShiftBlock``)
and other production MLFFs (NequIP, Allegro):

* :func:`compute_atomic_references` — per-element energy contribution
  ``e(Z)`` solving the linear regression
  ``E_total_i = Σ_Z n_iZ · e_Z`` over the training set via least
  squares.  Returned as a ``{Z: e_Z}`` dict so callers don't have to
  track a column ordering.

* :func:`compute_energy_scale` — scalar
  ``std(E_total - e_atomic) / mean(n_atoms)`` over the same training
  set.  Used as the multiplicative gain on the interaction-energy
  output so the model only has to fit ``O(1)`` residuals rather than
  ``O(eV)`` ones.

* :func:`compute_avg_num_neighbors` — mean degree at the model's
  cutoff over the training set.  Used to normalise the sum-aggregated
  message inside :class:`EquivariantInteractionBlock`
  (``message_sum / sqrt(avg_num_neighbors)``) so the message
  magnitude is invariant to dataset density (sparse organic molecule
  vs. dense liquid).  Same role as MACE's
  ``avg_num_neighbors`` argument to ``RealAgnosticInteractionBlock``.

These are computed once before training (in the training entry
point) and passed to the model as constructor kwargs.  Whether they
are then frozen as buffers or used as parameter initialisations is a
backbone-level choice.
"""

from __future__ import annotations

import typing

import torch

# Minimal periodic table covering elements commonly found in molecular and
# materials datasets. Used for human-readable error messages only — not for
# any model logic. Extend as needed; missing entries fall back to "?".
ATOMIC_SYMBOLS: dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    53: "I",
    82: "Pb",
}


def compute_atomic_references(
    dataset: typing.Iterable[typing.Any],
) -> dict[int, float]:
    """Solve the per-element linear regression for atomic energies.

    Builds the species count matrix ``C[i, Z] = #{atoms of element Z
    in structure i}`` and the energy vector ``E[i] = E_total(i)``,
    then solves ``C @ e ≈ E`` via :func:`torch.linalg.lstsq`.

    Same construction as MACE's :class:`AtomicEnergiesBlock`
    (``mace/modules/blocks.py:359``) — subtract the per-element
    baseline so the interaction blocks only have to fit the small
    residual fluctuation.

    Parameters
    ----------
    dataset : iterable of AtomicGraph
        Yields ``AtomicGraph``-shaped objects with an integer
        ``atomic_numbers`` field and a scalar ``energy``.  Graphs
        whose ``energy`` is ``None`` are skipped.

    Returns
    -------
    dict
        ``{Z: e_Z}`` for every Z that appears in the dataset.
        Elements not present in the training set are not in the dict.
        Rank-deficient systems (e.g. a single-formula trajectory)
        return a minimum-norm solution — the per-structure
        prediction ``Σ_Z n_iZ · e_Z`` is still well-defined.
    """
    z_lists: list[list[int]] = []
    energies: list[float] = []
    observed_z: set[int] = set()

    for graph in dataset:
        e: typing.Any = getattr(graph, "energy", None)
        if e is None:
            continue
        z_list: list[int] = graph.atomic_numbers.to(torch.long).tolist()
        observed_z.update(z_list)
        z_lists.append(z_list)
        e_val: float = float(e.sum().item() if torch.is_tensor(e) else e)
        energies.append(e_val)

    if not z_lists:
        return {}

    sorted_z: list[int] = sorted(observed_z)
    z_to_col: dict[int, int] = {z: i for i, z in enumerate(sorted_z)}
    n_graphs: int = len(z_lists)
    n_cols: int = len(sorted_z)
    C: torch.Tensor = torch.zeros((n_graphs, n_cols), dtype=torch.float64)
    for i, zl in enumerate(z_lists):
        for z_val in zl:
            C[i, z_to_col[z_val]] += 1.0

    E: torch.Tensor = torch.tensor(energies, dtype=torch.float64)

    sol: torch.Tensor = torch.linalg.lstsq(C, E).solution  # (n_cols,)

    return {int(z): float(sol[z_to_col[z]]) for z in sorted_z}


def compute_energy_scale(
    dataset: typing.Iterable[typing.Any],
    atomic_references: dict[int, float],
) -> float:
    """Multiplicative gain for the interaction-energy output.

    Computed as ``std(E_total − e_atomic) / mean(n_atoms)`` over the
    training set.  Mirrors MACE's :class:`ScaleShiftBlock`
    (``mace/modules/blocks.py:1369``).  Returns ``1.0`` if the
    dataset is empty or all residuals are identical (zero divisor).

    Parameters
    ----------
    dataset : iterable of AtomicGraph
        Same training set used for
        :func:`compute_atomic_references`.
    atomic_references : dict
        Output of :func:`compute_atomic_references` — ``{Z: e_Z}``.

    Returns
    -------
    float
        The scale factor.
    """
    residuals: list[float] = []
    n_atoms_list: list[int] = []

    for graph in dataset:
        e: typing.Any = getattr(graph, "energy", None)
        if e is None:
            continue
        e_val: float = float(e.sum().item() if torch.is_tensor(e) else e)
        z_list: list[int] = graph.atomic_numbers.to(torch.long).tolist()
        e_atomic: float = sum(atomic_references.get(z_i, 0.0) for z_i in z_list)
        residuals.append(e_val - e_atomic)
        n_atoms_list.append(len(z_list))

    if not residuals:
        return 1.0

    residuals_t: torch.Tensor = torch.tensor(residuals, dtype=torch.float64)
    n_atoms_t: torch.Tensor = torch.tensor(n_atoms_list, dtype=torch.float64)

    std_val: float = float(residuals_t.std())
    mean_n: float = float(n_atoms_t.mean())
    if std_val == 0.0 or mean_n == 0.0:
        return 1.0
    return std_val / mean_n


def compute_unique_elements(
    dataset: typing.Iterable[typing.Any],
) -> list[int]:
    """Return a sorted list of unique atomic numbers present in the dataset.

    Walks every graph in the dataset and collects all distinct values of
    ``graph.atomic_numbers``.  Returns the result as a sorted list so the
    caller (training entry point) can inject it into the backbone's
    ``elements`` argument without any further processing.

    Parameters
    ----------
    dataset : iterable of AtomicGraph
        Training set graphs with an integer ``atomic_numbers`` field.

    Returns
    -------
    list of int
        Sorted unique atomic numbers.  Empty list if the dataset is empty.
    """
    observed: set[int] = set()
    for graph in dataset:
        z: typing.Any = getattr(graph, "atomic_numbers", None)
        if z is None:
            continue
        observed.update(z.to(torch.long).tolist())
    return sorted(observed)


def compute_pair_counts(
    dataset: typing.Iterable[typing.Any],
) -> dict[tuple[int, int], int]:
    """Count directed edges per unordered element pair over the training set.

    For each graph, walks every directed edge ``(i, j)`` and increments the
    counter for the canonical pair ``(min(Z_i, Z_j), max(Z_i, Z_j))``.
    Bidirectional edges are counted twice (once as ``i→j`` and once as
    ``j→i``) which matches the convention used in :class:`SimurghArtisanBank`
    (each undirected pair contributes exactly two directed edges).

    Used by :class:`SimurghArtisanBank` to decide whether a pair is common enough
    to warrant a dedicated :class:`PotentialArtisan` or should be routed to
    the shared :class:`RarePotentialArtisan`.

    Parameters
    ----------
    dataset : iterable of AtomicGraph
        Training set — graphs with ``atomic_numbers`` and ``edge_index``.

    Returns
    -------
    dict
        ``{(Z_lo, Z_hi): edge_count}`` covering every pair observed.
        Pairs not present in the dataset are absent from the dict
        (caller should treat missing pairs as count = 0).
    """
    counts: dict[tuple[int, int], int] = {}
    for graph in dataset:
        edge_index: typing.Any = getattr(graph, "edge_index", None)
        atomic_numbers: typing.Any = getattr(graph, "atomic_numbers", None)
        if edge_index is None or atomic_numbers is None:
            continue
        z: list[int] = atomic_numbers.to(torch.long).tolist()
        src_list: list[int] = edge_index[0].tolist()
        dst_list: list[int] = edge_index[1].tolist()
        for i, j in zip(src_list, dst_list):
            lo, hi = (min(z[i], z[j]), max(z[i], z[j]))
            key: tuple[int, int] = (lo, hi)
            counts[key] = counts.get(key, 0) + 1
    return counts


def compute_atom_statistics(
    dataset: typing.Iterable[typing.Any],
    descriptors: typing.Mapping[str, typing.Callable[[typing.Any], torch.Tensor]],
) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of per-atom descriptors over every atom of ``dataset``.

    ``descriptors`` maps a name to a function of one graph returning one
    value per atom ``(N,)``.  Accumulated in float64 as sums and counts, so
    the result does not depend on how the dataset is split into graphs.

    Returns
    -------
    dict
        ``{name: {"mean": m, "std": s}}`` (population s.d.).
    """
    sums = {name: torch.zeros(3, dtype=torch.float64) for name in descriptors}
    for graph in dataset:
        for name, fn in descriptors.items():
            with torch.no_grad():
                values = fn(graph).detach().to(torch.float64).flatten()
            sums[name] += torch.stack(
                [values.new_tensor(values.numel()), values.sum(), values.square().sum()]
            )
    out: dict[str, dict[str, float]] = {}
    for name, (count, total, squares) in sums.items():
        n = max(float(count), 1.0)
        mean = float(total) / n
        out[name] = {"mean": mean, "std": max(float(squares) / n - mean * mean, 0.0) ** 0.5}
    return out


def compute_avg_num_neighbors(
    dataset: typing.Iterable[typing.Any],
) -> float:
    """Mean number of neighbours per atom over the training set.

    Walks every graph and computes ``num_edges / num_atoms``; returns
    the global mean of that ratio.  Pre-supposes the dataset has
    already been neighbour-listed at the cutoff the model will use
    (true for ``GOALDataModule`` — edges are built once in
    ``AtomicGraph.from_ase`` and reused).  Skips graphs with no
    ``edge_index`` or zero atoms.

    Used by :class:`EquivariantInteractionBlock` to normalise the
    sum-aggregated message: ``msg_sum / sqrt(avg_num_neighbors)``.
    Matches MACE's ``avg_num_neighbors`` semantics — a single scalar
    derived from the training data, frozen as a buffer at
    construction time so it is identical on every rank.

    Returns
    -------
    float
        Mean degree.  ``1.0`` if the dataset is empty (so the
        ``sqrt`` divisor is a no-op).
    """
    degrees: list[float] = []
    for graph in dataset:
        edge_index: typing.Any = getattr(graph, "edge_index", None)
        atomic_numbers: typing.Any = getattr(graph, "atomic_numbers", None)
        if edge_index is None or atomic_numbers is None:
            continue
        n_atoms: int = int(atomic_numbers.numel())
        if n_atoms == 0:
            continue
        n_edges: int = int(edge_index.size(1))
        degrees.append(n_edges / n_atoms)
    if not degrees:
        return 1.0
    return float(torch.tensor(degrees, dtype=torch.float64).mean())
