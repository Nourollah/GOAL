"""Fragment decomposition — how a structure is split into fragments.

A *fragment* is a group of atoms that
:class:`~goal.ml.nn.blocks.fragment_interaction.EquivariantFragmentInteraction`
treats as one unit: their features are pooled, the pooled representations
exchange equivariant messages, and the result is handed back to every
atom.  What counts as "one unit" is a modelling choice, so five schemes
are available, selected by ``data.fragment_scheme``:

==================== ================================================= ===========
scheme               splits on                                         needs rdkit
==================== ================================================= ===========
``connected``        distance graph at ``covalent_cutoff``             no
``rdkit_components`` connected components of the *perceived* molecule  yes
``rotatable``        every non-ring single bond between heavy atoms    yes
``brics``            BRICS retrosynthetic rules (``rdkit.Chem.BRICS``) yes
``recap``            RECAP cleavage rules (``rdkit.Chem.Recap``)       yes
``custom``           your own SMARTS bond patterns (``smarts=``)       yes
==================== ================================================= ===========

Which to pick depends on what you want the module to model:

* ``connected`` (default) gives **physically separate molecules** — the
  cutoff is deliberately far shorter than the MLIP interaction cutoff, so
  two molecules 3 Å apart are two fragments even though every atom pair
  is inside the model's receptive field.  The fragment channel then
  models genuine non-bonded interaction.  On a single covalently
  connected molecule it yields ``K = 1`` and the channel is a no-op.
* ``rotatable`` / ``brics`` / ``recap`` cut a **single molecule** into
  chemically meaningful groups, turning the channel into a
  coarse-grained *intramolecular* pathway: pooled groups exchange
  messages directly instead of atom-by-atom along the chain.  Worth the
  cost mainly when the fragment cutoff exceeds the atom cutoff, so
  distant parts of one molecule couple in a single step.
* ``rdkit_components`` is ``connected`` computed on the perceived
  molecule.  It normally agrees with ``connected`` exactly; it exists so
  you can confirm that, and so bond perception (rather than a distance
  threshold) decides what "bonded" means.
* ``custom`` cuts exactly the bonds *you* name in ``smarts``, for when no
  rule set matches your chemistry.

Two hints apply on top of any scheme: ``keep_groups`` names SMARTS whose
atoms must share a fragment, and ``smarts`` drives ``custom``.  rdkit's
own BRICS/RECAP rules are hard-coded module data with no settings file, so
these are the supported way to encode chemistry knowledge.

Every scheme returns the same thing: contiguous ``(N,)`` int64 labels,
ordered by each fragment's lowest atom index, requiring no supervision.

Implementation notes
--------------------
Connected components are found by **label propagation** (iterated
``amin`` over the bond graph) rather than ``scipy.sparse.csgraph``:
``scipy`` is not a declared GOAL dependency, and label propagation is a
handful of vectorised torch ops with no host round-trips beyond the
convergence check.  The bond graph itself is built in row chunks so a
10 000-atom cell never materialises an ``(N, N, 3)`` displacement
tensor.

The rdkit schemes need bond perception (``rdDetermineBonds``, the
xyz2mol algorithm) which is far more expensive and can fail on radicals,
unusual valences or a wrong total charge.  Two things make that
practical: results are **cached on the bond graph** — perception depends
only on connectivity and charge, so every frame of an MD trajectory whose
topology is unchanged reuses one perception — and a failure falls back to
``connected`` with a single warning rather than killing the run.
"""

from __future__ import annotations

import typing
import warnings

import torch

# Rows of the pairwise distance matrix evaluated per chunk.  1024 rows of
# a 10k-atom system is ~80 MB in float64 — comfortably small, while still
# large enough that the chunk loop is never the bottleneck for the
# molecule-sized systems GOAL usually trains on.
_ROW_CHUNK: int = 1024

#: Every fragmentation scheme ``data.fragment_scheme`` accepts.
FRAGMENT_SCHEMES: tuple[str, ...] = (
    "connected",
    "rdkit_components",
    "rotatable",
    "brics",
    "recap",
    "custom",
)

#: Schemes that need rdkit bond perception.
_RDKIT_SCHEMES: frozenset[str] = frozenset(FRAGMENT_SCHEMES) - {"connected"}

#: Strict rotatable-bond SMARTS: a non-ring single bond between two
#: non-terminal, non-triple-bonded atoms.  The ``!D1`` on both ends is
#: what keeps terminal groups (a carbonyl O, a methyl H) attached to
#: their neighbour instead of being shaved off as one-atom fragments.
_ROTATABLE_SMARTS: str = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"

#: Perception + fragmentation cache, keyed by (scheme, charge, elements,
#: bond graph).  Perception depends only on connectivity and charge, so
#: every frame of a trajectory with unchanged topology reuses one result.
_LABEL_CACHE: dict[typing.Any, tuple[int, ...]] = {}
_CACHE_LIMIT: int = 4096

#: Warn once per distinct message, not once per frame.
_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        warnings.warn(message, stacklevel=3)


def compute_fragment_index(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    covalent_cutoff: float = 1.8,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    scheme: str = "connected",
    charge: int = 0,
    on_failure: str = "fallback",
    smarts: typing.Sequence[str] | None = None,
    keep_groups: typing.Sequence[str] | None = None,
) -> torch.Tensor:
    """Label every atom with the index of the fragment it belongs to.

    Parameters
    ----------
    positions : Tensor ``(N, 3)``
        Cartesian atomic positions (Angstrom).
    atomic_numbers : Tensor ``(N,)``
        Atomic numbers.  Used by the rdkit schemes to build the molecule;
        the ``connected`` scheme only checks that the two inputs describe
        the same structure (its bond criterion is a single
        element-agnostic distance threshold).
    covalent_cutoff : float
        Bond threshold in Angstrom (default ``1.8``).  Defines the bond
        graph for ``connected``, and for the rdkit schemes it is the
        fallback criterion and the cache key.
    cell : Tensor ``(3, 3)`` or ``(1, 3, 3)``, optional
        Lattice vectors.  Together with *pbc* this applies the minimum
        image convention, so a molecule straddling a periodic boundary
        stays one fragment instead of splitting in two.
    pbc : Tensor ``(3,)``, optional
        Periodic-boundary flags.  Ignored when *cell* is absent or
        all-zero (the molecular case).
    scheme : str
        One of :data:`FRAGMENT_SCHEMES` — see the module docstring for
        what each one splits on and when to prefer it.
    charge : int
        Total charge handed to rdkit's bond perception.  Wrong here means
        perception fails or invents bond orders, so set it for ionic
        systems.
    on_failure : ``"fallback"`` or ``"raise"``
        What to do when an rdkit scheme cannot perceive the molecule.
        ``"fallback"`` (default) warns once and uses ``connected`` for
        that structure; ``"raise"`` propagates, which is what you want
        when a silently degraded decomposition would ruin an experiment.
    smarts : sequence of str, optional
        Bond patterns to cut, for ``scheme="custom"``.  Each pattern
        either has exactly two atoms (the bond between them is cut) or
        marks the two ends with atom maps ``:1`` and ``:2``.  Ignored by
        every other scheme.
    keep_groups : sequence of str, optional
        SMARTS whose matched atoms must end up in the **same** fragment,
        applied on top of *any* scheme.  Two things happen: a bond is
        never cut when both its ends lie inside one match, and the final
        labels are merged so the guarantee holds even when the base
        scheme had already separated them.  This is how you tell a
        chemistry-driven scheme to leave a group alone — ``["C(=O)O"]``
        keeps a carboxyl intact instead of letting BRICS shed its ``OH``.
        Needs rdkit, so it adds a perception step to ``connected``.

    Returns
    -------
    Tensor ``(N,)`` int64
        Fragment labels, contiguous and 0-indexed
        (``labels.max() + 1 == n_fragments``).  Labels are ordered by
        the lowest atom index each fragment contains, so atom 0 always
        belongs to fragment 0 and the numbering is deterministic.

    Examples
    --------
    A single methane molecule yields one fragment::

        >>> labels = compute_fragment_index(ch4_pos, ch4_z)
        >>> int(labels.max()) + 1
        1

    Two water molecules 3 Å apart yield two::

        >>> labels = compute_fragment_index(dimer_pos, dimer_z)
        >>> int(labels.max()) + 1
        2

    One butanoic acid cut at its rotatable bonds yields five::

        >>> labels = compute_fragment_index(acid_pos, acid_z, scheme="rotatable")
        >>> int(labels.max()) + 1
        5
    """
    if positions.dim() != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}.")
    n_atoms: int = int(positions.shape[0])
    if int(atomic_numbers.shape[0]) != n_atoms:
        raise ValueError(
            f"positions ({n_atoms} atoms) and atomic_numbers "
            f"({int(atomic_numbers.shape[0])} atoms) describe different structures."
        )
    if covalent_cutoff <= 0.0:
        raise ValueError(f"covalent_cutoff must be > 0, got {covalent_cutoff}.")
    if scheme not in FRAGMENT_SCHEMES:
        raise ValueError(
            f"Unknown fragment scheme {scheme!r}.  Choose one of "
            f"{list(FRAGMENT_SCHEMES)} (data.fragment_scheme)."
        )
    if on_failure not in ("fallback", "raise"):
        raise ValueError(f"on_failure must be 'fallback' or 'raise', got {on_failure!r}.")
    validate_fragment_config(scheme, smarts, keep_groups)

    device: torch.device = positions.device
    if n_atoms <= 1:
        return torch.zeros(n_atoms, dtype=torch.long, device=device)

    src, dst = _bond_pairs(positions, covalent_cutoff, cell, pbc)
    smarts_key: tuple[str, ...] = tuple(as_pattern_list(smarts))
    keep_key: tuple[str, ...] = tuple(as_pattern_list(keep_groups))

    # ``connected`` needs no rdkit at all — unless keep_groups asks for
    # SMARTS matching, which does.
    if scheme == "connected" and not keep_key:
        return _components_from_bonds(src, dst, n_atoms, device)

    try:
        labels = _rdkit_labels(
            positions=positions,
            atomic_numbers=atomic_numbers,
            bond_src=src,
            bond_dst=dst,
            scheme=scheme,
            charge=charge,
            periodic=bool(pbc is not None and bool(pbc.any())),
            smarts=smarts_key,
            keep_groups=keep_key,
        )
    except Exception as exc:  # noqa: BLE001 — any rdkit failure is recoverable
        if on_failure == "raise":
            raise
        _warn_once(
            f"Fragment scheme {scheme!r} failed on a structure "
            f"({type(exc).__name__}: {exc}); falling back to 'connected' for "
            f"every structure that fails.  Common causes: a radical or "
            f"open-shell species, a wrong data.fragment_charge, or a "
            f"distorted MD frame mid-bond-breaking.  Set "
            f"data.fragment_on_failure: raise to make this fatal instead."
        )
        return _components_from_bonds(src, dst, n_atoms, device)

    return labels.to(device)


def attach_fragment_index(
    graph: typing.Any,
    covalent_cutoff: float = 1.8,
    scheme: str = "connected",
    charge: int = 0,
    on_failure: str = "fallback",
    smarts: typing.Sequence[str] | None = None,
    keep_groups: typing.Sequence[str] | None = None,
) -> typing.Any:
    """Label *graph* with ``fragment_index`` in place, and return it.

    The graph-level counterpart to :func:`compute_fragment_index`: it
    reads the positions, species, cell and pbc off an ``AtomicGraph``
    and writes the labels back onto it.  Already-labelled graphs are
    left alone, so calling it twice costs nothing — which is what lets a
    dataset holding its graphs in memory attach labels lazily on first
    access and keep them for the rest of the run.

    Shared by both dataset hierarchies: ``BaseAtomicDataset`` (the
    ``goal.ml.data.datasets`` loaders) and ``BenchmarkDataset`` (the
    ``goal.ml.data.benchmarks`` adapters), so the two cannot drift apart on
    what a fragment label means.

    Parameters
    ----------
    graph : AtomicGraph
        Mutated in place; also returned for convenience.
    covalent_cutoff, scheme, charge, on_failure, smarts, keep_groups
        As in :func:`compute_fragment_index`.
    """
    if graph.get("fragment_index", None) is not None:
        return graph

    # PyG stores a single graph's cell as ``(1, 3, 3)`` (see
    # ``AtomicGraph.__init__``); the fragment code works in ``(3, 3)``.
    cell: torch.Tensor | None = graph.get("cell", None)
    if cell is not None and cell.dim() == 3 and cell.shape[0] == 1:
        cell = cell.squeeze(0)

    graph.fragment_index = compute_fragment_index(
        positions=graph.pos,
        atomic_numbers=graph.z,
        covalent_cutoff=covalent_cutoff,
        cell=cell,
        pbc=graph.get("pbc", None),
        scheme=scheme,
        charge=charge,
        on_failure=on_failure,
        smarts=smarts,
        keep_groups=keep_groups,
    )
    return graph


def validate_fragment_config(
    scheme: str,
    smarts: typing.Sequence[str] | None = None,
    keep_groups: typing.Sequence[str] | None = None,
) -> None:
    """Fail fast on a bad fragment config, at setup rather than mid-epoch.

    Checks the scheme name, that ``custom`` was given patterns, and that
    every SMARTS parses and (for ``custom``) unambiguously identifies one
    bond.  Datasets call this in ``__init__`` so a typo surfaces before
    the first frame is read, not thousands of frames later.
    """
    if scheme not in FRAGMENT_SCHEMES:
        raise ValueError(
            f"Unknown fragment scheme {scheme!r}.  Choose one of "
            f"{list(FRAGMENT_SCHEMES)} (data.fragment_scheme)."
        )
    smarts_list: list[str] = as_pattern_list(smarts)
    keep_list: list[str] = as_pattern_list(keep_groups)
    if smarts_list and scheme != "custom":
        # Easy and costly mistake: setting fragment_smarts alongside
        # `brics`/`rotatable`/... looks like it defines the fragments, but
        # those schemes carry their own rules and never read the patterns.
        # The run would silently fragment by the rule set instead.
        _warn_once(
            f"data.fragment_smarts is set but data.fragment_scheme is "
            f"{scheme!r}, which uses its own built-in rules and IGNORES the "
            f"patterns.  Set fragment_scheme: custom to cut the bonds you "
            f"named, or use data.fragment_keep_groups (honoured by every "
            f"scheme) to state which atoms must stay together."
        )
    if scheme == "custom" and not smarts_list:
        raise ValueError(
            "data.fragment_scheme: custom requires data.fragment_smarts — a "
            "list of SMARTS bond patterns to cut, e.g. ['[CX4]-[CX4]'].  Each "
            "pattern must either contain exactly two atoms or mark the two "
            "ends of the bond with atom maps :1 and :2."
        )
    for pattern in smarts_list:
        _compile_bond_pattern(pattern)
    for pattern in keep_list:
        _compile_smarts(pattern, key="data.fragment_keep_groups")


def as_pattern_list(value: typing.Any) -> list[str]:
    """Normalise a SMARTS config value to a list of patterns.

    ``fragment_smarts: "[CX4]-[CX4]"`` in YAML is a plain string, and
    iterating it would hand each *character* to the SMARTS parser — the
    resulting ``'[' is not a valid SMARTS pattern`` tells you nothing
    about the real mistake.  A lone string obviously means one pattern,
    so accept it.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _compile_smarts(pattern: str, key: str) -> typing.Any:
    """Parse a SMARTS string, with the config key named in the error."""
    from rdkit import Chem

    query = Chem.MolFromSmarts(pattern)
    if query is None:
        raise ValueError(f"{key}: {pattern!r} is not a valid SMARTS pattern.")
    return query


def _compile_bond_pattern(pattern: str) -> tuple[typing.Any, int, int]:
    """Parse a ``fragment_smarts`` entry into ``(query, end_a, end_b)``.

    The bond to cut is identified either by the pattern having exactly two
    atoms, or by atom maps ``:1`` and ``:2`` marking its ends.  Anything
    else is ambiguous and rejected — silently guessing which bond of a
    five-atom pattern to break is exactly the sort of thing that would
    quietly produce the wrong fragments for a whole run.
    """
    query = _compile_smarts(pattern, key="data.fragment_smarts")
    mapped: dict[int, int] = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in query.GetAtoms()
        if atom.GetAtomMapNum() in (1, 2)
    }
    if len(mapped) == 2:
        return query, mapped[1], mapped[2]
    if query.GetNumAtoms() == 2:
        return query, 0, 1
    raise ValueError(
        f"data.fragment_smarts: {pattern!r} has {query.GetNumAtoms()} atoms and "
        f"no ':1'/':2' atom maps, so which bond to cut is ambiguous.  Either "
        f"use a two-atom pattern or mark the bond ends, e.g. "
        f"'[C:1](=O)-[O:2]'."
    )


def _components_from_bonds(
    src: torch.Tensor,
    dst: torch.Tensor,
    n_atoms: int,
    device: torch.device,
) -> torch.Tensor:
    """Connected components of a symmetric bond list, via label propagation."""
    labels: torch.Tensor = torch.arange(n_atoms, dtype=torch.long, device=device)

    if src.numel() > 0:
        # Iterated min-propagation: every atom adopts the smallest label
        # in its neighbourhood.  Converges in at most (graph diameter)
        # sweeps; the n_atoms bound is a safety net, never reached in
        # practice.
        for _ in range(n_atoms):
            propagated: torch.Tensor = labels.scatter_reduce(
                0, dst, labels[src], reduce="amin", include_self=True
            )
            if bool(torch.equal(propagated, labels)):
                break
            labels = propagated

    return _renumber(labels)


def _renumber(labels: torch.Tensor) -> torch.Tensor:
    """Contiguous 0..K-1 labels ordered by each group's lowest atom index.

    ``torch.unique`` sorts, so the inverse mapping is exactly that order —
    atom 0 always lands in fragment 0 and the numbering is deterministic
    across frames.
    """
    _, inverse = torch.unique(labels, return_inverse=True)
    return inverse.to(torch.long)


def _rdkit_labels(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    bond_src: torch.Tensor,
    bond_dst: torch.Tensor,
    scheme: str,
    charge: int,
    periodic: bool,
    smarts: tuple[str, ...] = (),
    keep_groups: tuple[str, ...] = (),
) -> torch.Tensor:
    """Fragment labels from an rdkit-perceived molecule.

    Perception (``rdDetermineBonds``, the xyz2mol algorithm) depends only
    on the connectivity and the total charge, so the result is cached on
    the bond graph: an MD trajectory whose topology never changes pays
    for perception exactly once, not once per frame.
    """
    n_atoms: int = int(positions.shape[0])
    elements: tuple[int, ...] = tuple(int(z) for z in atomic_numbers.tolist())

    # Cache key: same elements + same bonds ⇒ same perception ⇒ same
    # fragmentation, whatever the coordinates did in between.
    pairs = torch.stack([bond_src, bond_dst], dim=-1) if bond_src.numel() else None
    bond_key: tuple[tuple[int, int], ...] = (
        tuple(sorted({(min(a, b), max(a, b)) for a, b in pairs.tolist()}))
        if pairs is not None
        else ()
    )
    key = (scheme, int(charge), elements, bond_key, smarts, keep_groups)
    cached = _LABEL_CACHE.get(key)
    if cached is not None:
        return torch.tensor(cached, dtype=torch.long)

    if periodic:
        _warn_once(
            f"Fragment scheme {scheme!r} runs rdkit bond perception, which has "
            f"no notion of periodicity: bonds across a cell boundary are not "
            f"seen, so a molecule wrapped by the box may be split.  The "
            f"'connected' scheme handles PBC correctly via the minimum image "
            f"convention."
        )

    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    table = Chem.GetPeriodicTable()
    coords = positions.detach().to(torch.float64).tolist()
    block: str = f"{n_atoms}\n\n" + "\n".join(
        f"{table.GetElementSymbol(z)} {x:.8f} {y:.8f} {zc:.8f}"
        for z, (x, y, zc) in zip(elements, coords)
    )
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        raise ValueError("rdkit could not read the structure as an XYZ block.")
    # Perception writes its complaints straight to stderr from C++, where
    # neither ``warnings`` nor logging can reach them.  On a corpus with a
    # non-trivial failure rate (measured: ~7% of OMol25 rows) that is both
    # a flood of unattributable lines and a real cost — 300 structures/s
    # became 70 with the logger live.  Nothing is lost by blocking it:
    # a failure still raises, and ``compute_fragment_index`` reports the
    # exception message through ``_warn_once``.  ``BlockLogs`` restores
    # whatever state was there on exit, so a caller who wants rdkit's
    # chatter elsewhere keeps it.
    from rdkit.rdBase import BlockLogs

    with BlockLogs():
        rdDetermineBonds.DetermineBonds(mol, charge=int(charge))

    # Atoms that must stay together, matched once and used twice: to veto
    # cuts inside a group, and to merge labels afterwards.
    protected: list[tuple[int, ...]] = _keep_group_matches(mol, keep_groups)

    break_bonds: list[int] = _scheme_break_bonds(mol, scheme, smarts)
    if protected:
        break_bonds = _drop_protected_bonds(mol, break_bonds, protected)

    fragmented = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False) if break_bonds else mol

    labels = torch.zeros(n_atoms, dtype=torch.long)
    for group_idx, group in enumerate(Chem.GetMolFrags(fragmented)):
        labels[list(group)] = group_idx
    if protected:
        # Vetoing cuts is not enough on its own: ``connected`` never cuts
        # anything, and a group can straddle fragments the base scheme
        # separated for other reasons.  Merging makes the guarantee real.
        labels = _merge_by_groups(labels, protected)
    labels = _renumber(labels)

    if len(_LABEL_CACHE) < _CACHE_LIMIT:
        _LABEL_CACHE[key] = tuple(int(v) for v in labels.tolist())
    return labels


def _keep_group_matches(
    mol: typing.Any,
    keep_groups: tuple[str, ...],
) -> list[tuple[int, ...]]:
    """Atom-index tuples for every match of every ``keep_groups`` pattern."""
    matches: list[tuple[int, ...]] = []
    for pattern in keep_groups:
        query = _compile_smarts(pattern, key="data.fragment_keep_groups")
        matches.extend(mol.GetSubstructMatches(query))
    return matches


def _drop_protected_bonds(
    mol: typing.Any,
    bond_ids: list[int],
    protected: list[tuple[int, ...]],
) -> list[int]:
    """Remove cuts whose two ends sit inside one protected match."""
    groups: list[set[int]] = [set(match) for match in protected]
    kept: list[int] = []
    for bond_id in bond_ids:
        bond = mol.GetBondWithIdx(bond_id)
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if any(begin in group and end in group for group in groups):
            continue
        kept.append(bond_id)
    return kept


def _merge_by_groups(
    labels: torch.Tensor,
    groups: list[tuple[int, ...]],
) -> torch.Tensor:
    """Union the labels of atoms sharing a protected match.

    Union-find over *labels* (not atoms), so overlapping and chained
    groups merge transitively — protecting ``C(=O)O`` and ``O-H`` on the
    same carboxyl collapses all three fragments into one, as intended.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for match in groups:
        first = int(labels[match[0]])
        for atom in match[1:]:
            union(first, int(labels[atom]))

    if not parent:
        return labels
    return torch.tensor([find(int(v)) for v in labels.tolist()], dtype=torch.long)


def _scheme_break_bonds(
    mol: typing.Any,
    scheme: str,
    smarts: tuple[str, ...] = (),
) -> list[int]:
    """Bond indices to cut for *scheme* (empty = keep the molecule whole)."""
    from rdkit import Chem

    if scheme == "connected":
        # Only reachable when keep_groups forced the rdkit path; the
        # fragments are the distance graph's components, which for a
        # perceived molecule are its own connected components.
        return []

    if scheme == "custom":
        found: set[int] = set()
        for pattern in smarts:
            query, end_a, end_b = _compile_bond_pattern(pattern)
            for match in mol.GetSubstructMatches(query):
                bond = mol.GetBondBetweenAtoms(match[end_a], match[end_b])
                if bond is not None:
                    found.add(bond.GetIdx())
        return sorted(found)

    if scheme == "rdkit_components":
        # Cut nothing: the fragments are the perceived molecule's own
        # connected components, which is what makes this the rdkit mirror
        # of the ``connected`` scheme.
        return []

    if scheme == "rotatable":
        pattern = Chem.MolFromSmarts(_ROTATABLE_SMARTS)
        return sorted(
            {
                mol.GetBondBetweenAtoms(i, j).GetIdx()
                for i, j in mol.GetSubstructMatches(pattern)
                if mol.GetBondBetweenAtoms(i, j) is not None
            }
        )

    if scheme == "brics":
        from rdkit.Chem import BRICS

        found: set[int] = set()
        for (begin, end), _labels in BRICS.FindBRICSBonds(mol):
            bond = mol.GetBondBetweenAtoms(int(begin), int(end))
            if bond is not None:
                found.add(bond.GetIdx())
        return sorted(found)

    if scheme == "recap":
        return _recap_break_bonds(mol)

    raise ValueError(f"Unhandled fragment scheme {scheme!r}.")


def _recap_break_bonds(mol: typing.Any) -> list[int]:
    """Bonds that RECAP's cleavage rules mark, as an atom partition.

    ``Recap.RecapDecompose`` returns a hierarchy of fragment *molecules*
    with dummy atoms, not a partition of the input atoms, and some of its
    reactions drop the bridging atom entirely — neither is usable as a
    per-atom label.  Instead each reaction's reactant template is matched
    directly and every bond it marks ``!@`` (explicitly acyclic, i.e. the
    bond RECAP would cleave) is cut.

    Consequence worth knowing: when both bonds flanking a bridging
    heteroatom are cut — an ether oxygen, say — that atom becomes its own
    single-atom fragment.  That is a faithful reading of the rules, not a
    bug, but it is why ``recap`` produces more, smaller fragments than
    ``RecapDecompose`` reports leaves.
    """
    from rdkit.Chem import AllChem, Recap

    found: set[int] = set()
    for smarts in Recap.reactionDefs:
        reaction = AllChem.ReactionFromSmarts(smarts)
        template = reaction.GetReactantTemplate(0)
        for match in mol.GetSubstructMatches(template):
            for template_bond in template.GetBonds():
                if "!@" not in template_bond.GetSmarts():
                    continue
                bond = mol.GetBondBetweenAtoms(
                    match[template_bond.GetBeginAtomIdx()],
                    match[template_bond.GetEndAtomIdx()],
                )
                if bond is not None and not bond.IsInRing():
                    found.add(bond.GetIdx())
    return sorted(found)


def _bond_pairs(
    positions: torch.Tensor,
    covalent_cutoff: float,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric ``(src, dst)`` index pairs of all contacts within *covalent_cutoff*."""
    n_atoms: int = int(positions.shape[0])
    device: torch.device = positions.device
    use_mic: bool = _use_mic(cell, pbc)
    cell_2d: torch.Tensor | None = None
    if use_mic and cell is not None:
        cell_2d = cell.squeeze(0) if cell.dim() == 3 else cell

    cutoff_sq: float = float(covalent_cutoff) ** 2
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []

    for start in range(0, n_atoms, _ROW_CHUNK):
        stop: int = min(start + _ROW_CHUNK, n_atoms)
        disp: torch.Tensor = positions[start:stop].unsqueeze(1) - positions.unsqueeze(0)
        if use_mic and cell_2d is not None and pbc is not None:
            from goal.ml.data.graph import _apply_mic

            rows: int = stop - start
            disp = _apply_mic(disp.reshape(-1, 3), cell_2d, pbc).reshape(rows, n_atoms, 3)
        dist_sq: torch.Tensor = disp.pow(2).sum(dim=-1)  # (rows, N)

        local_src, local_dst = torch.nonzero(dist_sq < cutoff_sq, as_tuple=True)
        global_src: torch.Tensor = local_src + start
        keep: torch.Tensor = global_src != local_dst  # drop self-contacts
        src_parts.append(global_src[keep])
        dst_parts.append(local_dst[keep])

    src: torch.Tensor = (
        torch.cat(src_parts) if src_parts else torch.zeros(0, dtype=torch.long, device=device)
    )
    dst: torch.Tensor = (
        torch.cat(dst_parts) if dst_parts else torch.zeros(0, dtype=torch.long, device=device)
    )
    return src, dst


def _use_mic(cell: torch.Tensor | None, pbc: torch.Tensor | None) -> bool:
    """Whether the minimum image convention applies to this structure."""
    if cell is None or pbc is None:
        return False
    if not bool(pbc.any()):
        return False
    # Molecular graphs carry an all-zero (singular) cell — inverting it
    # would produce NaNs, and there is nothing to wrap anyway.
    return bool(cell.abs().sum() > 0)
