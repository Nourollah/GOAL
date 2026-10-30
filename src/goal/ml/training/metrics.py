"""MLIP diagnostic metrics computed at every train/val step.

Standard MLFF-paper metrics (MAE/RMSE per atom for energy, MAE/RMSE
and cosine-similarity / magnitude split for forces, Newton-violation
for translation invariance).  Computed alongside the loss so the
progress bar and loggers can show *physically meaningful* numbers
even when the loss values themselves are weighted sums of several
components.

The entry point is :func:`mlip_metrics(predictions, batch)` which
returns a flat ``{name: scalar tensor}`` dict.  The training loop
turns each name into a logger key with :func:`metric_section` (which
files energy numbers under ``{split}_energy/``, force numbers under
``{split}_forces/`` and gate diagnostics under ``{split}_gate/``) and
feeds them to Lightning's ``log_dict``.

Metrics emitted
---------------
``energy_mae_per_atom``, ``energy_rmse_per_atom``, ``forces_mae``,
``forces_rmse``, ``forces_cosine_similarity``, ``forces_magnitude_mae``,
and the Newton violation as **four** numbers:
``newton_violation_{x,y,z}`` per Cartesian axis plus ``newton_violation``
for their sum.
"""

from __future__ import annotations

import collections
import itertools
import typing

import numpy as np
import torch
from ase.data import chemical_symbols

# Names elevated to the Lightning progress bar.  The training loop
# reads this so it can route only the headline metrics through
# ``prog_bar=True``.  The per-axis Newton components are deliberately
# absent: the total carries the headline, and the components are there
# for the loggers when it goes bad.
PROG_BAR_METRICS: frozenset[str] = frozenset(
    {
        "energy_mae_per_atom",
        "forces_cosine_similarity",
        "newton_violation",
    }
)


# ----------------------------------------------------------------------
# Logger sections
# ----------------------------------------------------------------------
#
# W&B and TensorBoard group panels by the text before the first ``/``,
# so the section a curve lands in is chosen by the *prefix* the training
# loop builds: ``{split}`` plus the sub-section resolved here.  One
# table, used for both the diagnostic metrics above and the loss labels
# coming out of ``CompositeLoss``, so a name cannot be filed in two
# different sections depending on which logging helper emitted it.
#
#   {split}/         the run's verdict: ``total``, plus any loss
#                    component that is neither energy nor forces
#                    (``stress``, ``dipole``, ...).
#   {split}_energy/  everything about the energy: its loss term, the
#                    per-atom MAE/RMSE, the per-round decomposition.
#   {split}_forces/  everything about the forces: its loss term, the
#                    MAE/RMSE/cosine/magnitude split, and the Newton
#                    violation (a force diagnostic, four numbers).
#   {split}_gate/    every gate diagnostic — numerous and for
#                    occasional inspection, never for judging a run.
#
# Splitting energy from forces matters because the two live on
# different scales and are read for different reasons: an energy MAE
# panel next to a force cosine-similarity panel is two unrelated
# questions sharing a row.

#: Sub-section for a metric or loss label, matched by exact name.
METRIC_SECTIONS: dict[str, str] = {
    # Loss labels (``WeightedLoss.label`` in the CompositeLoss).
    "energy": "energy",
    "forces": "forces",
    "aux_loss": "gate",
    "gate_reg": "gate",
    # Energy diagnostics.
    "energy_mae_per_atom": "energy",
    "energy_rmse_per_atom": "energy",
    # Force diagnostics.
    "forces_mae": "forces",
    "forces_rmse": "forces",
    "forces_cosine_similarity": "forces",
    "forces_magnitude_mae": "forces",
}

#: Names routed by prefix rather than by exact match, in priority
#: order.  Two families cannot be enumerated: the per-round /
#: per-element-pair diagnostics (``energy_round_{L}``,
#: ``gate_mean_round_{L}``, ``artisan/round{L}/{pair}``,
#: ``artisan_load/{pair}``) and the composite sub-loss labels, which
#: ``build_loss`` names ``{property}_{fn}`` — ``forces_mse``,
#: ``energy_huber`` and so on for any registered loss function.  So a
#: name simply beginning with ``energy`` or ``forces`` follows its
#: property, which is also what keeps a sub-loss in the same section as
#: the group total it belongs to.
_SECTION_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("energy", "energy"),
    ("forces", "forces"),
    ("newton_violation", "forces"),
    ("gate", "gate"),
    ("artisan", "gate"),
)


def metric_section(name: str) -> str | None:
    """Sub-section for *name*, or ``None`` when it belongs in ``{split}/``.

    ``"forces_mae"`` -> ``"forces"`` (logged as ``val_forces/forces_mae``);
    ``"total"`` -> ``None`` (logged as ``val/total``).
    """
    section: str | None = METRIC_SECTIONS.get(name)
    if section is not None:
        return section
    for prefix, by_prefix in _SECTION_BY_PREFIX:
        if name.startswith(prefix):
            return by_prefix
    return None


def _scatter_sum_batch(
    values: torch.Tensor,  # (N, ...)
    batch_index: torch.Tensor,  # (N,)
) -> torch.Tensor:
    """``scatter_sum`` per graph index — local helper to avoid a
    top-level import of ``torch_geometric``'s scatter for a single
    operation."""
    from torch_geometric.utils import scatter

    return scatter(values, batch_index, dim=0, reduce="sum")


def mlip_metrics(
    predictions: dict[str, torch.Tensor],
    batch: typing.Any,
) -> dict[str, torch.Tensor]:
    """Compute the standard MLIP diagnostic metrics for one batch.

    Parameters
    ----------
    predictions : dict
        Model output.  Must contain ``"energy"`` shape ``(G,)`` and
        ``"num_atoms"`` shape ``(G,)``.  ``"forces"`` shape
        ``(N, 3)`` is optional — when absent the force-side metrics
        are skipped.
    batch : AtomicGraph / pyg.Batch
        Reference labels.  Reads ``batch.energy`` (per graph) and
        ``batch.forces`` (per atom) when present.  ``batch.batch``
        is the per-atom graph index (PyG convention).

    Returns
    -------
    dict
        ``{metric_name: 0-d Tensor}``.  Metrics that can't be
        computed (e.g. forces requested but no labels available) are
        simply omitted from the dict.
    """
    out: dict[str, torch.Tensor] = {}

    # ---------- Energy: per-atom MAE / RMSE ----------
    pred_e: torch.Tensor | None = predictions.get("energy", None)
    num_atoms: torch.Tensor | None = predictions.get("num_atoms", None)
    target_e: torch.Tensor | None = getattr(batch, "energy", None)
    if pred_e is not None and target_e is not None and num_atoms is not None:
        # Flatten labels in case ``Batch`` returns ``(G, 1)``.
        t_e: torch.Tensor = target_e.to(pred_e.dtype).flatten()
        n_at: torch.Tensor = num_atoms.to(pred_e.dtype).clamp(min=1.0)
        per_atom_err: torch.Tensor = (pred_e - t_e) / n_at  # (G,)
        out["energy_mae_per_atom"] = per_atom_err.abs().mean()
        out["energy_rmse_per_atom"] = per_atom_err.pow(2).mean().sqrt()

    # ---------- Forces: MAE, RMSE, cos-sim, magnitude MAE ----------
    pred_f: torch.Tensor | None = predictions.get("forces", None)
    target_f: torch.Tensor | None = getattr(batch, "forces", None)
    if pred_f is not None and target_f is not None and pred_f.shape == target_f.shape:
        t_f: torch.Tensor = target_f.to(pred_f.dtype)
        err: torch.Tensor = pred_f - t_f  # (N, 3)
        out["forces_mae"] = err.abs().mean()
        out["forces_rmse"] = err.pow(2).mean().sqrt()

        # Per-atom cosine similarity in [-1, 1].  Uses
        # ``F.cosine_similarity`` for its well-behaved eps semantics
        # (denominator becomes ``max(‖p‖·‖t‖, eps)`` rather than
        # ``‖p‖·‖t‖ + eps``, so a near-zero numerator divided by a
        # near-zero magnitude product still produces a number close
        # to the true direction agreement instead of being squashed
        # to ~0).  Atoms whose reference force is *exactly* zero
        # (symmetric sites, or pure-energy datasets that fill
        # ``batch.forces`` with zeros) are dropped from the average:
        # they carry no directional information so including them
        # would silently bias the metric toward 0.  If every atom in
        # the batch has zero reference force the metric is omitted
        # entirely.
        norm_p: torch.Tensor = pred_f.norm(dim=-1)  # (N,)
        norm_t: torch.Tensor = t_f.norm(dim=-1)  # (N,)
        cos_sim: torch.Tensor = torch.nn.functional.cosine_similarity(
            pred_f, t_f, dim=-1, eps=1e-8
        )  # (N,) in [-1, 1]
        mask: torch.Tensor = norm_t > 0.0
        if bool(mask.any()):
            out["forces_cosine_similarity"] = cos_sim[mask].mean()

        out["forces_magnitude_mae"] = (norm_p - norm_t).abs().mean()

        # Per-element MAE (``forces_mae_O`` …): element-resolved errors are
        # where a compositional failure shows first (O on GMD).
        numbers: typing.Any = getattr(batch, "atomic_numbers", None)
        if numbers is not None and numbers.shape[0] == err.shape[0]:
            atom_err: torch.Tensor = err.abs().mean(dim=-1)  # (N,)
            for z in torch.unique(numbers).tolist():
                out[f"forces_mae_{chemical_symbols[z]}"] = atom_err[numbers == z].mean()

    # ---------- Newton violation (translational-invariance check) ----------
    #
    # The net force on an isolated molecule must vanish:
    # ``Σ_i F_i = 0``.  Reported **per Cartesian component** plus a
    # total, four numbers in all:
    #
    #   newton_violation_x = mean_m |Σ_i F_i,x|      (same for y, z)
    #   newton_violation   = _x + _y + _z            (exact sum)
    #
    # Splitting by axis is what makes the metric actionable: a single
    # number tells you translation invariance is broken, whereas three
    # tell you *where*.  An anisotropic pattern (e.g. z far worse than
    # x and y) points at a direction-dependent cause — a slab/surface
    # normal, a cell vector, a mis-set PBC flag, a direct force head
    # whose projection is unbalanced along one axis — while three
    # comparable values point at an isotropic cause such as an
    # unconverged direct head or plain numerical noise.
    #
    # Note the total is the **L1** sum of the components, not the
    # Euclidean norm ‖Σ F‖₂ this metric reported before the split, so
    # that the four logged numbers are exactly consistent
    # (total = x + y + z).  The two differ by at most √3, so absolute
    # scale and any threshold you had in mind are unaffected.
    #
    # ~0 for autograd-derived forces (the energy is translation
    # invariant by construction); meaningfully > 0 for direct heads.
    # Logged whether or not a newton loss is configured — it is a
    # diagnostic, not a training signal.
    batch_idx: typing.Any = getattr(batch, "batch", None)
    if pred_f is not None and batch_idx is not None:
        net_force_per_graph: torch.Tensor = _scatter_sum_batch(pred_f, batch_idx)  # (G, 3)
        per_component: torch.Tensor = net_force_per_graph.abs().mean(dim=0)  # (3,)
        for axis_idx, axis in enumerate(("x", "y", "z")):
            out[f"newton_violation_{axis}"] = per_component[axis_idx]
        out["newton_violation"] = per_component.sum()

    return out


# ----------------------------------------------------------------------
# Dataset-level evaluation metrics
# ----------------------------------------------------------------------
#
# Computed once over a whole evaluation set from per-structure
# predictions, never per batch: offsets per molecule and force tail
# quantiles do not average over batches.  The input is GOAL's evaluation
# format (``goal-eval predictions_file=…`` writes it; an external
# baseline exports the same keys so both are scored by the same code):
#
#   energy, energy_ref           (S,)    eV, total energy per structure
#   num_atoms                    (S,)    atoms per structure
#   atomic_numbers               (N,)    every atom, structures concatenated
#   forces, forces_ref           (N, 3)  eV/Å, same atom order

#: Keys of the evaluation format; ``forces``/``forces_ref`` are optional.
EVALUATION_KEYS: tuple[str, ...] = (
    "energy",
    "energy_ref",
    "num_atoms",
    "atomic_numbers",
    "forces",
    "forces_ref",
)


def _formula(numbers: np.ndarray) -> str:
    counts = collections.Counter(chemical_symbols[int(z)] for z in numbers)
    order = sorted(counts, key=lambda s: (s not in ("C", "H"), s != "C", s))  # Hill order
    return "".join(f"{s}{counts[s] if counts[s] > 1 else ''}" for s in order)


def evaluation_metrics(data: typing.Mapping[str, np.ndarray]) -> dict[str, typing.Any]:
    """Energy and force metrics over a whole evaluation set.

    Energies are reported per structure (eV) and per atom (eV/atom).
    With residual ``r_mj = E_pred - E_ref`` of conformer ``j`` of molecule
    ``m`` and offset ``δ_m = mean_j r_mj``:

    * ``energy_mae``: ``mean |r|`` (the GMD paper's total-energy MAE);
    * ``energy_centred_mae``: ``mean |r_mj - δ_m|``, the error left after
      removing each molecule's constant offset;
    * ``energy_offset_mae``: ``mean_m |δ_m|``, the offset part alone;
    * ``per_molecule``: ``δ_m`` (the mean signed error) per molecule with
      its carbon count, sorted by chain length.

    Forces: MAE, RMSE, cosine similarity, magnitude MAE, per-element MAE
    (``forces_mae_<symbol>``) and the tail of the per-component absolute
    error (``forces_q95``, ``forces_q99``, ``forces_max``).

    shortcut: molecules are grouped by composition, which is exact for GMD
    (one molecule per trajectory file); add an explicit id key if isomers
    ever share an evaluation set.
    """
    n_at = np.asarray(data["num_atoms"], dtype=float)
    numbers = np.asarray(data["atomic_numbers"])
    offsets = np.concatenate([[0], np.cumsum(data["num_atoms"])]).astype(int)
    out: dict[str, typing.Any] = {"num_structures": len(n_at)}

    r = np.asarray(data["energy"], dtype=float) - np.asarray(data["energy_ref"], dtype=float)
    molecules = [_formula(numbers[lo:hi]) for lo, hi in itertools.pairwise(offsets)]
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for s, m in enumerate(molecules):
        groups[m].append(s)
    delta = np.empty_like(r)
    rows: list[dict[str, typing.Any]] = []
    for m, idx in groups.items():
        delta[idx] = r[idx].mean()
        z = numbers[offsets[idx[0]] : offsets[idx[0] + 1]]
        rows.append(
            {
                "molecule": m,
                "num_carbon": int(np.count_nonzero(z == 6)),
                "num_atoms": len(z),
                "num_structures": len(idx),
                "mean_signed_error": float(r[idx].mean()),
                "mean_signed_error_per_atom": float(r[idx].mean() / len(z)),
            }
        )
    for suffix, scale in (("", 1.0), ("_per_atom", n_at)):
        out[f"energy_mae{suffix}"] = float(np.mean(np.abs(r / scale)))
        out[f"energy_rmse{suffix}"] = float(np.sqrt(np.mean((r / scale) ** 2)))
        out[f"energy_centred_mae{suffix}"] = float(np.mean(np.abs((r - delta) / scale)))
    out["energy_offset_mae"] = float(np.mean([abs(row["mean_signed_error"]) for row in rows]))
    out["energy_offset_mae_per_atom"] = float(
        np.mean([abs(row["mean_signed_error_per_atom"]) for row in rows])
    )
    out["per_molecule"] = sorted(rows, key=lambda row: (row["num_carbon"], row["molecule"]))

    if data.get("forces") is not None and data.get("forces_ref") is not None:
        f = np.asarray(data["forces"], dtype=float)
        t = np.asarray(data["forces_ref"], dtype=float)
        err = np.abs(f - t)
        out["forces_mae"] = float(err.mean())
        out["forces_rmse"] = float(np.sqrt(np.mean((f - t) ** 2)))
        out["forces_q95"], out["forces_q99"] = (float(q) for q in np.quantile(err, [0.95, 0.99]))
        out["forces_max"] = float(err.max())
        norm_f, norm_t = np.linalg.norm(f, axis=1), np.linalg.norm(t, axis=1)
        out["forces_magnitude_mae"] = float(np.mean(np.abs(norm_f - norm_t)))
        nonzero = norm_t > 0
        cos = np.sum(f * t, axis=1)[nonzero] / np.maximum(norm_f * norm_t, 1e-8)[nonzero]
        out["forces_cosine_similarity"] = float(cos.mean()) if nonzero.any() else float("nan")
        for z in np.unique(numbers):
            out[f"forces_mae_{chemical_symbols[int(z)]}"] = float(err[numbers == z].mean())
    return out
