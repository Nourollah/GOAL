"""Trainable MACE foundation-model backbone (``mace_finetune``).

Loads a pre-trained MACE model as a raw ``nn.Module`` and exposes it to GOAL
as a monolithic backbone returning ``{energy, forces, num_atoms}``.  Ride the
existing :class:`goal.ml.training.module.GOALModule` with ``head: null``.

Input/output key names and the model call signature were taken from the MACE
source (``mace.data.atomic_data.AtomicData`` and
``mace.modules.models.ScaleShiftMACE.forward``), **not guessed**:

* ``forward(data, training=..., compute_force=True)`` where ``data`` carries
  ``positions (N,3)``, ``node_attrs (N, n_elements) one-hot``,
  ``edge_index (2,E)``, ``shifts (E,3)``, ``unit_shifts (E,3)``,
  ``cell (3·G, 3)``, ``batch (N,)``, ``ptr (G+1,)``.
* one-hot columns follow the model's ``atomic_numbers`` buffer (its z-table).
* the E0 baseline lives in ``atomic_energies_fn.atomic_energies``
  (shape ``[n_heads, n_elements]`` or ``[n_elements]``).
* output dict keys are ``"energy" (G,)`` and ``"forces" (N,3)``.

MACE (``mace-torch``) is imported lazily so this module still imports (and the
backbone still registers) in environments where MACE is not installed — it is
only required when a model is actually loaded.
"""

from __future__ import annotations

import logging
import typing

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.models.foundation.base import FoundationModelBackbone
from goal.ml.registry import BACKBONE_REGISTRY

log = logging.getLogger(__name__)

# Friendly names → mace_mp variant identifiers.
_MACE_MP_ALIASES: dict[str, str] = {
    "mace-mp-0-small": "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large": "large",
    "mace-mp-0": "medium",
    "mace-mp": "medium",
}


def atomicgraph_to_mace_batch(
    graph: AtomicGraph,
    z_list: typing.Sequence[int],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Convert a (possibly batched) ``AtomicGraph`` to MACE's input dict.

    Follows ``mace.data.atomic_data.AtomicData`` exactly: ``node_attrs`` is a
    one-hot over ``z_list`` (the model's z-table); ``shifts`` are the Cartesian
    per-edge shifts ``unit_shifts @ cell`` (zeros for non-periodic systems).

    Float tensors are cast to ``dtype`` (the model's precision); integer
    tensors (edge_index, batch, ptr) stay ``long``.
    """
    device = graph.pos.device
    n_atoms = graph.pos.shape[0]

    # --- node_attrs: one-hot over the model z-table ---
    z = graph.z.to(torch.long)
    z_table = torch.tensor(list(z_list), dtype=torch.long, device=device)
    if z_table.numel() == 0:
        raise ValueError(
            "MACE z-table is empty; cannot build node_attrs. Load a real MACE "
            "model or pass `elements`."
        )
    lut = torch.full((int(z_table.max().item()) + 1,), -1, dtype=torch.long, device=device)
    lut[z_table] = torch.arange(z_table.numel(), device=device)
    safe_z = torch.where(z <= lut.numel() - 1, z, torch.full_like(z, -1))
    col = torch.where(safe_z >= 0, lut[safe_z.clamp_min(0)], torch.full_like(z, -1))
    if bool((col < 0).any()):
        missing = sorted({int(v) for v, c in zip(z.tolist(), col.tolist()) if c < 0})
        raise ValueError(
            f"Structure contains element(s) Z={missing} absent from the MACE "
            f"model's z-table {sorted(int(x) for x in z_list)}. The foundation "
            f"model cannot represent them."
        )
    node_attrs = torch.zeros((n_atoms, z_table.numel()), dtype=dtype, device=device)
    node_attrs[torch.arange(n_atoms, device=device), col] = 1.0

    # --- batch / ptr ---
    batch = (
        graph.batch
        if getattr(graph, "batch", None) is not None
        else torch.zeros(n_atoms, dtype=torch.long, device=device)
    )
    num_graphs = int(batch.max().item()) + 1 if n_atoms > 0 else 1
    ptr = getattr(graph, "ptr", None)
    if ptr is None:
        ptr = torch.tensor([0, n_atoms], dtype=torch.long, device=device)

    # --- cell: (G, 3, 3) → (3·G, 3) as MACE batches it ---
    cell = getattr(graph, "cell", None)
    if cell is None:
        cell_ggg = torch.zeros((num_graphs, 3, 3), dtype=dtype, device=device)
    else:
        cell_ggg = cell.to(dtype).reshape(num_graphs, 3, 3)

    # --- shifts = unit_shifts @ cell(of the edge's graph) ---
    edge_index = graph.edge_index.to(torch.long)
    n_edges = edge_index.shape[1]
    unit_shifts = getattr(graph, "unit_shifts", None)
    if unit_shifts is None or unit_shifts.numel() == 0:
        unit_shifts = torch.zeros((n_edges, 3), dtype=dtype, device=device)
        shifts = torch.zeros((n_edges, 3), dtype=dtype, device=device)
    else:
        unit_shifts = unit_shifts.to(dtype)
        edge_cell = cell_ggg[batch[edge_index[0]]]  # (E, 3, 3)
        shifts = torch.bmm(unit_shifts.unsqueeze(1), edge_cell).squeeze(1)  # (E, 3)

    return {
        "positions": graph.pos.detach().to(dtype),
        "node_attrs": node_attrs,
        "edge_index": edge_index,
        "shifts": shifts,
        "unit_shifts": unit_shifts,
        "cell": cell_ggg.reshape(num_graphs * 3, 3),
        "batch": batch,
        "ptr": ptr,
    }


@BACKBONE_REGISTRY.register("mace_finetune")
class MACEFinetune(FoundationModelBackbone):
    """Fine-tune a pre-trained MACE model inside the GOAL training loop.

    Parameters
    ----------
    checkpoint : str or None
        A MACE-MP variant (``"small"``/``"medium"``/``"large"`` or
        ``"mace-mp-0-medium"``) or a path to a ``.model``/``.pt`` file.
    model : nn.Module or None
        A pre-loaded MACE model (bypasses ``checkpoint``).  Mainly for tests.
    strategy, dtype, reestimate_e0s, lora_rank, lora_alpha, lora_target_modules
        See :class:`FoundationModelBackbone`.
    elements : list[int] or None
        Fallback z-table when the model does not expose an ``atomic_numbers``
        buffer (e.g. a stub in tests).
    device : str
        Device passed to the MACE loader.
    """

    def __init__(
        self,
        checkpoint: str | None = "medium",
        model: nn.Module | None = None,
        strategy: str = "head_only",
        dtype: str | torch.dtype = "float64",
        reestimate_e0s: bool = True,
        lora_rank: int = 4,
        lora_alpha: float = 16.0,
        lora_target_modules: list[str] | None = None,
        elements: list[int] | None = None,
        device: str = "cpu",
        **_ignored: typing.Any,
    ) -> None:
        super().__init__(
            strategy=strategy,
            dtype=dtype,
            reestimate_e0s=reestimate_e0s,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
        )
        loaded = model if model is not None else self._load_model(checkpoint, device)
        self._model = loaded.to(self.torch_dtype)

        # Resolve the z-table (element order for one-hot node_attrs).
        z_buf = getattr(self._model, "atomic_numbers", None)
        if z_buf is not None:
            self._z_list: tuple[int, ...] = tuple(int(z) for z in z_buf.tolist())
        elif elements is not None:
            self._z_list = tuple(int(z) for z in elements)
        else:
            raise ValueError(
                "MACEFinetune could not determine the element z-table: the model "
                "has no `atomic_numbers` buffer and no `elements` were provided."
            )

        r_max = getattr(self._model, "r_max", None)
        self._cutoff: float = float(r_max) if r_max is not None else 0.0

        self._apply_strategy()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint: str | None, device: str) -> nn.Module:
        """Load a MACE model via ``mace_mp`` (name/path) → raw ``nn.Module``."""
        try:
            from mace.calculators.foundations_models import mace_mp
        except ImportError as exc:
            raise ImportError(
                "MACEFinetune requires mace-torch. Install into a standalone venv "
                "(mace pins e3nn==0.4.4, incompatible with GOAL's pixi env). See "
                "the 'MACE (mace-torch)' note in pyproject.toml and docs/finetuning/mace.md."
            ) from exc

        variant = _MACE_MP_ALIASES.get(str(checkpoint), str(checkpoint))
        model = mace_mp(
            model=variant,
            device=device,
            default_dtype=str(self.torch_dtype).replace("torch.", ""),
            return_raw_model=True,
        )
        return model

    # ------------------------------------------------------------------
    # Backbone surface
    # ------------------------------------------------------------------

    @property
    def cutoff(self) -> float:
        """Neighbour-list cutoff (MACE ``r_max``), in Å."""
        return self._cutoff

    @property
    def elements(self) -> tuple[int, ...]:
        """The model's z-table (element order used for one-hot node_attrs)."""
        return self._z_list

    def _readout_name_fragments(self) -> tuple[str, ...]:
        # MACE's per-layer energy heads live under `readouts.*`.
        return ("readouts",)

    # ------------------------------------------------------------------
    # E0 re-estimation
    # ------------------------------------------------------------------

    def set_atomic_energies(self, references: dict[int, float]) -> None:
        """Write ``{Z: e_Z}`` into ``atomic_energies_fn.atomic_energies`` in place.

        The buffer is indexed by the model's z-table; every head's column for a
        given element is set to the new reference (re-baselining to the target
        dataset).  Elements absent from the z-table are skipped with a warning.
        """
        ae_fn = getattr(self._model, "atomic_energies_fn", None)
        if ae_fn is None or not hasattr(ae_fn, "atomic_energies"):
            log.warning("[finetune] model has no atomic_energies buffer — skipping E0 update.")
            return
        buf: torch.Tensor = ae_fn.atomic_energies
        z_index = {z: i for i, z in enumerate(self._z_list)}
        with torch.no_grad():
            for z, e in references.items():
                idx = z_index.get(int(z))
                if idx is None:
                    log.warning("[finetune] Z=%d not in MACE z-table — E0 not updated for it.", z)
                    continue
                if buf.dim() == 1:
                    buf[idx] = float(e)
                else:  # (n_heads, n_elements)
                    buf[:, idx] = float(e)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Run MACE and adapt its output to GOAL's ``{energy, forces, num_atoms}``."""
        data = self._adapt_input(graph)
        with torch.enable_grad():
            raw = self._model(data, training=self.training, compute_force=True)
        return self._adapt_output(raw, graph)

    def _adapt_input(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        return atomicgraph_to_mace_batch(graph, self._z_list, self.torch_dtype)

    def _adapt_output(
        self,
        raw: dict[str, torch.Tensor],
        graph: AtomicGraph,
    ) -> dict[str, torch.Tensor]:
        energy = raw["energy"]
        forces = raw["forces"]
        # Cast predictions back to the target/label precision for the loss.
        target_dtype = graph.pos.dtype
        energy = energy.to(target_dtype)
        forces = forces.to(target_dtype)

        batch = (
            graph.batch
            if getattr(graph, "batch", None) is not None
            else torch.zeros(graph.pos.shape[0], dtype=torch.long, device=graph.pos.device)
        )
        num_atoms = scatter(
            torch.ones(graph.pos.shape[0], device=graph.pos.device, dtype=energy.dtype),
            batch,
            dim=0,
            reduce="sum",
        )
        return {"energy": energy, "forces": forces, "num_atoms": num_atoms}
