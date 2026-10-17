"""Trainable UMA / FairChem foundation-model backbone (``uma_finetune``).

Follows the same contract as :class:`goal.ml.nn.models.foundation.mace.MACEFinetune`:
load a pre-trained model as a raw ``nn.Module`` and expose it monolithically as
``{energy, forces, num_atoms}`` for the standard GOAL training loop.

Reality check (documented, not hidden): fairchem-core does **not** publicly
document loading a UMA model as a bare, fine-tunable ``nn.Module`` — the
supported entry points return an inference predictor / ASE calculator.  So
:meth:`_load_model` attempts the most likely documented path and, if it is not
available, raises :class:`UnsupportedOperationError` with a link to the
fairchem docs.  The rest of the wrapper (adapters, strategies, E0
re-estimation) is complete, so it works as soon as a raw-module load path is
available.

UMA uses **dataset-type embeddings**: organic molecules use ``omol``,
inorganic materials use ``omat``.  This is surfaced via the ``head`` config
key and injected in :meth:`_adapt_input` (field name marked for confirmation
against the installed fairchem version).

fairchem-core is imported lazily so this module imports (and the backbone
registers) even when fairchem is absent.
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

_FAIRCHEM_DOCS = "https://fair-chem.github.io/"

# UMA dataset-type embedding names.
_UMA_HEADS: frozenset[str] = frozenset({"omol", "omat", "omc", "odac", "oc20"})


@BACKBONE_REGISTRY.register("uma_finetune")
class UMAFinetune(FoundationModelBackbone):
    """Fine-tune a pre-trained UMA / FairChem model inside the GOAL loop.

    Parameters
    ----------
    checkpoint : str or None
        UMA model name (e.g. ``"uma-s-1"``) or a path to a checkpoint.
    model : nn.Module or None
        Pre-loaded model (bypasses ``_load_model``; mainly for tests).
    head : str
        UMA dataset-type embedding: ``"omol"`` (organic) / ``"omat"``
        (inorganic) / ….
    strategy, dtype, reestimate_e0s, lora_* :
        See :class:`FoundationModelBackbone`.
    """

    def __init__(
        self,
        checkpoint: str | None = "uma-s-1",
        model: nn.Module | None = None,
        head: str = "omol",
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
        if head not in _UMA_HEADS:
            log.warning(
                "[finetune] UMA head=%r is not a recognised dataset type %s; "
                "passing through unchanged.",
                head,
                sorted(_UMA_HEADS),
            )
        self.head_name: str = head
        self._elements: tuple[int, ...] = tuple(elements or ())

        loaded = model if model is not None else self._load_model(checkpoint, device)
        self._model = loaded.to(self.torch_dtype)
        self._apply_strategy()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint: str | None, device: str) -> nn.Module:
        """Load a UMA model as a raw ``nn.Module`` — or explain why we can't."""
        # Lazy import of the (heavy, dependency-free) shared exception type.
        from goal.md.calculators.base import UnsupportedOperationError

        try:
            import fairchem  # noqa: F401
        except ImportError as exc:
            raise UnsupportedOperationError(
                "UMA fine-tuning requires fairchem-core, which is not installed "
                "(pip install fairchem-core). Note that even when installed, "
                "loading a UMA model as a raw trainable nn.Module is not a "
                f"publicly documented fairchem API — see {_FAIRCHEM_DOCS}."
            ) from exc

        # Most likely documented path (fairchem-core v2): a pretrained predict
        # unit that exposes the underlying nn.Module. Wrapped defensively —
        # the public API returns an inference predictor, not a trainable model.
        try:
            from fairchem.core import pretrained_mlip  # type: ignore

            predictor = pretrained_mlip.get_predict_unit(str(checkpoint), device=device)
            raw = getattr(predictor, "model", None) or getattr(predictor, "module", None)
            if isinstance(raw, nn.Module):
                return raw
            raise AttributeError("predict unit does not expose a raw nn.Module")
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedOperationError(
                "Loading a UMA model as a raw, fine-tunable nn.Module is not "
                "supported by the installed fairchem-core version — its public "
                "API returns an inference predictor/calculator, not a trainable "
                f"module. See {_FAIRCHEM_DOCS} for the current fine-tuning entry "
                f"points. Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Backbone surface
    # ------------------------------------------------------------------

    @property
    def elements(self) -> tuple[int, ...]:
        return self._elements

    def _readout_name_fragments(self) -> tuple[str, ...]:
        # EquiformerV2/UMA energy+force heads are typically named `*_head`/`energy_block`.
        # Confirm against the installed fairchem version if head_only misbehaves.
        return ("output_head", "energy_head", "force_head", "_head")

    def set_atomic_energies(self, references: dict[int, float]) -> None:
        """Write per-element references into the model if it exposes them.

        UMA/EquiformerV2 handle references differently across versions; when no
        recognised buffer is present this is a no-op (logged), so E0
        re-estimation degrades gracefully rather than crashing.
        """
        for attr in ("atomic_energies", "energy_references", "references"):
            buf = getattr(self._model, attr, None)
            if isinstance(buf, torch.Tensor):
                with torch.no_grad():
                    for z, e in references.items():
                        if 0 <= int(z) < buf.shape[-1]:
                            buf[..., int(z)] = float(e)
                log.info("[finetune] UMA references written into `%s`.", attr)
                return
        log.warning(
            "[finetune] UMA model exposes no recognised atomic-energy buffer; "
            "E0 re-estimation skipped (fine-tuning will still run)."
        )

    # ------------------------------------------------------------------
    # Forward / adapters
    # ------------------------------------------------------------------

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Run UMA and adapt its output to GOAL's ``{energy, forces, num_atoms}``.

        The graph is converted to a FairChem-style batch dict (positions
        detached and cast to ``self.torch_dtype``, plus the per-graph
        ``dataset`` embedding name from ``head``), the wrapped model is
        evaluated under ``torch.enable_grad()``, and its ``energy`` /
        ``forces`` are cast back to ``graph.pos.dtype``.  Forces are
        whatever the upstream model returns; no autograd is done here.

        Parameters
        ----------
        graph : AtomicGraph
            Batched input graph; reads ``pos`` ``(N, 3)``, ``z`` ``(N,)``,
            ``edge_index`` ``(2, E)``, ``batch`` ``(N,)`` and optional
            ``cell``.

        Returns
        -------
        dict[str, Tensor]
            ``'energy'`` ``(B,)`` in eV, ``'forces'`` ``(N, 3)`` in eV/Å
            and ``'num_atoms'`` ``(B,)`` (float atom counts per graph).

        Raises
        ------
        KeyError
            If the upstream output lacks ``'energy'`` or ``'forces'``.
        """
        data = self._adapt_input(graph)
        with torch.enable_grad():
            raw = self._model(data)
        return self._adapt_output(raw, graph)

    def _adapt_input(self, graph: AtomicGraph) -> dict[str, typing.Any]:
        """Convert ``AtomicGraph`` to FairChem's batch format.

        NOTE: the exact key names and the dataset-embedding field must be
        confirmed against the installed fairchem-core version.  UMA takes a
        single per-graph dataset-type integer (organic→omol, inorganic→omat);
        it is attached under ``dataset`` here (adjust if your fairchem version
        differs).
        """
        device = graph.pos.device
        n_atoms = graph.pos.shape[0]
        batch = (
            graph.batch
            if getattr(graph, "batch", None) is not None
            else torch.zeros(n_atoms, dtype=torch.long, device=device)
        )
        num_graphs = int(batch.max().item()) + 1 if n_atoms > 0 else 1
        return {
            "pos": graph.pos.detach().to(self.torch_dtype),
            "atomic_numbers": graph.z.to(torch.long),
            "edge_index": graph.edge_index.to(torch.long),
            "cell": (
                graph.cell.to(self.torch_dtype)
                if getattr(graph, "cell", None) is not None
                else None
            ),
            "batch": batch,
            "natoms": scatter(
                torch.ones(n_atoms, dtype=torch.long, device=device), batch, dim=0, reduce="sum"
            ),
            # Dataset-type embedding (confirm field name against fairchem).
            "dataset": [self.head_name] * num_graphs,
        }

    def _adapt_output(
        self,
        raw: typing.Any,
        graph: AtomicGraph,
    ) -> dict[str, torch.Tensor]:
        get = raw.get if isinstance(raw, dict) else (lambda k, d=None: getattr(raw, k, d))
        energy = get("energy")
        forces = get("forces")
        if energy is None or forces is None:
            raise KeyError(
                "UMA output did not contain 'energy'/'forces'; adjust _adapt_output "
                f"for your fairchem version. Got keys: {list(raw) if isinstance(raw, dict) else type(raw)}."
            )
        target_dtype = graph.pos.dtype
        energy = energy.to(target_dtype).reshape(-1)
        forces = forces.to(target_dtype)
        device = graph.pos.device
        batch = (
            graph.batch
            if getattr(graph, "batch", None) is not None
            else torch.zeros(graph.pos.shape[0], dtype=torch.long, device=device)
        )
        num_atoms = scatter(
            torch.ones(graph.pos.shape[0], device=device, dtype=energy.dtype),
            batch,
            dim=0,
            reduce="sum",
        )
        return {"energy": energy, "forces": forces, "num_atoms": num_atoms}
