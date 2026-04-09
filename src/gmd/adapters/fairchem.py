"""FairChem (UMA / EquiformerV2) foundation model adapter.

Wraps pre-trained FairChem models as GMD backbones. Requires the
``fairchem-core`` package to be installed.

Supports loading from:
- Pre-trained FairChem model variants via OCPCalculator
- Local fine-tuned checkpoints
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from gmd.data.graph import AtomicGraph, NodeFeatures
from gmd.registry import BACKBONE_REGISTRY


def _require_fairchem() -> None:
    """Check that fairchem-core is installed, with a helpful error message."""
    try:
        import fairchem  # noqa: F401
    except ImportError:
        raise ImportError(
            "FairChem adapter requires fairchem-core.\n"
            "Install with: pixi add --pypi fairchem-core\n"
            "Or: pip install gmd[fairchem]"
        ) from None


class UMAAdapter:
    """Wraps a pre-trained FairChem UMA / EquiformerV2 model as a GMD backbone.

    Parameters
    ----------
    fairchem_model : nn.Module
        A loaded FairChem model instance.
    dtype : torch.dtype
        Compute precision.
    """

    def __init__(self, fairchem_model: typing.Any, dtype: torch.dtype = torch.float64) -> None:
        _require_fairchem()
        self._model: typing.Any = fairchem_model
        self.dtype: torch.dtype = dtype

    @classmethod
    def from_pretrained(cls, variant: str = "small") -> UMAAdapter:
        """Load a pre-trained FairChem model.

        Parameters
        ----------
        variant : str
            Model variant identifier.
        """
        _require_fairchem()
        from fairchem.core import OCPCalculator

        calc: typing.Any = OCPCalculator(model_name=variant, local_cache="/tmp/fairchem_cache")
        return cls(calc.trainer.model)

    @classmethod
    def from_local(cls, checkpoint_path: typing.Union[str, Path], **kwargs: typing.Any) -> UMAAdapter:
        """Load a FairChem model from a local checkpoint file.

        Use this to load fine-tuned FairChem/OCP models.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the checkpoint file (.pt).
        **kwargs
            Additional keyword arguments.
        """
        _require_fairchem()
        path: Path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"FairChem checkpoint not found: {path}")

        checkpoint: typing.Any = torch.load(str(path), map_location="cpu", weights_only=False)

        # FairChem checkpoints typically contain config + state_dict
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            # NOTE: Full model reconstruction from FairChem checkpoint
            # requires their registry and config system.
            # This is best-effort; for reliable loading use their Trainer.
            raise NotImplementedError(
                "Loading FairChem models from a config+state_dict checkpoint "
                "is not yet fully supported. This requires reconstructing the "
                "model architecture from the FairChem config registry. "
                "For now, please load via OCPCalculator or save the full model "
                "object. This capability will be added in a future version."
            )
        return cls(checkpoint)

    @property
    def irreps_out(self) -> typing.Any:
        """Output representation description."""
        if hasattr(self._model, "irreps_out"):
            return self._model.irreps_out
        return "fairchem_hidden"

    @property
    def num_interactions(self) -> int:
        """Number of message-passing layers."""
        if hasattr(self._model, "num_layers"):
            return self._model.num_layers
        return 0

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Translate AtomicGraph -> FairChem input, run model, return NodeFeatures."""
        batch: typing.Dict[str, typing.Any] = self._to_fairchem_batch(graph)
        out: typing.Any = self._model(batch)
        node_feats: typing.Any = out.get("node_feats", out.get("hidden_feats"))
        return NodeFeatures(
            node_feats=node_feats,
            irreps=str(self.irreps_out),
            node_energies=out.get("node_energy"),
        )

    def _to_fairchem_batch(self, graph: AtomicGraph) -> typing.Dict[str, typing.Any]:
        """Convert GMD AtomicGraph to FairChem's expected input format."""
        return {
            "pos": graph.positions,
            "atomic_numbers": graph.atomic_numbers,
            "edge_index": graph.edge_index,
            "cell": graph.cell,
            "batch": graph.batch,
            "natoms": graph.num_atoms,
        }

    def parameters(self) -> typing.Iterator[typing.Any]:
        """Proxy to underlying model parameters (for freezing)."""
        return self._model.parameters()

    def requires(self) -> typing.List[str]:
        """Required pip packages for this adapter."""
        return ["fairchem-core>=1.0"]
