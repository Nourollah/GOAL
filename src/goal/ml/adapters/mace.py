"""MACE foundation model adapter.

Wraps a pre-trained MACE model as a GOAL backbone. Uses MACE's own
loader — no weight translation, no source copying. Requires the
``mace-torch`` package to be installed.

Supports loading from:
- Pre-trained MACE-MP variants (small/medium/large)
- Local fine-tuned checkpoints (.pt files)

Also satisfies ``FeatureExtractorBackbone`` — provides intermediate
layer feature extraction via ``extract_features()``.
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch
from e3nn.o3 import Irreps

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.registry import BACKBONE_REGISTRY


def _require_mace() -> None:
    """Check that mace-torch is installed, with a helpful error message."""
    try:
        import mace  # noqa: F401
    except ImportError:
        raise ImportError(
            "MACE adapter requires mace-torch.\n"
            "Install with: pixi add --pypi mace-torch\n"
            "Or: pip install goal[mace]"
        ) from None


class MACEAdapter:
    """Wraps a pre-trained MACE foundation model as a GOAL backbone.

    Satisfies the ``EquivariantBackbone`` protocol without inheriting
    from any base class.

    Parameters
    ----------
    mace_model : nn.Module
        A loaded MACE model instance.
    dtype : torch.dtype
        Compute precision.
    """

    def __init__(self, mace_model: typing.Any, dtype: torch.dtype = torch.float64) -> None:
        _require_mace()
        self._model: typing.Any = mace_model
        self.dtype: torch.dtype = dtype

    @classmethod
    def from_pretrained(cls, variant: str = "large") -> MACEAdapter:
        """Load a pre-trained MACE-MP model.

        Parameters
        ----------
        variant : str
            Model variant: ``'small'``, ``'medium'``, or ``'large'``.
        """
        _require_mace()
        from mace.calculators import mace_mp

        model: typing.Any = mace_mp(model=variant, return_raw_model=True)
        return cls(model)

    @classmethod
    def from_local(cls, checkpoint_path: typing.Union[str, Path], **kwargs: typing.Any) -> MACEAdapter:
        """Load a MACE model from a local checkpoint file.

        Use this to load fine-tuned MACE models saved as ``.pt`` files.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the ``.pt`` checkpoint file.
        **kwargs
            Additional keyword arguments passed to ``torch.load``.
        """
        _require_mace()
        path: Path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"MACE checkpoint not found: {path}")

        model: typing.Any = torch.load(str(path), map_location="cpu", weights_only=False, **kwargs)
        # Handle case where checkpoint is a state_dict rather than full model
        if isinstance(model, dict) and "model" in model:
            from mace.tools.scripts_utils import get_default_args
            # NOTE: Full model reconstruction from state_dict requires
            # MACE model config. This path is best-effort.
            # For reliable loading, save the full model object.
            raise NotImplementedError(
                "Loading from a state_dict is not yet supported. "
                "Please save the full MACE model using torch.save(model, path) "
                "rather than torch.save(model.state_dict(), path). "
                "This capability will be added in a future version."
            )
        return cls(model)

    @property
    def irreps_out(self):
        """Output irreducible representations from the MACE model."""
        return self._model.irreps_out

    @property
    def num_interactions(self) -> int:
        """Number of message-passing interactions in the MACE model."""
        return len(self._model.interactions)

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Translate AtomicGraph -> MACE input, run model, return NodeFeatures."""
        mace_batch: typing.Dict[str, typing.Any] = self._to_mace_batch(graph)
        out: typing.Dict[str, torch.Tensor] = self._model(mace_batch)
        return NodeFeatures(
            node_feats=out["node_feats"],
            irreps=str(self.irreps_out),
            node_energies=out.get("node_energy"),
        )

    def _to_mace_batch(self, graph: AtomicGraph) -> typing.Dict[str, typing.Any]:
        """Convert GOAL AtomicGraph to MACE's expected input format."""
        return {
            "positions": graph.positions,
            "node_attrs": graph.atomic_numbers,
            "edge_index": graph.edge_index,
            "shifts": graph.edge_vectors,
            "unit_shifts": torch.zeros_like(graph.edge_vectors),
            "cell": graph.cell,
            "batch": graph.batch,
            "ptr": graph.ptr,
        }

    def parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        """Proxy to underlying model parameters (for freezing)."""
        return self._model.parameters()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True,
    ) -> typing.Iterator[typing.Tuple[str, torch.nn.Parameter]]:
        """Proxy to underlying model named_parameters."""
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def requires(self) -> typing.List[str]:
        """Required pip packages for this adapter."""
        return ["mace-torch>=0.3"]

    # ------------------------------------------------------------------
    # FeatureExtractorBackbone interface
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        """Number of message-passing / interaction layers."""
        return len(self._model.interactions)

    def irreps_at_layer(self, layer: int) -> Irreps:
        """Irreducible representations of features at a given layer.

        MACE interaction blocks all share the same hidden irreps,
        so every layer returns the same value.

        Args:
            layer: 0-indexed interaction layer.

        Returns:
            ``e3nn.o3.Irreps`` for that layer's output.
        """
        if layer < 0:
            layer = self.num_layers + layer
        if layer < 0 or layer >= self.num_layers:
            raise IndexError(
                f"Layer {layer} out of range for model with "
                f"{self.num_layers} interaction layers."
            )
        return Irreps(str(self.irreps_out))

    def extract_features(
        self,
        graph: AtomicGraph,
        layer: typing.Union[int, str] = -1,
    ) -> NodeFeatures:
        """Extract MACE node features at any interaction layer.

        The features at each layer are equivariant tensors — they contain
        both scalar (L=0) and higher-order (L=1, L=2, L=3) components.
        Use ``goal.ml.utils.extraction.extract_scalars()`` to get invariant
        features for standard MLP downstream models.
        Use e3nn linear layers to preserve equivariance end-to-end.

        Args:
            graph: Input ``AtomicGraph``.
            layer: Which layer to extract from:
                   ``-1`` or ``'final'`` → last interaction block (default).
                   ``0, 1, ..., N-1`` → specific interaction block.
                   ``'all'`` → concatenation across all layers.

        Returns:
            ``NodeFeatures`` with the extracted features and corresponding
            irreps string.
        """
        from goal.ml.utils.extraction import HookBasedExtractor

        with HookBasedExtractor(
            self._model,
            blocks_attr="interactions",
            output_index=0,
            detach=not torch.is_grad_enabled(),
        ) as extractor:
            mace_batch: dict = self._to_mace_batch(graph)
            self._model(mace_batch)
            captured: typing.Dict[str, torch.Tensor] = extractor.captured

        n_layers: int = self.num_layers

        if layer == "all":
            parts: typing.List[torch.Tensor] = [
                captured[f"layer_{i}"] for i in range(n_layers)
            ]
            node_feats: torch.Tensor = torch.cat(parts, dim=-1)
            combined_irreps: str = "+".join(
                str(self.irreps_at_layer(i)) for i in range(n_layers)
            )
            return NodeFeatures(node_feats=node_feats, irreps=combined_irreps)

        resolved: int
        if layer == -1 or layer == "final":
            resolved = n_layers - 1
        elif isinstance(layer, int):
            resolved = layer if layer >= 0 else n_layers + layer
        else:
            raise ValueError(
                f"Invalid layer specifier: {layer!r}. "
                "Use an int, -1, 'final', or 'all'."
            )

        if resolved < 0 or resolved >= n_layers:
            raise IndexError(
                f"Layer {resolved} out of range for model with "
                f"{n_layers} interaction layers."
            )

        key: str = f"layer_{resolved}"
        return NodeFeatures(
            node_feats=captured[key],
            irreps=str(self.irreps_at_layer(resolved)),
        )
