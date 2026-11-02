"""``*_atomw``: exact epoch-level force metrics, next to the graph-weighted step metrics."""

from __future__ import annotations

import lightning as L
import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.loader import DataLoader

from goal.ml.data.graph import AtomicGraph
from goal.ml.training.loss import CompositeLoss, ForcesLoss, WeightedLoss
from goal.ml.training.module import GOALModule


class _Linear(nn.Module):
    """Monolithic toy: forces ``w · r``, energy the sum of positions."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        n = torch.bincount(graph.batch, minlength=graph.num_graphs).to(graph.pos.dtype)
        energy = torch.zeros(graph.num_graphs, dtype=graph.pos.dtype).index_add(
            0, graph.batch, self.w * graph.pos.sum(-1)
        )
        return {"energy": energy, "forces": self.w * graph.pos, "num_atoms": n}


def _frames() -> list[AtomicGraph]:
    g = torch.Generator().manual_seed(0)
    frames = []
    for n in (2, 3, 20, 25):  # sizes differ, so graph and atom weights disagree
        pos = torch.randn(n, 3, generator=g, dtype=torch.float64)
        frame = AtomicGraph(
            positions=pos,
            atomic_numbers=torch.ones(n, dtype=torch.long),
            cell=torch.zeros(3, 3),
            pbc=torch.zeros(3, dtype=torch.bool),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )
        frame.energy = torch.zeros(1, dtype=torch.float64)
        frame.forces = torch.randn(n, 3, generator=g, dtype=torch.float64)
        frames.append(frame)
    return frames


def test_atom_weighted_force_metrics_are_dataset_level() -> None:
    cfg = OmegaConf.create(
        {"training": {"ema": {"enabled": False}, "gradient_clip": 0.0, "optimizer": {"lr": 1e-3}}}
    )
    loss = CompositeLoss([WeightedLoss(ForcesLoss(loss_fn="mae"), 1.0, label="forces")])
    module = GOALModule(backbone=_Linear(), head=None, loss=loss, config=cfg)
    frames = _frames()
    seen: dict[str, float] = {}

    class _Monitor(L.Callback):  # where GOALCheckpointManager reads its metric
        def on_validation_epoch_end(self, trainer: L.Trainer, _: L.LightningModule) -> None:
            seen.update({k: float(v) for k, v in trainer.callback_metrics.items()})

    trainer = L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[_Monitor()],
    )
    trainer.validate(module, DataLoader(frames, batch_size=2), verbose=False)
    got = trainer.callback_metrics

    err = torch.cat([0.3 * f.pos - f.forces for f in frames])
    target = torch.cat([f.forces for f in frames])
    cos = nn.functional.cosine_similarity(err + target, target, dim=-1)
    assert abs(float(got["val_forces/forces_mae_atomw"]) - float(err.abs().mean())) < 1e-12
    assert (
        abs(float(got["val_forces/forces_rmse_atomw"]) - float(err.square().mean().sqrt())) < 1e-12
    )
    assert abs(float(got["val_forces/forces_cosine_similarity_atomw"]) - float(cos.mean())) < 1e-12
    assert seen["val_forces/forces_mae_atomw"] == float(got["val_forces/forces_mae_atomw"])
    # the existing metric keeps its name and its graph-weighted batch average
    assert abs(float(got["val_forces/forces_mae"]) - float(err.abs().mean())) > 1e-3
