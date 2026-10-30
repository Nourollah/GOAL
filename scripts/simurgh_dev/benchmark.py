#!/usr/bin/env python
"""Time and memory of SIMURGH training steps on one GMD batch (report only).

Builds the model from a task config with backbone overrides, takes one
batch from the training set, and times forward + backward (and the
optimiser step) after warm-up.

Example::

    python scripts/simurgh_dev/benchmark.py --config simurgh_dev_gmd26_FD \\
        --variant channels=64 --variant channels=128 \\
        --variant mfc=energy --variant mfc=energy,core_force
"""

from __future__ import annotations

import argparse
import time

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def build(cfg, overrides: dict, stats: dict, device: str):
    """Model of ``cfg`` with ``overrides`` and dataset statistics injected."""
    from goal.ml.nn.models.simurgh.dev import SimurghDev

    backbone = OmegaConf.to_container(cfg.model.backbone, resolve=True)
    backbone.pop("name")
    backbone.update(stats)
    channels = overrides.pop("channels", None)
    if channels:
        backbone["num_channels"] = channels
        backbone["message_irreps"] = f"{channels}x0e + {channels}x1o + {channels}x2e"
    objectives = overrides.pop("mfc", None)
    if objectives is not None:
        backbone["mfc"].update(
            enabled=objectives != "off", start_epoch=0, objective=objectives.split(",")
        )
    return SimurghDev(**backbone).to(device).train()


def time_steps(model, batch, steps: int) -> tuple[float, float]:
    """Mean seconds per training step and peak memory (GiB) after warm-up."""
    opt = model.build_optimizer({"name": "amsgrad", "lr": 1e-3})
    cuda = batch.pos.is_cuda

    def step() -> None:
        out = model(batch)
        loss = (out["energy"] - batch.energy).square().mean() + out["forces"].square().mean()
        loss = loss + out.get("aux_loss", 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    if cuda:
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else float("nan")
    return (time.perf_counter() - start) / steps, peak


def main(argv: list[str] | None = None) -> None:
    from goal.ml.cli import CONFIGS_ML_DIR
    from goal.ml.data.datamodule import GOALDataModule
    from goal.ml.data.statistics import compute_avg_num_neighbors

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="simurgh_dev_gmd26_FD")
    parser.add_argument("--variant", action="append", default=[], help="k=v[;k=v] overrides")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    with initialize_config_dir(config_dir=CONFIGS_ML_DIR, version_base=None):
        cfg = compose(config_name=args.config, overrides=[f"data.batch_size={args.batch_size}"])
    dm = GOALDataModule(cfg)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader())).to(args.device)
    stats = {"avg_num_neighbors": compute_avg_num_neighbors(dm.data_train), "scale": 1.0}
    for variant in args.variant or ["channels=128"]:
        overrides: dict = {}
        for item in variant.split(";"):
            key, value = item.split("=")
            overrides[key] = int(value) if key == "channels" else value
        model = build(cfg, dict(overrides), stats, args.device)
        params = sum(p.numel() for p in model.parameters())
        seconds, peak = time_steps(model, batch, args.steps)
        print(
            f"{variant:32s} params {params:9d}  {1e3 * seconds:8.1f} ms/step  peak {peak:6.2f} GiB"
        )


if __name__ == "__main__":
    main()
