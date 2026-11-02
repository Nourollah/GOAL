#!/usr/bin/env python
"""Time and memory of SIMURGH training steps on one GMD batch (report only).

Builds the model from a task config with backbone overrides, takes one
batch from the training set, and times forward + backward (and the
optimiser step) after warm-up.

Example::

    python scripts/simurgh_dev/benchmark.py --config simurgh_dev_gmd26_FD \\
        --variant channels=64 --variant channels=128 \\
        --variant mfc=energy --variant mfc=energy,core_force \\
        --variant "mfc=energy,core_force;mfc_every=2" \\
        --variant "mfc=energy,core_force;mfc_fraction=0.5"

``--profile`` also splits one step by component (CUDA events around module
forwards; the forces' autograd pass runs inside the model forward).
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
    core = overrides.pop("core", None)
    if core:
        backbone["core"] = core
    kernels = overrides.pop("kernels", None)
    if kernels:
        backbone["kernels"] = kernels
    objectives = overrides.pop("mfc", None)
    if objectives is not None:
        backbone["mfc"].update(enabled=objectives != "off", start_epoch=0)
        if objectives != "off":
            backbone["mfc"]["objective"] = objectives.split(",")
    for key in ("every_n_steps", "fraction"):
        value = overrides.pop(f"mfc_{key.split('_')[0]}", None)
        if value is not None:
            backbone["mfc"][key] = float(value) if key == "fraction" else int(value)
    import yaml

    for key in [k for k in overrides if "." in k]:  # any backbone key, e.g. readout.pgs=false
        *path, leaf = key.split(".")
        node = backbone
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(str(overrides.pop(key)))
    if overrides:
        raise ValueError(f"unknown variant keys {sorted(overrides)}")
    return SimurghDev(**backbone).to(device).train()


#: Component of each timed module, by the end of its qualified name.
COMPONENTS: dict[str, str] = {
    ".tp": "A-basis and tensor products",
    ".linear_up": "A-basis and tensor products",
    ".linear_a": "A-basis and tensor products",
    ".attention": "A-basis and tensor products",
    ".product": "product basis",
    ".linear_b": "product basis",
    ".radial": "radial network",
    "atom_line.rbf": "radial network",
    "atom_line.pair_code": "radial network",
    "atom_line.readouts": "readout",
    "atom_line.pgs": "readout",
    "pair_line": "pair line",
}


def profile(model, batch) -> dict[str, float]:
    """Milliseconds per component of one training step (CUDA only, MFC off)."""
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    open_calls: list[tuple[str, torch.cuda.Event]] = []
    handles = []
    for name, module in model.named_modules():
        component = next((c for k, c in COMPONENTS.items() if name.endswith(k)), None)
        if component is None or (component == "pair line" and name != "pair_line"):
            continue

        def pre(_m, _i, component=component) -> None:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            open_calls.append((component, start))

        def post(_m, _i, _o) -> None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events.append((*open_calls.pop(), end))

        handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    opt = model.build_optimizer({"name": "amsgrad", "lr": 1e-3})
    marks = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for warm in (True, False):
        events.clear()
        marks[0].record()
        out = model(batch)
        loss = (out["energy"] - batch.energy).square().mean() + out["forces"].square().mean()
        loss = loss + out.get("aux_loss", 0.0)
        marks[1].record()
        opt.zero_grad()
        loss.backward()
        marks[2].record()
        opt.step()
        marks[3].record()
        torch.cuda.synchronize()
    for h in handles:
        h.remove()
    result: dict[str, float] = {}
    for component, start, end in events:
        result[component] = result.get(component, 0.0) + start.elapsed_time(end)
    forward = marks[0].elapsed_time(marks[1])
    result["forces (autograd in forward) and the rest"] = forward - sum(result.values())
    result["loss backward"] = marks[1].elapsed_time(marks[2])
    result["optimiser step"] = marks[2].elapsed_time(marks[3])
    return result


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
    parser.add_argument("--profile", action="store_true", help="split a step by component")
    args = parser.parse_args(argv)
    with initialize_config_dir(config_dir=CONFIGS_ML_DIR, version_base=None):
        cfg = compose(config_name=args.config, overrides=[f"data.batch_size={args.batch_size}"])
    dm = GOALDataModule(cfg)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader())).to(args.device)
    stats = {"avg_num_neighbors": compute_avg_num_neighbors(dm.data_train), "scale": 1.0}
    stats["atom_statistics"] = {  # what goal-train would inject for the PAN gate
        "pan_rho": {"mean": 10.0, "std": 3.0},
        "pan_eta": {"mean": 0.3, "std": 0.1},
    }
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
        if args.profile and args.device.startswith("cuda") and model.mfc is None:
            for component, ms in profile(model, batch).items():
                print(f"    {component:42s} {ms:8.1f} ms")


if __name__ == "__main__":
    main()
