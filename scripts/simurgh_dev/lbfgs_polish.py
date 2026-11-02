#!/usr/bin/env python
"""Float64 full-batch L-BFGS polish of a trained SIMURGH checkpoint (arXiv:2311.16326, App. C).

    python scripts/simurgh_dev/lbfgs_polish.py logs/<task>/runs/<run>/checkpoints \\
        [--kappa-f 500] [--l2-scope product_basis] [--max-iter 2000]

Objective (Eqs. C4-C5), MSE in eV and eV/A::

    kappa_E/N_s sum_n (E_n - E_n^ref)^2 + kappa_F/N_c sum |F - F^ref|^2 + kappa_L2 sum c^2

with N_c the number of force components and c the product-basis weights of
every layer (``--l2-scope all_non_embedding``: the wider arm).  Full batch,
accumulated exactly over fixed chunks; torch.optim.LBFGS, history 10, strong
Wolfe line search, at most 20 evaluations per iteration.  Validation force
MAE every 10 iterations keeps the best iterate; stop after 50 iterations
without improvement or at ``--max-iter``.  EMA, MFC windows, denoising copies
and Hessian rows are off.  Starts from the best EMA weights (``--from-init``:
the source-faithful arm P0, L-BFGS from the initialisation).

float64 runs at a small fraction of the float32 rate on consumer and
workstation GPUs: the script refuses a GPU whose name contains "RTX" unless
``--allow-rtx`` is given.  It writes ``lbfgs_polish.ckpt`` next to the source
checkpoint (its EMA shadow set to the polished weights).
"""

from __future__ import annotations

import argparse
import copy
import typing
from pathlib import Path

import torch


def l2_parameters(model: torch.nn.Module, scope: str) -> list[torch.nn.Parameter]:
    """Product-basis weights of every layer, or every non-embedding weight."""
    if scope == "product_basis":
        return [p for n, p in model.named_parameters() if ".product." in n]
    if scope == "all_non_embedding":
        return [p for n, p in model.named_parameters() if "embedding" not in n]
    raise ValueError(f"l2 scope must be product_basis or all_non_embedding, got {scope!r}")


def deterministic(model: torch.nn.Module) -> torch.nn.Module:
    """float64, training mode (forces stay differentiable), stochastic terms off."""
    model = model.double().train()
    backbone = getattr(model, "backbone", model)
    for attr in ("mfc", "denoise"):
        if getattr(backbone, attr, None) is not None:
            setattr(backbone, attr, None)
    if hasattr(backbone, "config") and "aux" in backbone.config:
        backbone.config["aux"] = {**backbone.config["aux"], "hessian_distill": False}
    return model


def objective(
    model: torch.nn.Module,
    chunks: typing.Sequence[typing.Any],
    kappa: tuple[float, float, float],
    l2: typing.Sequence[torch.nn.Parameter],
    backward: bool = True,
) -> float:
    """Full-batch loss; with ``backward``, its exact gradient accumulated over ``chunks``."""
    kappa_e, kappa_f, kappa_l2 = kappa
    n_s = sum(int(c.num_graphs) for c in chunks)
    n_c = sum(3 * int(c.num_nodes) for c in chunks)
    total = 0.0
    for chunk in chunks:
        out = model(chunk.clone())
        e = (out["energy"] - chunk.energy.reshape(-1).double()).square().sum()
        f = (out["forces"] - chunk.forces.double()).square().sum()
        loss = kappa_e * e / n_s + kappa_f * f / n_c
        if backward:
            loss.backward()
        total += float(loss.detach())
    penalty = kappa_l2 * sum(p.square().sum() for p in l2) if l2 else None
    if penalty is not None:
        if backward:
            penalty.backward()
        total += float(penalty.detach())
    return total


def force_mae(model: torch.nn.Module, chunks: typing.Sequence[typing.Any]) -> float:
    """Validation force MAE over all components (eV/A)."""
    err, count = 0.0, 0
    for chunk in chunks:
        out = model(chunk.clone())
        err += float((out["forces"].detach() - chunk.forces.double()).abs().sum())
        count += 3 * int(chunk.num_nodes)
    return err / max(count, 1)


def polish(
    model: torch.nn.Module,
    train: typing.Sequence[typing.Any],
    val: typing.Sequence[typing.Any],
    kappa: tuple[float, float, float] = (1.0, 500.0, 5e-5),
    l2_scope: str = "product_basis",
    max_iter: int = 2000,
    patience: int = 50,
    every: int = 10,
) -> list[float]:
    """Run the stage in place (best validation iterate restored); returns the losses."""
    model = deterministic(model)
    params = [p for p in model.parameters() if p.requires_grad]
    l2 = l2_parameters(model, l2_scope)
    if max_iter <= 0:
        return []
    opt = torch.optim.LBFGS(
        params, lr=1.0, max_iter=1, max_eval=20, history_size=10, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        return torch.tensor(objective(model, train, kappa, l2))

    best = (force_mae(model, val), copy.deepcopy(model.state_dict()))
    since, losses, it = 0, [], 0

    def check() -> bool:
        """Keep the iterate if validation is no worse; False once patience runs out."""
        nonlocal best, since
        mae = force_mae(model, val)
        if mae <= best[0]:
            best, since = (mae, copy.deepcopy(model.state_dict())), 0
            return True
        since += every
        return since < patience

    for it in range(1, max_iter + 1):
        before = [p.detach().clone() for p in params]
        losses.append(float(opt.step(closure)))
        if all(torch.equal(a, p) for a, p in zip(before, params)):
            break  # converged: the line search made no move
        if it % every == 0 and not check():
            break
    if it % every:
        check()  # the last iterate
    model.load_state_dict(best[1])
    return losses


def refuse_consumer_gpu(device: str, allow: bool) -> None:
    """float64 L-BFGS belongs on A100/H100/H200: refuse "RTX" GPUs unless overridden."""
    if device.startswith("cuda") and not allow:
        name = torch.cuda.get_device_name(torch.device(device))
        if "RTX" in name:
            raise SystemExit(
                f"{name}: float64 runs at a small fraction of the float32 rate; use a "
                "datacenter GPU or pass --allow-rtx"
            )


def main() -> None:
    from goal.ml.data.datamodule import GOALDataModule
    from goal.ml.utils.calculator import load_goal_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-rtx", action="store_true")
    parser.add_argument("--kappa-e", type=float, default=1.0)
    parser.add_argument("--kappa-f", type=float, default=500.0, help="1000 for benzene")
    parser.add_argument("--kappa-l2", type=float, default=5e-5)
    parser.add_argument("--l2-scope", default="product_basis")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--from-init", action="store_true", help="arm P0: start from init")
    args = parser.parse_args()
    refuse_consumer_gpu(args.device, args.allow_rtx)
    torch.set_default_dtype(torch.float64)
    loaded = load_goal_checkpoint(args.checkpoint, device=args.device, use_ema=True)
    module = loaded.module
    if args.from_init:
        for m in module.modules():
            if hasattr(m, "reset_parameters") and m is not module:
                m.reset_parameters()
    cfg = loaded.config
    cfg.data.batch_size = args.chunk_size
    dm = GOALDataModule(cfg)
    dm.setup("fit")

    def chunks(loader):
        return [b.to(args.device) for b in loader]

    val = dm.val_dataloader()
    val = val[0] if isinstance(val, list) else val
    train = chunks(dm.train_dataloader())  # built once: fixed chunks of the full batch
    losses = polish(
        module,
        train,
        chunks(val),
        (args.kappa_e, args.kappa_f, args.kappa_l2),
        args.l2_scope,
        args.max_iter,
    )
    source = Path(args.checkpoint)
    ckpt_file = (
        source if source.is_file() else next(iter(sorted(source.glob("best_*.ckpt"))), None)
    )
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False) if ckpt_file else {}
    state["state_dict"] = {k: v.cpu() for k, v in module.state_dict().items()}
    if "goal_ema" in state:
        state["goal_ema"] = {"shadow": [p.detach().cpu().clone() for p in module.parameters()]}
    out = (source if source.is_dir() else source.parent) / "lbfgs_polish.ckpt"
    torch.save(state, out)
    print(
        f"{len(losses)} iterations, final loss {losses[-1] if losses else float('nan'):.6g} -> {out}"
    )


if __name__ == "__main__":
    main()
