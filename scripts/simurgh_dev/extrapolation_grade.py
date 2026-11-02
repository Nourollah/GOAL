#!/usr/bin/env python
"""Build or apply the per-atom extrapolation grade γ of a SIMURGH checkpoint (output only).

    python scripts/simurgh_dev/extrapolation_grade.py build CKPT [--subset 0.05]
    python scripts/simurgh_dev/extrapolation_grade.py score CKPT [--split test] [--out g.npz]

``build`` streams the run's training split three times (clusters, moments,
thresholds) and writes ``extrapolation_grade.pt`` next to the checkpoint.
``score`` grades a split and reports the flagged fraction and γ quantiles;
with force labels, also Spearman(γ_nbr, abs ΔF), ROC-AUC at fixed abs ΔF
thresholds and error quantiles per γ bin.  See
``goal.ml.nn.models.simurgh.dev.extrapolation`` (arXiv:2609.40060).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch


def _model_and_data(ckpt: str, device: str):
    from goal.ml.data.datamodule import GOALDataModule
    from goal.ml.utils.calculator import load_goal_checkpoint

    loaded = load_goal_checkpoint(ckpt, device=device)
    model = loaded.module.backbone
    model.config["diagnostics"]["calm"] = True
    for layer in model.atom_line.layers:
        layer.record_basis = True
    dm = GOALDataModule(loaded.config)
    dm.setup("fit")
    dm.setup("test")
    return model.eval(), dm, loaded


def _batches(loader, model, device: str):
    """``(z, numbers, out, batch)`` per batch, with z the grade's feature."""
    from goal.ml.nn.models.simurgh.dev.extrapolation import features, projection

    r = None
    for batch in loader:
        batch = batch.to(device)
        with torch.enable_grad():
            out = model(batch)
        basis = out["calm_basis"].detach().cpu()
        if r is None:
            r = projection(basis.shape[-1])
        z = features(basis, out["calm_layer_norms"].detach().cpu(), r)
        yield z, batch.atomic_numbers.cpu(), out, batch, r


def build(ckpt: str, device: str, subset: float) -> Path:
    from goal.ml.nn.models.simurgh.dev.extrapolation import GradeBuilder

    model, dm, _ = _model_and_data(ckpt, device)
    loader = dm.train_dataloader()
    zs, numbers, r = [], [], None
    for z, n, _, _, r in _batches(loader, model, device):  # pass 1 input (subset)
        keep = torch.rand(len(n)) < subset
        zs.append(z[keep])
        numbers.append(n[keep])
    builder = GradeBuilder(zs[0].shape[-1])
    builder.pass1(torch.cat(zs), torch.cat(numbers))
    for z, n, *_ in _batches(loader, model, device):
        builder.pass2(z, n)
    builder.finish_moments()
    for z, n, *_ in _batches(loader, model, device):
        builder.pass3(z, n)
    grade = builder.grade(r, seed=42)
    out = Path(ckpt) if Path(ckpt).is_dir() else Path(ckpt).parent
    path = out / "extrapolation_grade.pt"
    torch.save(grade, path)
    total = sum(int(g["count"].sum()) for g in grade["elements"].values())
    print(f"grade: {len(grade['elements'])} elements, {total} training atoms -> {path}")
    return path


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    """ROC-AUC by the rank-sum formula."""
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if not n_pos or not n_neg:
        return float("nan")
    ranks = np.argsort(np.argsort(score)) + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def score(ckpt: str, device: str, split: str, out: str | None) -> dict:
    from scipy.stats import spearmanr

    from goal.ml.nn.models.simurgh.dev.extrapolation import neighbourhood
    from goal.ml.nn.models.simurgh.dev.extrapolation import score as grade_score

    model, dm, loaded = _model_and_data(ckpt, device)
    base = Path(ckpt) if Path(ckpt).is_dir() else Path(ckpt).parent
    grade = torch.load(base / "extrapolation_grade.pt", weights_only=False)
    loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()
    loader = loader[0] if isinstance(loader, list) else loader
    gammas, nbrs, errors = [], [], []
    for z, n, pred, batch, _ in _batches(loader, model, device):
        gamma = grade_score(grade, z, n)
        lengths = (batch.pos[batch.edge_index[1]] - batch.pos[batch.edge_index[0]]).norm(dim=-1)
        nbrs.append(
            neighbourhood(gamma, batch.edge_index.cpu(), lengths.detach().cpu(), loaded.cutoff)
        )
        gammas.append(gamma)
        if getattr(batch, "forces", None) is not None:
            errors.append((pred["forces"] - batch.forces).detach().norm(dim=-1).cpu())
    gamma, nbr = torch.cat(gammas).numpy(), torch.cat(nbrs).numpy()
    report = {
        "flagged_fraction": float((gamma > 1).mean()),
        "gamma_quantiles": np.quantile(gamma, [0.5, 0.9, 0.99]).tolist(),
    }
    if errors:
        err = torch.cat(errors).numpy()
        report["spearman_nbr_abs_dF"] = float(spearmanr(nbr, err).statistic)
        report["roc_auc"] = {t: _auc(nbr, err > t) for t in (0.05, 0.1, 0.5, 1.0)}
        bins = [0, 0.5, 1, 1.5, 2, 5, np.inf]
        report["error_quantiles_by_gamma"] = {
            f"{lo}-{hi}": np.quantile(err[(nbr >= lo) & (nbr < hi)], [0.5, 0.95]).tolist()
            for lo, hi in itertools.pairwise(bins)
            if ((nbr >= lo) & (nbr < hi)).any()
        }
    if out:
        np.savez(out, gamma=gamma, gamma_nbr=nbr)
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "score"):
        p = sub.add_parser(name)
        p.add_argument("checkpoint")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub.choices["build"].add_argument("--subset", type=float, default=1.0, help="pass-1 fraction")
    sub.choices["score"].add_argument("--split", choices=["test", "val"], default="test")
    sub.choices["score"].add_argument("--out", default=None)
    args = parser.parse_args()
    if args.command == "build":
        build(args.checkpoint, args.device, args.subset)
    else:
        score(args.checkpoint, args.device, args.split, args.out)


if __name__ == "__main__":
    main()
