#!/usr/bin/env python
"""Score prediction files in GOAL's evaluation format with GOAL's metrics.

Every model is scored by the same code: ``goal-eval
predictions_file=<file>.npz`` writes the format for GOAL checkpoints, and
the external reference baseline's exporter writes the same keys
(``b_ref_predictions_<task>_<seed>.npz``).  See
:func:`goal.ml.training.metrics.evaluation_metrics` for the keys.

``--task`` adds the best published GMD in-distribution values
(arXiv:2605.08988v1, Table 7) next to the scores; pass an ID test set for
that comparison.

Example::

    python scripts/simurgh_dev/score_predictions.py fd_id.npz --task duplication --json out.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from goal.ml.training.metrics import evaluation_metrics

#: Best published GMD values: ID force MAE (meV/Å), ID energy MAE (meV per
#: molecule), best from-scratch OOD force MAE (meV/Å).
PUBLISHED_GMD: dict[str, dict[str, float]] = {
    "chain_extension": {"id_forces": 1.5, "id_energy": 1.6, "ood_forces": 4.0},
    "fusion": {"id_forces": 4.7, "id_energy": 3.6, "ood_forces": 88.9},
    "duplication": {"id_forces": 4.9, "id_energy": 2.9, "ood_forces": 39.0},
    "combination": {"id_forces": 5.3, "id_energy": 2.9, "ood_forces": 13.3},
}


def score(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return evaluation_metrics(dict(data))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", help="evaluation-format .npz files")
    parser.add_argument("--task", choices=sorted(PUBLISHED_GMD), help="GMD task for targets")
    parser.add_argument("--json", help="write every score (with per-molecule rows) here")
    args = parser.parse_args(argv)
    scores = {path: score(path) for path in args.files}
    for path, m in scores.items():
        print(path)
        for key, value in m.items():
            if not isinstance(value, list):
                print(f"  {key}: {value:.6g}")
        if args.task:
            target = PUBLISHED_GMD[args.task]
            print(
                f"  forces_mae {1e3 * m.get('forces_mae', np.nan):.2f} meV/Å "
                f"(published ID {target['id_forces']}); energy_mae {1e3 * m['energy_mae']:.2f} "
                f"meV/molecule (published ID {target['id_energy']})"
            )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(scores, fh, indent=2)


if __name__ == "__main__":
    main()
