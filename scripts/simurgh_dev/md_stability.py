#!/usr/bin/env python
"""Summarise MD stability and NVE energy drift.

* ``--batch DIR``: a ``goal-simulate-batch`` output directory (its
  ``stability.json``).  Reports the survival fraction, the instability
  onset times of the replicas that failed (median and interquartile
  range), and the mean temperature.
* ``--nve-log FILE``: the ASE MD log of an NVE run (``goal-simulate`` with
  ``velocity_verlet`` and a ``log_file``; columns ``Time[ps]``,
  ``Etot/N[eV]``, ...).  Reports the total-energy drift slope in
  meV/atom/ps from a least-squares line.

Protocol (MD17 screening): the original MD17 thermostat counted 3N
degrees of freedom instead of 3N - 6, so run at the effective temperature
``T_eff = 500 K * 3N / (3N - 6)`` (``--md17-temperature N`` prints it):
552.6 K for aspirin (N = 21) and 642.9 K for ethanol (N = 9).  Temperatures
reported here are ASE's ``get_temperature`` (3N degrees of freedom).

Example::

    goal-simulate-batch --checkpoint ckpt --structures aspirin_32.xyz \\
        --temperature 552.6 --timestep 0.5 --friction 0.01 --steps 200000 \\
        --output-dir outputs/md/aspirin
    python scripts/simurgh_dev/md_stability.py --batch outputs/md/aspirin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def effective_md17_temperature(num_atoms: int, nominal_K: float = 500.0) -> float:
    """MD17's 3N-degree-of-freedom thermostat target, expressed for 3N - 6."""
    return nominal_K * 3 * num_atoms / (3 * num_atoms - 6)


def survival_summary(stability: dict) -> dict[str, float]:
    """Survival fraction, onset median / IQR (ps, failed replicas) and mean T."""
    rows = stability["structures"]
    onsets = np.array([r["onset_ps"] for r in rows if r["onset_ps"] is not None], dtype=float)
    q25, q50, q75 = np.percentile(onsets, [25, 50, 75]) if onsets.size else (np.nan,) * 3
    return {
        "replicas": len(rows),
        "survival_fraction": 1.0 - onsets.size / len(rows),
        "run_ps": stability["steps"] * stability["timestep_fs"] / 1e3,
        "onset_median_ps": float(q50),
        "onset_iqr_ps": float(q75 - q25),
        "mean_temperature_K": float(np.nanmean([r["mean_temperature_K"] for r in rows])),
    }


def nve_drift(log_path: str | Path) -> float:
    """Total-energy drift slope (meV/atom/ps) of a per-atom ASE MD log."""
    header = Path(log_path).read_text().split("\n", 1)[0].split()
    if header[:2] != ["Time[ps]", "Etot/N[eV]"]:
        raise ValueError(f"{log_path}: expected a per-atom ASE MD log, got header {header}")
    data = np.loadtxt(log_path, skiprows=1, usecols=(0, 1))
    slope, _ = np.polyfit(data[:, 0], data[:, 1], 1)
    return float(1e3 * slope)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", nargs="*", default=[], help="goal-simulate-batch output dirs")
    parser.add_argument("--nve-log", nargs="*", default=[], help="ASE MD logs of NVE runs")
    parser.add_argument(
        "--md17-temperature", type=int, metavar="N", help="print T_eff for N atoms"
    )
    args = parser.parse_args(argv)
    if args.md17_temperature:
        print(
            f"T_eff({args.md17_temperature} atoms) = "
            f"{effective_md17_temperature(args.md17_temperature):.1f} K"
        )
    for d in args.batch:
        summary = survival_summary(json.loads((Path(d) / "stability.json").read_text()))
        print(f"{d}: " + ", ".join(f"{k}={v:.4g}" for k, v in summary.items()))
    for log in args.nve_log:
        print(f"{log}: drift {nve_drift(log):+.4g} meV/atom/ps")


if __name__ == "__main__":
    main()
