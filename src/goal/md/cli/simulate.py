"""goal-simulate CLI — run MD simulations from Hydra configs.

Usage examples
--------------
Run the default Langevin simulation::

    goal-simulate

Override molecule and output directory::

    goal-simulate molecule.smiles=CC output_dir=outputs/ethane

Run a pre-configured MTS simulation::

    goal-simulate --config-name mts/simulations/mts_respa

With a trained goal.ml model::

    goal-simulate calculator.checkpoint=outputs/train/run/last.ckpt
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

from goal.md.cli import CONFIGS_MD_DIR


@hydra.main(
    version_base="1.3",
    config_path=CONFIGS_MD_DIR,
    config_name="langevin_with_model_sim",
)
def simulate(cfg: DictConfig) -> None:
    """Hydra entrypoint for running MD simulations.

    All configuration is driven by Hydra.  Override any field from the
    command line using Hydra's standard ``key=value`` syntax.

    Parameters
    ----------
    cfg : DictConfig
        Hydra-resolved configuration for the simulation.
    """
    from goal.md.core.simulation import simulate_from_config

    log.info("Starting MD simulation")
    log.info("Molecule config: %s", dict(cfg.get("molecule", {})))
    log.info("Calculator config: %s", dict(cfg.get("calculator", {})))
    log.info("Dynamics config: %s", dict(cfg.get("dynamics", {})))

    result = simulate_from_config(cfg)

    log.info(
        "Simulation complete — %d steps, final energy=%.4f eV, final T=%.1f K",
        result.steps_completed,
        result.final_energy or float("nan"),
        result.final_temperature or float("nan"),
    )


@hydra.main(
    version_base="1.3",
    config_path=CONFIGS_MD_DIR,
    config_name="mts/simulations/mts_respa",
)
def simulate_mts(cfg: DictConfig) -> None:
    """Hydra entrypoint for multi-timescale MD simulations.

    Parameters
    ----------
    cfg : DictConfig
        Hydra-resolved MTS configuration.
    """
    from goal.md.mts.ipi_adapter import MTSSimulation

    log.info("Starting MTS simulation")
    log.info(
        "MTS ratio: %d, outer timestep: %.1f fs",
        cfg.get("mts", {}).get("mts_ratio", 4),
        cfg.get("mts", {}).get("timestep_fs", 1.0) * cfg.get("mts", {}).get("mts_ratio", 4),
    )

    sim = MTSSimulation.from_config(cfg)
    result = sim.run()

    log.info(
        "MTS complete — %d outer steps, final energy=%.4f eV, final T=%.1f K",
        result["steps_completed"],
        result.get("final_energy") or float("nan"),
        result.get("final_temperature") or float("nan"),
    )
    log.info("MTS stats: %s", result.get("mts_stats", {}))
