"""Hyperparameter tuning utilities — Lightning Tuner and Ray Tune.

Two levels of tuning are provided:

1. **Basic** — Lightning's built-in ``Tuner`` for learning rate and
   batch size auto-discovery.  Zero extra dependencies.
2. **Advanced** — Ray Tune with ASHA / Population-Based Training
   schedulers and Optuna / HyperOpt search algorithms.  Requires
   ``ray[tune]`` (optional dependency).

Both are fully driven by Hydra config under the ``hparams_search``
config group.  They are **not imported or activated** unless the user
explicitly selects a tuning config — invisible to normal users.
"""

from __future__ import annotations

import typing
from pathlib import Path

import lightning as L
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Basic: Lightning Tuner
# ---------------------------------------------------------------------------


def run_lightning_tuner(
    module: L.LightningModule,
    datamodule: L.LightningDataModule,
    trainer: L.Trainer,
    cfg: DictConfig,
) -> typing.Dict[str, typing.Any]:
    """Run Lightning's built-in Tuner for LR and batch size discovery.

    Parameters
    ----------
    module : LightningModule
        The model to tune.
    datamodule : LightningDataModule
        The data module providing train/val loaders.
    trainer : Trainer
        A pre-configured Lightning Trainer.
    cfg : DictConfig
        Tuning config (``cfg.hparams_search``).

    Returns
    -------
    dict
        Results with keys ``lr`` and/or ``batch_size`` if found.
    """
    tuner_cfg: typing.Dict[str, typing.Any] = dict(cfg.hparams_search.get("tuner", {}))
    tuner: L.pytorch.tuner.Tuner = L.pytorch.tuner.Tuner(trainer)

    results: typing.Dict[str, typing.Any] = {}

    if tuner_cfg.get("lr_find", False):
        lr_finder = tuner.lr_find(
            module,
            datamodule=datamodule,
            min_lr=tuner_cfg.get("min_lr", 1e-8),
            max_lr=tuner_cfg.get("max_lr", 1.0),
            num_training=tuner_cfg.get("num_training_steps", 100),
            mode=tuner_cfg.get("lr_find_mode", "exponential"),
        )
        if lr_finder is not None:
            results["lr"] = lr_finder.suggestion()
            if tuner_cfg.get("auto_apply", True):
                module.hparams.lr = results["lr"]

    if tuner_cfg.get("scale_batch_size", False):
        optimal_batch = tuner.scale_batch_size(
            module,
            datamodule=datamodule,
            mode=tuner_cfg.get("batch_size_mode", "power"),
            steps_per_trial=tuner_cfg.get("steps_per_trial", 3),
            init_val=tuner_cfg.get("init_batch_size", 2),
            max_trials=tuner_cfg.get("max_trials", 25),
        )
        if optimal_batch is not None:
            results["batch_size"] = optimal_batch

    return results


# ---------------------------------------------------------------------------
# Advanced: Ray Tune
# ---------------------------------------------------------------------------


def _require_ray_tune() -> None:
    """Guard import — raises clear error if ray[tune] is missing."""
    try:
        import ray.tune  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Ray Tune is required for advanced hyperparameter search.\n"
            "Install with: pip install 'ray[tune]' optuna\n"
            "Or:           pip install -e '.[tune]'"
        ) from exc


def _build_search_space(space_cfg: DictConfig) -> typing.Dict[str, typing.Any]:
    """Convert Hydra config to Ray Tune search space.

    Supports the standard Ray Tune primitives via config strings:

    .. code-block:: yaml

        search_space:
          training.optimizer.lr:
            type: loguniform
            lower: 1.0e-5
            upper: 1.0e-2
          training.losses.0.weight:
            type: uniform
            lower: 1.0
            upper: 20.0
          model.backbone.num_interactions:
            type: choice
            values: [2, 3, 4, 5]
    """
    from ray import tune

    _SPACE_BUILDERS: typing.Dict[str, typing.Callable[..., typing.Any]] = {
        "uniform": lambda c: tune.uniform(c.lower, c.upper),
        "loguniform": lambda c: tune.loguniform(c.lower, c.upper),
        "quniform": lambda c: tune.quniform(c.lower, c.upper, c.get("q", 1)),
        "choice": lambda c: tune.choice(list(c.values)),
        "randint": lambda c: tune.randint(c.lower, c.upper),
        "grid": lambda c: tune.grid_search(list(c.values)),
    }

    space: typing.Dict[str, typing.Any] = {}
    for param_name, param_cfg in space_cfg.items():
        builder = _SPACE_BUILDERS.get(param_cfg.type)
        if builder is None:
            available = ", ".join(sorted(_SPACE_BUILDERS))
            raise ValueError(
                f"Unknown search space type '{param_cfg.type}' for "
                f"'{param_name}'. Available: {available}"
            )
        space[param_name] = builder(param_cfg)
    return space


def run_ray_tune(
    train_fn: typing.Callable[..., None],
    cfg: DictConfig,
) -> typing.Any:
    """Run Ray Tune hyperparameter search.

    Parameters
    ----------
    train_fn : callable
        A function that accepts a single ``config`` dict and runs one
        training trial.  Typically wraps ``gmd.cli.train.train``.
    cfg : DictConfig
        Full Hydra config with ``hparams_search`` section.

    Returns
    -------
    ray.tune.ResultGrid
        The results of the search.
    """
    _require_ray_tune()

    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

    hp_cfg: DictConfig = cfg.hparams_search

    # Search space
    search_space: typing.Dict[str, typing.Any] = _build_search_space(
        hp_cfg.search_space
    )

    # Scheduler
    scheduler_name: str = hp_cfg.get("scheduler", "asha")
    if scheduler_name == "asha":
        scheduler = ASHAScheduler(
            max_t=hp_cfg.get("max_epochs", cfg.training.get("max_epochs", 500)),
            grace_period=hp_cfg.get("grace_period", 10),
            reduction_factor=hp_cfg.get("reduction_factor", 3),
        )
    elif scheduler_name == "pbt":
        scheduler = PopulationBasedTraining(
            perturbation_interval=hp_cfg.get("perturbation_interval", 5),
            hyperparam_mutations=search_space,
        )
    else:
        raise ValueError(
            f"Unknown scheduler '{scheduler_name}'. Use 'asha' or 'pbt'."
        )

    # Search algorithm (optional)
    search_alg: typing.Any = None
    search_alg_name: typing.Optional[str] = hp_cfg.get("search_algorithm")
    if search_alg_name == "optuna":
        from ray.tune.search.optuna import OptunaSearch
        search_alg = OptunaSearch()
    elif search_alg_name == "hyperopt":
        from ray.tune.search.hyperopt import HyperOptSearch
        search_alg = HyperOptSearch()
    elif search_alg_name is not None:
        raise ValueError(
            f"Unknown search algorithm '{search_alg_name}'. "
            "Use 'optuna', 'hyperopt', or omit for random search."
        )

    # Run directory
    storage_path: str = hp_cfg.get(
        "storage_path",
        str(Path(cfg.training.checkpoint_dir).parent / "ray_tune"),
    )

    tuner = tune.Tuner(
        tune.with_parameters(train_fn, base_cfg=cfg),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=hp_cfg.get("metric", "val/total"),
            mode=hp_cfg.get("mode", "min"),
            num_samples=hp_cfg.get("num_samples", 10),
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=tune.RunConfig(
            name=hp_cfg.get("experiment_name", "gmd_tune"),
            storage_path=storage_path,
        ),
    )
    return tuner.fit()


# ---------------------------------------------------------------------------
# Advanced: W&B Sweeps
# ---------------------------------------------------------------------------


def _build_wandb_sweep_config(hp_cfg: DictConfig) -> typing.Dict[str, typing.Any]:
    """Convert Hydra ``hparams_search`` config to W&B sweep config dict.

    Maps the GMD YAML schema to the W&B sweep spec
    (https://docs.wandb.ai/guides/sweeps/define-sweep-configuration).
    """
    sweep_config: typing.Dict[str, typing.Any] = {
        "method": hp_cfg.get("sweep_method", "bayes"),
        "metric": {
            "name": hp_cfg.get("metric", "val/total"),
            "goal": "minimize" if hp_cfg.get("mode", "min") == "min" else "maximize",
        },
    }

    # Parameters — nested dotted keys flattened for W&B
    parameters: typing.Dict[str, typing.Any] = {}
    for param_name, param_cfg in hp_cfg.parameters.items():
        param_spec: typing.Dict[str, typing.Any] = dict(param_cfg)
        parameters[param_name] = param_spec
    sweep_config["parameters"] = parameters

    # Early termination (optional)
    et_cfg = hp_cfg.get("early_terminate")
    if et_cfg is not None:
        sweep_config["early_terminate"] = dict(et_cfg)

    return sweep_config


def run_wandb_sweep(
    train_fn: typing.Callable[[typing.Dict[str, typing.Any]], None],
    cfg: DictConfig,
) -> str:
    """Create (or resume) a W&B sweep and run the agent.

    Parameters
    ----------
    train_fn : callable
        A function that accepts a ``config`` dict (sampled hyperparams)
        and runs one training trial.
    cfg : DictConfig
        Full Hydra config with ``hparams_search`` section.

    Returns
    -------
    str
        The W&B sweep ID.
    """
    import wandb

    hp_cfg: DictConfig = cfg.hparams_search

    project: str = hp_cfg.get("project", "gmd")
    entity: typing.Optional[str] = hp_cfg.get("entity")
    sweep_id: typing.Optional[str] = hp_cfg.get("sweep_id")

    if sweep_id is None:
        sweep_config: typing.Dict[str, typing.Any] = _build_wandb_sweep_config(hp_cfg)
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)

    count: typing.Optional[int] = hp_cfg.get("count", 20)

    wandb.agent(
        sweep_id,
        function=train_fn,
        project=project,
        entity=entity,
        count=count,
    )
    return sweep_id
