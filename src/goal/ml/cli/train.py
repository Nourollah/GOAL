"""Hydra-based training entry point with automatic SLURM resumption.

This is the main training command for GOAL. It:
1. Builds components via the registry system
2. Automatically resumes from ``last.ckpt`` if it exists
3. Continues a *finished* run when ``trainer.max_epochs`` is raised, and
   otherwise refuses to start — which is what keeps SLURM from requeuing
   a completed job (see :mod:`goal.ml.training.resume`)
4. Supports DDP, FSDP, ModelParallel strategies via trainer configs
5. Compatible with Lightning 2.6+ features (EMAWeightAveraging, torch.compile)
6. Performance-tuned: TF32 matmul, cuDNN benchmark, gradient accumulation
"""

from __future__ import annotations

import typing
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

from goal.ml.cli import CONFIGS_ML_DIR
from goal.ml.cli.builders import build_head as _build_head
from goal.ml.cli.builders import build_loss as _build_loss
from goal.ml.cli.builders import instantiate_callbacks as _instantiate_callbacks
from goal.ml.cli.builders import instantiate_loggers as _instantiate_loggers
from goal.ml.data.datamodule import GOALDataModule
from goal.ml.data.statistics import (
    ATOMIC_SYMBOLS,
    compute_atomic_references,
    compute_avg_num_neighbors,
    compute_energy_scale,
    compute_pair_counts,
    compute_unique_elements,
)
from goal.ml.nn.blocks.artisans import encode_pair_counts
from goal.ml.registry import BACKBONE_REGISTRY
from goal.ml.training import dataset_stats
from goal.ml.training.callbacks.checkpoint_manager import GOALCheckpointManager
from goal.ml.training.callbacks.logging import RichLoggingCallback
from goal.ml.training.loss import CompositeLoss
from goal.ml.training.module import GOALModule
from goal.ml.training.resume import (
    SENTINEL_NAME,
    LRScheduleRetarget,
    ResumeDecision,
    resolve_resume,
)
from goal.ml.training.strategies.factory import build_strategy


def _gpu_supports_tf32() -> bool:
    """Check whether the current CUDA device supports TF32 (Ampere+, sm_80+)."""
    if not torch.cuda.is_available():
        return False
    capability: tuple[int, int] = torch.cuda.get_device_capability()
    return capability[0] >= 8  # Ampere = 8.0, Hopper = 9.0


def _setup_performance(cfg: DictConfig) -> None:
    """Apply global PyTorch performance settings before training.

    Settings are applied conditionally based on GPU availability and
    hardware capability. TF32 and cuDNN options are silently skipped
    when running on CPU or pre-Ampere GPUs.

    - **TF32 matmul precision**: Uses TF32 tensor cores for float32 ops,
      giving ~3× speedup with negligible precision loss. Requires Ampere+
      (A100, H100, RTX 30xx/40xx).  Falls back to ``"highest"`` (full fp32)
      on older hardware.
    - **cuDNN benchmark**: Auto-tunes convolution algorithms for the
      given input sizes.  Adds startup overhead but faster thereafter.
      Only meaningful when CUDA is available.
    - **cuDNN deterministic**: Forces deterministic cuDNN algorithms.
      Only meaningful when CUDA is available.
    """
    perf: dict[str, typing.Any] = cfg.training.get("performance", {})
    has_cuda: bool = torch.cuda.is_available()

    # TF32 for matmul — only effective on Ampere+ GPUs (sm_80+)
    if _gpu_supports_tf32():
        matmul_precision: str = perf.get("float32_matmul_precision", "high")
    else:
        matmul_precision = perf.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(matmul_precision)

    # cuDNN benchmark — only effective when CUDA is present
    if has_cuda:
        torch.backends.cudnn.benchmark = perf.get("cudnn_benchmark", True)
        torch.backends.cudnn.deterministic = perf.get("cudnn_deterministic", False)


#: Backbone arguments ``_run_training`` may derive from the training set.
_DERIVED_BACKBONE_KEYS: tuple[str, ...] = (
    "elements",
    "atomic_energies",
    "scale",
    "avg_num_neighbors",
    "dressing_kwargs",
)


def _persist_derived_backbone_args(
    cfg: DictConfig,
    backbone_kwargs: dict[str, typing.Any],
) -> None:
    """Write dataset-derived backbone kwargs back into ``cfg.model.backbone``.

    Only keys whose value differs from the config's are written, so a
    config that already pins them is left byte-for-byte unchanged.
    """
    from omegaconf import OmegaConf

    backbone_cfg: typing.Any = cfg.model.backbone
    with open_dict(cfg):
        for key in _DERIVED_BACKBONE_KEYS:
            if key not in backbone_kwargs:
                continue
            value: typing.Any = backbone_kwargs[key]
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            current: typing.Any = backbone_cfg.get(key, None)
            if OmegaConf.is_config(current):
                current = OmegaConf.to_container(current, resolve=True)
            if current != value:
                backbone_cfg[key] = value


def _check_reused_stats(
    reused: list[str],
    stored_fingerprint: dict[str, typing.Any],
    current_fingerprint: dict[str, typing.Any],
    backbone_kwargs: dict[str, typing.Any],
    datamodule: GOALDataModule,
) -> None:
    """Report a changed training set; refuse elements the model cannot represent.

    Reused statistics describe the data the checkpoint was trained on.  When
    the current ``data:`` block or training-set size differs, the change is
    printed with the way to recompute.  The one hard failure is a dataset
    with elements outside the reused ``elements``: the model has no
    parameters for them, so training cannot continue on it.
    """
    if not stored_fingerprint:
        print(
            "[stats] the checkpoint's config predates dataset fingerprints — cannot check "
            "whether the training data changed since the statistics were computed."
        )
        return
    changes = dataset_stats.fingerprint_changes(stored_fingerprint, current_fingerprint)
    if not changes:
        return
    lines: str = "\n".join(f"    {k}: {old!r} -> {new!r}" for k, (old, new) in changes.items())
    print(
        "[stats] WARNING: the training data differs from the data the reused statistics "
        f"were computed on:\n{lines}\n"
        "  The model keeps the checkpoint's values, so it stays consistent with its "
        "weights.  If they do not suit the new data, recompute the ones you want, e.g.\n"
        "    'dataset_stats.recompute=[scale,avg_num_neighbors]'   (or dataset_stats.recompute=all)"
    )
    if "elements" in reused:
        missing: list[int] = dataset_stats.uncovered_elements(
            backbone_kwargs.get("elements"), datamodule.data_train
        )
        if missing:
            symbols: str = ", ".join(f"{ATOMIC_SYMBOLS.get(z, '?')}(Z={z})" for z in missing)
            raise ValueError(
                f"The training data contains {symbols}, which the resumed model has no "
                f"parameters for (its elements: {list(backbone_kwargs.get('elements') or [])}). "
                f"Continuing this checkpoint on that data is not possible. Either remove "
                f"those structures, or start a new model without resume.from "
                f"(dataset_stats.recompute=[elements] rebuilds the element tables, so the "
                f"checkpoint's weights would no longer fit)."
            )


def _prepare_resumed_module(
    module: GOALModule, resume: ResumeDecision, recompute: set[str]
) -> None:
    """Make recomputed statistics take effect on a resumed module.

    * Buffer statistics (``scale``, ``atomic_energies``) are restored from the
      checkpoint by Lightning; ``keep_fresh_buffers`` keeps the recomputed
      values instead.
    * Structural statistics (``elements``, ``pair_counts``) change parameter
      shapes; a mismatch is reported here, in plain terms, before Lightning
      fails on it.
    """
    module.keep_fresh_buffers = set(recompute & dataset_stats.BUFFER_STATS)
    if module.keep_fresh_buffers:
        print(
            f"[stats] {', '.join(sorted(module.keep_fresh_buffers))}: training with the "
            f"recomputed values instead of the checkpoint's."
        )
    if not recompute & dataset_stats.STRUCTURAL_STATS or resume.ckpt_path is None:
        return
    problems: list[str] = dataset_stats.structural_mismatches(module, resume.ckpt_path)
    if problems:
        shown: str = "\n".join(f"    {p}" for p in problems[:10])
        more: str = f"\n    … and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise ValueError(
            f"Recomputing {', '.join(sorted(recompute & dataset_stats.STRUCTURAL_STATS))} "
            f"changes the model's structure, so the weights in {resume.ckpt_path} no longer "
            f"fit:\n{shown}{more}\n"
            f"Keep the stored values (remove them from dataset_stats.recompute), or train a "
            f"new model without resume.from."
        )


def _run_training(cfg: DictConfig) -> None:
    """Core training logic — shared by train() and train_ml() entry points."""
    # Seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Apply global PyTorch performance settings
    _setup_performance(cfg)

    # Resume / extension is resolved before the datamodule so a launch
    # with nothing left to train exits without paying for dataset setup.
    checkpoint_dir: Path = Path(cfg.training.checkpoint_dir)
    resume: ResumeDecision = resolve_resume(cfg)
    if not resume.should_train:
        print(resume.stop_reason)
        return
    if resume.is_resume:
        verb: str = "Extending finished run" if resume.is_extension else "Resuming"
        print(
            f"[ckpt/resume] {verb} from {resume.ckpt_path} — "
            f"{resume.epochs_completed} epochs done, continuing to "
            f"{cfg.trainer.max_epochs}."
        )

    # Build datamodule first and prime it so we can compute
    # dataset-derived buffers (per-element atomic energies, scale/shift,
    # avg_num_neighbors) before constructing the model.  Lightning will
    # call ``setup("fit")`` again later — the second call is a no-op
    # because the datasets are already populated.
    datamodule: GOALDataModule = GOALDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Build components via registry
    backbone_cls: typing.Any = BACKBONE_REGISTRY.get(cfg.model.backbone.name)
    backbone_kwargs: dict[str, typing.Any] = {
        k: v for k, v in cfg.model.backbone.items() if k != "name"
    }

    # Dataset statistics recorded with the checkpoint being resumed.  They
    # are applied before anything is computed, so a resumed / extended run
    # rebuilds the model exactly as it was trained (and skips the scans).
    # ``dataset_stats.recompute`` names statistics to derive again anyway.
    stats_policy: typing.Any = cfg.get("dataset_stats") or {}
    recompute: set[str] = dataset_stats.normalise_recompute(stats_policy.get("recompute"))
    reused: list[str] = []
    stored_fingerprint: dict[str, typing.Any] = {}
    if resume.is_resume and bool(stats_policy.get("reuse", True)):
        saved_cfg: typing.Any = dataset_stats.load_stored(resume.source_dir, resume.ckpt_path)
        if saved_cfg is None:
            print(
                f"[stats] no saved config next to {resume.ckpt_path} — dataset statistics "
                f"are recomputed from the current training set."
            )
        else:
            stored, stored_fingerprint = dataset_stats.stored_values(saved_cfg)
            reused = dataset_stats.apply(backbone_kwargs, stored, skip=recompute)
            if reused:
                print(
                    f"[stats] reusing {', '.join(reused)} recorded with the checkpoint "
                    f"({resume.source_dir}); set dataset_stats.recompute to derive any of "
                    f"them from the current data instead."
                )

    # Auto-extract elements from the training set for SIMURGH-family backbones.
    # This replaces any manually-specified elements list in the config —
    # the dataset is always authoritative. Other backbones (hyperspec,
    # invariant_gnn) use a fixed-size embedding table and ignore this key.
    backbone_name: str = cfg.model.backbone.name
    if (
        backbone_name
        in (
            "simurgh",
            "simurgh_ace_first",
            "simurgh_arace",
            "simurgh_monolithic",
            "monolithic_arace",
        )
        and "elements" not in reused
    ):
        elements: list[int] = compute_unique_elements(datamodule.data_train)
        symbol_str: str = " ".join(f"{ATOMIC_SYMBOLS.get(z, '?')}({z})" for z in elements)
        print(f"[stats] elements discovered in training set: {symbol_str}")
        # Overwrite whatever the config said — dataset is authoritative.
        backbone_kwargs["elements"] = elements

    # Atomic-energy baseline + ScaleShift.  When the backbone exposes
    # an ``atomic_energies`` sub-config and its mode resolves to
    # ``"dataset"``, run the LSQ regression on the training set and
    # inject the values + scale.  ``"learned"`` and ``"provided"``
    # modes pass through unchanged.
    ae_cfg: dict[str, typing.Any] = dict(backbone_kwargs.get("atomic_energies") or {})
    mode: str = str(ae_cfg.get("mode", "learned"))
    if ae_cfg.get("compute_from_dataset", False) and mode == "learned":
        mode = "dataset"

    if mode == "dataset" and ae_cfg.get("values") is None:
        refs: dict[int, float] = compute_atomic_references(datamodule.data_train)
        ae_cfg["values"] = refs
        ae_cfg["mode"] = "dataset"
        backbone_kwargs["atomic_energies"] = ae_cfg
        print(
            "[stats] atomic_references "
            + ", ".join(f"Z={z}:{e:+.4f}" for z, e in sorted(refs.items()))
        )

    # ScaleShift gain — computed for both "dataset" and "provided" modes
    # so the interaction network starts at the right order of magnitude
    # regardless of where the baseline came from.  Skipped for "learned"
    # (no meaningful baseline yet) and when the user has already set a value.
    if mode in ("dataset", "provided") and backbone_kwargs.get("scale") is None:
        current_refs: dict[int, float] = {
            int(z): float(e) for z, e in (ae_cfg.get("values") or {}).items()
        }
        if current_refs:
            scale_val: float = compute_energy_scale(datamodule.data_train, current_refs)
            backbone_kwargs["scale"] = scale_val
            print(f"[stats] scale={scale_val:.6f}")

    # avg_num_neighbors — MACE-style sum-aggregation normaliser.
    dressing_cfg: typing.Any = backbone_kwargs.get("dressing_kwargs")
    if isinstance(dressing_cfg, (DictConfig, dict)):
        dressing_dict: dict[str, typing.Any] = dict(dressing_cfg)
        if dressing_dict.get("avg_num_neighbors") is None:
            avg_nn: float = compute_avg_num_neighbors(datamodule.data_train)
            dressing_dict["avg_num_neighbors"] = avg_nn
            backbone_kwargs["dressing_kwargs"] = dressing_dict
            print(f"[stats] avg_num_neighbors={avg_nn:.4f}")

    # avg_num_neighbors for the ARACE backbone — it has no dressing_kwargs
    # sub-config; the normaliser is a top-level backbone key instead.
    if backbone_name == "simurgh_arace" and backbone_kwargs.get("avg_num_neighbors") is None:
        avg_nn_arace: float = compute_avg_num_neighbors(datamodule.data_train)
        backbone_kwargs["avg_num_neighbors"] = avg_nn_arace
        print(f"[stats] avg_num_neighbors={avg_nn_arace:.4f}")

    # pair_counts — for data-driven expert routing.
    # Injected when a backbone declares rare_artisan.enabled=true.  The
    # two model families spell the artisan sub-config differently —
    # "artisan_config" on the legacy SIMURGH backbones, "artisan" on the
    # ARACE ones (simurgh_arace / monolithic_arace / nf_simurgh) — so both
    # keys are checked.  The routing layout inside is identical.
    for artisan_key in ("artisan_config", "artisan"):
        artisan_cfg_raw: typing.Any = backbone_kwargs.get(artisan_key)
        if not isinstance(artisan_cfg_raw, (DictConfig, dict)):
            continue
        ecfg: dict[str, typing.Any] = dict(artisan_cfg_raw)
        ge: dict[str, typing.Any] = dict(ecfg.get("rare_artisan") or {})
        if bool(ge.get("enabled", False)) and ecfg.get("pair_counts") is None:
            pc: dict[tuple[int, int], int] = compute_pair_counts(datamodule.data_train)
            total: int = max(1, sum(pc.values()))
            print(f"[stats] pair_counts computed ({len(pc)} pairs, {total} total directed edges)")
            # YAML-safe "Zlo_Zhi" keys, and written into BOTH the kwargs
            # and the config.  Routing decides which artisan modules
            # exist, so a checkpoint trained with it cannot load into a
            # model rebuilt from a config that has lost the counts — the
            # saved config.yaml has to carry them.
            encoded: dict[str, int] = encode_pair_counts(pc)
            ecfg["pair_counts"] = encoded
            backbone_kwargs[artisan_key] = ecfg
            with open_dict(cfg):
                cfg.model.backbone[artisan_key]["pair_counts"] = encoded

    # Persist every dataset-derived argument in the config the run saves
    # (hparams + checkpoints/config.yaml), as pair_counts already is.  A
    # model rebuilt from that config — GOALCalculator on a bare .ckpt,
    # goal-eval, goal-inspect — otherwise falls back to the defaults
    # (elements (1, 6, 7, 8), scale 1.0, no atomic references), and
    # anything sized from them (``artisan.n_elements: auto``) changes shape.
    _persist_derived_backbone_args(cfg, backbone_kwargs)

    # Which data the statistics describe, for the next resume to compare.
    n_train: int | None = len(datamodule.data_train) if datamodule.data_train is not None else None
    current_fingerprint: dict[str, typing.Any] = dataset_stats.dataset_fingerprint(cfg, n_train)
    if reused:
        _check_reused_stats(
            reused, stored_fingerprint, current_fingerprint, backbone_kwargs, datamodule
        )
        origin: str = f"reused from {resume.source_dir}"
        if recompute:
            origin += f"; recomputed: {', '.join(sorted(recompute))}"
        dataset_stats.record(
            cfg,
            backbone_kwargs,
            stored_fingerprint or current_fingerprint,
            origin=origin,
            trained_on=current_fingerprint,
        )
    else:
        dataset_stats.record(
            cfg, backbone_kwargs, current_fingerprint, origin="computed from the training set"
        )

    backbone: typing.Any = backbone_cls(**backbone_kwargs)

    head: typing.Any = _build_head(cfg, backbone)
    loss: CompositeLoss = _build_loss(cfg)

    module: GOALModule = GOALModule(
        backbone=backbone,
        head=head,
        loss=loss,
        config=cfg,
        compile_model=cfg.training.get("compile_model", False),
    )

    if resume.is_resume:
        _prepare_resumed_module(module, resume, recompute)

    callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="epoch"),
        RichLoggingCallback(),
        # NOTE: the stage-aware ``GOALRichProgressBar`` is registered
        # via the Hydra callback config (``configs/callbacks/
        # rich_progress_bar.yaml``) so the user has a single knob to
        # disable / replace it.  Don't re-add it here — Lightning
        # rejects multiple progress-bar callbacks.
    ]

    # Lightning restores LR-scheduler state wholesale, which silently
    # overrides a raised trainer.max_epochs — see goal.ml.training.resume.
    if resume.is_resume:
        callbacks.append(
            LRScheduleRetarget(
                policy=str((cfg.get("resume") or {}).get("lr_schedule", "stretch")),
                base_lr=float(cfg.training.optimizer.lr),
                epochs_completed=resume.epochs_completed,
            )
        )

    # Optionally add SLURM plugin
    plugins: list[typing.Any] = []
    if cfg.training.get("slurm_mode", False):
        from lightning.pytorch.plugins.environments import SLURMEnvironment

        plugins.append(SLURMEnvironment(auto_requeue=True))

    # Instantiate callbacks and loggers from Hydra config (if present)
    hydra_callbacks: list[Callback] = _instantiate_callbacks(cfg.get("callbacks"))
    loggers: list[Logger] = _instantiate_loggers(cfg.get("logger"))

    # Instantiate GOALCheckpointManager from the top-level checkpoint_manager block.
    # It lives outside cfg.callbacks deliberately (it is a first-class concern, not
    # an optional callback), so we handle it here explicitly.
    ckpt_manager_cfg = cfg.get("checkpoint_manager")
    if ckpt_manager_cfg is not None:
        ckpt_manager: GOALCheckpointManager = hydra.utils.instantiate(ckpt_manager_cfg)
        # Restore pool state if resuming.  The pools may name files in
        # another run's directory; the manager keeps them for ranking but
        # never deletes outside its own dirpath.
        if resume.is_resume and resume.source_dir is not None:
            GOALCheckpointManager.restore_pools_from_dir(ckpt_manager, str(resume.source_dir))
            print(
                f"[ckpt/resume] Pool state restored from "
                f"{resume.source_dir}/checkpoint_state.json"
            )
            if resume.is_extension:
                ckpt_manager.record_extension(
                    from_epoch=resume.epochs_completed,
                    to_epoch=int(cfg.trainer.max_epochs),
                    source_ckpt=str(resume.ckpt_path),
                )
        hydra_callbacks.append(ckpt_manager)

    # Guard: ModelCheckpoint + GOALCheckpointManager active at the same time
    # leads to double saves and conflicting deletion logic.
    from lightning.pytorch.callbacks import ModelCheckpoint

    has_model_checkpoint = any(isinstance(cb, ModelCheckpoint) for cb in hydra_callbacks)
    has_goal_manager = any(isinstance(cb, GOALCheckpointManager) for cb in hydra_callbacks)
    if has_model_checkpoint and has_goal_manager:
        raise ValueError(
            "Both ModelCheckpoint and GOALCheckpointManager are active simultaneously. "
            "They conflict: use one or the other.  For SIMURGH experiments use the "
            "top-level checkpoint_manager block.  For other models use ModelCheckpoint "
            "inside callbacks:."
        )

    # Merge callbacks: GOAL-specific + Hydra-configured
    all_callbacks: list[Callback] = callbacks + (hydra_callbacks or [])

    # Strategy: if cfg.strategy exists, use the strategy factory;
    # otherwise fall through to hydra.utils.instantiate (backward compat).
    strategy_override: dict[str, typing.Any] = {}
    if cfg.get("strategy") is not None:
        strategy_override["strategy"] = build_strategy(cfg)

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=all_callbacks,
        logger=loggers or True,
        plugins=plugins or None,
        **strategy_override,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume.ckpt_path)

    # Mark completion so SLURM doesn't requeue after success
    if trainer.is_global_zero:
        (checkpoint_dir / SENTINEL_NAME).touch()


@hydra.main(version_base=None, config_path=CONFIGS_ML_DIR, config_name="simurgh_gmd26")
def train_ml(cfg: DictConfig) -> None:
    """GOAL training entry point for self-contained src/goal/configs/ml/ experiment files.

    Usage:
        goal-train-ml                                  # loads simurgh_gmd26.yaml (default)
        goal-train-ml --config-name simurgh_md17        # loads simurgh_md17.yaml
        goal-train-ml --config-name hyperspec_md17     # loads hyperspec_md17.yaml

    All parameters live in a single file under src/goal/configs/ml/.
    Override any parameter from the CLI:
        goal-train-ml trainer.max_epochs=100
        goal-train-ml training.optimizer.lr=0.001

    Continue a run that already finished by raising the budget and
    pointing at its checkpoints:
        goal-train-ml trainer.max_epochs=2000 \
            resume.from=logs/simurgh_gmd26/runs/<old_run>/checkpoints
    """
    _run_training(cfg)


if __name__ == "__main__":
    train_ml()
