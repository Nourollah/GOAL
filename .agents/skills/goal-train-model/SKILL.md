---
name: goal-train-model
description: Trains GOAL interatomic potentials with goal-train (Hydra). Picks or copies an experiment config in src/goal/configs/ml, applies command-line overrides, runs smoke tests (+debug=fdr, overfit, limit), names, resumes and extends runs, runs multirun sweeps and hyperparameter searches, and launches multi-GPU DDP or SLURM jobs. Use when the user asks to train, retrain, resume, extend, sweep or scale a GOAL model, or a training run fails to start.
license: MIT
---

# Training a GOAL model

References: `docs/ml/training/training.md`, `docs/ml/reference/configuration.md`, `docs/ml/reference/cli.md`, `docs/ml/training/losses.md`, `docs/ml/training/resuming.md`, `docs/ml/training/distributed.md`, `docs/ml/training/hyperparameter-tuning.md`.

Prefix every command with `pixi run -e cuda` (NVIDIA), `-e rocm` (AMD) or `-e cpu`.

## 1. Choose the experiment file

One YAML file in `src/goal/configs/ml/` is one experiment. For new data, copy the closest one and edit `task_name`, `data.*` and `atomic_energies`:

```bash
ls src/goal/configs/ml/
cp src/goal/configs/ml/<closest>.yaml src/goal/configs/ml/my_system.yaml
pixi run -e cuda goal-train --config-name my_system --cfg job -p data             # check the data block
pixi run -e cuda goal-train --config-name my_system --cfg job -p model.backbone   # check the model block
```

## 2. Smoke-test before spending GPU hours

The debug presets live in `src/goal/configs/debug/`, outside goal-train's own config directory, hence `--config-dir` (without it, `+debug=fdr` only adds a key `debug: fdr` and changes nothing):

```bash
pixi run -e cuda goal-train --config-dir src/goal/configs --config-name my_system +debug=fdr        # 1 train/val/test step
pixi run -e cuda goal-train --config-dir src/goal/configs --config-name my_system +debug=overfit \
    trainer.accelerator=gpu                                                                         # loss must reach ~0
pixi run -e cuda goal-train --config-dir src/goal/configs --config-name my_system +debug=limit      # 3 short epochs
```

## 3. Run

```bash
# shakedown
pixi run -e cuda goal-train --config-name my_system trainer.max_epochs=30 \
    training.performance.float32_matmul_precision=high run_name_suffix=_shakedown logger.wandb.offline=true
# full run with a fixed name and seed
pixi run -e cuda goal-train --config-name my_system run_name=my_system_seed42 seed=42 \
    training.performance.float32_matmul_precision=high
```

Override syntax:
- Loss entries are a list: `training.losses.1.fn=huber training.losses.1.weight=10.0`; `training.stages=null` drops the curriculum.
- New keys need `+` (`'+data.test_paths=[...]'`); `~key` deletes one.
- Quote values that contain brackets, parentheses, `=` or spaces.

## 4. Seeds, sweeps and hyperparameter search

- Replicates: `pixi run -e cuda goal-train --config-name my_system -m seed=42,43,44`.
- Grids: `-m training.optimizer.lr=2e-3,5e-3,1e-2 seed=42,43`.
- `goal-tune` runs the Lightning LR finder, Ray Tune or a W&B sweep: `--config-dir src/goal/configs --config-name <experiment> +hparams_search@_global_=basic` (or `ray_tune`, `wandb_sweep`).

## 5. Resume and extend

```bash
pixi run -e cuda goal-train --config-name <experiment> ++resume.from=logs/<task>/runs/<old_run>     # interrupted run
pixi run -e cuda goal-train --config-name <experiment> trainer.max_epochs=3000 \
    ++resume.from=logs/<task>/runs/<old_run>/checkpoints                                          # extend a finished run
pixi run -e cuda goal-train --config-name <experiment> trainer.max_epochs=3000 \
    ++resume.from=logs/<task>/runs/<old_run>/checkpoints ++resume.lr_schedule=restart             # fresh cosine cycle
```

`++` sets the key whether or not the config has a `resume:` block; the `simurgh_dev` configs have none, so a plain `resume.from=...` fails there with "Key 'resume' is not in struct". `resume.from` takes a run directory, a checkpoint directory or a `.ckpt` file; `resume.lr_schedule` is `stretch` (default), `restart` or `keep`.

## 6. Several GPUs and clusters

- One node: `++strategy.name=ddp trainer.devices=4` (`++` because the `simurgh_dev` configs have no top-level `strategy:` block). In a config that has a `strategy:` block, `trainer.strategy=ddp` is silently ignored.
- Several nodes: add `trainer.num_nodes=2`. The SLURM job template is in `docs/ml/training/training.md` (`srun pixi run -e cuda python -m goal.ml.cli.train ... training.slurm_mode=true`).
- Pin a GPU on a shared machine with `CUDA_VISIBLE_DEVICES=1`. Remote machines: `goal-remote-training`.

## 7. Precision

- Develop in float32 with `training.performance.float32_matmul_precision=high`.
- Consumer and workstation GPUs (RTX 5090, RTX 6000 Ada) run float64 at about 1/64 of their float32 rate. Train in float64 only on datacenter GPUs (A100, H100, H200).

## Pitfalls

- Monolithic models need `model.head: null` and modular ones need a head; `goal-train` raises on a mismatch.
- Forces come from autograd, so evaluation needs `trainer.inference_mode: false` (the shipped configs set it).
- Dataset-derived constructor values (`elements`, element energies, `scale`, `avg_num_neighbors`) are injected only for specific backbones: see "Statistics derived from the data" in `docs/ml/data/loading.md`. The code in `src/goal/ml/cli/train.py` is authoritative; it also injects `elements`, `avg_num_neighbors` (when null) and `scale: force_rms` for `simurgh_dev`, which that page does not list yet.
- Test-set losses use the base loss weights, so they are not comparable with validation losses of a staged run; the metrics are comparable.
- Never use OOD test numbers for checkpoint selection, early stopping or hyperparameter choice.

## Outputs

`logs/<task_name>/runs/<run>/checkpoints/` holds the best and interval `.ckpt` files with JSON sidecars, `last.ckpt`, `config.yaml`, `metadata.json` and `frozen_source/`; at the end of training a `.simurgh` archive is packed (`docs/ml/deployment/archives.md`).

## Done when

- [ ] `+debug=fdr` and `+debug=overfit` pass for the config.
- [ ] The composed config shows the intended data, cutoff, element references and loss.
- [ ] The run directory contains `config.yaml` and checkpoints, and metrics appear in the logger.
