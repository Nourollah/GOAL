# Pixi tasks

This page lists every task defined in `[tool.pixi.tasks]` in `pyproject.toml`, the exact command each one runs, and when to use it. It assumes you installed GOAL with pixi (see [Installation](../../getting-started/installation.md#option-1-pixi-from-a-clone)).

Run a task with `pixi run -e <env> <task> [extra args]`, for example `pixi run -e cuda train`. On other hardware swap `cuda` for `cpu` or `rocm`, and use the `-full` variants for the MD module (see [Environments](../../getting-started/installation.md#environments)). Every task is available in every environment except `docs`, which has its own `docs-serve` and `docs-build` tasks. Pixi appends any extra arguments to the task's command, which is how Hydra overrides reach the training scripts.

## Task reference

| Task | Runs | Purpose |
|---|---|---|
| `train` | `python -c 'from goal.ml.cli.train import train_ml; train_ml()'` | Train. Same as `goal-train`, default config `src/goal/configs/ml/simurgh_gmd26.yaml`. |
| `train-ml` | same as `train` | Alias |
| `eval` | `goal-eval` | Evaluate a checkpoint: `pixi run -e cuda eval ckpt_path=<checkpoint>` |
| `finetune` | `goal-finetune` | Fine-tuning; default config `finetune/simurgh_lora` (LoRA on a SIMURGH checkpoint) |
| `simulate` | `goal-simulate` | Langevin MD with a trained model (`-full` environments) |
| `simulate-mts` | `goal-simulate-mts` | RESPA MTS MD or online learning (`-full` environments) |
| `test` | `pytest -k 'not slow'` | Fast test suite |
| `test-full` | `pytest` | Full test suite, including `slow` tests |
| `lint` | `ruff check src/ tests/` | Lint |
| `lint-configs` | `pytest tests/test_cli_and_packaging.py -k stale -q` | Fail if a script, README, doc or config comment uses the removed config-group syntax (`model=…`, `data=…`, `+experiment=…`, `python src/train.py`) |
| `format` | `ruff format src/ tests/` | Auto-format |
| `typecheck` | `mypy src/goal/ml/` | Static type check of `goal.ml` |
| `clean` | see below | Remove build artefacts and caches |
| `clean-logs` | `rm -rf logs/**` | Delete **all** training outputs |
| `docs-serve` | `mkdocs serve` | Live documentation preview (`docs` environment only) |
| `docs-build` | `mkdocs build --strict` | Build the documentation site into `site/` (`docs` environment only) |

## Training and evaluation

```bash
# Default experiment on an NVIDIA GPU / an AMD GPU / the CPU
pixi run -e cuda train
pixi run -e rocm train
pixi run -e cpu train

# Any experiment and overrides
pixi run -e cuda train --config-name nf_simurgh_gmd26 trainer.max_epochs=500
pixi run -e cuda train trainer.devices=4 strategy.name=ddp

# DeepSpeed is part of every ML environment
pixi run -e cuda train trainer.devices=4 strategy.name=deepspeed_zero2

# UMA fine-tuning needs FairChem, installed on demand (see Installation → FairChem)
pixi run -e cuda train --config-name finetune/uma

# Evaluation: the model comes from the checkpoint, the test split from its saved config
pixi run -e cuda eval ckpt_path=logs/simurgh_gmd26/runs/<run>/checkpoints checkpoint=last
```

`train` and `train-ml` call `train_ml()` through `python -c`, which behaves exactly like the `goal-train` console script. All Hydra flags (`--config-name`, `-m`, `--cfg job`, …) work. Read [CLI](cli.md) for what each command does.

> [!TIP]
> Outputs go to `logs/` under `paths.root_dir`, which is `$PROJECT_ROOT` if set and otherwise the directory you ran pixi from. To keep runs on scratch storage, use `PROJECT_ROOT=/scratch/me/goal pixi run -e cuda train`. Note that `data_dir` moves with it.

Arguments after a task name are appended to its command. There is no pixi task for `goal-tune`, `goal-inspect`, `goal-pack-archive`, `goal-pack-zarr` or `goal-finetune-upet`. Run the console scripts inside the environment instead:

```bash
pixi run -e cuda goal-tune
pixi run -e cuda goal-inspect logs/simurgh_gmd26/runs/<run>/checkpoints/
pixi run -e cuda goal-pack-archive --checkpoint-dir logs/simurgh_gmd26/runs/<run>/checkpoints
```

## MD tasks

`simulate` and `simulate-mts` run the console scripts, so every Hydra argument works; use a `-full` environment, which has the MD module's dependencies:

```bash
pixi run -e cuda-full simulate calculator.checkpoint=models/simurgh_gmd26.simurgh steps=5000
pixi run -e cuda-full simulate-mts fast_calculator.checkpoint=models/simurgh_gmd26.simurgh slow_calculator=null
pixi run -e cuda-full goal-simulate-batch --checkpoint models/simurgh_gmd26.simurgh --structures frames.xyz
```

See the [MD CLI reference](../../md/reference/cli.md). (Before October 2026 both tasks ran `python -m goal.md.cli.simulate`, which had no `__main__` block and exited without simulating anything.)

## Quality and tests

```bash
pixi run -e cuda test          # pytest -k 'not slow'   (also runs doctests: --doctest-modules)
pixi run -e cuda test-full     # everything, including @pytest.mark.slow
pixi run -e cuda lint
pixi run -e cuda format
pixi run -e cuda typecheck

# Pass pytest options through
pixi run -e cuda test tests/ml/simurgh -x -q
pixi run -e cuda-full test -m 'not integration'
```

The dev tools (`pytest`, `ruff`, `mypy`, `pre-commit`) are in every environment, so these tasks run anywhere. Use a `-full` environment to include the MD module's tests. Tests marked `integration` need a real foundation model or external engine; deselect them with `-m 'not integration'` on machines without one.

## Housekeeping

`clean` runs:

```bash
rm -rf dist build *.egg-info && \
find . -type f -name '*.DS_Store' -ls -delete && \
find . | grep -E '(__pycache__|\.pyc|\.pyo)' | xargs rm -rf && \
find . | grep -E '.pytest_cache' | xargs rm -rf && \
find . | grep -E '.ipynb_checkpoints' | xargs rm -rf && \
rm -f .coverage
```

> [!WARNING]
> `clean-logs` runs `rm -rf logs/**` and deletes **every** run, checkpoint and `.simurgh` archive under `logs/`, with no confirmation. Copy anything you want to keep (for example a packed archive) out of `logs/` first.

## Adding your own tasks

Tasks are plain strings in `pyproject.toml`:

```toml
[tool.pixi.tasks]
train-md17 = "python -c 'from goal.ml.cli.train import train_ml; train_ml()' --config-name nf_simurgh_md17"
```

Extra arguments are still appended: `pixi run -e cuda train-md17 trainer.max_epochs=100`.
