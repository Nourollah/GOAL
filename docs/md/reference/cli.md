# CLI

`goal.md` installs three console scripts. `goal-simulate` (standard MD) and `goal-simulate-mts` (multiple-time-step MD and online learning) are **Hydra** applications defined in [`src/goal/md/cli/simulate.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/md/cli/simulate.py): they take Hydra flags and `key=value` overrides, not `--option` style arguments. `goal-simulate-batch` (MD or relaxation of many structures at once) is a plain `--option` command, documented in [Batched simulation](../../integrations/batched_simulation.md). This page lists what each command runs and the flags it accepts.

| Command | Entry point | Default config | Builds and runs |
| --- | --- | --- | --- |
| `goal-simulate` | `goal.md.cli.simulate:simulate` | `md/simulations/langevin_with_model` | `simulate_from_config(cfg)` → `Simulation` (Langevin NVT) |
| `goal-simulate-mts` | `goal.md.cli.simulate:simulate_mts` | `md/mts/simulations/mts_respa` | `MTSSimulation.from_config(cfg).run()` (RESPA, or the online learner when the config has `base_calculator` + `ml_calculator`) |
| `goal-simulate-batch` | `goal.md.cli.simulate_batch:main` | `--config` YAML optional (`md/simulations/batch.yaml`) | `run_batched_simulation` — nvalchemi GPU batch or ASE ([details](../../integrations/batched_simulation.md)) |

## Before you start

The bundled configs (`goal/configs/`, path in `goal.CONFIGS_DIR`) are found automatically from any directory, in a clone or in a pip/conda install, so `goal-simulate calculator.checkpoint=...` works out of the box. Two conventions matter for **your own** config files:

1. **Start sub-directory configs with `# @package _global_`.** Without it, Hydra nests a config stored under `md/simulations/` inside `md.simulations`, and the command finds no `molecule:`. See [Configuration](configuration.md#writing-a-simulation-config-that-works).
2. **Your own config directory needs an absolute `--config-path`.** A relative `--config-path` is interpreted relative to `simulate.py`, not your shell. Pass `"$PWD/configs/md_runs"`, or use `--config-dir configs/md_runs` (relative to your shell), which adds the directory to the search path next to the bundled configs.

## `goal-simulate`

The command composes a config, builds the molecule and calculator, runs Langevin NVT through `Simulation`, and logs a summary:

```text
[...][goal.md.cli.simulate][INFO] - Starting MD simulation
[...][goal.md.cli.simulate][INFO] - Simulation complete — 20000 steps, final energy=-4198.2490 eV, final T=301.7 K
```

### Minimal invocation

The default config runs Langevin NVT of ethanol for 1000 steps at 0.5 fs with a trained GOAL model. The checkpoint is the only required value:

```bash
pixi run -e cuda-full goal-simulate calculator.checkpoint=logs/my_exp/runs/<run>/checkpoints
pixi run -e cuda-full goal-simulate calculator.checkpoint=models/simurgh_gmd26.simurgh calculator.device=cuda \
    'molecule.smiles="CC(=O)O"' steps=5000 output_dir=outputs/md/acetic_acid
```

`calculator.checkpoint` accepts a `.simurgh` archive, a checkpoint directory or a `.ckpt`; `calculator.select` picks `best` / `last` / `epoch=N` inside a directory, and the EMA weights are used when the checkpoint has them. Check the composed config with `--cfg job` before a long run.

### With your own config

Create `configs/md_runs/ethanol_goal_nvt.yaml` in your working directory (your own config directory, not the bundled one):

```yaml
# @package _global_
molecule:
  key: from_smiles
  smiles: CCO
calculator:
  key: goal_model
  checkpoint: ???
  device: cuda
steps: 20000
timestep_fs: 0.5
friction: 0.01
temperature_K: 300.0
log_interval: 20
trajectory_file: md.traj
log_file: md.log
output_dir: outputs/md/ethanol_goal_nvt
```

Then run it:

```bash
pixi run -e cuda-full goal-simulate --config-path "$PWD/configs/md_runs" --config-name ethanol_goal_nvt \
    calculator.checkpoint=logs/my_exp/runs/2026-10-01_12-00-00_run/checkpoints/best_val_forces_mae.simurgh
```

### Common overrides

```bash
CFG=(--config-path "$PWD/configs/md_runs" --config-name ethanol_goal_nvt)
CKPT=calculator.checkpoint=path/to/best.simurgh

# Different molecule (quote SMILES with parentheses or '=')
pixi run -e cuda-full goal-simulate "${CFG[@]}" $CKPT 'molecule.smiles="CC(=O)O"' output_dir=outputs/md/acetic_acid

# Start from a file instead of SMILES (quote '~...' so the shell does not expand it)
pixi run -e cuda-full goal-simulate "${CFG[@]}" $CKPT molecule.key=from_file '~molecule.smiles' +molecule.path=structures/aspirin.xyz

# Different backend: delete the calculator mapping, then add a new one
pixi run -e cuda-full goal-simulate "${CFG[@]}" '~calculator' '+calculator={key: tblite, method: GFN2-xTB}'

# Longer, hotter, GPU 1, log and frame every 100 steps
pixi run -e cuda-full goal-simulate "${CFG[@]}" $CKPT steps=200000 temperature_K=500 calculator.device=cuda:1 log_interval=100

# Temperature sweep (Hydra multirun, one job per value)
pixi run -e cuda-full goal-simulate "${CFG[@]}" $CKPT -m temperature_K=250,300,350 'output_dir=outputs/md/ethanol_${temperature_K}K'
```

Assigning a mapping to an existing key (`calculator={...}`) **merges** into it and fails on keys the old mapping does not have. That is why the backend swap deletes the mapping with `~calculator` first.

### Without a config file

Every key can also be given on the command line. Keys that the default config lacks need `+`:

```bash
pixi run -e cuda-full goal-simulate calculator.checkpoint=path/to/best.simurgh \
    'molecule.smiles="c1ccccc1"' steps=2000 temperature_K=350 +show_progress=false
```

## `goal-simulate-mts`

This command runs RESPA multiple-time-step MD with a cheap fast calculator (typically a GOAL model) and an expensive slow correction (typically ORCA). `MTSSimulation.from_config` reads `molecule`, `fast_calculator`, `slow_calculator` (optional, `null` = fast-only) and an `mts:` block. The physics and all `mts:` parameters are explained in [MTS overview](../mts/overview.md) and [RESPA integrator](../mts/respa.md).

The default config (`mts_respa.yaml`) needs the fast model and the ORCA executable:

```bash
pixi run -e cuda-full goal-simulate-mts \
    fast_calculator.checkpoint=path/to/best.simurgh \
    slow_calculator.orca_path=/opt/orca/orca \
    mts.mts_ratio=4 mts.total_steps=5000

# Fast-only dry run (no QM): checks the set-up before you spend CPU hours
pixi run -e cuda-full goal-simulate-mts \
    fast_calculator.checkpoint=path/to/best.simurgh slow_calculator=null mts.total_steps=100
```

At the end it logs the step count, final energy and temperature, and the MTS call statistics:

```text
MTS complete — 5000 outer steps, final energy=... eV, final T=... K
MTS stats: {'fast_calculator_calls': ..., 'slow_calculator_calls': ..., 'mts_ratio': 4, 'inner_timestep_fs': 1.0, 'outer_timestep_fs': 4.0}
```

### Online learning

When the config has `base_calculator` and `ml_calculator` sections (`online_learner.yaml`, `orca_ml_learner.yaml`), `goal-simulate-mts` drives a `QMLearnerCalculator` instead: QM labels every step until the ML model is accurate enough, then ML takes over and QM runs only once per `learner.eval_interval` steps to re-calibrate and refresh the fine-tuning buffer. A GOAL model is fine-tuned on the fly (`learner.fine_tuning`). See [How the QM→ML switch works](../mts/overview.md#how-the-qmml-switch-works).

```bash
pixi run -e cuda-full goal-simulate-mts --config-name md/mts/simulations/online_learner \
    ml_calculator.checkpoint=path/to/best.simurgh base_calculator.orca_path=/opt/orca/orca
```

## Hydra flags

Both commands accept the standard Hydra flags:

| Flag | Purpose |
| --- | --- |
| `--config-path DIR` | Primary config directory, replacing the bundled one. **Absolute path required** here. Only needed for your own config directory. |
| `--config-name NAME` | Config to load, relative to the config path, without `.yaml` |
| `--config-dir DIR` | Add a directory to the search path (for `defaults:` lookups; does not replace `--config-path`) |
| `--cfg job` | Print the composed config and exit. Use it to check overrides before a long run. |
| `--resolve` | With `--cfg`, resolve interpolations |
| `--help` | Hydra help, including the composed config |
| `--info` | Search path and defaults-tree diagnostics |
| `-m` / `--multirun` | Sweep over comma-separated values |
| `hydra.run.dir=PATH` | Where Hydra writes `.hydra/` and the log. Default `outputs/<date>/<time>`. |

Override syntax (`key=`, `+key=`, `++key=`, `~key`) is summarised in [Configuration](configuration.md#composition-and-overrides).

Hydra does not change the working directory (the commands use `version_base="1.3"`), so relative `output_dir` and checkpoint paths are resolved against the directory you launch from.

## Pixi tasks

`pyproject.toml` defines two tasks that run the console scripts; arguments after the task name are appended:

```toml
simulate     = "goal-simulate"
simulate-mts = "goal-simulate-mts"
```

```bash
pixi run -e cuda-full simulate calculator.checkpoint=path/to/best.simurgh steps=5000
pixi run -e cuda-full simulate-mts fast_calculator.checkpoint=path/to/best.simurgh slow_calculator=null
```

Use a `-full` environment, which has the MD dependencies. `python -m goal.md.cli.simulate` also runs `goal-simulate`.

## The `goal-md` Typer app

`goal/md/cli/main.py` holds a placeholder Typer application named `goal-md` with `version` and `simulate -c CONFIG` commands. It is **not** installed as a console script; `python -m goal.md.cli.main --help` runs it (needs `typer`). Its `simulate` command prints "Not yet implemented". Older docs that show `goal-md simulate -c ...` refer to this stub; use `goal-simulate` instead.

## Batch jobs

A single `goal-simulate` run uses one process, and one GPU if the calculator is on CUDA. On a cluster, submit one job per config or per multirun value. A minimal SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=md-ethanol
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
cd "$SLURM_SUBMIT_DIR"
pixi run -e cuda-full goal-simulate --config-path "$PWD/configs/md_runs" --config-name ethanol_goal_nvt \
    calculator.checkpoint=path/to/best.simurgh calculator.device=cuda \
    output_dir=outputs/md/ethanol_${SLURM_JOB_ID} hydra.run.dir=outputs/hydra/${SLURM_JOB_ID}
```

For QM backends (CP2K, Quantum ESPRESSO, VASP, ORCA), MPI ranks and threads are calculator arguments (`n_mpi`, `omp_threads`, `num_threads`, …); see the relevant [calculator page](../calculators/overview.md).
