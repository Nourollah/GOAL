# goal.ml

`goal.ml` is GOAL's machine-learning half: a framework for training, fine-tuning, evaluating and deploying machine-learning interatomic potentials (MLIPs) on top of PyTorch Lightning, PyTorch Geometric, e3nn and Hydra. This page gives the overview: what is in the package, how it is laid out, the shortest path to a trained model, and where each topic is documented in detail.

## What it is

GOAL treats an MLIP as a **backbone** that turns an atomic structure into per-atom features, plus a **head** that turns those features into properties (energy, forces, stress, dipole, …), trained by one universal Lightning module, `GOALModule`. A backbone that already returns properties is a **monolithic model** and runs with `head: null`. Everything is assembled from registries by name, and every run is described by a single self-contained YAML file. Swapping architecture, dataset or loss is therefore a config change, not a code change.

The native model is **SIMURGH**, an E(3)-equivariant ACE-type force field with conservative forces. It is in development under the registry key `simurgh_dev`; see [SIMURGH (development)](models/simurgh-dev.md).

## Features

| | Feature | Details |
|---|---|---|
| 🔬 | Native SIMURGH force field | `simurgh_dev` (default, monolithic), one config per benchmark task, and one `simurgh_dev_abl_*` arm per ablated mechanism |
| 🧱 | Modular and monolithic models | Backbone → head pipeline, or self-contained models with `head: null`. Baselines: `hyperspec`, `invariant_gnn`, `monolithic_example`. |
| 🎯 | Task heads | `energy`, `energy_forces`, `direct_forces`, `dual_forces`, `stress`, `dipole`, `scalar`, `multi` |
| 🧠 | Foundation-model fine-tuning | `mace_finetune` and `uma_finetune` trained in-process (`head_only` / `full` / `lora`); UPET/PET through metatrain (`goal-finetune-upet`); frozen MACE/UMA adapters for feature extraction |
| 📂 | Data loading | ExtXYZ, ASE trajectory/db, HDF5, LMDB, FairChem ASE-LMDB (OMol25, OMat24, ODAC23); four loading modes; multi-source merging; automatic splits; label key mapping; fragment decomposition |
| 📊 | Benchmarks | GMD-26 from the Hugging Face Hub; MD17/rMD17, MD22, QM9, ANI-1/1x, SPICE adapters; ready-made configs for GMD-26, rMD17, MD22 and 3BPA |
| 🧮 | Losses | Per-property `energy`, `forces`, `stress`, `dipole`, `charge`, `scalar_property`; `mse` / `mae` / `rmse` / `huber` / `smooth_l1` or any dotted torchmetrics function; composite sub-losses; staged loss schedules |
| ⚡ | Distributed training | `ddp`, `fsdp`, `fsdp2`, `deepspeed_zero1/2/3`, `deepspeed_zero3_offload` through one `strategy.name` key |
| 🚀 | Performance | TF32 on Ampere+, cuDNN autotuning, `torch.compile`, mixed precision, gradient accumulation, EMA |
| 💾 | Checkpointing | Three-pool `GOALCheckpointManager` (top-k / interval / last), frozen model source, self-contained `.simurgh` archives, auto-resume, run extension with LR-schedule retargeting, SLURM completion sentinel |
| 🔍 | Tuning | Lightning LR / batch-size finder, Ray Tune (ASHA/PBT + Optuna/HyperOpt), W&B Sweeps |
| 📈 | Logging | CSV and W&B by default; TensorBoard, MLflow, Neptune, Aim and Comet available; metrics grouped by section (`val/`, `val_energy/`, `val_forces/`, `val_gate/`); optional live OOD monitoring |
| 🧪 | Deployment | `GOALCalculator` (ASE) for single points, optimisation and MD; bridges into `goal.md` |
| 🧰 | Custom loops | Override `GOALModule` hooks, `FabricTrainer` (multi-GPU, full control), or the pure-PyTorch `MiniTrainer`; feature extraction utilities |

## Package layout and imports

```text
goal.ml
├── registry          # MODEL / BACKBONE / HEAD / LOSS / DATASET / TRANSFORM / STRATEGY registries
├── cli               # console-script entry points (goal-train, goal-eval, …)
├── data              # AtomicGraph, GOALDataModule, datasets, neighbour lists, statistics, fragments
├── nn
│   ├── primitives    # e3nn wrappers: tensor products, linear, norms, radial bases
│   ├── blocks        # interaction, embedding, readout, ECE block
│   ├── heads         # task heads
│   └── models        # backbones and monolithic models (simurgh/, foundation/, …)
├── adapters          # frozen foundation-model adapters (MACE, FairChem)
├── training          # GOALModule, losses, metrics, EMA, resume, archives, callbacks, strategies, tuning
└── utils             # GOALCalculator, feature extraction, MiniTrainer, FabricTrainer
```

The imports you will use most:

```python
# Data
from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.data.datamodule import GOALDataModule

# Components by name
from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY, DATASET_REGISTRY
import goal.ml.cli  # side effect: populates every registry (models, heads, datasets, adapters)

# Training
from goal.ml.training.module import GOALModule
from goal.ml.training.loss import CompositeLoss, WeightedLoss

# Deployment and analysis
from goal.ml.utils.calculator import GOALCalculator
from goal.ml.training.archive import inspect_checkpoint, pack_simurgh_archive
from goal.ml.utils.mini_trainer import MiniTrainer
from goal.ml.utils.extraction import HookBasedExtractor, extract_scalars
```

Build a component by registry name, exactly as `goal-train` does:

```python
import goal.ml.cli  # noqa: F401 (registers everything)
from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY

print(BACKBONE_REGISTRY.list_available())   # ['hyperspec', 'invariant_gnn', ..., 'simurgh_dev', ...]

model = BACKBONE_REGISTRY.get("simurgh_dev")(   # monolithic: no head
    elements=[1, 6, 8],
    cutoff=5.0,
    avg_num_neighbors=12.0,
    scale=1.0,
    atomic_energies={"mode": "provided", "values": {1: -13.6, 6: -1028.5, 8: -2041.1}},
)
head = HEAD_REGISTRY.get("energy_forces")(irreps_in="16x0e + 16x1o + 16x2e", hidden_dim=128)  # for a modular backbone
```

## Quick start

Install GOAL first ([Installation](../getting-started/installation.md)). Then, from the repository root:

```bash
# 1. Train the default model (SIMURGH on GMD-26, src/goal/configs/ml/simurgh_dev_gmd26_FD.yaml)
pixi run -e cuda goal-train

# Equivalent invocations
pixi run -e cuda goal-train-ml
pixi run -e cuda python -m goal.ml.cli.train
pixi run -e cuda train
```

The default config downloads the GMD-26 FragmentDuplication task from the Hugging Face Hub on first use and expects a GPU. Point it at your own data and hardware with overrides:

```bash
# CPU, short run
pixi run -e cuda goal-train trainer.accelerator=cpu trainer.max_epochs=5

# Your own ExtXYZ data, one file split by data.split_ratio
pixi run -e cuda goal-train data.dataset_type=xyz '~data.task' data.root=/data/mysys/all.xyz

# Another ready-made experiment
pixi run -e cuda goal-train --config-name simurgh_dev_rmd17_aspirin       # rMD17 aspirin
pixi run -e cuda goal-train --config-name simurgh_dev_gmd26_FF            # GMD-26 FragmentFusion

# Tweak the model and optimiser
pixi run -e cuda goal-train model.backbone.num_interactions=3 training.optimizer.lr=5e-3 training.ema.decay=0.999

# Multi-GPU
pixi run -e cuda goal-train trainer.devices=4 ++strategy.name=ddp

# Sweep three learning rates
pixi run -e cuda goal-train -m training.optimizer.lr=1e-2,5e-3,1e-3

# Just print the composed config
pixi run -e cuda goal-train --cfg job
```

Each run writes to `logs/<task_name>/runs/<timestamp>_<dataset>_<backbone><suffix>/`. When training finishes, `checkpoints/` contains the top-k, interval and `last.ckpt` checkpoints plus `best_<metric>.simurgh`, a single-file archive. Use it right away:

```python
from ase.build import molecule
from goal.ml.utils.calculator import GOALCalculator

calc = GOALCalculator(
    checkpoint_path="logs/simurgh_dev_gmd26_FD/runs/<run>/checkpoints/best_val_forces_forces_mae_atomw.simurgh",
    device="cuda",
)
atoms = molecule("CH3COOH")
atoms.calc = calc
print(atoms.get_potential_energy(), atoms.get_forces().shape)
```

`GOALCalculator` reads the fragment settings (`data.compute_fragment_index` and friends) from the saved config, so inference graphs carry the same labels as in training. See [ASE calculator](deployment/ase-calculator.md).

```bash
pixi run -e cuda goal-inspect logs/simurgh_dev_gmd26_FD/runs/<run>/checkpoints/     # provenance and metrics, no model build
```

If you prefer to learn in Python rather than YAML, start with the [notebooks](../getting-started/notebooks.md).

## Where to go next

| I want to… | Read |
|---|---|
| Train a model end to end and understand every config block | [Training a model](training/training.md) |
| Use several GPUs or nodes, or SLURM | [Distributed training](training/distributed.md) |
| Continue an interrupted or finished run | [Resuming & extending](training/resuming.md) |
| Make training faster | [Performance engineering](training/performance.md) |
| Choose and weight loss terms, define loss stages | [Loss functions](training/losses.md) |
| Configure checkpoints, callbacks and loggers | [Callbacks & logging](training/callbacks-logging.md) |
| Search hyperparameters | [Hyperparameter tuning](training/hyperparameter-tuning.md) |
| Write my own training loop | [Custom training loops](training/custom-loops.md) |
| Load my own data (ExtXYZ, traj, HDF5, LMDB) | [Loading data](data/loading.md) |
| Train on OMol25 or other FairChem `.aselmdb` data | [ASE-LMDB & OMol25](data/aselmdb-omol25.md) |
| Use MD17, MD22, QM9, ANI-1, SPICE | [Benchmark datasets](data/benchmarks.md) |
| Compare the available architectures | [Overview & model zoo](models/overview.md) |
| Understand SIMURGH and its ablation programme | [SIMURGH (development)](models/simurgh-dev.md) |
| Use HyperSpec, InvariantGNN or the monolithic example | [Other backbones](models/other-backbones.md) |
| Predict stress, dipoles, scalars or several properties | [Heads](models/heads.md) |
| Add my own architecture | [Writing a new model](models/custom-model.md) |
| Fine-tune a foundation model | [Fine-tuning overview](finetuning/overview.md), [MACE](finetuning/mace.md), [UMA / FairChem](finetuning/uma.md), [UPET / PET](finetuning/upet.md) |
| Extract features from a (foundation) model | [Feature extraction](features/feature-extraction.md) |
| Train small heads quickly in a notebook | [Mini trainer](features/mini-trainer.md) |
| Evaluate a trained model on a test set | [Evaluation](deployment/evaluation.md) |
| Run MD or optimisations with my model | [ASE calculator](deployment/ase-calculator.md), then [Bridge to goal.ml](../md/guide/ml-bridge.md) |
| Ship a model or understand checkpoint files | [Checkpoints & archives](deployment/archives.md) |
| Look up config keys, CLI flags or pixi tasks | [Configuration](reference/configuration.md), [CLI](reference/cli.md), [Pixi tasks](reference/pixi-tasks.md) |
