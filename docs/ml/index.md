# goal.ml

`goal.ml` is GOAL's machine-learning half: a framework for training, fine-tuning, evaluating and deploying machine-learning interatomic potentials (MLIPs) on top of PyTorch Lightning, PyTorch Geometric, e3nn and Hydra. This page gives the overview: what is in the package, how it is laid out, the shortest path to a trained model, and where each topic is documented in detail.

## What it is

GOAL treats an MLIP as a **backbone** that turns an atomic structure into per-atom features, plus a **head** that turns those features into properties (energy, forces, stress, dipole, …), trained by one universal Lightning module, `GOALModule`. A backbone that already returns properties is a **monolithic model** and runs with `head: null`. Everything is assembled from registries by name, and every run is described by a single self-contained YAML file. Swapping architecture, dataset or loss is therefore a config change, not a code change.

The native model family is **SIMURGH**, an E(3)-equivariant force field built around a bank of *potential artisans* (one small equivariant network per element pair). The default architecture is **ARACE** (`simurgh_arace`), which alternates artisan layers on edges with ACE message-passing blocks on nodes.

## Features

| | Feature | Details |
|---|---|---|
| 🔬 | Native SIMURGH force fields | `simurgh_arace` (default, modular), `monolithic_arace`, `nf_simurgh` (fragment virtual nodes), `simurgh_simplified_routed` / `_monolithic`, legacy ACE-first `simurgh`, plus the isolated `simurgh_ablation` study model |
| 🧱 | Modular and monolithic models | Backbone → head pipeline, or self-contained models with `head: null`. Baselines: `hyperspec`, `invariant_gnn`, `monolithic_example`. |
| 🎯 | Task heads | `energy`, `energy_forces`, `direct_forces`, `dual_forces`, `stress`, `dipole`, `scalar`, `multi` |
| 🧠 | Foundation-model fine-tuning | `mace_finetune` and `uma_finetune` trained in-process (`head_only` / `full` / `lora`); UPET/PET through metatrain (`goal-finetune-upet`); frozen MACE/UMA adapters for feature extraction |
| 📂 | Data loading | ExtXYZ, ASE trajectory/db, HDF5, LMDB, FairChem ASE-LMDB (OMol25, OMat24, ODAC23); four loading modes; multi-source merging; automatic splits; label key mapping; fragment decomposition |
| 📊 | Benchmarks | MD17/rMD17, MD22, QM9, ANI-1/1x, SPICE adapters with ready-made configs |
| 🧮 | Losses | Per-property `energy`, `forces`, `stress`, `dipole`, `charge`, `gate_reg`, `scalar_property`; `mse` / `mae` / `rmse` / `huber` / `smooth_l1` or any dotted torchmetrics function; composite sub-losses; staged loss schedules |
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
│   ├── blocks        # artisans, ACE block, env dressing, symmetric contraction, add-ons
│   ├── heads         # task heads
│   └── models        # backbones and monolithic models (simurgh/, ablations/, foundation/, …)
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

print(BACKBONE_REGISTRY.list_available())   # ['hyperspec', 'invariant_gnn', ..., 'simurgh_arace', ...]

backbone = BACKBONE_REGISTRY.get("simurgh_arace")(
    elements=[1, 6, 8],
    cutoff=5.0,
    avg_num_neighbors=12.0,
)
head = HEAD_REGISTRY.get("energy_forces")(irreps_in="16x0e + 16x1o + 16x2e", hidden_dim=128)
```

## Quick start

Install GOAL first ([Installation](../getting-started/installation.md)). Then, from the repository root:

```bash
# 1. Train the default model (SIMURGH ARACE on GMD-26, src/goal/configs/ml/simurgh_gmd26.yaml)
pixi run -e cuda goal-train

# Equivalent invocations
pixi run -e cuda goal-train-ml
pixi run -e cuda python -m goal.ml.cli.train
pixi run -e cuda train
```

The default config expects the GMD-26 FragmentDuplication split under `data/GMD/FragmentDuplication/{Training,TestID,TestOOD}` and a GPU (`trainer.accelerator: gpu`). Point it at your own data and hardware with overrides:

```bash
# CPU, short run
pixi run -e cuda goal-train trainer.accelerator=cpu trainer.max_epochs=5

# Your own ExtXYZ data, one directory per split
pixi run -e cuda goal-train data.dataset_type=xyz \
    data.train_dir=/data/mysys/train data.val_dir=/data/mysys/val data.test_dir=/data/mysys/test

# Another ready-made experiment
pixi run -e cuda goal-train --config-name nf_simurgh_md17          # NF_SIMURGH on rMD17 aspirin
pixi run -e cuda goal-train --config-name monolithic_arace_gmd26   # monolithic ARACE

# Tweak the model and optimiser
pixi run -e cuda goal-train model.backbone.num_rounds=3 training.optimizer.lr=5e-3 training.ema.decay=0.999

# Multi-GPU
pixi run -e cuda goal-train trainer.devices=4 strategy.name=ddp

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
    checkpoint_path="logs/simurgh_gmd26/runs/<run>/checkpoints/best_val_forces_mae.simurgh",
    device="cuda",
)
atoms = molecule("CH3COOH")
atoms.calc = calc
print(atoms.get_potential_energy(), atoms.get_forces().shape)
```

> [!WARNING]
> The shipped `simurgh_gmd26.yaml` enables the `fragment_interaction` add-on, so the trained model needs a `fragment_index` on every graph. `GOALCalculator` builds graphs with `AtomicGraph.from_ase(...)` **without** fragment labels, and the snippet above then raises `ValueError: model.backbone.fragment_interaction is enabled but the graphs carry no 'fragment_index'`. You have two options. Train with the add-on removed (`goal-train '~model.backbone.fragment_interaction'`) if you want plain `GOALCalculator` inference. Otherwise use a fragment-aware calculator subclass that passes the training-time fragment settings to `from_ase`, as `NFSimurghCalculator` does in [`notebooks/nf_simurgh_gmd_inference.ipynb`](https://github.com/Nourollah/GOAL/blob/main/notebooks/nf_simurgh_gmd_inference.ipynb). The same applies to `nf_simurgh` models. See [ASE calculator](deployment/ase-calculator.md).

```bash
pixi run -e cuda goal-inspect logs/simurgh_gmd26/runs/<run>/checkpoints/     # provenance and metrics, no model build
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
| Understand SIMURGH / ARACE | [SIMURGH](models/simurgh.md) |
| Use fragment virtual nodes | [NF_SIMURGH](models/nf-simurgh.md) |
| Use the three-line routed / monolithic models | [SIMURGH Simplified](models/simurgh-simplified.md) |
| Reproduce or extend the module ablations | [Ablation study](models/ablation-study.md) |
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
