<div align="center">

# 🧬 GMD — General Molecular Dynamics

[![python](https://img.shields.io/badge/-Python_3.14+-blue?logo=python&logoColor=white)](https://python.org)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![cuda](https://img.shields.io/badge/CUDA-12.6+-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

**Train equivariant graph neural networks on atomistic systems.**

Energy, forces, stress, and dipole prediction — from single-GPU prototyping to multi-node distributed training.

> **Python 3.14** — GIL-free interpreter with real multithreading support for data loading and preprocessing.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
  - [Single GPU](#single-gpu)
  - [Multi-GPU: DDP](#multi-gpu-ddp)
  - [Multi-GPU: FSDP](#multi-gpu-fsdp)
  - [Multi-GPU: FSDP2 / ModelParallel](#multi-gpu-fsdp2--modelparallel)
  - [Multi-GPU: DeepSpeed](#multi-gpu-deepspeed)
  - [Strategy Factory](#strategy-factory)
  - [Apple Silicon (MPS)](#apple-silicon-mps)
  - [CPU](#cpu)
- [Evaluation](#evaluation)
- [ASE Calculator](#ase-calculator)
- [Fine-Tuning](#fine-tuning)
- [Data Loading](#data-loading)
  - [Supported Formats](#supported-formats)
  - [Loading Modes](#loading-modes)
  - [Merge Strategies](#merge-strategies)
  - [Auto-Splitting](#auto-splitting)
- [Models](#models)
- [Heads](#heads)
- [Loss Functions](#loss-functions)
- [Benchmark Datasets](#benchmark-datasets)
- [Foundation Model Adapters](#foundation-model-adapters)
- [Feature Extraction](#feature-extraction)
- [Performance Engineering](#performance-engineering)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Callbacks](#callbacks)
- [Logging](#logging)
- [Configuration System](#configuration-system)
- [CLI Reference](#cli-reference)
- [Pixi Tasks](#pixi-tasks)

---

## Overview

GMD is a framework for training machine-learning interatomic potentials (MLIPs) using equivariant and invariant graph neural networks. Built on PyTorch Lightning 2.6+ and Hydra, it provides:

- 🔬 **Equivariant & invariant backbones** — HyperSpec (E(3)-equivariant) and SchNet-like invariant GNN
- 🎯 **Multiple task heads** — energy, forces, stress, dipole, direct forces
- 🧠 **Foundation model adapters** — MACE and FairChem (UMA) pre-trained models
- 📂 **Flexible data loading** — XYZ, HDF5, LMDB, ASE trajectory; multi-file merge, directory-based loading, and auto-splitting
- ⚡ **Distributed training** — DDP, FSDP, FSDP2 (ModelParallel), and DeepSpeed ZeRO (Stages 1/2/3 + CPU offload)
- 🧮 **Configurable loss functions** — per-property loss type selection (MSE, MAE, Huber, Smooth L1) + arbitrary weights
- 🔧 **Strategy factory** — unified `build_strategy(cfg)` entry point for all distributed strategies
- 🚀 **Performance engineering** — TF32, cuDNN benchmark, `torch.compile`, gradient accumulation, EMA/SWA
- 📊 **Experiment management** — Hydra config composition, W&B / TensorBoard / CSV / MLflow / Neptune / Aim / Comet loggers
- 🖥️ **SLURM-aware** — automatic checkpoint resumption and completion sentinels
- 🐍 **Python 3.14** — GIL-free interpreter with true multithreading for data loading

---

## Installation

### With pip

```bash
git clone https://github.com/user/gmd.git
cd gmd
pip install -e .
```

Optional extras:

```bash
pip install -e ".[mace]"       # MACE adapter
pip install -e ".[fairchem]"   # FairChem/UMA adapter
pip install -e ".[deepspeed]"  # DeepSpeed ZeRO strategies
pip install -e ".[all]"        # All optional dependencies
pip install -e ".[dev]"        # pytest, ruff, mypy
```

### With pixi

GMD ships a pixi workspace configuration in `pyproject.toml`. Pixi manages conda + pip dependencies and named environments:

```bash
pixi install                    # default (CPU)
pixi install -e cuda            # CUDA 12.6
pixi install -e dev             # CPU + dev tools
pixi install -e dev-cuda        # CUDA 12.6 + dev tools
pixi install -e mace            # CUDA + MACE adapter
pixi install -e fairchem        # CUDA + FairChem adapter
pixi install -e cuda-deepspeed  # CUDA + DeepSpeed
```

> **Note:** Minimum CUDA version is **12.6**. Older CUDA versions are not supported.

---

## Project Structure

<details>
<summary>📁 Click to expand full project tree</summary>

```
├── configs/                    # Hydra configuration groups
│   ├── train.yaml              # Training defaults composition
│   ├── eval.yaml               # Evaluation defaults composition
│   ├── callbacks/              # Callback configs (checkpoint, EMA, SWA, ...)
│   ├── data/                   # Dataset configs (xyz, hdf5, lmdb, trajectory)
│   ├── logger/                 # Logger configs (wandb, tensorboard, csv, ...)
│   ├── model/                  # Model configs (hyperspec, invariant_gnn)
│   ├── strategy/               # 🆕 Strategy configs (ddp, fsdp, fsdp2, deepspeed_*)
│   ├── trainer/                # Trainer configs (gpu, ddp, fsdp, model_parallel, ...)
│   ├── training/               # Training hyperparameters (optimizer, EMA, losses, ...)
│   ├── paths/                  # Path definitions
│   └── hydra/                  # Hydra runtime settings
├── src/
│   ├── gmd/                    # Main package
│   │   ├── cli/                # Entry points: train, evaluate, finetune
│   │   ├── data/               # DataModule, datasets (base, xyz, hdf5, lmdb, trajectory, concat)
│   │   ├── nn/                 # Neural network components
│   │   │   ├── models/         # Backbones: HyperSpec, invariant GNN
│   │   │   ├── heads/          # Task heads: energy, forces, stress, dipole
│   │   │   ├── blocks/         # Building blocks: embedding, interaction, readout
│   │   │   └── primitives/     # Low-level ops: tensor products, radial basis, norms
│   │   ├── adapters/           # Foundation model wrappers: MACE, FairChem
│   │   ├── training/           # LightningModule, loss, EMA, callbacks
│   │   │   └── strategies/     # 🆕 Strategy factory: DDP, FSDP, FSDP2, DeepSpeed
│   │   ├── utils/              # Feature extraction, utilities
│   │   └── registry.py         # Lazy component registry
├── data/                       # Dataset storage
├── logs/                       # Training outputs (checkpoints, metrics)
├── tests/                      # Test suite
└── pyproject.toml              # Package metadata + pixi workspace config
```

</details>

---

## Quick Start

Train the default model (HyperSpec) on XYZ data:

```bash
gmd-train data.root=/path/to/dataset
```

Or equivalently via module:

```bash
python -m gmd.cli.train data.root=/path/to/dataset
```

This loads `configs/train.yaml` which composes: `data=xyz`, `model=hyperspec`, `training=default`, `trainer=default`.

---

## Training

### Single GPU

```bash
gmd-train trainer=gpu data.root=/path/to/dataset
```

The `gpu` trainer config sets `accelerator: gpu` and `devices: 1`.

### Multi-GPU: DDP

Distributed Data Parallel — replicates the full model on each GPU and synchronizes gradients. Use when the model fits in a single GPU's memory.

```bash
gmd-train trainer=ddp data.root=/path/to/dataset
```

Override the number of GPUs:

```bash
gmd-train trainer=ddp trainer.devices=8
```

Multi-node:

```bash
gmd-train trainer=ddp trainer.devices=4 trainer.num_nodes=2
```

DDP key settings (in `configs/trainer/ddp.yaml`):
- `find_unused_parameters: false` — set `true` if you have frozen layers
- `static_graph: false` — set `true` for models with fixed computation graphs (faster)
- `gradient_as_bucket_view: true` — minor memory optimisation
- `sync_batchnorm: true` — synchronize batch norm statistics across GPUs

### Multi-GPU: FSDP

Fully Sharded Data Parallel — shards model parameters, gradients, and optimizer states across GPUs. Use when the model doesn't fit in a single GPU's memory.

```bash
gmd-train trainer=fsdp data.root=/path/to/dataset
```

FSDP settings (in `configs/trainer/fsdp.yaml`):
- `auto_wrap_policy` — controls how modules are wrapped for sharding
- `activation_checkpointing_policy` — trade compute for memory by recomputing activations
- `cpu_offload: false` — offload parameters to CPU (slower, saves GPU memory)
- `precision: "bf16-mixed"` — recommended for FSDP on Ampere+ GPUs

### Multi-GPU: FSDP2 / ModelParallel

ModelParallelStrategy (Lightning 2.4+) — supports FSDP2, tensor parallelism, `torch.compile`, and FP8. Recommended for very large models (500M+ parameters).

```bash
gmd-train trainer=model_parallel data.root=/path/to/dataset
```

Or via the strategy factory:

```bash
gmd-train +strategy=fsdp2 data.root=/path/to/dataset
```

### Multi-GPU: DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) ZeRO enables training of very large models by partitioning optimizer states, gradients, and parameters across GPUs. Requires `pip install -e ".[deepspeed]"`.

**ZeRO Stage 1** — optimizer state partitioning only (lowest communication overhead):
```bash
gmd-train +strategy=deepspeed_zero1 data.root=/path/to/dataset
```

**ZeRO Stage 2** — optimizer state + gradient partitioning:
```bash
gmd-train +strategy=deepspeed_zero2 data.root=/path/to/dataset
```

**ZeRO Stage 3** — full parameter partitioning (maximum memory savings):
```bash
gmd-train +strategy=deepspeed_zero3 data.root=/path/to/dataset
```

**ZeRO Stage 3 + CPU offload** — offload parameters to CPU (for extremely large models):
```bash
gmd-train +strategy=deepspeed_zero3_offload data.root=/path/to/dataset
```

<details>
<summary>📋 DeepSpeed configuration options</summary>

```yaml
# configs/strategy/deepspeed_zero3.yaml
name: deepspeed_zero3
stage: 3
allgather_bucket_size: 200_000_000
reduce_bucket_size: 200_000_000
logging_level: WARNING
```

</details>

### Strategy Factory

GMD provides a unified **strategy factory** (`build_strategy()`) that maps config to Lightning strategies. When `cfg.strategy` is present, it takes priority over the trainer's built-in strategy.

```yaml
# Two ways to select a strategy:
# 1. Via trainer config group (backward compatible):
gmd-train trainer=ddp

# 2. Via strategy config group (new, more options):
gmd-train +strategy=fsdp2
gmd-train +strategy=deepspeed_zero3_offload
```

| Strategy Config | Lightning Strategy | Use Case |
|---|---|---|
| `ddp` | `DDPStrategy` | Model fits on one GPU |
| `fsdp` | `FSDPStrategy` | Model too large for one GPU |
| `fsdp2` | `ModelParallelStrategy` | Very large models, torch.compile |
| `deepspeed_zero1` | `DeepSpeedStrategy` (stage 1) | Optimizer state partitioning |
| `deepspeed_zero2` | `DeepSpeedStrategy` (stage 2) | + gradient partitioning |
| `deepspeed_zero3` | `DeepSpeedStrategy` (stage 3) | Full parameter partitioning |
| `deepspeed_zero3_offload` | `DeepSpeedStrategy` (stage 3) | + CPU offload |

### Apple Silicon (MPS)

```bash
gmd-train trainer=mps data.root=/path/to/dataset
```

### CPU

```bash
gmd-train trainer=cpu data.root=/path/to/dataset
```

---

## Evaluation

Evaluate a trained checkpoint on the test split:

```bash
gmd-eval ckpt_path=/path/to/checkpoint.ckpt data.root=/path/to/dataset
```

The evaluation entry point supports the same trainer configs for distributed evaluation:

```bash
gmd-eval trainer=ddp ckpt_path=/path/to/checkpoint.ckpt data.root=/path/to/dataset
```

---

## ASE Calculator

Any trained GMD model can be used as an [ASE Calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html) for molecular dynamics, geometry optimisation, phonons, and more.

### From a Checkpoint

```python
from gmd.utils.calculator import GMDCalculator

calc = GMDCalculator(checkpoint_path="logs/train/runs/.../last.ckpt")
```

### From a Pre-loaded Module

```python
from gmd.utils.calculator import GMDCalculator

calc = GMDCalculator(module=my_module, cutoff=5.0, device="cuda")
```

### Single-Point Calculation

```python
from ase.build import molecule

atoms = molecule("H2O")
atoms.calc = calc

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV/Å
stress = atoms.get_stress()             # eV/ų (Voigt, 6-component)
```

### Geometry Optimisation

```python
from ase.optimize import BFGS

opt = BFGS(atoms)
opt.run(fmax=0.01)
```

### Molecular Dynamics

```python
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_path` | — | Path to `.ckpt` file (mutually exclusive with `module`) |
| `module` | — | Pre-loaded `GMDModule` instance |
| `cutoff` | from config | Neighbour-list cutoff (Å). Auto-detected from checkpoint |
| `device` | `"cpu"` | `"cpu"`, `"cuda"`, `"cuda:0"`, etc. |
| `dtype` | `float64` | Precision for positions and cell |
| `head` | `None` | Multi-head tag for multi-task models |

---

## Fine-Tuning

Fine-tune a pre-trained foundation model on a downstream dataset:

```bash
gmd-finetune model.backbone.name=mace-large model.backbone.pretrained=true data.root=/path/to/dataset
```

### Backbone Loading Modes

**1. Pre-trained hub model:**

```bash
gmd-finetune model.backbone.name=mace-large model.backbone.pretrained=true model.backbone.variant=large
```

**2. Local checkpoint:**

```bash
gmd-finetune model.backbone.name=mace-large model.backbone.local_checkpoint=/path/to/model.pt
```

**3. Fresh backbone (train from scratch):**

```bash
gmd-finetune model.backbone.name=mace-large
```

### Freeze Backbone (Linear Probing)

```bash
gmd-finetune training.freeze_backbone=true model.backbone.name=mace-large model.backbone.pretrained=true
```

### Gradual Unfreezing

Use the backbone finetuning callback:

```bash
gmd-finetune callbacks=backbone_finetuning model.backbone.name=mace-large model.backbone.pretrained=true
```

This freezes the backbone initially, then unfreezes at epoch 10 with a reduced learning rate (10% of head LR).

### Available Adapters

| Adapter | Registry Names | Source |
|---------|---------------|--------|
| MACE | `mace-large`, `mace-medium`, `mace-small` | `mace-torch` |
| FairChem/UMA | `uma-small` | `fairchem-core` |

---

## Data Loading

### Supported Formats

| Format | Config | File Types | Description |
|--------|--------|-----------|-------------|
| ExtXYZ | `data=xyz` | `.xyz`, `.extxyz` | ASE-readable extended XYZ files |
| HDF5 | `data=hdf5` | `.h5`, `.hdf5` | Pre-processed atomic graphs with random access |
| LMDB | `data=lmdb` | `data.mdb` | FairChem/OCP-compatible format |
| Trajectory | `data=trajectory` | `.traj` | ASE trajectory files from MD simulations |

### Loading Modes

GMD supports four data loading modes, automatically detected from the config:

#### Mode 1 — Single source, auto-split (default)

Point to a single directory. GMD first looks for named split files (`train.xyz`, `val.xyz`, `test.xyz`). If those don't exist, it loads everything and splits by ratio.

```bash
gmd-train data.root=/path/to/dataset
```

```yaml
# configs/data/xyz.yaml
data:
  dataset_type: xyz
  root: ${paths.data_dir}
  split_ratio: [0.8, 0.1, 0.1]
  split_seed: 42
```

#### Mode 2 — Per-split paths

Specify separate file lists for train, validation, and test. Each split can load from multiple files.

```bash
gmd-train data.train_paths='[/data/A/train.xyz,/data/B/train.xyz]' \
          data.val_paths='[/data/A/val.xyz]' \
          data.test_paths='[/data/A/test.xyz]'
```

Or in a config file:

```yaml
data:
  dataset_type: xyz
  train_paths:
    - /data/dataset_A/train.xyz
    - /data/dataset_B/train.xyz
  val_paths:
    - /data/dataset_A/val.xyz
  test_paths:
    - /data/dataset_A/test.xyz
```

#### Mode 3 — Merged multi-source, auto-split

Provide a list of roots. All datasets are loaded, merged into one, then split by ratio.

```yaml
data:
  dataset_type: xyz
  root:
    - /data/dataset_A
    - /data/dataset_B
    - /data/dataset_C
  merge_strategy: random
  split_ratio: [0.8, 0.1, 0.1]
  split_seed: 42
```

#### Mode 4 — Directory-based per-split 🆕

Point to directories containing data files. All matching files (`.xyz`, `.extxyz`, `.h5`, `.hdf5`, `.lmdb`, `.traj`, `.db`) inside each directory are automatically discovered and loaded.

```bash
gmd-train data.train_dir=/data/train/ \
          data.val_dir=/data/val/ \
          data.test_dir=/data/test/
```

```yaml
data:
  dataset_type: xyz
  train_dir: /data/splits/train/
  val_dir: /data/splits/val/
  test_dir: /data/splits/test/     # optional
```

> **Tip:** Mode 4 is ideal when you have pre-organized split directories. Files are loaded in sorted order for reproducibility.

### Merge Strategies

When loading multiple files (Mode 2 or Mode 3), datasets are merged using one of two strategies:

| Strategy | Behaviour |
|----------|-----------|
| `sequential` | Concatenate datasets in order (default) |
| `random` | Shuffle all indices after concatenation (seed-controlled) |

```bash
gmd-train data.merge_strategy=random data.split_seed=123
```

### Auto-Splitting

When splits aren't provided explicitly, GMD splits the dataset numerically:

```yaml
data:
  split_ratio: [0.8, 0.1, 0.1]   # train / val / test
  split_seed: 42                   # reproducible splits
```

A two-element ratio creates train/val only (no test split):

```yaml
data:
  split_ratio: [0.9, 0.1]         # train / val only
```

### DataLoader Options

All data configs support these performance options:

```yaml
data:
  batch_size: 32
  num_workers: 4              # parallel data loading workers
  pin_memory: true            # pin tensors in CPU memory for faster GPU transfer
  persistent_workers: true    # keep workers alive between epochs
  prefetch_factor: 2          # batches prefetched per worker
```

---

## Models

### HyperSpec (equivariant)

E(3)-equivariant graph neural network using spherical harmonics and tensor products.

```bash
gmd-train model=hyperspec
```

Key parameters:
- `hidden_channels: 128` — feature dimension
- `num_interactions: 3` — message passing layers
- `lmax: 2` — maximum spherical harmonics order
- `cutoff: 5.0` — interaction radius (Å)
- `num_radial_basis: 8` — radial basis functions

Output irreps: `128x0e+128x1o+128x2e` (scalars + vectors + rank-2 tensors)

### Invariant GNN

SchNet-like invariant backbone using only scalar features. Faster than equivariant models; use for baselines or when equivariance isn't needed.

```bash
gmd-train model=invariant_gnn
```

Output irreps: `128x0e` (scalars only)

---

## Heads

Task-specific output heads registered via the head registry:

| Head | Description | Config key |
|------|-------------|-----------|
| `energy_forces` | Energy prediction + force via autograd | `head.name: energy_forces` |
| `energy` | Energy prediction only | `head.name: energy` |
| `direct_forces` | Direct force prediction (no autograd) | `head.name: direct_forces` |
| `stress` | Stress tensor prediction | `head.name: stress` |
| `dipole` | Dipole moment prediction | `head.name: dipole` |

Override the head:

```bash
gmd-train model.head.name=stress model.head.compute_stress=true
```

---

## Loss Functions

Each property loss supports a configurable loss function via the `fn` parameter:

| Key | Function | Notes |
|-----|----------|-------|
| `mse` | Mean Squared Error | Default — good for smooth regression |
| `mae` / `l1` | Mean Absolute Error | Robust to outliers |
| `rmse` | Root Mean Squared Error | Penalises large errors more than MAE |
| `huber` | Huber Loss | Combines MSE + MAE (delta = 1.0) |
| `smooth_l1` | Smooth L1 | Like Huber with beta = 1.0 |

Configure per-property in `configs/training/default.yaml`:

```yaml
losses:
  - name: energy
    weight: 4.0
    fn: mse          # ← loss function
  - name: forces
    weight: 1.0
    fn: huber         # ← robust to noisy forces
  - name: stress
    weight: 0.01
    fn: mae
```

### Composite Loss per Property

Use **multiple loss functions simultaneously** for the same property, each with its own weight and separate logging panel:

```yaml
losses:
  - name: energy
    weight: 4.0
    fn: mse
  - name: forces
    fn:                    # ← list of sub-losses
      - name: mse
        weight: 4.0
      - name: rmse
        weight: 8.0
```

This produces **five** logged metrics in W&B / TensorBoard:

| Logged metric | Description |
|---------------|-------------|
| `train/energy` | Energy MSE × 4.0 |
| `train/forces_mse` | Forces MSE × 4.0 |
| `train/forces_rmse` | Forces RMSE × 8.0 |
| `train/forces` | Sum of forces sub-losses |
| `train/total` | Grand total |

Each sub-loss gets its own chart in W&B automatically.

### Custom / torchmetrics Loss Functions

Use any callable via a dotted import path:

```yaml
losses:
  - name: forces
    fn:
      - name: mse
        weight: 4.0
      - name: torchmetrics.functional.mean_squared_error
        weight: 2.0
```

Install torchmetrics first: `pip install -e ".[torchmetrics]"`

Override from the CLI:

```bash
# Switch forces loss to MAE
gmd-train 'training.losses=[{name: energy, weight: 4.0, fn: mse}, {name: forces, weight: 1.0, fn: mae}]'
```

> **Tip:** Use `huber` or `mae` for forces when your dataset has noisy DFT reference forces — they're more robust to outliers than MSE.

---

## Benchmark Datasets

Ready-to-use benchmark datasets for training and evaluating MLIPs. **Completely optional** — the core framework works without them.

| Dataset | Structures | Elements | Properties | Size | Config |
|---------|-----------|----------|------------|------|--------|
| **MD17** | ~10k/mol | H, C, N, O | energy, forces | ~100 MB | `data=md17_aspirin` |
| **rMD17** | ~10k/mol | H, C, N, O | energy, forces | ~100 MB | `data=rmd17_aspirin` |
| **ANI-1** | ~20M | H, C, N, O | energy, forces | ~30 GB | `data=ani1` |
| **ANI-1x** | ~5M | H, C, N, O | energy, forces | ~7 GB | `data=ani1x` |
| **QM9** | 134k | H, C, N, O, F | 19 properties | ~1 GB | `data=qm9` |
| **SPICE** | ~1.1M | 10 elements | energy, forces | ~15 GB | — |

```bash
# Train on MD17 aspirin
gmd-train data=md17_aspirin

# Train on ANI-1x, subsample 50k for quick experiment
gmd-train data=ani1x data.max_structures=50000

# Train on QM9 predicting HOMO-LUMO gap
gmd-train data=qm9 data.target=gap

# Override cutoff
gmd-train data=md17_aspirin data.cutoff=6.0
```

Install optional dependencies for SPICE (HDF5):

```bash
pip install -e ".[examples]"
```

See [examples/datasets/README.md](examples/datasets/README.md) for full documentation, citations, and unit conversion details.

---

## Foundation Model Adapters

Adapters wrap pre-trained foundation models (MACE, FairChem/UMA) as GMD backbones. They translate between the foundation model's interface and GMD's backbone protocol.

```bash
# Fine-tune MACE-large
gmd-finetune model.backbone.name=mace-large model.backbone.pretrained=true

# Fine-tune UMA-small
gmd-finetune model.backbone.name=uma-small model.backbone.pretrained=true
```

Install adapter dependencies:

```bash
pip install -e ".[mace]"       # for MACE adapters
pip install -e ".[fairchem]"   # for FairChem/UMA adapters
```

---

## Feature Extraction

Extract intermediate node features from any backbone for downstream analysis, transfer learning, or custom heads.

### HookBasedExtractor

Attach forward hooks to interaction blocks — works with any model whose layers are a `nn.ModuleList`:

```python
from gmd.utils.extraction import HookBasedExtractor

with HookBasedExtractor(model, blocks_attr="interactions", output_index=0) as ext:
    output = model(batch)
    features = ext.captured  # {"layer_0": Tensor, "layer_1": Tensor, ...}
```

### Composable Backbone Wrappers

| Wrapper | Description |
|---------|-------------|
| `LayerBackbone` | Returns features from a single interaction layer |
| `MultiScaleBackbone` | Concatenates features from multiple layers |
| `FrozenBackbone` | Freezes all backbone parameters for feature extraction |

### Irrep Helpers

```python
from gmd.utils.extraction import extract_scalars, extract_irrep_channels, pool_nodes

scalars = extract_scalars(node_feats, irreps)           # l=0 channels only
channels = extract_irrep_channels(node_feats, irreps)   # dict by irrep type
graph_feats = pool_nodes(node_feats, batch_idx)          # per-graph pooling
```

### Pre-built Extractors

Registered as Hydra targets for zero-code feature extraction:

```yaml
backbone:
  _target_: gmd.utils.extraction._build_mace_large_final       # last layer
  # or: gmd.utils.extraction._build_mace_large_multiscale      # all layers
  # or: gmd.utils.extraction._build_mace_large_frozen           # frozen weights
```

---

## Performance Engineering

### TF32 Matmul Precision

On Ampere+ GPUs (A100, H100, RTX 30xx/40xx), TF32 tensor cores provide ~3× speedup for float32 operations with negligible precision loss:

```yaml
# configs/training/default.yaml
training:
  performance:
    float32_matmul_precision: high  # "highest" = fp32, "high" = TF32+fp32, "medium" = TF32
```

### cuDNN Benchmark

Auto-tunes convolution algorithms for fixed input sizes:

```yaml
training:
  performance:
    cudnn_benchmark: true
    cudnn_deterministic: false  # set true only for debugging
```

### torch.compile

Compile the backbone with `torch.compile` for faster training (PyTorch 2.0+):

```bash
gmd-train training.compile_model=true
```

Configure compilation mode:

```yaml
training:
  compile_model: true
  compile:
    mode: default           # 'default', 'reduce-overhead', 'max-autotune'
    fullgraph: false        # true = compile the entire graph (faster, stricter)
    dynamic: null           # null, true, false — dynamic shape support
```

### Mixed Precision

```bash
gmd-train trainer.precision=bf16-mixed    # bfloat16 (Ampere+, recommended)
gmd-train trainer.precision=16-mixed       # float16
gmd-train trainer.precision=64-true        # double precision
```

### Gradient Accumulation

Simulate larger batch sizes without increasing GPU memory:

```bash
gmd-train trainer.accumulate_grad_batches=4   # effective batch = batch_size × 4
```

Or use the dynamic scheduler callback:

```bash
gmd-train callbacks=grad_accumulation
```

### Exponential Moving Average (EMA)

Maintains a shadow copy of weights for more stable evaluation:

```yaml
training:
  ema:
    enabled: true
    decay: 0.999
```

### Stochastic Weight Averaging (SWA)

Alternative to EMA — averages weights during the last portion of training:

```bash
gmd-train callbacks=swa
```

### Sanity Validation Check

Before the first training epoch, Lightning runs a short validation sanity check to catch data loading, metric computation, or model errors early. This is enabled by default:

```yaml
# configs/trainer/default.yaml
num_sanity_val_steps: 2   # run 2 val batches before training
                          # 0 = skip, -1 = full validation set
```

Override from the command line:

```bash
# Skip sanity check (faster startup)
gmd-train trainer.num_sanity_val_steps=0

# Full validation run before training (thorough check)
gmd-train trainer.num_sanity_val_steps=-1
```

---

## Hyperparameter Tuning

GMD provides two levels of hyperparameter optimisation, both fully config-driven.

### Basic: Lightning Tuner

Built-in learning rate and batch size auto-discovery. **Zero extra dependencies.**

```bash
gmd-tune hparams_search=basic
```

```yaml
# configs/hparams_search/basic.yaml
hparams_search:
  method: tuner
  tuner:
    lr_find: true             # find optimal learning rate
    scale_batch_size: true    # find max batch size that fits in memory
```

### Advanced: Ray Tune

Full hyperparameter search with ASHA early stopping, Optuna Bayesian optimisation, or Population-Based Training. **Requires optional dependencies.**

```bash
pip install -e ".[tune]"   # installs ray[tune] + optuna

gmd-tune hparams_search=ray_tune
```

<details>
<summary>Example Ray Tune config</summary>

```yaml
# configs/hparams_search/ray_tune.yaml
hparams_search:
  method: ray
  num_samples: 20
  max_epochs: 100
  metric: val/total
  mode: min
  scheduler: asha
  search_algorithm: optuna

  search_space:
    training.optimizer.lr:
      type: loguniform
      lower: 1.0e-5
      upper: 1.0e-2
    training.optimizer.weight_decay:
      type: loguniform
      lower: 1.0e-8
      upper: 1.0e-3
    training.ema.decay:
      type: uniform
      lower: 0.99
      upper: 0.9999
```

</details>

| Scheduler | Description |
|-----------|-------------|
| `asha` | Asynchronous Successive Halving — prunes bad trials early (recommended) |
| `pbt` | Population-Based Training — mutates hyperparams during training |

| Search algorithm | Description |
|-----------------|-------------|
| `null` | Random search (no extra deps) |
| `optuna` | Bayesian optimisation via [Optuna](https://optuna.org/) |
| `hyperopt` | Tree-structured Parzen Estimators |

### Advanced: W&B Sweeps

Cloud-managed hyperparameter search via [Weights & Biases](https://wandb.ai/). Supports Bayesian, grid, and random search with Hyperband early termination. **Requires W&B (already a core dependency).**

```bash
gmd-tune hparams_search=wandb_sweep
```

<details>
<summary>Example W&B Sweep config</summary>

```yaml
# configs/hparams_search/wandb_sweep.yaml
hparams_search:
  method: wandb
  project: gmd
  sweep_method: bayes          # 'bayes', 'grid', 'random'
  metric: val/total
  mode: min
  count: 20

  early_terminate:
    type: hyperband
    min_iter: 10
    eta: 3

  parameters:
    training.optimizer.lr:
      distribution: log_uniform_values
      min: 1.0e-5
      max: 1.0e-2
    training.optimizer.weight_decay:
      distribution: log_uniform_values
      min: 1.0e-8
      max: 1.0e-3
```

</details>

Resume an existing sweep:

```bash
gmd-tune hparams_search=wandb_sweep hparams_search.sweep_id=<SWEEP_ID>
```

| Sweep method | Description |
|-------------|-------------|
| `bayes` | Bayesian optimisation (Gaussian process) — recommended |
| `grid` | Exhaustive grid search |
| `random` | Random search |

---

## Callbacks

### Default Callbacks

The default callback group (`callbacks=default`) includes:
- **ModelCheckpoint** — save top-k checkpoints by validation loss, plus `last.ckpt`
- **EarlyStopping** — stop training after 100 epochs with no improvement
- **RichModelSummary** — rich-formatted model summary
- **RichProgressBar** — rich-formatted training progress

### Additional Callbacks

| Callback | Config | Description |
|----------|--------|-------------|
| Stochastic Weight Averaging | `callbacks=swa` | Average weights during late training |
| Backbone Finetuning | `callbacks=backbone_finetuning` | Gradual unfreezing for fine-tuning |
| Gradient Accumulation Scheduler | `callbacks=grad_accumulation` | Dynamic accumulation steps |

Override callback parameters:

```bash
gmd-train callbacks.model_checkpoint.save_top_k=5
gmd-train callbacks.early_stopping.patience=200
```

---

## Logging

GMD supports all Lightning loggers. Enable via the `logger` config group:

```bash
gmd-train logger=wandb
gmd-train logger=tensorboard
gmd-train logger=csv
```

| Logger | Config | Notes |
|--------|--------|-------|
| Weights & Biases | `logger=wandb` | Project: `gmd`, requires `wandb` login |
| TensorBoard | `logger=tensorboard` | Saves to `output_dir/tensorboard/` |
| CSV | `logger=csv` | Simple CSV file logging |
| MLflow | `logger=mlflow` | MLflow tracking server |
| Neptune | `logger=neptune` | Requires `NEPTUNE_API_TOKEN` |
| Aim | `logger=aim` | Local `.aim` repo, open with `aim up` |
| Comet | `logger=comet` | Comet.ml experiment tracking |

Use multiple loggers:

```bash
gmd-train logger=wandb,csv
```

---

## Configuration System

GMD uses [Hydra](https://hydra.cc/) for composable configuration. Every aspect of training is controlled by YAML config files that can be overridden from the command line.

### Config Groups

| Group | Path | Options |
|-------|------|---------|
| Data | `configs/data/` | `xyz`, `hdf5`, `lmdb`, `trajectory`, `md17_aspirin`, `md17_ethanol`, `rmd17_aspirin`, `ani1`, `ani1x`, `qm9` |
| Model | `configs/model/` | `hyperspec`, `invariant_gnn` |
| Trainer | `configs/trainer/` | `default`, `gpu`, `ddp`, `fsdp`, `model_parallel`, `cpu`, `mps`, `ddp_sim` |
| Training | `configs/training/` | `default` |
| Strategy | `configs/strategy/` | `ddp`, `fsdp`, `fsdp2`, `deepspeed_zero1`, `deepspeed_zero2`, `deepspeed_zero3` |
| Callbacks | `configs/callbacks/` | `default`, `none`, `swa`, `backbone_finetuning`, `grad_accumulation` |
| Logger | `configs/logger/` | `wandb`, `tensorboard`, `csv`, `mlflow`, `neptune`, `aim`, `comet` |
| Hparams Search | `configs/hparams_search/` | `basic`, `ray_tune`, `wandb_sweep` |

### Override Examples

```bash
# Change model and data format
gmd-train model=invariant_gnn data=hdf5

# Override nested parameters
gmd-train training.optimizer.lr=0.0005 training.ema.decay=0.9999

# Change loss weights
gmd-train training.losses.0.weight=1.0 training.losses.1.weight=50.0

# Multi-run sweep
gmd-train -m training.optimizer.lr=0.001,0.0005,0.0001

# Disable callbacks
gmd-train callbacks=none
```

### Output Directory

Each run creates a timestamped output directory:

```
logs/train/runs/2026-01-24_04-45-12/
├── checkpoints/
│   ├── epoch_001.ckpt
│   └── last.ckpt
├── train.log
└── .hydra/
    ├── config.yaml          # resolved config
    ├── hydra.yaml
    └── overrides.yaml       # command-line overrides
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `gmd-train` | Train a model |
| `gmd-eval` | Evaluate a checkpoint on test data |
| `gmd-finetune` | Fine-tune a pre-trained model |
| `gmd-tune` | Hyperparameter search (LR finder, Ray Tune) |

All commands accept Hydra overrides:

```bash
gmd-train trainer=ddp data=hdf5 model=invariant_gnn logger=wandb seed=42
```

Module-based invocation (equivalent):

```bash
python -m gmd.cli.train trainer=ddp data=hdf5
python -m gmd.cli.evaluate ckpt_path=/path/to/ckpt
python -m gmd.cli.finetune model.backbone.pretrained=true
python -m gmd.cli.tune hparams_search=basic
```

---

## Pixi Tasks

If using pixi as your environment manager, these tasks are available:

| Task | Command | Description |
|------|---------|-------------|
| `pixi run train` | `python -m gmd.cli.train` | Train a model |
| `pixi run eval` | `python -m gmd.cli.evaluate` | Evaluate a checkpoint |
| `pixi run finetune` | `python -m gmd.cli.finetune` | Fine-tune a model |
| `pixi run tune` | `python -m gmd.cli.tune` | Hyperparameter search |
| `pixi run test` | `pytest -k 'not slow'` | Run fast tests |
| `pixi run test-full` | `pytest` | Run all tests |
| `pixi run lint` | `ruff check src/ tests/` | Lint code |
| `pixi run format` | `ruff format src/ tests/` | Format code |
| `pixi run typecheck` | `mypy src/gmd/` | Type check |
| `pixi run clean` | — | Remove build artifacts |
| `pixi run clean-logs` | `rm -rf logs/**` | Remove training logs |

Pass Hydra overrides through pixi:

```bash
pixi run train trainer=ddp data.root=/path/to/data
```

Use the `cuda-deepspeed` environment for DeepSpeed training:

```bash
pixi run -e cuda-deepspeed train strategy=deepspeed_zero2
```

Use the `tune` environment for hyperparameter search:

```bash
pixi run -e tune tune hparams_search=ray_tune
```

---

## License

This project is licensed under the MIT License.
