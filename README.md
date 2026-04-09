<div align="center">

<!-- GOAL: General Open Atomistic Laboratory -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/GOAL-General_Open_Atomistic_Laboratory-00c853?style=for-the-badge&labelColor=1a1a2e">
  <img alt="GOAL" src="https://img.shields.io/badge/GOAL-General_Open_Atomistic_Laboratory-00c853?style=for-the-badge&labelColor=263238">
</picture>

<br/>

# ⚛️ GOAL

### *Your atoms. Your rules. Your laboratory.*

A modular, open-source framework for building, training, and deploying machine-learning interatomic potentials — from quick experiments to production-scale distributed workflows.

<br/>

[![python](https://img.shields.io/badge/Python_3.14+-3776AB?logo=python&logoColor=white)](https://python.org)
[![pytorch](https://img.shields.io/badge/PyTorch_2.10+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/Lightning_2.6+-792EE5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/Hydra_1.3-89B8CD?logo=dropbox&logoColor=white)](https://hydra.cc/)
[![cuda](https://img.shields.io/badge/CUDA_12+-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)
[![repo](https://img.shields.io/badge/GitHub-MC--GLOW-181717?logo=github)](https://github.com/Nourollah/GOAL)

> **Python 3.14** · GIL-free interpreter with true multithreading for data loading and preprocessing.

</div>

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🗺️ Navigation
<!-- ═══════════════════════════════════════════════════════════════════ -->

> 🔵 **Getting started** · 🟢 **Core workflow** · 🟡 **Advanced features** · 🟣 **Infrastructure**

| 🔵 Start Here | 🟢 Train & Evaluate | 🟡 Go Deeper | 🟣 Under the Hood |
|:---|:---|:---|:---|
| [Overview](#-overview) | [Training](#-training) | [Models & Heads](#-models) | [Configuration System](#-configuration-system) |
| [Installation](#-installation) | [Evaluation](#-evaluation) | [Loss Functions](#-loss-functions) | [Logging](#-logging) |
| [Project Structure](#-project-structure) | [ASE Calculator](#-ase-calculator) | [Foundation Model Adapters](#-foundation-model-adapters) | [Callbacks](#-callbacks) |
| [Quick Start](#-quick-start) | [Fine-Tuning](#-fine-tuning) | [Feature Extraction](#-feature-extraction) | [CLI Reference](#-cli-reference) |
| | [Data Loading](#-data-loading) | [Performance Engineering](#-performance-engineering) | [Pixi Tasks](#-pixi-tasks) |
| | [Benchmark Datasets](#-benchmark-datasets) | [Hyperparameter Tuning](#-hyperparameter-tuning) | |
| | | [Mini Trainer](#-mini-trainer) | |
| | | [Customising the Training Loop](#-customising-the-training-loop) | |

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🔵 Overview
<!-- ═══════════════════════════════════════════════════════════════════ -->

**GOAL** (**G**eneral **O**pen **A**tomistic **L**aboratory) is a modular framework for training machine-learning interatomic potentials (MLIPs). Built on **PyTorch Lightning 2.6+** and **Hydra**, it provides:

| | Feature | Details |
|---|---|---|
| 🔬 | **Equivariant & invariant backbones** | HyperSpec (E(3)-equivariant) and SchNet-like invariant GNN |
| 🎯 | **Multiple task heads** | Energy, forces, stress, dipole, direct forces |
| 🧠 | **Foundation model adapters** | MACE and FairChem (UMA) pre-trained models |
| 📂 | **Flexible data loading** | XYZ, HDF5, LMDB, ASE trajectory; multi-file merge, directory-based loading, auto-splitting |
| ⚡ | **Distributed training** | DDP, FSDP, FSDP2 (ModelParallel), DeepSpeed ZeRO (Stages 1/2/3 + CPU offload) |
| 🧮 | **Configurable loss** | Per-property loss type (MSE, MAE, Huber, Smooth L1) + composite sub-losses |
| 🔧 | **Strategy factory** | Unified `build_strategy(cfg)` for all distributed strategies |
| 🚀 | **Performance engineering** | TF32, cuDNN benchmark, `torch.compile`, gradient accumulation, EMA/SWA |
| 📊 | **Experiment management** | Hydra config composition · W&B · TensorBoard · CSV · MLflow · Neptune · Aim · Comet |
| 🖥️ | **SLURM-aware** | Auto checkpoint resumption and completion sentinels |
| 🧪 | **ASE integration** | Use any trained model as an ASE Calculator for MD, geometry optimisation, phonons |
| 🐍 | **Python 3.14** | GIL-free interpreter with real multithreading for data loading |
| 🔬 | **Mini Trainer** | Standalone notebook-friendly training loop for rapid prototyping on extracted features |
| 🧰 | **Custom training loops** | Three levels of loop customisation: GOALModule hooks, Fabric-based multi-GPU, or pure PyTorch |

> **Package layout** · The top-level namespace is `goal`. The ML training module lives at `goal.ml`:
> ```python
> from goal.ml.training.module import GOALModule
> from goal.ml.data.datamodule import GOALDataModule
> from goal.ml.utils.calculator import GOALCalculator
> ```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🔵 Installation
<!-- ═══════════════════════════════════════════════════════════════════ -->

### With pip

```bash
git clone https://github.com/Nourollah/GOAL.git
cd GOAL
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

### With pixi (recommended)

<details>
<summary><b>💡 What is pixi?</b></summary>

<br/>

[**Pixi**](https://pixi.sh/) is a fast, cross-platform package manager built on top of conda-forge. It manages **both** conda and pip dependencies in a single lockfile, giving you:

- **Reproducible environments** — a `pixi.lock` pins every package version (conda *and* pip)
- **Named environments** — switch between CPU, CUDA, dev, and adapter-specific setups instantly
- **No `conda activate`** — just `pixi run <task>` or `pixi shell`
- **Fast solves** — written in Rust; resolves environments in seconds

**Install pixi** (one-liner):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Or see the [official installation guide](https://pixi.sh/latest/#installation) for Homebrew, Windows, and other methods.

</details>

<br/>

GOAL ships a pixi workspace configuration in `pyproject.toml`. After installing pixi:

```bash
pixi install                    # default (CPU)
pixi install -e cuda            # CUDA 12+
pixi install -e dev             # CPU + dev tools (pytest, ruff, mypy)
pixi install -e dev-cuda        # CUDA + dev tools
pixi install -e fairchem        # CUDA + FairChem adapter
pixi install -e cuda-deepspeed  # CUDA + DeepSpeed
```

> **Note:** Some optional dependencies have compatibility constraints:
> - **MACE adapter** — pins `e3nn==0.4.4` which conflicts with the core `e3nn>=0.5` requirement. Install via `pip install -e ".[mace]"` instead.
> - **Ray Tune / Optuna** — no Python 3.14 wheels yet. Install via `pip install -e ".[tune]"` on Python ≤3.13.

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🔵 Project Structure
<!-- ═══════════════════════════════════════════════════════════════════ -->

<details>
<summary>📁 <b>Click to expand full project tree</b></summary>

```
├── configs/                        # Hydra configuration groups
│   ├── train.yaml                  #   Training defaults composition
│   ├── eval.yaml                   #   Evaluation defaults composition
│   ├── callbacks/                  #   Callback configs (checkpoint, EMA, SWA, …)
│   ├── data/                       #   Dataset configs (xyz, hdf5, lmdb, trajectory, benchmarks)
│   ├── hparams_search/             #   Hyperparameter search (basic, ray_tune, wandb_sweep)
│   ├── logger/                     #   Logger configs (wandb, tensorboard, csv, …)
│   ├── model/                      #   Model configs (hyperspec, invariant_gnn)
│   ├── strategy/                   #   Strategy configs (ddp, fsdp, fsdp2, deepspeed_*)
│   ├── trainer/                    #   Trainer configs (gpu, ddp, fsdp, model_parallel, …)
│   ├── training/                   #   Training hyperparameters (optimizer, EMA, losses, …)
│   ├── paths/                      #   Path definitions
│   └── hydra/                      #   Hydra runtime settings
├── src/
│   └── goal/                       # Top-level namespace package
│       └── ml/                     #   ML training module
│           ├── cli/                #     Entry points: train, evaluate, finetune, tune
│           ├── data/               #     DataModule, datasets (xyz, hdf5, lmdb, trajectory, concat)
│           ├── nn/                 #     Neural network components
│           │   ├── models/         #       Backbones: HyperSpec, invariant GNN
│           │   ├── heads/          #       Task heads: energy, forces, stress, dipole
│           │   ├── blocks/         #       Building blocks: embedding, interaction, readout
│           │   └── primitives/     #       Low-level ops: tensor products, radial basis, norms
│           ├── adapters/           #     Foundation model wrappers: MACE, FairChem
│           ├── training/           #     LightningModule, loss, EMA, tuning
│           │   ├── callbacks/      #       Checkpoint, logging callbacks
│           │   └── strategies/     #       Strategy factory: DDP, FSDP, FSDP2, DeepSpeed
│           ├── utils/              #     ASE calculator, feature extraction, mini trainer
│           └── registry.py         #     Lazy component registry
├── examples/
│   └── datasets/                   # Benchmark dataset loaders (MD17, ANI-1, QM9, SPICE)
├── notebooks/                      # Demo notebooks (feature extraction, mini trainer)
├── scripts/                        # SLURM job scripts
├── tests/                          # Test suite
├── data/                           # Dataset storage
├── logs/                           # Training outputs (checkpoints, metrics)
└── pyproject.toml                  # Package metadata + pixi workspace config
```

</details>

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🔵 Quick Start
<!-- ═══════════════════════════════════════════════════════════════════ -->

Train the default model (HyperSpec) on XYZ data:

```bash
goal-train data.root=/path/to/dataset
```

Or equivalently via module:

```bash
python -m goal.ml.cli.train data.root=/path/to/dataset
```

This loads `configs/train.yaml` which composes: `data=xyz`, `model=hyperspec`, `training=default`, `trainer=default`.

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟢 Training
<!-- ═══════════════════════════════════════════════════════════════════ -->

### Single GPU

```bash
goal-train trainer=gpu data.root=/path/to/dataset
```

The `gpu` trainer config sets `accelerator: gpu` and `devices: 1`.

### Multi-GPU: DDP

Distributed Data Parallel — replicates the full model on each GPU and synchronizes gradients. Use when the model fits in a single GPU's memory.

```bash
goal-train trainer=ddp data.root=/path/to/dataset
```

Override the number of GPUs:

```bash
goal-train trainer=ddp trainer.devices=8
```

Multi-node:

```bash
goal-train trainer=ddp trainer.devices=4 trainer.num_nodes=2
```

DDP key settings (in `configs/trainer/ddp.yaml`):
- `find_unused_parameters: false` — set `true` if you have frozen layers
- `static_graph: false` — set `true` for models with fixed computation graphs (faster)
- `gradient_as_bucket_view: true` — minor memory optimisation
- `sync_batchnorm: true` — synchronize batch norm statistics across GPUs

### Multi-GPU: FSDP

Fully Sharded Data Parallel — shards model parameters, gradients, and optimizer states across GPUs. Use when the model doesn't fit in a single GPU's memory.

```bash
goal-train trainer=fsdp data.root=/path/to/dataset
```

FSDP settings (in `configs/trainer/fsdp.yaml`):
- `auto_wrap_policy` — controls how modules are wrapped for sharding
- `activation_checkpointing_policy` — trade compute for memory by recomputing activations
- `cpu_offload: false` — offload parameters to CPU (slower, saves GPU memory)
- `precision: "bf16-mixed"` — recommended for FSDP on Ampere+ GPUs

### Multi-GPU: FSDP2 / ModelParallel

ModelParallelStrategy (Lightning 2.4+) — supports FSDP2, tensor parallelism, `torch.compile`, and FP8. Recommended for very large models (500M+ parameters).

```bash
goal-train trainer=model_parallel data.root=/path/to/dataset
```

Or via the strategy factory:

```bash
goal-train +strategy=fsdp2 data.root=/path/to/dataset
```

### Multi-GPU: DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) ZeRO enables training of very large models by partitioning optimizer states, gradients, and parameters across GPUs. Requires `pip install -e ".[deepspeed]"`.

**ZeRO Stage 1** — optimizer state partitioning only (lowest communication overhead):
```bash
goal-train +strategy=deepspeed_zero1 data.root=/path/to/dataset
```

**ZeRO Stage 2** — optimizer state + gradient partitioning:
```bash
goal-train +strategy=deepspeed_zero2 data.root=/path/to/dataset
```

**ZeRO Stage 3** — full parameter partitioning (maximum memory savings):
```bash
goal-train +strategy=deepspeed_zero3 data.root=/path/to/dataset
```

**ZeRO Stage 3 + CPU offload** — offload parameters to CPU (for extremely large models):
```bash
goal-train +strategy=deepspeed_zero3_offload data.root=/path/to/dataset
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

GOAL provides a unified **strategy factory** (`build_strategy()`) that maps config to Lightning strategies. When `cfg.strategy` is present, it takes priority over the trainer's built-in strategy.

```yaml
# Two ways to select a strategy:
# 1. Via trainer config group (backward compatible):
goal-train trainer=ddp

# 2. Via strategy config group (new, more options):
goal-train +strategy=fsdp2
goal-train +strategy=deepspeed_zero3_offload
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
goal-train trainer=mps data.root=/path/to/dataset
```

### CPU

```bash
goal-train trainer=cpu data.root=/path/to/dataset
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟢 Evaluation
<!-- ═══════════════════════════════════════════════════════════════════ -->

Evaluate a trained checkpoint on the test split:

```bash
goal-eval ckpt_path=/path/to/checkpoint.ckpt data.root=/path/to/dataset
```

The evaluation entry point supports the same trainer configs for distributed evaluation:

```bash
goal-eval trainer=ddp ckpt_path=/path/to/checkpoint.ckpt data.root=/path/to/dataset
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟢 ASE Calculator
<!-- ═══════════════════════════════════════════════════════════════════ -->

Any trained GOAL model can be used as an [ASE Calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html) for molecular dynamics, geometry optimisation, phonons, and more.

### From a Checkpoint

```python
from goal.ml.utils.calculator import GOALCalculator

calc = GOALCalculator(checkpoint_path="logs/train/runs/.../last.ckpt")
```

### From a Pre-loaded Module

```python
from goal.ml.utils.calculator import GOALCalculator

calc = GOALCalculator(module=my_module, cutoff=5.0, device="cuda")
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
| `module` | — | Pre-loaded `GOALModule` instance |
| `cutoff` | from config | Neighbour-list cutoff (Å). Auto-detected from checkpoint |
| `device` | `"cpu"` | `"cpu"`, `"cuda"`, `"cuda:0"`, etc. |
| `dtype` | `float64` | Precision for positions and cell |
| `head` | `None` | Multi-head tag for multi-task models |

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟢 Fine-Tuning
<!-- ═══════════════════════════════════════════════════════════════════ -->

Fine-tune a pre-trained foundation model on a downstream dataset:

```bash
goal-finetune model.backbone.name=mace-large model.backbone.pretrained=true data.root=/path/to/dataset
```

### Backbone Loading Modes

**1. Pre-trained hub model:**

```bash
goal-finetune model.backbone.name=mace-large model.backbone.pretrained=true model.backbone.variant=large
```

**2. Local checkpoint:**

```bash
goal-finetune model.backbone.name=mace-large model.backbone.local_checkpoint=/path/to/model.pt
```

**3. Fresh backbone (train from scratch):**

```bash
goal-finetune model.backbone.name=mace-large
```

### Freeze Backbone (Linear Probing)

```bash
goal-finetune training.freeze_backbone=true model.backbone.name=mace-large model.backbone.pretrained=true
```

### Gradual Unfreezing

Use the backbone finetuning callback:

```bash
goal-finetune callbacks=backbone_finetuning model.backbone.name=mace-large model.backbone.pretrained=true
```

This freezes the backbone initially, then unfreezes at epoch 10 with a reduced learning rate (10% of head LR).

### Available Adapters

| Adapter | Registry Names | Source |
|---------|---------------|--------|
| MACE | `mace-large`, `mace-medium`, `mace-small` | `mace-torch` |
| FairChem/UMA | `uma-small` | `fairchem-core` |

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟢 Data Loading
<!-- ═══════════════════════════════════════════════════════════════════ -->

### Supported Formats

| Format | Config | File Types | Description |
|--------|--------|-----------|-------------|
| ExtXYZ | `data=xyz` | `.xyz`, `.extxyz` | ASE-readable extended XYZ files |
| HDF5 | `data=hdf5` | `.h5`, `.hdf5` | Pre-processed atomic graphs with random access |
| LMDB | `data=lmdb` | `data.mdb` | FairChem/OCP-compatible format |
| Trajectory | `data=trajectory` | `.traj` | ASE trajectory files from MD simulations |

### Loading Modes

GOAL supports four data loading modes, automatically detected from the config:

#### Mode 1 — Single source, auto-split (default)

Point to a single directory. GOAL first looks for named split files (`train.xyz`, `val.xyz`, `test.xyz`). If those don't exist, it loads everything and splits by ratio.

```bash
goal-train data.root=/path/to/dataset
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
goal-train data.train_paths='[/data/A/train.xyz,/data/B/train.xyz]' \
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

#### Mode 4 — Directory-based per-split

Point to directories containing data files. All matching files (`.xyz`, `.extxyz`, `.h5`, `.hdf5`, `.lmdb`, `.traj`, `.db`) inside each directory are automatically discovered and loaded.

```bash
goal-train data.train_dir=/data/train/ \
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
goal-train data.merge_strategy=random data.split_seed=123
```

### Auto-Splitting

When splits aren't provided explicitly, GOAL splits the dataset numerically:

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

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟢 Benchmark Datasets
<!-- ═══════════════════════════════════════════════════════════════════ -->

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
goal-train data=md17_aspirin

# Train on ANI-1x, subsample 50k for quick experiment
goal-train data=ani1x data.max_structures=50000

# Train on QM9 predicting HOMO-LUMO gap
goal-train data=qm9 data.target=gap

# Override cutoff
goal-train data=md17_aspirin data.cutoff=6.0
```

Install optional dependencies for SPICE (HDF5):

```bash
pip install -e ".[examples]"
```

See [examples/datasets/README.md](examples/datasets/README.md) for full documentation, citations, and unit conversion details.

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟡 Models
<!-- ═══════════════════════════════════════════════════════════════════ -->

### HyperSpec (equivariant)

E(3)-equivariant graph neural network using spherical harmonics and tensor products.

```bash
goal-train model=hyperspec
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
goal-train model=invariant_gnn
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
goal-train model.head.name=stress model.head.compute_stress=true
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟡 Loss Functions
<!-- ═══════════════════════════════════════════════════════════════════ -->

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
goal-train 'training.losses=[{name: energy, weight: 4.0, fn: mse}, {name: forces, weight: 1.0, fn: mae}]'
```

> **Tip:** Use `huber` or `mae` for forces when your dataset has noisy DFT reference forces — they're more robust to outliers than MSE.

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟡 Foundation Model Adapters
<!-- ═══════════════════════════════════════════════════════════════════ -->

Adapters wrap pre-trained foundation models (MACE, FairChem/UMA) as GOAL backbones. They translate between the foundation model's interface and GOAL's backbone protocol.

```bash
# Fine-tune MACE-large
goal-finetune model.backbone.name=mace-large model.backbone.pretrained=true

# Fine-tune UMA-small
goal-finetune model.backbone.name=uma-small model.backbone.pretrained=true
```

Install adapter dependencies:

```bash
pip install -e ".[mace]"       # for MACE adapters
pip install -e ".[fairchem]"   # for FairChem/UMA adapters
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟡 Feature Extraction
<!-- ═══════════════════════════════════════════════════════════════════ -->

Extract intermediate node features from any backbone for downstream analysis, transfer learning, or custom heads.

### HookBasedExtractor

Attach forward hooks to interaction blocks — works with any model whose layers are a `nn.ModuleList`:

```python
from goal.ml.utils.extraction import HookBasedExtractor

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
from goal.ml.utils.extraction import extract_scalars, extract_irrep_channels, pool_nodes

scalars = extract_scalars(node_feats, irreps)           # l=0 channels only
channels = extract_irrep_channels(node_feats, irreps)   # dict by irrep type
graph_feats = pool_nodes(node_feats, batch_idx)          # per-graph pooling
```

### Pre-built Extractors

Registered as Hydra targets for zero-code feature extraction:

```yaml
backbone:
  _target_: goal.ml.utils.extraction._build_mace_large_final       # last layer
  # or: goal.ml.utils.extraction._build_mace_large_multiscale      # all layers
  # or: goal.ml.utils.extraction._build_mace_large_frozen           # frozen weights
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟡 Mini Trainer
<!-- ═══════════════════════════════════════════════════════════════════ -->

A standalone, lightweight training loop for rapid prototyping in Jupyter notebooks. Completely decoupled from the Lightning / Hydra pipeline — operates on raw PyTorch primitives.

**Typical workflow:**
1. Freeze a foundation model (MACE, FairChem, etc.) and extract representations
2. Cache extracted features as a `TensorDataset`
3. Train a downstream head with `MiniTrainer` — iterate fast without re-running the backbone

### Basic Usage

```python
from goal.ml.utils.mini_trainer import MiniTrainer

trainer = MiniTrainer(
    model=my_head,
    loss_fn=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam(my_head.parameters(), lr=1e-3),
    device="auto",
)
history = trainer.fit(train_loader, val_loader=val_loader, epochs=50)
history.plot()  # loss curves in the notebook
```

### Features

| Feature | Description |
|---------|-------------|
| **Early stopping** | Stop when validation loss plateaus (`early_stopping_patience`) |
| **Best checkpoint** | In-memory best model state, restore with `trainer.load_best()` |
| **LR scheduling** | Any PyTorch scheduler (ReduceLROnPlateau, cosine, etc.) |
| **Gradient clipping** | Max-norm clipping via `grad_clip` parameter |
| **Progress bars** | `tqdm.auto` progress bars per epoch |
| **History** | `TrainingHistory` with `.plot()`, `.best_val_loss`, `.best_epoch` |
| **Prediction** | `trainer.predict(loader)` returns `(preds, targets)` tensors |
| **Custom step** | Plug in `step_fn` for `AtomicGraph` batches or arbitrary logic |

### With AtomicGraph Batches

For training on graph data with `CompositeLoss`, use the built-in `graph_step`:

```python
from goal.ml.utils.mini_trainer import MiniTrainer, graph_step

trainer = MiniTrainer(
    model=my_backbone_plus_head,
    loss_fn=composite_loss,
    optimizer=optimizer,
    step_fn=graph_step,  # handles AtomicGraph batches
)
history = trainer.fit(graph_train_loader, graph_val_loader, epochs=50)
```

### Notebook Demo

See [`notebooks/mini_trainer_demo.ipynb`](notebooks/mini_trainer_demo.ipynb) for a complete walkthrough — from feature extraction to model evaluation with parity plots.

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟡 Customising the Training Loop
<!-- ═══════════════════════════════════════════════════════════════════ -->

GOAL provides **three levels** of training loop customisation, from least to most control:

| Level | Tool | Multi-GPU | Loop Control | Best For |
|:---:|---|:---:|---|---|
| 1 | **GOALModule** hooks + callbacks | ✅ | Partial — override hooks | Standard workflows with minor tweaks |
| 2 | **FabricTrainer** | ✅ | Full — write your own `for` loop | Custom optimisation, multi-optimiser, GAN-style |
| 3 | **MiniTrainer** | ❌ | Full — pure PyTorch | Quick notebook prototyping on extracted features |

### Level 1: Override GOALModule Hooks

The standard Lightning path. Subclass `GOALModule` and override any hook:

```python
from goal.ml.training.module import GOALModule

class MyModule(GOALModule):
    """Custom training step with auxiliary loss."""

    def training_step(self, batch, batch_idx):
        predictions = self(batch)
        losses = self.loss(predictions, batch)

        # --- Your custom logic here ---
        aux_loss = self.compute_auxiliary_loss(predictions, batch)
        losses["total"] = losses["total"] + 0.1 * aux_loss
        # --------------------------------

        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            batch_size=batch.num_graphs, sync_dist=True,
        )
        return losses["total"]
```

Register it in Hydra and use the standard `goal-train` CLI as usual.

**What you can override:**

| Hook | When it runs |
|------|-------------|
| `training_step(batch, batch_idx)` | Each training batch |
| `validation_step(batch, batch_idx)` | Each validation batch |
| `configure_optimizers()` | Optimizer + scheduler setup |
| `configure_model()` | Pre-training model transforms (compile, FSDP wrap) |
| `on_before_optimizer_step(optimizer)` | Before each optimizer step (gradient clipping) |
| `on_train_batch_end(outputs, batch, batch_idx)` | After each training step (EMA update) |

You can also inject logic via **Lightning callbacks** without subclassing:

```python
from lightning import Callback

class GradientMonitorCallback(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(pl_module.parameters(), float("inf"))
        pl_module.log("grad_norm", grad_norm)
```

### Level 2: FabricTrainer (Full Loop Control + Multi-GPU)

When Lightning hooks are not enough — you need full control over the `for` loop **and** distributed training. Built on [Lightning Fabric](https://lightning.ai/docs/fabric/).

```python
from goal.ml.utils.fabric_trainer import FabricTrainer, graph_fabric_step

ft = FabricTrainer(
    model=my_model,
    loss_fn=composite_loss,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    # --- Distributed config (same options as Lightning Trainer) ---
    accelerator="gpu",
    strategy="ddp",       # or "fsdp", "deepspeed", etc.
    devices=4,
    precision="bf16-mixed",
    # --- Loop options ---
    step_fn=graph_fabric_step,
    grad_clip=10.0,
    grad_accumulation_steps=4,
)
history = ft.fit(epochs=100, early_stopping_patience=20)
```

**Or write the loop from scratch** using the `setup_fabric()` helper:

```python
from goal.ml.utils.fabric_trainer import setup_fabric

fabric = setup_fabric(strategy="ddp", devices=4, precision="bf16-mixed")

model, optimizer = fabric.setup(model, optimizer)
train_loader = fabric.setup_dataloaders(train_loader)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch)
        losses = loss_fn(predictions, batch)
        fabric.backward(losses["total"])

        # Your custom logic — anything goes:
        if epoch > 50:
            fabric.clip_gradients(model, optimizer, max_norm=1.0)

        optimizer.step()

    # Validation, logging, checkpointing — all under your control
    fabric.save("checkpoint.pt", {"model": model, "optimizer": optimizer})
```

**FabricTrainer features:**

| Feature | Description |
|---------|-------------|
| Multi-GPU / multi-node | DDP, FSDP, DeepSpeed — same strategies as Lightning |
| Mixed precision | bf16, fp16, fp64 |
| Gradient accumulation | Efficient sync-skipping via `fabric.no_backward_sync()` |
| Gradient clipping | `fabric.clip_gradients()` |
| Checkpointing | `save_checkpoint()` / `load_checkpoint()` — handles sharded saves |
| Early stopping | Built-in patience counter |
| History | Reuses `TrainingHistory` from MiniTrainer (`.plot()`, `.best_val_loss`) |

### Level 3: MiniTrainer (Pure PyTorch)

Single-device, no Lightning dependency at all. Ideal for notebook prototyping on pre-extracted features. See the [Mini Trainer](#-mini-trainer) section above.

### Choosing the Right Level

```
Need multi-GPU?
  ├── No  → MiniTrainer (Level 3)
  └── Yes
        ├── Standard loop is fine, just need custom loss/hook? → GOALModule (Level 1)
        └── Need full loop control? → FabricTrainer (Level 2)
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟡 Performance Engineering
<!-- ═══════════════════════════════════════════════════════════════════ -->

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
goal-train training.compile_model=true
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
goal-train trainer.precision=bf16-mixed    # bfloat16 (Ampere+, recommended)
goal-train trainer.precision=16-mixed       # float16
goal-train trainer.precision=64-true        # double precision
```

### Gradient Accumulation

Simulate larger batch sizes without increasing GPU memory:

```bash
goal-train trainer.accumulate_grad_batches=4   # effective batch = batch_size × 4
```

Or use the dynamic scheduler callback:

```bash
goal-train callbacks=grad_accumulation
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
goal-train callbacks=swa
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
goal-train trainer.num_sanity_val_steps=0

# Full validation run before training (thorough check)
goal-train trainer.num_sanity_val_steps=-1
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟡 Hyperparameter Tuning
<!-- ═══════════════════════════════════════════════════════════════════ -->

GOAL provides three levels of hyperparameter optimisation, all fully config-driven.

### Basic: Lightning Tuner

Built-in learning rate and batch size auto-discovery. **Zero extra dependencies.**

```bash
goal-tune hparams_search=basic
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

goal-tune hparams_search=ray_tune
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
goal-tune hparams_search=wandb_sweep
```

<details>
<summary>Example W&B Sweep config</summary>

```yaml
# configs/hparams_search/wandb_sweep.yaml
hparams_search:
  method: wandb
  project: goal
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
goal-tune hparams_search=wandb_sweep hparams_search.sweep_id=<SWEEP_ID>
```

| Sweep method | Description |
|-------------|-------------|
| `bayes` | Bayesian optimisation (Gaussian process) — recommended |
| `grid` | Exhaustive grid search |
| `random` | Random search |

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟣 Callbacks
<!-- ═══════════════════════════════════════════════════════════════════ -->

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
goal-train callbacks.model_checkpoint.save_top_k=5
goal-train callbacks.early_stopping.patience=200
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟣 Logging
<!-- ═══════════════════════════════════════════════════════════════════ -->

GOAL supports all Lightning loggers. Enable via the `logger` config group:

```bash
goal-train logger=wandb
goal-train logger=tensorboard
goal-train logger=csv
```

| Logger | Config | Notes |
|--------|--------|-------|
| Weights & Biases | `logger=wandb` | Project: `goal`, requires `wandb` login |
| TensorBoard | `logger=tensorboard` | Saves to `output_dir/tensorboard/` |
| CSV | `logger=csv` | Simple CSV file logging |
| MLflow | `logger=mlflow` | MLflow tracking server |
| Neptune | `logger=neptune` | Requires `NEPTUNE_API_TOKEN` |
| Aim | `logger=aim` | Local `.aim` repo, open with `aim up` |
| Comet | `logger=comet` | Comet.ml experiment tracking |

Use multiple loggers:

```bash
goal-train logger=wandb,csv
```

### Run Naming Convention

Every run is automatically named with a **timestamp + dataset + model** pattern:

```
{date}_{time}_{dataset_type}_{model_backbone}
```

For example: `2026-04-09_14-30-45_xyz_hyperspec`

This naming is applied consistently to:
- Output directories (`logs/train/runs/...`)
- Logger run names (W&B, TensorBoard, MLflow, etc.)
- Hydra sweep directories

Override the name from the CLI:

```bash
goal-train run_name=my_custom_experiment
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟣 Configuration System
<!-- ═══════════════════════════════════════════════════════════════════ -->

GOAL uses [Hydra](https://hydra.cc/) for composable configuration. Every aspect of training is controlled by YAML config files that can be overridden from the command line.

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
goal-train model=invariant_gnn data=hdf5

# Override nested parameters
goal-train training.optimizer.lr=0.0005 training.ema.decay=0.9999

# Change loss weights
goal-train training.losses.0.weight=1.0 training.losses.1.weight=50.0

# Multi-run sweep
goal-train -m training.optimizer.lr=0.001,0.0005,0.0001

# Disable callbacks
goal-train callbacks=none
```

### Output Directory

Each run creates a timestamped output directory:

```
logs/train/runs/2026-04-09_14-30-45_xyz_hyperspec/
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

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟣 CLI Reference
<!-- ═══════════════════════════════════════════════════════════════════ -->

| Command | Description |
|---------|-------------|
| `goal-train` | Train a model |
| `goal-eval` | Evaluate a checkpoint on test data |
| `goal-finetune` | Fine-tune a pre-trained model |
| `goal-tune` | Hyperparameter search (LR finder, Ray Tune, W&B Sweeps) |

All commands accept Hydra overrides:

```bash
goal-train trainer=ddp data=hdf5 model=invariant_gnn logger=wandb seed=42
```

Module-based invocation (equivalent):

```bash
python -m goal.ml.cli.train trainer=ddp data=hdf5
python -m goal.ml.cli.evaluate ckpt_path=/path/to/ckpt
python -m goal.ml.cli.finetune model.backbone.pretrained=true
python -m goal.ml.cli.tune hparams_search=basic
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟣 Pixi Tasks
<!-- ═══════════════════════════════════════════════════════════════════ -->

If using pixi as your environment manager, these tasks are available:

| Task | Command | Description |
|------|---------|-------------|
| `pixi run train` | `python -m goal.ml.cli.train` | Train a model |
| `pixi run eval` | `python -m goal.ml.cli.evaluate` | Evaluate a checkpoint |
| `pixi run finetune` | `python -m goal.ml.cli.finetune` | Fine-tune a model |
| `pixi run test` | `pytest -k 'not slow'` | Run fast tests |
| `pixi run test-full` | `pytest` | Run all tests |
| `pixi run lint` | `ruff check src/ tests/` | Lint code |
| `pixi run format` | `ruff format src/ tests/` | Format code |
| `pixi run typecheck` | `mypy src/goal/ml/` | Type check |
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

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 📦 Tested Versions
<!-- ═══════════════════════════════════════════════════════════════════ -->

| Package | Version |
|---------|---------|
| Python | 3.14.4 |
| PyTorch | 2.10.0 |
| Lightning | 2.6.1 |
| e3nn | 0.6.0 |
| PyG (torch-geometric) | 2.7.0 |
| Hydra | 1.3.2 |
| ASE | 3.28.0 |
| W&B | 0.25.1 |
| Rich | 13.9.4 |

---

## License

This project is licensed under the MIT License.
