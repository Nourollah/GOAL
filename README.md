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
[![repo](https://img.shields.io/badge/GitHub-GOAL-181717?logo=github)](https://github.com/Nourollah/GOAL)

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
| [Project Structure](#-project-structure) | [ASE Calculator](#-ase-calculator) · [QM Calculators](#-qm-calculators) | [Foundation Model Adapters](#-foundation-model-adapters) | [Callbacks](#-callbacks) |
| [Quick Start](#-quick-start) | [Fine-Tuning](#-fine-tuning) | [Feature Extraction](#-feature-extraction) | [CLI Reference](#-cli-reference) |
| [**Tutorial Notebook**](notebooks/getting_started.ipynb) | [Data Loading](#-data-loading) | [Performance Engineering](#-performance-engineering) | [Pixi Tasks](#-pixi-tasks) |
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
| 🔬 | **Native SIMURGH force field** | E(3)-equivariant potential-artisan models — **ARACE** (artisan + ACE alternating, the default) and the legacy ACE-first variant |
| 🧱 | **Modular & monolithic models** | Backbone→head pipeline (SIMURGH ARACE, HyperSpec, InvariantGNN) or self-contained monolithic models for external architectures |
| 🎯 | **Multiple task heads** | Energy, forces, stress, dipole, direct forces, generic scalar, **multi-head** |
| 🧠 | **Foundation-model fine-tuning** | Fine-tune MACE & UMA in-process (`head_only` / `full` / LoRA) through GOALModule; feature-extraction adapters too |
| ⚗️ | **QM & pretrained calculators** | ORCA, CP2K, Quantum ESPRESSO, VASP, UPET/PET-MAD, xTB, MACE, FlashMD via `CalculatorFactory` |
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
pip install -e ".[fairchem]"   # FairChem/UMA adapter
pip install -e ".[deepspeed]"  # DeepSpeed ZeRO strategies
pip install -e ".[all]"        # All optional dependencies
pip install -e ".[dev]"        # pytest, ruff, mypy
```

Fine-tuning & calculator dependencies (installed separately):

```bash
pip install "mace-torch>=0.3.16"  # MACE fine-tuning backbone (standalone venv — see note)
pip install fairchem-core         # UMA fine-tuning backbone
pip install peft                  # LoRA fine-tuning strategy
pip install upet                  # UPET / PET-MAD calculator (MD inference + fine-tune target)
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
> - **MACE** — `mace-torch` pins `e3nn==0.4.4`, which conflicts with the core `e3nn>=0.5`. Install it in a **standalone virtualenv** (not the pixi/core env); see [`docs/finetuning/mace.md`](docs/finetuning/mace.md).
> - **Ray Tune / Optuna** — no Python 3.14 wheels yet. Install via `pip install -e ".[tune]"` on Python ≤3.13.

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🔵 Project Structure
<!-- ═══════════════════════════════════════════════════════════════════ -->

<details>
<summary>📁 <b>Click to expand full project tree</b></summary>

```
├── configs/                        # Hydra configuration
│   ├── ml/                         #   Self-contained ML experiment configs
│   │   ├── simurgh_gmd26.yaml      #     DEFAULT — SIMURGH ARACE on GMD-26
│   │   ├── simurgh_gmd26_hpo.yaml  #     ARACE hyperparameter search (Ray Tune)
│   │   ├── simurgh_ace_first_gmd26.yaml  # legacy ACE-first SIMURGH
│   │   ├── monolithic_arace_gmd26.yaml   # monolithic ARACE (ablation)
│   │   ├── simurgh_md17.yaml       #     legacy SIMURGH on MD17
│   │   └── hyperspec_md17.yaml     #     HyperSpec baseline on MD17
│   ├── md/                         #   MD simulation configs (calculators, dynamics, MTS)
│   ├── hparams_search/             #   Hyperparameter search schemas (basic, ray_tune, wandb_sweep)
│   ├── debug/                      #   Debug presets (overfit, profiler, limit, fdr)
│   ├── extras/, hydra/, paths/     #   Runtime settings
├── src/
│   └── goal/                       # Top-level namespace package
│       └── ml/                     #   ML training module
│           ├── cli/                #     Entry points: train, evaluate, finetune, tune, pack, inspect
│           ├── data/               #     DataModule, datasets (xyz, hdf5, lmdb, trajectory, concat)
│           ├── nn/                 #     Neural network components
│           │   ├── models/         #       Backbones: SIMURGH (ARACE + legacy), HyperSpec, InvariantGNN; monolithic models
│           │   │   └── simurgh/    #         backbone_arace (default), backbone (legacy), monolithic, arace
│           │   ├── heads/          #       Task heads: energy, forces, stress, dipole, scalar, multi
│           │   ├── blocks/         #       Building blocks: artisans, ace_block, env_dressing, interaction, embedding, readout
│           │   └── primitives/     #       Low-level ops: tensor products, radial basis, norms
│           ├── adapters/           #     Foundation model wrappers: MACE, FairChem
│           ├── training/           #     LightningModule, loss, EMA, tuning
│           │   ├── callbacks/      #       Checkpoint manager, logging, progress callbacks
│           │   └── strategies/     #       Strategy factory: DDP, FSDP, FSDP2, DeepSpeed
│           ├── utils/              #     ASE calculator, feature extraction, mini trainer
│           └── registry.py         #     Lazy component registry
├── examples/
│   └── datasets/                   # Benchmark dataset loaders (MD17, ANI-1, QM9, SPICE)
├── notebooks/                      # Tutorials & demos (getting started, feature extraction, mini trainer)
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

Train the default model — **SIMURGH ARACE** — with the default experiment config:

```bash
goal-train                        # loads configs/ml/simurgh_gmd26.yaml
```

Or equivalently via module:

```bash
python -m goal.ml.cli.train
```

Experiment configs live in `configs/ml/` as **self-contained files** — one file describes the whole run (data, model, trainer, losses, checkpointing, logging). Pick one with `--config-name` and override any field from the CLI:

```bash
goal-train --config-name simurgh_md17            # legacy SIMURGH on MD17
goal-train --config-name monolithic_arace_gmd26  # monolithic ARACE (ablation)
goal-train data.train_dir=/path/to/train trainer.max_epochs=100
```

> **📓 New to GOAL?** Work through [`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb) — a step-by-step tutorial covering model building (modular & monolithic), dataset loading, training, and using trained models as ASE calculators.

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟢 Training
<!-- ═══════════════════════════════════════════════════════════════════ -->

### Single GPU

The default configs already set `accelerator: gpu` and `devices: 1`:

```bash
goal-train                                    # SIMURGH ARACE, single GPU
goal-train trainer.accelerator=cpu            # CPU instead
```

### Multi-GPU: DDP

Distributed Data Parallel — replicates the full model on each GPU and synchronizes gradients. Use when the model fits in a single GPU's memory.

```bash
goal-train strategy.name=ddp trainer.devices=8
```

Multi-node:

```bash
goal-train strategy.name=ddp trainer.devices=4 trainer.num_nodes=2
```

DDP key settings (in the `strategy:` section of every `configs/ml/*.yaml`):

- `find_unused_parameters: false` — set `true` if you have frozen layers.
  Not needed for SIMURGH/ARACE: the artisan bank uses a static-shape
  zero-masking schedule so every parameter participates every step.
- `static_graph: false` — set `true` for models with fixed computation graphs (faster)
- `gradient_as_bucket_view: true` — minor memory optimisation

### Multi-GPU: FSDP

Fully Sharded Data Parallel — shards model parameters, gradients, and optimizer states across GPUs. Use when the model doesn't fit in a single GPU's memory.

```bash
goal-train strategy.name=fsdp trainer.precision=bf16-mixed
```

FSDP settings (in the `strategy:` section):

- `auto_wrap_policy` — controls how modules are wrapped for sharding
- `activation_checkpointing` — trade compute for memory by recomputing activations
- `cpu_offload: false` — offload parameters to CPU (slower, saves GPU memory)

### Multi-GPU: FSDP2 / ModelParallel

ModelParallelStrategy (Lightning 2.4+) — supports FSDP2, tensor parallelism, `torch.compile`, and FP8. Recommended for very large models (500M+ parameters).

```bash
goal-train strategy.name=fsdp2
```

### Multi-GPU: DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) ZeRO enables training of very large models by partitioning optimizer states, gradients, and parameters across GPUs. Requires `pip install -e ".[deepspeed]"`.

```bash
goal-train strategy.name=deepspeed_zero1   # optimizer state partitioning
goal-train strategy.name=deepspeed_zero2   # + gradient partitioning
goal-train strategy.name=deepspeed_zero3   # + parameter partitioning
goal-train strategy.name=deepspeed_zero3_offload   # + CPU offload
```

### Strategy Factory

GOAL provides a unified **strategy factory** (`build_strategy()`) that maps the `strategy:` config section to Lightning strategies. When `cfg.strategy` is present, it takes priority over the trainer's built-in strategy. All strategy parameters (bucket sizes, wrap policies, ZeRO stage, …) live in the `strategy:` section of the self-contained experiment configs.

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
goal-train trainer.accelerator=mps
```

### CPU

```bash
goal-train trainer.accelerator=cpu
```

### Resuming Training

If training is interrupted (crash, preemption, timeout), GOAL **automatically resumes**: on launch, `goal-train` looks for `last.ckpt` in the run's `training.checkpoint_dir` and, if present, restores the full training state (model weights, optimiser, scheduler, epoch counter — logging continues from the restored epoch). The `GOALCheckpointManager` pool state is restored alongside from `checkpoint_state.json`, and a `TRAINING_COMPLETE` sentinel prevents SLURM requeue loops after a successful finish.

Because `training.checkpoint_dir` defaults to the current (timestamped) output directory, a fresh launch starts a fresh run. To resume a *previous* run, point both checkpoint keys at its checkpoint folder:

```bash
goal-train \
  training.checkpoint_dir=logs/simurgh_gmd26/runs/<old_run>/checkpoints \
  checkpoint_manager.dirpath=logs/simurgh_gmd26/runs/<old_run>/checkpoints
```

`training.checkpoint_dir` is where `last.ckpt` and the pool state are found; `checkpoint_manager.dirpath` is where the continued run writes new checkpoints. If the old run finished normally, remove its `TRAINING_COMPLETE` sentinel first — otherwise `goal-train` exits immediately. Note the EMA shadow weights are not stored in checkpoints; the EMA re-initialises from the restored weights and re-converges within a few hundred steps.

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

calc = GOALCalculator(checkpoint_path="logs/simurgh_gmd26/runs/.../checkpoints/last.ckpt")
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
## 🟢 QM Calculators
<!-- ═══════════════════════════════════════════════════════════════════ -->

Beyond running trained GOAL models (above), GOAL builds ASE calculators for reference/QM engines and pretrained potentials through `CalculatorFactory` — for MD, single-points, and multi-timescale (MTS) QM/ML learning.

```python
from goal.md import CalculatorFactory

calc = CalculatorFactory.create("cp2k", preset="molecular", n_mpi=4)
```

| Key | Engine | Notes |
|-----|--------|-------|
| `orca` | ORCA | reference DFT / wavefunction |
| `cp2k` | CP2K | `set_pos_file=True` by default — fixes stdin stalls on MPI builds (needs CP2K ≥ 2024.2); `molecular`/`bulk` presets |
| `espresso` | Quantum ESPRESSO | **new** — `pw.x`; SSSP pseudopotentials required |
| `vasp` | VASP | **new** — commercial; needs `$VASP_PP_PATH` |
| `upet` | UPET / PET-MAD | pretrained PET (`pet-mad`/`pet-omat`/`pet-oam`/`pet-spice`); `pip install upet` |
| `xtb`, `mace`, `flashmd`, `nequip` | semi-empirical / pretrained ML | drop-in ASE calculators |

Calculator configs live in `configs/md/calculator/`. See [`docs/calculators/`](docs/calculators/cp2k.md).

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟢 Fine-Tuning
<!-- ═══════════════════════════════════════════════════════════════════ -->

GOAL fine-tunes pre-trained foundation models **in-process**: each becomes a normal GOAL backbone and trains through the same `GOALModule` + PyTorch Lightning trainer as native models — reusing the loss, stage curriculum, `GOALCheckpointManager`, DDP/FSDP, and HPO. No separate training loop.

| Model | Backbone | Entry point |
|-------|----------|-------------|
| MACE | `mace_finetune` | `goal-train --config-name finetune/mace` |
| UMA / FairChem | `uma_finetune` | `goal-train --config-name finetune/uma` |
| UPET / PET | metatrain subprocess | `goal-finetune-upet` |

MACE and UMA are monolithic backbones (`head: null`) returning `{energy, forces}`. Choose a strategy via `model.backbone.strategy`:

| Strategy | Trains | When |
|----------|--------|------|
| `head_only` | final readout layers only (rest frozen) | small datasets; fastest, most stable |
| `full` | all parameters | larger datasets; use a much smaller LR (~1e-5) |
| `lora` | frozen base + LoRA adapters on every `nn.Linear` (needs `peft`) | parameter-efficient adaptation |

```bash
goal-train --config-name finetune/mace model.backbone.strategy=head_only data.root=/path/to/dataset
```

Per-element reference energies (E0s) are **re-estimated from your training set** before fitting (`reestimate_e0s: true`) — otherwise the mismatch with the foundation model's original E0s is the most common cause of fine-tuning failure. MACE needs `mace-torch` in a standalone venv (e3nn conflict). UPET uses metatensor internally, so it is fine-tuned by a thin CLI that runs `mtt train`; the result loads back as a UPET calculator. See [`docs/finetuning/`](docs/finetuning/overview.md).

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

| Dataset | Structures | Elements | Properties | Size | `data.dataset_type` |
|---------|-----------|----------|------------|------|------|
| **MD17** | ~10k/mol | H, C, N, O | energy, forces | ~100 MB | `md17` |
| **rMD17** | ~10k/mol | H, C, N, O | energy, forces | ~100 MB | `rmd17` |
| **ANI-1** | ~20M | H, C, N, O | energy, forces | ~30 GB | `ani1` |
| **ANI-1x** | ~5M | H, C, N, O | energy, forces | ~7 GB | `ani1x` |
| **QM9** | 134k | H, C, N, O, F | 19 properties | ~1 GB | `qm9` |
| **SPICE** | ~1.1M | 10 elements | energy, forces | ~15 GB | `spice` |

```bash
# Train on MD17 aspirin (ready-made experiment config)
goal-train --config-name simurgh_md17

# Switch molecule / dataset via overrides
goal-train --config-name simurgh_md17 data.molecule=ethanol
goal-train data.dataset_type=ani1x data.max_structures=50000

# Override cutoff (keep data and model cutoffs in sync)
goal-train --config-name simurgh_md17 data.cutoff=6.0
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

GOAL supports two model paradigms:

| Paradigm | How it works | Config | Best for |
|----------|-------------|--------|----------|
| **Modular** | Backbone → NodeFeatures → Head → property dict | `model.backbone` + `model.head` | Mixing backbones and heads freely |
| **Monolithic** | Model → property dict directly | `model.backbone` + `model.head: null` | External self-contained architectures |

**Modular** models separate the backbone (feature extraction) from the head (property prediction). Any backbone can be paired with any compatible head. The built-in modular backbones are SIMURGH ARACE (the default), the legacy ACE-first SIMURGH, HyperSpec, and InvariantGNN.

**Monolithic** models handle everything internally — embedding, interaction, readout, and property prediction — in a single `forward()` call. They return a dictionary of predicted properties directly (the same format heads produce). Set `head: null` in the config to use a monolithic model.

The `MonolithicModel` protocol in `goal.ml.nn.models.base` defines the contract: `forward(graph) → dict[str, Tensor]` and an `output_keys` property declaring which keys consumers can expect.

---

### SIMURGH — the native GOAL force field

SIMURGH is an E(3)-equivariant interatomic potential built around **Potential Artisans**: one dedicated pairwise energy module per unordered element pair (H–H, H–C, C–O, …). This is *not* a generic mixture-of-experts — routing is deterministic by chemistry (each edge goes to exactly one artisan, selected by its element pair), every artisan runs every step under a static-shape zero-masking schedule (DDP-safe, no `find_unused_parameters`), and each artisan carries a learnable gate that is monitored to detect expert collapse.

#### ARACE — the default architecture (`simurgh_arace`)

**ARACE = ARtisan + Atomic Cluster Expansion.** Potential artisans are the **primary computation at every layer**, alternating with ACE message passing:

```
nodes → [Artisan layer → ACE block] × N rounds → energy + forces

┌─────────────────────────────────────────────────────┐
│ ARTISAN LAYER L                                     │
│   For each edge (i→j) routed to its pair artisan:  │
│   h_AB   = EquivLinear(h[row]) + EquivLinear(h[col])│
│   h_pair = TP(h_AB, Y^l(r̂), RadialMLP(d))          │
│   E_L    = MLP(scalars(h_pair)) → (E,) scalar      │
│   E_atom += index_add(row, 0.5 × E_L × cutoff)     │
│   h_pair forwarded to ACE block                     │
└─────────────────────────────────────────────────────┘
          ↓ h_pair (E, irreps) equivariant edge features
┌─────────────────────────────────────────────────────┐
│ ACE MESSAGE PASSING BLOCK L                         │
│   agg_i  = Σ_j h_pair_{j→i} / N̄                    │
│   h_new  = RMSNorm(EquivLinear(agg) + h)  residual  │
└─────────────────────────────────────────────────────┘
          ↓ h_new (N, irreps) updated node features
          repeat for num_rounds

E_total = E_atomic + scale × Σ_L E_artisan_L
F       = −∂E_total/∂r   via autograd
```

What makes this different from MACE-style models (and from the legacy SIMURGH below):

- **Artisans are primary, not auxiliary.** Each round *starts* with the element-pair artisans; there is no separate environment-dressing phase. The first artisan layer operates directly on the atomic-number embeddings.
- **The ACE block aggregates artisan edge features, not node→node messages.** Where MACE builds messages inside the interaction block, ARACE's ACE block simply pools the equivariant pair representations `h_pair` that the artisans already produced onto the destination nodes, so artisan layer `L+1` sees a richer chemical context than layer `L`.
- **Per-round energy decomposition.** Every round contributes its own pair energy `E_L` (logged as `energy_round_{L}` during training) rather than a single readout at the end — a built-in diagnostic for collapsed rounds.
- **Pair-specific computation.** Every element pair gets its own equivariant network (shared symmetric endpoint embedding → CG tensor product with the bond direction, radially weighted → invariant scalar readout), instead of one shared interaction conditioned on species.
- **E(3) equivariance is maintained throughout.** The only transition to invariants is the scalar energy readout inside each artisan.

Building blocks: `ArtisanLayer` ([blocks/artisans.py](src/goal/ml/nn/blocks/artisans.py)), `AceBlock` / `AraceRound` ([blocks/ace_block.py](src/goal/ml/nn/blocks/ace_block.py)), assembled by `SimurghAraceBackbone` ([models/simurgh/backbone_arace.py](src/goal/ml/nn/models/simurgh/backbone_arace.py)).

#### ACE-first — the legacy architecture (`simurgh` / `simurgh_ace_first`)

The original SIMURGH pipeline: ACE-style **environment dressing first** (MACE-style tensor-product message passing, optional body-order expansion), then the artisan bank as an **auxiliary readout at the end** contributing gated, cosine-tapered pair energies. Preserved and fully supported — select it via `model.backbone.name: simurgh` (see `configs/ml/simurgh_ace_first_gmd26.yaml`), just no longer the default.

#### Model zoo

| Registry name | Kind | Description |
|---|---|---|
| `simurgh_arace` | modular | **Default.** ARACE architecture — artisan layer + ACE block alternating each round |
| `simurgh` (alias `simurgh_ace_first`) | modular | Legacy — ACE dressing first, artisan bank as final readout |
| `monolithic_arace` | monolithic | ARACE as a self-contained model (computes its own forces) — for research/ablation |
| `simurgh_monolithic` | monolithic | Legacy ACE-first SIMURGH without the backbone/head split |
| `hyperspec` | modular | E(3)-equivariant GNN baseline (spherical harmonics + tensor products) |
| `invariant_gnn` | modular | SchNet-like invariant baseline (scalars only) |
| `monolithic_example` | monolithic | Minimal reference implementation of the `MonolithicModel` protocol |

```bash
goal-train                                        # simurgh_arace (default config)
goal-train --config-name simurgh_ace_first_gmd26  # legacy ACE-first
goal-train --config-name monolithic_arace_gmd26   # monolithic ARACE
goal-train --config-name hyperspec_md17           # HyperSpec baseline
```

---

### Monolithic Models

Monolithic models bypass the backbone→head split. They take an `AtomicGraph` and return a property dictionary directly. `monolithic_arace` and `simurgh_monolithic` are the self-contained counterparts of the two SIMURGH architectures; `monolithic_example` is a minimal demonstration model (embedding → MLP readout → autograd forces) for users bringing their own architecture.

---

### Implementing a New Monolithic Model

Create a model that satisfies the `MonolithicModel` protocol:

```python
import torch
import torch.nn as nn
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("my_monolithic")
class MyModel(nn.Module):
    def __init__(self, cutoff: float = 5.0) -> None:
        super().__init__()
        # Use any GOAL modules: AtomicNumberEmbedding, BesselBasis, etc.
        ...

    @property
    def output_keys(self) -> list[str]:
        return ["energy", "forces"]

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        # Compute everything internally, return property dict
        return {"energy": energy, "forces": forces}
```

Then in the config, set `head: null`:

```yaml
model:
  backbone:
    name: my_monolithic
    cutoff: 5.0
  head: null
```

---

## Heads

Task-specific output heads registered via the head registry. Used with **modular** backbones — monolithic models set `head: null` and skip the head entirely.

| Head | Description | Config key |
|------|-------------|-----------|
| `energy_forces` | Energy prediction + force via autograd | `head.name: energy_forces` |
| `energy` | Energy prediction only | `head.name: energy` |
| `direct_forces` | Direct force prediction (no autograd) | `head.name: direct_forces` |
| `stress` | Stress tensor prediction | `head.name: stress` |
| `dipole` | Dipole moment prediction | `head.name: dipole` |
| `scalar` | Generic scalar property (any name) | `head.name: scalar` |
| `multi` | Compose multiple heads | `head.name: multi` |

Override the head:

```bash
goal-train model.head.name=stress model.head.compute_stress=true
```

### Generic Scalar Head

The `scalar` head predicts any per-structure scalar property. The `property_name` parameter sets the output key (and must match the target key on the graph):

```yaml
head:
  name: scalar
  irreps_in: "128x0e"
  hidden_dim: 64
  property_name: band_gap   # ← becomes the key in predictions dict
  reduction: mean            # "mean" (intensive) or "sum" (extensive)
```

### Multi-Head — Multiple Properties at Once

The `multi` head composes several sub-heads that share the same backbone features. Each sub-head independently produces its output keys, which are merged into a single dictionary:

```yaml
model:
  backbone:
    name: invariant_gnn
    hidden_channels: 128
    # ...

  head:
    name: multi
    heads:
      - name: energy_forces
        irreps_in: "128x0e"
        hidden_dim: 64
      - name: scalar
        irreps_in: "128x0e"
        hidden_dim: 64
        property_name: homo
        reduction: mean
      - name: scalar
        irreps_in: "128x0e"
        hidden_dim: 64
        property_name: lumo
        reduction: mean
```

Each sub-head has its own readout MLP, so they learn separate representations for each property. The corresponding losses reference the property names:

```yaml
training:
  losses:
    - name: energy
      weight: 4.0
    - name: forces
      weight: 100.0
    - name: scalar_property
      property_name: homo
      weight: 1.0
    - name: scalar_property
      property_name: lumo
      weight: 1.0
```

Loss components declared in `training.losses` are matched to the merged prediction keys by name.

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

Configure per-property in the `training.losses` section of your `configs/ml/*.yaml`:

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

`MACEAdapter` / `UMAAdapter` wrap pre-trained MACE and FairChem/UMA models as GOAL backbones for **feature extraction and inference**, translating between the foundation model's interface and GOAL's backbone protocol.

To **fine-tune** these models, use the trainable `mace_finetune` / `uma_finetune` backbones instead — see [Fine-Tuning](#-fine-tuning).

Install adapter dependencies:

```bash
pip install "mace-torch>=0.3.16"   # MACE (standalone venv — see Installation note)
pip install fairchem-core          # FairChem / UMA
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

Or use the dynamic scheduler — uncomment the `grad_accumulation` block in the `callbacks:` section of your `configs/ml/*.yaml`.

### Exponential Moving Average (EMA)

Maintains a shadow copy of weights for more stable evaluation:

```yaml
training:
  ema:
    enabled: true
    decay: 0.999
```

### Stochastic Weight Averaging (SWA)

Alternative to EMA — averages weights during the last portion of training. Uncomment the `swa` block in the `callbacks:` section of your `configs/ml/*.yaml` (and disable `training.ema` first — they are mutually exclusive).

### Sanity Validation Check

Before the first training epoch, Lightning runs a short validation sanity check to catch data loading, metric computation, or model errors early. This is enabled by default:

```yaml
# trainer: section of configs/ml/*.yaml
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

### ARACE search space (recommended starting point)

`configs/ml/simurgh_gmd26_hpo.yaml` composes the default ARACE experiment and defines a Ray Tune search space over the architecture's key dimensions — `num_rounds`, `artisan.hidden_irreps`, `artisan.num_layers`, `artisan.radial_hidden`, `artisan.n_scalar_out`, `share_artisan_weights`, plus log-uniform `lr` and `weight_decay`:

```bash
pip install -e ".[tune]"   # installs ray[tune] + optuna
goal-tune --config-dir configs/ml --config-name simurgh_gmd26_hpo
```

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

Callbacks are declared inline in the `callbacks:` section of each `configs/ml/*.yaml` (plus a top-level `checkpoint_manager:` block). The default set includes:

- **GOALCheckpointManager** — three checkpoint pools: top-k by validation metric, fixed-interval, and `last.ckpt` for crash recovery; freezes model source and packs shippable `.simurgh` archives
- **EarlyStopping** — stop when `val/forces_mae` plateaus
- **RichModelSummary** — rich-formatted model summary
- **GOALRichProgressBar** — stage-aware training progress (shows the active loss-curriculum stage)
- **LearningRateMonitor** and **RichLoggingCallback** — always active via code

### Additional Callbacks

Uncomment the corresponding block in the `callbacks:` section to enable: `StochasticWeightAveraging`, `BackboneFinetuning` (gradual unfreezing), `GradientAccumulationScheduler`, `EMAWeightAveraging`, `ThroughputMonitor`.

Override callback parameters:

```bash
goal-train callbacks.early_stopping.patience=200
goal-train checkpoint_manager.top_k.k=3
```

---

<!-- ═══════════════════════════════════════════════════════════════════ -->
## 🟣 Logging
<!-- ═══════════════════════════════════════════════════════════════════ -->

GOAL supports all Lightning loggers. They are declared inline in the `logger:` section of each `configs/ml/*.yaml` — CSV and W&B are enabled by default; uncomment the others to activate them. Multiple loggers run simultaneously.

| Logger | Notes |
|--------|-------|
| Weights & Biases | Project: `goal`, requires `wandb` login (enabled by default) |
| CSV | Simple CSV file logging (enabled by default) |
| TensorBoard | Saves to `output_dir/tensorboard/` |
| MLflow | MLflow tracking server |
| Neptune | Requires `NEPTUNE_API_TOKEN` |
| Aim | Local `.aim` repo, open with `aim up` |
| Comet | Comet.ml experiment tracking |

ARACE runs additionally log the architecture-specific diagnostics: per-round energy contributions (`train/energy_round_{L}`, `val/energy_round_{L}` — watch for a round collapsing to zero) and every artisan gate at validation time (`gates/round{L}/{pair}` — a gate drifting to near-zero signals expert collapse).

### Run Naming Convention

Every run is automatically named with a **timestamp + dataset + model** pattern:

```
{date}_{time}_{dataset_type}_{model_backbone}{run_name_suffix}
```

For example: `2026-07-07_14-30-45_trajectory_simurgh_arace_fragment_duplication`

This naming is applied consistently to:

- Output directories (`logs/{task_name}/runs/...`)
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

GOAL uses [Hydra](https://hydra.cc/) with **self-contained experiment configs**: every run is described by a single file in `configs/ml/` (data, model, trainer, losses, checkpointing, logging, distributed strategy — all in one place, every option documented inline). Select a file with `--config-name` and override any field from the command line.

### Experiment Configs (`configs/ml/`)

| Config | Model | Notes |
|--------|-------|-------|
| `simurgh_gmd26` | `simurgh_arace` | **Default** — ARACE on GMD-26 |
| `simurgh_gmd26_hpo` | `simurgh_arace` | ARACE hyperparameter search (Ray Tune) |
| `simurgh_ace_first_gmd26` | `simurgh` | Legacy ACE-first SIMURGH on GMD-26 |
| `monolithic_arace_gmd26` | `monolithic_arace` | Monolithic ARACE (research/ablation) |
| `simurgh_md17` | `simurgh` | Legacy SIMURGH on MD17 |
| `hyperspec_md17` | `hyperspec` | HyperSpec baseline on MD17 |

### The default model config (ARACE)

The `model:` section of `configs/ml/simurgh_gmd26.yaml`:

```yaml
model:
  backbone:
    name: simurgh_arace
    num_rounds: 2                  # artisan + ACE rounds
    share_artisan_weights: false   # true = one artisan bank for all rounds
    cutoff: ${data.cutoff}         # must match the data cutoff
    embedding_dim: 32
    num_elements: 120
    avg_num_neighbors: null        # auto-computed from the training set
    artisan:                       # per-element-pair equivariant artisans
      architecture: equivariant
      hidden_irreps: "16x0e + 16x1o + 16x2e"
      num_layers: 1                # depth of each artisan (tunable)
      num_rbf: 8
      radial_hidden: 32
      n_scalar_out: 16
      final_hidden: 16
      element_conditioned: true
    atomic_energies:
      mode: learned                # or "dataset" (LSQ) / "provided" (DFT)
    scale: null
  head:
    name: energy_forces            # conservative forces via -∂E/∂r
    irreps_in: ${model.backbone.artisan.hidden_irreps}
    hidden_dim: 128
```

Switching architectures is a one-line change — `model.backbone.name: simurgh` selects the legacy ACE-first backbone (which uses `dressing_kwargs` / `artisan_config` keys instead; see `configs/ml/simurgh_ace_first_gmd26.yaml` for a ready-made file).

### Override Examples

```bash
# Pick a different experiment file
goal-train --config-name simurgh_ace_first_gmd26

# Override nested parameters
goal-train training.optimizer.lr=0.0005 training.ema.decay=0.9999
goal-train model.backbone.num_rounds=3 model.backbone.share_artisan_weights=true

# Change loss weights
goal-train training.losses.0.weight=1.0 training.losses.1.weight=50.0

# Multi-run sweep
goal-train -m training.optimizer.lr=0.001,0.0005,0.0001
```

### Output Directory

Each run creates a timestamped output directory:

```
logs/simurgh_gmd26/runs/2026-07-07_14-30-45_trajectory_simurgh_arace_fragment_duplication/
├── checkpoints/
│   ├── epoch_001.ckpt
│   ├── last.ckpt
│   ├── frozen_source/       # frozen model source for self-contained checkpoints
│   ├── config.yaml          # resolved config
│   └── metadata.json
├── csv_logs/
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
| `goal-train` (alias `goal-train-ml`) | Train a model |
| `goal-eval` | Evaluate a checkpoint on test data |
| `goal-finetune` | Fine-tune via a foundation adapter + head |
| `goal-finetune-upet` | Fine-tune a UPET/PET model via metatrain (`mtt train`) |
| `goal-tune` | Hyperparameter search (LR finder, Ray Tune, W&B Sweeps) |
| `goal-simulate` (`goal-simulate-mts`) | Run MD / multi-timescale MD |
| `goal-inspect` | Inspect a trained checkpoint |
| `goal-pack-archive` | Pack a checkpoint into a self-contained `.simurgh` archive |

All commands accept Hydra overrides:

```bash
goal-train trainer=ddp data=hdf5 model=invariant_gnn logger=wandb seed=42
```

Module-based invocation (equivalent):

```bash
python -m goal.ml.cli.train trainer=ddp data=hdf5
python -m goal.ml.cli.evaluate ckpt_path=/path/to/ckpt
python -m goal.ml.cli.train --config-name finetune/mace
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
