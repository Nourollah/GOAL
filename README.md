<div align="center">

<img src="https://raw.githubusercontent.com/Nourollah/GOAL/main/docs/assets/goal-banner.jpeg" alt="GOAL — General Open Atomistic Laboratory" width="100%"/>

### *Your atoms. Your rules. Your laboratory.*

Train machine-learning interatomic potentials, then put them to work in molecular dynamics, all in one framework.

<br/>

[![python](https://img.shields.io/badge/Python_3.14+-3776AB?logo=python&logoColor=white)](https://python.org)
[![pytorch](https://img.shields.io/badge/PyTorch_2.10+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/Lightning_2.6+-792EE5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/Hydra_1.3-89B8CD?logo=dropbox&logoColor=white)](https://hydra.cc/)
[![ase](https://img.shields.io/badge/ASE_3.25+-0A6EBD)](https://wiki.fysik.dtu.dk/ase/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license)
[![docs](https://img.shields.io/badge/docs-website-ff6d00)](https://nourollah.github.io/GOAL/)

</div>

---

## What is GOAL?

**GOAL** (**G**eneral **O**pen **A**tomistic **L**aboratory) is organised as modules under the `goal` namespace. Currently available modules:

| | Module | What it does |
|---|---|---|
| 🧠 | **[`goal.ml`](https://github.com/Nourollah/GOAL/blob/main/docs/ml/index.md)** | Trains and fine-tunes ML interatomic potentials with PyTorch Lightning + Hydra. Ships the native E(3)-equivariant **SIMURGH** force field (ARACE by default, plus NF_SIMURGH and SIMURGH Simplified variants), HyperSpec and invariant baselines, MACE / UMA / UPET fine-tuning, multi-property heads, and distributed training (DDP, FSDP, DeepSpeed). |
| ⚛️ | **[`goal.md`](https://github.com/Nourollah/GOAL/blob/main/docs/md/index.md)** | Runs molecular dynamics with any calculator: trained GOAL models, foundation models (MACE, UPET/PET-MAD, FlashMD, NequIP), GFN-xTB tight binding (tblite) or QM codes (CP2K, Quantum ESPRESSO, VASP, ORCA). tblite and Quantum ESPRESSO (`pw.x`) ship in the `-full` pixi environments. Also supports **multiple-time-step MD** with on-the-fly QM→ML learning (RESPA, i-PI). |

Modules are deliberately decoupled. For example, `goal.md` uses `goal.ml` models only through a narrow bridge, so each module can be installed and used on its own.

## Installation

GOAL is distributed as **`goal-atomistic`** on PyPI and conda-forge; the import name is `goal` (`import goal.ml`, `import goal.md`). Configs and benchmark dataset adapters are bundled, so every CLI works from any directory, no clone needed.

**pip** (Python 3.14; install the [PyTorch build](https://pytorch.org/get-started/locally/) for your hardware first):

```bash
pip install goal-atomistic              # goal.ml
pip install "goal-atomistic[md]"        # + goal.md (FlashMD, tblite, i-PI, CLI)
pip install "goal-atomistic[all]"       # every extra except mace
```

**conda-forge** (conda has no extras, so there are three packages: `goal-atomistic`, `goal-atomistic-md` adding tblite, Quantum ESPRESSO and i-PI, and `goal-atomistic-full` adding the training extras):

```bash
pixi add goal-atomistic-full            # or: conda install -c conda-forge goal-atomistic-full
```

**pixi from a clone** (locked, reproducible environments; recommended for development and for the notebooks). Pick the environment for your hardware; each one works on x86-64 and aarch64 (ROCm: x86-64 only):

```bash
git clone https://github.com/Nourollah/GOAL.git && cd GOAL
```

| Hardware | ML module | ML + MD modules |
|---|---|---|
| NVIDIA GPU | `pixi install -e cuda` | `pixi install -e cuda-full` |
| AMD GPU | `pixi install -e rocm` | `pixi install -e rocm-full` |
| CPU | `pixi install -e cpu` | `pixi install -e cpu-full` |

> [!NOTE]
> GOAL targets **Python 3.14**. MACE pins an old `e3nn` and needs its own virtualenv, and FairChem pins an older PyTorch, so both are installed on demand. FlashMD and UPET are not on conda-forge; `pip install` them into a conda environment if you need them. See [Installation](https://github.com/Nourollah/GOAL/blob/main/docs/getting-started/installation.md) for the details.

## Quick start

**Train a potential** (SIMURGH on GMD-26, the default experiment). The commands below use a pixi environment from a clone; after a pip or conda install, drop the `pixi run -e cuda` prefix:

```bash
pixi run -e cuda goal-train                                     # src/goal/configs/ml/simurgh_gmd26.yaml
pixi run -e cuda goal-train --config-name nf_simurgh_md17       # pick another experiment
pixi run -e cuda goal-train trainer.max_epochs=100 data.batch_size=8   # override anything
pixi run -e cuda goal-eval ckpt_path=logs/simurgh_gmd26/runs/<run>/checkpoints   # test-set metrics
```

**Run MD with it** (inside `pixi shell -e cuda-full`, or any environment with the MD module):

```python
from goal.md.core.molecule_factory import MoleculeFactory
from goal.md.core.md_factory import DynamicsFactory
from goal.md.adapters.model_loader import load_goal_calculator

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
atoms.calc = load_goal_calculator("path/to/best.ckpt", device="cuda")

dyn = DynamicsFactory.create("langevin_ase", atoms, timestep_fs=0.5, temperature_K=300)
dyn.run(1000)
```

MD can also be driven config-first from the command line:

```bash
pixi run -e cuda-full goal-simulate calculator.checkpoint=logs/simurgh_gmd26/runs/<run>/checkpoints steps=5000
pixi run -e cuda-full goal-simulate-batch --checkpoint <checkpoint> --structures frames.xyz   # many structures; one GPU batch on NVIDIA
```

See the [MD CLI reference](https://github.com/Nourollah/GOAL/blob/main/docs/md/reference/cli.md) and, for the NVIDIA GPU fast paths and their portable fallbacks, the [integrations overview](https://github.com/Nourollah/GOAL/blob/main/docs/integrations/nvalchemi_overview.md).

> [!TIP]
> New to GOAL? Start with the [`getting_started`](https://github.com/Nourollah/GOAL/blob/main/notebooks/getting_started.ipynb) notebook, then browse the [notebook catalogue](https://github.com/Nourollah/GOAL/blob/main/docs/getting-started/notebooks.md).

## Documentation

The full documentation is published at **[nourollah.github.io/GOAL](https://nourollah.github.io/GOAL/)**. The same pages live in [`docs/`](https://github.com/Nourollah/GOAL/blob/main/docs/index.md) and read fine on GitHub; preview them locally with `pixi run -e docs docs-serve`.

| Getting started | 🧠 goal.ml | ⚛️ goal.md |
|:---|:---|:---|
| [Installation](https://github.com/Nourollah/GOAL/blob/main/docs/getting-started/installation.md) | [Overview](https://github.com/Nourollah/GOAL/blob/main/docs/ml/index.md) | [Overview](https://github.com/Nourollah/GOAL/blob/main/docs/md/index.md) |
| [Project structure](https://github.com/Nourollah/GOAL/blob/main/docs/getting-started/project-structure.md) | [Training](https://github.com/Nourollah/GOAL/blob/main/docs/ml/training/training.md) · [Distributed](https://github.com/Nourollah/GOAL/blob/main/docs/ml/training/distributed.md) | [Usage patterns](https://github.com/Nourollah/GOAL/blob/main/docs/md/guide/usage-patterns.md) |
| [Notebooks](https://github.com/Nourollah/GOAL/blob/main/docs/getting-started/notebooks.md) | [Data loading](https://github.com/Nourollah/GOAL/blob/main/docs/ml/data/loading.md) · [Benchmarks](https://github.com/Nourollah/GOAL/blob/main/docs/ml/data/benchmarks.md) | [Molecules](https://github.com/Nourollah/GOAL/blob/main/docs/md/guide/molecules.md) · [Dynamics](https://github.com/Nourollah/GOAL/blob/main/docs/md/guide/dynamics.md) |
| | [Models & SIMURGH](https://github.com/Nourollah/GOAL/blob/main/docs/ml/models/overview.md) · [Heads](https://github.com/Nourollah/GOAL/blob/main/docs/ml/models/heads.md) | [Calculators](https://github.com/Nourollah/GOAL/blob/main/docs/md/calculators/overview.md) |
| | [Fine-tuning MACE / UMA / UPET](https://github.com/Nourollah/GOAL/blob/main/docs/ml/finetuning/overview.md) | [Multiple time-step MD](https://github.com/Nourollah/GOAL/blob/main/docs/md/mts/overview.md) |
| | [ASE calculator](https://github.com/Nourollah/GOAL/blob/main/docs/ml/deployment/ase-calculator.md) · [Archives](https://github.com/Nourollah/GOAL/blob/main/docs/ml/deployment/archives.md) | [Bridge to goal.ml](https://github.com/Nourollah/GOAL/blob/main/docs/md/guide/ml-bridge.md) |
| | [Configuration](https://github.com/Nourollah/GOAL/blob/main/docs/ml/reference/configuration.md) · [CLI](https://github.com/Nourollah/GOAL/blob/main/docs/ml/reference/cli.md) | [Configuration](https://github.com/Nourollah/GOAL/blob/main/docs/md/reference/configuration.md) · [CLI](https://github.com/Nourollah/GOAL/blob/main/docs/md/reference/cli.md) |

## Tested versions

| Python | PyTorch | Lightning | e3nn | PyG | Hydra | ASE |
|---|---|---|---|---|---|---|
| 3.14.4 | 2.10.0 | 2.6.1 | 0.6.0 | 2.7.0 | 1.3.2 | 3.28.0 |

## License

GOAL is released under the MIT License.
