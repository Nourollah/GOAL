# Installation

This page explains how to install GOAL: the core package, its optional extras, the pixi environments, and the separate setups that some foundation models need. It is written for people who already manage Python environments. Pick the route that matches your hardware and the parts of GOAL you plan to use.

GOAL is a single Python distribution, published as **`goal-atomistic`** on PyPI and conda-forge (the name `goal` was taken). The import name is still `goal`: `import goal.ml`, `import goal.md`. Currently available modules:

| Module | What it does | Install |
|---|---|---|
| `goal.ml` | Training, fine-tuning, evaluating and deploying ML interatomic potentials | `pip install goal-atomistic`; conda-forge `goal-atomistic`; every pixi environment |
| `goal.md` | Molecular dynamics, QM/foundation-model calculators, multiple-time-step MD | `pip install "goal-atomistic[md]"`; conda-forge `goal-atomistic-md`; the `*-full` pixi environments |

The Hydra configs (`goal/configs/`) and the benchmark dataset adapters (`goal.ml.data.benchmarks`) are part of the package, so every install route gives you working CLIs from any directory. You only need a clone of the repository to develop GOAL, to use the pixi environments, or to run the notebooks.

There are four ways to install GOAL:

1. [pixi from a clone](#option-1-pixi-from-a-clone): locked, reproducible environments for every hardware target. Recommended for working in the repository.
2. [pip from PyPI](#option-2-pip-from-pypi): `pip install goal-atomistic` into any Python 3.14 virtualenv.
3. [conda-forge packages](#option-3-conda-forge-packages): `goal-atomistic`, `goal-atomistic-md`, `goal-atomistic-full`, with `pixi add` or `conda install`.
4. [Development install from a clone](#option-4-development-install-from-a-clone): `pip install -e ".[dev]"`.

| Core libraries | `torch>=2.4`, `torch-geometric>=2.7`, `e3nn>=0.5`, `lightning>=2.6`, `hydra-core>=1.3.2`, `ase>=3.23`, `wandb`, `rich`, `lmdb`, `rdkit` |

## Option 1: pixi from a clone

[Pixi](https://pixi.sh/) is a fast package manager built on conda-forge. It locks conda and PyPI dependencies together in `pixi.lock`, so every machine gets the same environment. The GOAL workspace is defined in `pyproject.toml` under `[tool.pixi.*]`, so there is no separate `pixi.toml`.

```bash
# install pixi once
curl -fsSL https://pixi.sh/install.sh | bash

git clone https://github.com/Nourollah/GOAL.git
cd GOAL
pixi install -e cuda          # pick the environment for your hardware (table below)
```

> [!TIP]
> Always name the environment with `-e` (`pixi install -e …`, `pixi run -e …`, `pixi shell -e …`). It makes every command say exactly which hardware build it runs on, and it keeps commands copy-pasteable between machines.

### Environments

There are two axes: **hardware** (which PyTorch build) and **modules** (ML only, or ML + MD).

| | ML module | ML + MD modules | Platforms | PyTorch build |
|---|---|---|---|---|
| **NVIDIA GPU** | `cuda` | `cuda-full` | x86-64, aarch64 | conda-forge `pytorch-gpu` (CUDA 12) |
| **CPU** | `cpu` | `cpu-full` | x86-64, aarch64 | conda-forge `pytorch-cpu` |
| **AMD GPU** | `rocm` | `rocm-full` | x86-64 | PyTorch ROCm 7.2 wheels |
| **Docs** | `docs` | | x86-64, aarch64 | none (MkDocs only) |

Each environment is defined once and solved for every platform it supports. `pixi.lock` stores one solve per platform, and pixi picks the right one for the machine it runs on, so x86-64 and aarch64 users run the same commands.

**What every ML environment contains.** GOAL in editable mode with all its core dependencies, plus the ML extras: `peft` (LoRA), `zarr` (Zarr datasets), `deepspeed`, `ray[tune]` and `optuna`, `matscipy`, `torchmetrics`, `h5py` and `requests` (benchmark downloaders), `py3dmol`, `dftd3-python`, Jupyter, and the dev tools (`pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`). The `cuda*` environments also add `nvalchemi-toolkit-ops` (GPU neighbour lists, NVIDIA only).

**What `*-full` adds.** The MD module's dependencies: `flashmd`, `ipi`, `typer` and `pandas`, plus two simulation engines from conda-forge:

- **tblite** (`tblite-python`): GFN1-/GFN2-xTB tight binding for the [`tblite` calculator](../md/calculators/tblite.md). It is the cheap, physics-based option for exploratory MD, pre-equilibration and screening.
- **Quantum ESPRESSO 7.6** (`qe`, with OpenMPI and an explicit `elpa` pin, because the conda package does not declare that dependency): `pw.x` and friends on `PATH` for the [`espresso` calculator](../md/calculators/espresso.md). It provides plane-wave DFT with no further setup apart from pseudopotentials, which you download yourself.

Both are available on x86-64 and aarch64.

```bash
pixi install -e cuda-full                      # NVIDIA workstation, everything
pixi run -e cuda goal-train trainer.max_epochs=10
pixi run -e cuda-full goal-simulate …
pixi shell -e cuda                             # interactive shell; then just: goal-train …
pixi run -e docs docs-serve                    # documentation preview
```

The defined tasks (`pixi run -e cuda train`, `pixi run -e cuda test`, …) are listed in [Pixi tasks](../ml/reference/pixi-tasks.md).

> [!NOTE]
> The CUDA environments declare a CUDA 12 system requirement. On a machine without an NVIDIA driver, `pixi install -e cuda` stops with an error. Use `-e cpu` (or `-e rocm` on AMD hardware) instead.

#### FairChem (UMA)

`fairchem-core` pins `torch~=2.13.0`. Adding it to the environments would downgrade PyTorch everywhere, so it is installed on demand into an existing environment:

```bash
uv pip install --python .pixi/envs/cuda/bin/python "fairchem-core>=1.0"
```

### Adding packages to a pixi environment by hand

Pixi's uv-managed environments do not ship a `pip` binary. To add packages to one of them, point `uv pip` at the environment's interpreter, as above.

> [!WARNING]
> Pixi does not track packages installed this way. The next `pixi install -e cuda` or `pixi run -e cuda …` may reconcile the environment back to `pixi.lock` and remove or downgrade them. That is harmless for purely additive packages. It is **not** harmless for anything that changes a locked dependency, such as FairChem's torch pin or MACE's `e3nn` downgrade (see below).

## Option 2: pip from PyPI

Any Python 3.14 virtual environment works. Install the PyTorch build for your hardware **first**, because pip keeps whichever `torch` is already installed and would otherwise pull PyPI's default build:

```bash
python3.14 -m venv .venv && source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu126     # NVIDIA
# pip install torch --index-url https://download.pytorch.org/whl/rocm7.2  # AMD
# pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU

pip install goal-atomistic              # ML module
pip install "goal-atomistic[md]"        # + MD module
pip install "goal-atomistic[all]"       # every extra except mace
```

With the environment activated, the console scripts work from any directory; no clone is needed:

```bash
goal-train --cfg job -p model.backbone.name              # prints: simurgh_arace
goal-train --config-name nf_simurgh_md17 --cfg job       # bundled experiment configs
python -c 'import goal; print(goal.CONFIGS_DIR)'         # where the bundled configs live
```

> [!NOTE]
> Quantum ESPRESSO (`pw.x`) is not a Python package, so `[md]` does not install it. Take it from conda-forge (`qe`, or the `goal-atomistic-md` package below), your cluster's module system, or a source build.

## Option 3: conda-forge packages

conda packages have no extras, so GOAL is split into three packages on conda-forge:

| Package | Contains |
|---|---|
| `goal-atomistic` | The core ML module (`goal.ml`) and the CLIs |
| `goal-atomistic-md` | `goal-atomistic` + the MD stack: `tblite-python`, Quantum ESPRESSO (`qe`, with OpenMPI and ELPA), `dftd3-python`, `i-pi`, `typer`, `pandas` |
| `goal-atomistic-full` | `goal-atomistic-md` + `peft`, `deepspeed`, `ray-tune`, `optuna`, `matscipy`, `torchmetrics`, `h5py`, `requests` |

```bash
pixi add goal-atomistic-full                          # in your own pixi project
# or
conda install -c conda-forge goal-atomistic-full      # in a conda/mamba environment
```

The solver picks a CUDA or CPU build of PyTorch for your machine automatically. ROCm builds are not on conda-forge; for AMD GPUs use the pixi `rocm` / `rocm-full` environments from a clone ([Option 1](#option-1-pixi-from-a-clone)) or pip with the ROCm index ([Option 2](#option-2-pip-from-pypi)).

`flashmd` and `upet` are not on conda-forge, so none of the three packages include them. Install them with pip into the same environment if you need them (see [Packages installed separately](#packages-installed-separately)).

## Option 4: development install from a clone

To work on GOAL itself without pixi, install the clone in editable mode with the `dev` extra:

```bash
git clone https://github.com/Nourollah/GOAL.git
cd GOAL
python3.14 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126     # or rocm7.2 / cpu, as above
pip install -e ".[dev]"            # add md, all, docs, ... as needed: -e ".[dev,md]"
```

Edits to `src/goal/` (including `src/goal/configs/`) take effect immediately. The pixi environments already contain everything in `dev`.

### Optional extras

The extras below come from `[project.optional-dependencies]`. Append them to the distribution name, for example `pip install "goal-atomistic[md,tune]"` (or `pip install -e ".[md,tune]"` in a clone). The pixi environments already include all of them except `mace` and `fairchem`.

| Extra | Installs | Enables |
|---|---|---|
| `md` | `flashmd`, `tblite` (x86-64 only; PyPI has no aarch64 wheels), `ase>=3.25`, `typer`, `pandas`, `ipi>=3.0` | The MD module: `goal-simulate` CLIs, FlashMD, tblite (GFN-xTB) and i-PI. Quantum ESPRESSO is not a Python package; with pip, install `pw.x` separately |
| `mace` | `mace-torch>=0.3.16` | MACE adapters and the `mace_finetune` backbone. **Not installable together with GOAL's core `e3nn`**. Use the standalone venv below. |
| `fairchem` | `fairchem-core>=1.0` | UMA / EquiformerV2 adapters and the `uma_finetune` backbone (pins `torch~=2.13`) |
| `matscipy` | `matscipy` | matscipy neighbour-list backend |
| `nvalchemiops` | `nvalchemi-toolkit-ops[torch]` | GPU neighbour lists (NVIDIA only) |
| `nvalchemiops-cu12` / `nvalchemiops-cu13` | `nvalchemi-toolkit-ops[torch-cu12]` / `[torch-cu13]` | The same, with the matching CUDA runtime wheels |
| `zarr` | `zarr>=3` | Zarr datasets (`dataset_type: zarr`) and `goal-pack-zarr` ([Zarr data](../integrations/zarr_data.md)) |
| `lora` | `peft>=0.13` | LoRA fine-tuning (SIMURGH `simurgh_lora`, and `strategy: lora` for MACE/UMA) |
| `deepspeed` | `deepspeed>=0.14` | `deepspeed_zero1/2/3` and `deepspeed_zero3_offload` strategies |
| `tune` | `ray[tune]>=2.55.1`, `optuna>=3.0` | `goal-tune` with `method: ray` (Ray Tune + Optuna) |
| `torchmetrics` | `torchmetrics>=1.9.0` | torchmetrics-backed loss functions (`fn: torchmetrics.…`) |
| `examples` | `h5py>=3.10`, `requests>=2.31` | Downloaders for MD17/rMD17, MD22, QM9, ANI-1/1x, SPICE |
| `dev` | `pytest`, `pytest-cov`, `rootutils`, `ruff`, `mypy`, `ipython`, `jupyter`, `ipykernel` | Running the test suite, linters and notebooks |
| `docs` | `mkdocs`, `mkdocs-material`, `markdown-callouts` | Building this documentation |
| `all` | `goal-atomistic[fairchem,matscipy,nvalchemiops,deepspeed,tune,torchmetrics,examples,md,zarr,lora]`: every extra above except `mace`, the CUDA-suffixed `nvalchemiops-*`, `dev` and `docs` | Convenience bundle |

### Packages installed separately

Some integrations are not declared as extras at all:

```bash
pip install upet        # UPET / PET-MAD calculator in goal.md (CalculatorFactory.create("upet", ...))
pip install metatrain   # provides `mtt`, which goal-finetune-upet launches as a subprocess
```

**nvalchemi-toolkit** (NVIDIA's batched GPU MD engine, used by the [nvalchemi model wrapper](../integrations/nvalchemi_wrapper.md) and the GPU path of [batched simulation](../integrations/batched_simulation.md)) is deliberately not an extra: every PyPI release so far declares `requires-python <3.14`, and GOAL needs Python 3.14. Install it from git `main` into a separate venv layered on a pixi environment; the recipe is in [Integrations overview](../integrations/nvalchemi_overview.md). Everything that uses it falls back to a portable path when it is absent.

**`xtb-python`** is not on PyPI and GOAL no longer uses it: GFN1/GFN2-xTB comes from **tblite** (the [`tblite` calculator](../md/calculators/tblite.md), in the `-full` pixi environments and, on x86-64, in the `[md]` extra), and ORCA's built-in `XTB2` keyword is a second route. Older versions of the `[md]` extra required `xtb-python` and pinned `rdkit<2026` against the core `rdkit>=2026.3.3`, so `pip install -e ".[md]"` could not resolve; both pins are gone.

External simulation engines other than Quantum ESPRESSO (ORCA, CP2K, VASP) are separate binaries that you install yourself. The [MD calculators](../md/calculators/overview.md) pages cover them. Quantum ESPRESSO comes with the `-full` pixi environments; with pip, install `pw.x` yourself (for example from conda-forge `qe`, or your cluster's module).

## MACE (`mace-torch`) in a standalone venv

`mace-torch` hard-pins `e3nn==0.4.4`, while GOAL's own equivariant blocks (SIMURGH, ARACE, NF_SIMURGH, …) need `e3nn>=0.5` (0.6.x in practice). No version of e3nn satisfies both, so:

- `pip install "goal-atomistic[mace]"` is **unsatisfiable by construction**;
- you **cannot** train SIMURGH and run MACE in the same interpreter.

Do not try to fix this inside a pixi environment by downgrading e3nn by hand. It looks as if it works, but pixi does not track the downgrade. The next `pixi install -e cuda` or `pixi run -e cuda …` (for example, whatever launches your Jupyter kernel) silently moves e3nn back to 0.6.x. `mace_mp()` then fails with a confusing `too many values to unpack` error deep inside `e3nn/util/codegen/_mixin.py`.

Instead, create a plain venv that pixi never touches:

```bash
uv venv --python 3.14 .venv-mace
uv pip install --python .venv-mace/bin/python torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-mace/bin/python goal-atomistic   # or, from a clone: -e .
uv pip install --python .venv-mace/bin/python "mace-torch>=0.3.16" jupyter ipykernel
.venv-mace/bin/python -m ipykernel install --user --name goal-mace-cuda \
    --display-name "GOAL (mace-cuda, standalone)"
```

Use the **"GOAL (mace-cuda, standalone)"** kernel for the MACE and UPET notebooks (`mace_foundation_gmd_gallery.ipynb`, `upet_petmad_gmd_gallery.ipynb`, `mof74_mn_bulk.ipynb`, `feature_extraction_demo.ipynb`). Run MACE fine-tuning with the same interpreter:

```bash
.venv-mace/bin/goal-train --config-name finetune/mace
```

> [!IMPORTANT]
> e3nn 0.4.4 loads its bundled Wigner-3j constants with a bare `torch.load(...)`, and torch ≥ 2.6 blocks that by default (`weights_only=True`). Run this **before** importing `mace` or `e3nn`:
>
> ```python
> import torch
> torch.serialization.add_safe_globals([slice])
> ```

This setup has been verified end to end with Python 3.14, torch 2.13+cu126, e3nn 0.4.4 and mace-torch 0.3.16. `mace_mp()` loads a real checkpoint on the GPU, and `goal.ml.data.graph.AtomicGraph` still imports, because it does not touch e3nn's API. GOAL's own e3nn-based models (SIMURGH etc.) are **not** expected to work in this venv.

## Ray Tune and Optuna

`ray[tune]` and `optuna` are needed for `goal-tune` with `method: ray`, which `src/goal/configs/ml/simurgh_gmd26_hpo.yaml` uses. Ray publishes Python 3.14 wheels for x86-64 and aarch64, and both packages are already in every ML pixi environment. With pip, use `pip install "goal-atomistic[tune]"`. `method: tuner` (Lightning's LR / batch-size finder) and `method: wandb` (W&B Sweeps) need no extra packages.

See [Hyperparameter tuning](../ml/training/hyperparameter-tuning.md).

## Verify your install

Inside the environment you installed (any directory works):

```python
import torch, lightning, e3nn, torch_geometric, hydra, ase

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("lightning", lightning.__version__, "| e3nn", e3nn.__version__, "| PyG", torch_geometric.__version__)
print("hydra", hydra.__version__, "| ase", ase.__version__)

import goal.ml.cli  # importing the CLI package populates every registry
from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY

print("backbones:", BACKBONE_REGISTRY.list_available())
print("heads:", HEAD_REGISTRY.list_available())
```

The backbone list should include `simurgh_arace`, `nf_simurgh`, `monolithic_arace`, `simurgh_simplified_routed` and friends. Then check that the console scripts are on your `PATH` and that Hydra can compose the default config without training anything:

```bash
pixi run -e cuda goal-train --cfg job -p model.backbone.name   # prints: simurgh_arace
pixi run -e cuda goal-inspect --help
pixi run -e cuda test                          # or: pytest -k 'not slow'
```

`--cfg job` makes Hydra print the composed config and exit. It is the cheapest way to confirm that `src/goal/configs/ml/` was found.

In a `-full` environment, also check the MD module and its two bundled engines:

```bash
pixi run -e cuda-full python -c "from goal.md import CalculatorFactory; from tblite.ase import TBLite; print(sorted(CalculatorFactory._builders))"
pixi run -e cuda-full which pw.x mpirun
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `MissingConfigException: Cannot find primary config ...` | The `--config-name` you passed does not exist in `goal/configs/ml/` (every command's default config exists since October 2026). List them with `ls src/goal/configs/ml` (outside a clone: `goal.CONFIGS_DIR`). See [CLI](../ml/reference/cli.md). |
| `too many values to unpack` inside `e3nn/util/codegen` | You are running MACE against e3nn ≥ 0.5. Use the `.venv-mace` interpreter. |
| `ModuleNotFoundError` for `flashmd`, `ipi` or `tblite` | The MD module's optional dependencies are not installed. Use a `-full` environment (`pixi install -e cuda-full`), `pip install "goal-atomistic[md]"`, or conda-forge `goal-atomistic-md`. (`goal-simulate` itself no longer needs `typer`.) |
| `pixi install -e cuda` fails with a missing `__cuda` virtual package | The machine has no NVIDIA driver (or one older than CUDA 12). Use `-e cpu`, or `-e rocm` on AMD hardware. |
| The pixi environment "lost" a package you installed | You installed it with `uv pip` into a pixi environment, and pixi reconciled the environment back to the lockfile. Re-install it, or add it to `pyproject.toml`. |
