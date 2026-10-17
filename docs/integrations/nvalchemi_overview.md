# NVIDIA nvalchemi integration and hardware fallback

GOAL can use two NVIDIA packages for GPU fast paths: batched GPU-resident MD and relaxation, and GPU neighbour lists. Every one of these paths is optional. On AMD ROCm, Apple MPS or CPU, or when the packages are not installed, GOAL runs a portable implementation instead and tells you which one it picked. This page explains the rule all integrations follow, what runs on which hardware, and how to install the NVIDIA packages.

Source:
[`goal.ml.utils.hardware`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/utils/hardware.py).

## What it is

The two packages, with their distribution and import names:

| Distribution | Import | What GOAL uses it for | Upstream |
|---|---|---|---|
| `nvalchemi-toolkit` | `nvalchemi` | Batched, GPU-resident MD and optimisation engine (`NVTLangevin`, `FIRE2`, `FusedStage`, hooks), the model protocol `nvalchemi.models.base.BaseModelMixin`, and the data containers `nvalchemi.data.AtomicData` / `Batch` | [NVIDIA/nvalchemi-toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) |
| `nvalchemi-toolkit-ops` | `nvalchemiops` | NVIDIA Warp kernels, among them the O(N) cell-list neighbour list | [NVIDIA/nvalchemi-toolkit-ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops) |

The integration pages:

| Page | Feature | Needs NVIDIA? |
|---|---|---|
| [LoRA fine-tuning of SIMURGH](lora_finetune.md) | Low-rank adapters on a trained SIMURGH/ARACE checkpoint | No |
| [nvalchemi model wrapper](nvalchemi_wrapper.md) | A GOAL checkpoint as an nvalchemi `BaseModelMixin` model | Yes |
| [Neighbour lists](neighbor_list.md) | `torch`, `auto` and batched GPU neighbour-list backends | Fast path only |
| [Batched simulation](batched_simulation.md) | `goal-simulate-batch`: MD or relaxation of many structures with one model | Fast path only |
| [Zarr data](zarr_data.md) | Chunked, compressed dataset storage and `goal-pack-zarr` | No |
| [Roadmap stubs](roadmap_stubs.md) | Metadynamics, REMD and `goal-distill` placeholders | Planned features |

**The rule.** Every NVIDIA-specific path is used only when the package imports **and** an NVIDIA CUDA GPU is present. Otherwise:

- a portable fallback runs (pure PyTorch, PyG, ASE or matscipy), and one INFO line names the backend that was chosen;
- nothing raises because the NVIDIA packages are missing;
- options that pick a backend default to `"auto"`;
- an explicit request for an unavailable fast path (for example `--backend nvalchemi` on a CPU machine) degrades to the fallback with a WARNING instead of failing.

The one exception is the [nvalchemi model wrapper](nvalchemi_wrapper.md): it *is* the nvalchemi integration, so constructing it without nvalchemi or an NVIDIA GPU raises `RuntimeError` with installation instructions. Importing its module still works everywhere.

## Hardware requirements

What runs on which hardware:

| Feature | NVIDIA CUDA | AMD ROCm | Apple MPS | CPU |
|---|---|---|---|---|
| [LoRA fine-tuning](lora_finetune.md) | Yes (PyTorch + peft) | Yes | Yes | Yes |
| [Neighbour lists](neighbor_list.md) | nvalchemiops Warp kernel: `auto` uses it for single structures with at least 1000 atoms, and the batched builder always uses it for CUDA tensors | Pure PyTorch, matscipy or ASE | Pure PyTorch, matscipy or ASE | Pure PyTorch, matscipy or ASE |
| [nvalchemi model wrapper](nvalchemi_wrapper.md) | Yes | No (`RuntimeError`; use `GOALCalculator`) | No | No |
| [Batched simulation](batched_simulation.md) | nvalchemi: all structures in one GPU batch | ASE + `GOALCalculator`, one structure after another | ASE + `GOALCalculator` | ASE + `GOALCalculator` |
| [Zarr data](zarr_data.md) | Yes | Yes | Yes | Yes |
| `nvalchemi_collate_fn` ([Zarr data](zarr_data.md)) | nvalchemi `Batch` | PyG `Batch` | PyG `Batch` | PyG `Batch` |

How GOAL tells the hardware apart: `detect_compute_backend()` returns `"nvidia"` when `torch.cuda.is_available()` is true on a CUDA build of PyTorch. ROCm builds of PyTorch also report `torch.cuda.is_available()` as true, so GOAL checks `torch.version.hip` and reports `"amd"` for them. Next comes `"mps"` (Apple Silicon), and `"cpu"` is the fallback. nvalchemi supports neither ROCm nor MPS, so on those backends every NVIDIA gate is closed even if the packages happen to be installed.

## Installation

### nvalchemi-toolkit-ops (neighbour-list kernels)

`nvalchemi-toolkit-ops[torch]` is already part of the `cuda` and `cuda-full` pixi environments. Nothing to do there. The CPU and ROCm environments do not contain it, by design.

With pip, use one of the extras:

| Extra | Installs |
|---|---|
| `goal-atomistic[nvalchemiops]` | `nvalchemi-toolkit-ops[torch]` |
| `goal-atomistic[nvalchemiops-cu12]` | `nvalchemi-toolkit-ops[torch-cu12]` (with the CUDA 12 runtime wheels) |
| `goal-atomistic[nvalchemiops-cu13]` | `nvalchemi-toolkit-ops[torch-cu13]` (with the CUDA 13 runtime wheels) |
| `goal-atomistic[zarr]` | `zarr>=3` ([Zarr data](zarr_data.md)) |
| `goal-atomistic[lora]` | `peft>=0.13` ([LoRA fine-tuning](lora_finetune.md)) |

`peft` and `zarr` are in every pixi environment. `goal-atomistic[all]` includes `nvalchemiops`, `zarr` and `lora`.

### nvalchemi-toolkit (batched MD engine)

`nvalchemi-toolkit` is deliberately **not** a GOAL extra. Every PyPI release so far (the latest is 0.2.0) declares `requires-python <3.14`, while GOAL requires Python ≥ 3.14, and a git URL inside an extra would block GOAL's own PyPI upload. On Python 3.14, install it from git `main`.

Do not `pip install` it into a pixi environment. Its dependency `nvidia-physicsnemo` makes pip or uv re-install torch from PyPI (CUDA 13 wheels, several GB) on top of the environment's torch, and pixi does not track the change (the same drift problem as with [MACE](../ml/finetuning/mace.md)). The verified recipe is a separate venv layered on the `cuda-full` pixi environment, which pixi never touches:

```bash
pixi install -e cuda-full    # the base environment the venv reuses

uv venv --python .pixi/envs/cuda-full/bin/python --system-site-packages .venv-nvalchemi
uv pip install --python .venv-nvalchemi/bin/python --no-deps \
    "nvalchemi-toolkit @ git+https://github.com/NVIDIA/nvalchemi-toolkit" nvidia-physicsnemo tensordict
uv pip install --python .venv-nvalchemi/bin/python loguru "jaxtyping>=0.3.2" "plum-dispatch>=2.5.7" \
    "zarr>=3" "periodictable==2.0.2" "plotext<6" click
uv pip install --python .venv-nvalchemi/bin/python --no-deps \
    pyvers cloudpickle s3fs aiobotocore botocore jmespath wrapt aioitertools treelib termcolor
.venv-nvalchemi/bin/python -c "from goal.ml.utils.hardware import nvalchemi_available; print(nvalchemi_available())"
```

The last line prints `True` on a machine with an NVIDIA GPU. `--system-site-packages` makes the venv see torch, GOAL (the editable install) and everything else from the pixi environment, and `--no-deps` stops `nvalchemi-toolkit` and `nvidia-physicsnemo` from pulling in a second torch.

This recipe was verified with nvalchemi-toolkit `main` at commit `f59e28dc`, nvalchemi-toolkit-ops 0.5.0, torch 2.14.1 and Python 3.14. The list of pure-Python dependencies may change upstream. If an import fails with `No module named ...`, install that module with `--no-deps` as well.

Run GOAL with the venv's interpreter. The `goal-*` console scripts in `.pixi/envs/cuda-full/bin` start the pixi interpreter, which does not see the venv, so call the modules instead:

```bash
.venv-nvalchemi/bin/python -m goal.md.cli.simulate_batch --checkpoint model.simurgh --structures frames.xyz
.venv-nvalchemi/bin/python my_script.py
```

On Python ≤ 3.13 (outside GOAL's environments) the PyPI release installs normally: `pip install "nvalchemi-toolkit[cu12]"` (or `[cu13]`).

## Quick start

Check what GOAL detects:

```bash
pixi run -e cuda python -c "from goal.ml.utils.hardware import detect_compute_backend, nvalchemi_available, nvalchemiops_available; print(detect_compute_backend(), nvalchemi_available(), nvalchemiops_available())"
```

On a workstation with an NVIDIA RTX 6000 Ada this prints `nvidia False True` in `cuda-full` (the kernels are there, the engine is not), and `cpu False False` in `cpu-full`. Run the same line with `.venv-nvalchemi/bin/python` to confirm that `nvalchemi_available()` is `True` there.

Then use the features with their defaults. Each one picks its backend by itself:

```python
import logging
logging.basicConfig(level=logging.INFO)    # show the one-line backend choices

from goal.md.core.batched import BatchSimulationConfig, run_batched_simulation

result = run_batched_simulation(
    BatchSimulationConfig(checkpoint="models/water.simurgh", structures=["frames.xyz"], steps=500)
)
print(result.backend)    # "nvalchemi" on NVIDIA with nvalchemi-toolkit, "ase" everywhere else
```

## API reference

All functions live in `goal.ml.utils.hardware` and import without any extra installed.

| Function | Returns | Notes |
|---|---|---|
| `detect_compute_backend()` | `"nvidia"`, `"amd"`, `"mps"` or `"cpu"` | Cached after the first call. Logs the result at INFO |
| `nvalchemi_available()` | `bool` | `nvalchemi` imports **and** `detect_compute_backend() == "nvidia"`. Cached |
| `nvalchemiops_available()` | `bool` | `nvalchemiops.torch.neighbors` imports **and** the backend is `"nvidia"`. Cached. `nvalchemi_ops_available` is an alias |
| `resolve_backend(requested, *, fast, fallback, available, feature)` | `str` | Resolves an `"auto"` or explicit backend request for one feature (below) |
| `clear_cache()` | `None` | Forgets every cached detection result and log line. For tests, or after changing `CUDA_VISIBLE_DEVICES` in-process |

`resolve_backend` parameters:

| Parameter | Type | Meaning |
|---|---|---|
| `requested` | `str` | `"auto"`, the fast backend's name, or any other backend name (returned unchanged) |
| `fast` | `str` | Name of the NVIDIA fast path, e.g. `"nvalchemi"` or `"nvalchemiops"` |
| `fallback` | `str` | Name of the portable fallback, e.g. `"ase"` or `"torch"` |
| `available` | callable | Availability probe for the fast path, e.g. `nvalchemi_available` |
| `feature` | `str` | Human-readable feature name used in the log line |

Resolution: `"auto"` gives `fast` when `available()` is true, else `fallback`, and logs one INFO line per feature and choice. An explicit `fast` that is not available gives `fallback` and a WARNING. Any other name is returned unchanged.

## How it works

1. **Detection.** `detect_compute_backend()` inspects `torch.cuda.is_available()`, `torch.version.hip` and `torch.backends.mps`, in that order. The result is cached for the life of the process.
2. **Package probes.** `nvalchemi_available()` and `nvalchemiops_available()` first check with `importlib.util.find_spec` that the package exists, then import it. An install that is present but fails to import (for example a broken CUDA build) counts as *not available*: GOAL logs the exception at INFO and falls back instead of raising.
3. **Per-feature resolution.** Each feature calls `resolve_backend` with its own fast path and fallback:

   | Feature | `fast` | `fallback` | Log label |
   |---|---|---|---|
   | Batched simulation | `nvalchemi` | `ase` | `GOAL batched simulation backend: ...` |
   | Batched neighbour list | `nvalchemiops` | `torch` | `GOAL batched neighbour-list backend: ...` |

   The per-structure neighbour list (`backend="auto"` in datasets and `GOALCalculator`) and `nvalchemi_collate_fn` use the same probes with their own rules; see [Neighbour lists](neighbor_list.md) and [Zarr data](zarr_data.md).
4. **Logging once.** Resolution can run on every MD step, so each `(feature, backend)` choice is logged only once per process. The messages go through Python's `logging` module. The GOAL CLIs configure logging; in your own scripts, call `logging.basicConfig(level=logging.INFO)` to see them.

The fallback warning has this form:

```text
GOAL batched simulation: backend 'nvalchemi' requested but it needs nvalchemi-toolkit and an NVIDIA CUDA GPU; falling back to 'ase'.
```

## Configuration reference

Hardware detection has **no YAML configuration** and no environment variable of its own. Each feature exposes its own backend option:

| Feature | Where | Key or argument | Default |
|---|---|---|---|
| Neighbour lists in datasets | training config | `data.neighbor_list_backend` | `ase` |
| Neighbour lists at inference | `GOALCalculator(...)` | `neighbor_list_backend` | the training config's value, else `"ase"` |
| nvalchemi wrapper | `SIMURGHNvalchemiWrapper(...)` | `neighbor_backend` | `"auto"` |
| Batched simulation | `goal-simulate-batch` / `batch.yaml` | `--backend` / `backend` | `auto` |
| LoRA fine-tuning | training config | `model.backbone.name: simurgh_lora` | — |
| Zarr datasets | training config | `data.dataset_type: zarr` | — |

To force the portable path on an NVIDIA machine, pass the fallback explicitly (`backend: ase`, `neighbor_backend="torch"`), or hide the GPU with `CUDA_VISIBLE_DEVICES=""`. The latter makes `detect_compute_backend()` return `"cpu"`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvalchemi_available()` is `False` on a machine with an NVIDIA GPU | `nvalchemi-toolkit` is not in the pixi environments | Use the `.venv-nvalchemi` interpreter from [Installation](#installation), not `pixi run` |
| `nvalchemiops_available()` is `False` in `cpu`, `cpu-full`, `rocm` or `rocm-full` | Those environments have no NVIDIA GPU support and do not contain `nvalchemi-toolkit-ops` | Expected. The portable backends run instead |
| `detect_compute_backend()` returns `"amd"` and every NVIDIA gate is closed on an AMD GPU | ROCm build of PyTorch (`torch.version.hip` is set); nvalchemi does not support ROCm | Expected. Batched simulation uses ASE, neighbour lists use PyTorch, matscipy or ASE |
| `pip install nvalchemi-toolkit` fails with a message that the package requires a different Python | The PyPI releases declare `requires-python <3.14` | Install from git `main` as shown above |
| pip or uv starts downloading torch and several GB of CUDA 13 wheels | `nvidia-physicsnemo` (a dependency of nvalchemi-toolkit) requests torch from PyPI | Install `nvalchemi-toolkit`, `nvidia-physicsnemo` and `tensordict` with `--no-deps` into the layered venv. Never into the pixi environment |
| `ModuleNotFoundError: No module named '...'` when importing `nvalchemi` in the venv | A pure-Python dependency that the `--no-deps` install skipped | `uv pip install --python .venv-nvalchemi/bin/python --no-deps <module>` |
| `.venv-nvalchemi/bin/goal-simulate-batch: no such file` or the run uses the ASE backend although the venv has nvalchemi | Console scripts live in the pixi environment and start the pixi interpreter | `.venv-nvalchemi/bin/python -m goal.md.cli.simulate_batch ...` |
| `UserWarning: found nvalchemi-toolkit-ops version 0.5.0, this code was only tested with ...` | Emitted by `metatomic_ase` (a dependency of the MD module in the `-full` environments), not by GOAL | Harmless |
| The backend log line appears only once, or not at all | Choices are logged once per process; scripts that do not configure `logging` hide INFO lines | `logging.basicConfig(level=logging.INFO)`; `clear_cache()` re-runs detection |
