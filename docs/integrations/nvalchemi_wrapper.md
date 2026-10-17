# nvalchemi model wrapper

`SIMURGHNvalchemiWrapper` exposes a trained GOAL checkpoint (SIMURGH or any other GOAL model) through nvalchemi-toolkit's model protocol, `nvalchemi.models.base.BaseModelMixin`. nvalchemi's batched, GPU-resident engines (`NVTLangevin`, `FIRE2`, `FusedStage` pipelines and their hooks) can then drive a GOAL model directly, with many structures in one GPU batch. [`goal-simulate-batch`](batched_simulation.md) is built on this wrapper; use the wrapper itself when you want to write your own nvalchemi pipeline.

Source:
[`goal.ml.adapters.nvalchemi_wrapper`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/adapters/nvalchemi_wrapper.py).
Background and installation: [NVIDIA nvalchemi integration](nvalchemi_overview.md).

## What it is

A thin adapter between nvalchemi's data containers and GOAL's graphs:

- nvalchemi hands the model an `AtomicData` or `Batch` (positions, atomic numbers, cells, ...);
- the wrapper builds GOAL's neighbour list on the GPU, assembles the same PyG `Batch` of `AtomicGraph` objects the model was trained on, and runs the model;
- it returns `energy` `[B, 1]` in eV and `forces` `[V, 3]` in eV/Å. These are nvalchemi's units, which are GOAL's, so no conversion happens.

On an NVIDIA RTX 6000 Ada the wrapper reproduces `GOALCalculator` to about 1e-15 eV in energies and 1e-17 eV/Å in forces, for batches of molecules and for batches of periodic crystals, and `NVTLangevin` and `FIRE2` run with it.

## Hardware requirements

An **NVIDIA CUDA GPU**, and `nvalchemi-toolkit` installed. nvalchemi supports neither AMD ROCm nor Apple MPS. On those machines, and on CPU, use the ASE route instead:

- `GOALCalculator` with any ASE integrator ([ASE calculator](../ml/deployment/ase-calculator.md)), or
- `goal-simulate-batch --backend ase` ([Batched simulation](batched_simulation.md)), which `--backend auto` selects there by itself.

Importing `goal.ml.adapters.nvalchemi_wrapper` always works. Constructing the wrapper without nvalchemi-toolkit or an NVIDIA GPU raises `RuntimeError` with installation instructions (`INSTALL_HINT`).

## Installation

`nvalchemi-toolkit` is not part of the pixi environments and not a GOAL extra: its PyPI releases do not support Python 3.14 yet, and installing it from git into a pixi environment would replace the environment's torch. Use the layered `.venv-nvalchemi` described in [NVIDIA nvalchemi integration](nvalchemi_overview.md#installation), and run your scripts with `.venv-nvalchemi/bin/python`.

The neighbour-list kernels (`nvalchemi-toolkit-ops`) come with the `cuda` and `cuda-full` environments and are therefore visible in the venv as well.

Check:

```bash
.venv-nvalchemi/bin/python -c "from goal.ml.utils.hardware import nvalchemi_available; print(nvalchemi_available())"
```

## Quick start

Langevin MD of several molecules in one GPU batch:

```python
import torch
from ase.build import molecule
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.hooks import NaNDetectorHook

from goal.ml.adapters.nvalchemi_wrapper import SIMURGHNvalchemiWrapper

model = SIMURGHNvalchemiWrapper("logs/simurgh_gmd26/runs/<run>/checkpoints", device="cuda")
dtype = next(model.parameters()).dtype         # build the batch in the model's precision

datas = []
for atoms in [molecule("CH4"), molecule("C2H6"), molecule("C2H2")]:
    data = AtomicData.from_atoms(atoms, device="cuda", dtype=dtype)
    data.forces = torch.zeros(len(atoms), 3, device="cuda", dtype=dtype)   # output buffers
    data.energy = torch.zeros(1, 1, device="cuda", dtype=dtype)
    datas.append(data)
batch = Batch.from_data_list(datas, device=torch.device("cuda"))

nvt = NVTLangevin(
    model=model,
    dt=0.5,              # fs
    temperature=300.0,   # K
    friction=0.01,       # 1/fs
    n_steps=1000,
    hooks=[NaNDetectorHook()],
)
batch = nvt.run(batch)
print(batch.energy.squeeze(-1))   # eV, one per structure
```

The `forces` and `energy` buffers matter: nvalchemi 0.2 silently drops model outputs that have no preallocated buffer (`main` allocates them lazily, but preallocating works with both).

Geometry relaxation uses `FIRE2` the same way. Its `dt` is **not** converted from fs upstream; it is a raw optimiser step, and GOAL's batched simulation uses `dt=0.05`:

```python
from nvalchemi.dynamics import FIRE2, ConvergenceHook

fire = FIRE2(model=model, dt=0.05, hooks=[NaNDetectorHook()],
             convergence_hook=ConvergenceHook.from_fmax(0.05))
batch = fire.run(batch, n_steps=500)
```

## API reference

### `SIMURGHNvalchemiWrapper`

```python
SIMURGHNvalchemiWrapper(checkpoint_path, device="cuda", dtype=None, checkpoint="best",
                        use_ema=True, neighbor_backend="auto")
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `checkpoint_path` | `str` or `Path` | — | `.simurgh` archive, checkpoint directory or `.ckpt` ([Checkpoints & archives](../ml/deployment/archives.md)) |
| `device` | `str` or `torch.device` | `"cuda"` | CUDA device, e.g. `"cuda:1"` |
| `dtype` | `torch.dtype` or `None` | `None` | Cast the model; `None` keeps the checkpoint's precision |
| `checkpoint` | `str` | `"best"` | Selector inside a checkpoint directory (`best`, `last`, `epoch=N`) |
| `use_ema` | `bool` | `True` | Use the EMA weights when the checkpoint has them |
| `neighbor_backend` | `str` | `"auto"` | `"auto"` (batched Warp kernel), `"nvalchemiops"` or `"torch"`; see [Neighbour lists](neighbor_list.md) |

Construction returns an instance of a class that derives from `torch.nn.Module`, `nvalchemi.models.base.BaseModelMixin` **and** `SIMURGHNvalchemiWrapper`, so both `isinstance(model, BaseModelMixin)` and `isinstance(model, SIMURGHNvalchemiWrapper)` hold.

Attributes: `model` (the GOAL module, in eval mode), `cutoff` (the checkpoint's cutoff, Å), `weights` (`"ema"` or `"raw"`), `neighbor_backend`, `model_config`.

`model_config` (nvalchemi `ModelConfig`):

| Field | Value |
|---|---|
| `outputs` | `{energy, forces}` |
| `autograd_outputs` | `{forces}` |
| `autograd_inputs` | `{positions}` |
| `required_inputs` | `{}` |
| `optional_inputs` | `{cell, pbc}` |
| `supports_pbc` | `True` |
| `needs_pbc` | `False` |
| `neighbor_config` | `None`: the wrapper builds the neighbour list itself, so no `NeighborListHook` is needed |

`compute_embeddings()` raises `NotImplementedError`, and `embedding_shapes` is empty.

### Field mapping

| nvalchemi | Shape | GOAL | Notes |
|---|---|---|---|
| `positions` | `[V, 3]` | `pos` | Cast to the model dtype; forces are $-\partial E/\partial$ `positions` |
| `atomic_numbers` | `[V]` | `z` | |
| `batch_idx` | `[V]` | `batch` | Absent means one structure |
| `cell` | `[B, 3, 3]` | `cell` | Lattice vectors as rows |
| `pbc` | `[B, 3]` | `pbc` | `cell` without `pbc` is treated as fully periodic; a batch whose `pbc` is all `False` is treated as non-periodic |
| — | — | `edge_index`, `unit_shifts` | Rebuilt at the checkpoint's cutoff on every call |
| `energy` (output) | `[B, 1]` | `energy` | eV |
| `forces` (output) | `[V, 3]` | `forces` | eV/Å |

### Helpers for custom pipelines

| Function | Returns | Meaning |
|---|---|---|
| `atomic_data_to_goal_batch(data, cutoff, dtype, neighbor_backend="auto")` | `(goal_batch, positions_leaf)` | nvalchemi `AtomicData`/`Batch` to a GOAL PyG `Batch`. `positions_leaf` requires grad and the batch's `pos` is computed from it, so `-dE/d(positions_leaf)` are the forces |
| `goal_outputs_to_model_outputs(out, active)` | `OrderedDict` | GOAL prediction dict to nvalchemi's canonical shapes (`energy` `[B, 1]`, `forces` `[V, 3]`, `stress` `[B, 3, 3]` if present and requested) |
| `INSTALL_HINT` | `str` | The message of the `RuntimeError` |

The reverse direction (GOAL graphs to nvalchemi containers) is `goal.ml.data.collate.nvalchemi_collate_fn`; see [Zarr data](zarr_data.md).

## How it works

1. **Construction.** `SIMURGHNvalchemiWrapper.__new__` checks `nvalchemi_available()`. If it is false it raises `RuntimeError(INSTALL_HINT)`. Otherwise it builds (once, cached) a concrete subclass of `nn.Module`, `BaseModelMixin` and `SIMURGHNvalchemiWrapper`, and allocates that. The subclass loads the checkpoint with `load_goal_checkpoint` (the loader `GOALCalculator` uses), refuses fragment-aware checkpoints, and records the cutoff and dtype.
2. **Forward.** For each call nvalchemi makes:
   - positions are detached, cast to the model dtype and made a fresh autograd leaf;
   - `batch_from_tensors` builds the neighbour list for the whole batch with `build_neighbor_list_batch` (one Warp kernel launch on CUDA tensors) and slices it into one `AtomicGraph` per structure;
   - the model runs under `torch.enable_grad()`, so it computes forces by autograd exactly as in training;
   - the outputs are detached and reshaped to nvalchemi's conventions.
3. **Equivalence.** Because the graphs are the training graphs (same cutoff, same `unit_shifts` convention, same `AtomicGraph` fields), the predictions are those of `GOALCalculator` up to floating-point summation order.

### Limitations

- **Fragment-aware checkpoints** (trained with `data.compute_fragment_index`) raise `NotImplementedError`: fragment perception is not available on the batched GPU path. Use `GOALCalculator`.
- **No stress output** yet, so no NPT or cell relaxation through nvalchemi.
- **No embeddings**: `compute_embeddings` raises `NotImplementedError`.
- **Mixed periodic and non-periodic batches.** Upstream `Batch.from_data_list` keeps only the fields of the **first** structure. `AtomicData.from_atoms` gives a molecule `cell=None`, so a batch that starts with a molecule silently drops the cells of the periodic structures after it. Keep periodic and non-periodic structures in separate batches, or give every structure `cell` and `pbc` (identity cell and `pbc=False` for molecules). `goal-simulate-batch` does the latter automatically.

## Configuration reference

The wrapper has **no YAML configuration**; it is a Python API. The YAML-driven way to run a GOAL model through nvalchemi is [`goal-simulate-batch`](batched_simulation.md) with `backend: nvalchemi` (or `auto`) in [`src/goal/configs/md/simulations/batch.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/simulations/batch.yaml).

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: SIMURGHNvalchemiWrapper requires nvalchemi-toolkit and an NVIDIA CUDA GPU.` | nvalchemi-toolkit not importable (for example under `pixi run`), or no NVIDIA GPU (CPU, ROCm, MPS) | Run with `.venv-nvalchemi/bin/python` ([installation](nvalchemi_overview.md#installation)); on other hardware use `GOALCalculator` or `goal-simulate-batch --backend ase` |
| `NotImplementedError: This checkpoint was trained with fragment labels (data.compute_fragment_index); ...` | Fragment labels need bond perception, which the batched GPU path does not do | Use `GOALCalculator` (or `goal-simulate-batch --backend ase`) |
| `energy`/`forces` of the batch stay zero or missing after `run` | nvalchemi 0.2 drops model outputs without preallocated buffers | Preallocate `data.forces = zeros(n, 3)` and `data.energy = zeros(1, 1)` per structure, as in the quick start |
| Periodic structures in a mixed batch are evaluated without their periodic images (no error) | The batch starts with a molecule, whose missing `cell` made `Batch.from_data_list` drop the cells of every later structure | Separate batches, or set `cell` and `pbc` on every structure |
| `NotImplementedError: GOAL checkpoints expose no embeddings to nvalchemi.` | A hook or pipeline stage asked for embeddings | Not supported; remove that stage |
| `FIRE2` relaxation unstable after reusing the MD time step | `FIRE2`'s `dt` is a raw optimiser step, not fs | Start from `dt=0.05` |
