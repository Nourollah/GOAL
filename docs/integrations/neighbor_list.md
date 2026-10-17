# Neighbour lists

GOAL builds neighbour lists in two places: once per structure when a dataset is processed or `GOALCalculator` evaluates a structure, and for whole batches of structures as device tensors when atoms move every step (the [nvalchemi wrapper](nvalchemi_wrapper.md) and [batched simulation](batched_simulation.md)). This page describes the backends for both, including the pure-PyTorch backend `"torch"`, the `"auto"` choice, and the batched builder that uses NVIDIA's Warp kernel on NVIDIA GPUs.

Source:
[`goal.ml.data.neighbor_list`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/data/neighbor_list.py) (the implementation) and
[`goal.ml.nn.primitives.neighbor_list`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/primitives/neighbor_list.py) (a tensor-level front end).

## What it is

Every backend returns the same thing: for a cutoff $r_c$, all ordered pairs $(i, j)$ and integer cell shifts $S$ with

$$\lVert \mathbf{r}_j - \mathbf{r}_i + S \cdot \mathbf{h} \rVert \le r_c,$$

where the rows of $\mathbf{h}$ are the lattice vectors. `edge_index[0]` is the centre atom $i$, the edge vector is $\mathbf{r}_j - \mathbf{r}_i + S\mathbf{h}$ (ASE's convention), and periodic images of an atom itself are neighbours, as in ASE. GOAL models never call a neighbour list in their forward pass: they re-derive differentiable edge vectors from `graph.pos` and the stored `unit_shifts`, so a neighbour list only decides *which* edges exist.

Per-structure backends, for `goal.ml.data.neighbor_list.build_neighbor_list(atoms, ...)`:

| Backend | Implementation | PBC | Notes |
|---|---|---|---|
| `"ase"` | `ase.neighborlist.neighbor_list` | Yes | **Default.** No extra dependency |
| `"matscipy"` | `matscipy.neighbours.neighbour_list` | Yes | Faster for large cells. Falls back to `"ase"` with a warning when matscipy is missing |
| `"nvalchemiops"` | `nvalchemiops.torch.neighbors.neighbor_list` (Warp, O(N) cell list) | Yes | Copies the structure to the GPU and back. Falls back to `"ase"` with a warning without the package or CUDA |
| `"torch"` | Pure PyTorch: dense distances plus explicit periodic image shifts | Yes | No extra dependency; runs on CUDA, ROCm, MPS and CPU |
| `"radius_graph"` | `torch_geometric.nn.radius_graph` | **No** | Legacy. PyG ≥ 2.7 needs `pyg-lib` or `torch-cluster` for it, and neither is in the pixi environments, so it falls back to `"torch"` with a warning |
| `"auto"` | Resolved per size class | Yes | `"nvalchemiops"` for structures with at least `AUTO_GPU_MIN_ATOMS` (1000) atoms when the kernels and an NVIDIA GPU are available; otherwise `"matscipy"` if installed, else `"ase"` |

Batched builder, for many structures given as tensors on a device: `build_neighbor_list_batch(positions, cutoff, batch, cell, pbc, backend)`, with backends `"auto"`, `"nvalchemiops"` and `"torch"` (see [API reference](#api-reference)).

All backends were checked against ASE and give identical edge sets, including the shift vectors, for a molecule, an fcc bulk cell, a small triclinic cell where atoms see their own images, a slab periodic in two directions, and positions far outside the cell. The batched builder matches per-structure ASE on CPU, on CUDA and with the nvalchemiops kernel.

## Hardware requirements

| Backend | NVIDIA CUDA | AMD ROCm | Apple MPS | CPU |
|---|---|---|---|---|
| `ase`, `matscipy` | CPU code | CPU code | CPU code | Yes |
| `torch` | Yes | Yes | Yes | Yes |
| `nvalchemiops` | Yes | No (falls back) | No (falls back) | No (falls back) |
| `auto` (per structure) | nvalchemiops for ≥ 1000 atoms, else matscipy/ASE | matscipy or ASE | matscipy or ASE | matscipy or ASE |
| `auto` (batched) | nvalchemiops for CUDA tensors | `torch` | `torch` | `torch` |

The `"torch"` backend is dense: per structure it computes distances for every atom pair and every periodic image shift, in blocks of up to 27 shifts, so memory grows as $O(N^2)$ per block. That is fine for molecules and cells of a few hundred atoms. For structures beyond a few thousand atoms, use `nvalchemiops` on NVIDIA or `matscipy` elsewhere.

## Installation

| Backend | Comes with |
|---|---|
| `ase`, `torch` | Core GOAL dependencies; nothing to install |
| `matscipy` | Every pixi environment; with pip, `goal-atomistic[matscipy]` |
| `nvalchemiops` | The `cuda` and `cuda-full` pixi environments; with pip, `goal-atomistic[nvalchemiops]`, `[nvalchemiops-cu12]` or `[nvalchemiops-cu13]` |
| `radius_graph` | Needs `pyg-lib` or `torch-cluster`, which GOAL does not install |

Check that the kernels are usable:

```bash
pixi run -e cuda python -c "from goal.ml.utils.hardware import nvalchemiops_available; print(nvalchemiops_available())"
```

## Quick start

One structure:

```python
from ase.build import bulk
from goal.ml.data.neighbor_list import build_neighbor_list

atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
nl = build_neighbor_list(atoms, cutoff=5.0, backend="torch")
print(nl.edge_index.shape, nl.unit_shifts.shape)   # (2, E), (E, 3)
```

A batch of structures as tensors (on the GPU when available):

```python
import numpy as np
import torch
from ase.build import bulk
from goal.ml.data.neighbor_list import build_neighbor_list_batch

frames = [bulk("C", "diamond", a=3.57), bulk("C", "diamond", a=3.6) * (2, 1, 1)]
device = "cuda" if torch.cuda.is_available() else "cpu"
positions = torch.tensor(np.concatenate([a.positions for a in frames]), device=device)
batch = torch.cat([torch.full((len(a),), i) for i, a in enumerate(frames)]).to(device)
cell = torch.tensor(np.stack([np.array(a.cell) for a in frames]), device=device)
pbc = torch.tensor(np.stack([a.pbc for a in frames]), device=device)

nl = build_neighbor_list_batch(positions, 4.0, batch=batch, cell=cell, pbc=pbc)   # backend="auto"
print(nl.edge_index.shape)   # global indices into `positions`
```

Training data: set the backend in the data config (`data.neighbor_list_backend: torch`, for example). At inference, `GOALCalculator(checkpoint_path=..., neighbor_list_backend="auto")`.

## API reference

### `build_neighbor_list(atoms, cutoff, backend="ase", dtype=torch.float64)`

In `goal.ml.data.neighbor_list`. Per-structure neighbour list of an ASE `Atoms`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `atoms` | `ase.Atoms` | — | The structure |
| `cutoff` | `float` | — | Cutoff radius, Å |
| `backend` | `str` | `"ase"` | `"ase"`, `"matscipy"`, `"nvalchemiops"`, `"torch"`, `"radius_graph"` or `"auto"`. Unknown names fall back to `"ase"` with a warning |
| `dtype` | `torch.dtype` | `torch.float64` | Dtype of `edge_vectors` and `edge_lengths` |

`build_neighbor_list_from_tensors(positions, atomic_numbers, cell, pbc, cutoff, backend="ase", dtype=torch.float64)` does the same for one structure given as tensors (it builds a temporary `Atoms`).

### `build_neighbor_list_batch(positions, cutoff, batch=None, cell=None, pbc=None, backend="auto", sort=True)`

In `goal.ml.data.neighbor_list`. Neighbour list for many structures at once, on the tensors' device.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `positions` | `Tensor (N, 3)` | — | Concatenated positions, any device |
| `cutoff` | `float` | — | Cutoff radius, Å |
| `batch` | `Tensor (N,)` or `None` | `None` | Sorted structure index per atom (PyG convention); `None` means one structure |
| `cell` | `Tensor (B, 3, 3)` or `(3, 3)` | `None` | Lattice vectors as rows; a single cell is shared by all structures |
| `pbc` | `Tensor (B, 3)` or `(3,)` | `None` | Periodic flags. **Required whenever `cell` is given** |
| `backend` | `str` | `"auto"` | `"auto"`, `"nvalchemiops"` or `"torch"`; anything else raises `ValueError` |
| `sort` | `bool` | `True` | Sort edges lexicographically by `(i, j, unit_shift)`, so every backend and device returns the same order |

Backend resolution: `"auto"` uses the nvalchemiops kernel when `nvalchemiops_available()` is true **and** the tensors are on a CUDA device, otherwise `"torch"`. An explicit `"nvalchemiops"` that cannot run degrades to `"torch"` with a warning; with CPU tensors it quietly uses `"torch"`.

> [!NOTE]
> In cells smaller than twice the cutoff, the same pair `(i, j)` appears several times with different `unit_shifts` (one edge per periodic image). Because the sort key includes the shift, these repeated pairs also come out in the same order on every backend; a test checks this element by element for a 2-atom diamond cell at a 4 Å cutoff, on CPU, CUDA and the nvalchemiops kernel.

### `NeighborListResult`

| Field | Shape, dtype | Meaning |
|---|---|---|
| `edge_index` | `(2, E)` int64 | `edge_index[0]` = centre atom $i$, `edge_index[1]` = neighbour $j$. Global (batch-level) indices for the batched builder |
| `edge_vectors` | `(E, 3)` float | $\mathbf{r}_j - \mathbf{r}_i + S\mathbf{h}_{\mathrm{cell}(i)}$ |
| `edge_lengths` | `(E,)` float | Norms of `edge_vectors` |
| `unit_shifts` | `(E, 3)` int64 | Integer cell shifts $S$; zero for non-periodic directions |

### Front end: `goal.ml.nn.primitives.neighbor_list.build_neighbor_list`

```python
from goal.ml.nn.primitives.neighbor_list import build_neighbor_list

edge_index, edge_vec = build_neighbor_list(pos, cutoff, batch=None, cell=None, pbc=None, backend="auto")
```

A thin wrapper around `build_neighbor_list_batch` that returns `(edge_index, edge_vec)` only. It accepts the aliases `auto`, `nvalchemi` and `nvalchemiops` (the kernel), and `torch` and `pyg` (both the pure-PyTorch implementation; `"pyg"` does **not** call PyG's `radius_graph`). Any other name raises `ValueError`.

### Constants

| Name | Value | Meaning |
|---|---|---|
| `AUTO_GPU_MIN_ATOMS` | `1000` | Per-structure `"auto"` sends structures at least this large to the GPU kernel; below it, the host-device round trip costs more than it saves |

## How it works

**Where each builder is used.**

| Caller | Builder | Backend setting |
|---|---|---|
| Datasets (training, evaluation) | `build_neighbor_list`, once per structure | `data.neighbor_list_backend`, default `ase` |
| `GOALCalculator` (ASE MD, `goal-simulate`) | `build_neighbor_list`, every evaluation | `neighbor_list_backend=None` → the training config's `data.neighbor_list_backend`, else `"ase"`. `"auto"` is accepted |
| [nvalchemi wrapper](nvalchemi_wrapper.md), [batched simulation](batched_simulation.md) | `build_neighbor_list_batch`, every step | `neighbor_backend="auto"` |
| [Zarr packing](zarr_data.md) with `--compute-edges` | `build_neighbor_list`, once per structure | `--neighbor-list-backend`, default `ase` |

The dataset default stays `"ase"` on purpose: dataset processing is per structure, the cached graphs should not depend on the hardware they were built on, and a GPU only wins for large systems whose lists are rebuilt every step.

**The `"torch"` backend** wraps periodic coordinates into the cell, enumerates the image shifts that can hold a neighbour within the cutoff (for each periodic direction, $\lceil r_c / d \rceil$ images on each side, where $d$ is the distance between the two cell faces spanned by the other lattice vectors), computes all distances with `torch.cdist` block by block, and translates the shifts back to the unwrapped positions. Non-periodic structures use a single `cdist`.

**The batched kernel path** passes `batch_idx` and `batch_ptr` to `nvalchemiops.torch.neighbors.neighbor_list`, so the whole batch is one kernel launch, and edges never link two structures. The portable path loops over the structures with the `"torch"` implementation.

**`"auto"` logging.** Per-structure `"auto"` resolves once per size class (below or at least 1000 atoms) and logs one INFO line per class, for example `GOAL neighbour-list backend (auto, < 1000 atoms structures): matscipy`. The batched builder logs `GOAL batched neighbour-list backend: ...` once.

## Configuration reference

| Key or argument | Where | Values | Default |
|---|---|---|---|
| `data.neighbor_list_backend` | training config | `ase`, `matscipy`, `nvalchemiops`, `torch`, `radius_graph`, `auto` | `ase` |
| `neighbor_list_backend` | `GOALCalculator(...)` | same | `None` (from the training config) |
| `neighbor_backend` | `SIMURGHNvalchemiWrapper(...)`, `batch_from_tensors(...)` | `auto`, `nvalchemiops`, `torch` | `auto` |
| `--neighbor-list-backend` | `goal-pack-zarr` | as `data.neighbor_list_backend` | `ase` |

For example:

```yaml
data:
  dataset_type: xyz
  cutoff: 5.0
  neighbor_list_backend: torch   # portable, no extra dependency
```

The batched simulation (`goal-simulate-batch`) has no neighbour-list key; it always uses `"auto"`.

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `UserWarning: radius_graph is unavailable ('radius_graph' requires 'pyg-lib>=0.6.0'); using the pure-PyTorch backend.` | PyG ≥ 2.7 without `pyg-lib` or `torch-cluster` | Harmless; set `backend="torch"` to silence it. `radius_graph` has no PBC anyway |
| `ValueError: pbc is required whenever cell is given.` | `build_neighbor_list_batch` (or the front end) got a `cell` without `pbc` | Pass `pbc` of shape `(B, 3)` or `(3,)` |
| `ValueError: build_neighbor_list_batch backend must be 'auto', 'nvalchemiops' or 'torch', got '...'` | Per-structure names (`ase`, `matscipy`) passed to the batched builder | Use one of the three; the front end additionally accepts `nvalchemi` and `pyg` |
| `WARNING GOAL batched neighbour-list: backend 'nvalchemiops' requested but it needs nvalchemi-toolkit-ops and an NVIDIA CUDA GPU; falling back to 'torch'.` | Explicit `nvalchemiops` on a machine without the kernels or an NVIDIA GPU | Expected on CPU, ROCm and MPS. Use `"auto"` to avoid the warning |
| `UserWarning: nvalchemi-toolkit-ops is not installed; falling back to ASE neighbor list.` / `... requires CUDA but no GPU was found ...` | Per-structure `backend="nvalchemiops"` without the package or a GPU | Use `"auto"`, or a `cuda`/`cuda-full` environment on an NVIDIA machine |
| `UserWarning: matscipy is not installed; falling back to ASE neighbor list.` | matscipy missing (pip install without the extra) | `pip install "goal-atomistic[matscipy]"` |
| `UserWarning: found nvalchemi-toolkit-ops version 0.5.0, this code was only tested with ...` | Emitted by `metatomic_ase` in the `-full` environments, not by GOAL | Harmless |
| `CUDA out of memory` or very slow neighbour lists for a large structure with `backend="torch"` | The `"torch"` backend is dense, $O(N^2)$ per image block | Use `nvalchemiops` (NVIDIA) or `matscipy` for structures beyond a few thousand atoms |
| Edges link atoms of different structures | Positions of several structures passed without `batch` (treated as one structure), or a `batch` vector that is not sorted | Pass a sorted `batch` vector (PyG convention: structure 0's atoms first, then structure 1's, ...) |
