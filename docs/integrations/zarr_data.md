# Zarr data

GOAL can store atomic structures and their labels in a [Zarr](https://zarr.dev) store: chunked, compressed arrays on disk with random access to any structure. A store holds several splits (train, val, test, or MD trajectories) side by side and can also keep precomputed neighbour lists. Train on it with `data.dataset_type: zarr`, write it with `goal-pack-zarr` or `write_zarr`. This page also covers the collate functions that turn GOAL graphs into batches, including an nvalchemi `Batch` for nvalchemi pipelines.

Source:
[`goal.ml.data.zarr_dataset`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/data/zarr_dataset.py),
[`goal.ml.cli.pack_zarr`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/cli/pack_zarr.py),
[`goal.ml.data.collate`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/data/collate.py).
Other dataset formats: [Loading data](../ml/data/loading.md).

## What it is

A GOAL Zarr store (format tag `goal-zarr/1`) has one group per split. Structures are concatenated along the first axis and delimited by per-structure counts:

```text
store.zarr/                    attrs: format = "goal-zarr/1"
  {split}/                     attrs: num_structures, [cutoff]
    n_atoms         (S,)       int64
    positions       (N, 3)     float64   Å
    atomic_numbers  (N,)       int64
    cell            (S, 3, 3)  float64   Å, lattice vectors as rows
    pbc             (S, 3)     bool
    energies        (S,)       float64   eV          (optional)
    forces          (N, 3)     float64   eV/Å        (optional)
    stress          (S, 3, 3)  float64   eV/Å³       (optional)
    n_edges         (S,)       int64                 (optional, with cutoff)
    edge_index      (E, 2)     int64     per-structure atom indices
    unit_shifts     (E, 3)     int64
```

$S$ is the number of structures, $N$ the total number of atoms and $E$ the total number of edges in the split.

Compared with one ExtXYZ file per split, a store reads a single structure without parsing the rest of the file, compresses well, appends in blocks while it is written, and can carry neighbour lists, so a dataset does not have to rebuild them when it is opened with the same cutoff. `ZarrDataset` returns the same `AtomicGraph` fields as the xyz dataset; this is verified field by field against `AtomicGraph.from_ase` on the same frames.

> [!NOTE]
> nvalchemi-toolkit has its own Zarr format (`AtomicDataZarrWriter` / `AtomicDataZarrReader`, a CSR layout with `meta/atoms_ptr` and `core/<field>` arrays). It is a different layout: GOAL stores and nvalchemi stores are not interchangeable.

## Hardware requirements

None. Zarr storage is CPU-side I/O and works with every GOAL environment: NVIDIA CUDA, AMD ROCm, Apple MPS and CPU. Only `nvalchemi_collate_fn` has a hardware-dependent result: an nvalchemi `Batch` on NVIDIA GPUs with nvalchemi-toolkit, and a PyG `Batch` everywhere else.

## Installation

`zarr>=3` is part of every pixi environment (the default feature), so there is nothing to install. With pip:

```bash
pip install "goal-atomistic[zarr]"
```

zarr 2.x is rejected with the installation hint below. Check:

```bash
pixi run -e cuda python -c "import zarr; print(zarr.__version__)"
```

## Quick start

Pack the splits into one store (any format `ase.io.read` understands works as input):

```bash
pixi run -e cuda goal-pack-zarr --input data/train.xyz --output data/gmd.zarr --split train
pixi run -e cuda goal-pack-zarr --input data/val.xyz   --output data/gmd.zarr --split val
pixi run -e cuda goal-pack-zarr --input data/test.xyz  --output data/gmd.zarr --split test
```

Each call prints a summary such as `Packed 1200 structures into data/gmd.zarr/train: atomic_numbers, cell, energies, forces, pbc, positions`.

Train on it. Point every split at the same store; the split name selects the group:

```yaml
data:
  dataset_type: zarr
  cutoff: 5.0
  batch_size: 8
  train_paths: [data/gmd.zarr]
  val_paths: [data/gmd.zarr]
  test_paths: [data/gmd.zarr]
```

or on the command line. The default experiment (`simurgh_gmd26`) loads its splits from `data.train_dir` / `val_dir` / `test_dir`, which take precedence over per-split paths, so switch those off and add the paths with `+`:

```bash
pixi run -e cuda goal-train data.dataset_type=zarr \
    data.train_dir=null data.val_dir=null data.test_dir=null data.root=data/gmd.zarr \
    '+data.train_paths=[data/gmd.zarr]' '+data.val_paths=[data/gmd.zarr]' '+data.test_paths=[data/gmd.zarr]'
```

From Python:

```python
from ase.io import iread
from goal.ml.data.zarr_dataset import ZarrDataset, write_zarr

write_zarr(iread("data/train.xyz", index=":"), "data/gmd.zarr", split="train", cutoff=5.0)

ds = ZarrDataset("data/gmd.zarr", cutoff=5.0, split="train")
graph = ds[0]                       # AtomicGraph: pos, z, cell, pbc, edge_index, unit_shifts, energy, forces, ...
print(len(ds), graph.pos.shape, graph.energy)
```

## API reference

### `write_zarr`

```python
write_zarr(frames, store, split="train", cutoff=None, neighbor_list_backend="ase",
           overwrite=False, chunk_rows=4096, flush_every=1000) -> Path
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `frames` | iterable of `ase.Atoms` | — | Any iterable, including a lazy `ase.io.iread` generator |
| `store` | `str` or `Path` | — | Store directory (`*.zarr`); created if missing, other splits in it are kept |
| `split` | `str` | `"train"` | Group name: `train`, `val`, `test` or anything else |
| `cutoff` | `float` or `None` | `None` | Also store neighbour lists for this cutoff (Å) |
| `neighbor_list_backend` | `str` | `"ase"` | Backend for those neighbour lists ([Neighbour lists](neighbor_list.md)) |
| `overwrite` | `bool` | `False` | Replace an existing `split` group; without it an existing group is an error |
| `chunk_rows` | `int` | `4096` | Zarr chunk length along the first axis |
| `flush_every` | `int` | `1000` | Append buffered structures to the store every this many structures |

Labels are read the way the xyz and trajectory datasets read them: `atoms.info` / `atoms.arrays` first (`energy`, `forces`, `stress`), then an attached calculator. A label array is written only if **every** structure in the split has that label; otherwise it is dropped for the whole split. Stress is normalised to 3×3 (Voigt 6-vectors are accepted). Other `atoms.info` and `atoms.arrays` entries are not stored.

### `ZarrDataset`

```python
ZarrDataset(root, cutoff, split="train", dtype=torch.float64, neighbor_list_backend="ase", **fragment_options)
```

Registered as `dataset_type: zarr`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `root` | `str` or `Path` | — | The `*.zarr` store |
| `cutoff` | `float` | — | Neighbour-list cutoff, Å |
| `split` | `str` | `"train"` | Group to read |
| `dtype` | `torch.dtype` | `torch.float64` | Floating-point dtype of the graphs |
| `neighbor_list_backend` | `str` | `"ase"` | Backend used when neighbour lists are rebuilt |
| fragment options | — | — | `compute_fragment_index` and the other `fragment_*` options, as for every GOAL dataset |

`ds[i]` returns an `AtomicGraph` with `pos`, `z`, `cell`, `pbc`, `edge_index`, `unit_shifts`, `energy` (shape `(1,)`), `forces`, `stress`, `weight` and, with fragment options, `fragment_index`. Negative indices work.

### `goal-pack-zarr`

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | One or more structure files, any `ase.io` format, read lazily with `iread` |
| `--output` | required | Store directory, created if missing |
| `--split` | `train` | Group name inside the store |
| `--compute-edges` | off | Also store neighbour lists; needs `--cutoff` |
| `--cutoff` | — | Cutoff in Å for `--compute-edges` |
| `--neighbor-list-backend` | `ase` | `ase`, `matscipy`, `nvalchemiops`, `torch` or `auto` |
| `--overwrite` | off | Replace an existing split group |

Exit code 0 on success, 1 on errors (printed as one line).

### Collate functions (`goal.ml.data.collate`)

| Function | Returns | Use |
|---|---|---|
| `goal_collate_fn(graphs)` | PyG `Batch` | What GOAL models consume. Works on every device |
| `batch_from_tensors(positions, atomic_numbers, cutoff, batch=None, cell=None, pbc=None, neighbor_backend="auto")` | PyG `Batch` | A GOAL batch built straight from device tensors, with the neighbour list built on the same device. `positions` is used as is, so a caller that set `requires_grad` can differentiate through it. Used by the nvalchemi wrapper and batched simulation |
| `atomic_graph_to_atomic_data(graph, device=None)` | `nvalchemi.data.AtomicData` | One graph to nvalchemi's container, labels in nvalchemi's shapes (`energy` `[1, 1]`, `forces` `[N, 3]`, `stress` `[1, 3, 3]`); `energy`/`forces` buffers are always allocated. Needs nvalchemi |
| `nvalchemi_collate_fn(graphs, device="cuda")` | nvalchemi `Batch`, or PyG `Batch` | GPU-resident nvalchemi `Batch` on NVIDIA with nvalchemi-toolkit; otherwise falls back to `goal_collate_fn` and logs `GOAL collate backend: pyg (nvalchemi unavailable)` once |

`nvalchemi_collate_fn` is deliberately **not** the default collate of `GOALDataModule`. GOAL models take `AtomicGraph` batches; an nvalchemi `Batch` is for feeding GOAL datasets into nvalchemi pipelines, for example:

```python
from torch.utils.data import DataLoader
from goal.ml.data.collate import nvalchemi_collate_fn
from goal.ml.data.zarr_dataset import ZarrDataset

ds = ZarrDataset("data/gmd.zarr", cutoff=5.0, split="train")
loader = DataLoader(ds, batch_size=64, collate_fn=nvalchemi_collate_fn)
batch = next(iter(loader))   # nvalchemi.data.Batch on cuda (NVIDIA + nvalchemi), else a PyG Batch
```

## How it works

**Writing.** `write_zarr` opens the store in append mode, sets the root attribute `format = "goal-zarr/1"`, refuses to replace an existing split unless `overwrite=True`, and creates the split group. Structures are buffered and appended to growable arrays in blocks of `flush_every`, so arbitrarily long inputs stream through constant memory. At the end, label arrays that some structure lacked are deleted, and the group attributes `num_structures` and (with neighbour lists) `cutoff` are written.

**Reading.** `ZarrDataset` reads `n_atoms` once and builds atom offsets, so `ds[i]` reads exactly the rows of structure `i`. Stored neighbour lists are reused only when the store's `cutoff` attribute equals the dataset's cutoff (to 1e-9 Å); otherwise neighbour lists are built per structure with `neighbor_list_backend`, exactly as the xyz dataset does. The store is opened once per process and reopened after a fork or pickling, so `num_workers > 0` in the DataLoader is safe.

**Splits in the datamodule.** `GOALDataModule` creates every dataset with the split it is building (`train`, `val`, `test`), and `ZarrDataset` reads the group of that name. With `train_paths` / `val_paths` / `test_paths` (above), `test_paths` is optional. With `data.root: data/gmd.zarr`, a store behaves like a directory of named split files: a missing group raises `FileNotFoundError`, so a store with `train` and `val` groups but no `test` group simply has no test split, and a store with only a `train` group is split by `data.split_ratio` (the store directory counts as one source, `ZarrDataset.ROOT_IS_DIRECTORY`).

**Per-structure metadata.** `write_zarr(..., info_keys=("step", "temperature"))` keeps scalar `atoms.info` values, one per structure, under `{split}/info/<key>`; `ds.structure_info(i)` returns them as a dict. Every structure must have each listed key. They are not attached to the `AtomicGraph`, so training is unaffected.

## Configuration reference

| Key | Value | Meaning |
|---|---|---|
| `data.dataset_type` | `zarr` | Select `ZarrDataset` |
| `data.train_paths`, `data.val_paths`, `data.test_paths` | `[path/to/store.zarr]` | Store per split; the split name selects the group |
| `data.root` | `path/to/store.zarr` | Alternative to the per-split paths; see [How it works](#how-it-works) |
| `data.cutoff` | float | Model cutoff; stored neighbour lists are reused only if it matches the store's |
| `data.neighbor_list_backend` | `ase` (default), ... | Backend for neighbour lists that are rebuilt |
| `data.compute_fragment_index`, `data.fragment_*` | | Fragment labels, as for other datasets |
| `data.num_workers` | int | Safe with Zarr stores |

`goal-pack-zarr` and `write_zarr` have no YAML configuration. In `goal-simulate-batch`, `trajectory_format: zarr` writes trajectories as a GOAL store ([Batched simulation](batched_simulation.md)).

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ImportError: Zarr storage needs zarr>=3: pip install 'goal-atomistic[zarr]' (or 'zarr>=3')` | zarr missing, or zarr 2.x (the message then ends with `(found zarr 2.x)`). In a pixi environment: the environment was installed before zarr was added to it | `pip install "goal-atomistic[zarr]"`; in pixi, `pixi install -e <env>` |
| `FileExistsError: data/gmd.zarr already has a 'train' group (pass overwrite=True).` | Writing a split that already exists | `overwrite=True`, or `--overwrite` with `goal-pack-zarr`; or choose another `--split` |
| `FileNotFoundError: "data/gmd.zarr has no 'test' group; groups: ['train', 'val']"` | Your code (or `test_paths`) asked for a split the store does not have | Pack that split with `goal-pack-zarr --split test`, or drop `test_paths` |
| `goal-pack-zarr: --compute-edges needs --cutoff` (exit 1) | `--compute-edges` without a cutoff | Add `--cutoff <Å>` |
| `FileNotFoundError: Zarr store not found: ...` | Wrong `root` / path | Fix the path; relative paths resolve from the launch directory |
| `graph.energy` (or `forces`) is `None` for every structure of a split, although most input frames had it | One structure in the split lacked that label, so the whole array was dropped when the split was written | Check the input; every frame needs the label. Re-pack with `--overwrite` |
| Training still reads `train_dir` data, or `Could not override 'data.train_paths'` | The experiment config uses `train_dir`/`val_dir`/`test_dir` (Mode 4 wins), and has no `train_paths` key | `data.train_dir=null data.val_dir=null data.test_dir=null '+data.train_paths=[...]'`, as in the quick start |
| Dataset access is slower than expected | The store was written without neighbour lists, or with another cutoff, so they are rebuilt per structure | Re-pack with `--compute-edges --cutoff <model cutoff>` |
