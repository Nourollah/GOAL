# Benchmark datasets

GOAL ships loaders for the standard MLIP benchmarks (rMD17/MD17, MD22, QM9,
plus ANI-1/ANI-1x and SPICE) and ready-made configs for training on them. The
loaders download on first use, convert units to eV and eV/Å, cache the
processed graphs, and split deterministically. They are **optional**: they
live in [`src/goal/ml/data/benchmarks/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/ml/data/benchmarks),
outside the installed `goal` package, and nothing in the core library needs
them.

## Available loaders

| `dataset_type` | Class | Structures | Elements | Labels | Default split |
|---|---|---|---|---|---|
| `rmd17`, `md17` | `MD17Dataset` | 100k frames per molecule (rMD17) | H C N O | energy, forces | 950 / 50 / rest |
| `md22` | `MD22Dataset` | 5k to 85k frames per molecule | H C N O | energy, forces | per-molecule (sGDML paper) / 500 / rest |
| `qm9` | `QM9Dataset` | ~134k molecules | H C N O F | one of 19 properties (**no forces**) | 110k / 10k / rest |
| `ani1x`, `ani1` | `ANI1Dataset` | ~5M (1x) / ~20M (1) | H C N O | energy, forces | 80% / 10% / rest |
| `spice` | `SPICEDataset` | ~1.1M | H C N O F P S Cl Br I | energy, forces | 80% / 10% / rest |
| `gmd` | `GMDDataset` | 282k frames, 140 molecules | H C N O | energy, forces | the paper's per-task windows (train / TestID / TestOOD) |

`md17` and `rmd17` map to the **same class**, and `revised: true` is the
default, so `dataset_type: md17` loads **rMD17** unless you set
`revised: false`. Likewise `ani1` and `ani1x` both default to
`version: "1x"`.

### How they plug in

The adapters live in the `goal.ml.data.benchmarks` module
([`src/goal/ml/data/benchmarks/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/ml/data/benchmarks)).
Importing `goal.ml.data.datasets`, which every CLI does, imports it and
registers the names above lazily. They are part of the installed package, so
they work the same in a clone and in a pip or conda install.

A benchmark loader takes the `split` argument and does its own split. With
`root` pointing at the dataset directory, `GOALDataModule` builds it three
times, with `split="train"`, `"val"` and `"test"`, and **ignores
`split_ratio`**. Split sizes are set with the loader's own keys instead
(`train_size`, `val_size`, `test_size`, `train_fraction`, ...).

In Python:

```python
import goal.ml.data.datasets  # registers md17, rmd17, md22, qm9, ani1, ani1x, spice
from goal.ml.registry import DATASET_REGISTRY

train = DATASET_REGISTRY.build(
    "rmd17", root="data/md17", molecule="aspirin", cutoff=5.0, split="train",
)
g = train[0]
print(len(train), g.pos.shape, g.energy, g.forces.shape)   # 950 (21, 3) ... (21, 3)
print(train.statistics())    # num_structures, avg_num_atoms, energy_mean/std, forces_rms
print(train.citation())      # BibTeX
```

## GMD

[GMD](https://huggingface.co/datasets/AmirMasoud/GMD) (arXiv:2605.08988), the
compositional-generalisation benchmark, is a native dataset: `dataset_type: gmd`
downloads, on first use, only the files a task needs from the Hugging Face Hub
(into the standard Hugging Face cache) and caches the graphs under `data.root`.
Nothing is added by hand, and no molecular data lives in this repository.

```yaml
data:
  dataset_type: gmd
  task: fragment_fusion          # fragment_{chain_extension,fusion,duplication,combination,fission},
                                 # the two *_augmented variants, all, or custom
  format: traj                   # traj | lmdb | zarr | npz: the same labels in four layouts
  seed: null                     # null: the paper's frame windows; an int: random disjoint frames
  n_train: 1000                  # frames per training molecule (paper: [0, 1000))
  n_val: 200                     # validation frames per training molecule (paper: [1000, 1200))
  n_test: 200                    # test frames per molecule (paper: [1400, 1600))
  val_split: test_id             # GOAL's validation set: test_id, val or test_ood
  test_split: test_ood           # GOAL's test set: test_ood (the unseen molecules) or test_id
  train_molecules: null          # replace the task's molecules: "Family/Molecule" or a "Family"
  test_molecules: null
  extra_train_molecules: []      # or add trajectories, e.g. [Alkanes/C8H18]
  extra_test_molecules: []
  fractions: [0.8, 0.1, 0.1]     # task: all
  revision: main                 # a commit hash pins the data
  local_root: null               # a local copy of the repository: nothing is downloaded
  root: ${paths.data_dir}/GMD    # graph cache
```

The selection is the dataset's own: with `seed: null`, frames [0, n_train) of
each training molecule for training, [1000, 1000 + n_val) for `val`,
[1400, 1400 + n_test) for TestID, and [1400, 1400 + n_test) of each unseen
molecule for TestOOD; with a seed, disjoint random frames per molecule, drawn
exactly as the dataset card's `gmd.load(..., seed=...)` draws them. GOAL
validates and selects checkpoints on TestID and tests on TestOOD, so the OOD
number is never used for selection. Frames are non-periodic (the stored cell is
the padding box of the sampling MD), and the isolated-atom references are in
`goal.ml.data.benchmarks.gmd.ATOM_REFS`.

Every GMD config in `src/goal/configs/ml/` (`*_gmd26_*`) uses it. Typical
variations:

```bash
# the paper protocol on Fusion
goal-train --config-name simurgh_dev_gmd26_FF
# a random draw of 500 training frames per molecule, from the Zarr copy
goal-train --config-name simurgh_dev_gmd26_FF data.seed=42 data.n_train=500 data.format=zarr
# your own selection of trajectories
goal-train --config-name simurgh_dev_gmd26_FD data.task=custom \
  'data.train_molecules=[CarboxylicAcids,Alkanes/C8H18]' 'data.test_molecules=[DiCarboxylicAcids]'
```

On a cluster whose compute nodes have no network, load the task once on a
login node (or any machine that shares the Hugging Face cache): the dataset
falls back to the cached files when the Hub cannot be reached.

## Processed caches and cache keys

The first construction downloads the raw files into `<root>/raw/`, builds
neighbour lists, and saves the processed graph list to
`<root>/processed/<name>_cutoff<cutoff>_<dtype>.pt`. Later runs load that
file.

| Loader | `<name>` | Example |
|---|---|---|
| rMD17 / MD17 | `rmd17_<molecule>` / `md17_<molecule>` | `data/md17/processed/rmd17_aspirin_cutoff5.0_float64.pt` |
| MD22 | `md22_<molecule>` | `data/md22/processed/md22_Ac-Ala3-NHMe_cutoff5.0_float64.pt` |
| QM9 | `qm9_target<idx>` (or `qm9_all`) | `data/qm9/processed/qm9_target7_cutoff5.0_float64.pt` |
| ANI | `ani<version>[_max<N>][_elems<Zs>]` | `ani1x_max50000_cutoff5.0_float64.pt` |
| SPICE | `spice<version>_<subset>[_max<N>]` | `spice1_all_cutoff5.0_float64.pt` |

What the key does and does not include matters:

- **Cutoff and dtype are in the key.** Changing `data.cutoff` triggers a
  full rebuild. That takes about 90 s for MD22 Ac-Ala3-NHMe and about
  9 minutes for all of QM9.
- **Split sizes and seed are not in the key.** Splits are drawn from the
  cached list at construction time, so changing `train_size` or `seed`
  costs nothing.
- **Fragment labels are not cached.** They are attached lazily, the first
  time each graph is read, and then kept on the in-memory graph. A cache
  built under one `fragment_scheme` therefore cannot be reused silently
  under another, and each structure is decomposed once per process, not
  once per epoch.

Delete the `.pt` file to force a rebuild, for example after a loader bug
fix.

## Ready-made configs

All of these are self-contained files in
[`src/goal/configs/ml/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/configs/ml) and
run with `goal-train --config-name <name>`.

| Config | Backbone | Dataset | Notes |
|---|---|---|---|
| `simurgh_dev_rmd17_<molecule>` | `simurgh_dev` | rMD17, one file per molecule | 950 train / 50 val; `elements`, `avg_num_neighbors` and `scale` derived from the training set |
| `simurgh_dev_md22_<system>` | `simurgh_dev` | MD22, one file per system | the published training size per system (table below) |
| `hyperspec_md17` | `hyperspec` | rMD17 aspirin | fixed 120-entry embedding, nothing to sync |

```bash
pixi run -e cuda goal-train --config-name simurgh_dev_rmd17_aspirin
pixi run -e cuda goal-train --config-name simurgh_dev_md22_Ac-Ala3-NHMe   # 42 atoms
```

The rMD17 files leave `test_size: null` and switch off
`data.monitor_test_during_training`: the test set is every frame that train
and val did not take (about 99,000 for aspirin), evaluated once at the end of
the run rather than every validation epoch.

## Switching molecules

SIMURGH has one config per molecule (`simurgh_dev_rmd17_ethanol`,
`simurgh_dev_md22_DHA`, ...), and `goal-train` derives its element-dependent
settings from the training set, so a different molecule is a different
`--config-name`. For `hyperspec_md17` it is one override:

```bash
pixi run -e cuda goal-train --config-name hyperspec_md17 data.molecule=ethanol
pixi run -e cuda goal-train --config-name hyperspec_md17 data.molecule=aspirin data.revised=false   # original MD17
```

rMD17 molecules: `aspirin`, `azobenzene`, `benzene`, `ethanol`,
`malonaldehyde`, `naphthalene`, `paracetamol`, `salicylic_acid`, `toluene`,
`uracil`. Original MD17 has the same list without `azobenzene` and
`paracetamol`.

To pin the derived values (for `goal-tune`, which does not derive them),
measure them for the molecule and cutoff:

```python
import goal.ml.data.datasets  # registers the benchmark adapters
from goal.ml.registry import DATASET_REGISTRY
from goal.ml.data.statistics import compute_avg_num_neighbors, compute_unique_elements

ds = DATASET_REGISTRY.build("md22", root="data/md22", molecule="AT-AT", cutoff=5.0, split="train")
print("elements:", compute_unique_elements(ds))
print("avg_num_neighbors:", round(compute_avg_num_neighbors(ds), 4))
```

MD22 molecules and their default (sGDML paper) training sizes:

| Molecule | Atoms | Frames | default `train_size` |
|---|---|---|---|
| `Ac-Ala3-NHMe` | 42 | 85,109 | 6,000 |
| `DHA` | 56 | 69,753 | 8,000 |
| `stachyose` | 87 | 27,272 | 8,000 |
| `AT-AT` | 60 | 20,001 | 3,000 |
| `AT-AT-CG-CG` | 118 | 10,153 | 2,000 |
| `buckyball-catcher` | 148 | 6,102 | 600 |
| `double-walled_nanotube` | 370 | 5,032 | 800 |

Each `simurgh_dev_md22_<system>.yaml` sets its `train_size`. If you change
`data.molecule` instead of the config, override `train_size` too, or the
loader raises because `train_size + val_size` exceeds the frame count. The
MD22 reader checks the `e_unit`/`r_unit` declared in each `.npz` and the
expected atom count rather than assuming them. Only Ac-Ala3-NHMe and AT-AT
are typically pre-downloaded. The others download from `quantum-machine.org`
on first use.

## Keeping the data and model cutoffs in sync

The neighbour list is built at `data.cutoff`, and the model has its own
cutoff for radial bases and envelopes. They must agree. How each config ties
them together differs:

| Config | Model cutoff | Safe override |
|---|---|---|
| `simurgh_dev_*` | `cutoff: ${data.cutoff}` | `data.cutoff=6.0` (`avg_num_neighbors` is derived again when null) |
| `hyperspec_md17` | `model.backbone.cutoff: 5.0` (not interpolated) | `data.cutoff=6.0 model.backbone.cutoff=6.0` |

```bash
pixi run -e cuda goal-train --config-name hyperspec_md17 data.cutoff=6.0 model.backbone.cutoff=6.0
```

A new cutoff also means a new processed cache (it is part of the key), and
more edges per structure, so revisit `batch_size`.

## QM9: no forces

QM9 is a property-prediction set. Each molecule has a single relaxed geometry
and 19 properties, and **no forces**. `data.target` picks the property, and
it is stored as the graph's `energy` (in eV for the energy-like targets):

`mu`, `alpha`, `homo` (default), `lumo`, `gap`, `r2`, `zpve`, `u0`, `u298`,
`h298`, `g298`, `cv`, `u0_atom`, `u298_atom`, `h298_atom`, `g298_atom`, `A`,
`B`, `C`. An integer index from 0 to 18 also works. `target: null` stores
all 19 as separate graph attributes (with `u0` as `energy`) for multi-property
heads.

A force-trained config cannot simply be pointed at QM9. The changes needed:

- the `forces` loss is removed and the stage curriculum collapses to one
  stage;
- checkpoint selection monitors `val_energy/energy_mae_per_atom`. Leaving the
  usual `val_forces/forces_mae` would crash `EarlyStopping` (`strict: true`)
  because force metrics are never emitted without force labels.

> [!WARNING]
> `QM9Dataset` (like `ANI1Dataset` and `SPICEDataset`) does not convert a
> string `data.dtype` into a `torch.dtype`, unlike the MD17 and MD22
> loaders. Building the QM9 cache from scratch with `dtype: float64` in the
> config fails with
> `Invalid device string: 'float64'`. An existing cache loads fine. Until the
> loader is fixed, build the cache once without the key (float64 is the
> default anyway):
> `goal-train --config-name <your_qm9_config> '~data.dtype'`.

## ANI-1 / ANI-1x and SPICE

| Key | ANI | SPICE |
|---|---|---|
| version | `version: "1x"` (default) or `"1"` | `version: "1"` (SPICE 1.1.4, default) or `"2"` (2.0.1) |
| subset | `elements: [1, 6, 7, 8]` filter | `subset:` `all`, `small_molecules`, `amino_acids` or `dipeptides` |
| size cap | `max_structures` | `max_structures` |
| split | `train_fraction: 0.8`, `val_fraction: 0.1` | same |

```bash
# hyperspec has a 120-entry element embedding, so SPICE's Br/I are covered
pixi run -e cuda goal-train --config-name hyperspec_md17 data.dataset_type=spice data.root=data/spice \
    data.subset=dipeptides data.max_structures=50000 \
    '~data.molecule' '~data.revised' '~data.train_size' '~data.val_size' '~data.dtype'
```

The `~` overrides delete the MD17-only keys, which the SPICE loader would
otherwise reject as unexpected arguments.

SPICE needs `h5py` and `requests` (`pip install "goal-atomistic[examples]"`; already in the pixi environments and in conda-forge `goal-atomistic-full`) and
downloads its HDF5 file from Zenodo. Neither loader accepts the
`fragment_*` options: they are dropped when `compute_fragment_index` is
false, and rejected with an error when it is true. Hartree is converted to
eV and Hartree/Bohr to eV/Å.

> [!WARNING]
> `ANI1Dataset` imports `ANI1` / `ANI1x` from `torch_geometric.datasets`, and
> current PyG releases (2.7) do not provide those classes. Building an ANI
> cache therefore fails with a misleading
> `ImportError: torch_geometric is required`. Until the loader gets its own
> downloader, convert ANI-1x to ExtXYZ or HDF5 and use a core reader.

## Fragment options on benchmarks

The MD17, MD22 and QM9 loaders accept the same `compute_fragment_index` /
`fragment_*` keys as the core readers (see
[Loading data](loading.md#fragment-labels)). Every benchmark here is a single
molecule per structure, so `fragment_scheme: connected` returns **one
fragment** and a fragment-interaction channel has no pairs to work with. Use
`rotatable`, `brics`, `recap` or `custom`. A trajectory never changes its
bond graph, so RDKit perception is cached and a whole MD trajectory costs a
single decomposition.

> [!TIP]
> Do not use `rotatable` blindly on MD17 or MD22. It cuts every non-ring
> single bond, which strips a bare `O` and a lone `HO` off aspirin and
> splits the peptide bond of Ac-Ala3-NHMe into `CO` + `HN`. That is why the
> shipped configs use `custom` SMARTS cuts instead:
>
> - aspirin: `fragment_smarts: ["[C:1](=O)-[O:2]", "c-[CX3]"]`, `fragment_keep_groups: ["C(=O)[OX2H1]"]`
> - Ac-Ala3-NHMe: `fragment_smarts: ["[CX4:1]-[CX3:2](=O)"]`, `fragment_keep_groups: ["C(=O)N"]`
>
> A pattern with three or more atoms needs `:1`/`:2` atom maps to say which
> bond to cut. Unmapped, it is rejected as ambiguous.

## Units

All loaders store energies in **eV** and forces in **eV/Å**:

| Source unit | Factor | Target |
|---|---|---|
| kcal/mol (MD17, MD22) | × 0.043364 | eV |
| kcal/mol/Å | × 0.043364 | eV/Å |
| Hartree (ANI, SPICE) | × 27.211396 | eV |
| Hartree/Bohr | × 51.422065 | eV/Å |

Citations for each dataset are in
[`src/goal/ml/data/benchmarks/README.md`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/data/benchmarks/README.md)
and available programmatically via `dataset.citation()`.
