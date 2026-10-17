# ASE-LMDB & OMol25

`dataset_type: aselmdb` reads the `.aselmdb` shards that FairChem uses to
distribute its large corpora (OMol25, OMat24, ODAC23) directly. You need
neither a conversion step nor `fairchem-core`. This page covers the format,
OMol25's download layout, the catalogue cache, the subset filters, and the
worked NF_SIMURGH config and exploration notebook.

## Why a separate loader

FairChem writes these files with ASE's own database layer (`ase.db` with the
LMDB backend). They do not use the layout that `dataset_type: lmdb` expects:

| | `lmdb` | `aselmdb` |
|---|---|---|
| Keys | `"0"`, `"1"`, ... (0-based) | `ase.db` row ids `"1"`, `"2"`, ... plus `nextid`, `metadata`, `deleted_ids` |
| Values | pickled dict of tensors and arrays | `zlib.compress(json.dumps(row))`, arrays in ASE's `__ndarray__` encoding |
| Extras | none | `row["data"]`: corpus-specific fields (OMol25: `charge`, `spin`, `data_id`, `source`, population analyses, ...) |

Rows are decoded with `ase.io.jsonio` and `ase.db.row.AtomsRow`, the same
classes that wrote them. Pointing `lmdb` at an `.aselmdb` file fails to
unpickle.

## OMol25 download layout

OMol25 is distributed as tarballs of per-split directories. GOAL expects the
`train_4M` split extracted under `data/`:

```text
data/OMol25/
├── train_4M.tar.gz                        # the download; can be deleted after extraction
└── train_4M/
    ├── data0000.aselmdb                   # 80 shards, ~50k rows each
    ├── data0000.aselmdb-lock              # LMDB lock file (ignored)
    ├── data0000.aselmdb.goal-catalogue.npz   # GOAL catalogue cache (built once, see below)
    ├── ...
    └── data0079.aselmdb
```

```bash
mkdir -p data/OMol25 && cd data/OMol25
# download train_4M.tar.gz from the OMol25 release (Meta FAIR Chemistry), then:
tar -xzf train_4M.tar.gz
```

`train_4M` holds **3,986,754 structures** (ωB97M-V/def2-TZVPD) in 80 shards,
about 20 GB on disk. If you also download the official validation split, put
it next to `train_4M` (for example `data/OMol25/val_4M`) and load it with
`val_dir` (see [below](#using-the-official-splits)).

> [!NOTE]
> `data/` is git-ignored. On a fresh clone, or after moving the data, the
> catalogue caches are missing and have to be rebuilt.

## The catalogue: lazy rows, eager metadata

[`aselmdb.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/data/datasets/aselmdb.py)
has two layers:

- **`ASELMDBReader`** gives random access over one or many shards. It is
  independent of the training stack (it never builds an `AtomicGraph`), so
  notebooks use it too. Rows are addressed by a flat position running over
  the shards in sorted file-name order. LMDB environments open lazily and
  are re-opened after a `fork`, so a reader created in the parent process
  works inside DataLoader workers.
- **`ASELMDBDataset`** is the `dataset_type: aselmdb` dataset. **One
  instance covers one shard.** It filters that shard down to a trainable
  subset and returns `AtomicGraph`s.

Geometry is decoded only in `__getitem__`. The one eager step is the
**catalogue** (`ASELMDBCatalogue`): one pass over every row that records its
element bitmask, atom count, energy, charge, spin, Hill formula and
sub-corpus tag. This summary, about 40 bytes per row, is what lets every
filter below run as a vectorised numpy operation without decoding any
geometry.

Decoding the JSON runs at about 6k rows/s, which is roughly **8 s per
50k-row shard** and about ten minutes for all of `train_4M`. The result is
cached next to each shard as `<shard>.goal-catalogue.npz` (about 1 MB per
50k rows). A cache whose row count no longer matches its shard is rebuilt
automatically. To keep the caches somewhere else, for example when the data
sits on a read-only mount, set

```yaml
data:
  catalogue_cache: true
  catalogue_cache_dir: /scratch/$USER/omol25_catalogues
```

`catalogue_cache: false` skips the cache and rescans every shard each time
the dataset is built.

## Preparing a split: `scripts/omol25_prepare.py`

Build every cache up front, in parallel, so the first training run starts
immediately instead of leaving the GPU idle for ten minutes:

```bash
pixi run -e cuda python scripts/omol25_prepare.py --root data/OMol25/train_4M --workers 8
```

[The script](https://github.com/Nourollah/GOAL/blob/main/scripts/omol25_prepare.py)
does two jobs:

1. **Indexes** every shard (unless you pass `--no-index`), caching each
   catalogue.
2. **Prints the model-side numbers** for the filtered subset, as a YAML block
   ready to paste into `model.backbone`:
   - `elements`
   - `atomic_energies.values`: a least-squares fit of
     $E \approx \sum_Z n_Z e_Z$ over the catalogue's formulas, taking
     seconds instead of hours
   - `scale`, computed as $\mathrm{std}(E - \sum_Z n_Z e_Z) / \langle N_\text{atoms}\rangle$
   - `avg_num_neighbors`, from real neighbour lists over a random sample

| Flag | Default | Meaning |
|---|---|---|
| `--root` | `data/OMol25/train_4M` | split directory |
| `--workers` | `8` | parallel indexing processes |
| `--no-index` | off | skip the scan and use the cached catalogues |
| `--stats-shards N` | all | compute statistics from the first N shards only |
| `--elements Z ...` | `1 6 7 8 9 15 16 17` | allowed atomic numbers |
| `--charges Q ...` | `0` | allowed total charges (pass the flag with no values to allow all) |
| `--spins S ...` | `1` | allowed spin multiplicities (pass the flag with no values to allow all) |
| `--max-atoms` | `128` | atom-count ceiling |
| `--cutoff` | `6.0` | cutoff for `avg_num_neighbors`, in Å |
| `--neighbor-sample` | `4000` | structures sampled for `avg_num_neighbors` (`0` skips it) |
| `--seed` | `0` | sampling seed |

> [!IMPORTANT]
> The energy baseline, `scale` and `avg_num_neighbors` describe the
> **filtered subset at a given cutoff**, not the corpus. Re-run the script
> (`--no-index` is enough once the caches exist) every time you change the
> filters or `data.cutoff`. The script's filters (`--elements`, `--charges`,
> `--spins`, `--max-atoms`) mirror the dataset's, but it has no `data_ids`
> or `min_atoms` option. If you narrow by those, the printed numbers come
> from a slightly wider set. Neither the elements nor the per-shard
> `max_samples` subsample changes the baseline much.

OMol25 energies are raw total energies of order $10^4$ eV. Without a
per-element baseline, the interaction network would have to fit numbers five
orders of magnitude larger than the chemistry it is meant to learn. That is
why these numbers are not cosmetic.

## Filtering: deciding what problem the model sees

`train_4M` spans 83 elements, total charges of roughly −10 to +10, and spin
multiplicities 1 to 11. Each filter is a key in `data:` and is applied
against the catalogue before any geometry is decoded:

| Key | Default | Keeps a row when |
|---|---|---|
| `allowed_elements` | `null` | **every** atom is in the list |
| `excluded_elements` | `null` | it contains none of these |
| `charges` | `null` | its `data["charge"]` is in the list |
| `spins` | `null` | its `data["spin"]` is in the list |
| `data_ids` | `null` | its sub-corpus is in the list |
| `min_atoms` / `max_atoms` | `null` | its atom count is inside the window |
| `require_energy` | `true` | it has a finite energy |
| `sample_stride` | `1` | it is every n-th survivor (applied before `max_samples`) |
| `max_samples` | `null` | it is among the first N survivors **per shard**, or N random ones when `sample_seed` is set |
| `sample_seed` | `null` | (makes `max_samples` draw a deterministic random subset) |

OMol25 sub-corpora (`data_ids`): `biomolecules`, `elytes`, `metal_complexes`,
`reactivity`, `ani2x`, `trans1x`, `geom_orca6`, `rgd`, `orbnet_denali`,
`spice`.

### Charge and spin: pin them

SIMURGH-family backbones take only (Z, positions) as input. OMol25 contains
the **same geometry at different total charge or spin**, and those energies
differ by electron-volts. Leave `charges` and `spins` open and the target is
no longer a function of the input: the loss has an irreducible floor set by
the spread across charge states, not by the model. For any model without
charge or spin inputs, pin them:

```yaml
data:
  charges: [0]
  spins: [1]
```

### Elements: model capacity, not well-posedness

An element filter is a question of what the model can afford. A
SIMURGH-family backbone builds one artisan per element **pair**, so 83
elements means 3,486 pairs per round. Two ways to handle that:

- **Restrict the chemistry** with `allowed_elements`, for example CHNOFPSCl
  `[1, 6, 7, 8, 9, 15, 16, 17]`. This removes the `metal_complexes`
  sub-corpus entirely (about a fifth of the split).
- **Keep every element and route rare pairs** into one shared artisan
  (`model.backbone.artisan.rare_artisan.enabled: true`). The pairs that
  carry the corpus keep dedicated artisans and the long tail shares one.
  This is what `nf_simurgh_omol25.yaml` does now.

> [!WARNING]
> If you set `allowed_elements`, make `model.backbone.elements` the same
> list. An edge whose element pair has no artisan is zero-masked, so that
> pair contributes exactly zero energy and **nothing warns**.

### How much data

`max_samples` caps each shard **after** filtering, so the total is roughly
`80 × max_samples`. Rows inside a shard are grouped by upstream corpus and
are often consecutive conformers of one molecule. The first N rows are
therefore a sample of whichever corpus was written first, not of the shard.
Set `sample_seed` to draw a random subset instead, and use `sample_stride`
to thin out dense conformer runs.

```yaml
data:
  max_samples: 250      # ~20k structures: smoke test
  # max_samples: 4000   # ~320k: a serious run
  # max_samples: null   # the whole filtered subset
  sample_seed: 97
```

## Label keys and row extras

By default the loader reads each row's own `energy` and `forces`, which ASE
attaches as a `SinglePointCalculator`. Every **scalar** field in
`row["data"]` is also copied into `atoms.info`, so any scalar the corpus
records can be used as the target:

```yaml
data:
  energy_key: nl_energy    # OMol25's non-local dispersion term, instead of the total
```

Because a row always carries a calculator with the total energy, a misspelled
key would otherwise fall back to the total energy silently. `ASELMDBDataset`
therefore checks every explicitly configured key against the first row when
it is constructed, and fails with the list of available `row` and `data`
fields.

Two further switches:

- `attach_charge_spin: true` adds `total_charge` and `spin_multiplicity` to
  every graph. It is off by default because no SIMURGH-family backbone reads
  them, and because `total_charge` is a *prediction* key of the dipole head:
  attaching it turns the row's charge into that head's regression target.
- `fragment_charge_from_data: true` uses each row's own charge for RDKit
  bond perception instead of the fixed `fragment_charge`. It only matters
  when `charges` allows more than one value.

## Loading the split

A split directory contains no `train.aselmdb` or `val.aselmdb`, so with
`root` pointing at it, `ASELMDBDataset` deliberately raises
`FileNotFoundError` rather than load the whole directory three times (which
would make train, val and test identical). `GOALDataModule` then falls back
to discovering every shard, building one dataset per shard, concatenating
them, and splitting by `split_ratio`
([Mode 1](loading.md#mode-1-one-source-ratio-split)):

```yaml
data:
  dataset_type: aselmdb
  root: ${paths.data_dir}/OMol25/train_4M
  cutoff: 6.0
  split_ratio: [0.96, 0.02, 0.02]
  split_seed: 97
  charges: [0]
  spins: [1]
  max_atoms: 128
  max_samples: 4000
  sample_seed: 97
  num_workers: 8
  persistent_workers: true
  prefetch_factor: 4
  pin_memory: true
```

### Using the official splits

A random split of `train_4M` is in-distribution by construction. OMol25's own
validation and test splits hold out whole molecules. To use them, switch to
[Mode 4](loading.md#mode-4-per-split-directories):

```yaml
data:
  dataset_type: aselmdb
  train_dir: ${paths.data_dir}/OMol25/train_4M
  val_dir:   ${paths.data_dir}/OMol25/val_4M
  # split_ratio is ignored in Mode 4
  monitor_test_during_training: false   # true only with a genuinely held-out test_dir
```

### Workers matter here

Unlike the eager `xyz` and `trajectory` readers, every `__getitem__`
unzips a JSON row, builds an ASE neighbour list and (if enabled) labels
fragments. The measured rate is about **564 structures/s per process** with
`fragment_scheme: connected`. Set `num_workers` high enough that loading runs
ahead of the training step. Eight workers is what the shipped config uses.

> [!WARNING]
> RDKit-based fragment schemes (`rotatable`, `brics`, ...) run inside
> `__getitem__` for this dataset. On large flexible molecules, xyz2mol bond
> perception can take **seconds per structure**: a 114-atom peptide took
> 5.3 s. One such structure stalls its worker. If you need those schemes,
> close the size window as well (for example `max_atoms: 60`).

## Worked example: `nf_simurgh_omol25.yaml`

[`src/goal/configs/ml/nf_simurgh_omol25.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/nf_simurgh_omol25.yaml)
is a complete, commented NF_SIMURGH run on OMol25:

| Choice | Value | Reason |
|---|---|---|
| `allowed_elements` | `null` (no element filter) | rare-pair routing (`artisan.rare_artisan.enabled: true`, `min_pair_frequency: 0.02`) keeps the artisan bank tractable; keeps `metal_complexes` |
| `charges` / `spins` | `[0]` / `[1]` | the model has no charge or spin input |
| `max_atoms` | `128` | bounds VRAM per batch and the O(K²) fragment-pair enumeration |
| `max_samples` / `sample_seed` | `4000` / `97` | ~320k structures, random within each shard |
| `split_ratio` | `[0.96, 0.02, 0.02]` | 2% is already thousands of structures per evaluation set |
| `cutoff` | `6.0` | at 8 Å the edge count roughly doubles, for interactions the virtual-node channel is meant to carry |
| `fragment_scheme` | `connected` | about half the subset is multi-species (electrolyte clusters, reacting complexes, solvated ions), so `connected` gives K ≥ 2 on ~50% of structures, with whole molecules and ions as fragments |
| `model.backbone.elements`, `atomic_energies`, `scale`, `avg_num_neighbors` | hard-coded | `nf_simurgh` is not auto-instrumented by `train.py`; the values come from `omol25_prepare.py` |

After the charge, spin and size filters, the pool contains about 1.9 M
structures. Run it with:

```bash
pixi run -e cuda python scripts/omol25_prepare.py --root data/OMol25/train_4M --workers 8   # once
pixi run -e cuda goal-train --config-name nf_simurgh_omol25
# smoke test: ~20k structures
pixi run -e cuda goal-train --config-name nf_simurgh_omol25 data.max_samples=250
```

To restrict the chemistry again:

```bash
pixi run -e cuda python scripts/omol25_prepare.py --root data/OMol25/train_4M --no-index \
    --elements 1 6 7 8 9 15 16 17 --charges 0 --spins 1 --max-atoms 128 --cutoff 6.0
```

Then paste the printed block into `model.backbone`, and set
`data.allowed_elements` to the same element list.

## Browsing the corpus: `notebooks/omol25_explore.ipynb`

[`notebooks/omol25_explore.ipynb`](https://github.com/Nourollah/GOAL/blob/main/notebooks/omol25_explore.ipynb)
is the way into the corpus. It uses only `ASELMDBReader` and
`ASELMDBCatalogue`:

- an **overview** of sub-corpora, size distribution, charges, spins and
  elements, computed from the catalogue without decoding any geometry;
- the **training subset**, read straight from `nf_simurgh_omol25.yaml` so
  the notebook cannot drift from the config;
- **lookup** by reader position, by `(shard, row id)` address, by formula,
  by metadata combinations, by provenance (`data["source"]`), and by
  **SMILES** (a formula prefilter, then RDKit `rdDetermineBonds`
  perception);
- **py3Dmol views** with DFT force arrows and atoms coloured by the
  fragment decomposition, which is exactly what NF_SIMURGH pools onto its
  virtual nodes.

The same tools are available in plain Python:

```python
import numpy as np
from goal.ml.data.datasets.aselmdb import (
    ASELMDBReader,
    linear_reference_energies,
    residual_energy_scale,
)

reader = ASELMDBReader("data/OMol25/train_4M")
cat = reader.catalogue(progress=print)      # cached; prints only on a cold scan
print(len(reader), "rows;", cat.unique_elements()[:10], "...")

# neutral singlets of at most 128 atoms, CHNO only
keep = cat.element_mask(allowed=[1, 6, 7, 8])
keep &= (cat.charge == 0) & (cat.spin == 1) & (cat.num_atoms <= 128)
subset = cat.select(keep)

refs = linear_reference_energies(subset)     # {Z: e_Z} in eV
scale = residual_energy_scale(subset, refs)
print(len(subset), refs, scale)

# pull one structure out as ase.Atoms
position = int(reader.positions_of(subset)[0])
atoms = reader.atoms(position)
print(atoms.get_chemical_formula(), atoms.info["data_id"], atoms.info["charge"])
```

`ASELMDBDataset` exposes the same objects for the rows it serves:
`dataset.catalogue` (row *i* describes `dataset[i]`), `dataset.reader`, and
`dataset.atoms(i)` for the raw `ase.Atoms`.
