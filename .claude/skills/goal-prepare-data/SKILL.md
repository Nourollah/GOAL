---
name: goal-prepare-data
description: Prepares atomistic datasets for GOAL. Converts quantum-chemistry or MD outputs into extxyz or ASE trajectory files with total energies in eV and forces in eV/Å, chooses data.dataset_type, key_mapping and a loading mode (ratio split, file lists, per-split folders), packs large corpora into Zarr, and checks a split before training. Use when the user wants to train or fine-tune on their own data, asks which file formats GOAL reads, or sees a missing-label error at load time.
license: MIT
---

# Preparing data for GOAL

Full reference: `docs/ml/data/loading.md`. Built-in benchmarks: `docs/ml/data/benchmarks.md`. Large corpora: `docs/integrations/zarr_data.md`.

## 1. Pick the format

| You have | `data.dataset_type` |
|---|---|
| extxyz or xyz files that fit in memory | `xyz` |
| ASE `.traj` files (the GMD benchmark uses these) | `trajectory` |
| more frames than fit in memory | `zarr`, built with `goal-pack-zarr` |
| pre-built graphs in HDF5 | `hdf5` |
| FairChem/OCP pickled LMDB | `lmdb` |
| FairChem ASE-DB shards (`.aselmdb`, e.g. OMol25) | `aselmdb` (not interchangeable with `lmdb`) |
| rMD17, MD17, MD22, QM9, ANI-1, ANI-1x, SPICE | the built-in loaders (`goal-standard-benchmarks`) |

## 2. Write frames that GOAL reads correctly

- Units: positions in Å, the **total** energy of each structure in eV (not per atom), forces in eV/Å, stress in eV/Å³.
  - 1 Hartree = 27.211386 eV; 1 Hartree/Bohr = 51.42208 eV/Å.
  - Forces are minus the energy gradient. Gradient files (for example ORCA `.engrad`) store +dE/dR.
- Energies and forces must come from the same calculation at the same level of theory, including any dispersion correction.
- Keep the raw QM output files. Dispersion terms, dipoles, charges and relabelling all need them later, and an extxyz file cannot give them back.
- Store labels through an ASE `SinglePointCalculator` (read with no configuration), or under your own keys plus `key_mapping`:

```python
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

frames = []
for atoms, energy_ev, forces_ev_per_a in records:        # from your own parser
    atoms.calc = SinglePointCalculator(atoms, energy=energy_ev, forces=forces_ev_per_a)
    frames.append(atoms)
write("data/my_system/all_frames.extxyz", frames)
```

```yaml
data:
  dataset_type: xyz
  root: /data/my_system/all_frames.extxyz
  key_mapping: {energy: REF_energy, forces: REF_forces}   # atoms.info / atoms.arrays keys
```

- Optional Hessian labels (Hessian distillation): a per-atom extxyz array named `hessian`, shape (N, 9N) in eV/Å², holding each atom's three Hessian rows. `AtomicGraph.from_ase` reads it under that fixed name (not through `key_mapping`).
- Frames taken from MD: subsample to decorrelated frames before labelling (a stride, or farthest-point selection on a descriptor). Consecutive frames add cost without information and leak between splits.

## 3. Point GOAL at the data

```bash
# one file, ratio split
pixi run -e cuda goal-train --config-name <experiment> \
    data.dataset_type=xyz data.root=/data/my_system/all_frames.extxyz \
    data.train_dir=null data.val_dir=null data.test_dir=null \
    'data.split_ratio=[0.9,0.05,0.05]'

# pre-split files
pixi run -e cuda goal-train --config-name <experiment> data.dataset_type=xyz \
    data.train_dir=null data.val_dir=null data.test_dir=null \
    '+data.train_paths=[/data/A/train.xyz]' '+data.val_paths=[/data/A/val.xyz]' \
    '+data.test_paths=[/data/A/test.xyz]'
```

Per-split folders use `data.train_dir`, `data.val_dir` and `data.test_dir` (the GMD configs do this). Merging several sources is Mode 3 in `docs/ml/data/loading.md`.

## 4. Pitfalls

- A wrong label key fails at load time and lists the keys the frame has. `key_mapping` wins over the older `energy_key`/`forces_key` form.
- One malformed frame aborts the whole load.
- The named-split lookup checks one extension per reader: `xyz` does not match `train.extxyz`.
- Very small datasets can produce an empty validation or test split.
- `data.cutoff` builds the neighbour list. Configs that interpolate `model.backbone.cutoff: ${data.cutoff}` follow it. Configs that hard-code `elements` or `avg_num_neighbors` must be updated by hand when the data change: a structure with an element missing from `elements` raises `ValueError` (SIMURGH and, since November 2026, the legacy SIMURGH models; it used to contribute zero energy silently).
- Element reference energies: prefer isolated-atom energies at the labels' level of theory (`atomic_energies: {mode: provided, values: {...}}`). `mode: dataset` fits them by least squares on the training compositions, which is unidentifiable when compositions are collinear (homologous series such as GMD; see `goal-gmd-benchmark`). For a single-composition dataset it only sets a constant, which is harmless.
- Frames load as float64 on the CPU; the trainer precision sets the compute dtype.

## 5. Check before training

```python
from omegaconf import OmegaConf
from goal.ml.data.datamodule import GOALDataModule

cfg = OmegaConf.create({"data": {"dataset_type": "xyz", "root": "/data/my_system/all_frames.extxyz",
                                 "cutoff": 5.0, "batch_size": 4, "split_ratio": [0.8, 0.1, 0.1]}})
dm = GOALDataModule(cfg)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
print(len(dm.data_train), len(dm.data_val), batch.num_graphs, batch.pos.shape, batch.energy.shape)
```

Then print the composed config (`goal-train --config-name <experiment> --cfg job -p data`) and run one `+debug=fdr` step (`goal-train-model`).

Large corpora: `pixi run -e cuda goal-pack-zarr --input data/train.xyz --output data/my.zarr --split train`, once per split.

## Done when

- [ ] Split sizes and the shapes of `batch.pos` and `batch.energy` look right, and energies are totals in eV.
- [ ] Per-atom energy and force statistics are physically plausible (no Hartree values, no sign-flipped forces).
- [ ] A `+debug=fdr` training step runs.
