# GMD Benchmark Datasets

Ready-to-use benchmark datasets for training and evaluating MLIPs. **Completely decoupled from the core `gmd` package** — the framework works perfectly without them.

## Quick Start

```bash
# Train on MD17 aspirin (revised)
gmd-train data=md17_aspirin

# Train on ANI-1x, subsample 50k structures
gmd-train data=ani1x data.max_structures=50000

# Train on QM9 predicting HOMO-LUMO gap
gmd-train data=qm9 data.target=gap

# Override cutoff
gmd-train data=md17_aspirin data.cutoff=6.0
```

## Programmatic Usage

```python
import examples.datasets  # triggers lazy registration

from gmd.registry import DATASET_REGISTRY

train_data = DATASET_REGISTRY.build(
    "md17",
    root="data/md17",
    molecule="aspirin",
    revised=True,
    cutoff=5.0,
    split="train",
)

graph = train_data[0]
print(graph.positions.shape)   # (21, 3)
print(graph.energy)            # scalar in eV
print(graph.forces.shape)      # (21, 3) in eV/Å
```

## Available Datasets

| Dataset | Structures | Elements | Properties | Download | Source |
|---------|-----------|----------|------------|----------|--------|
| **MD17** | ~10k/mol | H, C, N, O | energy, forces | ~100 MB | PyG |
| **rMD17** | ~10k/mol | H, C, N, O | energy, forces | ~100 MB | PyG |
| **ANI-1** | ~20M | H, C, N, O | energy, forces | ~30 GB | PyG |
| **ANI-1x** | ~5M | H, C, N, O | energy, forces | ~7 GB | PyG |
| **QM9** | 134k | H, C, N, O, F | 19 properties | ~1 GB | PyG |
| **SPICE** | ~1.1M | H, C, N, O, F, P, S, Cl, Br, I | energy, forces | ~15 GB | Zenodo |

### MD17 / rMD17

Molecular dynamics trajectories for small organic molecules. MD17 computed at CCSD(T)/cc-pVTZ level; **rMD17** (revised) recomputed at PBE/def2-SVP with more consistent reference frame.

**Molecules:** aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic_acid, toluene, uracil (rMD17 adds azobenzene, paracetamol)

**Standard split:** 950 train / 50 val / rest test (Schütt et al. 2017)

**Units:** energy kcal/mol → eV, forces kcal/mol/Å → eV/Å

### ANI-1 / ANI-1x

Large-scale datasets of organic molecule conformations. ANI-1 at ωB97x/6-31G\* level (~20M structures); ANI-1x extended with active learning (~5M structures, higher quality).

**Note:** ANI-1 is ~30 GB, ANI-1x is ~7 GB. First download takes time.

**Units:** energy Hartree → eV, forces Hartree/Bohr → eV/Å

### QM9

134k drug-like organic molecules (≤9 heavy atoms) with 19 quantum chemical properties at B3LYP/6-31G(2df,p) level. **No forces** — property prediction only.

**Targets:** dipole moment, polarisability, HOMO, LUMO, gap, ZPVE, internal energy (0 K, 298 K), enthalpy, free energy, heat capacity, atomisation energies, rotational constants.

**Standard split:** 110k train / 10k val / ~14k test (DimeNet++ protocol)

### SPICE

Drug-like molecules and protein fragments at ωB97M-D3BJ/def2-TZVPPD level. Higher quality than ANI-1, more representative of pharmaceutical chemistry.

**Subsets:** `all`, `small_molecules`, `amino_acids`, `dipeptides`

**Data source:** Downloaded directly from Zenodo as HDF5. Requires `h5py` and `requests`:

```bash
pip install -e ".[examples]"
```

## Unit Conversions

All datasets store energies in **eV** and forces in **eV/Å** internally:

| Source unit | Conversion factor | Target |
|------------|------------------|--------|
| kcal/mol | × 0.043364 | eV |
| kcal/mol/Å | × 0.043364 | eV/Å |
| Hartree | × 27.211396 | eV |
| Hartree/Bohr | × 51.422065 | eV/Å |

## Dependencies

| Dataset | Required packages |
|---------|------------------|
| MD17, rMD17, ANI-1, ANI-1x, QM9 | `torch-geometric` (already a core dep) |
| SPICE | `h5py`, `requests` (`pip install -e ".[examples]"`) |

## Citations

<details>
<summary>BibTeX entries for all datasets</summary>

```bibtex
% MD17
@article{chmiela2017machine,
  title={Machine learning of accurate energy-conserving molecular force fields},
  author={Chmiela, Stefan and Tkatchenko, Alexandre and Sauceda, Huziel E and
          Poltavsky, Igor and Sch{\"u}tt, Kristof T and M{\"u}ller, Klaus-Robert},
  journal={Science advances},
  year={2017}
}

% ANI-1
@article{smith2017ani,
  title={ANI-1: an extensible neural network potential with DFT accuracy
         at force field computational cost},
  author={Smith, Justin S and Isayev, Olexandr and Roitberg, Adrian E},
  journal={Chemical science},
  year={2017}
}

% ANI-1x
@article{smith2020ani1x,
  title={The ANI-1ccx and ANI-1x data sets, coupled-cluster and density
         functional theory properties for molecules},
  author={Smith, Justin S and Zubatyuk, Roman and Nebgen, Benjamin and
          Lubbers, Nicholas and Barros, Kipton and Roitberg, Adrian E and
          Isayev, Olexandr and Tretiak, Sergei},
  journal={Scientific data},
  year={2020}
}

% QM9
@article{ramakrishnan2014quantum,
  title={Quantum chemistry structures and properties of 134 kilo molecules},
  author={Ramakrishnan, Raghunathan and Dral, Pavlo O and Rupp, Matthias
          and von Lilienfeld, O Anatole},
  journal={Scientific data},
  year={2014}
}

% SPICE
@article{eastman2023spice,
  title={SPICE, A Dataset of Drug-like Molecules and Peptides for Training
         Machine Learning Potentials},
  author={Eastman, Peter and Behara, Pavan Kumar and Dotson, David L and
          Galvelis, Raimondas and Herr, John E and Horton, Josh T and
          Mao, Yuezhi and Chodera, John D and Pritchard, Benjamin P and
          Wang, Yuanqing and others},
  journal={Scientific Data},
  year={2023}
}
```

</details>
