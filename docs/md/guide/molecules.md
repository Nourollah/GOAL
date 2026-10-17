# Molecules

Every simulation starts from an `ase.Atoms` object. `MoleculeFactory` builds one from a SMILES string or from any structure file ASE can read, and `molecule_tools` adds the two helpers you usually need next: SMILES embedding and putting a molecule in a periodic box.

Because the result is a plain `ase.Atoms`, you can also skip the factory and pass in anything you built with `ase.build`, `pymatgen` or your own code. The factory exists so that configs can say `method: from_smiles` and so that you can [register your own sources](extending.md#custom-molecule-builders).

## MoleculeFactory

```python
from goal.md import MoleculeFactory

atoms = MoleculeFactory.create(key, **kwargs)
```

`create` looks up the builder registered under `key` and forwards every other argument to its `build` method. If the key is unknown, a `ValueError` lists the registered keys. The class lives in `goal.md.core.molecule_factory` and is re-exported from `goal.md` and `goal.md.core`.

| Key | Builder | Arguments | Returns | Status |
| --- | --- | --- | --- | --- |
| `from_smiles` | `FromSmilesBuilder` | `smiles: str` | `ase.Atoms` (no cell, `pbc=False`) | Working; needs RDKit |
| `from_file` | `FromFileBuilder` | `path: str`, `**kwargs` forwarded to `ase.io.read` | `ase.Atoms` or `list[ase.Atoms]` | Working |
| `from_db` | `FromDBBuilder` | `db_path: str`, `**kwargs` | — | **Not implemented**: always raises `NotImplementedError` |

You can list the registered keys at runtime (custom builders included):

```python
from goal.md import MoleculeFactory
print(sorted(MoleculeFactory._builders))   # ['from_db', 'from_file', 'from_smiles']
```

### `from_smiles`

```python
atoms = MoleculeFactory.create("from_smiles", smiles="CC(=O)Oc1ccccc1C(=O)O")  # aspirin
print(atoms.get_chemical_formula(), atoms.pbc, atoms.cell.volume)   # C9H8O4 [False False False] 0.0
```

Internally this calls `generate_3d_coordinates_from_smiles` (below) and wraps the result in `ase.Atoms(symbols, positions=positions)`. The molecule has explicit hydrogens, an MMFF94-relaxed geometry, **no cell and no periodicity**.

Use it when you want a reasonable gas-phase starting geometry for a small or medium organic molecule and do not have a structure file. It is the default in every MD and MTS simulation config (`smiles: CCO`).

Be aware of these points before using it in production:

- **Not reproducible by default.** ETKDG is called without a fixed random seed, so two calls can give different conformers (different torsions or ring puckers). For a reproducible study, generate the structure once, save it with `atoms.write("start.xyz")`, and load it with `from_file`.
- **One conformer only.** For flexible molecules the conformer you get is arbitrary. If conformational sampling matters, enumerate conformers with RDKit yourself and loop over them.
- **Charge and spin are not carried over.** `"[NH4+]"` gives the right atoms, but `ase.Atoms` does not record the formal charge. Pass `charge=` / `multiplicity=` to the calculator yourself (`tblite` and `orca` both take these two arguments).
- **MMFF coverage.** MMFF94 has no parameters for most metals and some unusual valences. In that case RDKit skips the optimisation without raising an error, and you get the unrelaxed ETKDG embedding. Relax the structure with your actual calculator before starting MD.

### `from_file`

```python
# Single frame (for multi-frame files ASE returns the LAST frame by default)
atoms = MoleculeFactory.create("from_file", path="structures/ts_guess.xyz")

# Explicit frame / format, forwarded to ase.io.read
first = MoleculeFactory.create("from_file", path="md.traj", index=0)
frames = MoleculeFactory.create("from_file", path="md.traj", index=":")   # list[Atoms]
poscar = MoleculeFactory.create("from_file", path="POSCAR", format="vasp")
```

All formats `ase.io.read` understands are supported: XYZ/extended XYZ, CIF, POSCAR, `.traj`, XSF, PDB, and so on. Cell, PBC, and (for extended XYZ or `.traj`) stored energies and forces come through exactly as ASE reads them.

> [!IMPORTANT]
> `Simulation` and the dynamics builders need **one** `Atoms` object. If you pass `index=":"` you get a list back, so pick a frame (`frames[-1]`) before simulating.

Use `from_file` for anything periodic (crystals, MOFs, solvated boxes), for structures pre-optimised with a QM code, and for restarting from the last frame of a previous trajectory:

```python
atoms = MoleculeFactory.create("from_file", path="outputs/md/run1/md.traj")  # last frame
```

Velocities stored in a `.traj` frame are restored by ASE. Note that `Simulation` re-draws Maxwell–Boltzmann velocities on start unless you pass `initialise_velocities=False` (see [Dynamics](dynamics.md#simulation)).

### `from_db`

This key is registered as a placeholder for loading from an ASE database, but its builder raises `NotImplementedError("Database loading not yet implemented")`. Until it is implemented, read the database with ASE directly:

```python
from ase.db import connect
atoms = connect("structures.db").get_atoms(id=42)
```

## molecule_tools

`goal.md.core.molecule_tools` holds the geometry helpers. Both are re-exported from `goal.md`.

### `generate_3d_coordinates_from_smiles(smiles)`

```python
from goal.md import generate_3d_coordinates_from_smiles

symbols, positions = generate_3d_coordinates_from_smiles("c1ccccc1")
# symbols: ['C', 'C', ..., 'H'] (12 atoms); positions: (12, 3) numpy array in Å
```

The pipeline is:

1. `Chem.MolFromSmiles`. An unparsable SMILES raises `ValueError("Invalid SMILES: ...")`.
2. `Chem.AddHs`, which makes all hydrogens explicit.
3. `AllChem.EmbedMolecule(mol, AllChem.ETKDG())`, a single conformer. If embedding fails, it raises `ValueError("Could not generate 3D conformation ...")`; this happens mostly with large, strained or cage molecules.
4. `AllChem.MMFFOptimizeMolecule(mol, confId=0)`.

If RDKit is not importable, the function raises `ImportError`. RDKit is a core GOAL dependency, so this only happens in a stripped-down environment.

Call it directly when you want the raw arrays, for example to build `Atoms` with extra info:

```python
from ase import Atoms
symbols, pos = generate_3d_coordinates_from_smiles("O")
water = Atoms(symbols, positions=pos, info={"charge": 0, "smiles": "O"})
```

### `box_molecule(molecule, box_scale=10.0, pbc=True, **kwargs)`

`box_molecule` puts a molecule in a cubic cell **in place** and returns `None`. The cell edge is `box_scale × d`, where `d` is the largest extent of the molecule along x, y or z. The molecule is centred in the box, and PBC is switched on (or off with `pbc=False`).

```python
from goal.md import MoleculeFactory, box_molecule

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
box_molecule(atoms, box_scale=4.0)            # ethanol: d ≈ 2.8 Å → ~11 Å cube
print(atoms.cell.lengths(), atoms.pbc)
```

Use it when a calculator needs a cell:

- **Plane-wave or GPW codes** (CP2K, Quantum ESPRESSO, VASP) always need a simulation cell. A molecule-in-a-box with Γ-point sampling is the standard set-up; see [CP2K](../calculators/cp2k.md), [Quantum ESPRESSO](../calculators/espresso.md) and [VASP](../calculators/vasp.md).
- **ML potentials trained on periodic data** may expect a cell even for a molecule.

Choosing `box_scale`:

- The default `10.0` is very generous: ethanol ends up in a ~38 Å box. For a plane-wave code the cost grows with box volume, so you usually want a vacuum gap (box edge minus molecular size) of about 10–15 Å, which means `box_scale` ≈ 3–5 for small molecules. For tighter control, call `atoms.center(vacuum=7.5); atoms.pbc = True` yourself.
- For a single atom, `d = 0`, so the box size is zero. Set the cell by hand in that case.
- Pass `pbc=False` if you want the cell recorded but the system kept non-periodic (for example, for visualisation, or for a model that should not see periodic images).

> [!WARNING]
> Extra `**kwargs` are forwarded to **both** `atoms.set_cell` and `atoms.center`. Most keyword arguments are valid for only one of them (`scale_atoms` for `set_cell`, `vacuum`/`axis` for `center`), so passing them raises `TypeError`. Leave `kwargs` empty, and post-process with ASE if you need finer control.

## Molecule configs

Two templates live in [`src/goal/configs/md/molecules/`](https://github.com/Nourollah/GOAL/tree/main/src/goal/configs/md/molecules):

| File | Contents |
| --- | --- |
| `ethanol.yaml` | `method: from_smiles`, `smiles: CCO` |
| `from_file.yaml` | `method: from_file`, `path: ???` (mandatory, set on the command line) |

Both files also contain `_target_: goal.md.core.molecule_factory.MoleculeFactory.create`. That `_target_` is **not usable** with `hydra.utils.instantiate` as written, because the factory expects the key as its first positional argument and the files name it `method`. The working way to describe a molecule in a simulation config is a plain `molecule:` mapping, which `Simulation.from_config` reads by popping `method` (default `from_smiles`) and passing the remaining keys to the builder:

```yaml
molecule:
  method: from_file
  path: structures/aspirin.xyz
  index: -1          # any extra key goes to ase.io.read
```

See [Configuration](../reference/configuration.md#molecule-templates) for how to load the template files from Python, and [Usage patterns](usage-patterns.md#pattern-3-config-first) for a complete config-first run.

## Recipes

### A reproducible start for a set of molecules

```python
from pathlib import Path
from goal.md import MoleculeFactory

smiles = {"ethanol": "CCO", "acetone": "CC(C)=O", "phenol": "Oc1ccccc1"}
out = Path("structures"); out.mkdir(exist_ok=True)
for name, smi in smiles.items():
    MoleculeFactory.create("from_smiles", smiles=smi).write(out / f"{name}.xyz")
# From now on use from_file, so every run starts from the same geometry.
```

### Several conformers for one molecule

```python
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
params = AllChem.ETKDGv3(); params.randomSeed = 42
conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)
AllChem.MMFFOptimizeMoleculeConfs(mol)
symbols = [a.GetSymbol() for a in mol.GetAtoms()]
conformers = [Atoms(symbols, positions=mol.GetConformer(i).GetPositions()) for i in conf_ids]
```

### Relax with the production calculator before MD

The MMFF94 geometry is only a starting guess. A short relaxation with the calculator you will actually use for MD avoids a large potential-energy drop (and a temperature spike) in the first few hundred femtoseconds:

```python
from ase.optimize import BFGS
from goal.md import CalculatorFactory, MoleculeFactory

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
atoms.calc = CalculatorFactory.create("goal_model", checkpoint="path/to/best.simurgh")
BFGS(atoms, logfile=None).run(fmax=0.05)
```
