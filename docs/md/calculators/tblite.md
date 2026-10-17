# tblite (GFN-xTB)

The `tblite` backend runs Grimme's extended tight-binding methods (GFN1-xTB, GFN2-xTB and IPEA1-xTB) through [tblite](https://github.com/tblite/tblite), the maintained library implementation of these Hamiltonians. The builder is [`TBLiteBuilder`](https://github.com/Nourollah/GOAL/blob/main/src/goal/md/core/calculator_factory.py), selected with `key: tblite`, and it returns `tblite.ase.TBLite`.

tblite runs **in-process**: it starts no subprocess and writes no input or output files. It returns **analytic forces**, plus the **stress tensor** for periodic systems. These have been checked against finite differences of the energy. This makes it GOAL's cheap, physics-based option.

## When to use it

| Use tblite for | Use something else for |
|----------------|------------------------|
| Exploratory MD of a new system before you have any training data | Labels you intend to train on or publish: use DFT ([Quantum ESPRESSO](espresso.md), [CP2K](cp2k.md), [VASP](vasp.md), [ORCA](orca.md)) |
| Pre-equilibrating a structure (removing RDKit/MMFF strain, relaxing a solvated box) before an expensive DFT or MTS run | Production MD on a system you already have data for: a [trained GOAL model](goal-model.md) is faster per step |
| Conformer screening and pre-optimisation ahead of DFT | Reaction barriers, spin-state energetics or transition-metal chemistry where xTB is known to be unreliable |
| A cheap slow level for [multiple-time-step MD](../mts/overview.md), or a cheap reference while you set up an [online-learning](../mts/overview.md#online-learning-with-a-qm-reference) workflow | |

In terms of cost, tblite sits between the ML potentials and DFT. GFN2-xTB on ethanol (9 atoms) takes about 2 ms per MD step on a workstation CPU. A plane-wave DFT single point on a single water molecule with Quantum ESPRESSO takes about 12 s on 2 MPI ranks. Expect tblite to be three to four orders of magnitude cheaper than DFT for small molecules. The gap narrows as systems grow, because xTB still diagonalises a dense Hamiltonian.

## Installation

tblite is part of every `-full` pixi environment (conda-forge `tblite-python`), on both x86-64 and aarch64:

```bash
pixi install -e cuda-full          # or cpu-full / rocm-full
pixi run -e cuda-full python -c "from tblite.ase import TBLite; print('tblite OK')"
```

With pip, `pip install "goal-atomistic[md]"` installs the `tblite` wheel on x86-64 only, because PyPI publishes no aarch64 wheels. On aarch64 without pixi, install `tblite-python` from conda-forge. If tblite is missing, `CalculatorFactory.create("tblite")` raises `ImportError` and names the fix.

## Quick start

```python
from ase.build import molecule
from goal.md import CalculatorFactory

atoms = molecule("H2O")
atoms.calc = CalculatorFactory.create("tblite", method="GFN2-xTB")
print(atoms.get_potential_energy())   # -137.968 eV
print(atoms.get_forces())             # eV/Å, analytic
```

## Parameters

| Argument | Default | Meaning |
|----------|---------|---------|
| `method` | `"GFN2-xTB"` | `"GFN2-xTB"`, `"GFN1-xTB"` or `"IPEA1-xTB"` |
| `charge` | `None` | Total charge. `None` → sum of the atoms' initial charges (see [Charge and spin](#charge-and-spin)) |
| `multiplicity` | `None` | Spin multiplicity $2S+1$. `None` → from the atoms' initial magnetic moments |
| `accuracy` | `1.0` | Numerical accuracy; smaller is tighter (SCC convergence, integral cutoffs) |
| `electronic_temperature` | `300.0` | Fermi smearing temperature, K |
| `max_iterations` | `250` | Maximum SCC iterations |
| `solvation` | `None` | Implicit solvent as a tuple/list, e.g. `("alpb", "water")` (see [Implicit solvation](#implicit-solvation)) |
| `verbosity` | `0` | tblite print level. `0` is silent, which you want inside an MD loop; `1` prints the SCC iterations |

Which method to pick:

- **GFN2-xTB**, the default, is the most accurate general-purpose choice for molecules. It has multipole electrostatics and D4 dispersion.
- **GFN1-xTB** is the simpler, older parameterisation. It is sometimes more robust for inorganic and periodic systems.
- **IPEA1-xTB** is a GFN1 reparameterisation for ionisation potentials and electron affinities. It is rarely the right choice for MD.

Energies from different methods have different zeros: GFN2 and GFN1 differ by about 19 eV for water. Never mix methods within one dataset or one MTS run.

## Charge and spin

tblite's ASE calculator resolves charge and spin like this:

- **Total charge:** the `charge` argument if given, otherwise `atoms.get_initial_charges().sum()`.
- **Unpaired electrons:** `multiplicity - 1` if `multiplicity` is given, otherwise `round(atoms.get_initial_magnetic_moments().sum())`.

Passing them explicitly is the least ambiguous route:

```python
from ase.build import molecule
from goal.md import CalculatorFactory

oh = molecule("OH")

oh.calc = CalculatorFactory.create("tblite", charge=-1, multiplicity=1)   # hydroxide anion
print(oh.get_potential_energy())   # -127.395 eV

oh.calc = CalculatorFactory.create("tblite", multiplicity=2)              # neutral OH radical
print(oh.get_potential_energy())   # -120.497 eV
```

To keep charge and spin with the structure instead, for example when they differ per frame, set them on the atoms and leave both arguments at `None`:

```python
import numpy as np

atoms.set_initial_charges(np.array([-1.0] + [0.0] * (len(atoms) - 1)))   # total charge -1
atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))                 # closed shell
```

> [!WARNING]
> **Initial magnetic moments that come with the structure count.** Some structure sources set initial magnetic moments. For example, `ase.build.molecule("OH")` puts 0.5 on each atom, and many CIF and extxyz files carry them too. If you then set only a charge (say −1 on OH), tblite still sees one unpaired electron from the magnetic moments and fails with `Total number of electrons (8) and number unpaired electrons (1) is not compatible`. Pass `multiplicity` explicitly, or zero the magnetic moments.

`MoleculeFactory.create("from_smiles", ...)` does not carry the SMILES formal charge into `ase.Atoms` (see [Molecules](../guide/molecules.md)), so give `charge=` for ions built from SMILES.

## Implicit solvation

`solvation` is passed to tblite unchanged. Its first element is the model name, and the remaining elements are the model's arguments:

| Model | Example | Notes |
|-------|---------|-------|
| ALPB | `("alpb", "water")`, `("alpb", "dmso")`, `("alpb", "water", "bar1mol")` | Analytical linearised Poisson–Boltzmann; recommended for GFN2. The optional third element is the solution state |
| GBSA | `("gbsa", "water")` | Generalised Born with surface area |
| GB | `("gbe", 78.4, "p16")` | Generalised Born from a dielectric constant and a Born kernel |

```python
atoms = molecule("H2O")
atoms.calc = CalculatorFactory.create("tblite", solvation=("alpb", "water"))
print(atoms.get_potential_energy())   # -138.365 eV (gas phase: -137.968 eV)
```

> [!NOTE]
> The tblite 0.7 build in the pixi environments rejects `cpcm`, `cosmo` and `pcm` with `Interaction or post processing '…-solvation' is not supported in this calculator`. Use ALPB or GBSA.

## Periodic systems and stress

Set `atoms.pbc` and a cell, and tblite switches to periodic boundary conditions. It then returns the stress tensor in addition to energy and forces, so NPT dynamics and cell relaxations work:

```python
from ase.build import bulk

si = bulk("Si", "diamond", a=5.43)
si.calc = CalculatorFactory.create("tblite", method="GFN1-xTB")
print(si.get_potential_energy(), si.get_stress())   # stress in eV/Å³, Voigt order
```

tblite has no k-point sampling: periodic systems are treated at the Γ point of the cell you give it. Use a supercell large enough to converge the electronic structure. xTB parameters were fitted to molecules, so validate against DFT before you trust periodic xTB numbers for anything quantitative.

## Hydra configuration

[`src/goal/configs/md/calculators/tblite.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/calculators/tblite.yaml):

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: all
key: tblite

method: GFN2-xTB               # GFN2-xTB, GFN1-xTB or IPEA1-xTB
charge: null                   # null -> from the atoms' initial charges
multiplicity: null             # 2S+1; null -> from the atoms' initial magnetic moments
accuracy: 1.0
electronic_temperature: 300.0  # K
max_iterations: 250
solvation: null                # e.g. [alpb, water]
verbosity: 0
```

Here `key: tblite` selects the backend, and `method:` is a builder argument (the xTB Hamiltonian), so the file instantiates directly:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load("src/goal/configs/md/calculators/tblite.yaml")
cfg.solvation = ["alpb", "water"]
calc = instantiate(cfg)            # -> tblite.ase.TBLite
```

In a simulation config (`goal-simulate`, `MTSSimulation.from_config`), use a plain mapping with `type:` and no `_target_`/`_convert_`:

```yaml
calculator:
  type: tblite
  method: GFN2-xTB
  solvation: [alpb, water]
```

To swap the backend of an existing simulation config from the command line (see [CLI](../reference/cli.md#common-overrides)):

```bash
pixi run -e cuda-full goal-simulate "${CFG[@]}" '~calculator' '+calculator={type: tblite, method: GFN2-xTB}'
```

## Examples

### Langevin MD

```python
from goal.md import CalculatorFactory, MoleculeFactory, Simulation

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
calc = CalculatorFactory.create("tblite", method="GFN2-xTB")

sim = Simulation(
    atoms, calc,
    steps=4000, temperature_K=300.0, timestep_fs=0.5, friction=0.01,
    log_interval=20,
    trajectory_file="ethanol.traj", log_file="ethanol.log",
    output_dir="outputs/md/ethanol_gfn2",
)
result = sim.run()
print(result.steps_completed, result.final_energy, result.final_temperature)
```

### Pre-equilibrate, then hand over to DFT

```python
from ase.optimize import BFGS

from goal.md import CalculatorFactory, MoleculeFactory

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
atoms.calc = CalculatorFactory.create("tblite")
BFGS(atoms).run(fmax=0.05)                 # remove force-field strain cheaply
atoms.write("ethanol_gfn2.xyz")            # starting geometry for the DFT or MTS run
```

### As the slow level of RESPA

When DFT is too expensive even every few steps, tblite can serve as the slow (reference) level, with a GOAL model or a foundation model as the fast level:

```python
from goal.md import CalculatorFactory, MoleculeFactory
from goal.md.mts import MTSConfig, MTSSimulation

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
fast = CalculatorFactory.create("goal_model", checkpoint="models/ethanol.simurgh", device="cpu")
slow = CalculatorFactory.create("tblite", method="GFN2-xTB")

cfg = MTSConfig(timestep_fs=0.5, mts_ratio=4, total_steps=2000, temperature_K=300.0, friction=0.01,
                output_dir="outputs/md/mts/ethanol_gfn2")
result = MTSSimulation(atoms, fast, config=cfg, slow_calculator=slow).run()
```

The same works from a config with `slow_calculator: {type: tblite, method: GFN2-xTB}`. RESPA samples the slow level's potential energy surface, so the result is GFN2-xTB-quality dynamics at a reduced number of xTB calls. That pays off most when the fast model was trained on GFN2-xTB data, so that the slow correction is small. See [Multiple time-step MD](../mts/overview.md) and [RESPA](../mts/respa.md).

## Limitations

- **Semi-empirical accuracy.** GFN2-xTB is good for geometries, conformational preferences and non-covalent interactions of organic and many organometallic systems. Reaction and activation energies have errors of several kcal/mol, and spin-state ordering and open-shell transition-metal chemistry are unreliable. Use it to explore and screen, not as ground truth.
- **Not a substitute for DFT labels.** A model trained on xTB data learns xTB, including its errors. For datasets you train on, use a DFT backend, and do not mix xTB and DFT energies in one dataset.
- **SCC convergence.** Stretched bonds, high temperatures and open-shell systems can fail to converge. ASE then raises `CalculationFailed`. Raising `electronic_temperature` (for example to 1000–3000 K) or `max_iterations` usually helps. A higher electronic temperature changes the energy slightly, so keep it fixed within a run.
- **Charges and spin are your responsibility** (see [above](#charge-and-spin)).
