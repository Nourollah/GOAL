# ORCA calculator

The `orca` backend runs [ORCA](https://www.faccts.de/orca/) (DFT, double hybrids, coupled cluster, and ORCA's built-in xTB) as an ASE calculator. It is the natural choice for **Gaussian-basis, all-electron reference labels**, such as the ωB97M-V/def2-TZVPD level used by OMol25 or the PBE/def2-SVP level of GMD. It is also the default *slow* level in the shipped [MTS configs](../mts/overview.md). The builder is [`ORCABuilder`](https://github.com/Nourollah/GOAL/blob/main/src/goal/md/core/calculator_factory.py), selected with `key: orca`. It wraps `ase.calculators.orca.ORCA` with an `OrcaProfile` (current ASE API, ≥ 3.23).

## Installing ORCA

ORCA is not in any GOAL environment and cannot be installed with pixi or pip. It is free for academic use: register at the [ORCA forum](https://orcaforum.kofo.mpg.de/), download the binary package for your platform, and unpack it, for example to `/opt/orca`. Parallel runs (`num_threads > 1`) also need the OpenMPI version that matches your ORCA release on `PATH` and `LD_LIBRARY_PATH`. The ORCA release notes name it.

> [!TIP]
> **Always pass the absolute path to the ORCA executable**, for example `orca_path="/opt/orca/orca"`. ORCA needs its full path to launch its parallel sub-programs. On many Linux desktops a bare `orca` also resolves to `/usr/bin/orca`, the GNOME screen reader, which silently does nothing useful. `which orca` tells you which one you would get.

## Builder interface

```python
from goal.md import CalculatorFactory

calc = CalculatorFactory.create(
    "orca",
    orca_path="/opt/orca/orca",
    keywords="wB97M-V def2-TZVPD RIJCOSX TightSCF DEFGRID3",
    charge=0,
    multiplicity=1,
    num_threads=8,
    blocks="%maxcore 3000\n%scf MaxIter 500 end",
    directory="orca_files",
)
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `orca_path` | required | **Absolute** path to the `orca` executable |
| `keywords` | `"wB97M-V def2-TZVPD RIJCOSX TightSCF DEFGRID3"` | ORCA simple-input line, without the leading `!`. `EnGrad` is prepended automatically if missing |
| `charge` | `0` | Total charge |
| `multiplicity` | `1` | Spin multiplicity $2S+1$ |
| `num_threads` | `1` | MPI processes, written as `%pal nprocs N end` |
| `blocks` | `""` | Extra ORCA `%` input blocks, appended after `%pal` (`%maxcore`, `%scf`, `%method`, …) |
| `directory` | `"orca_files"` | Working directory for `orca.inp` and ORCA's output |

The call above writes this `orca.inp` (for a water molecule):

```text
! EnGrad wB97M-V def2-TZVPD RIJCOSX TightSCF DEFGRID3
%pal nprocs 8 end
%maxcore 3000
%scf MaxIter 500 end
*xyz 0 1
O   0.0 0.0 0.119262
H   0.0 0.763239 -0.477047
H   0.0 -0.763239 -0.477047
*
```

**`EnGrad` is automatic.** MD and geometry optimisation need forces. If `keywords` does not contain `EnGrad`, the builder prepends it, so `keywords="PBE D3BJ def2-SVP"` becomes `! EnGrad PBE D3BJ def2-SVP`. The check is case-insensitive, and a keyword line that already has `EnGrad` is left as it is.

**`%maxcore` is per process**, in MB. With `num_threads=8` and `%maxcore 3000`, ORCA may use about 24 GB.

If ASE's ORCA module cannot be imported, the builder raises `ImportError`.

> [!NOTE]
> **ASE version.** The builder needs ASE ≥ 3.23, where `ORCA` takes a `profile=OrcaProfile(command=...)` plus `orcasimpleinput` / `orcablocks`; GOAL requires `ase>=3.25` for the MD module and is tested with ASE 3.29. A `TypeError` such as `create() missing 1 required positional argument: 'key'` or `got an unexpected keyword argument 'method'` is not an ASE problem: it came from older GOAL templates that named the backend `method: orca`; the shipped ones now use `key: orca`.

## ORCA's built-in xTB

ORCA can run GFN-xTB itself through the same builder:

```python
calc = CalculatorFactory.create("orca", orca_path="/opt/orca/orca", keywords="XTB2")   # GFN2-xTB
calc = CalculatorFactory.create("orca", orca_path="/opt/orca/orca", keywords="XTB1")   # GFN1-xTB
```

This is convenient when the rest of a workflow already runs in ORCA. For xTB on its own, GOAL's [`tblite`](tblite.md) backend is the better tool: it is in the `-full` environments, it runs in-process with no file I/O, and it is much faster per MD step.

## Example

```python
from goal.md import CalculatorFactory, MoleculeFactory

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
atoms.calc = CalculatorFactory.create(
    "orca",
    orca_path="/opt/orca/orca",
    keywords="PBE D3BJ def2-SVP",
    num_threads=4,
    directory="orca_ethanol",
)
print(atoms.get_potential_energy(), atoms.get_forces())
```

ASE writes `orca.inp` into `directory`, runs ORCA there, and parses the energy and the gradient from the output.

### As the slow level of RESPA

```python
from goal.md import CalculatorFactory, MoleculeFactory
from goal.md.mts import MTSConfig, MTSSimulation

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
fast = CalculatorFactory.create("goal_model", checkpoint="models/ethanol.simurgh")
slow = CalculatorFactory.create("orca", orca_path="/opt/orca/orca",
                                keywords="PBE D3BJ def2-SVP", num_threads=8)

sim = MTSSimulation(atoms, fast, config=MTSConfig(timestep_fs=0.5, mts_ratio=4, total_steps=2000),
                    slow_calculator=slow)
result = sim.run()
```

From a simulation config, the equivalent `slow_calculator` block is:

```yaml
slow_calculator:
  type: orca
  orca_path: /opt/orca/orca
  keywords: "PBE D3BJ def2-SVP"
  charge: 0
  multiplicity: 1
  num_threads: 8
```

See [RESPA integrator](../mts/respa.md) for how to choose `mts_ratio`.

## Configs

[`src/goal/configs/md/mts/simulations/mts_respa.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/mts/simulations/mts_respa.yaml) uses ORCA as its default `slow_calculator` (`orca_path: ???`, so you must set it). The template [`src/goal/configs/md/mts/base_calculators/orca_reference.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/mts/base_calculators/orca_reference.yaml) is the reference level of the online-learning configs:

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none

key: orca
orca_path: ??? # ABSOLUTE path to the orca executable
keywords: "EnGrad wB97M-V def2-TZVPD RIJCOSX TightSCF DEFGRID3"
charge: 0
multiplicity: 1
```

`hydra.utils.instantiate(cfg, orca_path="/opt/orca/orca")` builds it directly.

## Practical notes

- **Labels for GMD-style training:** to match a published dataset, reproduce its level of theory exactly: functional, basis, dispersion, grid and RI approximation. For example, GMD uses PBE/def2-SVP, so use `keywords="PBE def2-SVP"`, plus `D3BJ` if the reference includes it.
- **Comparing with plane-wave codes:** ORCA energies are all-electron. Do not compare absolute energies with CP2K-SIRIUS, QE or VASP; compare forces and relative energies. See [CP2K](cp2k.md#dispersion-dft-d3bj-dispersiond3bj) for a PBE-D3BJ setup whose D3 parameters match ORCA's.
- **Cost:** each MD step is a fresh ORCA run. ORCA reuses the previous orbitals (`orca.gbw`) as its initial guess when it finds them in the same directory, so keep `directory` fixed across a trajectory, and use a different `directory` for each concurrent calculator.
- **Charge and spin:** `MoleculeFactory.create("from_smiles", ...)` does not carry formal charges into `ase.Atoms`, so set `charge` and `multiplicity` explicitly for ions and radicals.
