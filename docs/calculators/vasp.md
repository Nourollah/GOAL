# VASP calculator

Uses `ase.calculators.vasp.Vasp`. Builder:
[`VaspBuilder`](../../src/goal/md/core/calculator_factory.py), `key: vasp`.

## Commercial software

VASP is **commercial** and is not bundled. GOAL degrades gracefully: if ASE's
VASP support is missing, or no POTCAR path is configured, the builder raises
`GOALCalculatorNotFound` with instructions rather than crashing on import.

## Environment variables

| var | meaning |
|-----|---------|
| `$VASP_PP_PATH` | root of the POTCAR library (contains `potpaw_PBE/` …). Required — set it or pass `pp_path`. |
| `$ASE_VASP_COMMAND` | launch command, e.g. `mpirun -n 8 vasp_std`. |

`pp_path` (config) is written into `$VASP_PP_PATH` for the run;
`n_mpi` builds `mpirun -n N vasp_std` when `command`/`$ASE_VASP_COMMAND` is
unset.

## Configuration

`configs/md/calculator/vasp.yaml`:

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none
key: vasp

command: null            # null -> $ASE_VASP_COMMAND
pp_path: null            # null -> $VASP_PP_PATH
preset: molecular        # "molecular" or "bulk"
n_mpi: null
omp_threads: null

parameters:
  xc: PBE
  encut: 520             # eV
  ediff: 1.0e-6
  nsw: 0
  ibrion: -1
  lwave: false
  lcharg: false
```

### Presets

| preset | kpts | ismear | sigma |
|--------|------|--------|-------|
| `molecular` | `[1,1,1]` | `0` | `0.01` |
| `bulk` | `[4,4,4]` | `1` | `0.2` |

Anything in `parameters` overrides preset defaults.

## Example

```python
import os
from goal.md import CalculatorFactory
from ase.build import bulk

os.environ["VASP_PP_PATH"] = "/data/vasp/potpaw"
atoms = bulk("Si")
atoms.calc = CalculatorFactory.create("vasp", preset="bulk", n_mpi=8)
print(atoms.get_potential_energy())
```
