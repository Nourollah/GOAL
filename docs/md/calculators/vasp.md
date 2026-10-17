# VASP calculator

The `vasp` backend runs VASP through `ase.calculators.vasp.Vasp`. The builder is [`VaspBuilder`](https://github.com/Nourollah/GOAL/blob/main/src/goal/md/core/calculator_factory.py), selected with `key: vasp`. It adds GOAL's `molecular`/`bulk` presets and MPI/OpenMP launch handling, and otherwise passes parameters to ASE's `Vasp` unchanged.

## Commercial software

VASP is **commercial** and is not bundled. GOAL degrades gracefully: if ASE's VASP support is missing, or no POTCAR path is configured, the builder raises `GOALCalculatorNotFound` with instructions rather than failing on import.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `$VASP_PP_PATH` | Root of the POTCAR library (contains `potpaw_PBE/` …). **Required**: set it or pass `pp_path`. |
| `$ASE_VASP_COMMAND` | Launch command, e.g. `mpirun -n 8 vasp_std`. |

Resolution order:

- **POTCARs:** `pp_path` → `$VASP_PP_PATH`. The resolved value is written back into `os.environ["VASP_PP_PATH"]` for the rest of the process.
- **Command:** `command` → `$ASE_VASP_COMMAND` → `mpirun -n {n_mpi} vasp_std` (only if `n_mpi` is set). If none of these is set, ASE's own default lookup applies. Use `command="… vasp_gam"` for Γ-only molecular runs to save memory and time.
- `omp_threads` exports `OMP_NUM_THREADS` into the current process environment.

## Configuration

[`src/goal/configs/md/calculators/vasp.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/calculators/vasp.yaml):

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none
key: vasp

command: null            # null -> $ASE_VASP_COMMAND
pp_path: null            # null -> $VASP_PP_PATH (POTCAR library root)
preset: molecular        # "molecular" (kpts [1,1,1], ismear 0, sigma 0.01) or "bulk"
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

`parameters` takes any `ase.calculators.vasp.Vasp` keyword (INCAR tags in lower case, plus `kpts`, `gamma`, `setups`, …). Calculations run in `directory="vasp_files"`.

The defaults are single-point settings driven by ASE: `nsw: 0` and `ibrion: -1` mean VASP does not move the ions, and ASE's dynamics does. `lwave`/`lcharg: false` keep the run directory small.

This file instantiates cleanly with `hydra.utils.instantiate`.

### Presets

| `preset` | `kpts` | `ismear` | `sigma` |
|----------|--------|----------|---------|
| `molecular` | `[1,1,1]` | `0` (Gaussian) | `0.01` |
| `bulk` | `[4,4,4]` | `1` (Methfessel–Paxton) | `0.2` |

The preset values are applied with `setdefault`, so anything in `parameters` wins.

> [!TIP]
> `ismear=1` (the `bulk` preset) is meant for metals. For semiconductors and insulators, override it with `parameters: {ismear: 0, sigma: 0.05}`, or use `ismear: -5` for accurate total energies with a dense k-mesh (not for forces in MD).

## Example

```python
import os
from ase.build import bulk
from goal.md import CalculatorFactory

os.environ["VASP_PP_PATH"] = "/data/vasp/potpaw"
atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = CalculatorFactory.create(
    "vasp",
    preset="bulk",
    n_mpi=8,
    parameters={"xc": "PBE", "encut": 450, "ismear": 0, "sigma": 0.05},
)
print(atoms.get_potential_energy())
```

> [!NOTE]
> Like ASE's other file-based calculators, each evaluation is a fresh VASP run (no WAVECAR reuse when `lwave: false`). For MD, consider `lwave: true` so each step restarts from the previous wavefunctions, at the cost of disk space.
