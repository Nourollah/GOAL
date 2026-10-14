# Quantum ESPRESSO calculator

Uses `ase.calculators.espresso.Espresso` with `EspressoProfile`
(requires ASE `>= 3.23`). Builder:
[`EspressoBuilder`](../../src/goal/md/core/calculator_factory.py), `key: espresso`.

## Pseudopotentials (required)

Quantum ESPRESSO needs a UPF pseudopotential per element. The recommended set
is the **SSSP** library (Standard Solid-State Pseudopotentials):

1. Download an SSSP efficiency/precision tarball from
   <https://www.materialscloud.org/discover/sssp/table/efficiency>.
2. Extract it and point `pseudo_dir` (or `$ESPRESSO_PSEUDO`) at the folder.
3. List one UPF filename per element in `pseudopotentials`.

If `pseudopotentials` is empty — or, when `elements` is provided, missing any
element — the builder raises `GOALCalculatorNotFound` naming the missing
element(s).

## Configuration

`configs/md/calculator/espresso.yaml`:

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none
key: espresso

command: null            # null -> $ASE_ESPRESSO_COMMAND
pseudo_dir: null         # null -> $ESPRESSO_PSEUDO
pseudopotentials:        # REQUIRED
  H: H.pbe-rrkjus_psl.1.0.0.UPF
  O: O.pbe-n-rrkjus_psl.1.0.0.UPF
preset: molecular        # "molecular" (assume_isolated=mt) or "bulk" (kpts [2,2,2])
kpts: null               # null = gamma only
n_mpi: null
omp_threads: null
# elements: [H, O]       # optional: verify a pseudopotential exists per element

input_data:
  control: {calculation: scf, verbosity: low}
  system: {ecutwfc: 80, ecutrho: 640}   # Ry
  electrons: {conv_thr: 1.0e-8}
```

### Presets

- `molecular` → adds `system.assume_isolated: "mt"` (Martyna–Tuckerman),
  gamma-point only.
- `bulk` → defaults `kpts` to `[2, 2, 2]` when not set.

## Parallelism

`n_mpi: 4` builds the command `mpirun -n 4 pw.x` (when `command` is not set);
`omp_threads` sets `OMP_NUM_THREADS`.

## Example

```python
from goal.md import CalculatorFactory
from ase.build import molecule

atoms = molecule("H2O")
atoms.calc = CalculatorFactory.create(
    "espresso",
    pseudo_dir="/data/SSSP_efficiency",
    pseudopotentials={"H": "H.pbe-rrkjus_psl.1.0.0.UPF",
                      "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF"},
    preset="molecular",
)
print(atoms.get_potential_energy())
```
