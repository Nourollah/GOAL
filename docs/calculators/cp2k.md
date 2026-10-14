# CP2K calculator

GOAL drives CP2K through ASE's `cp2k_shell` mode (a persistent subprocess that
GOAL talks to over pipes). The builder lives in
[`goal.md.core.calculator_factory.CP2KBuilder`](../../src/goal/md/core/calculator_factory.py)
and is selected with `key: cp2k`.

## The `set_pos_file` fix

With MPI builds of CP2K, streaming a large structure to the shell through
**stdin** can stall (the classic "CP2K hangs on big systems" symptom). CP2K
`>= 2024.2` added `set_pos_file`, which sends atomic positions via a temporary
**file** instead of stdin. GOAL sets `set_pos_file=True` **by default**.

If you are on an older CP2K, set `set_pos_file: false` (and expect stalls on
large MPI runs).

## Requirements

- CP2K `>= 2024.2` (for `set_pos_file`)
- ASE `>= 3.23`
- A `cp2k_shell` binary on `PATH` (e.g. `cp2k_shell.psmp` for MPI+OpenMP). If
  you only have a plain `cp2k.psmp`, the builder symlinks it to
  `cp2k_shell.ssmp` in the working directory.

If CP2K cannot be imported/found you get a clear
`GOALCalculatorNotFound`.

## Configuration

`configs/md/calculator/cp2k.yaml`:

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none
key: cp2k

command: null            # null -> $ASE_CP2K_COMMAND, then "cp2k_shell.psmp"
preset: molecular        # "molecular" (MT poisson, no stress) or "bulk"
set_pos_file: true       # positions via temp file (CP2K >= 2024.2)
n_mpi: null              # MPI ranks; adds "mpirun -n N" prefix
omp_threads: null        # sets OMP_NUM_THREADS per rank
cutoff_ry: 400.0         # plane-wave cutoff in Rydberg (converted to eV for ASE)
xc: PBE
basis_set: DZVP-MOLOPT-SR-GTH
pseudo_potential: GTH-PBE
basis_set_file: BASIS_MOLOPT
potential_file: GTH_POTENTIALS
charge: 0
uks: false
max_scf: 50
```

### Presets

| preset | poisson_solver | stress_tensor | intended system |
|--------|----------------|---------------|-----------------|
| `molecular` | `MT` (Martyna–Tuckerman) | `false` | isolated molecule |
| `bulk` | periodic | `true` | periodic solid |

Any explicit key overrides the preset value.

## Parallelism

```yaml
command: cp2k_shell.psmp
n_mpi: 8
omp_threads: 4
```

builds the launch command `mpirun -n 8 cp2k_shell.psmp` and exports
`OMP_NUM_THREADS=4`.

## Example (library use)

```python
from goal.md import CalculatorFactory
from ase.build import molecule

atoms = molecule("H2O")
atoms.calc = CalculatorFactory.create(
    "cp2k", preset="molecular", n_mpi=4, cutoff_ry=400.0,
)
print(atoms.get_potential_energy())
```

## Pure plane-wave DFT (SIRIUS) — `method="sirius"`

The default `method="quickstep"` is CP2K's **GPW** scheme: the Kohn–Sham orbitals
live in a **Gaussian** basis (`basis_set`) with a plane-wave *density* grid
(`cutoff_ry`). If you want a genuinely **basis-set-free, pure plane-wave** DFT —
the wavefunctions themselves expanded in plane waves, like Quantum ESPRESSO /
VASP — use CP2K's **SIRIUS** backend:

```python
from goal.md import CalculatorFactory

calc = CalculatorFactory.create(
    "cp2k",
    method="sirius",            # METHOD SIRIUS + &PW_DFT (pure plane wave)
    preset="molecular",         # Γ-point, no stress
    command="cp2k_shell",       # or a podman wrapper (below)
    xc="PBE",                   # -> libxc XC_GGA_X_PBE + XC_GGA_C_PBE
    pseudo_potential="GTH-PBE",
    pw_cutoff=18.0,             # density  |G|max  [bohr^-1]  (~324 Ry)
    gk_cutoff=9.0,              # wavefunc |G+k|max [bohr^-1]  (~81 Ry)
    kpts=(1, 1, 1),             # Γ-point (molecule in a box)
)
```

What changes vs GPW:

* **No Gaussian basis** — `basis_set`/`basis_set_file` are forced to `None`, so no
  `BASIS_SET` line is emitted per `&KIND`; only the norm-conserving
  `pseudo_potential` (default `GTH-PBE`, from `GTH_POTENTIALS`) is used.
* The GPW `&MGRID CUTOFF` and `&SCF` are dropped; the plane-wave cutoffs and SCF
  loop live in `&PW_DFT` (`PW_CUTOFF`, `GK_CUTOFF`, `NUM_DFT_ITER`, mixer, …).
* Cutoffs are in **bohr⁻¹** (`|G|_max`). If `pw_cutoff`/`gk_cutoff` are left
  `None` they derive from `cutoff_ry`: `pw = sqrt(cutoff_ry)`, `gk = pw/2`.
* `xc="PBE"` is auto-mapped to the explicit libxc pair SIRIUS understands
  (`XC_GGA_X_PBE XC_GGA_C_PBE`); the native CP2K `&PBE` shortcut is *not*
  recognised by SIRIUS.

Requires a **SIRIUS-enabled** CP2K build (`cp2k --version` lists `sirius`). The
official `cp2k/cp2k` container qualifies. Config: `configs/md/calculator/cp2k_pw.yaml`.

### Dispersion (DFT-D3(BJ)) — `dispersion="d3bj"`

> ⚠️ **CP2K's `&VDW_POTENTIAL` is silently ignored under `METHOD SIRIUS`** —
> verified: the total energy is bitwise identical with and without the block,
> and CP2K reports zero warnings. Putting D3 "in the CP2K input" therefore gives
> plain PBE mislabelled as PBE-D3BJ.

Because D3(BJ) is a geometry-only pairwise term, the builder adds it as an
**independent additive** calculator (ASE `SumCalculator`) using Grimme's
`simple-dftd3` (`dftd3-python`) — physically exact and letting the parameters
match ORCA:

```python
calc = CalculatorFactory.create(
    "cp2k", method="sirius", xc="PBE",
    dispersion="d3bj",      # DFT-D3 with Becke-Johnson damping
    dispersion_atm=False,   # 2-body only (ORCA's default 'D3BJ'); True adds ABC
)
```

The 2-body parameters are the canonical per-functional values (`_D3BJ_PARAMS`,
e.g. PBE: `s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407`), identical to what
Grimme/ORCA use. Needs `dftd3-python` (`pixi add dftd3-python`); a clear
`GOALCalculatorNotFound` is raised if it is missing.

> **Comparing to Gaussian references:** absolute energies from a pseudopotential
> plane-wave run are *not* comparable to an all-electron Gaussian reference (e.g.
> the PBE/def2-SVP GMD labels) — different energy zeros. Compare **forces** and
> **relative** energies.

## Running CP2K from a podman container (no local install)

You don't need CP2K on the host — ASE can drive a **containerised** `cp2k_shell`.
Pull the image once:

```bash
podman pull docker.io/cp2k/cp2k:latest
```

Then install this wrapper as `$HOME/bin/cp2k_shell` (and set
`export ASE_CP2K_COMMAND="$HOME/bin/cp2k_shell"`, or put `$HOME/bin` on `PATH`):

```bash
#!/usr/bin/env bash
IMAGE="${CP2K_IMAGE:-docker.io/cp2k/cp2k:latest}"
OMP="${CP2K_OMP:-8}"
DATA_DIR="${CP2K_CONTAINER_DATA_DIR:-/opt/cp2k/share/cp2k/data}"
HOSTCWD="$(pwd -P)"
podman run --rm -i \
  --userns=keep-id \
  -v "$HOSTCWD:$HOSTCWD:z" -w "$HOSTCWD" \
  -e OMP_NUM_THREADS="$OMP" \
  -e CP2K_DATA_DIR="$DATA_DIR" \
  "$IMAGE" cp2k_shell "$@" \
  | stdbuf -oL tee -a "${CP2K_SHELL_LOG:-/dev/null}" \
  | grep --line-buffered -E '^[[:space:]]*([*]|CP2K Shell Version:|[-+0-9.])'
```

The non-obvious bits (each one is load-bearing):

| Piece | Why |
|-------|-----|
| named `cp2k_shell` | ASE asserts the launch command contains `cp2k_shell`. |
| `-i` | ASE streams the shell protocol over stdin. |
| `$(pwd -P)`, not `$PWD` | ASE launches the wrapper with `subprocess(shell=True)`; the `$PWD` env var can be stale after `os.chdir`. The mount must match where ASE writes `cp2k.inp/out/pos`. |
| `:z` on the mount | SELinux-enforcing hosts otherwise deny the container write access (*Permission denied*). |
| `-e CP2K_DATA_DIR=…` | the image leaves it unset, so `GTH_POTENTIALS` wouldn't resolve. |
| `grep` whitelist | the SIRIUS build prints a banner + `[info]/[fft]/…` to **stdout**, which would corrupt ASE's line protocol. Only real protocol lines (`* …`, the version line, numeric data) are passed through; set `CP2K_SHELL_LOG=/path` to capture the raw stream (incl. SIRIUS errors) for debugging. |

Threads: `CP2K_OMP` sets `OMP_NUM_THREADS` inside the container (single MPI rank,
OpenMP-parallel).

### End-to-end example: recompute a GMD trajectory

See `notebooks/gmd_cp2k_pw_recompute.ipynb` (verify one frame interactively) and
`dump/recompute_gmd_cp2k_pw.py` (batch the whole trajectory in the background,
with resume):

```bash
export ASE_CP2K_COMMAND="$HOME/bin/cp2k_shell" CP2K_OMP=16
nohup pixi run python dump/recompute_gmd_cp2k_pw.py \
    --traj data/GMD/FragmentDuplication/TestOOD/O4C4H6_PBE.traj \
    --pw-cutoff 18 --gk-cutoff 9 \
    > logs/recompute_O4C4H6.log 2>&1 &
```
