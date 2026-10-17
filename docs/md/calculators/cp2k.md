# CP2K calculator

GOAL drives CP2K through ASE's `cp2k_shell` mode: a persistent CP2K subprocess that ASE talks to over pipes, so the SCF setup is reused between MD steps. One builder covers two different DFT discretisations. The default `method="quickstep"` gives Gaussian-and-plane-waves (GPW). `method="sirius"` gives pure plane-wave pseudopotential DFT. The builder is [`CP2KBuilder`](https://github.com/Nourollah/GOAL/blob/main/src/goal/md/core/calculator_factory.py), selected with `key: cp2k`.

## Requirements

- CP2K **≥ 2024.2** (for `set_pos_file`), either native or in the official container (see [podman](#running-cp2k-from-a-podman-container-no-local-install))
- ASE **≥ 3.23**
- A `cp2k_shell` entry point, e.g. `cp2k_shell.psmp` for MPI+OpenMP. If you only have a plain `cp2k.psmp`, the builder symlinks it to `cp2k_shell.ssmp` **in the current working directory**, because ASE requires the shell entry point.
- For `method="sirius"`: a SIRIUS-enabled build (`cp2k --version` lists `sirius`). The official `cp2k/cp2k` image qualifies.
- For `dispersion="d3bj"`: `dftd3-python` (Grimme's `simple-dftd3`). It is already a dependency of the pixi environment.

If ASE's CP2K support cannot be imported, or the binary cannot be launched, you get a `GOALCalculatorNotFound` that names the command it tried.

## The `set_pos_file` fix

With MPI builds of CP2K, streaming a large structure to the shell through **stdin** can stall: the classic "CP2K hangs on big systems" symptom. CP2K ≥ 2024.2 added `set_pos_file`, which sends atomic positions through a temporary **file** instead of stdin. GOAL sets `set_pos_file=True` **by default**.

On an older CP2K, set `set_pos_file: false` and expect stalls on large MPI runs.

## GPW (default): `method="quickstep"`

[`src/goal/configs/md/calculators/cp2k.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/calculators/cp2k.yaml):

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none
key: cp2k

command: null            # null -> $ASE_CP2K_COMMAND, then "cp2k_shell.psmp"
preset: molecular        # "molecular" (MT poisson, no stress) or "bulk"
set_pos_file: true       # send positions via temp file (CP2K >= 2024.2)
n_mpi: null              # MPI ranks; null = no mpirun prefix
omp_threads: null        # OMP_NUM_THREADS per rank

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

In GPW the Kohn–Sham orbitals live in a **Gaussian** basis (`basis_set`). `cutoff_ry` is the `&MGRID` cutoff of the auxiliary plane-wave *density* grid, and the builder converts it to eV for ASE.

### Presets

| `preset` | `poisson_solver` | `stress_tensor` | Intended system |
|----------|------------------|-----------------|-----------------|
| `molecular` | `MT` (Martyna–Tuckerman) | `false` | isolated molecule in a box |
| `bulk` | `PERIODIC` | `true` | periodic solid / liquid |

An explicit `poisson_solver=` or `stress_tensor=` overrides the preset.

> [!TIP]
> The MT Poisson solver needs the box to be roughly twice the extent of the electron density in each direction. If you see a large energy change when you enlarge the box, it is too small.

### Parallelism

```yaml
command: cp2k_shell.psmp
n_mpi: 8
omp_threads: 4
```

builds the launch command `mpirun -n 8 cp2k_shell.psmp` and exports `OMP_NUM_THREADS=4` into the Python process environment. If `command` already contains `mpirun` or `srun`, no prefix is added, so you can pass a full `srun …` string on clusters.

### Example (library use)

```python
from ase.build import molecule
from goal.md import CalculatorFactory

atoms = molecule("H2O")
atoms.center(vacuum=6.0)
atoms.calc = CalculatorFactory.create("cp2k", preset="molecular", n_mpi=4, cutoff_ry=400.0)
print(atoms.get_potential_energy(), atoms.get_forces())
```

## Pure plane-wave DFT (SIRIUS): `method="sirius"`

The default GPW still uses a Gaussian basis. If you want **basis-set-free, pure plane-wave** DFT, with the wavefunctions themselves expanded in plane waves as in Quantum ESPRESSO or VASP, use CP2K's **SIRIUS** backend. Accepted spellings: `"sirius"`, `"pw"`, `"plane_wave"`, `"planewave"`. `"quickstep"`/`"gpw"` select GPW, and anything else raises `ValueError`.

```python
from goal.md import CalculatorFactory

calc = CalculatorFactory.create(
    "cp2k",
    method="sirius",            # METHOD SIRIUS + &PW_DFT (pure plane wave)
    preset="molecular",         # Γ-point, no stress
    command="cp2k_shell",       # or the podman wrapper (below)
    xc="PBE",                   # -> libxc XC_GGA_X_PBE + XC_GGA_C_PBE
    pseudo_potential="GTH-PBE",
    pw_cutoff=18.0,             # density  |G|max   [bohr^-1]  (~324 Ry)
    gk_cutoff=9.0,              # wavefunc |G+k|max [bohr^-1]  (~81 Ry)
    kpts=(1, 1, 1),             # Γ-point (molecule in a box)
)
```

Config: [`src/goal/configs/md/calculators/cp2k_pw.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/calculators/cp2k_pw.yaml). It has the same `key: cp2k`, plus `method: sirius`, `pw_cutoff: 18.0`, `gk_cutoff: 9.0`, `kpts: [1, 1, 1]`, `smearing_width: 0.001`, and `num_dft_iter: 100`.

What changes compared with GPW:

- **No Gaussian basis.** `basis_set`/`basis_set_file` are forced to `None`, so no `BASIS_SET` line is emitted per `&KIND`. Only the norm-conserving `pseudo_potential` (default `GTH-PBE`, from `GTH_POTENTIALS`) is used.
- The GPW `&MGRID CUTOFF`, `&SCF` (`max_scf`), and Poisson solver are dropped. The plane-wave cutoffs and the SCF loop live in an auto-generated `&PW_DFT` block (Davidson solver, Broyden2 mixer, `USE_SYMMETRY FALSE`, `VERBOSITY 0`).
- **Cutoffs are in bohr⁻¹** (`|G|_max`). If `pw_cutoff`/`gk_cutoff` are `None` they derive from `cutoff_ry`: `pw_cutoff = sqrt(cutoff_ry)` and `gk_cutoff = pw_cutoff / 2` (so $E_\text{wfc} = E_\rho/4$). In Rydberg, $E_\text{cut} = |G|_\text{max}^2$, so 18 bohr⁻¹ ≈ 324 Ry for the density and 9 bohr⁻¹ ≈ 81 Ry for the wavefunctions. That is well converged for H/C/O GTH-PBE. `12 / 6` is a fast smoke test.
- **XC through libxc.** SIRIUS does not recognise CP2K's `&PBE` shortcut. The builder maps friendly names to libxc pairs: `PBE`, `PBESOL`, `BLYP`, `BP`, `LDA`/`PADE`/`PZ`. Names that already contain `XC_` are passed through unchanged.
- **Spin.** `uks=True` sets `NUM_MAG_DIMS 1` inside `&PW_DFT`, rather than the `&DFT UKS` flag.
- **Presets.** `molecular` → `stress_tensor=False`. `bulk` → `stress_tensor=True`. The preset does **not** change `kpts`, so set a Monkhorst–Pack grid yourself for solids.

Extra SIRIUS knobs, which are ignored for GPW:

| Argument | Default | `&PW_DFT` keyword |
|----------|---------|-------------------|
| `kpts` | `(1, 1, 1)` | `NGRIDK` |
| `smearing_width` | `0.001` (Ha) | `SMEARING_WIDTH` |
| `num_dft_iter` | `100` | `NUM_DFT_ITER` |
| `scf_energy_tol` | `1e-6` | `ENERGY_TOL` |
| `scf_density_tol` | `1e-6` | `DENSITY_TOL` |
| `solver_energy_tol` | `1e-4` | Davidson `ENERGY_TOLERANCE` (tighten for accurate forces) |
| `mixer_beta` | `0.5` | Broyden2 `BETA` |
| `processing_unit` | `"cpu"` | `PROCESSING_UNIT` (`"gpu"` for GPU builds) |

Passing `inp=` (raw CP2K input string) or `input_file=` (path) **replaces** the auto-generated `&PW_DFT` block. In GPW mode, ASE merges it with the structured keywords instead.

### Dispersion (DFT-D3(BJ)): `dispersion="d3bj"`

> [!WARNING]
> **CP2K's `&VDW_POTENTIAL` is silently ignored under `METHOD SIRIUS`.** The total energy is bitwise identical with and without the block, and CP2K prints no warning. Putting D3 "in the CP2K input" therefore gives plain PBE mislabelled as PBE-D3BJ.

D3(BJ) depends only on geometry, so the builder adds it as an **independent additive** calculator (ASE `SumCalculator([cp2k, DFTD3])`). This is exact, and it works for both GPW and SIRIUS:

```python
calc = CalculatorFactory.create(
    "cp2k", method="sirius", xc="PBE",
    dispersion="d3bj",      # DFT-D3 with Becke-Johnson damping ("d3" is accepted as an alias)
    dispersion_atm=False,   # 2-body only (ORCA's default 'D3BJ'); True adds the ATM 3-body term
)
```

For `PBE`, `PBE0`, `BLYP`, `B3LYP`, and `TPSS`, the builder uses explicit canonical 2-body parameters (Grimme et al., *J. Comput. Chem.* **32**, 1456 (2011)). For PBE these are `s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407`, the same values ORCA uses, so a CP2K PBE-D3BJ number is directly comparable to an ORCA `PBE D3BJ` reference. Other functionals fall back to `simple-dftd3`'s built-in method lookup. If `dftd3-python` is missing, you get a `GOALCalculatorNotFound`.

> [!IMPORTANT]
> **Comparing to Gaussian references.** Absolute energies from a pseudopotential plane-wave run are *not* comparable to an all-electron Gaussian reference (for example, the PBE/def2-SVP GMD labels), because the energy zeros differ by thousands of eV. Compare **forces** and **relative** energies.

> [!WARNING]
> **Varying cells and plane-wave energies.** At a fixed cutoff, the number of plane waves depends on the cell. If every frame of a trajectory carries a slightly different box, which is common in GMD-style datasets, the recomputed *energy differences* pick up a spurious basis-set-variation term. One example was a ~6 eV spread where the reference varied by < 1 eV. **Forces are unaffected.** For comparable relative energies, put every frame in one common box. That is also faster, because SIRIUS can reuse one FFT grid.

## Full argument reference

| Argument | Default | Notes |
|----------|---------|-------|
| `command` | `None` | `None` → `$ASE_CP2K_COMMAND` → `"cp2k_shell.psmp"` |
| `preset` | `"molecular"` | `"molecular"` or `"bulk"` |
| `method` | `"quickstep"` | `"quickstep"`/`"gpw"` or `"sirius"`/`"pw"`/`"plane_wave"` |
| `n_mpi`, `omp_threads` | `None` | see [Parallelism](#parallelism) |
| `set_pos_file` | `True` | positions via a temp file |
| `cutoff_ry` | `400.0` | GPW `&MGRID` cutoff; seeds `pw_cutoff` for SIRIUS |
| `xc` | `"PBE"` | |
| `basis_set`, `basis_set_file` | `DZVP-MOLOPT-SR-GTH`, `BASIS_MOLOPT` | GPW only |
| `pseudo_potential`, `potential_file` | `GTH-PBE`, `GTH_POTENTIALS` | |
| `stress_tensor`, `poisson_solver` | `None` | `None` = preset value |
| `charge`, `uks` | `0`, `False` | |
| `max_scf` | `50` | GPW only |
| `inp`, `input_file` | `None` | raw CP2K input |
| `print_level` | `"LOW"` | |
| `directory` | `None` | ASE working directory |
| `pw_cutoff`, `gk_cutoff`, `kpts`, `smearing_width`, `num_dft_iter`, `scf_energy_tol`, `scf_density_tol`, `solver_energy_tol`, `mixer_beta`, `processing_unit` | see the SIRIUS table | SIRIUS only |
| `dispersion`, `dispersion_atm` | `None`, `False` | additive D3(BJ) |
| `**overrides` | | forwarded verbatim to `ase.calculators.cp2k.CP2K`, overriding anything above |

## Running CP2K from a podman container (no local install)

You don't need CP2K on the host, because ASE can drive a **containerised** `cp2k_shell`. Pull the image once:

```bash
podman pull docker.io/cp2k/cp2k:latest
```

Then install this wrapper as `$HOME/bin/cp2k_shell`, make it executable, and run `export ASE_CP2K_COMMAND="$HOME/bin/cp2k_shell"` (or put `$HOME/bin` on `PATH`):

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

Each non-obvious piece is load-bearing:

| Piece | Why |
|-------|-----|
| named `cp2k_shell` | ASE asserts that the launch command contains `cp2k_shell`, and the builder only skips its symlink shim for such names. |
| `-i` | ASE streams the shell protocol over stdin. |
| `$(pwd -P)`, not `$PWD` | ASE launches the wrapper with `subprocess(shell=True)`, and the `$PWD` env var can be stale after `os.chdir`. The mount must match where ASE writes `cp2k.inp/out/pos`. |
| `:z` on the mount | SELinux-enforcing hosts otherwise deny the container write access (*Permission denied*). |
| `--userns=keep-id` | Files written in the bind mount stay owned by you. |
| `-e CP2K_DATA_DIR=…` | The image leaves it unset, so `GTH_POTENTIALS` / `BASIS_MOLOPT` wouldn't resolve. |
| `grep` whitelist | The SIRIUS build prints a banner and `[info]/[fft]/…` diagnostics to **stdout**, which would corrupt ASE's line protocol. Only real protocol lines (`* …`, the version line, numeric data) pass through. Set `CP2K_SHELL_LOG=/path` to capture the raw stream, including SIRIUS errors, for debugging. |

Threads: `CP2K_OMP` sets `OMP_NUM_THREADS` inside the container (one MPI rank, OpenMP-parallel). The builder's `omp_threads` does **not** reach the container, because the wrapper overrides it.

A quick handshake test before any DFT:

```bash
printf 'VERSION\nEXIT\n' | "$HOME/bin/cp2k_shell"     # must print "CP2K Shell Version: ..."
```

> [!TIP]
> ASE writes `cp2k.inp`, `cp2k.out`, and the position file into the **current working directory**, which is what the wrapper bind-mounts. `chdir` into a scratch directory under your project before building the calculator, and call `calc.close()` when done to shut the shell down cleanly.

## End-to-end example: recompute a GMD trajectory at PW-PBE-D3(BJ)

[`notebooks/gmd_cp2k_pw_recompute.ipynb`](https://github.com/Nourollah/GOAL/blob/main/notebooks/gmd_cp2k_pw_recompute.ipynb) creates and checks the wrapper, recomputes one frame of a GMD *FragmentDuplication* OOD trajectory with `method="sirius"` and `dispersion="d3bj"`, prints the generated input (`METHOD SIRIUS`, `&PW_DFT`, `&KIND` with only a `POTENTIAL`), and plots force parity against the reference. The core of it:

```python
import os
from pathlib import Path
from ase.io import read
from goal.md.core.calculator_factory import CalculatorFactory

os.environ["ASE_CP2K_COMMAND"] = str(Path.home() / "bin" / "cp2k_shell")
os.environ["CP2K_OMP"] = "16"

work = Path("notebooks/cp2k_pw_work"); work.mkdir(parents=True, exist_ok=True)
frames = read("data/GMD/FragmentDuplication/TestOOD/O4C4H6_PBE.traj", index=":")
os.chdir(work)

calc = CalculatorFactory.create(
    "cp2k", method="sirius", preset="molecular",
    command=os.environ["ASE_CP2K_COMMAND"],
    xc="PBE", pseudo_potential="GTH-PBE",
    pw_cutoff=18.0, gk_cutoff=9.0, smearing_width=0.001,
    dispersion="d3bj",
)
atoms = frames[0].copy(); atoms.calc = calc
e, f = atoms.get_potential_energy(), atoms.get_forces()
calc.close()
```

Expect about 1–2.5 min per frame at `pw/gk = 18/9` for a ~14-atom molecule in a ~15 Å box. Against the all-electron PBE/def2-SVP labels, the force-direction cosine is ≈ 0.99 at 18/9. At 12/6 it drops to ~0.91 and force magnitudes come out about 2× too large, which is plane-wave under-convergence rather than a pipeline problem.

> [!NOTE]
> The notebook hands the full trajectory off to a batch script, `dump/recompute_gmd_cp2k_pw.py`, which offers resume, `--stride`, `--stop`, and `--fixed-cell`. That script lives in the author's local `dump/` scratch directory and is **not tracked in the repository**. Outside that checkout, loop over frames yourself with the snippet above and write results incrementally (e.g. with `ase.io.write(..., append=True)`).
