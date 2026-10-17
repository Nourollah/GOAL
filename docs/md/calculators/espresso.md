# Quantum ESPRESSO calculator

The `espresso` backend runs plane-wave pseudopotential DFT with Quantum ESPRESSO's `pw.x` through `ase.calculators.espresso.Espresso` and `EspressoProfile` (ASE ≥ 3.23). The builder is [`EspressoBuilder`](https://github.com/Nourollah/GOAL/blob/main/src/goal/md/core/calculator_factory.py), selected with `key: espresso`.

It is GOAL's ready-to-use open-source DFT backend. Quantum ESPRESSO 7.6 (`pw.x`, MPI-parallel through OpenMPI) ships in every `-full` pixi environment on x86-64 and aarch64, so the only thing you have to supply is pseudopotentials. Use it for DFT-quality forces and labels: training and validation data, reference MD, and the slow level of [multiple-time-step MD](../mts/overview.md). For cheap exploratory dynamics, use [tblite](tblite.md) instead.

## Installation

```bash
pixi install -e cuda-full                 # or cpu-full / rocm-full
pixi run -e cuda-full which pw.x          # .pixi/envs/cuda-full/bin/pw.x
```

The `-full` environments install conda-forge's `qe` package plus an explicit `elpa` pin, because that `qe` build links ELPA without declaring it. `mpirun` comes from the same OpenMPI. With pip, `pip install "goal-atomistic[md]"` gives you the Python side only. Install `pw.x` yourself (conda-forge `qe`, your cluster's module, or a source build) and put it on `PATH` or in `$ASE_ESPRESSO_COMMAND`.

## Pseudopotentials (required)

Quantum ESPRESSO needs one UPF pseudopotential file per element, and GOAL ships none. Two good sources:

- **SSSP** (Standard Solid-State Pseudopotentials), a verified set with recommended cutoffs per element. Download the *efficiency* or *precision* tarball from <https://www.materialscloud.org/discover/sssp/table/efficiency> and extract it.
- **PSlibrary** (<https://dalcorso.github.io/pslibrary/>). Individual files are also served from the QE pseudopotential table at <https://pseudopotentials.quantum-espresso.org/>, for example:

  ```bash
  mkdir -p ~/pseudo && cd ~/pseudo
  curl -LO https://pseudopotentials.quantum-espresso.org/upf_files/H.pbe-rrkjus_psl.1.0.0.UPF
  curl -LO https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-rrkjus_psl.1.0.0.UPF
  ```

Point `pseudo_dir` at the folder, or export `$ESPRESSO_PSEUDO`. Then list one filename per element in `pseudopotentials`. Use the same functional throughout: the default `input_data` takes the functional from the UPF files, so PBE pseudopotentials give PBE.

If `pseudopotentials` is empty, the builder raises `GOALCalculatorNotFound`. It does the same when you pass `elements` and one of those elements has no entry, and the message names the missing element(s). Passing `elements=sorted(set(atoms.get_chemical_symbols()))` makes a run fail at build time instead of mid-trajectory.

> [!TIP]
> SSSP recommends a wavefunction and density cutoff per element. Set `ecutwfc`/`ecutrho` to the **maximum** over the elements in your system. The builder default of `80/640` Ry is conservative for most first- and second-row chemistry. Ultrasoft and PAW pseudopotentials need `ecutrho` ≈ 8–12 × `ecutwfc`; norm-conserving ones need 4 ×.

## Launch command

The builder resolves the `pw.x` command in this order:

1. `command`: a full launch string, used as given. Examples: `"pw.x"`, `"mpirun -n 8 pw.x"`, `"srun pw.x -nk 2"`.
2. `n_mpi=N`: builds `mpirun -n N pw.x`.
3. `$ASE_ESPRESSO_COMMAND`.
4. `pw.x` found on `PATH`, which is the case in every `-full` environment.

In a `-full` environment you therefore need no command configuration at all. Pass `n_mpi` to run in parallel. `omp_threads` exports `OMP_NUM_THREADS` into the current process environment (it persists for the rest of the session). With MPI, `omp_threads=1` avoids oversubscribing the cores.

## Forces and stress

QE only prints forces and stress when asked, so the builder asks for them:

- `control.tprnfor = True` is always set, so every calculation returns forces for MD and optimisation.
- With `preset="bulk"`, `control.tstress = True` is also set, unless you set `tstress` yourself in `input_data`. Stress is then available for NPT and cell relaxation.

## Presets

| `preset` | Effect | Intended system |
|----------|--------|-----------------|
| `molecular` (default) | adds `system.assume_isolated = "mt"` (Martyna–Tuckerman); `kpts=None` → Γ-only routines | molecule in a vacuum box with `atoms.pbc = True` |
| `bulk` | `kpts` defaults to `[2, 2, 2]` if not given; `control.tstress = True` unless set | periodic solids and liquids |

`kpts=None` makes QE use its Γ-point-optimised code path, which roughly halves memory and CPU time compared with `kpts=(1, 1, 1)`. Martyna–Tuckerman needs the box to be at least about twice the extent of the molecule's charge density, so use generous vacuum (≥ 6 Å on each side).

## Parameters

| Argument | Default | Meaning |
|----------|---------|---------|
| `command` | `None` | Launch string (see [Launch command](#launch-command)) |
| `pseudo_dir` | `None` | Pseudopotential directory; `None` → `$ESPRESSO_PSEUDO` |
| `pseudopotentials` | required | `{element: UPF filename}` |
| `preset` | `"molecular"` | `"molecular"` or `"bulk"`; anything else raises `ValueError` |
| `kpts` | `None` | Monkhorst–Pack grid; `None` → Γ-only (`bulk` → `[2, 2, 2]`) |
| `input_data` | `None` | `pw.x` namelists, deep-merged over the defaults below |
| `n_mpi` | `None` | MPI ranks → `mpirun -n N pw.x` |
| `omp_threads` | `None` | Sets `OMP_NUM_THREADS` |
| `elements` | `None` | Element symbols to check against `pseudopotentials` |
| `directory` | `"espresso_files"` | Working directory for `espresso.pwi` / `espresso.pwo` |

The `input_data` defaults are `control.calculation="scf"`, `control.verbosity="low"`, `control.tprnfor=True`, `system.ecutwfc=80`, `system.ecutrho=640` and `electrons.conv_thr=1e-8`. Your `input_data` is deep-merged over them: setting `{"system": {"ecutwfc": 60}}` changes only `ecutwfc` and keeps everything else. Any `pw.x` namelist key can go here, for example `system.nspin`, `system.tot_charge`, `system.occupations`/`smearing`/`degauss` for metals, `system.input_dft` or `system.vdw_corr`, and `electrons.mixing_beta`.

## Worked example: water

```python
from ase.build import molecule
from goal.md import CalculatorFactory

atoms = molecule("H2O")
atoms.center(vacuum=6.0)
atoms.pbc = True

atoms.calc = CalculatorFactory.create(
    "espresso",
    pseudo_dir="/home/me/pseudo",
    pseudopotentials={"H": "H.pbe-rrkjus_psl.1.0.0.UPF",
                      "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF"},
    elements=["H", "O"],
    preset="molecular",
    n_mpi=2,                                             # mpirun -n 2 pw.x
    input_data={"system": {"ecutwfc": 30, "ecutrho": 240}},
)
print(atoms.get_potential_energy())   # -471.467 eV, about 12 s on 2 ranks
print(atoms.get_forces())
```

The `ecutwfc = 30` Ry cutoff here is a smoke test and is far from converged. The forces it gives on ASE's water geometry are around 1 eV/Å. For production, start from the SSSP-recommended cutoffs and converge forces to the accuracy you need.

### MD with QE forces

```python
from goal.md import Simulation

sim = Simulation(atoms, atoms.calc, steps=3, temperature_K=300.0, timestep_fs=0.5, friction=0.01,
                 log_interval=1, output_dir="outputs/md/water_qe")
result = sim.run()
```

Each MD step is one full SCF, so a picosecond of DFT MD is thousands of `pw.x` runs. For anything longer than a short reference trajectory, put QE at the slow level of [RESPA](../mts/overview.md), with a GOAL model or foundation model as the fast level.

## Configuration

[`src/goal/configs/md/calculators/espresso.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/calculators/espresso.yaml):

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none
key: espresso

command: null            # null -> $ASE_ESPRESSO_COMMAND, then pw.x on PATH
pseudo_dir: null         # null -> $ESPRESSO_PSEUDO
pseudopotentials: {}     # REQUIRED, e.g. {H: H.pbe-rrkjus.UPF, O: O.pbe-rrkjus.UPF}
preset: molecular        # "molecular" (assume_isolated=mt) or "bulk" (kpts default [2,2,2])
kpts: null               # null = gamma point only
n_mpi: null
omp_threads: null
# elements: [H, C, O]    # optional: verify a pseudopotential exists per element

input_data:
  control:
    calculation: scf
    verbosity: low
    tprnfor: true        # print forces (needed for MD / optimisation)
  system:
    ecutwfc: 80          # Ry
    ecutrho: 640         # Ry
  electrons:
    conv_thr: 1.0e-8
```

Instantiate it with Hydra. The preset is merged into `input_data` without losing the cutoffs:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load("src/goal/configs/md/calculators/espresso.yaml")
cfg.pseudo_dir = "/home/me/pseudo"
cfg.pseudopotentials = {"H": "H.pbe-rrkjus_psl.1.0.0.UPF", "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF"}
calc = instantiate(cfg)
# input_data: control {scf, low, tprnfor}, system {ecutwfc 80, ecutrho 640, assume_isolated mt}, ...
```

In a simulation config (`goal-simulate`, `MTSSimulation.from_config`), use a plain mapping with `type: espresso` and no `_target_`/`_convert_`:

```yaml
slow_calculator:
  type: espresso
  pseudo_dir: /home/me/pseudo
  pseudopotentials: {H: H.pbe-rrkjus_psl.1.0.0.UPF, C: C.pbe-n-rrkjus_psl.1.0.0.UPF, O: O.pbe-n-rrkjus_psl.1.0.0.UPF}
  n_mpi: 8
  input_data: {system: {ecutwfc: 50, ecutrho: 400}}
```

## Practical notes

- **Start-up cost per step.** ASE's file-based Espresso calculator starts a fresh `pw.x` for every evaluation, so each MD step pays the full start-up and initial-guess cost. For long trajectories, CP2K's persistent `cp2k_shell` is usually more efficient.
- **Keep `directory` fixed** across a run so files do not pile up in different places. Use a distinct `directory` for each concurrent calculator.
- **Charged and open-shell molecules** need `system.tot_charge` and `system.nspin = 2` (with `starting_magnetization` or `tot_magnetization`) in `input_data`.
- **Energy zero.** Plane-wave pseudopotential energies are not comparable with ORCA or xTB energies. Compare forces and relative energies only.
