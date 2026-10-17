# ASE calculator

`GOALCalculator` wraps a trained GOAL model as a standard [ASE calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html). Anything ASE can drive then runs on your potential: single points, geometry optimisation, NEB, vibrational analysis, phonons and molecular dynamics.

```python
from goal.ml.utils.calculator import GOALCalculator   # also: from goal.ml.utils import GOALCalculator
```

> [!TIP]
> For production MD (thermostats, trajectory output, multiple time-step integration) use `goal.md`. It wraps this same class through `CalculatorFactory`. See [Trained GOAL models](../../md/calculators/goal-model.md).

## Loading a model

`GOALCalculator` accepts three checkpoint formats plus an already-built module. The format is detected from the path:

| What you pass | Format | Self-contained? | When to use |
| --- | --- | --- | --- |
| `checkpoint_path="model.simurgh"` | `.simurgh` archive | yes: weights, config, metadata and frozen source in one zip | sharing, deployment, long-term reproducibility |
| `checkpoint_path="…/checkpoints"` | checkpoint directory written by `GOALCheckpointManager` | yes: loads with the directory's `frozen_source/` | working with your own runs; choose the weights with `checkpoint=` |
| `checkpoint_path="…/last.ckpt"` | bare Lightning checkpoint (legacy) | **no**: rebuilt from the live source, with a warning | quick checks right after training, while the code is unchanged |
| `module=my_module, cutoff=…` | a `GOALModule` in memory | — | notebooks, freshly trained or modified models |

The two self-contained formats work like this: the model classes are imported from the frozen copy of the source that trained them (`frozen_source_imports`), so later refactors of `src/goal` do not break loading. How these files are produced is covered in [Checkpoints & archives](archives.md).

### From an archive

```python
calc = GOALCalculator(checkpoint_path="best_val_forces_forces_mae.simurgh", device="cuda")
```

The archive is unpacked into a temporary directory. That directory lives as long as the calculator does.

### From a checkpoint directory

```python
calc = GOALCalculator(
    checkpoint_path="logs/simurgh_gmd26/runs/<run>/checkpoints",
    checkpoint="best",        # "best" (default) | "last" | "epoch=N" | "<name>.ckpt"
)
```

- `"best"` takes the best entry of the top-k pool, using `checkpoint_state.json` and the pool's `min`/`max` mode. It falls back to parsing `best_*=<value>_epoch=*.ckpt` filenames, lower is better.
- `"last"` is `last.ckpt`.
- `"epoch=N"` is `interval_epoch=000N.ckpt`.
- Anything ending in `.ckpt` is used as a filename relative to the directory, or as an absolute path.

A directory without `config.yaml`, `metadata.json` and `frozen_source/` is rejected with `FileNotFoundError`.

### From a bare `.ckpt`

```python
calc = GOALCalculator(checkpoint_path="logs/…/checkpoints/last.ckpt")
```

This logs a warning: the model is rebuilt from the **current** source using the config stored in the checkpoint's hyper-parameters. It fails if a backbone signature has changed since training. Prefer the directory or archive form.

### From a pre-loaded module

```python
calc = GOALCalculator(module=my_goal_module, cutoff=5.0, device="cuda")
```

`cutoff` is required here because there is no saved config to read it from. It must equal the cutoff the model was trained with. Passing both `module` and `checkpoint_path`, or neither, raises `ValueError`.

## Constructor reference

| Argument | Default | Meaning |
| --- | --- | --- |
| `checkpoint_path` | `None` | `.simurgh` archive, checkpoint directory, or `.ckpt` file |
| `module` | `None` | an instantiated `GOALModule`, mutually exclusive with `checkpoint_path` |
| `cutoff` | `None` | neighbour-list cutoff in Å. When loading from a checkpoint it is read from `data.cutoff` (falling back to `model.backbone.cutoff`, then 5.0) |
| `device` | `"cpu"` | any torch device string: `"cuda"`, `"cuda:1"`, … |
| `dtype` | `None` | dtype of positions and cell. `None` uses the model's parameter dtype, so a float32 model gets float32 inputs. Set it only to override |
| `head` | `None` | head tag stored on the graph, for multi-head models |
| `checkpoint` | `"best"` | weights selector, only used for checkpoint directories |
| `use_ema` | `True` | load the EMA weights (`goal_ema` in the checkpoint) when present: the weights validation and `best_*` selection used. `False` loads the raw optimiser weights |
| `fragment_options` | `None` | fragment-label kwargs for `AtomicGraph.from_ase`. `None` reads them from the training config (`data.compute_fragment_index`, `data.fragment_*`); `{}` disables them |
| `fragment_on_failure` | `"fallback"` | overrides the rdkit schemes' failure mode at inference (bond perception can fail on geometries an optimiser or MD walks into). `None` keeps the training value |
| `neighbor_list_backend` | `None` | `"ase"`, `"matscipy"`, `"nvalchemiops"`, `"torch"` or `"auto"`; `None` uses the training config's `data.neighbor_list_backend`, else `"ase"` ([Neighbour lists](../../integrations/neighbor_list.md)) |
| `**kwargs` | — | forwarded to `ase.calculators.calculator.Calculator` (for example `label`) |

Public attributes: `calc.module` (the `GOALModule`, shared, in eval mode), `calc.cutoff`, `calc.config` (the training config), `calc.weights` (`"ema"`, `"raw"`, or `"module"` when a module was passed in), `calc.fragment_options`, and `calc.build_graph(atoms, energy=None, forces=None, stress=None)`, which builds exactly the graph the model sees at inference.

When loading from a checkpoint the model is built in the dtype its weights were saved in, so a run trained with `trainer.precision: 64` is evaluated in float64. (Before October 2026 the rebuilt model stayed float32 and the float64 weights were silently rounded.)

The loader is also available on its own: `goal.ml.utils.calculator.load_goal_checkpoint(path, checkpoint="best", device="cpu", use_ema=True)` returns the rebuilt module, cutoff, config and which weights were loaded. `goal-eval`, LoRA fine-tuning and the nvalchemi wrapper use it.

`implemented_properties` is `["energy", "forces", "stress"]`. Units follow ASE: eV, eV/Å and eV/Å³.

## Single point

```python
from ase.build import molecule

atoms = molecule("CH3COOH")
atoms.calc = calc

energy = atoms.get_potential_energy()   # eV (total energy, including atomic reference energies)
forces = atoms.get_forces()             # (N, 3) eV/Å
```

Each call builds an `AtomicGraph` from the `Atoms` (neighbour list with `cutoff`, honouring `atoms.pbc` and `atoms.cell`), runs the model under `torch.enable_grad()` so autograd force heads can differentiate the energy, and stores detached NumPy results. ASE caches the results until positions, numbers, cell or PBC change. Call `calc.reset()` to force a recomputation after you modify the model in place.

## Geometry optimisation

```python
from ase.build import molecule
from ase.optimize import BFGS

atoms = molecule("CH3CH2OH")
atoms.rattle(stdev=0.05, seed=0)
atoms.calc = calc

opt = BFGS(atoms, trajectory="opt.traj", logfile="opt.log")
opt.run(fmax=0.01)        # eV/Å
print(atoms.get_potential_energy())
```

> [!TIP]
> Tight `fmax` thresholds (≤ 0.01 eV/Å) amplify float32 noise in the forces. Checkpoints trained in float64 now load in float64 automatically. For a float32 model, cast it with `calc.module.double()` **and** pass `dtype=torch.float64` (or reload with it), because the input dtype must match the parameter dtype.

## Molecular dynamics with ASE

```python
from ase import units
from ase.build import molecule
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

atoms = molecule("CH3COOH")
atoms.calc = calc

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=300, friction=0.01 / units.fs)
traj = Trajectory("md.traj", "w", atoms)
dyn.attach(traj.write, interval=10)
dyn.attach(lambda: print(f"{dyn.get_number_of_steps():6d}  Epot={atoms.get_potential_energy():.4f} eV"), interval=100)
dyn.run(2000)             # 1 ps
```

Each step is a single-structure forward pass, so throughput is limited by Python overhead and the neighbour list. For anything beyond short tests, use `goal.md` ([Dynamics](../../md/guide/dynamics.md)).

## Periodic systems and stress

PBC is handled automatically: `AtomicGraph.from_ase` copies `atoms.cell` and `atoms.pbc` and builds a periodic neighbour list (ASE backend by default).

```python
from ase.build import bulk

atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
atoms.calc = calc
atoms.get_forces()
```

`atoms.get_stress()` returns a value **only if the model's output dict contains `"stress"`**. In GOAL that means a model built with the `stress` head (`StressHead`). The SIMURGH energy/force heads do not predict stress. For them ASE raises `PropertyNotImplementedError` and variable-cell optimisation or NPT is not available.

> [!WARNING]
> `StressHead` returns stress with shape `(B, 3, 3)`. The calculator only converts `(3, 3)` and `(6,)` correctly. A `(1, 3, 3)` tensor falls into a fallback that takes the first six flattened entries, which is **not** the Voigt order. Until this is fixed, check stresses against a finite-difference reference before using them.

## Limitations

### Models that need fragment labels

Models with fragment-level components (SIMURGH ARACE with `fragment_interaction`, NF_SIMURGH, and SIMURGH Simplified's fragment line) need `fragment_index` on every graph. The calculator reads `data.compute_fragment_index` and the `data.fragment_*` options from the checkpoint's own training config and applies them in `build_graph`, so these checkpoints work out of the box:

```python
calc = GOALCalculator(checkpoint_path="best_val_forces_mae.simurgh")
calc.fragment_options   # {'compute_fragment_index': True, 'fragment_scheme': 'custom', ..., 'fragment_on_failure': 'fallback'}
```

The settings come from the checkpoint, so build the calculator **from that checkpoint** (or pass a `GOALModule` whose `config` carries them). A freshly constructed model has no training config and therefore no fragment settings; pass `fragment_options=` yourself in that case.

`fragment_on_failure` defaults to `"fallback"` at inference even if training used `"raise"`: rdkit's bond perception can fail on distorted geometries an MD run or optimiser reaches ("unable to determine valence of atom N"), and aborting there is rarely what you want. The fallback uses connected-component labels for that frame; with single-molecule GMD frames that means one fragment, i.e. the fragment channel is silent for that frame. Pass `fragment_on_failure=None` to keep the training value.

Before October 2026 the calculator ignored these options and such models failed with `ValueError: ... the graphs carry no 'fragment_index'`.

### `atomic_energies.mode: dataset`

This affects checkpoints written before October 2026 only: `goal-train` now writes every dataset-derived backbone argument (`elements`, `atomic_energies.values`, `scale`, `avg_num_neighbors`) back into the config it saves, so newer checkpoints rebuild exactly. Older ones may fail to build:

```text
ValueError: atomic_energies.mode='dataset' requires 'values' to be a {Z: e_Z} dict
```

The fitted values are in the weights, as the `backbone.atomic_energies` buffer. Copy them back into the embedded config and save a patched checkpoint **inside the same directory**, so the frozen source is still used:

```python
from pathlib import Path

import torch
from omegaconf import open_dict

from goal.ml.training.archive import resolve_checkpoint
from goal.ml.utils.calculator import GOALCalculator

ckpt_dir = Path("logs/<task>/runs/<run>/checkpoints")
src = resolve_checkpoint(ckpt_dir, "best")
raw = torch.load(src, map_location="cpu", weights_only=False)

table = raw["state_dict"]["backbone.atomic_energies"]          # indexed by atomic number
values = {int(z): float(table[z]) for z in torch.nonzero(table).flatten()}
with open_dict(raw["hyper_parameters"]["config"]):
    raw["hyper_parameters"]["config"].model.backbone.atomic_energies["values"] = values

patched = ckpt_dir / f"{src.stem}_patched.ckpt"
torch.save(raw, patched)
calc = GOALCalculator(checkpoint_path=ckpt_dir, checkpoint=patched.name)
```

### EMA weights

For runs with `training.ema.enabled: true` the checkpoint stores the EMA shadow under `goal_ema`, and validation, `best_*` selection and `goal-eval` all use those averaged weights. The calculator loads them too (`use_ema=True`, the default) and reports which weights it used:

```python
calc = GOALCalculator(checkpoint_path="logs/<task>/runs/<run>/checkpoints", checkpoint="best")
calc.weights      # "ema" — or "raw" for a checkpoint without a usable EMA shadow
GOALCalculator(checkpoint_path=..., use_ema=False).weights   # "raw": the optimiser weights
```

A shadow that does not line up with the rebuilt model (an older checkpoint, a changed architecture) is ignored with a warning, and the raw weights are used. Before October 2026 the calculator always loaded the raw weights, so its predictions differed from the validation metrics the checkpoint was selected on.

### Throughput

`GOALCalculator` evaluates one `Atoms` object per call. To score many structures at once, build a batch with `torch_geometric.data.Batch.from_data_list([calc.build_graph(a) for a in frames])` and call `calc.module` directly, or use the Lightning test loop shown in [Evaluation](evaluation.md). For MD or relaxation of many structures, see [Batched simulation](../../integrations/batched_simulation.md); on NVIDIA GPUs it runs all of them in one batch.

## Using it through goal.md

`goal.md`'s `CalculatorFactory` builds a `GOALCalculator` for the `goal_model` calculator and adds MD drivers, mixed QM/ML set-ups and the MTS integrators on top. Configuration, examples and caveats are in [Trained GOAL models](../../md/calculators/goal-model.md). The factory forwards `select` (the `checkpoint` selector), `use_ema`, `fragment_on_failure` and `neighbor_list_backend`, and leaves the dtype to the checkpoint.
