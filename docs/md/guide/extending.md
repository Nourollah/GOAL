# Extending goal.md

`goal.md` has three registries: `CalculatorFactory`, `DynamicsFactory` and `MoleculeFactory`. Each maps a string key to a builder object. Adding a key makes your calculator, integrator or structure source available everywhere a key is accepted: `Factory.create(...)` calls, `Simulation.from_config`, the MTS configs and the CLI (with one caveat, covered [below](#making-your-keys-available-to-configs-and-the-cli)).

## How the registries work

The three registries share one pattern (see `goal.md.core.calculator_factory`, `md_factory` and `molecule_factory`):

| Registry | Abstract builder | Registration decorator |
| --- | --- | --- |
| `CalculatorFactory` | `CalculatorBuilder` | `register_calculator(key)` |
| `DynamicsFactory` | `DynamicsBuilder` | `register_dynamics(key)` |
| `MoleculeFactory` | `MoleculeBuilder` | `register_molecule_set(key)` |

- A builder is a subclass with one abstract method, `build(self, *args, **kwargs)`. `Factory.create(key, *args, **kwargs)` forwards its arguments to `build` unchanged, so **the `build` signature is your public API**. Config keys become keyword arguments.
- The decorator **instantiates the class once, with no arguments**, and stores that instance under `key` in a class-level dict (`Factory._builders`). Builders should therefore be stateless. Put the expensive work, such as loading a model, inside `build`.
- Registration happens when the module containing the decorated class is **imported**.
- Registering an existing key **silently replaces** the previous builder. You can use this to override a built-in deliberately, but pick unique names (for example a project prefix) to avoid doing it by accident.
- Each factory also has a plain `Factory.register(key, builder_instance)` classmethod if you prefer not to use decorators.

To see what is currently registered:

```python
from goal.md import CalculatorFactory, DynamicsFactory, MoleculeFactory
for f in (CalculatorFactory, DynamicsFactory, MoleculeFactory):
    print(f.__name__, sorted(f._builders))
```

## Custom calculators

Any object that implements the ASE `Calculator` interface can be returned. The example below is a complete plugin module. Save it as `my_md_plugins.py` somewhere on your `PYTHONPATH`; the following sections add to the same file.

```python
"""Project-local goal.md extensions. Import this module before using the keys."""
from __future__ import annotations

import typing

import ase
from ase import units
from ase.calculators.calculator import Calculator

from goal.md.calculators import GOALCalculatorNotFound
from goal.md.core.calculator_factory import CalculatorBuilder, register_calculator
from goal.md.core.md_factory import DynamicsBuilder, register_dynamics
from goal.md.core.molecule_factory import MoleculeBuilder, register_molecule_set


@register_calculator("emt")
class EMTBuilder(CalculatorBuilder):
    """ASE's effective-medium toy potential (Al, Cu, Ag, Au, Ni, Pd, Pt; H, C, N, O)."""

    def build(self, **kwargs: typing.Any) -> Calculator:
        from ase.calculators.emt import EMT
        return EMT(**kwargs)


@register_calculator("dftd3_only")
class DFTD3Builder(CalculatorBuilder):
    """Pure D3(BJ) dispersion via simple-dftd3, as an example of an optional backend."""

    def build(self, method: str = "PBE", damping: str = "d3bj") -> Calculator:
        try:
            from dftd3.ase import DFTD3
        except ImportError as exc:
            raise GOALCalculatorNotFound(
                "simple-dftd3 is not installed: conda install -c conda-forge dftd3-python"
            ) from exc
        return DFTD3(method=method, damping=damping)
```

The built-in builders follow these conventions, and yours should too:

- **Import optional packages inside `build`** (or in a guarded `try/except` at module level). Importing the plugin should never fail just because a backend is missing.
- **Raise `goal.md.calculators.GOALCalculatorNotFound`** when an external program, its Python bindings or required data (pseudopotentials, basis files, licences) is missing. Say exactly what is missing and how to provide it. Raise `UnsupportedOperationError` for something that is recognised but not implemented. Both exceptions live in the dependency-free `goal.md.calculators.base`.
- **Keep arguments config-friendly.** Use scalars, strings, lists and dicts, with explicit defaults. A `torch.dtype` or a callable cannot be written in YAML.
- **Return a fresh calculator on every call.** The MTS and comparison workflows create several calculators side by side.

For production QM backends, the existing builders (`orca`, `cp2k`, `espresso`, `vasp`) are good templates: they handle executable discovery, MPI/OpenMP launch strings and molecular vs. bulk presets. See [Calculators](../calculators/overview.md).

## Custom dynamics

A dynamics builder receives the `Atoms` and returns any object with ASE's dynamics interface: `.run(steps)`, `.attach(fn, interval)` and an `.atoms` attribute. That is what `setup_observers` and `run_dynamics` rely on.

```python
@register_dynamics("bussi")
class BussiBuilder(DynamicsBuilder):
    """Stochastic velocity rescaling NVT. Takes fs, like Simulation does."""

    def build(self, atoms: ase.Atoms, timestep_fs: float = 0.5,
              temperature_K: float = 300.0, taut_fs: float = 100.0, **kwargs: typing.Any):
        from ase.md.bussi import Bussi
        return Bussi(atoms, timestep=timestep_fs * units.fs, temperature_K=temperature_K,
                     taut=taut_fs * units.fs, **kwargs)


@register_dynamics("nve")
class VelocityVerletBuilder(DynamicsBuilder):
    def build(self, atoms: ase.Atoms, timestep_fs: float = 0.5, **kwargs: typing.Any):
        from ase.md.verlet import VelocityVerlet
        return VelocityVerlet(atoms, timestep=timestep_fs * units.fs, **kwargs)
```

> [!TIP]
> State the units in the argument names (`timestep_fs`, `taut_fs`) and convert inside the builder. The built-in `langevin_ase` passes raw ASE units while `langevin_flashmd` takes femtoseconds, and that inconsistency is an easy source of 10× time-step errors (see [Dynamics](dynamics.md#langevin_ase)).

Using a registered integrator:

```python
import my_md_plugins  # noqa: F401  (registers the keys)
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from goal.md import CalculatorFactory, DynamicsFactory, MoleculeFactory, run_dynamics, setup_observers

atoms = MoleculeFactory.create("from_g2", name="CH3CH2OH", vacuum=6.0)   # defined below
atoms.calc = CalculatorFactory.create("emt")
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

dyn = DynamicsFactory.create("bussi", atoms, timestep_fs=0.5, temperature_K=300)
obs = setup_observers(dyn, atoms, trajectory_file="outputs/md/bussi/md.traj", log_interval=10)
run_dynamics(dyn, steps=2_000)
```

> [!NOTE]
> `Simulation` builds an ASE or FlashMD Langevin integrator via `DynamicsFactory`. Other custom dynamics keys are therefore used through the [lower-level loop](dynamics.md#building-your-own-loop).

## Custom molecule builders

A molecule builder returns `ase.Atoms`, or a list of them. `Simulation.from_config` pops `method` from the `molecule:` mapping and passes the remaining keys as keyword arguments, so a builder with keyword-only parameters works directly from YAML.

```python
@register_molecule_set("from_g2")
class G2Builder(MoleculeBuilder):
    """Molecules from ASE's G2 collection, e.g. 'CH3CH2OH', 'H2O', 'C6H6'."""

    def build(self, name: str, vacuum: float | None = None) -> ase.Atoms:
        from ase.build import molecule
        atoms = molecule(name)
        if vacuum is not None:
            atoms.center(vacuum=vacuum)
            atoms.pbc = True
        return atoms


@register_molecule_set("from_ase_db")
class ASEDBBuilder(MoleculeBuilder):
    """Rows from an ASE database: one Atoms for `id=`, a list for a selection string."""

    def build(self, db_path: str, selection: str | None = None, id: int | None = None):
        from ase.db import connect
        db = connect(db_path)
        if id is not None:
            return db.get_atoms(id=id)
        return [row.toatoms() for row in db.select(selection)]
```

The built-in `from_db` key is a placeholder that raises `NotImplementedError`. The `from_ase_db` builder above is a working replacement, registered under its own key so the placeholder is left alone. Register it as `from_db` if you want to override the placeholder instead.

```yaml
molecule:
  method: from_ase_db
  db_path: structures.db
  id: 42
```

## Making your keys available to configs and the CLI

The registration has to be imported in the process that runs the simulation. In your own scripts and notebooks, `import my_md_plugins` is enough, and then config dicts can use the new keys:

```python
import my_md_plugins  # noqa: F401
from omegaconf import OmegaConf
from goal.md import simulate_from_config

simulate_from_config(OmegaConf.create({
    "molecule": {"method": "from_g2", "name": "H2O"},
    "calculator": {"type": "emt"},
    "steps": 1000, "output_dir": "outputs/md/water_emt",
}))
```

The installed `goal-simulate` command imports only `goal.md`, so it never sees your plugin module. Write a small entry point of your own that imports the plugins and then hands over to `simulate_from_config`. It accepts the same Hydra overrides as `goal-simulate`:

```python
# run_md.py
"""goal-simulate with project-local plugins registered."""
from pathlib import Path

import hydra
from omegaconf import DictConfig

import my_md_plugins  # noqa: F401  -- registers emt, dftd3_only, bussi, nve, from_g2, from_ase_db
from goal.md import simulate_from_config

CONFIG_DIR = str(Path(__file__).resolve().parent / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="water_emt")
def main(cfg: DictConfig) -> None:
    print(simulate_from_config(cfg))


if __name__ == "__main__":
    main()
```

```yaml
# configs/water_emt.yaml  (top-level file → already global, no @package line needed)
molecule:
  method: from_g2
  name: H2O
calculator:
  type: emt
steps: 5000
output_dir: outputs/md/water_emt
```

```bash
pixi run -e cuda-full python run_md.py steps=200 calculator.type=dftd3_only
```

### Contributing a builder to GOAL

To ship a backend with GOAL itself, add the builder to `src/goal/md/core/calculator_factory.py` (or `md_factory.py` / `molecule_factory.py`), following the guarded-import pattern at the top of that file. Add a template under `src/goal/configs/md/calculators/`, and add tests next to `tests/md/test_factories.py` that skip cleanly when the backend is absent. Builders in those modules are registered as soon as `goal.md` is imported, so they work with `goal-simulate` without any extra entry point.
