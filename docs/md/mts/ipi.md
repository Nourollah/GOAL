# i-PI adapter

[i-PI](https://github.com/i-pi/i-pi) is the reference implementation of advanced MD integrators: multiple time-stepping, path-integral MD, and sophisticated thermostats. `goal.md.mts.ipi_adapter` mirrors i-PI's force-provider interface so that GOAL calculators are i-PI-shaped, and its `MTSSimulation` reproduces i-PI's RESPA physics in-process. This page explains what the adapter actually does today and how to drive a **real** i-PI server with GOAL calculators.

## What the module provides

| Object | What it does | Status |
|--------|--------------|--------|
| `IPIForceProvider` | `typing.Protocol` for i-PI's force-field interface: `get_forces(pos_bohr, cell_bohr) -> (forces_au, energy_au, virial_au)` | interface only |
| `IPICompatibleDriver(calculator, atoms)` | Wraps any ASE calculator to satisfy that protocol. Handles Å↔bohr, eV↔Hartree, and eV/Å↔Hartree/bohr, and computes the virial as $-\sigma V$ (zeros if the calculator has no stress). `get_forces_ase(atoms)` returns `(forces_eV/Å, energy_eV)`. | working, in-process |
| `MTSConfig`, `MTSSimulation` | High-level RESPA runner (see [RESPA integrator](respa.md)). It builds `IPICompatibleDriver` wrappers for both levels, but integration is done by GOAL's own `RespaIntegrator`. | working |
| `MTSSimulation.use_ipi_backend()` | Checks that `ipi` is importable and sets a flag. A subsequent `run()` then raises `NotImplementedError`. | **planned**: forward-compatibility hook only |

> [!IMPORTANT]
> Nothing in `goal.md` talks to an i-PI server yet. There is no socket client and no XML generation. `IPICompatibleDriver` is an in-process unit-conversion layer designed so that a future Python-level i-PI API can call GOAL calculators directly. To use i-PI's integrators today, connect GOAL calculators to i-PI through **ASE's socket client**, as shown below.

### `IPICompatibleDriver` example

```python
import numpy as np
from ase import units
from goal.md import CalculatorFactory, MoleculeFactory
from goal.md.mts import IPICompatibleDriver

atoms = MoleculeFactory.create("from_smiles", smiles="O")
calc = CalculatorFactory.create("goal_model", checkpoint="models/water.simurgh")
driver = IPICompatibleDriver(calc, atoms)

pos_bohr = atoms.get_positions() / units.Bohr
cell_bohr = np.eye(3) * 20.0 / units.Bohr
f_au, e_au, vir_au = driver.get_forces(pos_bohr, cell_bohr)   # Hartree/bohr, Hartree, Hartree
f_ev, e_ev = driver.get_forces_ase(atoms)                      # eV/Å, eV
```

## Running GOAL calculators under a real i-PI server

i-PI is a *server*. It owns the positions and integrates them, and it asks *clients* for forces over a socket. Any ASE calculator becomes a client through `ase.calculators.socketio.SocketClient`, and that includes every `CalculatorFactory` backend.

### 1. Install

`ipi` is in the `md` extra (`pip install "goal-atomistic[md]"`, which pulls `ipi>=3.0`) and provides the `i-pi` command. It is not part of the default pixi environment.

### 2. Write the starting structure in i-PI's xyz format

```python
from goal.md import MoleculeFactory

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
box = 20.0  # Å; large vacuum box for an isolated molecule
with open("init.xyz", "w") as fh:
    fh.write(f"{len(atoms)}\n")
    fh.write(f"# CELL(abcABC): {box} {box} {box} 90.0 90.0 90.0 cell{{angstrom}} positions{{angstrom}}\n")
    for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        fh.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")
```

### 3. i-PI input with QM/ML multiple time-stepping

The same force split as GOAL's RESPA, written in i-PI's MTS syntax. `<nmts>` gives the subdivisions per level, outermost first, and `<mts_weights>` says which level each force contributes to. QM enters only the outer level, and ML enters the inner level, subtracted from the outer one.

```xml
<simulation verbosity='medium'>
  <output prefix='ethanol'>
    <properties stride='1' filename='out'>
      [ step, time{picosecond}, conserved, temperature{kelvin}, kinetic_md, potential ]
    </properties>
    <trajectory filename='pos' stride='10' format='xyz'> positions{angstrom} </trajectory>
  </output>
  <total_steps> 2000 </total_steps>
  <prng><seed> 31415 </seed></prng>

  <ffsocket name='qm' mode='unix' pbc='false'><address> goal-qm </address></ffsocket>
  <ffsocket name='ml' mode='unix' pbc='false'><address> goal-ml </address></ffsocket>

  <system>
    <initialize nbeads='1'>
      <file mode='xyz'> init.xyz </file>
      <velocities mode='thermal' units='kelvin'> 300 </velocities>
    </initialize>
    <forces>
      <force forcefield='qm'> <mts_weights> [1, 0] </mts_weights> </force>
      <force forcefield='ml'> <mts_weights> [-1, 1] </mts_weights> </force>
    </forces>
    <motion mode='dynamics'>
      <dynamics mode='nvt'>
        <timestep units='femtosecond'> 2.0 </timestep>   <!-- OUTER step -->
        <nmts> [1, 4] </nmts>                            <!-- 4 inner steps of 0.5 fs -->
        <thermostat mode='svr'><tau units='femtosecond'> 100 </tau></thermostat>
      </dynamics>
    </motion>
    <ensemble><temperature units='kelvin'> 300 </temperature></ensemble>
  </system>
</simulation>
```

The XML above follows i-PI's documented MTS input. Check element names against the manual of the i-PI version you install.

### 4. Start the server, then one client per force field

```bash
i-pi input.xml > ipi.log &
sleep 5                                    # let i-PI open its sockets
pixi run -e cuda-full python client.py --socket goal-ml --backend goal_model &
pixi run -e cuda-full python client.py --socket goal-qm --backend cp2k &
wait
```

`client.py`:

```python
import argparse
from ase.calculators.socketio import SocketClient
from ase.io import read
from goal.md import CalculatorFactory

p = argparse.ArgumentParser()
p.add_argument("--socket", required=True)
p.add_argument("--backend", required=True)
args = p.parse_args()

atoms = read("init.xyz")        # only species and order matter; i-PI sends positions and cell
if args.backend == "goal_model":
    atoms.calc = CalculatorFactory.create("goal_model", checkpoint="models/ethanol.simurgh")
else:
    atoms.calc = CalculatorFactory.create("cp2k", method="sirius", preset="molecular",
                                          pw_cutoff=18.0, gk_cutoff=9.0, dispersion="d3bj")

# ASE connects to /tmp/ipi_<name>, which matches <ffsocket mode='unix'><address>name</address>
SocketClient(unixsocket=args.socket).run(atoms)
```

ASE's socket client does the unit conversion itself, so `IPICompatibleDriver` is not involved in this path.

## When to use i-PI vs GOAL's RESPA

| Need | Use |
|------|-----|
| Two-level QM/ML MTS, Langevin or NVE, everything in one Python process | `MTSSimulation` / [`RespaIntegrator`](respa.md) |
| Path-integral MD (nuclear quantum effects), PIGLET/GLE thermostats, more than two MTS levels, replica exchange, NPT with advanced barostats | i-PI server + ASE `SocketClient` |
| Several force evaluations in parallel on different nodes | i-PI (clients can connect over TCP with `mode='inet'` and `SocketClient(host=..., port=...)`) |
| Online learning with `QMLearnerCalculator` | either. The learner is just an ASE calculator, so it can serve forces to i-PI as a socket client too. |

Related reading: the FlashMD project ships its own i-PI integration ([`flashmd/ipi.py`](https://github.com/lab-cosmo/flashmd/blob/main/src/flashmd/ipi.py)), which GOAL's adapter was modelled on.
