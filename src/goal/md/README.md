# GOAL MD Module

**Molecular Dynamics with Machine Learning Integration**

The `goal.md` module brings molecular dynamics (MD) capabilities to GOAL, seamlessly integrated with the `goal.ml` machine learning framework. It enables:

- **Run MD simulations** with trained ML models from goal.ml
- **Flexible calculator backends** (ML, quantum chemistry, pretrained potentials)
- **Config-driven workflows** via Hydra (reproducible science)
- **Direct library usage** for custom scripts and notebooks
- **Data format bridges** between goal.ml and ASE ecosystems

> **Design Philosophy**: `goal.md` is a satellite module around the `goal.ml` heart. It does NOT modify or depend on goal.ml internals, only uses its public APIs.

---

## Quick Start

### Installation

```bash
# With md dependencies (rdkit, ase, flashmd, etc.)
pip install goal[md]

# Or via pixi (recommended for reproducibility)
cd /path/to/GOAL
pixi run python -c "from goal.md import *; print('✓ ready')"
```

### Minimal Example

```python
from goal.md.core.molecule_factory import MoleculeFactory
from goal.md.adapters.model_loader import load_goal_calculator

# Create molecule from SMILES
atoms = MoleculeFactory.create("from_smiles", smiles="CCO")

# Load a trained model
calc = load_goal_calculator("outputs/train/2024-01-15_model/last.ckpt")

# Run MD
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

---

## Architecture

### Module Structure

```
goal/
├── ml/                    # ❤️ Heart & Head (ML training, data, models)
│   ├── data/
│   ├── nn/
│   ├── training/
│   └── ...
│
└── md/                    # 🛰️ Satellite (MD simulations, integrators)
    ├── core/              # Core MD functionality
    │   ├── molecule_factory.py      # Create molecules
    │   ├── calculator_factory.py    # Create calculators
    │   ├── md_factory.py            # Create dynamics
    │   └── molecule_tools.py        # Geometry utilities
    │
    ├── adapters/          # 🌉 Bridges to goal.ml
    │   ├── ase_converter.py         # ASE ↔ AtomicGraph conversion
    │   └── model_loader.py          # Load trained models
    │
    ├── cli/               # Command-line interface (isolated)
    ├── utils/             # Utilities
    └── config/            # Hydra configuration templates
```

### Key Bridges to goal.ml

#### 1. Model Loading Bridge
Load trained models from goal.ml checkpoints and use them in MD:

```python
from goal.md.adapters.model_loader import load_goal_calculator

calc = load_goal_calculator("path/to/checkpoint.ckpt")
atoms.calc = calc
```

Under the hood, this wraps `goal.ml.utils.calculator.GOALCalculator` with ASE compatibility.

#### 2. Data Conversion Bridge
Convert between ASE `Atoms` and goal.ml `AtomicGraph`:

```python
from goal.md.adapters import atomic_graph_from_ase, ase_atoms_from_atomic_graph
from goal.ml.data.graph import AtomicGraph

# ASE → goal.ml (e.g., for analysis or fine-tuning)
graph = atomic_graph_from_ase(atoms, cutoff=5.0)

# goal.ml → ASE (e.g., use goal.ml structures in MD)
atoms_new = ase_atoms_from_atomic_graph(graph)
```

#### 3. Calculator Factory with ML Support
The `CalculatorFactory` includes a `goal_model` builder for trained models:

```python
from goal.md.core.calculator_factory import CalculatorFactory

calc = CalculatorFactory.create(
    "goal_model",
    checkpoint="path/to/model.ckpt",
    device="cuda"
)
```

---

## Usage Patterns

### Pattern 1: Direct Import (Library Mode)

Use factories directly in Python scripts or notebooks:

```python
from goal.md.core import MoleculeFactory, CalculatorFactory, DynamicsFactory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Create molecule
atoms = MoleculeFactory.create("from_smiles", smiles="CCO")

# Create calculator (any registered backend)
calc = CalculatorFactory.create("flashmd")
atoms.calc = calc

# Setup and run dynamics
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = DynamicsFactory.create(
    "langevin_ase",
    atoms=atoms,
    temperature_K=300.0,
    timestep=1.0
)
dyn.run(1000)
```

### Pattern 2: Using Trained ML Models

Load and use models trained in goal.ml:

```python
from goal.md.adapters.model_loader import load_goal_calculator
from ase.md.langevin import Langevin
from ase import units

# Load checkpoint from goal.ml training
calc = load_goal_calculator(
    "outputs/train/2024-01-15_energy_model/last.ckpt",
    device="cuda"
)

# Attach to molecule and compute
atoms.calc = calc
e = atoms.get_potential_energy()
f = atoms.get_forces()

# Run MD
dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

### Pattern 3: Config-First (Recommended)

Use Hydra configs for reproducible workflows:

```python
from hydra import compose, initialize_config_dir
import os

# Initialize config directory
with initialize_config_dir(
    config_dir=os.path.abspath("configs/md"),
    version_base="1.3"
):
    # Compose config
    cfg = compose(config_name="langevin_with_model_sim")

    # Instantiate everything from config
    molecule = MoleculeFactory.create(**cfg.molecule)
    calculator = CalculatorFactory.create(**cfg.calculator)
    dynamics = DynamicsFactory.create(atoms=molecule, **cfg.dynamics)

    # Run
    dynamics.run(cfg.num_steps)
```

See `configs/md/langevin_with_model_sim.yaml` for a complete example.

### Pattern 4: Data Conversion Workflows

Convert between formats for different tools:

```python
from goal.md.adapters import atomic_graph_from_ase, ase_atoms_from_atomic_graph
from goal.md.core import MoleculeFactory

# Create molecule in ASE format
atoms = MoleculeFactory.create("from_smiles", smiles="C1=CC=CC=C1")

# Convert to goal.ml format for ML analysis
graph = atomic_graph_from_ase(atoms, cutoff=5.0)

# Use in goal.ml training pipeline
# ... dataset preparation, feature extraction, etc.

# Convert back to ASE for MD
atoms_reloaded = ase_atoms_from_atomic_graph(graph)
```

---

## API Reference

### Core Factories

#### MoleculeFactory

Create molecules from various sources:

```python
from goal.md.core.molecule_factory import MoleculeFactory

# From SMILES
atoms = MoleculeFactory.create("from_smiles", smiles="CCO")

# From file (XYZ, XSF, trajectory, etc.)
atoms = MoleculeFactory.create("from_file", path="structure.xyz")

# From database (not yet implemented)
# atoms = MoleculeFactory.create("from_db", db_path="structures.db")
```

**Registered builders**: `from_smiles`, `from_file`, `from_db`

#### CalculatorFactory

Create calculators for energy/force computation:

```python
from goal.md.core.calculator_factory import CalculatorFactory

# ML models (from goal.ml)
calc = CalculatorFactory.create("goal_model", checkpoint="model.ckpt")

# Pretrained models
calc = CalculatorFactory.create("flashmd")
calc = CalculatorFactory.create("mace", model_name="small")

# Quantum chemistry
calc = CalculatorFactory.create("orca", orca_path="/path/to/orca")
calc = CalculatorFactory.create("cp2k", cp2k_path="cp2k.ssmp")
```

**Registered builders**: `goal_model`, `flashmd`, `mace`, `orca`, `cp2k`, `psi4` (placeholder), `xtb` (placeholder)

#### DynamicsFactory

Create MD integrators:

```python
from goal.md.core.md_factory import DynamicsFactory

# Standard ASE Langevin
dyn = DynamicsFactory.create(
    "langevin_ase",
    atoms=atoms,
    timestep=1.0,
    temperature_K=300.0,
    friction=0.5
)

# Accelerated FlashMD Langevin
dyn = DynamicsFactory.create(
    "langevin_flashmd",
    atoms=atoms,
    timestep=16.0,
    temperature_K=300.0
)
```

**Registered builders**: `langevin_ase`, `langevin_flashmd`

### Adapters

#### Model Loading

```python
from goal.md.adapters.model_loader import load_goal_calculator, get_model_cutoff

# Load model
calc = load_goal_calculator(
    checkpoint="path/to/model.ckpt",
    device="cuda",
    cutoff=5.0  # Optional, read from checkpoint if None
)

# Extract model parameters
cutoff = get_model_cutoff("path/to/model.ckpt")
```

#### Data Conversion

```python
from goal.md.adapters.ase_converter import atomic_graph_from_ase, ase_atoms_from_atomic_graph

# ASE → goal.ml
graph = atomic_graph_from_ase(
    atoms,
    cutoff=5.0,
    energy=None,      # Optional: training target
    forces=None,      # Optional: training target
    dtype=torch.float64
)

# goal.ml → ASE
atoms = ase_atoms_from_atomic_graph(graph)
```

---

## Configuration System

### Hydra Config Structure

Configs are organized hierarchically:

```
configs/md/
├── molecule/           # Molecule source configs
│   ├── ethanol.yaml    # From SMILES
│   └── from_file.yaml  # From file
├── calculator/         # Calculator configs
│   ├── goal_model.yaml # Trained ML models
│   ├── cp2k.yaml       # 
│   ├── cp2k_pw.yaml    # 
│   ├── espresso.yaml   # 
│   ├── mace.yaml       # 
│   ├── tblite.yaml     # 
│   ├── upet.yaml       # 
│   ├── vasp.yaml       # 
│   ├── xtb.yaml        # 
│   ├── flashmd.yaml    # FlashMD pretrained
│   └── orca.yaml       # Quantum chemistry
├── dynamics/           # Dynamics configs
│   └── langevin.yaml   # Langevin thermostat
└── langevin_with_model_sim.yaml    # Complete example
```

### Example: Complete Simulation Config

`configs/md/langevin_with_model_sim.yaml`:

```yaml
defaults:
  - molecule: ethanol
  - calculator: goal_model
  - dynamics: langevin

# Simulation parameters
steps: 1000
trajectory_file: trajectory.traj
log_file: md.log
log_interval: 10
```

### Using Configs from Command Line

```bash
# Run with defaults
python simulation.py

# Override model checkpoint
python simulation.py calculator.checkpoint=path/to/model.ckpt

# Change molecule
python simulation.py molecule.smiles="CC(=O)O"

# Run with different config
python simulation.py --config-name=langevin_with_flashmd
```

---

## Available Calculators

### ML Models (goal.ml)

**`goal_model`** - Use trained models from goal.ml
- Load from checkpoint files
- Supports multi-task models with `head` parameter
- Auto-detects cutoff from config or accepts explicit value

### Pretrained Models

**`flashmd`** - FlashMD universal potential (accelerated MD)
- Fast inference with GPU support
- Pre-downloaded models

**`mace`** - MACE universal models
- Options: `"small"`, `"medium"`, `"large"` (or custom path)
- GPU-accelerated

### Quantum Chemistry

**`orca`** - DFT/quantum chemistry via ORCA
- Requires ORCA binary in PATH
- Configurable keywords and basis sets

**`cp2k`** - DFT via CP2K
- Requires CP2K binary
- Input configuration via file

**`psi4`** - DFT via Psi4 (placeholder, not implemented yet)

**`xtb`** - Semi-empirical xTB (placeholder, not implemented yet)

---

## Available Dynamics

### `langevin_ase` (Default)
Standard ASE Langevin thermostat:
```python
dyn = DynamicsFactory.create(
    "langevin_ase",
    atoms=atoms,
    timestep=1.0,        # fs
    temperature_K=300.0,
    friction=0.5,
    fixcm=True
)
```

### `langevin_flashmd` (Accelerated)
FlashMD-accelerated Langevin:
```python
dyn = DynamicsFactory.create(
    "langevin_flashmd",
    atoms=atoms,
    timestep=16.0,       # fs (optimized for FlashMD)
    temperature_K=300.0,
    time_constant=100.0  # fs
)
```

---

## Integration with goal.ml

### Workflow 1: Train → Simulate → Analyze

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Train Model in goal.ml                                   │
│    goal.ml.cli.train → checkpoint.ckpt                      │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Run MD with goal.md                                      │
│    load_goal_calculator(checkpoint.ckpt)                    │
│    → Run MD simulation → trajectory.traj                    │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Analyze in goal.ml                                       │
│    Convert trajectory to goal.ml format                     │
│    → Use for analysis, fine-tuning, etc.                    │
└─────────────────────────────────────────────────────────────┘
```

### Workflow 2: Multi-Model Comparison

```python
from goal.md.adapters.model_loader import load_goal_calculator
from goal.md.core import MoleculeFactory

# Create test molecule
atoms = MoleculeFactory.create("from_smiles", smiles="CCO")

# Compare multiple models
models = [
    "outputs/train/model_v1/last.ckpt",
    "outputs/train/model_v2/last.ckpt",
]

for model_path in models:
    calc = load_goal_calculator(model_path)
    atoms.calc = calc

    e = atoms.get_potential_energy()
    f = atoms.get_forces()

    print(f"{model_path}: E={e:.4f}, max|F|={f.max():.4f}")
```

---

## Examples & Notebooks

See `notebooks/md_introduction.py` for introduction and usage patterns.

Example configs:
- `configs/md/molecule/ethanol.yaml` - Create ethanol from SMILES
- `configs/md/calculator/goal_model.yaml` - Load trained model
- `configs/md/dynamics/langevin.yaml` - Langevin thermostat
- `configs/md/langevin_with_model_sim.yaml` - Complete workflow

---

## Design Principles

### 1. goal.ml is Untouched
The md module does NOT modify goal.ml:
- Uses only public APIs (`AtomicGraph`, `GOALCalculator`, etc.)
- No dependencies on goal.ml internals
- Can coexist with other goal.ml extensions

### 2. CLI is Isolated
CLI functionality is confined to the md module:
- Doesn't affect goal.ml CLI commands
- Separate Typer app in `goal.md.cli`
- Optional dependency tree

### 3. Config-First by Default
Reproducible science via Hydra:
- All workflows driven by configs
- Defaults can be overridden via CLI
- Configs are version-controlled for reproducibility

### 4. Direct Import Always Works
Can be used as a library without configs:
- Factory pattern for programmatic access
- No mandatory CLI or Hydra usage
- Easy integration into custom scripts

### 5. Optional Dependencies
MD module dependencies are optional:
```toml
[project.optional-dependencies]
md = [
  "flashmd>=0.2.5,<0.3",
  "ase>=3.25.0,<4",
  "xtb-python>=22.1,<23",
  "tblite>=0.7.0,<0.8",
  "upet>=0.1.2,<0.3",
  "rdkit>=2025.9.3,<2026",
  "typer>=0.16.0,<0.17",
  "pandas>=2.3.0,<3",
  "ipi>=3.0",
]
```
Install with: `pip install goal[md]`

---

## Extending the Module

### Register a Custom Calculator

```python
from goal.md.core.calculator_factory import register_calculator, CalculatorBuilder

@register_calculator("my_calculator")
class MyCalculatorBuilder(CalculatorBuilder):
    def build(self, param1, param2=default):
        # Create and return your calculator
        return MyCalculator(param1, param2)

# Use it
calc = CalculatorFactory.create("my_calculator", param1=value)
```

### Register a Custom Dynamics

```python
from goal.md.core.md_factory import register_dynamics, DynamicsBuilder

@register_dynamics("my_dynamics")
class MyDynamicsBuilder(DynamicsBuilder):
    def build(self, atoms, **kwargs):
        return MyDynamics(atoms, **kwargs)
```

### Register a Custom Molecule Builder

```python
from goal.md.core.molecule_factory import register_molecule_set, MoleculeBuilder

@register_molecule_set("my_source")
class MyMoleculeBuilder(MoleculeBuilder):
    def build(self, **kwargs):
        return my_build_logic(**kwargs)
```

---

## Known Limitations

- Some calculators (PSI4, xTB) are placeholders and raise `NotImplementedError`
- FlashMD requires GPU for best performance
- CP2K/ORCA require local binaries and environment setup

---

## Troubleshooting

### ImportError: No module named 'rdkit'
Install md dependencies:
```bash
pip install goal[md]
```

### ImportError: No module named 'flashmd'
Install FlashMD explicitly:
```bash
pip install flashmd
```

### Model checkpoint not found
Verify checkpoint path is correct:
```python
from pathlib import Path
ckpt = Path("outputs/train/model/last.ckpt")
assert ckpt.exists(), f"Not found: {ckpt}"
```

### Device errors with GPU
Specify device explicitly:
```python
calc = load_goal_calculator(
    checkpoint,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## Related Documentation

- **goal.ml**: `src/goal/ml/README.md` - Machine learning framework
- **ASE**: https://wiki.fysik.dtu.dk/ase/ - Atomic Simulation Environment
- **Hydra**: https://hydra.cc/ - Configuration management
- **FlashMD**: https://github.com/flashmd - Accelerated MD
- **MACE**: https://github.com/ACEsuit/mace - Universal potentials

---

## Citation

If you use the md module in research, please cite GOAL:

```bibtex
@software{goal,
  title={GOAL: General Open Atomistic Laboratory},
  author={Nourollah, Amir Masoud},
  url={https://github.com/Nourollah/GOAL},
  year={2024}
}
```

---

## License

MIT License - See LICENSE file in repository root

---

## Multiple-Time-Step (MTS) MD with Dual-Calculator Fine-Tuning

### Overview

The `goal.md.mts` module enables advanced MTS simulations where a fast ML model
learns from an accurate but expensive base calculator (QM or expensive ML).

**Key Features:**
- Run QM/expensive ML as reference (base) calculator
- Train fast ML model on-the-fly from base data
- Automatically switch to ML once accuracy meets threshold
- Continuous monitoring with fallback mechanisms
- Config-driven orchestration

### Workflow

```
┌─────────────────────────────────────┐
│ 1. Initialize: Setup QM + ML model  │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 2. Training Phase: QM running,      │
│    ML model learning from QM data   │
└──────────────┬──────────────────────┘
               ↓
        ┌──────────────┐
        │ Accurate?    │
        └─┬────────┬───┘
      no │         │ yes
        ↓         ↓
    Continue   Switch to ML
       QM      (QM standby)
```

### Quick Start

```python
from goal.md.mts.dual_calculator import QMLearnerCalculator

# Create dual calculator
dual_calc = QMLearnerCalculator(
    base_calculator=qm_calc,           # e.g., ORCA
    ml_calculator=ml_model,            # e.g., trained goal.ml model
    force_rmse_threshold=0.01,         # eV/Å
    energy_mae_threshold=0.001,        # eV/atom
    min_training_steps=500
)

# Use in MD
atoms.calc = dual_calc
dyn = Langevin(atoms, ...)
dyn.run(10000)

# Monitor
metrics = dual_calc.get_training_history()
print(f"Active calculator: {metrics['active_calculator']}")
print(f"Force RMSE: {metrics['current_force_rmse']}")
```

### Configuration-Based Usage

```python
from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir="configs/md/mts"):
    cfg = compose(config_name="simulations/orca_ml_learner")

    # Auto-build dual calculator + dynamics
    sim = build_mts_simulation(cfg)
    sim.run()
```

### Advanced Features

#### Switching Strategies

- **Accuracy-Based** (default): Switch when force/energy errors below threshold
- **Time-Based**: Switch after fixed number of steps
- **Hybrid**: Blend calculator results with dynamic weighting
- **Fallback**: Auto-revert to base if ML diverges

#### Training Modes

- **Online Gradient**: Update model every step
- **Batch Replay**: Collect data, train periodically
- **Experience Replay**: Buffer with random sampling

#### Monitoring & Control

```python
# Get live metrics
state = dual_calc.get_state()
print(f"Step: {state.step}")
print(f"Force RMSE: {state.force_rmse:.6f} eV/Å")
print(f"Active: {state.active_calculator}")

# Add custom training data
dual_calc.add_training_data(atoms, forces, energy)

# Get history
history = dual_calc.get_training_history()
```

### Use Cases

1. **Accelerating QM-based MD**: Train fast ML surrogate from QM data
2. **Fine-tuning ML models**: Use simulation to refine model on relevant data
3. **Uncertainty quantification**: Compare QM vs ML predictions
4. **Multi-scale modeling**: Hierarchical force computation
5. **Online learning**: Model improves as simulation progresses

### Configuration Examples

See `configs/md/mts/` for:
- `simulations/orca_ml_learner.yaml` - ORCA + ML learner
- `simulations/qm_ml_adaptive.yaml` - Pre-trained QM + adaptive ML

### Architecture Details

See `MTS_DUAL_CALCULATOR_DESIGN.md` for comprehensive architecture documentation.

---
