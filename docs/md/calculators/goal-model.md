# Trained GOAL models as calculators

The `goal_model` backend loads a potential you trained with `goal.ml` and returns it as an ASE calculator (`goal.ml.utils.calculator.GOALCalculator`). You can use it for MD, geometry optimisation, NEB, phonons, or as the fast level of a [multiple-time-step](../mts/overview.md) run. This page covers the MD side. For how the calculator itself works, see [ASE calculator](../../ml/deployment/ase-calculator.md), and for the formats it loads, see [Checkpoints & archives](../../ml/deployment/archives.md).

## Quick start

```python
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from goal.md import CalculatorFactory, MoleculeFactory

atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
atoms.calc = CalculatorFactory.create(
    "goal_model",
    checkpoint="models/ethanol.simurgh",   # archive, checkpoint dir, or .ckpt
    device="cuda",
)

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.01)
dyn.run(2000)
```

## Accepted checkpoint formats

`checkpoint` is passed straight to `GOALCalculator(checkpoint_path=...)`, which dispatches on what it gets:

| You pass | Format | Notes |
|----------|--------|-------|
| `my_model.simurgh` | **A: self-contained archive** (weights, resolved config, metadata, *frozen model source*) | **Recommended for MD.** It loads from the frozen source, so the archive keeps working after you change model code. Build one with `goal-pack-archive`. |
| `logs/.../checkpoints/` | **B: checkpoint directory** written by `GOALCheckpointManager` (`config.yaml`, `metadata.json`, `frozen_source/`) | `select` picks the weights: `best` (default), `last`, `epoch=N` or a `.ckpt` name. |
| `last.ckpt` | **C: bare Lightning checkpoint** (legacy) | Rebuilt from the *live* source code, with a warning. It breaks if the model code changed since training. |

To produce an archive from a training run:

```bash
pixi run -e cuda-full goal-pack-archive \
    --checkpoint-dir logs/train/runs/<run>/checkpoints/ \
    --checkpoint best \
    --output models/ethanol.simurgh
```

## Builder arguments

`GOALModelBuilder.build()`:

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `checkpoint` | `str \| Path` | required | Archive, checkpoint directory, or `.ckpt` file |
| `select` | `str` | `"best"` | Weights inside a checkpoint directory: `best`, `last`, `epoch=N` or a `.ckpt` filename |
| `use_ema` | `bool` | `True` | Use the EMA weights when the checkpoint has them (the weights validation and `best` selection used) |
| `device` | `str` | `"cpu"` | `"cpu"`, `"cuda"`, `"cuda:1"`, … |
| `dtype` | `torch.dtype \| None` | `None` | Input precision. `None` matches the model, which is built in the dtype its checkpoint was saved in |
| `head` | `str \| None` | `None` | Head tag for multi-head models |
| `fragment_on_failure` | `str \| None` | `"fallback"` | For fragment-aware models: what to do when rdkit bond perception fails mid-run; `None` keeps the training setting |
| `neighbor_list_backend` | `str \| None` | `None` | `None` = the training config's backend (else `ase`); `"auto"` uses the GPU kernel for large systems ([Neighbour lists](../../integrations/neighbor_list.md)) |
| `cutoff` | `float \| None` | `None` | Ignored (logged): the cutoff always comes from the checkpoint, because a model evaluated with another neighbour list is a different model. Kept so older configs still load. |

Models trained with fragment labels (`data.compute_fragment_index`) get them at inference automatically; the settings are read from the checkpoint's training config. See [ASE calculator](../../ml/deployment/ase-calculator.md#models-that-need-fragment-labels).

`GOALCalculator` implements `energy`, `forces`, and `stress`. Each property is filled only if the model's heads predict it. Forces from an energy-gradient head are conservative ($\mathbf F = -\partial E / \partial \mathbf r$), which is what you want for NVE/NVT. A direct-force head is faster but not conservative. See [Heads](../../ml/models/heads.md).

There is also a thin functional alias, `goal.md.adapters.model_loader.load_goal_calculator(checkpoint_path, device="cpu", dtype=None, head=None, checkpoint="best", use_ema=True)`, and `get_model_cutoff(checkpoint_path)` reads a model's cutoff from any of the three formats without building it.

## From a config

[`src/goal/configs/md/calculators/goal_model.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/calculators/goal_model.yaml):

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none

key: goal_model
checkpoint: ??? # Path to model checkpoint (archive, directory or .ckpt)
select: best # checkpoint directories only: best | last | epoch=N | <name>.ckpt
use_ema: true # load the EMA weights when the checkpoint has them
device: cpu
head: null
```

It is the `calculator:` of `goal-simulate`'s default config, so `goal-simulate calculator.checkpoint=models/ethanol.simurgh` uses it. From Python, `hydra.utils.instantiate` builds it directly:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

calc = instantiate(OmegaConf.load("src/goal/configs/md/calculators/goal_model.yaml"),
                   checkpoint="models/ethanol.simurgh", select="last")
```

In MTS configs (`fast_calculator:` / `ml_calculator:` blocks) `key: goal_model` and `type: goal_model` both work.

## Practical advice

- **Check the model before long MD.** Run a few hundred NVE steps and watch total-energy drift. A drift that grows with time usually means the trajectory left the training distribution. In that case consider [online learning](../mts/overview.md) or more data.
- **Timestep:** ML potentials do not lift the physical constraint from the fastest vibration. Use 0.5 fs with explicit H and 1 fs for heavy-atom-only systems, or use RESPA to spend a cheap model on the inner steps.
- **Cutoff:** the cutoff always comes from the checkpoint. Message passing assumes the trained neighbourhood, so it cannot be changed at inference.
- **GPU:** single small molecules are often *faster on CPU*, because graph construction and kernel-launch overhead dominate. Benchmark both.
- **Units:** inputs are Å; outputs are eV, eV/Å, and eV/Å³ (stress in Voigt order, ASE sign convention).
