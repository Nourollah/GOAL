# UPET / PET-MAD calculator

The `upet` backend runs metatensor PET foundation models (PET-MAD, PET-OMat, PET-SPICE, …) through upstream's `upet.calculator.UPETCalculator`. It handles both stock released weights and your own fine-tuned checkpoints. The builder is [`UPETBuilder`](https://github.com/Nourollah/GOAL/blob/main/src/goal/md/core/calculator_factory.py), selected with `key: upet`.

Install: `pip install upet`. If it is missing, `create("upet", ...)` raises `GOALCalculatorNotFound`.

> [!NOTE]
> The `upet` package is the renamed successor of `pet-mad` (same project, `lab-cosmo/upet`). `pet-mad` 1.4.4 is the last release under the old name and old API. "UPET" and "PET-MAD" refer to the same family.

## Model families and sizes

| Family | Trained on | Sizes | Typical use |
| --- | --- | --- | --- |
| `pet-mad` | MAD dataset (r2SCAN from v1.5) | `xs, s` | general-purpose, molecules and materials |
| `pet-oam` | OAM | `l, xl` | inorganic materials |
| `pet-omat` | OMat | `xs, s, m, l, xl` | inorganic materials, off-equilibrium |
| `pet-omad` | OMat + MAD | `xs, s, l` | mixed |
| `pet-omatpes` | OMatPES | `l` | materials PES |
| `pet-spice` | SPICE (molecular) | `s, l` | organic / drug-like molecules |

The released set is **not** a family × size cross-product; for example, there is no `pet-mad-l`. The builder validates `model` (case-insensitive) against `upet._version.UPET_AVAILABLE_MODELS`, the upstream list, rather than a hand-maintained one. An unknown identifier raises `ValueError` listing exactly what upstream ships. Validation is skipped when `checkpoint_path` is given.

## Builder arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `model` | `"pet-mad-s"` | Released model identifier (see the table above) |
| `checkpoint_path` | `None` | Fine-tuned metatrain `.ckpt`. When given, it **takes precedence over `model`**. |
| `variants` | `None` | Head selection *within* an output, e.g. `{"energy": "r2scan"}`. `None` = each output's default head. |
| `version` | `None` | Pin released weights (e.g. `"1.5.0"`). `None` = upstream's latest. |
| `device` | `"cuda"` | Torch device |
| `non_conservative` | `False` | Use the direct force head. **Breaks energy conservation**; see below. |

## Configuration

[`src/goal/configs/md/calculators/upet.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/calculators/upet.yaml):

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none
key: upet

model: pet-mad-s
version: null            # null = latest published weights; pin (e.g. "1.5.0") for reproducibility
checkpoint_path: null    # OR path to a fine-tuned metatrain .ckpt
variants: null           # null = each output's default head.
device: cuda
non_conservative: false
```

### `version`: pin it for anything you will cite

`version: null` resolves to whatever weights are newest *at run time*. When upstream publishes new weights, the same config silently gives different numbers. Pin a version for production runs and publications.

### `variants`: leave it `null` unless you mean it

`variants` picks a named head *inside* an output, so `{"energy": "r2scan"}` selects the `energy/r2scan` head. There is no head called `energy/energy`, so passing `{"energy": "energy"}` raises `ValueError: variant 'energy' for output 'energy' not found in outputs`. `null` (the default) uses each output's default head, which is what you want for a stock released model.

> [!WARNING]
> **`non_conservative: true`** selects a **direct** force head that does **not** come from $-\partial E/\partial \mathbf r$, so it **violates energy conservation**. That causes energy drift and breaks thermostats in long MD. GOAL logs a loud warning (and emits a `UserWarning`), and the option is never the default. Use it only for fast single-point screening where conservation is irrelevant. PET-SPICE and PET-MAD < 1.1.0 don't support it.

## Using a fine-tuned model

After `goal-finetune-upet` (see [UPET / PET fine-tuning](../../ml/finetuning/upet.md)):

```python
from goal.md import CalculatorFactory

calc = CalculatorFactory.create(
    "upet",
    checkpoint_path="runs/upet_finetuned/model.ckpt",
    variants={"energy": "finetune"},   # the head written by goal-finetune-upet
)
```

## Example: short NVT run

```python
import torch
torch.serialization.add_safe_globals([slice])   # harmless here; needed if e3nn 0.4.4 is in the same env

from ase import units
from ase.build import molecule
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from goal.md import CalculatorFactory

atoms = molecule("CH3CH2OH")
atoms.center(vacuum=8.0)
atoms.calc = CalculatorFactory.create("upet", model="pet-spice-s", version=None, device="cuda")

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.01).run(1000)
print(atoms.get_potential_energy())
```

For a worked comparison of PET-MAD against GMD reference labels, see `notebooks/upet_petmad_gmd_gallery.ipynb` ([Notebooks](../../getting-started/notebooks.md)).
