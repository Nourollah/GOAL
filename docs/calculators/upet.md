# UPET calculator

Runs metatensor PET foundation models for MD/inference via
`upet.calculator.UPETCalculator`. Builder:
[`UPETBuilder`](../../src/goal/md/core/calculator_factory.py), `key: upet`.

Install: `pip install upet` (missing → `GOALCalculatorNotFound`).

> **Note** — the `upet` package is the renamed successor of `pet-mad`
> (same project, `lab-cosmo/upet`); `pet-mad` 1.4.4 is the last release under
> the old name and old API. "UPET" and "PET-MAD" refer to the same family.

## Model families and sizes

| family | trained on | sizes |
| --- | --- | --- |
| `pet-mad` | MAD dataset (r2SCAN from v1.5) | `xs, s` |
| `pet-oam` | OAM | `l, xl` |
| `pet-omat` | OMat | `xs, s, m, l, xl` |
| `pet-omad` | OMat + MAD | `xs, s, l` |
| `pet-omatpes` | OMatPES | `l` |
| `pet-spice` | SPICE (molecular) | `s, l` |

The released set is **not** a family × size cross-product — there is no
`pet-mad-l`. The builder validates against `upet._version.UPET_AVAILABLE_MODELS`
(the upstream list) rather than a hand-maintained one, so an unknown id raises
`ValueError` listing exactly what upstream ships.

## Configuration

`configs/md/calculator/upet.yaml`:

```yaml
_target_: goal.md.core.calculator_factory.CalculatorFactory.create
_convert_: none
key: upet

model: pet-mad-s
version: null            # null = latest; pin (e.g. "1.5.0") for reproducibility
checkpoint_path: null    # OR a fine-tuned metatrain .ckpt
variants: null           # null = each output's default head
device: cuda
non_conservative: false
```

### `variants` — leave it `null` unless you mean it

`variants` picks a named head *inside* an output, so `{"energy": "r2scan"}`
selects the `energy/r2scan` head. There is no head called `energy/energy`:
passing `{"energy": "energy"}` raises
`ValueError: variant 'energy' for output 'energy' not found in outputs`.
`null` (the default) uses each output's default head, which is what you want
for a stock released model.

## ⚠️ `non_conservative`

`non_conservative: true` selects a **direct** force head that does **not**
come from `-∂E/∂r`, so it **violates energy conservation**. This causes energy
drift and breaks thermostats in long MD. GOAL logs a loud warning and it is
**never** the default. Use only for fast single-point screening where
conservation is irrelevant.

## Using a fine-tuned model

After `goal-finetune-upet` (see [finetuning/upet.md](../finetuning/upet.md)):

```python
from goal.md import CalculatorFactory
calc = CalculatorFactory.create(
    "upet",
    checkpoint_path="runs/upet_finetuned/model.ckpt",
    variants={"energy": "finetune"},
)
```

## Example

```python
from goal.md import CalculatorFactory
from ase.build import molecule

atoms = molecule("CH4")
atoms.calc = CalculatorFactory.create("upet", model="pet-mad-s", device="cuda")
print(atoms.get_potential_energy())
```
