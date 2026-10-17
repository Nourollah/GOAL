# MACE, NequIP & FlashMD

These three backends give you energies and forces without training anything yourself. MACE, NequIP, and FlashMD are pretrained or externally trained ML potentials. Use them for exploratory MD, to pre-screen structures before DFT, or as the fast level of a [multiple-time-step](../mts/overview.md) run. For UPET / PET-MAD see [its own page](upet.md). For cheap physics-based (semi-empirical) energies, see [tblite (GFN-xTB)](tblite.md).

| Key | Builder | Python dependency | Missing dependency raises |
|-----|---------|-------------------|---------------------------|
| `mace` | `MACEBuilder` | `mace-torch` (**separate venv**, see below) | `ImportError` |
| `nequip` | `NequIPBuilder` | `nequip` | `ImportError` |
| `flashmd` | `FlashMDBuilder` | `flashmd` (in `goal-atomistic[md]`) | `ImportError` |

> [!NOTE]
> These are *inference* backends in `goal.md`. Fine-tuning MACE or UPET as a GOAL backbone is a different code path. See [Fine-tuning overview](../../ml/finetuning/overview.md) and [MACE fine-tuning](../../ml/finetuning/mace.md).

## MACE

```python
from goal.md import CalculatorFactory

calc = CalculatorFactory.create("mace", model_name_or_path="medium")            # MACE-MP-0 medium
calc = CalculatorFactory.create("mace", model_name_or_path="models/my.model")   # your own MACE model
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `model_name_or_path` | `"small"` | `"small"`, `"medium"`, or `"large"` → `mace.calculators.mace_mp(...)` (universal MACE-MP). Any other string is treated as a model path → `MACECalculator(model_paths=...)`. |
| `device` | `None` | `None` → CUDA if available, else CPU |

Config: [`src/goal/configs/md/calculators/mace.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/calculators/mace.yaml) (`key: mace`, `model_name_or_path: small`).

> [!NOTE]
> The builder does not expose `default_dtype`. `mace_mp()` defaults to `float32`, which is fine for screening but adds noise to long NVE energy conservation. For production MD, build the upstream calculator directly: `mace_mp(model="medium", device="cuda", default_dtype="float64")`. Any ASE calculator plugs into `goal.md`.

### Installing MACE: use a standalone venv

> [!WARNING]
> `mace-torch` hard-pins `e3nn==0.4.4`, while GOAL's own equivariant blocks need `e3nn>=0.5`. The `goal-atomistic[mace]` extra is therefore **unsatisfiable by construction**. Do **not** try to downgrade e3nn inside a pixi environment. Pixi doesn't track the downgrade, and the next `pixi install -e cuda` / `pixi run -e cuda …` silently restores e3nn 0.6.x. `mace_mp()` then fails with an opaque `too many values to unpack` inside `e3nn/util/codegen/_mixin.py`.

Use a plain venv that pixi never touches (this is the recipe from `pyproject.toml`, verified with Python 3.14, torch 2.13+cu126, e3nn 0.4.4, and mace-torch 0.3.16):

```bash
uv venv --python 3.14 .venv-mace
uv pip install --python .venv-mace/bin/python torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-mace/bin/python goal-atomistic   # or, from a clone: -e .
uv pip install --python .venv-mace/bin/python "mace-torch>=0.3.16" jupyter ipykernel
.venv-mace/bin/python -m ipykernel install --user --name goal-mace-cuda \
    --display-name "GOAL (mace-cuda, standalone)"
```

Then pick the **GOAL (mace-cuda, standalone)** kernel for MACE notebooks (e.g. `notebooks/mace_foundation_gmd_gallery.ipynb`, `notebooks/mof74_mn_bulk.ipynb`).

### `torch.load` and `add_safe_globals([slice])`

e3nn 0.4.4 loads its bundled Wigner-3j constants with a bare `torch.load(...)`. Since torch 2.6 (`weights_only=True` by default) that load is rejected because the pickle contains `slice` objects. Allow-list `slice` **before importing mace or e3nn**:

```python
import torch
torch.serialization.add_safe_globals([slice])

from goal.md import CalculatorFactory          # imports mace lazily on create()
calc = CalculatorFactory.create("mace", model_name_or_path="small", device="cuda")
```

The MACE and UPET gallery notebooks do exactly this in their first cell.

## NequIP and Allegro

```python
calc = CalculatorFactory.create(
    "nequip",
    model_path="deployed/ethanol_nequip.pth",
    device="cuda",
    species_to_type_name={"H": "H", "C": "C", "O": "O"},   # only if you used custom type names
)
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `model_path` | required | Deployed NequIP / Allegro model (`*.pth` / `*.pt`) |
| `device` | `None` | `None` → CUDA if available |
| `species_to_type_name` | `None` | Element → NequIP type-name map when the model was trained with custom type names |

The builder calls `nequip.ase.NequIPCalculator.from_deployed_model(...)`. There is no shipped YAML config.

> [!NOTE]
> `from_deployed_model` is the API of the NequIP 0.6 line (models produced by `nequip-deploy`). Newer NequIP releases changed how compiled/packaged models are loaded. If your NequIP version lacks `from_deployed_model`, load the model with that version's own ASE calculator and assign it to `atoms.calc` directly. This builder has not been tested against NequIP ≥ 0.7.

## FlashMD

```python
calc = CalculatorFactory.create("flashmd", model_name="pet-omatpes-v2", device=None)
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `model_name` | `"pet-omatpes-v2"` | FlashMD pretrained model identifier |
| `device` | `None` | `None` → CUDA if available |

Internally, the builder calls `flashmd.get_pretrained(model_name, 16)` and wraps the **energy model** in `flashmd.ase.EnergyCalculator`. What you get is therefore a conventional MLIP calculator: energies and forces, integrated by whatever dynamics you choose. It is *not* FlashMD's large-timestep trajectory predictor.

To use FlashMD's actual accelerated propagation (direct prediction of positions/momenta over ~16 fs strides), use the dynamics key `langevin_flashmd` in `DynamicsFactory`, which loads the FlashMD model alongside. See [Dynamics](../guide/dynamics.md).

Config: [`src/goal/configs/md/calculators/flashmd.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/md/calculators/flashmd.yaml).

> [!WARNING]
> `flashmd.yaml` names the backend with `method: flashmd` rather than `key: flashmd`, so `hydra.utils.instantiate` fails on it. Set `cfg.key = cfg.pop("method")` first, or use `CalculatorFactory.create("flashmd", ...)` directly.
