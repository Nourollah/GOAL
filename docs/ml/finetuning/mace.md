# Fine-tuning MACE

`mace_finetune` loads a pre-trained MACE model as a trainable `nn.Module` and
exposes it as a monolithic GOAL backbone that returns
`{energy, forces, num_atoms}`. It runs on the standard `GOALModule` and
Lightning trainer, so checkpoints, the stage curriculum, DDP and HPO work
unchanged. Read the [overview](overview.md) first for the strategies and why
E0 re-estimation matters.

Source:
[`goal.ml.nn.models.foundation.mace.MACEFinetune`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/foundation/mace.py).

## Install MACE in a standalone venv

`mace-torch` pins `e3nn==0.4.4`, which conflicts with the e3nn version that
GOAL's own equivariant blocks need. **Do not install it into the pixi
environment.** Even an isolated pixi environment drifts back to the locked
e3nn on the next `pixi install -e cuda` or `pixi run -e cuda …`. Use a plain venv that pixi
never touches (this is also described in the "MACE (mace-torch)" note in
[`pyproject.toml`](https://github.com/Nourollah/GOAL/blob/main/pyproject.toml)):

```bash
uv venv --python 3.14 .venv-mace
uv pip install --python .venv-mace/bin/python torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-mace/bin/python goal-atomistic   # or, from a clone: -e .
uv pip install --python .venv-mace/bin/python "mace-torch>=0.3.16"
# for strategy: lora
uv pip install --python .venv-mace/bin/python peft
```

Run fine-tuning with that interpreter (`.venv-mace/bin/goal-train`).

> [!WARNING]
> **torch ≥ 2.6 and e3nn 0.4.4.** e3nn 0.4.4 loads a constants file that
> contains a `slice` through `torch.load`, and torch ≥ 2.6 rejects that under
> the new `weights_only=True` default. GOAL's CLI imports e3nn (via its own
> model registry) before MACE gets a chance to relax the setting, so
> `.venv-mace/bin/goal-train` fails at start-up with
> `Unsupported global: GLOBAL slice`. Set the environment variable for the
> run:
>
> ```bash
> export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
> ```
>
> In a notebook, call `torch.serialization.add_safe_globals([slice])` before
> anything imports e3nn. Load only checkpoints you trust: both workarounds
> allow arbitrary pickles.

## Step by step

1. **Prepare data.** Any GOAL `dataset_type` works (see
   [Loading data](../data/loading.md)). The shipped config reads ExtXYZ from
   `data/finetune` with energy in `atoms.info["energy"]` and forces in
   `atoms.arrays["forces"]`. If your file uses other names, set
   `data.key_mapping`. To export an existing GOAL dataset:

   ```python
   from goal.ml.data.export import write_extxyz
   write_extxyz(my_dataset, "data/finetune/train.xyz")
   ```

2. **Check the config**,
   [`src/goal/configs/ml/finetune/mace.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/mace.yaml):
   - `data.root`: your file or directory.
   - `data.cutoff`: **must equal the checkpoint's `r_max`**. That is 6.0 Å
     for MACE-MP-0 small, medium and large (the config's comment claiming
     ~5.0 Å for small is wrong; the loaded model reports 6.0). The value is
     exposed as `backbone.cutoff`.
   - `model.backbone.checkpoint` and `model.backbone.strategy`.

3. **Run** with the MACE venv:

   ```bash
   export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
   .venv-mace/bin/goal-train --config-name finetune/mace \
       data.root=data/finetune model.backbone.strategy=head_only \
       model.backbone.device=cuda trainer.accelerator=gpu
   ```

4. **Use the result.** `GOALCheckpointManager` saves the checkpoint like any
   other GOAL model, so it loads through the normal calculator (from the
   MACE venv):

   ```python
   from goal.ml.utils.calculator import GOALCalculator

   calc = GOALCalculator(checkpoint_path="logs/finetune_mace/runs/<run>/checkpoints", checkpoint="best")
   atoms.calc = calc
   print(atoms.get_potential_energy(), atoms.get_forces())
   ```

   Reconstruction re-instantiates `mace_finetune`, which reloads the base
   MACE architecture and then overlays the fine-tuned weights. The base
   checkpoint must still be resolvable at load time. MACE-MP weights are
   cached under `~/.cache/mace/` after the first download, so this works
   offline.

## Backbone options

```yaml
model:
  backbone:
    name: mace_finetune
    checkpoint: "mace-mp-0-medium"   # see the table below, or a path to a .model/.pt
    strategy: head_only              # head_only | full | lora
    lora_rank: 4                     # lora only
    lora_alpha: 16.0
    lora_target_modules: null        # null = every nn.Linear
    dtype: float64
    reestimate_e0s: true             # read by FoundationE0Callback
    device: cpu                      # device the MACE loader puts the model on
    elements: null                   # fallback z-table, only if the model has no atomic_numbers buffer
  head: null
```

| `checkpoint` | Loads |
|---|---|
| `mace-mp-0-small`, `small` | MACE-MP-0 small |
| `mace-mp-0-medium`, `mace-mp-0`, `mace-mp`, `medium` | MACE-MP-0 medium |
| `mace-mp-0-large`, `large` | MACE-MP-0 large |
| `/path/to/model.model` | any MACE model file (MACE-OFF, MACE-OMol, your own) |

The model is loaded through `mace.calculators.mace_mp(..., return_raw_model=True)`,
using the backbone's `dtype` as MACE's default dtype.

- `head_only` leaves trainable only the parameters under `readouts.*`,
  MACE's per-layer energy heads.
- `full` trains everything. Drop `training.optimizer.lr` to about 1e-5.
- `lora` freezes the base model and injects LoRA adapters (`pip install peft`).

## What GOAL handles for you

- **E0 re-estimation.** `FoundationE0Callback` writes a least-squares fit on
  your training set into `atomic_energies_fn.atomic_energies` (every head's
  column for multi-head models).
- **dtype alignment.** Inputs are cast to the model dtype and outputs back
  to the label dtype.
- **AtomicGraph to MACE batch** (`atomicgraph_to_mace_batch`). It builds a
  one-hot `node_attrs` over the model's z-table (its `atomic_numbers`
  buffer) and Cartesian `shifts = unit_shifts @ cell`, matching
  `mace.data.AtomicData`. PBC systems work as long as the dataset provides
  `unit_shifts`.
- **Forces under validation.** `forward` runs MACE inside
  `torch.enable_grad()` with `compute_force=True`. The trainer still needs
  `inference_mode: false`.

> [!NOTE]
> A structure containing an element **outside the model's z-table** raises
> `ValueError: Structure contains element(s) Z=[...] absent from the MACE
> model's z-table`. MACE-MP-0 covers 89 elements, so this mostly affects
> specialised models such as MACE-OFF.

## Using the backbone outside training

Because `mace_finetune` returns GOAL's property dict, it doubles as a batched
inference engine and as an ASE calculator, with no subclassing:

```python
import torch
from ase.build import molecule
from torch_geometric.data import Batch
from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.models.foundation.mace import MACEFinetune
from goal.ml.utils.calculator import GOALCalculator

mace = MACEFinetune(checkpoint="medium", device="cpu")
graphs = [AtomicGraph.from_ase(molecule(m), cutoff=mace.cutoff) for m in ("CH3CH2OH", "H2O")]
out = mace(Batch.from_data_list(graphs))
print(out["energy"], out["forces"].shape)

calc = GOALCalculator(module=mace, cutoff=mace.cutoff)
```

## DDP with a frozen backbone

`head_only` and `lora` leave many parameters out of the graph. Under DDP set:

```yaml
strategy:
  name: ddp
  find_unused_parameters: true
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Unsupported global: GLOBAL slice` at start-up | e3nn 0.4.4 under torch ≥ 2.6. Set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| `ImportError: MACEFinetune requires mace-torch` | running in the pixi env. Use `.venv-mace/bin/goal-train` |
| "too many values to unpack" deep in `e3nn/util/codegen` | pixi re-synced e3nn to 0.6.x under MACE. Use the standalone venv |
| Loss starts at hundreds of eV | `FoundationE0Callback` missing from `callbacks:`, or `reestimate_e0s: false` |
| Forces wrong or noisy everywhere | `data.cutoff` does not match the model's `r_max` |
| DDP error about unused parameters | set `strategy.find_unused_parameters: true` |
