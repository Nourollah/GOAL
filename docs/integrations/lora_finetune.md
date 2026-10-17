# LoRA fine-tuning of SIMURGH

`simurgh_lora` adapts a trained SIMURGH or ARACE checkpoint to a new, usually small, dataset by training low-rank adapters (LoRA) on its artisan layers while the rest of the checkpoint stays frozen. It runs on the standard `GOALModule`, Lightning trainer and `GOALCheckpointManager`, so checkpoints, archives, resume and DDP work as in native training, and the result loads in `GOALCalculator` like any other model.

Typical use: a model trained on one chemical family (say monocarboxylic acids) that has to move to a neighbouring one (dicarboxylic acids) with a few hundred labelled structures. For foundation models (MACE, UMA, UPET) see [Fine-tuning overview](../ml/finetuning/overview.md) instead.

Source:
[`goal.ml.adapters.simurgh_lora`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/adapters/simurgh_lora.py),
config [`src/goal/configs/ml/finetune/simurgh_lora.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/simurgh_lora.yaml).

## What it is

LoRA replaces the update of a weight matrix $W$ by a low-rank product: the layer computes $W x + \frac{\alpha}{r} B A x$, where $A$ and $B$ have rank $r$ and are the only trainable weights of that layer. GOAL injects these adapters with [peft](https://github.com/huggingface/peft) into the `nn.Linear` layers inside the artisans of a SIMURGH model (see [SIMURGH](../ml/models/simurgh.md)):

- `element_linear`, the element conditioning;
- the radial MLP that produces the tensor-product weights;
- `energy_mlp`, the artisan's scalar energy readout (only when `head_trainable: false`, see below).

These are the invariant maps of the model: their inputs and outputs do not change under rotation, so a low-rank update cannot break equivariance. The equivariant e3nn `Linear` layers and tensor products are not `nn.Linear`, carry no adapter and stay frozen.

Properties that the tests check:

- **Exact no-op at the start.** peft initialises $B$ to zero, so before the first optimiser step the adapted model predicts exactly what the base checkpoint predicts.
- **Equivariance is preserved.** With random non-zero adapters, energies of a rotated molecule agree to 1e-9 and forces rotate with it to 1e-8.
- **Only adapters (and the head) move.** One training step leaves every frozen parameter bit-for-bit unchanged.

With `head_trainable: true` (the default), the task head is trained in full instead of being adapted: the modular head (`head.`), the artisans' energy readouts (`energy_mlp`) and learned `atomic_energies`. This usually converges faster on small datasets, because the per-element reference energies of the new data often differ from the base data.

## Hardware requirements

None beyond what training needs. LoRA fine-tuning is pure PyTorch plus peft and runs on NVIDIA CUDA, AMD ROCm, Apple MPS and CPU. A GPU is recommended for anything but toy datasets. Memory use is close to inference of the base model plus the activations of one training batch, because frozen weights need no optimiser state.

## Installation

`peft` is in every pixi environment, so there is nothing to install:

```bash
pixi run -e cuda python -c "import peft; print(peft.__version__)"
```

With pip, use the `lora` extra, which installs `peft>=0.13`:

```bash
pip install "goal-atomistic[lora]"
```

## Quick start

You need a trained SIMURGH/ARACE run and the new data as ExtXYZ (energy in `atoms.info["energy"]`, forces in `atoms.arrays["forces"]`; other formats work through `data.dataset_type`, see [Loading data](../ml/data/loading.md)).

```bash
pixi run -e cuda goal-train --config-name finetune/simurgh_lora \
    model.backbone.checkpoint_path=/abs/path/logs/simurgh_gmd26/runs/<run>/checkpoints \
    'data.train_paths=[data/dicarboxylic/train.xyz]' \
    'data.val_paths=[data/dicarboxylic/val.xyz]'
```

`goal-finetune` takes the same overrides; its default config is `finetune/simurgh_lora`, so `--config-name` can be left out:

```bash
pixi run -e cuda goal-finetune \
    model.backbone.checkpoint_path=/abs/path/logs/simurgh_gmd26/runs/<run>/checkpoints \
    'data.train_paths=[data/dicarboxylic/train.xyz]' \
    'data.val_paths=[data/dicarboxylic/val.xyz]'
```

`goal-train` is the recommended route; both use `GOALCheckpointManager`, write `.simurgh` archives and support resume. The run lands in `logs/finetune_simurgh_lora/runs/<timestamp>_xyz_simurgh_lora/`. Use the result like any GOAL model:

```python
from ase.io import read
from goal.ml.utils.calculator import GOALCalculator

atoms = read("data/dicarboxylic/val.xyz", index=0)
atoms.calc = GOALCalculator(
    checkpoint_path="logs/finetune_simurgh_lora/runs/<run>/checkpoints", checkpoint="best", device="cuda"
)
print(atoms.get_potential_energy(), atoms.get_forces())
```

> [!WARNING]
> A fine-tuned checkpoint stores the full adapted model, but loading it rebuilds the `simurgh_lora` backbone, and that re-loads the base checkpoint from `model.backbone.checkpoint_path`. Keep the base checkpoint where it was, and give `checkpoint_path` as an **absolute** path, so the fine-tuned model still loads from another working directory or machine.

### From Python

```python
from goal.ml.adapters.simurgh_lora import SIMURGHLoRAAdapter, SIMURGHLoRAConfig

adapter = SIMURGHLoRAAdapter(
    SIMURGHLoRAConfig(checkpoint_path="/abs/path/logs/simurgh_gmd26/runs/<run>/checkpoints", rank=8)
)
print(adapter.trainable_parameters(), "/", adapter.total_parameters())
print(adapter.adapted_layers()[:3])

# your own loop: out = adapter(graph) -> {"energy": ..., "forces": ...}
...

adapter.save("dicarboxylic")                         # writes dicarboxylic.lora (adapter delta only)
adapter = SIMURGHLoRAAdapter.load("dicarboxylic.lora")  # base checkpoint + delta
adapter.merge()                                      # optional: fold adapters into the weights
```

`adapter(graph)` takes a PyG batch of `AtomicGraph` objects and returns the wrapped model's property dict. Forces come from autograd, so run it with gradients enabled.

## API reference

### `SIMURGHLoRAConfig`

A dataclass. The same keys are the `model.backbone` keys of the `simurgh_lora` backbone.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `rank` | `int` | `8` | Rank $r$ of the update. 4–64 is the usual range; must be ≥ 1 |
| `alpha` | `float` | `16.0` | LoRA scale; the update is multiplied by `alpha / rank`. Keep `alpha >= rank`; doubling it doubles the adapters' effective learning rate |
| `dropout` | `float` | `0.0` | Dropout inside the adapters. 0.05–0.1 helps on very small datasets |
| `target_modules` | `list[str]` | `["artisan"]` | Substrings of `nn.Linear` names to adapt. Add `"fragment_interaction"` to also adapt that add-on's radial MLPs |
| `bias` | `str` | `"none"` | `"none"` (adapters only), `"all"` (every bias trainable) or `"lora_only"` (biases of the adapted layers) |
| `checkpoint_path` | `str` or `Path` | `None` (required) | Base checkpoint: `.simurgh` archive, checkpoint directory (frozen source) or bare `.ckpt`. Use an absolute path |
| `checkpoint` | `str` | `"best"` | Selector inside a checkpoint directory: `best`, `last`, `epoch=N` |
| `use_ema` | `bool` | `True` | Start from the base run's EMA weights when it has them |
| `head_trainable` | `bool` | `True` | Train the head (energy readouts, atomic reference energies) in full; head layers then get no adapter |
| `head_patterns` | `list[str]` | `["head.", "energy_mlp", "atomic_energies"]` | Parameter-name fragments that define the head |

### `SIMURGHLoRAAdapter(config, model=None)`

An `nn.Module` holding the frozen model with adapters injected in place (`peft.inject_adapter_in_model`; no `PeftModel` wrapper). Pass `model` to adapt an already built model (a `GOALModule`, anything with `backbone`/`head` attributes, or a monolithic model) instead of loading `config.checkpoint_path`.

| Member | Returns | Meaning |
|---|---|---|
| `forward(graph)` | `dict` | The wrapped model's `{"energy", "forces", ...}` |
| `trainable_parameters()` | `int` | Trainable scalars (adapters, plus head and biases if enabled) |
| `total_parameters()` | `int` | All scalars, frozen included |
| `adapted_layers()` | `list[str]` | Fully qualified names of the layers that carry an adapter |
| `save(path)` | `Path` | Writes the adapter delta; appends `.lora` if missing |
| `SIMURGHLoRAAdapter.load(path, model=None)` | adapter | Rebuilds from the base checkpoint named in the file, then loads the delta |
| `merge()` | `None` | Folds the adapters into the base weights for faster inference. Irreversible for this object; save first |
| `delta_state_dict()` | `dict` | The trainable tensors, as written by `save` |
| `cutoff` | `float` | The base checkpoint's cutoff (when loaded from a path) |
| `base_config` | config | The base run's training config |

Module constants: `LORA_SUFFIX = ".lora"`, `LORA_FORMAT = "goal-simurgh-lora/1"`, `DEFAULT_HEAD_PATTERNS`.

### `SimurghLoRABackbone(cutoff=None, **kwargs)`

The config-driven form, registered lazily as backbone and model `simurgh_lora` in `goal.ml.adapters`. It takes the `SIMURGHLoRAConfig` fields as keyword arguments plus `cutoff`. It is monolithic (`head: null`): it returns the full property dict. `cutoff` (set to `${data.cutoff}` in the config) must equal the base checkpoint's cutoff, otherwise construction fails with a `ValueError` that names the right value. Unknown keywords raise `TypeError`.

## How it works

1. **Load the base model** with the same loader as `GOALCalculator` (`goal.ml.utils.calculator.load_goal_checkpoint`): a `.simurgh` archive, a checkpoint directory (rebuilt from its frozen source) or a bare `.ckpt`, with EMA weights when present and `use_ema` is set.
2. **Pick the target layers**: every `nn.Linear` whose fully qualified name contains one of `target_modules`, minus the head layers when `head_trainable` is on. In an ARACE model the names look like

   ```text
   backbone.rounds.0.artisan_layer.artisans.z1_z6.radial_mlp.net.0
   backbone.rounds.0.artisan_layer.artisans.z1_z6.element_linear
   backbone.rounds.0.artisan_layer.artisans.z1_z6.energy_mlp.0      (head: trained in full by default)
   ```

   No match is an error that lists the model's linear layers.
3. **Inject the adapters** with `peft.inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, ...))`. $B$ starts at zero.
4. **Set trainability**: freeze everything, then unfreeze parameters whose name contains `lora_`, the head parameters (if `head_trainable`) and biases according to `bias`.
5. **Train** with the normal GOAL loop. The optimiser sees only trainable parameters; frozen ones have no gradient, which is why the config sets `strategy.find_unused_parameters: true` for DDP.
6. **Save.** `GOALCheckpointManager` saves the whole `GOALModule` (base weights plus adapters) like any run. `SIMURGHLoRAAdapter.save` instead writes only the delta: a `torch.save` dict with the format tag `goal-simurgh-lora/1`, the config, the adapted layer names and the trainable tensors. Both routes need the base checkpoint at load time.

> [!NOTE]
> **Why the switch is a backbone name and not `strategy.name: lora`.** In GOAL's training config, the top-level `strategy:` key is the distributed-training strategy (`ddp`, `fsdp`, `deepspeed`). LoRA for SIMURGH is therefore selected with `model.backbone.name: simurgh_lora`. Foundation-model fine-tuning has its own `model.backbone.strategy: lora` inside the backbone section; see [MACE](../ml/finetuning/mace.md).

## Configuration reference

The backbone section of `finetune/simurgh_lora.yaml`:

```yaml
model:
  backbone:
    name: simurgh_lora
    checkpoint_path: ???          # .simurgh archive, checkpoint directory or .ckpt (absolute path)
    checkpoint: best              # best | last | epoch=N
    use_ema: true
    cutoff: ${data.cutoff}        # checked against the base checkpoint
    rank: 8
    alpha: 16.0
    dropout: 0.0
    target_modules: [artisan]
    bias: none                    # none | all | lora_only
    head_trainable: true
    # head_patterns: ["head.", "energy_mlp", "atomic_energies"]   # accepted, not in the file
  head: null
```

The other settings in the file that matter for fine-tuning:

| Key | Value | Why |
|---|---|---|
| `data.dataset_type` | `xyz` | Any GOAL dataset type works |
| `data.train_paths`, `data.val_paths` | `???` | Required |
| `data.cutoff` | `8.0` | Must equal the base checkpoint's cutoff (`simurgh_gmd26` uses 8 Å) |
| `data.batch_size` | `4` | Small datasets |
| `trainer.precision` | `"32"` | Match the base run; use `"64"` for a float64 checkpoint |
| `trainer.inference_mode` | `false` | Required: forces need autograd during validation |
| `training.optimizer.lr` | `1.0e-3` | Adapters tolerate a larger learning rate than full fine-tuning |
| `training.optimizer.weight_decay` | `0.0` | |
| `training.ema` | `enabled: true`, `decay: 0.99` | |
| `training.losses` | energy (MAE, weight 1, per atom) and forces (MAE, weight 10) | |
| `checkpoint_manager.top_k.metric` | `val_forces/forces_mae` | Best checkpoint by validation force MAE |
| `strategy.find_unused_parameters` | `true` | Frozen parameters under DDP |

If the base run was trained with fragment labels, copy its `data.compute_fragment_index` and `data.fragment_*` keys into the fine-tuning config, so the new graphs carry the same labels.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ImportError: LoRA fine-tuning needs peft. Install peft: pip install 'peft>=0.13'` | `peft` missing (only possible outside the pixi environments) | `pip install "goal-atomistic[lora]"` |
| `ValueError: No nn.Linear matches target_modules=[...]. Linear layers in this model (first 10): [...]` | `target_modules` names nothing in this model, or every match is a head layer while `head_trainable` is on | Pick substrings from the listed names; the default `["artisan"]` fits SIMURGH/ARACE checkpoints |
| `ValueError: data.cutoff=X differs from the base checkpoint's cutoff Y; set data.cutoff=Y.` | The fine-tuning graphs would be built with another cutoff, which is a different model | Set `data.cutoff=Y` |
| `FileNotFoundError: Base SIMURGH checkpoint not found: ... must stay reachable.` | Wrong `checkpoint_path`, a relative path resolved from another directory, or a fine-tuned model whose base checkpoint was moved | Use an absolute path to the base run's `checkpoints/` directory or `.simurgh` file, and keep it there |
| `TypeError: simurgh_lora got unknown arguments [...]; valid: [...]` | A misspelt key under `model.backbone` | Use one of the listed names |
| `ValueError: bias must be 'none', 'all' or 'lora_only'` / `rank must be >= 1` | Invalid config value | Fix the value |
| `ValueError: ... is not a SIMURGH LoRA file` | `SIMURGHLoRAAdapter.load` was given a file that `save` did not write | Load `.lora` files only; checkpoints go through `GOALCalculator` |
| DDP error about parameters that did not receive gradients | Frozen parameters under DDP | Keep `strategy.find_unused_parameters: true` |
| Autograd errors during validation | `trainer.inference_mode: true` disables gradients that the force computation needs | Keep `trainer.inference_mode: false` |
