# Mini trainer

`MiniTrainer` is a small training loop for notebooks. You give it a model, a loss, an optimiser and `DataLoader`s, and it handles epochs, validation, early stopping, best-weights tracking, learning-rate scheduling and loss curves. It is independent of the Lightning/Hydra pipeline (no configs, no callbacks, no log directories), so you can iterate on a downstream model in a few cells.

```python
from goal.ml.utils.mini_trainer import MiniTrainer, TrainingHistory, default_step, graph_step
```

## When to use it, and when not

Use it for quick, single-GPU experiments where the Lightning machinery is overhead:

- training a regression head on **cached descriptors** from a frozen backbone (see [Feature extraction](feature-extraction.md)). This is the main use case: the backbone runs once, and each epoch afterwards takes milliseconds.
- probing what a representation encodes (linear probes, small MLPs, comparing layers);
- a short fine-tune of an existing `GOALModule` on a handful of structures, interactively.

Use `goal-train` ([Training a model](../training/training.md)) for anything you want to reproduce, scale or publish. `MiniTrainer` has no multi-GPU support, mixed precision, EMA, checkpoint files, loss-stage curricula, MLIP metrics or loggers. For a middle ground with Fabric and your own loop, see [Custom training loops](../training/custom-loops.md).

## Basic usage: tensors in, scalar out

The default step expects `(x, y)` batches, which is what `TensorDataset` yields. It computes `loss_fn(model(x).squeeze(-1), y)`.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from goal.ml.utils.mini_trainer import MiniTrainer

# X: (M, d) descriptors, y: (M,) targets, e.g. from the feature-extraction recipes
X, y = torch.randn(1000, 256), torch.randn(1000)
perm = torch.randperm(len(X))
tr, va, te = perm[:700], perm[700:850], perm[850:]
train_loader = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[va], y[va]), batch_size=128)
test_loader = DataLoader(TensorDataset(X[te], y[te]), batch_size=128)

head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)

trainer = MiniTrainer(
    model=head,
    loss_fn=nn.MSELoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    device="auto",          # cuda > mps > cpu
    grad_clip=10.0,
)
history = trainer.fit(
    train_loader,
    val_loader=val_loader,
    epochs=200,
    early_stopping_patience=20,
    checkpoint_best=True,
)

trainer.load_best()                      # restore the best-validation weights
preds, targets = trainer.predict(test_loader)
print(f"test MAE {(preds - targets).abs().mean():.4f}")
history.plot()                           # log-scale train/val curves, best epoch marked
```

Each epoch prints one line:

```text
Epoch  12/200  train_loss=0.041233  val_loss=0.052871  lr=1.00e-03  (0.1s)
```

## API

### `MiniTrainer(...)`

| Argument | Default | Meaning |
| --- | --- | --- |
| `model` | — | any `nn.Module`. Moved to `device` at construction |
| `loss_fn` | — | an `nn.Module` loss (`nn.MSELoss()`, a GOAL `CompositeLoss`, …). Also moved to `device` |
| `optimizer` | — | an optimiser already built over the parameters you want to train |
| `scheduler` | `None` | stepped once per epoch. `ReduceLROnPlateau` gets the validation loss (or the training loss without a validation loader) |
| `device` | `"auto"` | `"auto"` picks CUDA, then MPS, then CPU. Also accepts any device string or `torch.device` |
| `step_fn` | `None` → `default_step` | how one batch becomes a loss (see [Custom steps](#custom-steps)) |
| `grad_clip` | `None` | max global gradient norm (`clip_grad_norm_`). `None` disables clipping |
| `enable_progress` | `True` | per-epoch `tqdm.auto` bars for train and val (skipped silently if tqdm is missing) |

### `fit(train_loader, val_loader=None, epochs=10, early_stopping_patience=None, checkpoint_best=True, verbose=True)`

Runs the loop and returns a `TrainingHistory`:

- **Validation** runs after every training epoch when `val_loader` is given.
- **Best weights.** With `checkpoint_best=True` an in-memory `deepcopy` of `model.state_dict()` is kept whenever the validation loss improves. `load_best()` restores it and raises `RuntimeError` if none was kept. The best loss persists across `fit` calls on the same trainer, so a second `fit` only replaces the snapshot if it beats the first.
- **Early stopping** triggers after `early_stopping_patience` epochs without a validation improvement. It requires a `val_loader`.
- **Logged losses** are the mean of per-batch losses, without weighting by batch size.

### `predict(loader)`

Returns `(predictions, targets)` concatenated over the loader, on the CPU, in eval mode and without gradients. It always unpacks `(x, y)` batches and calls `model(x).squeeze(-1)`, so it does **not** work with graph batches. For those, write a short loop (see below).

### `TrainingHistory`

| Member | Contents |
| --- | --- |
| `train_loss`, `val_loss` | per-epoch means (`val_loss` is empty without a validation loader) |
| `lr` | learning rate of the first parameter group at each epoch, before the scheduler step |
| `epoch_times` | wall-clock seconds per epoch |
| `best_val_loss`, `best_epoch` | lowest validation loss and its 0-based epoch index (or `None`) |
| `extra` | a free-form `{name: list}` dict. `MiniTrainer` itself never fills it |
| `plot(figsize=(10, 4))` | matplotlib figure of the curves on a log y-axis |

## Custom steps

A step function has the signature `step_fn(batch, model, loss_fn, device) -> dict[str, Tensor]`. It must return a dict with a scalar `"loss"` or `"total"` key. That key is back-propagated during training and averaged for reporting. Two are built in:

- `default_step`: `(x, y)` tensors, `loss_fn(model(x).squeeze(-1), y)`.
- `graph_step`: moves an `AtomicGraph` / PyG `Batch` to the device, calls `predictions = model(batch)` (a dict, like `GOALModule.forward`) and returns `loss_fn(predictions, batch)`. With a `CompositeLoss` that dict carries `"total"` plus one entry per component.

You can write your own to weight samples, add a regulariser or train multi-task heads:

```python
def weighted_step(batch, model, loss_fn, device):
    x, y, w = (t.to(device) for t in batch)                 # TensorDataset(X, y, weights)
    per_sample = (model(x).squeeze(-1) - y) ** 2
    return {"loss": (w * per_sample).sum() / w.sum()}

trainer = MiniTrainer(head, nn.MSELoss(), optimizer, step_fn=weighted_step)
```

## Training on `AtomicGraph` batches

You can drive a full backbone and head, for example a loaded `GOALModule`, with `graph_step` and a `CompositeLoss`. Graphs batch with PyG's loader.

> [!WARNING]
> `MiniTrainer` runs validation under `torch.no_grad()`. Autograd force heads (`EnergyForcesHead`, SIMURGH's default, and `DualForcesHead` in autograd/hybrid mode) compute $\mathbf F = -\partial E / \partial \mathbf r$ and fail there with `RuntimeError: element 0 of tensors does not require grad`. Wrap the step in `torch.enable_grad()`, as below, when you pass a `val_loader`.

```python
import numpy as np
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from goal.ml.data.graph import AtomicGraph
from goal.ml.training.loss import CompositeLoss, EnergyLoss, ForcesLoss, WeightedLoss
from goal.ml.utils.calculator import GOALCalculator
from goal.ml.utils.mini_trainer import MiniTrainer, graph_step

# A trained model to adapt (any GOALModule works; see "Checkpoints & archives").
calc = GOALCalculator(checkpoint_path="models/simurgh_gmd26.simurgh")
model = calc.module

# Build labelled graphs with the model's own cutoff, dtype and fragment settings.
data = model.config.data
frag_keys = ("compute_fragment_index", "fragment_covalent_cutoff", "fragment_scheme",
             "fragment_charge", "fragment_on_failure", "fragment_smarts", "fragment_keep_groups")
frag_kwargs = {k: data[k] for k in frag_keys if data.get(k) is not None}

def to_graph(atoms):
    return AtomicGraph.from_ase(
        atoms, cutoff=calc._cutoff, dtype=calc.dtype,
        energy=atoms.get_potential_energy(), forces=atoms.get_forces(), **frag_kwargs,
    )

frames = [to_graph(a) for a in read("new_data.extxyz", index=":")]
rng = np.random.default_rng(0)
idx = rng.permutation(len(frames))
n_val = max(1, len(frames) // 10)
train_loader = DataLoader([frames[i] for i in idx[n_val:]], batch_size=8, shuffle=True)
val_loader = DataLoader([frames[i] for i in idx[:n_val]], batch_size=8)

loss = CompositeLoss([
    WeightedLoss(EnergyLoss("mse"), weight=1.0, label="energy"),
    WeightedLoss(ForcesLoss("mse"), weight=10.0, label="forces"),
])

def graph_step_with_grad(batch, model, loss_fn, device):
    with torch.enable_grad():                   # forces need the autograd tape, also in validation
        return graph_step(batch, model, loss_fn, device)

trainer = MiniTrainer(
    model=model,
    loss_fn=loss,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5),
    step_fn=graph_step_with_grad,
    grad_clip=1.0,
)
history = trainer.fit(train_loader, val_loader=val_loader, epochs=50, early_stopping_patience=10)
trainer.load_best()
```

Because `predict` only handles `(x, y)` batches, evaluate graph models with a loop:

```python
model.eval()
errors = []
for batch in val_loader:
    batch = batch.to(trainer.device)
    with torch.enable_grad():
        out = model(batch)
    errors.append((out["forces"].detach() - batch.forces).abs().mean().item())
print(f"val force MAE {np.mean(errors):.4f} eV/Å")
```

> [!NOTE]
> `model(batch)` on a `GOALModule` uses the raw weights. Its EMA shadow, if any, is neither updated nor used by `MiniTrainer`. After this kind of fine-tune you have a modified in-memory module. Wrap it as `GOALCalculator(module=model, cutoff=calc._cutoff)` to use it in ASE. There is no `.simurgh` export for it, because archives are produced only by `goal-train`'s checkpoint manager.

## Notebook

[`notebooks/mini_trainer_demo.ipynb`](https://github.com/Nourollah/GOAL/blob/main/notebooks/mini_trainer_demo.ipynb) walks through the whole descriptor workflow: building `AtomicGraph`s, extracting features (with a synthetic-feature fallback if `mace-torch` is not installed), `TensorDataset` splits, an MLP head trained with `ReduceLROnPlateau`, early stopping and gradient clipping, loss curves, `load_best()`, test-set MAE/RMSE and a parity plot. Its optional MACE cell uses the wrapper API affected by the issues listed in [Feature extraction](feature-extraction.md). The `MiniTrainer` cells use the API exactly as described on this page.

## See also

- [Feature extraction](feature-extraction.md): where cached descriptors come from.
- [Custom training loops](../training/custom-loops.md): Fabric-based loops with multi-GPU support.
- [Loss functions](../training/losses.md): `CompositeLoss`, `WeightedLoss` and the registered losses.
