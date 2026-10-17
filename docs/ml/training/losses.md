# Loss functions

This page explains how `goal-train` turns the `training.losses` list into the objective being optimised: the per-property losses that are available, the loss functions you can choose for each, weights and the stage curriculum, composite sub-losses, plugging in `torchmetrics` or your own loss, and the diagnostic metrics logged next to the losses. The code is in [`src/goal/ml/training/loss.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/training/loss.py) and [`src/goal/ml/training/metrics.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/training/metrics.py).

## How the loss is assembled

The shipped GMD-26 config contains:

```yaml
training:
  losses:
    - name: energy # property loss, looked up in LOSS_REGISTRY
      fn: mae # loss function applied to that property
      weight: 1.0 # base weight
      normalize_by_n_atoms: true

    - name: forces
      fn: mae
      weight: 1.0
      normalize_by_n_atoms: true

    - name: gate_reg
      fn: mse
      weight: 1.0e-4
```

For each entry, `_build_loss` in [`cli/train.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/cli/train.py) does the following:

1. Looks up `name` in `LOSS_REGISTRY` (`energy`, `forces`, …).
2. Builds the class with `loss_fn=fn`, plus any **other keys of the entry that the class's `__init__` accepts**. Keys the class does not accept are dropped silently. That is why `normalize_by_n_atoms: true` on the `energy` entry above is harmless: `EnergyLoss` has no such argument and never sees it.
3. Wraps the loss in a `WeightedLoss(weight, label=name)`.

All entries are combined in a `CompositeLoss`, which returns a dictionary:

```python
{"energy": w_E * L_E, "forces": w_F * L_F, "gate_reg": w_g * L_g, "total": sum_of_the_above}
```

`total` is what gets back-propagated. Every other key is logged so you can see which term dominates. Values in the dictionary are **already multiplied by their weights**.

$$
\mathcal{L}_\text{total} = \sum_k w_k\,\mathcal{L}_k \;+\; \mathcal{L}_\text{aux}
$$

$\mathcal{L}_\text{aux}$ is a model-supplied regulariser that is not part of `training.losses` (see [below](#model-supplied-auxiliary-loss)).

## Property losses

| `name` | Class | Compares | Notes |
| --- | --- | --- | --- |
| `energy` | `EnergyLoss` | `energy / num_atoms`, predicted vs reference | Always **per atom**, so molecules of different size are on the same scale. |
| `forces` | `ForcesLoss` | `forces` (N × 3) | `normalize_by_n_atoms` (default `true`), see below. |
| `stress` | `StressLoss` | `stress` | Periodic systems with a stress label. Plain reduction. |
| `dipole` | `DipoleLoss` | `dipole` | Needs a head that predicts `dipole`. |
| `charge` | `ChargeLoss` | `total_charge` | Reference defaults to 0 if the batch has no `total_charge`. |
| `gate_reg` | `GateRegLoss` | — | $\sum g^2$ over SIMURGH artisan gate scalars (see [below](#gate_reg-what-it-regularises)). Returns `0` for models without artisans, so the same config works everywhere. `fn` is accepted and ignored (the penalty is always L2). |
| `scalar_property` | `ScalarPropertyLoss` | any key `property_name` | For HOMO/LUMO, gaps and other scalar targets. Options: `property_name` (required), `per_atom` (divide by atom count). |

> [!NOTE]
> The comments in the shipped experiment files list a `newton` loss and show a commented-out `- name: newton` entry. **No `newton` loss is registered**, so uncommenting it raises a registry `KeyError`. The Newton violation is still computed as a *diagnostic metric* (`{split}_forces/newton_violation`) for every run. Referring to `newton` inside `training.stages` is harmless: unknown stage labels only trigger a warning.

### Per-structure force normalisation

`ForcesLoss(normalize_by_n_atoms=True)` (the default) first averages the per-component error inside each structure and then averages over structures:

$$
\mathcal{L}_F = \frac{1}{M}\sum_{m=1}^{M}\;\frac{1}{3N_m}\sum_{i\in m}\sum_{d\in\{x,y,z\}} \ell\!\left(F^\text{pred}_{i,d}, F^\text{ref}_{i,d}\right)
$$

Every molecule contributes equally to the gradient, whatever its size. With `normalize_by_n_atoms: false`, the loss is a flat mean over all $3\sum_m N_m$ components in the batch, so large structures dominate. Keep the default for datasets that mix sizes (GMD-26, OMol25, SPICE). The flat mean is fine for single-molecule benchmarks such as rMD17, where every frame has the same $N$. For `rmse`, the per-structure MSE is computed first, then square-rooted per structure, then averaged.

### Per-sample weights

`EnergyLoss` and `ForcesLoss` read an optional per-graph `weight` from the batch (MACE's `config_weight` convention). Graphs built by `AtomicGraph.from_ase` carry `weight = 1.0`, and HDF5 datasets can store a `weight` per structure. With weights present, the built-in functions compute $\sum_m w_m\,\ell_m / \sum_m w_m$. A dotted-path loss function cannot be decomposed per element, so it falls back to its own unweighted reduction.

## Loss functions (`fn`)

| `fn` | Function | Use it for |
| --- | --- | --- |
| `mse` | `F.mse_loss` | Smooth labels. Penalises large errors quadratically. Default when `fn` is omitted. |
| `mae` / `l1` | `F.l1_loss` | Robust to outliers in DFT labels. The GMD-26 default for both energy and forces. |
| `rmse` | $\sqrt{\text{MSE}}$ | Same units as the target. Often paired with `mse` or `mae` in a composite. |
| `huber` | `F.huber_loss` (δ = 1.0) | Quadratic near zero, linear in the tails. δ is fixed at 1.0 in the units of the target (eV/Å for forces, eV/atom for energy). |
| `smooth_l1` | `F.smooth_l1_loss` (β = 1.0) | Same shape as Huber, rescaled. |
| `pkg.module.fn` | any importable callable `(pred, target) -> scalar` | `torchmetrics`, or your own function (see below) |

> [!TIP]
> **Choosing between them.** Forces from DFT single points often have a few badly converged frames. With MSE those frames dominate the gradient, while MAE or Huber limit their influence. The cost of MAE is a gradient that does not shrink near the optimum, which a decaying LR (cosine) compensates for. Report RMSE as well either way: it is what most papers quote, and `{split}_forces/forces_rmse` is always logged.

### Relative weights

Energy (eV/atom) and force (eV/Å) errors live on different scales, and a model has $3N$ force labels for every energy label. Common choices:

- **Force-heavy** (most MLIP work, MACE-style): forces at 10–100× the energy weight early in training. Good forces give good MD.
- **Balanced late training**: raise the energy weight towards the end to tighten energies (the `stages` mechanism below does this without restarting).

## Composite losses per property

Several loss functions can supervise the same property, each with its own weight and its own logged curve:

```yaml
training:
  losses:
    - name: energy
      fn: mse
      weight: 1.0
    - name: forces
      fn: # a list instead of a string
        - name: mae
          weight: 1.0
        - name: rmse
          weight: 0.5
```

This builds two `ForcesLoss` objects labelled `forces_mae` and `forces_rmse`, both in group `forces`. Because the group has two members, `CompositeLoss` also emits their sum under `forces`. The logger then shows:

| Logged key (validation) | Value |
| --- | --- |
| `val_energy/energy_epoch` | 1.0 × energy MSE |
| `val_forces/forces_mae_epoch` | 1.0 × force MAE |
| `val_forces/forces_rmse_epoch` | 0.5 × force RMSE |
| `val_forces/forces_epoch` | sum of the two above |
| `val/total_epoch` | everything |

Sub-loss labels start with the property name, so they land in the same logger section as their group total (see [Callbacks & logging](callbacks-logging.md#logger-sections)).

> [!WARNING]
> Two limitations of composite entries in the current builder:
>
> - **Extra keys are not forwarded to the sub-losses.** Each sub-loss is built as `loss_cls(loss_fn=sub_name)` only. `normalize_by_n_atoms` therefore always takes its default (`true`) inside a composite, and `scalar_property` cannot be used as a composite because its required `property_name` never arrives.
> - **Stage overrides address labels, not groups.** A stage with `loss_weights: {forces: 0.0}` does **not** switch off `forces_mae` and `forces_rmse`. Name them individually, for example `loss_weights: {forces_mae: 0.0, forces_rmse: 0.0}`. GOAL warns about `forces` being an unknown label.

## Custom and `torchmetrics` loss functions

Any dotted import path to a callable taking `(prediction, target)` and returning a scalar tensor can be used as `fn`:

```yaml
training:
  losses:
    - name: forces
      fn:
        - name: mae
          weight: 1.0
        - name: torchmetrics.functional.mean_squared_error
          weight: 0.5 # logged as forces_mean_squared_error
```

```bash
pip install "goal-atomistic[torchmetrics]"
```

The label suffix is the last part of the path. Your own function works in the same way, provided it is importable when the loss is built:

```python
# my_project/losses.py
import torch


def log_cosh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth, outlier-robust alternative to Huber with no scale parameter."""
    return torch.log(torch.cosh(pred - target)).mean()
```

```yaml
    - name: forces
      fn: my_project.losses.log_cosh
      weight: 1.0
```

Dotted-path functions skip the per-sample-weight path. With `normalize_by_n_atoms: true` they are applied to the whole batch at once (a flat mean), because GOAL cannot decompose them per structure.

## A new property loss (custom class)

To supervise a property no built-in loss covers (atomic charges, per-atom spins, Hessians, …), register a new class in `LOSS_REGISTRY`:

```python
# my_project/losses.py
import torch
import torch.nn as nn

from goal.ml.registry import LOSS_REGISTRY
from goal.ml.training.loss import resolve_loss_fn


@LOSS_REGISTRY.register("partial_charges")
class PartialChargeLoss(nn.Module):
    """Per-atom partial charges, plus a soft total-charge constraint."""

    def __init__(self, loss_fn: str = "mse", neutrality_weight: float = 0.0) -> None:
        super().__init__()
        self.loss_fn = resolve_loss_fn(loss_fn)  # gives you mse/mae/huber/dotted paths for free
        self.neutrality_weight = neutrality_weight

    def forward(self, pred: dict[str, torch.Tensor], target) -> torch.Tensor:
        q_pred, q_ref = pred["charges"], target["charges"]
        loss = self.loss_fn(q_pred, q_ref)
        if self.neutrality_weight > 0:
            from torch_geometric.utils import scatter

            q_tot = scatter(q_pred, target.batch, dim=0, reduce="sum")
            loss = loss + self.neutrality_weight * q_tot.pow(2).mean()
        return loss
```

The contract is the following:

- `__init__` must accept `loss_fn` (the builder always passes it). Other arguments are filled from the matching keys of the config entry.
- `forward(pred, target)` returns one scalar. `pred` is the model's output dictionary, and `target` is the batched `AtomicGraph`, so labels are available as attributes or by key, and `target.batch` maps atoms to graphs.

The class has to be registered *before* `goal-train` builds the loss. The CLI does not scan for plugins, so launch through a small wrapper that imports your module first:

```python
# train_with_charges.py
import my_project.losses  # noqa: F401  (registers "partial_charges")
from goal.ml.cli.train import train_ml

if __name__ == "__main__":
    train_ml()
```

```bash
pixi run -e cuda python train_with_charges.py 'training.losses=[{name: energy, fn: mae, weight: 1.0}, {name: forces, fn: mae, weight: 1.0}, {name: partial_charges, fn: mse, weight: 10.0, neutrality_weight: 1.0}]'
```

The label is the entry's `name`. `partial_charges` is in neither `METRIC_SECTIONS` nor a prefix rule, so it is logged under the plain `{split}/` section.

## The stage curriculum

`training.stages` changes loss weights over time without a restart:

```yaml
training:
  stages:
    - name: "energy_only"
      epochs: 5
      loss_weights: { energy: 1.0, forces: 0.0, gate_reg: 1.0e-4 }
    - name: "energy_forces"
      epochs: null # open-ended: runs to the end
      loss_weights: { energy: 1.0, forces: 1.0, gate_reg: 1.0e-4 }
  strict_stage_labels: false # true: unknown labels raise instead of warn
```

Stage $i$ covers epochs $[s_i, s_i + \text{epochs}_i)$, where $s_0 = 0$. A stage with `epochs: null` captures everything after it, and once the last finite stage is over, training stays in the last stage. On every train and validation step, `GOALModule` patches `WeightedLoss.weight` for each label in the active stage's `loss_weights`. Labels without an override keep their base weight. Every change is logged as `[stage] epoch=5 → 'energy_forces' active_weights={…}`, and the progress bar shows `[energy_forces 3/∞] Epoch 7/1499`.

Why the default config starts with energy only: early in training the energy surface is still wrong, so its gradient, the force, is wrong too, and force supervision adds noise that slows down learning the overall energy scale. A few energy-only epochs settle the atomic baselines and the scale. A three-phase variant (raise the energy weight at the end to tighten energies) is shown, commented out, in the default config.

Points to know:

- **Validation uses the stage weights too**, so `val/total` is comparable with `train/total` epoch by epoch. Changing stages therefore shows up as a jump in `val/total`. Compare the weight-independent metrics (`val_forces/forces_mae`, `val_energy/energy_mae_per_atom`) across stage boundaries instead.
- **`trainer.test()` ignores stages** and reports every component at its base weight.
- **Early stopping or top-k on `val/total`** is affected by stage jumps for the same reason. The shipped configs select on `val_forces/forces_mae`.
- Labels in `loss_weights` that are not in `training.losses` produce a warning, or a `ValueError` with `strict_stage_labels: true`. Enable strict mode to catch typos such as `force` vs `forces`.

### `gate_reg`: what it regularises

Every SIMURGH artisan (one per element pair and round) multiplies its pair energy by a learnable scalar gate $P_{AB}$, initialised to 1. `gate_reg` adds $w_g \sum_{AB} P_{AB}^2$ to the loss, which keeps a single pair channel from absorbing the interaction energy through an ever-growing gate. `GOALModule.forward` collects the gates as `predictions["gate_values"]` from every artisan class — `PotentialArtisan` / `RarePotentialArtisan` (legacy `simurgh`, `simurgh_monolithic`) and the ARACE artisans (`simurgh_arace`, `monolithic_arace`, `nf_simurgh`) — on the modular and the monolithic path. Models without artisans (HyperSpec, InvariantGNN, SIMURGH Simplified, foundation models) contribute nothing.

With gates at 1.0 the term starts at the number of gates times $w_g$ (12 gates × 1e-4 for a two-round H/C/O ARACE), and its gradient pulls each gate by $2 w_g P_{AB}$.

> [!WARNING]
> **Before October 2026 `gate_reg` was always 0.** The collector looked for `artisan_bank.experts`, an attribute that the artisan rename turned into `.artisans`, and it only ran for modular backbones. Runs trained before the fix had the regulariser effectively switched off, whatever weight their config gave it. Re-running such a config now applies it. The ablation study configs do not declare `gate_reg` at all, so they are unaffected.

## Model-supplied auxiliary loss

Some models add their own regulariser to their outputs. Currently this is the ARACE adaptive depth gate's sparsity penalty $\lambda\,\overline{g}$. If the prediction dictionary contains `aux_loss`, `GOALModule` adds it to `total` and logs it as `{split}_gate/aux_loss`. Its weight is set where the term is defined (`model.backbone.adaptive_gate.aux_loss_weight`), **not** in `training.losses`. Do not declare it there.

## Metrics logged alongside the loss

Loss values are weighted sums and change when the weights change. `mlip_metrics(predictions, batch)` computes physically meaningful numbers on every train, validation and test step, independently of the loss configuration:

| Metric | Definition | Section |
| --- | --- | --- |
| `energy_mae_per_atom` | $\langle \lvert E^\text{pred}_m - E^\text{ref}_m\rvert / N_m\rangle_m$ | `{split}_energy/` |
| `energy_rmse_per_atom` | $\sqrt{\langle ((E^\text{pred}_m - E^\text{ref}_m)/N_m)^2\rangle_m}$ | `{split}_energy/` |
| `forces_mae` | mean absolute error over all force components | `{split}_forces/` |
| `forces_rmse` | RMSE over all force components | `{split}_forces/` |
| `forces_cosine_similarity` | mean per-atom cosine between predicted and reference force vectors, skipping atoms whose reference force is exactly zero | `{split}_forces/` |
| `forces_magnitude_mae` | mean $\lvert\,\lVert F^\text{pred}_i\rVert - \lVert F^\text{ref}_i\rVert\,\rvert$ | `{split}_forces/` |
| `newton_violation_{x,y,z}` | $\langle \lvert \sum_{i\in m} F^\text{pred}_{i,d}\rvert \rangle_m$ per axis | `{split}_forces/` |
| `newton_violation` | sum of the three axes | `{split}_forces/` |

Some notes on reading them:

- The force metrics are **flat means over atoms**, as published MLIP numbers are, even when the loss is normalised per structure.
- **Cosine similarity and magnitude MAE separate direction errors from size errors.** A model with a good cosine but poor magnitude has the right shape of the potential energy surface at the wrong stiffness.
- **Newton violation** must be ≈ 0 for autograd-derived forces on isolated molecules, because translation invariance makes $\sum_i \mathbf{F}_i = 0$ exact. A clearly non-zero value points to a direct (non-conservative) force head. A strongly anisotropic one (for example $z \gg x, y$) points to something direction-dependent: a surface normal, a periodic-boundary setting, or a cell vector.
- `energy_mae_per_atom`, `forces_cosine_similarity` and `newton_violation` appear on the progress bar (`PROG_BAR_METRICS`). Everything else goes only to the loggers.

The keys are listed in full in [Callbacks & logging](callbacks-logging.md#logger-sections).

## Overriding losses from the command line

```bash
# Change one field of an existing entry (list index addressing)
pixi run -e cuda goal-train training.losses.1.fn=huber training.losses.1.weight=10.0

# Replace the whole list
pixi run -e cuda goal-train 'training.losses=[{name: energy, fn: mae, weight: 1.0}, {name: forces, fn: huber, weight: 10.0}]'

# Drop the curriculum and use base weights throughout
pixi run -e cuda goal-train training.stages=null
```

If you replace the list and drop `gate_reg`, also remove it from `training.stages` (or set `training.stages=null`) to avoid the unknown-label warning.
