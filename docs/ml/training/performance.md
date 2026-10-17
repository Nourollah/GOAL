# Performance engineering

This page covers the settings that change how fast, and how stably, a GOAL model trains: TF32 matmuls, cuDNN autotuning, `torch.compile`, mixed precision, gradient accumulation, gradient clipping, EMA and SWA weight averaging, and Lightning's sanity validation. For each one it lists the config key, what the setting actually changes, and when to use it.

## Summary

| Setting | Config key | Shipped default | Typical change |
| --- | --- | --- | --- |
| TF32 matmul | `training.performance.float32_matmul_precision` | `highest` | `high` on Ampere+ |
| cuDNN autotune | `training.performance.cudnn_benchmark` | `false` | `true` for long runs |
| cuDNN determinism | `training.performance.cudnn_deterministic` | `false` | `true` only for bit-reproducibility checks |
| `torch.compile` | `training.compile_model`, `training.compile.{mode,fullgraph,dynamic}` | `false` | experiment per model |
| Precision | `trainer.precision` | `"32"` | `bf16-mixed` (with care), `64-true` for tests |
| Gradient accumulation | `trainer.accumulate_grad_batches` | `1` | `2`–`8` when memory limits batch size |
| Gradient clipping | `training.gradient_clip` | `10.0` | `1.0` if the early loss explodes |
| EMA | `training.ema.{enabled,decay}` | `true`, `0.99` | `0.999` for long runs |
| SWA | `callbacks.swa` (commented out) | off | alternative to EMA |
| Sanity validation | `trainer.num_sanity_val_steps` | `2` | `0` for faster restarts |

## TF32 matmul precision

```yaml
training:
  performance:
    float32_matmul_precision: highest # "highest" | "high" | "medium"
```

`goal-train` passes this value to `torch.set_float32_matmul_precision` before anything else runs.

- `highest`: true IEEE float32 matmuls.
- `high`: float32 matmuls may use TF32 tensor cores (10-bit mantissa inputs, float32 accumulation). On A100/H100/RTX 30xx–40xx/RTX 6000 Ada this is typically several times faster for the dense layers.
- `medium`: allows bfloat16-level internal precision as well.

If the key is **absent**, GOAL picks `high` on GPUs with compute capability ≥ 8.0 (Ampere and newer) and `highest` elsewhere. The shipped configs set `highest` explicitly, so you have to opt in:

```bash
pixi run -e cuda goal-train training.performance.float32_matmul_precision=high
```

**Should you?** For energy and force training in float32, TF32 matmuls are standard practice and the effect on accuracy is far below the noise of DFT labels. Two cases call for keeping `highest`:

- Force-conservation diagnostics. `val_forces/newton_violation` measures numerical noise for autograd forces, and TF32 raises that noise floor slightly.
- Comparing against a published number that was obtained in full fp32.

## cuDNN benchmark and determinism

```yaml
training:
  performance:
    cudnn_benchmark: false
    cudnn_deterministic: false
```

These map to `torch.backends.cudnn.benchmark` and `torch.backends.cudnn.deterministic`, and are applied only when CUDA is available. `benchmark=true` times several kernels for each new input shape and caches the fastest one. Atomistic batches change shape on every step (different atom and edge counts), and cuDNN is used mainly for convolutions, which GOAL's message-passing models barely use, so the gain is small. Enable it for long runs and measure. `cudnn_deterministic=true` together with `trainer.deterministic=true` is for checking bit-for-bit reproducibility. It is slower, and some scatter operations on GPU stay non-deterministic regardless.

## `torch.compile`

```yaml
training:
  compile_model: false
  compile:
    mode: "default" # "default" | "reduce-overhead" | "max-autotune"
    fullgraph: false # true = fail instead of falling back on graph breaks
    dynamic: null # null | true | false
```

```bash
pixi run -e cuda goal-train training.compile_model=true training.compile.dynamic=true
```

With `compile_model: true`, `GOALModule.configure_model()` replaces `self.backbone` with `torch.compile(self.backbone, mode=…, fullgraph=…, dynamic=…)`. Only the **backbone** is compiled, and the head and loss stay eager. Lightning calls `configure_model` once, before the optimizer is built, so nothing recompiles between epochs.

What to expect with GNN potentials:

- **Shapes vary from batch to batch.** With `dynamic: null`, PyTorch first specialises on the first shape and then recompiles with dynamic shapes when a second one appears. `dynamic: true` skips the specialised first compile. Without it, you may hit the recompilation limit, after which PyTorch falls back to eager mode for the remaining shapes.
- **Autograd forces need double backward.** Forces come from $-\partial E/\partial \mathbf{r}$ with `create_graph=True`, and training then differentiates that again. Inductor support for double backward through compiled regions depends on the PyTorch version and can fall back to eager or raise errors. Start with `mode: default`, `fullgraph: false`.
- **Measure.** Compare epoch times over 5–10 epochs, not the first one: the first epoch includes compilation, which can take minutes.
- `mode: reduce-overhead` uses CUDA graphs, which need static shapes, so it rarely suits variable-size molecular batches. `max-autotune` spends a long time autotuning and is only worth it for very long production runs.

## Mixed precision

```bash
pixi run -e cuda goal-train trainer.precision=32            # default ("32" / "32-true")
pixi run -e cuda goal-train trainer.precision=bf16-mixed    # bfloat16 autocast, Ampere+
pixi run -e cuda goal-train trainer.precision=16-mixed      # float16 autocast + loss scaling
pixi run -e cuda goal-train trainer.precision=64-true       # double precision everywhere
```

`trainer.precision` is passed straight to Lightning.

- **`bf16-mixed`** runs matmuls in bfloat16 and keeps the master weights in fp32. It roughly halves activation memory, so it is one of the first things to try when large structures hit out-of-memory errors. The cost is accuracy: bfloat16 has an 8-bit mantissa, and forces are *derivatives* of the energy, so their relative error is amplified. Before adopting it for a production model, compare `val_forces/forces_mae` and `val_forces/newton_violation` against an fp32 run of the same length.
- **`16-mixed`** has more mantissa bits but a narrow exponent range, so it needs loss scaling. Energies of whole molecules in eV (`atomic_energies` baselines run to thousands of eV per atom) are exactly the kind of values that overflow it. Prefer `bf16-mixed`.
- **`64-true`** is the right setting for unit tests, finite-difference force checks, and small benchmark systems where you want to remove numerical noise from the comparison. It is several times slower on consumer GPUs. The EMA shadow follows the parameters' dtype automatically.

On MPS, keep `"32"`, because float64 is not supported there.

## Gradient accumulation

```bash
pixi run -e cuda goal-train trainer.accumulate_grad_batches=4
```

This sums gradients over 4 batches before each optimizer step, so the effective batch size is `data.batch_size × 4` (multiplied further by the number of GPUs, see [Distributed training](distributed.md#batch-size-and-learning-rate-across-gpus)). Use it when memory, not throughput, limits the batch size, which is typical for large structures or wide irreps.

To ramp accumulation up over training, enable Lightning's scheduler in the `callbacks:` block:

```yaml
callbacks:
  grad_accumulation:
    _target_: lightning.pytorch.callbacks.GradientAccumulationScheduler
    scheduling: { 0: 1, 10: 2, 20: 4 } # epoch: accumulation factor
```

Two interactions to know about:

- `GOALModule.on_train_batch_end` updates the EMA after **every batch**, not every optimizer step. With accumulation factor $k$, the EMA is updated $k$ times between parameter changes, so its effective half-life in optimizer steps shrinks by a factor of $k$. Raise `training.ema.decay` to compensate. For $k = 4$, $0.99 \to 0.99^{1/4} \approx 0.9975$.
- Epoch-based LR schedulers are unaffected, because they step per epoch.

## Gradient clipping

```yaml
training:
  gradient_clip: 10.0 # clip-by-norm value; 0 disables

trainer:
  gradient_clip_val: null # must stay null
```

Clipping is applied by `GOALModule.on_before_optimizer_step`, using Lightning's `clip_gradients` with the global 2-norm. Lightning refuses to have both its own `trainer.gradient_clip_val` and a manual clip, so keep the trainer key `null`. `10.0` is a loose safety net for equivariant networks. Drop it to `1.0` if early training shows loss spikes.

> [!WARNING]
> Under FSDP, norm clipping is rejected by Lightning, so set `training.gradient_clip=0`. Under DeepSpeed it is silently skipped. Move it to `trainer.gradient_clip_val` there and set `training.gradient_clip=0`. See [Distributed training](distributed.md).

## Exponential moving average (EMA)

```yaml
training:
  ema:
    enabled: true
    decay: 0.99
```

`EMAWrapper` ([`src/goal/ml/training/ema.py`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/training/ema.py)) keeps a shadow copy $\bar\theta$ of every parameter and updates it after each training batch:

$$
\bar\theta \leftarrow \bar\theta + (1 - \beta)\,(\theta - \bar\theta), \qquad \beta = \texttt{decay}
$$

The shadow's memory has a half-life of $\ln 2 / (1-\beta)$ updates: about 69 batches for $\beta = 0.99$ and about 693 for $0.999$. Pick the decay so that the half-life is a small fraction of an epoch for short runs, and a few epochs' worth of steps for long ones.

Where the EMA weights are used:

- **Validation and test.** `validation_step` and `test_step` run the forward pass inside `ema.average_parameters()`, which swaps the shadow weights in and restores the raw ones afterwards. Every `val_*` and `test_*` metric, and therefore top-k checkpoint selection and early stopping, is measured on the **EMA** weights.
- **Checkpoints.** The `.ckpt` `state_dict` contains the **raw** weights, and the shadow is stored next to it under `checkpoint["goal_ema"]`. It is restored on resume and on any Lightning `trainer.test(ckpt_path=…)` / `trainer.validate(ckpt_path=…)` run.

> [!WARNING]
> `GOALCalculator` (the ASE calculator, also used to load `.simurgh` archives) builds the model with `load_state_dict(checkpoint["state_dict"])`, which is the **raw** weights. It does not apply `goal_ema`. A model loaded for MD can therefore be slightly different from the one whose validation error picked the checkpoint. The difference is usually small late in training, but it can matter for short runs with a strong decay. If you need the EMA weights at inference time, copy the shadow into the parameters yourself (the order of `checkpoint["goal_ema"]["shadow"]` matches `module.parameters()`).

**Lightning's EMA instead of GOAL's.** Lightning 2.6 ships `EMAWeightAveraging`. To use it, disable the built-in EMA so that the weights are not averaged twice:

```yaml
training:
  ema:
    enabled: false

callbacks:
  ema_weight_averaging:
    _target_: lightning.pytorch.callbacks.EMAWeightAveraging
    decay: 0.999
    # update_every_n_steps: 1
    # update_starting_at_epoch: 5   # skip the noisy start
```

Its `update_starting_at_*` options are useful together with a stage curriculum. For example, they let you start averaging only once forces are switched on.

## Stochastic weight averaging (SWA)

SWA averages the weights uniformly over the last part of training, usually with a constant or cyclical LR, instead of exponentially over all of it. It is an alternative to EMA, so do not combine the two:

```yaml
training:
  ema:
    enabled: false

callbacks:
  swa:
    _target_: lightning.pytorch.callbacks.StochasticWeightAveraging
    swa_lrs: 1.0e-4
    swa_epoch_start: 0.75 # fraction of max_epochs (or an int epoch)
    annealing_epochs: 10
    annealing_strategy: cos
```

Lightning's SWA swaps in its own LR schedule from `swa_epoch_start` onwards, so the cosine schedule from `training.optimizer` only applies before that point.

## Sanity validation

```yaml
trainer:
  num_sanity_val_steps: 2 # 0 = skip, -1 = the whole validation set
```

Before the first training epoch, Lightning runs this many validation batches. This catches data-loading, metric and force-head errors in seconds instead of at the end of epoch 1. Set it to `0` when restarting a known-good run many times, and to `-1` when you change the validation data or metrics.

## Other trainer flags that matter

```yaml
trainer:
  inference_mode: false # MUST stay false
  detect_anomaly: false # true raises at the first NaN/Inf; very slow
  deterministic: false
  log_every_n_steps: 1
```

- **`inference_mode: false` is required.** Validation and test wrap the forward pass in `torch.enable_grad()` so that autograd force heads can compute $-\partial E/\partial\mathbf{r}$. Under `torch.inference_mode`, which is Lightning's default, that is impossible, and validation fails.
- **`log_every_n_steps: 1`** logs every training step to every logger. With many gate curves (one per element pair per round) and W&B, raising it to `10`–`50` noticeably reduces overhead.
- **Data loading.** `data.num_workers: 0` is the shipped default because worker processes are fragile in some Python 3.14 environments. If GPU utilisation is low, try `data.num_workers=4 data.persistent_workers=true data.pin_memory=true data.prefetch_factor=2`.

## Profiling

To find out where time goes:

```bash
pixi run -e cuda goal-train --config-dir src/goal/configs +debug=profiler trainer.accelerator=gpu   # Lightning "simple" profiler, 1 epoch
pixi run -e cuda goal-train trainer.max_epochs=2 trainer.profiler=pytorch                   # PyTorch profiler traces
```

To measure throughput continuously, use Lightning's `ThroughputMonitor`. It needs a `batch_size_fn`, which has to be a Python callable, so point Hydra at a function in your own module:

```python
# my_project/throughput.py
def num_graphs(batch):
    return int(batch.num_graphs)
```

```yaml
callbacks:
  throughput_monitor:
    _target_: lightning.pytorch.callbacks.ThroughputMonitor
    batch_size_fn:
      _target_: hydra.utils.get_method
      path: my_project.throughput.num_graphs
```

The commented-out block in the shipped configs (`window_size: 100` only) is missing this required argument and fails to instantiate as written.
