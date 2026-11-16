---
name: goal-finetune-foundation
description: Fine-tunes pre-trained interatomic potentials inside GOAL (in-process foundation-model backbones configured under src/goal/configs/ml/finetune, UPET through goal-finetune-upet, LoRA on a trained SIMURGH checkpoint). Covers choosing head_only, lora or full, element reference energies, matching cutoff and dtype, DDP settings for frozen parameters, separate environments for packages with old e3nn pins, and zero-shot evaluation first. Use when the user wants to adapt a foundation model or a pre-trained GOAL model to new data.
license: MIT
---

# Fine-tuning pre-trained potentials

Reference: `docs/ml/finetuning/overview.md` (the per-model pages are linked from it) and `docs/integrations/lora_finetune.md`.

## Workflow

1. Get the data into a GOAL format (`goal-prepare-data`). `finetune/mace` and `finetune/uma` read extxyz from `data/finetune` with a 90/10 split; `finetune/simurgh_legacy_lora` needs `data.train_paths` and `data.val_paths`.
2. Evaluate candidate models zero-shot as MD calculators on your validation frames (`docs/md/calculators/foundation.md`, `docs/md/calculators/upet.md`). A matching training domain does not guarantee the best starting point.
3. Start with `model.backbone.strategy: head_only`; move to `lora` or `full` only if it plateaus and the data support it.
4. Train with `pixi run -e cuda goal-train --config-name finetune/<model>`; list the choices with `ls src/goal/configs/ml/finetune/` (`finetune/mace`, `finetune/uma`, and `finetune/simurgh_legacy_lora` for LoRA on a trained SIMURGH checkpoint, set through `model.backbone.checkpoint_path`).
5. Use the result like any GOAL checkpoint (`goal-evaluate-model`). The base model must still be resolvable at load time.

| `strategy` | What trains | Suggested LR |
|---|---|---|
| `head_only` | readout parameters only | about 1e-4 |
| `lora` | LoRA adapters on a frozen base (`lora_rank`, `lora_alpha`, `lora_target_modules`) | about 1e-4 |
| `full` | every parameter; risks forgetting on small datasets | about 1e-5 |

These keys belong to the foundation backbones (`mace_finetune`, `uma_finetune`). The SIMURGH LoRA backbone (`simurgh_lora`, `finetune/simurgh_legacy_lora`) has no `strategy` key and uses `rank`, `alpha`, `dropout`, `target_modules`, `bias` and `head_trainable` instead.

## Rules

- Element reference energies cause most fine-tuning failures. `reestimate_e0s: true` (with `FoundationE0Callback`) fits them by least squares on the training set before the first step. That is valid only when the training compositions identify every element. For homologous series such as GMD they do not: provide isolated-atom energies at the labels' level of theory, or report that the backbone cannot accept them. At present neither `mace_finetune` nor `uma_finetune` accepts provided element energies from the config: their constructors silently drop unknown keys such as `atomic_energies`, and `reestimate_e0s` is the only config route. Read the logged `[finetune] re-estimated atomic energies` line.
- Set `data.cutoff` to the foundation model's cutoff.
- Some foundation models are float64: set `model.backbone.dtype: float64`, `trainer.precision: "64"` and `trainer.inference_mode: false`, and run on datacenter GPUs.
- `head_only` and `lora` leave parameters out of the autograd graph, so DDP needs `strategy.find_unused_parameters: true`. This is the only place that flag belongs; native GOAL models keep a static graph.
- The foundation backbones are monolithic: `model.head: null`.
- `model.backbone.strategy` (fine-tuning) is not the top-level `strategy:` block (DDP, FSDP, DeepSpeed).
- `mace-torch` pins e3nn 0.4.4, older than GOAL needs, so it lives in its own virtual environment (`.venv-mace`; "Packages installed separately" in `docs/getting-started/installation.md`). Never install it into a pixi environment.
- UPET runs metatrain as a subprocess: `uv pip install --python .pixi/envs/cuda/bin/python metatrain upet`, then `goal-finetune-upet ... --dry-run` and read the generated options before the real run.
- Prefer `goal-train --config-name finetune/<model>`; the overview page documents the older `goal-finetune` entry point and the feature-extraction adapters as legacy.

## Done when

- [ ] A zero-shot baseline exists on the same frames.
- [ ] The log shows the intended number of trainable parameters and the intended element references.
- [ ] Fine-tuned errors are reported next to the zero-shot baseline.
