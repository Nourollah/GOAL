# GOAL: notes for coding agents

GOAL (General Open Atomistic Laboratory) trains machine-learning interatomic potentials (`goal.ml`) and runs molecular dynamics with them (`goal.md`): Python 3.14, PyTorch, Lightning, Hydra, e3nn, PyTorch Geometric and ASE, under the MIT licence. Human documentation lives in `docs/` and at https://nourollah.github.io/GOAL/.

## Environments and commands

- Run everything through pixi: `pixi run -e cuda <command>` on NVIDIA GPUs, `-e rocm` on AMD GPUs, `-e cpu` without a GPU. The `-full` variants (`cuda-full`, `cpu-full`, `rocm-full`) add the MD extras; use them for `goal.md` work.
- Packages that pin an older e3nn than GOAL needs go in separate virtual environments, never into a pixi environment (`docs/getting-started/installation.md`).
- If `.venv-mace` (the standalone environment for `mace-torch`) comes before pixi on `PATH`, a `goal-train` subprocess test fails; put `.pixi/envs/cuda-full/bin` first. If `ptxas` is not on `PATH`, set `TRITON_PTXAS_PATH` to a ptxas binary before GPU evaluation.
- Tests: `pixi run -e cuda test` (skips slow tests), `pixi run -e cuda test-full`, or one path: `pixi run -e cuda test tests/<path> -x -q`.
- Quality: `pixi run -e cuda lint`, `format`, `typecheck` and `lint-configs`. Ruff line length is 99.
- Docs: `pixi run -e docs docs-build` (strict mode: links from a docs page to files outside `docs/` must be absolute URLs).
- Agent skills: edit `.agents/skills/`, then run `pixi run sync-skills`; `pixi run check-skills` verifies the Claude Code copy.

## Layout

- `src/goal/ml/`: models (`nn/models/`), building blocks (`nn/blocks/`, `nn/primitives/`, `nn/heads/`), data, training, CLI entry points and registries (`registry.py`).
- `src/goal/md/`: calculators, molecules, dynamics, observers, batched simulation and CLIs.
- `src/goal/configs/`: Hydra configs; `ml/` holds one self-contained file per experiment.
- `tests/`, `docs/` (mkdocs), `scripts/` and `notebooks/`.

## Rules for every change

1. Inside SIMURGH (`goal-naming-policy` lists the paths) everything is native and names no other model: no imports or copied code from other model packages, sources cited by arXiv ID or DOI, and the external baseline called B-REF. SIMURGH names carry no version numbers. Outside SIMURGH there is no naming restriction; never rename GOAL's foundation-model support.
2. Native models give conservative forces only (F = −∂E/∂R by autograd) and keep a static DDP graph: every parameter reaches the loss on every step, without `find_unused_parameters`.
3. One config per experiment; optional capabilities default to off; unknown keys raise.
4. Validation and checkpoint selection use in-distribution data only; OOD test sets are evaluated once, at the end of a run.
5. No DFT or other quantum-chemistry calls inside models, training loops or inference; QM runs are offline data generation.
6. Train in float64 only on datacenter GPUs (A100, H100, H200).
7. Do not change the behaviour of existing models or shared modules without the maintainer's approval.
8. Never commit secrets, host names, IP addresses, user names or personal paths.
9. Ask before deleting runs, checkpoints or data, force-pushing or rewriting history.
10. Every number you report comes from a command you ran; state its unit, split and seed.

## Skills

Task-specific instructions live in `.agents/skills/<name>/SKILL.md` (read by Codex, Cursor and other tools that implement the Agent Skills standard) and are copied to `.claude/skills/` for Claude Code. Load the matching skill before you start a task.

| Skill | Load it for |
|---|---|
| `goal-prepare-data` | converting QM or MD outputs into GOAL datasets, choosing formats, label keys and loading modes |
| `goal-train-model` | training, smoke-testing, resuming, sweeping or scaling a model with `goal-train` |
| `goal-evaluate-model` | test metrics, comparing checkpoints, archives, units of reported errors |
| `goal-run-md` | MD, relaxations and batched simulations with `goal.md` |
| `goal-finetune-foundation` | fine-tuning foundation models or a trained SIMURGH checkpoint |
| `goal-add-model` | implementing, porting or registering a model or layer |
| `goal-equivariant-kernels` | e3nn and cuEquivariance layers, layouts and parity tests |
| `goal-mlip-invariant-tests` | the physical invariant test battery for any potential |
| `goal-gmd-benchmark` | GMD training, evaluation, element references and OOD reporting |
| `goal-standard-benchmarks` | rMD17, MD17, MD22 and 3BPA protocols and published targets |
| `goal-md-stability-benchmark` | MD stability and NVE drift comparisons between models |
| `goal-ablation-study` | planning, running and reporting ablations |
| `goal-naming-policy` | names of files, keys, configs, docs and commits |
| `goal-remote-training` | running and collecting jobs on remote GPU machines |
| `goal-simurgh` | SIMURGH configs, code, checkpoints and its ablation programme |
