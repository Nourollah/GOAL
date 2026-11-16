# Agent skills

GOAL ships instructions for AI coding agents such as Claude Code, Codex and Cursor. They use the open [Agent Skills](https://agentskills.io) format: each skill is a folder with a `SKILL.md` file whose short description tells the agent when to load it. An agent working in a GOAL checkout then knows how to prepare data, train, evaluate and simulate with GOAL, and which rules the project enforces.

## Where they live

| Path | Read by |
|---|---|
| `.agents/skills/` (the source) | Codex, Cursor and other tools that implement the Agent Skills standard |
| `.claude/skills/` (a generated copy) | Claude Code |
| `AGENTS.md` | Codex and most other agents; Claude Code reads it through `CLAUDE.md` |

The copy exists because Claude Code reads project skills only from `.claude/skills/`. `scripts/agents/sync_skills.py` writes it, and a test fails when it is stale.

## Using them

**Claude Code.** Start Claude Code in the repository. It loads a skill when your request matches its description, or when you call it by name, for example `/goal-train-model`.

**Codex.** Start Codex in the repository. It reads `AGENTS.md` and the skills in `.agents/skills/`; mention one with `$goal-train-model`, or let Codex choose.

**Other agents.** Tools that implement the Agent Skills standard read `.agents/skills/` in the project. Some also read `.claude/skills/`, so they may list each skill twice.

**Outside a GOAL clone.** `python scripts/agents/sync_skills.py --user` copies the skills into `~/.claude/skills/` and `~/.agents/skills/`, so that they are available in every project. It never overwrites a personal copy that differs unless you add `--force`.

## The skills

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

## Writing or changing a skill

1. Edit or add `.agents/skills/goal-<name>/SKILL.md`. The frontmatter `name` equals the folder name (lowercase letters, digits and single hyphens, at most 64 characters). The `description` says what the skill does and when to use it, in at most 1024 characters, with no angle brackets.
2. Keep the body under 500 lines; move long material into a `references/` folder inside the skill.
3. Every command, path and config key in a skill must exist in the repository. Mark anything not yet implemented as "(planned)".
4. Run `pixi run sync-skills` and `pixi run -e cuda test tests/test_agent_skills.py -q`.
5. Agents follow skills as instructions, so review changes to them as you would review code.
