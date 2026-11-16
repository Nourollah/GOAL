@AGENTS.md

## Claude Code

- Project skills load from `.claude/skills/`, a generated copy of `.agents/skills/`. Edit the files under `.agents/skills/`, run `pixi run sync-skills`, and never edit `.claude/skills/` by hand.
- Call a skill explicitly with `/goal-train-model` (and so on), or let Claude choose one from its description.
