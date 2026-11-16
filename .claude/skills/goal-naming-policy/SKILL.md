---
name: goal-naming-policy
description: Enforces GOAL's naming rules. Inside SIMURGH (its package, ablations, configs, tests, scripts, ECE modules, docs and skill) everything is native and names no other model, sources are cited by arXiv ID or DOI, and the external reference baseline is called B-REF. SIMURGH names carry no version numbers, older variants live under the legacy namespace, and the model in development uses the simurgh_dev key until release. Outside SIMURGH there is no naming restriction. Use before creating or renaming SIMURGH files, keys, configs or docs, and when the naming guard fails.
license: MIT
---

# GOAL naming policy

## 1. Scope

The rules apply inside SIMURGH:
- `src/goal/ml/nn/models/simurgh/dev/`, including `ablations/`;
- the ECE modules SIMURGH uses: `src/goal/ml/nn/primitives/edge_frame.py`, `src/goal/ml/nn/primitives/o2.py`, `src/goal/ml/nn/blocks/ece.py` and `src/goal/ml/nn/heads/edge_energy.py`;
- the `simurgh_dev*` configs and `src/goal/configs/ml/ablations/simurgh_dev/`;
- `tests/ml/simurgh_dev/` and `scripts/simurgh_dev/`;
- SIMURGH's docs page (`docs/ml/models/simurgh-dev.md`) and the `goal-simurgh` skill (both copies).

The legacy SIMURGH models and the rest of GOAL are outside the scope. GOAL's foundation-model support keeps the reference package's real name: never rename, move or remove it.

## 2. Inside SIMURGH

- **No other model's name**, anywhere: identifiers, file names, config keys and comments, docstrings, test names, log messages, docs. Cite a source by arXiv ID or DOI, with its equation or table number, and name techniques by what they do (for example `physical_long_range`, `coupled_product`, `quadrangle_term`).
- **The external baseline is B-REF.** Its predictions arrive as neutral files (`b_ref_predictions_<task>_<seed>.npz`), and its kernel fixtures as plain tensors (`reference_symmetric_contraction_*.pt`, `reference_interaction_*.pt`). It is trained in its own workspace outside the repository.
- **Native code only.** No imports from other model packages or from GOAL's foundation wrappers (`goal.ml.nn.models.foundation`, `goal.ml.adapters`), and no code copied from other repositories. Implement from the paper's equations. Libraries (torch, e3nn, cuequivariance, torch_geometric, ase, tad-dftd3, dftd3-python) are fine.
- **No version numbers** in any SIMURGH name, in any form: no hyphen and digit, no space and "v" with a digit, no digit glued to the name.

## 3. Keys and files

- During development the registry key is `simurgh_dev`: package `src/goal/ml/nn/models/simurgh/dev/`, configs `simurgh_dev_<task>.yaml`, tests `tests/ml/simurgh_dev/`, scripts `scripts/simurgh_dev/`. The `_dev` suffix is dropped at release, when the `simurgh` key moves to this model and the deprecated aliases are removed.
- Ablation files: `src/goal/ml/nn/models/simurgh/dev/ablations/<mechanism>.py`, registered as `simurgh_dev_abl_<mechanism>`.
- Older models live in `src/goal/ml/nn/models/simurgh/legacy/`; their old keys are deprecated aliases until release:

| Old key | Key |
|---|---|
| `simurgh` | `simurgh_legacy` |
| `simurgh_ace_first` | `simurgh_legacy_ace_first` |
| `simurgh_arace` | `simurgh_legacy_arace` |
| `monolithic_arace` | `simurgh_legacy_monolithic_arace` |
| `simurgh_monolithic` | `simurgh_legacy_monolithic` |
| `nf_simurgh` | `simurgh_legacy_nf` |
| `simurgh_simplified_monolithic` | `simurgh_legacy_simplified_monolithic` |
| `simurgh_simplified_routed` | `simurgh_legacy_simplified_routed` |

Old checkpoints are converted with `scripts/simurgh_dev/migrate_legacy_checkpoints.py` (a dry run unless told otherwise).

## 4. The guard

`tests/ml/simurgh_dev/test_naming_policy.py` checks the scope for:
- names from `tests/naming/other_model_names.txt` (one per line; `cs:` marks a case-sensitive entry, `prefix:` a name matched only at the start of a word; all others match case-insensitively, with no letter directly before or after);
- imports, parsed with `ast`, from the packages in `tests/naming/denied_imports.txt` (other model packages and GOAL's foundation wrappers);
- file names;
- version numbers in SIMURGH names.

Add a newly found model name to the list instead of special-casing it. Put genuine false positives in `tests/ml/simurgh_dev/naming_allowlist.txt`, one `path` or `path:line` per line followed by `# reason` (an entry without a reason fails the test).

```bash
pixi run -e cuda test tests/ml/simurgh_dev/test_naming_policy.py -q
pixi run -e cuda test tests/test_agent_skills.py -q
```

## Done when

- [ ] Both tests pass, and nothing in the scope names another model or imports one.
