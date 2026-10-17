# Project structure

This page is an annotated map of the GOAL repository: where the `goal.ml` and `goal.md` code lives, how the Hydra configs are organised, and what the notebooks, scripts, tests and docs contain. Use it to find the file you need before you open the source.

> [!NOTE]
> Local working directories are omitted: `data/`, `logs/`, `checkpoints/`, `dump/`, `outputs/`, `.venv*`, `.pixi/` and `__pycache__/`. Most of them are created at run time. `logs/` is where every training run writes its output (see [Configuration → output directory](../ml/reference/configuration.md#output-directory-layout)), and `data/` is the default dataset root (`paths.data_dir`).

## Top level

```text
GOAL/
├── pyproject.toml          # package metadata (distribution goal-atomistic), extras, console scripts, pytest config, AND the pixi workspace
├── pixi.lock               # locked conda + PyPI environment
├── mkdocs.yml              # this documentation site
├── README.md               # also the PyPI project page
├── LICENSE                 # MIT
├── .project-root           # marker used by rootutils to find the repository root
├── .env.example            # template for environment variables (API keys, PROJECT_ROOT, …)
├── .pre-commit-config.yaml
├── .github/                # CI: tests, code quality, release drafter, dependabot, and
│   └── workflows/          #   docs.yml (build + deploy this site), release.yml (publish to PyPI on a GitHub Release)
├── recipe/recipe.yaml      # conda-forge recipe: goal-atomistic, goal-atomistic-md, goal-atomistic-full
├── src/                    # the installable code (goal.ml, goal.md, bundled Hydra configs) + legacy template utils
├── notebooks/              # tutorials, inference notebooks, case studies
├── scripts/                # helper scripts (OMol25 indexing, SLURM, notebook generator)
├── tests/                  # pytest suite for ml/ and md/
└── docs/                   # this site's Markdown sources
```

## `src/goal/ml`: machine-learning potentials

```text
src/goal/
├── __init__.py                     # namespace root; CONFIGS_DIR; Python 3.14 / Hydra argparse compatibility shim
├── configs/                        # bundled Hydra configs (see below)
└── ml/
    ├── __init__.py                 # exposes goal.ml.trainer (= goal.ml.training) and __version__
    ├── registry.py                 # Registry class + MODEL/BACKBONE/HEAD/LOSS/DATASET/TRANSFORM/STRATEGY registries
    │
    ├── cli/                        # console-script entry points (see ML → Reference → CLI)
    │   ├── __init__.py             #   CONFIGS_DIR / CONFIGS_ML_DIR, registry population, torch.load patch
    │   ├── builders.py             #   loss / head / callback / logger builders shared by the Hydra CLIs
    │   ├── train.py                #   goal-train / goal-train-ml (Hydra, default config simurgh_gmd26)
    │   ├── evaluate.py             #   goal-eval (default config eval; model rebuilt from the checkpoint)
    │   ├── finetune.py             #   goal-finetune (default config finetune/simurgh_lora)
    │   ├── finetune_upet.py        #   goal-finetune-upet (argparse → metatrain `mtt train`)
    │   ├── tune.py                 #   goal-tune (Lightning Tuner / Ray Tune / W&B Sweeps)
    │   ├── inspect_ckpt.py         #   goal-inspect
    │   ├── pack.py                 #   goal-pack-archive
    │   ├── pack_zarr.py            #   goal-pack-zarr (structure files → Zarr store)
    │   └── distill.py              #   goal-distill (roadmap stub)
    │
    ├── data/
    │   ├── graph.py                #   AtomicGraph (PyG Data subclass), NodeFeatures: the central data contract
    │   ├── datamodule.py           #   GOALDataModule: loading modes, splits, merging, DataLoaders
    │   ├── datasets/               #   registered loaders
    │   │   ├── base.py             #     abstract base dataset
    │   │   ├── xyz.py              #     "xyz": ExtXYZ via ASE
    │   │   ├── trajectory.py       #     "trajectory": ASE .traj / .db
    │   │   ├── hdf5.py             #     "hdf5": pre-processed graphs
    │   │   ├── lmdb.py             #     "lmdb": FairChem/OCP LMDB
    │   │   ├── aselmdb.py          #     "aselmdb": FairChem .aselmdb shards (OMol25, OMat24, ODAC23) + catalogue
    │   │   └── concat.py           #     multi-source concatenation
    │   ├── benchmarks/             #   benchmark adapters: md17/rmd17, md22, qm9, ani1/ani1x, spice (see below)
    │   ├── neighbor_list.py        #   neighbour lists: ase/matscipy/nvalchemiops/torch/auto + batched tensor builder
    │   ├── collate.py              #   PyG / GPU-tensor / nvalchemi batch construction
    │   ├── zarr_dataset.py         #   "zarr": GOAL Zarr stores (write_zarr, ZarrDataset)
    │   ├── transforms.py           #   graph-construction transforms ("radius_graph")
    │   ├── fragments.py            #   fragment decomposition (fragment_index for NF_SIMURGH / fragment add-ons)
    │   ├── keys.py                 #   label key mapping (energy_key, forces_key, …)
    │   ├── statistics.py           #   dataset-derived buffers: elements, atomic references, scale, avg neighbours, pair counts
    │   └── export.py               #   AtomicGraph → ASE Atoms / ExtXYZ
    │
    ├── nn/
    │   ├── primitives/             #   thin typed e3nn wrappers: tp.py, linear.py, norm.py, radial.py; neighbor_list.py front end
    │   ├── blocks/                 #   reusable building blocks
    │   │   ├── artisans.py         #     SIMURGH element-pair bank of potential artisans (+ pair-count routing)
    │   │   ├── ace_block.py        #     ACE message-passing block (ARACE)
    │   │   ├── env_dressing.py     #     environment dressing (legacy ACE-first SIMURGH)
    │   │   ├── symmetric_contraction.py  # MACE-style per-element body-order contraction
    │   │   ├── adaptive_gate.py    #     optional adaptive depth gate for ARACE rounds
    │   │   ├── fragment_interaction.py   # optional equivariant fragment interaction for ARACE rounds
    │   │   ├── interaction.py, embedding.py, readout.py
    │   ├── heads/                  #   task heads: energy, energy_forces, direct_forces, dual_forces,
    │   │                           #   stress, dipole, scalar, multi
    │   └── models/
    │       ├── base.py             #     Backbone / Head / MonolithicModel protocols
    │       ├── simurgh/            #     native SIMURGH family
    │       │   ├── backbone_arace.py     # simurgh_arace: DEFAULT modular ARACE backbone
    │       │   ├── backbone.py           # simurgh / simurgh_ace_first: legacy ACE-first backbone
    │       │   ├── arace.py              # monolithic_arace: self-contained ARACE
    │       │   ├── monolithic.py         # simurgh_monolithic
    │       │   ├── nf_simurgh.py         # nf_simurgh: Native Fragment SIMURGH (fragment virtual nodes)
    │       │   ├── simurgh_simplified_layers.py      # shared skeleton of the three-line models
    │       │   ├── simurgh_simplified_routed.py      # simurgh_simplified_routed
    │       │   ├── simurgh_simplified_monolithic.py  # simurgh_simplified_monolithic
    │       │   └── geometry.py           # differentiable edge-geometry recomputation
    │       ├── ablations/          #     simurgh_ablation: isolated, frozen ablation-study model
    │       ├── foundation/         #     trainable foundation backbones: mace_finetune, uma_finetune
    │       ├── hyperspec.py        #     hyperspec: equivariant GNN baseline
    │       ├── invariant.py        #     invariant_gnn: scalar message passing
    │       └── monolithic_example.py  #  monolithic_example: minimal self-contained model
    │
    ├── adapters/                   #   frozen foundation-model adapters: mace-small/medium/large, uma-small;
    │                               #   simurgh_lora.py (LoRA on SIMURGH checkpoints), nvalchemi_wrapper.py
    │
    ├── training/
    │   ├── module.py               #   GOALModule: universal LightningModule for backbone (+ head)
    │   ├── loss.py                 #   CompositeLoss, WeightedLoss; losses energy/forces/stress/dipole/charge/gate_reg/scalar_property
    │   ├── metrics.py              #   MLIP diagnostics + logger-section naming
    │   ├── ema.py                  #   exponential moving average of weights
    │   ├── resume.py               #   auto-resume, run extension, LR-schedule retargeting, TRAINING_COMPLETE sentinel
    │   ├── archive.py              #   self-contained .simurgh archives, frozen source, inspect/pack helpers
    │   ├── tuning.py               #   Lightning Tuner, Ray Tune and W&B Sweep drivers
    │   ├── callbacks/
    │   │   ├── checkpoint_manager.py  # GOALCheckpointManager: top-k / interval / last pools
    │   │   ├── checkpoint.py       #     GOALCheckpoint (ModelCheckpoint + SLURM sentinel)
    │   │   ├── foundation.py       #     re-estimate foundation-model atomic energies
    │   │   ├── logging.py          #     RichLoggingCallback
    │   │   └── progress.py         #     stage-aware GOALRichProgressBar
    │   └── strategies/             #   build_strategy(): auto/single/ddp/fsdp/fsdp2/deepspeed_*
    │
    └── utils/
        ├── calculator.py           #   GOALCalculator + load_goal_checkpoint (EMA, dtype, fragment labels)
        ├── hardware.py             #   compute-backend detection, nvalchemi availability gates
        ├── enhanced_sampling.py    #   metadynamics / REMD (roadmap stubs)
        ├── extraction.py           #   HookBasedExtractor, Layer/MultiScale/FrozenBackbone, irrep helpers
        ├── mini_trainer.py         #   MiniTrainer: notebook-friendly pure-PyTorch loop
        └── fabric_trainer.py       #   FabricTrainer: custom loop with Lightning Fabric
```

`src/utils/` holds leftover helpers from the Lightning-Hydra project template (`instantiators.py`, `rich_utils.py`, `pylogger.py`, …). The current CLIs do not use them.

## `src/goal/md`: molecular dynamics

```text
src/goal/md/
├── __init__.py                 # public API (CalculatorFactory, Simulation, …)
├── README.md                   # module-level notes
├── cli/
│   ├── main.py                 # typer app (placeholder `goal-md simulate`; python -m goal.md.cli.main)
│   ├── simulate.py             # goal-simulate / goal-simulate-mts (Hydra)
│   └── simulate_batch.py       # goal-simulate-batch (argparse)
├── core/
│   ├── calculator_factory.py   # CalculatorFactory: goal_model, mace, flashmd, upet, nequip, tblite, orca, cp2k, espresso, vasp
│   ├── molecule_factory.py     # build ASE Atoms from SMILES / names / files
│   ├── molecule_tools.py       # geometry and setup utilities
│   ├── md_factory.py           # dynamics / integrator factory
│   ├── simulation.py           # Simulation: molecule + calculator + dynamics
│   ├── batched.py              # batched MD / relaxation: nvalchemi GPU batch or ASE
│   ├── runner.py               # SimulationRunner: multi-step runs with progress
│   └── observers.py            # observer protocol and implementations
├── adapters/
│   ├── model_loader.py         # load goal.ml checkpoints as calculators
│   └── ase_converter.py        # ASE ↔ AtomicGraph bridges
├── calculators/                # shared exception types for calculators
├── mts/                        # multiple time-step MD
│   ├── integrators/respa.py    # RESPA integrator
│   ├── ipi_adapter.py          # i-PI compatibility layer, MTSSimulation
│   ├── dual_calculator/        # dual-calculator base, switching policies, QM+ML online learner
│   ├── config_parsers/         # Hydra support for MTS workflows
│   └── ensemble_managers/      # multi-calculator orchestration
└── utils/
```

See the [goal.md overview](../md/index.md).

## `src/goal/configs/`: Hydra configuration

The configs are part of the package (installed as `goal/configs/`; the absolute path is `goal.CONFIGS_DIR`), so the CLIs find them from any directory, in a clone or in a pip/conda install.

```text
src/goal/configs/
├── ml/                                   # self-contained goal.ml experiment configs (goal-train's config dir)
│   ├── simurgh_gmd26.yaml                #   DEFAULT: SIMURGH ARACE on GMD-26 FragmentDuplication
│   ├── simurgh_gmd26_hpo.yaml            #   Ray Tune search space over ARACE (composes simurgh_gmd26)
│   ├── simurgh_ace_first_gmd26.yaml      #   legacy ACE-first SIMURGH on GMD-26
│   ├── monolithic_arace_gmd26.yaml       #   monolithic ARACE on GMD-26
│   ├── nf_simurgh_gmd26.yaml             #   NF_SIMURGH on GMD-26 FragmentDuplication
│   ├── nf_simurgh_gmd26_typecond.yaml    #   NF_SIMURGH + type-conditioned virtual nodes (FragmentDuplication)
│   ├── nf_simurgh_gmd26_FF.yaml          #   same model on the GMD FragmentFusion split
│   ├── nf_simurgh_md17.yaml              #   NF_SIMURGH on rMD17 aspirin
│   ├── nf_simurgh_md17_benzene.yaml      #   NF_SIMURGH on rMD17 benzene (fragment channel inert)
│   ├── nf_simurgh_md22.yaml              #   NF_SIMURGH on MD22 Ac-Ala3-NHMe
│   ├── nf_simurgh_qm9.yaml               #   NF_SIMURGH on QM9 U0
│   ├── nf_simurgh_omol25.yaml            #   NF_SIMURGH on the OMol25 4M organic subset
│   ├── simurgh_simplified_routed_{gmd26,md17}.yaml      # SIMURGH Simplified, Routed
│   ├── simurgh_simplified_monolithic_{gmd26,md17}.yaml  # SIMURGH Simplified, Monolithic
│   ├── simurgh_md17.yaml                 #   legacy SIMURGH on MD17 aspirin
│   ├── hyperspec_md17.yaml               #   HyperSpec baseline on MD17 aspirin
│   ├── eval.yaml                         #   goal-eval default (ckpt_path + test-loop settings)
│   ├── finetune/                         #   fine-tuning (run with goal-train)
│   │   ├── mace.yaml                     #     mace_finetune
│   │   ├── uma.yaml                      #     uma_finetune
│   │   └── simurgh_lora.yaml             #     simurgh_lora (also goal-finetune's default)
│   └── ablations/                        #   SIMURGH ablation study (simurgh_ablation model)
│       ├── README.md                     #     study design, ladder, controls, run order
│       ├── _ablation_base.yaml           #     shared base (not a run)
│       ├── 01_…14_*.yaml                 #     phase 1: one switch per rung
│       ├── 90_…93_*.yaml                 #     controls: seeds, width-matched, 3 rounds
│       ├── 201_…208_*.yaml               #     phase 2: body_order + one other module
│       └── make_combo_configs.py         #     generator for the 2xx combination configs
├── md/                                   # goal.md configs (goal-simulate's config tree)
│   ├── README.md
│   ├── calculators/                      #   cp2k, cp2k_pw, espresso, vasp, mace, flashmd, upet, tblite, goal_model
│   ├── dynamics/langevin.yaml
│   ├── molecules/                        #   ethanol, from_file
│   ├── simulations/langevin_with_model.yaml   # goal-simulate default
│   ├── simulations/batch.yaml          # goal-simulate-batch --config example
│   └── mts/
│       ├── README.md
│       ├── simulations/                  #   mts_respa (goal-simulate-mts default), online_learner, orca_ml_learner
│       ├── base_calculators/orca_reference.yaml
│       ├── dynamics/langevin_mts.yaml
│       └── ml_models/learner.yaml
├── hparams_search/                       # goal-tune search schemas: basic, ray_tune, wandb_sweep
├── debug/                                # debug presets: default, fdr, limit, overfit, profiler
├── hydra/default.yaml                    # legacy multi-file run-dir settings (ml configs inline their own)
├── paths/default.yaml                    # legacy path settings (ml configs inline their own)
├── extras/default.yaml                   # legacy template extras
└── local/                                # untracked machine-specific overrides
```

See [Configuration](../ml/reference/configuration.md) for how these fit together.

## `src/goal/ml/data/benchmarks/`: benchmark dataset adapters

```text
src/goal/ml/data/benchmarks/  # module goal.ml.data.benchmarks, registered into DATASET_REGISTRY on import
├── README.md                 # note: its quick-start still shows old config-group syntax (data=md17_aspirin)
├── base.py                   # shared download / caching base class
├── md17.py                   # "md17", "rmd17"
├── md22.py                   # "md22"
├── qm9.py                    # "qm9"
├── ani1.py                   # "ani1", "ani1x"
└── spice.py                  # "spice"
```

The adapters are part of the installed package and register automatically when `goal.ml.data.datasets` is imported, which every CLI does. See [Benchmark datasets](../ml/data/benchmarks.md).

## Packaging and releases

The distribution is `goal-atomistic` on PyPI and conda-forge; the import name stays `goal`. `.github/workflows/release.yml` builds the wheel and publishes it to PyPI (trusted publishing) when a GitHub Release is published, and `recipe/recipe.yaml` is the conda-forge recipe for the three conda packages (`goal-atomistic`, `goal-atomistic-md`, `goal-atomistic-full`). The release procedure is described in [PUBLISHING.md](https://github.com/Nourollah/GOAL/blob/main/PUBLISHING.md); install routes are on the [Installation](installation.md) page.

## `notebooks/`

Seventeen notebooks plus three support folders: `data/` (the bundled CPO-27-Mn CIF), `dobdc_work/` and `mof74_work/` (geometries and logs written by the MOF-74 series). See the [Notebooks catalogue](notebooks.md).

## `scripts/`

| Script | Purpose |
|---|---|
| `omol25_prepare.py` | Index an OMol25 (or any FairChem `.aselmdb`) split, build the per-shard catalogue caches, and report training statistics. Example: `python scripts/omol25_prepare.py --root data/OMol25/train_4M --workers 8` |
| `slurm_train.sh` | Multi-node SBATCH script: `EXPERIMENT=<name> sbatch scripts/slurm_train.sh` (DDP over the `#SBATCH` nodes and GPUs) |

`slurm_train.sh` uses the self-contained config system (`--config-name <experiment>` plus dotted overrides); `pixi run lint-configs` fails if a script, README or doc page reintroduces the removed config-group syntax. See [Distributed training](../ml/training/distributed.md).

## `tests/`

```text
tests/
├── conftest.py, helpers/        # shared fixtures; RunIf / package_available helpers
├── ml/
│   ├── test_registry.py, test_graph.py, test_data.py, test_aselmdb.py, test_example_datasets.py,
│   │   test_training.py, test_strategies.py, test_archive.py, test_extraction.py
│   ├── data/                    # fragment schemes, fragment-index pipeline, key mapping, traj loader
│   ├── simurgh/                 # ARACE, artisans, pair routing, body order, NF_SIMURGH, SIMURGH Simplified, forces, …
│   ├── ablations/               # simurgh_ablation model
│   ├── foundation/              # mace_finetune / uma_finetune (integration tests need the engines)
│   └── training/                # checkpoint manager, resume/extend, logger sections
└── md/                          # factories (incl. tblite), simulation, MTS, CP2K, ESPRESSO, VASP, UPET calculator
```

Markers: `slow` (skipped by `pixi run -e cuda test`) and `integration` (needs a real foundation model or external engine). Pytest also runs `--doctest-modules`.

## `docs/`

```text
docs/
├── index.md                    # site home
├── assets/                     # logo, banner
├── getting-started/            # installation, project structure, notebooks
├── ml/                         # goal.ml: training, data, models, fine-tuning, features, deployment, reference
└── md/                         # goal.md: guide, calculators, MTS, reference
```
