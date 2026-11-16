# SIMURGH ablation programme

One file per arm of the brief's Section 6 ablation programme.  Each composes a task
config (GMD Duplication by default) and changes one thing; the file name
gives the stage (`sNN_`).  Protocol: screen every arm with seed 42 on
Duplication and Fusion, run seeds 43 and 44 only for arms whose change exceeds
twice the base seed-to-seed spread, select on TestID only, and require the
smoothness and MD-stability gates before an accuracy counts.

Stages 11 and 12 build on `combined_base.yaml` (the combined base: stage 11;
an ECE branch next to the ACE core: stage 12).  Stage 13
(float64, datacenter GPUs only) is `model.backbone.dtype=float64
trainer.precision=64-true` on the final configurations.
