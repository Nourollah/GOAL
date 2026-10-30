# SIMURGH ablation programme

One file per arm of the brief's Section 6 ablation programme.  Each composes a task
config (GMD Duplication by default) and changes one thing; the file name
gives the stage (`sNN_`).  Protocol: screen every arm with seed 42 on
Duplication and Fusion, run seeds 43 and 44 only for arms whose change exceeds
twice the base seed-to-seed spread, select on TestID only, and require the
smoothness and MD-stability gates before an accuracy counts.

Stage 11
(float64, datacenter GPUs only) is `model.backbone.dtype=float64
trainer.precision=64-true` on the final configurations.
