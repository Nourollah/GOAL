# MTS MD Configuration Templates

Multiple-Time-Step MD with dual-calculator fine-tuning.

## Workflow

1. **Base Calculator**: High-accuracy QM or expensive ML computes reference forces
2. **ML Training**: Light ML model trains on-the-fly from base calculator data
3. **Switching**: Once ML accurate enough, it takes over the MD
4. **Monitoring**: Continuous error checking; fallback if drift detected

## Key Parameters

### Switching Strategy
- force_rmse_threshold: Force accuracy before switching (eV/Å)
- energy_mae_threshold: Energy accuracy before switching (eV/atom)
- evaluation_interval: Steps between accuracy checks
- min_training_steps: Min training before possible switch

## Simulations
- orca_ml_learner: ORCA QM + ML learner
- qm_ml_adaptive: Pre-trained QM model + adaptive ML
