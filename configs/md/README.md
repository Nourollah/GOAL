# Molecular Dynamics Configuration Templates

This directory contains Hydra configuration files for MD simulations,
designed to work seamlessly with goal.ml trained models.

## Structure

- **molecule/**: Molecule source configs (SMILES, files, databases)
- **calculator/**: Calculator backend configs (ML models, QM, pretrained)
- **dynamics/**: Dynamics/integrator configs (Langevin, etc)

## Usage Pattern

```python
from hydra import compose, initialize_config_dir
import os

with initialize_config_dir(
    config_dir=os.path.abspath("configs/md"),
    version_base="1.3"
):
    cfg = compose(config_name="langevin_with_model_sim")
    # Now use cfg to instantiate everything
```

## Example: Using a Trained Model

See `langevin_with_model_sim.yaml` for a complete example that:
1. Creates molecule from SMILES
2. Loads a trained model from goal.ml
3. Runs Langevin MD at 300 K
4. Logs to a file and saves the output trajectory
