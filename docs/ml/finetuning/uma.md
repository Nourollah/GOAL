# Fine-tuning UMA / FairChem

`uma_finetune` works like [`mace_finetune`](mace.md): it wraps a pre-trained
UMA model as a trainable monolithic backbone and fine-tunes it through the
standard GOAL loop. The wrapper itself is complete. Whether it runs depends
on whether your `fairchem-core` version exposes the model as a raw
`nn.Module`.

Source:
[`goal.ml.nn.models.foundation.uma.UMAFinetune`](https://github.com/Nourollah/GOAL/blob/main/src/goal/ml/nn/models/foundation/uma.py).

## Status: loading the raw module is not a documented FairChem API

fairchem-core's public API returns an **inference predictor or ASE
calculator**, not a bare trainable `nn.Module`. `UMAFinetune._load_model`
tries the most likely path:

```python
from fairchem.core import pretrained_mlip
predictor = pretrained_mlip.get_predict_unit(checkpoint, device=device)
model = predictor.model   # or predictor.module
```

If that import fails, or the predictor does not expose an `nn.Module`, it
raises `UnsupportedOperationError` with a link to the FairChem docs
(<https://fair-chem.github.io/>). Without `fairchem-core` installed, the same
error is raised.

Everything else is in place: the input and output adapters, the
`head_only`/`full`/`lora` strategies, and E0 re-estimation. `uma_finetune`
therefore starts working as soon as a raw-module load path exists. If you
know the right entry point for your fairchem version, adapt
`UMAFinetune._load_model`. Also confirm the batch field names in
`_adapt_input` (see below).

```bash
pip install fairchem-core   # or: pip install "goal-atomistic[fairchem]"
```

## Dataset-type embedding: `omol` vs `omat`

UMA conditions on a **dataset-type** embedding, chosen with the backbone's
`head` key. This is not the GOAL task head, which stays `null`:

```yaml
model:
  backbone:
    name: uma_finetune
    checkpoint: "uma-s-1"       # UMA model name or checkpoint path
    head: omol                  # omol (organic molecules) | omat (inorganic materials) | omc | odac | oc20
    strategy: head_only         # head_only | full | lora
    lora_rank: 4
    lora_alpha: 16.0
    lora_target_modules: null
    dtype: float64
    reestimate_e0s: true
    device: cpu
  head: null
```

A value outside `omol`, `omat`, `omc`, `odac`, `oc20` is passed through with a
warning.

## Parts to confirm against your fairchem version

These parts are written defensively and are marked in the source:

| Piece | Current behaviour |
|---|---|
| Input batch (`_adapt_input`) | dict with `pos`, `atomic_numbers`, `edge_index`, `cell`, `batch`, `natoms`, and the dataset embedding under the key **`dataset`** (one entry per graph) |
| Output (`_adapt_output`) | reads `energy` and `forces` from a dict or attributes, raising `KeyError` with the keys it got otherwise |
| `head_only` readout fragments | parameter names containing `output_head`, `energy_head`, `force_head` or `_head` |
| E0 buffer (`set_atomic_energies`) | first tensor found among `atomic_energies`, `energy_references`, `references`, indexed by Z. If none exists, re-estimation is skipped with a warning and training still runs |

If `head_only` leaves too few or too many parameters trainable (check the
`[finetune] strategy=head_only — n/N params trainable` log line), adjust
`_readout_name_fragments`.

## Run

```bash
pixi run -e cuda goal-train --config-name finetune/uma data.root=data/finetune
```

[`src/goal/configs/ml/finetune/uma.yaml`](https://github.com/Nourollah/GOAL/blob/main/src/goal/configs/ml/finetune/uma.yaml)
has the same structure as the MACE config: ExtXYZ in `data/finetune`, a 90/10
split, `cutoff: 6.0` (match your UMA model), `batch_size: 4`,
`precision: "64"`, `inference_mode: false`, and `FoundationE0Callback` in
`callbacks:`. Checkpointing, the curriculum, DDP and HPO behave exactly as in
[MACE fine-tuning](mace.md).

> [!TIP]
> If you only need UMA as a fixed potential (MD, geometry optimisation,
> zero-shot evaluation), use FairChem's own ASE calculator directly. The
> trainable wrapper is only needed when you want to update its weights.
