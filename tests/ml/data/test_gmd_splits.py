"""The ``gmd26_FD_plus_alkane`` split composes onto any GMD experiment config."""

from __future__ import annotations

from hydra import compose, initialize_config_dir

from goal.ml.cli import CONFIGS_ML_DIR


def test_fd_plus_alkane_adds_octane_to_duplication() -> None:
    with initialize_config_dir(config_dir=CONFIGS_ML_DIR, version_base=None):
        cfg = compose(
            config_name="simurgh_dev_gmd26_FD",
            overrides=["+splits@data=gmd26_FD_plus_alkane", "paths.root_dir=/root"],
        )
    data = cfg.data
    assert data.dataset_type == "gmd" and data.task == "fragment_duplication"
    assert list(data.extra_train_molecules) == ["Alkanes/C8H18"]
    assert data.train_dir is None and data.train_paths is None
