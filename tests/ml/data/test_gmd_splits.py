"""The ``gmd26_FD_plus_alkane`` split composes onto any GMD experiment config."""

from __future__ import annotations

from hydra import compose, initialize_config_dir

from goal.ml.cli import CONFIGS_ML_DIR


def test_fd_plus_alkane_replaces_the_directory_split() -> None:
    with initialize_config_dir(config_dir=CONFIGS_ML_DIR, version_base=None):
        cfg = compose(
            config_name="simurgh_legacy_simplified_monolithic_gmd26",
            overrides=["+splits@data=gmd26_FD_plus_alkane", "paths.root_dir=/root"],
        )
    data = cfg.data
    assert data.train_dir is None and data.val_dir is None and data.test_dir is None
    gmd = "/root/data/GMD"
    train = list(data.train_paths)
    assert [p for p in train if "FragmentDuplication" not in p] == [
        f"{gmd}/FragmentChainExtension/TestOOD/C8H18_PBE.traj"
    ]
    assert all(p.startswith(f"{gmd}/FragmentDuplication/Training/") for p in train[:-1])
    assert all("/TestID/" in p for p in data.val_paths)
    assert all("FragmentDuplication/TestOOD/O4" in p for p in data.test_paths)
    assert not set(train) & (set(data.val_paths) | set(data.test_paths))
