"""GOAL dataset implementations."""

from __future__ import annotations

from goal.ml.registry import DATASET_REGISTRY

DATASET_REGISTRY.register_lazy("xyz", "goal.ml.data.datasets.xyz:ExtXYZDataset")
DATASET_REGISTRY.register_lazy("hdf5", "goal.ml.data.datasets.hdf5:HDF5Dataset")
DATASET_REGISTRY.register_lazy("lmdb", "goal.ml.data.datasets.lmdb:LMDBDataset")
DATASET_REGISTRY.register_lazy("trajectory", "goal.ml.data.datasets.trajectory:TrajectoryDataset")
