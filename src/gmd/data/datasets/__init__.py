"""GMD dataset implementations."""

from __future__ import annotations

from gmd.registry import DATASET_REGISTRY

DATASET_REGISTRY.register_lazy("xyz", "gmd.data.datasets.xyz:ExtXYZDataset")
DATASET_REGISTRY.register_lazy("hdf5", "gmd.data.datasets.hdf5:HDF5Dataset")
DATASET_REGISTRY.register_lazy("lmdb", "gmd.data.datasets.lmdb:LMDBDataset")
DATASET_REGISTRY.register_lazy("trajectory", "gmd.data.datasets.trajectory:TrajectoryDataset")
