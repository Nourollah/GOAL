"""Where each metric lands: Newton axes, and the energy/forces/gate sections.

Four guarantees are locked in here, all about the shape of a run's
dashboard rather than about the physics:

1. ``mlip_metrics`` reports the Newton violation as **four** numbers —
   one per Cartesian axis plus their sum — so an anisotropic violation is
   visible instead of averaged away.
2. The model's own regularisation term (``aux_loss``) is logged under
   ``{split}_gate/``, never under ``{split}/``, so it cannot bury
   ``val_forces/forces_mae`` in the W&B panel list.
3. Energy and force curves get a section each — ``{split}_energy/`` and
   ``{split}_forces/`` — loss term included, leaving ``{split}/`` holding
   the loss ``total`` and nothing property-specific.
4. A second validation dataloader (the OOD set monitored during
   training) logs under ``test…/`` and never merges into ``val…/``.
"""

from __future__ import annotations

import typing
import warnings

import pytest
import torch
from ase.build import molecule
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.training.metrics import PROG_BAR_METRICS, metric_section, mlip_metrics

E0 = {1: -13.6, 6: -1028.5, 8: -2041.1}
SMALL: dict = {
    "elements": [1, 6, 8],
    "num_channels": 8,
    "message_irreps": "8x0e + 8x1o",
    "embedding_dim": 8,
    "radial_mlp": [16],
    "avg_num_neighbors": 6.0,
    "scale": 0.7,
    "dtype": "float64",
    "atomic_energies": {"mode": "provided", "values": E0},
}


def _graph(name: str) -> AtomicGraph:
    return AtomicGraph.from_ase(molecule(name), cutoff=5.0, dtype=torch.float64)


def _model():
    from goal.ml.nn.models.simurgh.dev import SimurghDev

    torch.manual_seed(0)
    return SimurghDev(**SMALL)


NEWTON_KEYS: list[str] = [
    "newton_violation_x",
    "newton_violation_y",
    "newton_violation_z",
    "newton_violation",
]


# ----------------------------------------------------------------------
# Newton violation — one number per axis, plus their sum
# ----------------------------------------------------------------------


def _batch_with_forces(forces: torch.Tensor) -> tuple[dict[str, torch.Tensor], Batch]:
    """Predictions carrying *forces* plus a matching single-graph batch (methane)."""
    batch = Batch.from_data_list([_graph("CH4")])
    batch.energy = torch.tensor([-17.0], dtype=torch.float64)
    batch.forces = torch.zeros_like(batch.pos)

    predictions: dict[str, torch.Tensor] = {
        "energy": torch.tensor([-17.0], dtype=torch.float64),
        "num_atoms": torch.tensor([5.0], dtype=torch.float64),
        "forces": forces,
    }
    return predictions, batch


class TestNewtonViolationPerComponent:
    def test_reports_four_numbers(self) -> None:
        predictions, batch = _batch_with_forces(torch.zeros(5, 3, dtype=torch.float64))
        metrics = mlip_metrics(predictions, batch)

        for key in NEWTON_KEYS:
            assert key in metrics, f"missing {key}"
            assert metrics[key].shape == (), f"{key} is not a scalar"

    def test_total_is_the_exact_sum_of_the_components(self) -> None:
        forces = torch.tensor(
            [
                [0.3, 0.0, 0.0],
                [0.0, -0.7, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        predictions, batch = _batch_with_forces(forces)
        metrics = mlip_metrics(predictions, batch)

        expected = sum(float(metrics[f"newton_violation_{a}"]) for a in "xyz")
        assert float(metrics["newton_violation"]) == pytest.approx(expected)

    def test_components_match_the_per_axis_net_force(self) -> None:
        forces = torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, -0.25, 0.0],
                [0.0, 0.0, 0.125],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        predictions, batch = _batch_with_forces(forces)
        metrics = mlip_metrics(predictions, batch)

        # Single graph → mean over graphs is just |Σ_i F_i,α|.
        assert float(metrics["newton_violation_x"]) == pytest.approx(1.0)
        assert float(metrics["newton_violation_y"]) == pytest.approx(0.25)
        assert float(metrics["newton_violation_z"]) == pytest.approx(0.125)
        assert float(metrics["newton_violation"]) == pytest.approx(1.375)

    def test_anisotropic_violation_is_visible(self) -> None:
        """The whole point of the split: a z-only violation must show up as
        z-only, not as a single blended number."""
        forces = torch.zeros(5, 3, dtype=torch.float64)
        forces[0, 2] = 0.9
        predictions, batch = _batch_with_forces(forces)
        metrics = mlip_metrics(predictions, batch)

        assert float(metrics["newton_violation_x"]) == pytest.approx(0.0)
        assert float(metrics["newton_violation_y"]) == pytest.approx(0.0)
        assert float(metrics["newton_violation_z"]) == pytest.approx(0.9)

    def test_conservative_forces_violate_nothing_on_any_axis(self) -> None:
        """Autograd forces from a real model: every axis at ~0."""
        batch = Batch.from_data_list([_graph("CH4")])
        batch.energy = torch.tensor([-17.0], dtype=torch.float64)
        batch.forces = torch.zeros_like(batch.pos)

        predictions = _model().eval()(batch)

        metrics = mlip_metrics(predictions, batch)
        for key in NEWTON_KEYS:
            assert float(metrics[key]) < 1e-8, f"{key} = {float(metrics[key]):.3e}"

    def test_only_the_total_reaches_the_progress_bar(self) -> None:
        assert "newton_violation" in PROG_BAR_METRICS
        for axis in "xyz":
            assert f"newton_violation_{axis}" not in PROG_BAR_METRICS

    def test_omitted_without_forces(self) -> None:
        predictions, batch = _batch_with_forces(torch.zeros(5, 3, dtype=torch.float64))
        del predictions["forces"]
        metrics = mlip_metrics(predictions, batch)
        assert not any(k.startswith("newton_violation") for k in metrics)


# ----------------------------------------------------------------------
# Gate metrics live in their own section
# ----------------------------------------------------------------------


class _LogSpy:
    """Records every ``self.log`` / ``self.log_dict`` key a step emits."""

    def __init__(self) -> None:
        self.logged: dict[str, float] = {}
        self.prog_bar_keys: set[str] = set()

    def log(self, name: str, value: typing.Any, **kwargs: typing.Any) -> None:
        self.logged[name] = float(value.compute() if hasattr(value, "compute") else value)
        if kwargs.get("prog_bar", False):
            self.prog_bar_keys.add(name)

    def log_dict(self, values: dict[str, typing.Any], **kwargs: typing.Any) -> None:
        for name, value in values.items():
            self.log(name, value, **kwargs)


def _module():
    from goal.ml.training.loss import (
        CompositeLoss,
        EnergyLoss,
        ForcesLoss,
        WeightedLoss,
    )
    from goal.ml.training.module import GOALModule

    loss = CompositeLoss(
        [
            WeightedLoss(EnergyLoss(), weight=1.0, label="energy"),
            WeightedLoss(ForcesLoss(), weight=1.0, label="forces"),
        ]
    )
    cfg = OmegaConf.create(
        {
            "training": {
                "ema": {"enabled": False},
                "gradient_clip": 0.0,
                "optimizer": {"lr": 1.0e-3},
            }
        }
    )
    return GOALModule(backbone=_model(), head=None, loss=loss, config=cfg)


def _labelled_training_batch() -> Batch:
    batch = Batch.from_data_list([_graph("CH3CH2OH")])
    batch.energy = torch.tensor([-4200.0], dtype=torch.float64)
    # Non-zero reference forces on purpose: ``forces_cosine_similarity``
    # is (correctly) omitted when every label is exactly zero, and this
    # test is about which section the metrics land in, so they all need to
    # be computable.
    torch.manual_seed(0)
    batch.forces = 0.1 * torch.randn_like(batch.pos)
    return batch


def _run_step(module, batch, step: str) -> _LogSpy:
    spy = _LogSpy()
    module.log = spy.log  # type: ignore[method-assign]
    module.log_dict = spy.log_dict  # type: ignore[method-assign]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(module, f"{step}_step")(batch, 0)
    return spy


class TestGateMetricsAreSectioned:
    def test_train_step_puts_aux_loss_in_train_gate(self) -> None:
        module = _module()
        module.train()
        keys = _run_step(module, _labelled_training_batch(), "training").logged

        assert "train_gate/aux_loss" in keys
        assert not any(
            k.startswith("train/") and ("gate" in k or "aux_loss" in k) for k in keys
        ), sorted(k for k in keys if k.startswith("train/"))

    def test_train_step_keeps_physics_metrics_in_train(self) -> None:
        module = _module()
        module.train()
        keys = _run_step(module, _labelled_training_batch(), "training").logged

        for expected in (
            "train/total",
            "train_energy/energy",
            "train_forces/forces",
            "train_energy/energy_mae_per_atom",
            "train_forces/forces_mae",
            "train_forces/newton_violation",
            "train_forces/newton_violation_x",
        ):
            assert expected in keys, f"{expected} left its section"

    def test_val_step_keeps_gate_terms_out_of_val(self) -> None:
        module = _module()
        module.eval()
        keys = _run_step(module, _labelled_training_batch(), "validation").logged

        assert not any(k.startswith("gates/") for k in keys)
        assert not any(
            k.startswith("val/") and ("gate" in k or "aux_loss" in k) for k in keys
        ), sorted(k for k in keys if k.startswith("val/"))

    def test_val_step_keeps_the_metrics_worth_monitoring(self) -> None:
        module = _module()
        module.eval()
        keys = _run_step(module, _labelled_training_batch(), "validation").logged

        for expected in (
            "val/total",
            "val_energy/energy_mae_per_atom",
            "val_forces/forces_mae",
            "val_forces/forces_cosine_similarity",
            "val_forces/newton_violation",
            "val_forces/newton_violation_z",
        ):
            assert expected in keys, f"{expected} left its section"

    def test_test_step_keeps_gate_terms_out_of_test(self) -> None:
        module = _module()
        module.eval()
        keys = _run_step(module, _labelled_training_batch(), "test").logged

        assert "test/total" in keys
        assert not any(k.startswith("test/") and ("gate" in k or "aux_loss" in k) for k in keys)

    def test_gate_curves_stay_off_the_progress_bar(self) -> None:
        module = _module()
        module.train()
        spy = _run_step(module, _labelled_training_batch(), "training")

        assert "train_gate/aux_loss" in spy.logged
        gate_on_bar = {k for k in spy.prog_bar_keys if "_gate/" in k}
        assert not gate_on_bar, f"gate metrics on the progress bar: {sorted(gate_on_bar)}"
        assert "train/total" in spy.prog_bar_keys

    def test_gate_prefix_helper(self) -> None:
        from goal.ml.training.module import GOALModule

        assert GOALModule._gate_prefix("train/") == "train_gate/"
        assert GOALModule._gate_prefix("val/") == "val_gate/"
        assert GOALModule._gate_prefix("test/") == "test_gate/"


# ----------------------------------------------------------------------
# Energy and forces get a section each
# ----------------------------------------------------------------------


class TestMetricSectionTable:
    """``metric_section`` is the single source of truth for routing."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("energy", "energy"),
            ("energy_mae_per_atom", "energy"),
            ("energy_rmse_per_atom", "energy"),
            ("forces", "forces"),
            ("forces_mae", "forces"),
            ("forces_rmse", "forces"),
            ("forces_cosine_similarity", "forces"),
            ("forces_magnitude_mae", "forces"),
            ("newton_violation", "forces"),
            ("newton_violation_y", "forces"),
            ("aux_loss", "gate"),
            ("mfc", "mfc"),
        ],
    )
    def test_routing(self, name: str, expected: str) -> None:
        assert metric_section(name) == expected

    @pytest.mark.parametrize("name", ["total", "stress", "dipole", "lr"])
    def test_everything_else_stays_in_the_plain_section(self, name: str) -> None:
        assert metric_section(name) is None

    def test_key_helper_composes_split_and_section(self) -> None:
        from goal.ml.training.module import GOALModule

        assert GOALModule._key("val/", "forces_mae") == "val_forces/forces_mae"
        assert (
            GOALModule._key("train/", "energy_mae_per_atom") == "train_energy/energy_mae_per_atom"
        )
        assert GOALModule._key("test/", "total") == "test/total"
        assert GOALModule._key("val/", "aux_loss") == "val_gate/aux_loss"


class TestEnergyAndForcesAreSeparateSections:
    """Every energy curve in ``{split}_energy/``, every force curve in
    ``{split}_forces/``, and ``{split}/`` left holding the total."""

    def test_no_energy_or_force_curve_stays_in_the_plain_section(self) -> None:
        module = _module()
        module.train()
        keys = _run_step(module, _labelled_training_batch(), "training").logged

        plain = {k for k in keys if k.startswith("train/")}
        assert "train/total" in plain
        assert not any(w in k for k in plain for w in ("energy", "forces", "newton")), plain

    def test_sections_hold_what_they_advertise(self) -> None:
        module = _module()
        module.eval()
        keys = _run_step(module, _labelled_training_batch(), "validation").logged

        energy = {k.split("/", 1)[1] for k in keys if k.startswith("val_energy/")}
        forces = {k.split("/", 1)[1] for k in keys if k.startswith("val_forces/")}

        assert {"energy", "energy_mae_per_atom", "energy_rmse_per_atom"} <= energy
        assert not any("forces" in name or "newton" in name for name in energy), energy
        assert {
            "forces",
            "forces_mae",
            "forces_rmse",
            "forces_cosine_similarity",
            "forces_magnitude_mae",
            "newton_violation",
        } <= forces
        assert not any(name.startswith("energy") for name in forces), forces

    def test_test_step_splits_the_same_way(self) -> None:
        module = _module()
        module.eval()
        keys = _run_step(module, _labelled_training_batch(), "test").logged

        assert "test/total" in keys
        assert "test_energy/energy_mae_per_atom" in keys
        assert "test_forces/forces_mae" in keys
        assert {k for k in keys if k.startswith("test/")} == {"test/total"}

    def test_headline_metrics_still_reach_the_progress_bar(self) -> None:
        module = _module()
        module.train()
        spy = _run_step(module, _labelled_training_batch(), "training")

        assert "train/total" in spy.prog_bar_keys
        assert "train_energy/energy_mae_per_atom" in spy.prog_bar_keys
        assert "train_forces/forces_cosine_similarity" in spy.prog_bar_keys
        assert "train_forces/newton_violation" in spy.prog_bar_keys
        # Non-headline diagnostics stay off it.
        assert "train_forces/newton_violation_x" not in spy.prog_bar_keys
        assert "train_forces/forces_rmse" not in spy.prog_bar_keys


# ----------------------------------------------------------------------
# The OOD set: a second validation dataloader, logged under test…/
# ----------------------------------------------------------------------


def _attach_fake_trainer(module: typing.Any, val_split_names: list[str] | None) -> None:
    """Give *module* just enough trainer/datamodule for ``_val_prefix``."""
    import types

    datamodule = types.SimpleNamespace()
    if val_split_names is not None:
        datamodule.val_split_names = val_split_names
    module._trainer = types.SimpleNamespace(current_epoch=0, datamodule=datamodule)


def _run_val_step(module: typing.Any, batch: Batch, dataloader_idx: int) -> _LogSpy:
    spy = _LogSpy()
    module.log = spy.log  # type: ignore[method-assign]
    module.log_dict = spy.log_dict  # type: ignore[method-assign]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module.validation_step(batch, 0, dataloader_idx)
    return spy


class TestSecondValidationDataloader:
    def test_prefix_comes_from_the_datamodule(self) -> None:
        module = _module()
        _attach_fake_trainer(module, ["val", "test"])

        assert module._val_prefix(0) == "val/"
        assert module._val_prefix(1) == "test/"

    def test_prefix_falls_back_without_a_datamodule(self) -> None:
        module = _module()
        _attach_fake_trainer(module, None)

        assert module._val_prefix(0) == "val/"
        assert module._val_prefix(1) == "test/"

    def test_unnamed_extra_dataloader_gets_its_own_section(self) -> None:
        """Never silently merge an unexpected dataloader into val/."""
        module = _module()
        _attach_fake_trainer(module, ["val", "test"])

        assert module._val_prefix(2) == "val2/"

    def test_ood_metrics_land_in_the_test_sections(self) -> None:
        module = _module()
        module.eval()
        _attach_fake_trainer(module, ["val", "test"])
        keys = _run_val_step(module, _labelled_training_batch(), 1).logged

        assert "test/total" in keys
        assert "test_energy/energy_mae_per_atom" in keys
        assert "test_forces/forces_mae" in keys
        # Nothing from the OOD set may reach the ID sections — mixing the
        # two is the whole thing this guards against.
        assert not any(k.startswith("val") for k in keys), sorted(keys)

    def test_id_metrics_are_unchanged_by_the_second_loader(self) -> None:
        module = _module()
        module.eval()
        _attach_fake_trainer(module, ["val", "test"])
        keys = _run_val_step(module, _labelled_training_batch(), 0).logged

        assert "val/total" in keys
        assert "val_forces/forces_mae" in keys
        assert not any(k.startswith("test") for k in keys), sorted(keys)

    def test_ood_set_stays_off_the_progress_bar(self) -> None:
        module = _module()
        module.eval()
        _attach_fake_trainer(module, ["val", "test"])
        spy = _run_val_step(module, _labelled_training_batch(), 1)

        assert not spy.prog_bar_keys, sorted(spy.prog_bar_keys)


class TestCompositeSubLossLabels:
    """``build_loss`` labels a per-property sub-loss ``{property}_{fn}``
    (``forces_mse``, ``energy_huber``, ...), one per registered loss
    function — so routing has to follow the property, not a fixed list of
    names, or a sub-loss would end up in a different section from the
    group total it sums into."""

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("energy_mse", "energy"),
            ("energy_mae", "energy"),
            ("energy_huber", "energy"),
            ("forces_mse", "forces"),
            ("forces_l1", "forces"),
            ("forces_smooth_l1", "forces"),
        ],
    )
    def test_sub_loss_joins_its_property(self, label: str, expected: str) -> None:
        assert metric_section(label) == expected

    def test_unknown_property_keeps_the_plain_section(self) -> None:
        for label in ("stress", "stress_mse", "dipole", "charge"):
            assert metric_section(label) is None
