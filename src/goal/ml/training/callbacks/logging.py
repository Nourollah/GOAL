"""Rich logging callback for beautiful terminal output."""

from __future__ import annotations

import typing

import lightning as L
from rich.console import Console
from rich.table import Table


class RichLoggingCallback(L.Callback):
    """Pretty-print training metrics to the terminal using Rich.

    Logs a formatted table at the end of each validation epoch showing
    key metrics like loss components, learning rate, and epoch number.
    """

    def __init__(self) -> None:
        super().__init__()
        self.console: Console = Console()

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Print a Rich table of current metrics."""
        if not trainer.is_global_zero:
            return

        metrics: typing.Dict[str, typing.Any] = trainer.callback_metrics
        if not metrics:
            return

        table: Table = Table(title=f"Epoch {trainer.current_epoch}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key in sorted(metrics):
            val: typing.Any = metrics[key]
            if hasattr(val, "item"):
                val = val.item()
            table.add_row(key, f"{val:.6f}" if isinstance(val, float) else str(val))

        self.console.print(table)
