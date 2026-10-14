"""MD module CLI interface (isolated to this module only).

Command-line interface for molecular dynamics simulations using Typer.
This CLI is completely separate from goal.ml and does not affect it.

Usage:
    python -m goal.md.cli.main --help
"""

import os as _os

import typer

app = typer.Typer(
    name="goal-md",
    help="GOAL Molecular Dynamics — MD simulations with goal.ml integration",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo("GOAL MD Module v0.1.0")
    typer.echo("Integrated with goal.ml for machine learning potentials")


@app.command()
def simulate(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to Hydra config file for simulation",
    ),
) -> None:
    """Run MD simulation from Hydra config.

    Example:
        goal-md simulate -c configs/md/langevin_with_model_sim.yaml
    """
    typer.echo(f"Running simulation from config: {config}")
    typer.echo("Not yet implemented - use notebooks for now")


CONFIGS_MD_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "configs", "md")
)

if __name__ == "__main__":
    app()
