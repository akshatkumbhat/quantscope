"""Command-line interface for QuantScope."""

from importlib.metadata import PackageNotFoundError, version

import typer

app = typer.Typer(
    name="quantscope",
    help="Hardware-aware neural-network quantization and numerical debugging.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Hardware-aware neural-network quantization and numerical debugging."""


@app.command()
def version_info() -> None:
    """Print the installed QuantScope version."""
    try:
        installed_version = version("quantscope")
    except PackageNotFoundError:
        installed_version = "development"

    typer.echo(f"QuantScope {installed_version}")


if __name__ == "__main__":
    app()
