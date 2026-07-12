"""Command-line interface for QuantScope."""

import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer

app = typer.Typer(
    name="quantscope",
    help="Hardware-aware neural-network quantization and numerical debugging.",
    no_args_is_help=True,
)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Hardware-aware neural-network quantization and numerical debugging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def version_info() -> None:
    """Print the installed QuantScope version."""
    try:
        installed_version = version("quantscope")
    except PackageNotFoundError:
        installed_version = "development"

    typer.echo(f"QuantScope {installed_version}")


@app.command()
def env_info() -> None:
    """Print interpreter, package, and quantization-backend information."""
    from quantscope.utilities import capture_environment

    typer.echo(json.dumps(capture_environment(), indent=2))


@app.command()
def train_fp32(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
) -> None:
    """Train the FP32 baseline described by an experiment config (measured)."""
    from quantscope.config import load_experiment_config
    from quantscope.evaluation.loop import train_fp32 as _train

    experiment = load_experiment_config(config)
    _, metrics = _train(experiment)
    typer.echo(f"fp32 eval accuracy (measured): {metrics['accuracy']:.4f}")


@app.command()
def ptq(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    checkpoint: Path | None = typer.Option(None, "--checkpoint"),
) -> None:
    """Post-training-quantize the FP32 baseline via FX graph mode (measured INT8 CPU)."""
    from quantscope.config import load_experiment_config
    from quantscope.quantization.ptq import run_ptq

    experiment = load_experiment_config(config)
    _, metrics = run_ptq(experiment, checkpoint=checkpoint)
    typer.echo(f"int8 eval accuracy (measured, CPU): {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    app()
