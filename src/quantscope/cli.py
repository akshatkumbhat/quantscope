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


@app.command()
def ablate(
    seed: int = typer.Option(0, "--seed"),
    runs_dir: str = typer.Option("runs/validation", "--runs-dir"),
    weight_bits: int = typer.Option(4, "--weight-bits"),
    act_bits: int = typer.Option(4, "--act-bits"),
    freq_step: float = typer.Option(0.30, "--freq-step"),
) -> None:
    """Per-group quantization ablation vs. an FP32 texture-bench checkpoint. Slow."""
    from quantscope.benchmark import benchmark_config, texture10_calibration
    from quantscope.data.synthetic import build_datasets
    from quantscope.quantization.simulate import SimQuantConfig
    from quantscope.sensitivity import ablate_groups

    config = benchmark_config(seed=seed, output_dir=runs_dir, freq_step=freq_step)
    checkpoint = Path(runs_dir) / f"texture-a-seed{seed}-fp32" / "model.pt"
    calibration = texture10_calibration(config)
    _, test_set = build_datasets(config.data, config.model)
    results = ablate_groups(
        config,
        calibration,
        test_set,
        checkpoint=checkpoint,
        target=SimQuantConfig(weight_bits, act_bits),
    )
    for group, deltas in sorted(results.items(), key=lambda kv: kv[1]["delta_nll"], reverse=True):
        typer.echo(
            f"{group:>14} (simulated): dNLL={deltas['delta_nll']:+.4f} "
            f"flips={deltas['prediction_flip_rate']:.4f} "
            f"dAcc={deltas['delta_accuracy']:+.4f}"
        )


@app.command()
def sweep(
    seed: int = typer.Option(0, "--seed"),
    runs_dir: str = typer.Option("runs/validation-012", "--runs-dir"),
    freq_step: float = typer.Option(0.12, "--freq-step"),
) -> None:
    """Exhaustive 256-config INT4/INT8 sweep vs an FP32 checkpoint. Very slow."""
    import json

    import torch

    from quantscope.benchmark import benchmark_config, texture10_calibration
    from quantscope.config import Provenance
    from quantscope.data.synthetic import build_datasets
    from quantscope.models.tiny_cnn import build_model
    from quantscope.search import exhaustive_sweep, pareto_frontier
    from quantscope.utilities import RunWriter

    config = benchmark_config(seed=seed, output_dir=runs_dir, freq_step=freq_step)
    checkpoint = Path(runs_dir) / f"texture-a-seed{seed}-fp32" / "model.pt"
    model = build_model(config.model)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    calibration = texture10_calibration(config)
    _, test_set = build_datasets(config.data, config.model)

    records = exhaustive_sweep(model, calibration, test_set, batch_size=config.training.batch_size)
    writer = RunWriter(config, kind="sweep")
    (writer.run_dir / "sweep_table.json").write_text(
        json.dumps([r.to_dict() for r in records], indent=1) + "\n"
    )
    frontier = pareto_frontier(records, quality="nll")
    writer.record_metric("num_configs", len(records), Provenance.SIMULATED)
    writer.record_metric(
        "pareto_size_nll",
        len(frontier),
        Provenance.SIMULATED,
        note="cost is estimated (normalized weight bits); nll simulated",
    )
    writer.finalize()
    typer.echo(f"sweep complete: {len(records)} configs, NLL-frontier size {len(frontier)}")


@app.command()
def texture_bench(
    seed: int = typer.Option(0, "--seed"),
    epochs: int = typer.Option(35, "--epochs"),
    boundary_fraction: float = typer.Option(0.45, "--boundary-fraction"),
    boundary_low: float = typer.Option(0.40, "--boundary-low"),
    boundary_high: float = typer.Option(0.50, "--boundary-high"),
    snr_db: float = typer.Option(4.0, "--snr-db"),
    freq_step: float = typer.Option(0.30, "--freq-step"),
    bottleneck_width: int = typer.Option(6, "--bottleneck-width"),
    output_dir: str = typer.Option("runs", "--output-dir"),
) -> None:
    """Run Texture-10 benchmark A: FP32 (measured) vs simulated W8A8/W4A4. Slow."""
    from quantscope.benchmark import benchmark_config, run_texture_benchmark

    config = benchmark_config(
        seed=seed,
        epochs=epochs,
        boundary_fraction=boundary_fraction,
        boundary_low=boundary_low,
        boundary_high=boundary_high,
        snr_db=snr_db,
        freq_step=freq_step,
        bottleneck_width=bottleneck_width,
        output_dir=output_dir,
    )
    results = run_texture_benchmark(config)
    for label, metrics in results.items():
        provenance = "measured" if label == "fp32" else "simulated"
        typer.echo(
            f"{label:>5} ({provenance}): accuracy={metrics['accuracy']:.4f} "
            f"nll={metrics['nll']:.4f} margin={metrics['mean_margin']:.3f}"
        )


if __name__ == "__main__":
    app()
